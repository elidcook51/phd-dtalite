#!/usr/bin/env python3
"""Test 4: Dynamic User Equilibrium as the information baseline (WINNER).

Merged from two independent implementations (/tmp/build_A and /tmp/build_B),
both targeting the MSA procedure for Dynamic User Equilibrium from Bingrong
Sun's 2018 dissertation (Chapter 4, Test 4; after Sheffi 1985):

  Step 0: assign every agent its shortest free-flow path (historical times),
          set n = 1, simulate with DTALite.
  Step 1: read link travel times from the current simulated pattern.
  Step 2: direction finding - all-or-nothing auxiliary assignment: every
          agent's shortest-travel-time route under the current link times.
  Step 3: move - each agent switches to its auxiliary route with probability
          alpha = 1/n, otherwise keeps its current route (agent-level MSA).
          With n = 1 the first move uses alpha = 1.0.
  Step 4: re-simulate, recompute the relative gap
          (current_TT - shortest_TT) / shortest_TT, repeat until the gap
          falls below 0.01% or the iteration cap is hit. The gap always
          uses FRESH link times from the just-finished DTALite run.

Key assumption (dissertation, Chapter 4): the operator assumes ALL drivers
minimize travel time only. The converged UE routes are written out; they are
the routes the operator would broadcast as guidance in Test 4. How users with
true preferences react to that guidance is a separate step (deferred here).

Merge notes (validator's verdict):
  Base is build A: ragged-CSV reader for DTALite's link outputs (handles the
  trailing-comma quirk; build B's plain pd.read_csv would silently fall back
  to free-flow), cross-platform DTALite launch (native ./DTALite, DTALite.exe
  on Windows, wine fallback on Linux; build B's bare exe path fails on
  Linux), tolerant demand parsing (NaN-safe numerics; departure column
  optional), OD mapping derived from choice-set route endpoints (robust to
  row ordering; build B assumes row-major (origin-1)*24+(dest-1)), trailing
  ";" path normalization, vehicle-hours reporting.
  Taken from build B: run.log file logging, delete-stale-outputs freshness
  discipline, CRLF line endings on input_agent.csv (Windows/DTALite
  convention), the spec-faithful loop guard (skip the MSA loop entirely when
  the Step-0 gap is already below tolerance), and CLI overrides for the
  choice-set / demand / network / exe paths plus the DTALite timeout.
  Fixed in the merge: network rows with no usable time now warn instead of
  being skipped silently; link-time aggregation is always mean-over-timestamp
  when a timestamp column exists.

  Final review (2026-10-11): fixed relative_gap never accumulating sp_sum
  (the gap was always inf, so the MSA could never report convergence);
  fixed the _run summary block referencing t_start from main's scope
  (NameError at the end of the run); added a warning when the choice set
  does not hold the expected 576 OD rows.
  Fix pass (2026-10-13): load_network is bounded to the first link table
  (the sheet's second sub-table shares (u,v) keys with empty free-flow
  cells and used to overwrite all good times with NaN); route_time scores
  pairs missing from both tables as inf instead of 0.0; missing DTALite
  link outputs now raise UEError instead of a free-flow fallback that
  could report false convergence; all-NaN vehicle-hours reported as n/a;
  write_input drops Unnamed:/helper columns; NaN template paths written
  as empty strings; outputs backed up before overwrite on re-run.

Usage:
  python scenarios_ue.py --data-dir <sandbox> [--result-dir results/ue_baseline]
                     [--seed 42] [--max-iters 50] [--gap-threshold 0.0001]
                     [--choice-mat ...] [--demand-csv ...] [--network-xlsx ...]
                     [--exe ...] [--timeout 3600]
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


class UEError(Exception):
    """Clean, user-facing failure (bad inputs, DTALite failure, ...)."""


def backup_existing(path):
    """Rename an existing output to a timestamped backup; return its new path."""
    p = Path(path)
    if not p.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = p.with_name(p.name + ".bak_" + stamp)
    n = 1
    while target.exists():
        n += 1
        target = p.with_name(p.name + ".bak_%s_%d" % (stamp, n))
    p.rename(target)
    return target


class Logger(object):
    """Console + run.log logger (from build B)."""

    def __init__(self, log_path=None):
        self.lines = []
        self._fh = None
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            backup_existing(log_path)
            self._fh = open(log_path, "w")

    def __call__(self, msg):
        line = str(msg)
        print(line, flush=True)
        self.lines.append(line)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def find_header_row(raw, required=("init", "term"), scan=15):
    for i in range(min(scan, len(raw))):
        cells = [str(c).lower() for c in raw.iloc[i].tolist()]
        joined = " ".join(cells)
        if all(r in joined for r in required):
            return i
    return None


def load_network(path, log):
    raw = pd.read_excel(path, header=None)
    hrow = find_header_row(raw)
    if hrow is None:
        raise UEError("could not find link-table header row in %s" % path)
    header = [str(c).strip().lower() for c in raw.iloc[hrow].tolist()]
    data = raw.iloc[hrow + 1:].reset_index(drop=True)

    def colkey(*needles):
        for i, name in enumerate(header):
            if all(nd in name for nd in needles):
                return i
        return None

    c_from = colkey("init")
    c_to = colkey("term")
    c_len = colkey("length")
    c_fft = colkey("free", "time")
    c_spd = colkey("speed")
    if c_from is None or c_to is None:
        raise UEError("link table in %s lacks init/term node columns" % path)

    freeflow = {}
    n_skipped = 0       # rows parsed but with no usable free-flow time
    n_unparseable = 0   # rows whose init/term node ids would not parse
    for _, row in data.iterrows():
        try:
            u = int(row[c_from])
            v = int(row[c_to])
        except (ValueError, TypeError):
            n_unparseable += 1
            if freeflow:
                # End of the first link table: the standard SiouxFalls sheet
                # carries a second sub-table further down whose rows share
                # the same (u, v) keys but have empty free-flow cells. Reading
                # to the bottom of the sheet would let those rows overwrite
                # the good times (silently: float(nan) raises nothing and
                # nan <= 0 is False).
                break
            continue
        fft = None
        if c_fft is not None:
            try:
                fft = float(row[c_fft])
            except (ValueError, TypeError):
                fft = None
        if ((fft is None or pd.isna(fft) or fft <= 0)
                and c_len is not None and c_spd is not None):
            try:
                length = float(row[c_len])
                speed = float(row[c_spd])
                if length > 0 and speed > 0:
                    fft = length / speed * 60.0
            except (ValueError, TypeError):
                pass
        if fft is None or pd.isna(fft) or fft <= 0:
            # Merge fix: warn instead of silently dropping the link.
            n_skipped += 1
            continue
        if (u, v) in freeflow:
            log("warning: duplicate link row for (%d, %d); keeping the first "
                "free-flow time %.2f" % (u, v, freeflow[(u, v)]))
            continue
        freeflow[(u, v)] = fft
    if n_unparseable:
        log("warning: %d rows had unparseable init/term node ids and were skipped"
            % n_unparseable)
    if n_skipped:
        log("warning: %d link rows had no usable free-flow time and were skipped"
            % n_skipped)
    if not freeflow:
        raise UEError("no usable free-flow link times in %s" % path)
    log("network: %d directed links with free-flow times" % len(freeflow))
    return freeflow


def unwrap_cell(value):
    if isinstance(value, np.ndarray) and value.size > 0:
        text = str(value.flat[0]).strip()
        if text:
            return text
    elif isinstance(value, str) and value.strip():
        return value.strip()
    return None


def load_choiceset(path, log):
    mat = sio.loadmat(path)
    if "finallist" not in mat:
        raise UEError("%s has no 'finallist' variable" % path)
    table = mat["finallist"].T  # [od_idx, route_idx]; (576, 10) for Sioux Falls
    od_lookup = {}
    od_routes = []
    for i in range(table.shape[0]):
        routes = []
        for j in range(table.shape[1]):
            text = unwrap_cell(table[i, j])
            if text is None:
                continue
            try:
                nodes = [int(x) for x in text.split(";") if x.strip()]
            except ValueError:
                continue
            if len(nodes) >= 2:
                routes.append(nodes)
        if routes:
            # OD identity from the route endpoints themselves: robust to any
            # row ordering of the choice-set matrix.
            od = (routes[0][0], routes[0][-1])
            od_lookup[od] = len(od_routes)
            od_routes.append(routes)
    if not od_routes:
        raise UEError("no valid routes found in %s" % path)
    if len(od_routes) != 576:
        # The Sioux Falls choice set is 576 OD rows; anything else usually
        # means the .mat orientation assumption above is wrong.
        log("warning: expected 576 OD rows, found %d" % len(od_routes))
    log("choiceset: %d OD pairs, %d total routes (matrix %s)"
        % (len(od_routes), sum(len(r) for r in od_routes), table.shape))
    return od_lookup, od_routes


def detect_columns(df, log):
    cols = list(df.columns)
    lowered = {c: str(c).strip().lower() for c in cols}

    def find(*needles, exclude=()):
        for c in cols:
            name = lowered[c]
            if all(n in name for n in needles) and not any(e in name for e in exclude):
                return c
        return None

    agent_col = find("agent_id")
    # Prefer node IDs over zone IDs: the choice set is node-based.
    origin_col = find("from_origin_node_id") or find("from_zone_id") or find("origin")
    dest_col = (find("to_destination_node_id") or find("to_zone_id")
                or find("destination", exclude=("origin",)))
    path_col = find("path_node_sequence")
    dep_col = find("departure")  # optional; detected but not required
    missing = [n for n, c in (("agent_id", agent_col), ("origin", origin_col),
                              ("destination", dest_col), ("path", path_col)) if c is None]
    if missing:
        raise UEError("demand file lacks columns: %s" % ", ".join(missing))
    return agent_col, origin_col, dest_col, path_col, dep_col


_missing_link_pairs = set()


def route_time(link_pairs, link_time, freeflow, log=None):
    """Sum travel time over link pairs.

    Pairs missing from both link-time tables used to score 0.0, biasing
    shortest-path choice *toward* routes over missing links. They now score
    infinite time instead, and the first miss logs a warning.
    """
    total = 0.0
    for pair in link_pairs:
        if pair in link_time:
            total += link_time[pair]
        elif pair in freeflow:
            total += freeflow[pair]
        else:
            total += float("inf")
            if pair not in _missing_link_pairs:
                _missing_link_pairs.add(pair)
                msg = ("warning: link %s missing from both link-time tables; "
                       "scoring it as infinite time" % (pair,))
                if log is None:
                    print(msg, flush=True)
                else:
                    log(msg)
    return total


def read_dtalite_csv(path, log):
    # DTALite's link-MOE files are ragged: data rows can carry more fields
    # (trailing comma) or fewer fields than the header row, which silently
    # misaligns a plain pd.read_csv. Align every row to the header width.
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = [ln for ln in text.splitlines() if ln.strip()]
    if not rows:
        raise ValueError("empty file")
    header = [c.strip() for c in rows[0].split(",")]
    n = len(header)
    data = []
    for ln in rows[1:]:
        fields = [f.strip() for f in ln.split(",")]
        while len(fields) > n and fields[-1] == "":
            fields.pop()
        if len(fields) > n:
            fields = fields[:n]
        while len(fields) < n:
            fields.append("")
        data.append(fields)
    return pd.DataFrame(data, columns=header)


# "output_link.csv" was listed here but DTALite never writes it (dead entry,
# removed 2026-10-13).
STALE_OUTPUTS = ("output_agent.csv",
                 "output_LinkTDMOE.csv", "Each iteration.csv")


def dtalite_command(data_dir, exe_override, log):
    if exe_override:
        exe = Path(exe_override)
        if not exe.exists():
            raise UEError("DTALite executable not found: %s" % exe_override)
        return [str(exe)]
    if sys.platform.startswith("win"):
        exe = data_dir / "DTALite.exe"
        if not exe.exists():
            raise UEError("DTALite.exe not found in %s" % data_dir)
        return [str(exe)]
    native = data_dir / "DTALite"
    if native.exists() and os.access(str(native), os.X_OK):
        return [str(native)]
    exe = data_dir / "DTALite.exe"
    if exe.exists() and shutil.which("wine"):
        return ["wine", str(exe)]
    raise UEError("no runnable DTALite found in %s "
                  "(need ./DTALite, or DTALite.exe plus wine)" % data_dir)


def run_dtalite(data_dir, exe_override, timeout_s, log):
    # Freshness discipline (from build B): delete stale outputs before each
    # run so nothing below can mistake a previous iteration's files for new.
    for name in STALE_OUTPUTS:
        p = data_dir / name
        if p.exists():
            p.unlink()
    cmd = dtalite_command(data_dir, exe_override, log)
    log("running DTALite (%s) ..." % " ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(data_dir), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise UEError("DTALite timed out after %d seconds" % timeout_s)
    except OSError as exc:
        raise UEError("could not launch DTALite (%s)" % exc)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else "no output captured"
        raise UEError("DTALite exited with code %d: %s" % (proc.returncode, detail))
    log("DTALite finished in %.0f s" % (time.time() - t0))
    refreshed = [name for name in STALE_OUTPUTS if (data_dir / name).exists()]
    if not refreshed:
        log("warning: DTALite wrote none of the watched outputs")
    return refreshed


def load_link_times(data_dir, refreshed, freeflow, log):
    for name in ("output_link.csv", "output_LinkTDMOE.csv", "Each iteration.csv"):
        if name not in refreshed:
            continue
        try:
            df = read_dtalite_csv(data_dir / name, log)
        except Exception as exc:
            log("warning: could not read %s (%s)" % (name, exc))
            continue
        cols = list(df.columns)
        lowered = {c: str(c).strip().lower() for c in cols}

        def find_exact(want):
            for c in cols:
                if lowered[c] == want:
                    return c
            return None

        def find(*needles, exclude=()):
            for c in cols:
                nm = lowered[c]
                if all(n in nm for n in needles) and not any(e in nm for e in exclude):
                    return c
            return None

        # Prefer the exact node-id headers: the old substring matcher took
        # the first column containing "to" but not "from", which could misfire
        # on a column like total_volume.
        c_from = find_exact("from_node_id") or find("from", exclude=("transform",))
        c_to = find_exact("to_node_id") or find("to", exclude=("from",))
        c_tt = find("travel", "time")
        if c_from is None or c_to is None or c_tt is None:
            log("warning: %s lacks link travel-time columns, skipped" % name)
            continue
        c_ts = find("timestamp")
        use = df[[c_from, c_to, c_tt]].copy()
        use.columns = ["u", "v", "tt"]
        use["u"] = pd.to_numeric(use["u"], errors="coerce")
        use["v"] = pd.to_numeric(use["v"], errors="coerce")
        use["tt"] = pd.to_numeric(use["tt"], errors="coerce")
        use = use.dropna()
        if use.empty:
            log("warning: %s has no usable link travel times, skipped" % name)
            continue
        if c_ts is not None:
            # Time-dependent file (e.g. Each iteration.csv): mean over the
            # reported timestamps, as in build A.
            link_time = use.groupby(["u", "v"])["tt"].mean().to_dict()
        else:
            link_time = dict(zip(zip(use["u"], use["v"]), use["tt"]))
        link_time = {(int(u), int(v)): float(t) for (u, v), t in link_time.items()}
        log("link times: %d links from %s" % (len(link_time), name))
        return link_time
    # No usable link times came out of this DTALite run. Falling back to
    # free-flow here would let Step-0's free-flow assignment show a zero
    # relative gap and report a false "converged" with exit 0, so abort.
    raise UEError("no usable link travel times in this DTALite run "
                  "(none of the watched link outputs was refreshed)")


def vehicle_hours(data_dir, refreshed, log):
    name = "output_agent.csv"
    if name not in refreshed:
        return None, "output_agent.csv not refreshed by DTALite"
    try:
        df = read_dtalite_csv(data_dir / name, log)
    except Exception as exc:
        return None, "could not read output_agent.csv (%s)" % exc
    cols = list(df.columns)
    tt_col = None
    for c in cols:
        nm = str(c).strip().lower()
        if "travel" in nm and "time" in nm:
            tt_col = c
            break
    if tt_col is None:
        return None, "no travel-time column in output_agent.csv"
    tt = pd.to_numeric(df[tt_col], errors="coerce")
    if tt.isna().all():
        # Summing an all-NaN column yields 0.0, which would be reported as
        # "0.00 vehicle-hours"; there is really nothing to report.
        return None, "travel-time column %s is all-NaN in output_agent.csv" % tt_col
    try:
        total_min = float(tt.sum())
    except Exception as exc:
        return None, "could not sum %s (%s)" % (tt_col, exc)
    return total_min / 60.0, "summed %s over %d agents" % (tt_col, len(df))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Test 4: MSA for Dynamic User Equilibrium")
    script_dir = Path(__file__).resolve().parent
    # DTALite_Files/ is read-only: every run must point --data-dir at a
    # writable sandbox copy. No default on purpose -- see the guard below.
    parser.add_argument("--data-dir", default=None,
                        help="writable sandbox copy of the DTALite inputs; REQUIRED")
    parser.add_argument("--result-dir", default=str(script_dir / "results" / "ue_baseline"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--gap-threshold", type=float, default=0.0001)
    parser.add_argument("--choice-mat", default=None)
    parser.add_argument("--demand-csv", default=None)
    parser.add_argument("--network-xlsx", default=None)
    parser.add_argument("--exe", default=None, help="DTALite executable override")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="seconds to wait for one DTALite run")
    args = parser.parse_args(argv)
    if not args.data_dir:
        parser.error("--data-dir is required: pass a writable sandbox copy of the DTALite inputs (DTALite_Files/ is read-only)")

    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(result_dir / "run.log")
    t_start = time.time()
    try:
        _run(args, data_dir, result_dir, log, t_start)
    except UEError as exc:
        log("ERROR: %s" % exc)
        log.close()
        return 1
    except Exception:
        import traceback
        log("ERROR: unexpected failure\n%s" % traceback.format_exc())
        log.close()
        return 1
    log("total wall time %.0f s" % (time.time() - t_start))
    log.close()
    return 0


def _run(args, data_dir, result_dir, log, t_start):
    mat_path = args.choice_mat or str(data_dir / "choice set no overlap try.mat")
    dem_path = args.demand_csv or str(data_dir / "input_agent_initial.csv")
    net_path = args.network_xlsx or str(data_dir / "SiouxFalls_net.xlsx")
    for label, path in (("choice set", mat_path), ("demand", dem_path),
                        ("network", net_path)):
        if not Path(path).is_file():
            raise UEError("%s not found: %s" % (label, path))

    log("Test 4 (UE baseline): MSA for Dynamic User Equilibrium")
    log("data dir: %s" % data_dir)
    log("result dir: %s" % result_dir)
    log("seed=%d, max_iters=%d, gap_threshold=%.6f, timeout=%ds"
        % (args.seed, args.max_iters, args.gap_threshold, args.timeout))

    freeflow = load_network(net_path, log)
    od_lookup, od_routes = load_choiceset(mat_path, log)
    od_linkpairs = [[[(r[k], r[k + 1]) for k in range(len(r) - 1)] for r in routes]
                    for routes in od_routes]

    template = pd.read_csv(dem_path)
    agent_col, origin_col, dest_col, path_col, dep_col = detect_columns(template, log)
    agent_ids = template[agent_col].to_numpy()
    n_agents = len(template)
    log("demand: %d agents" % n_agents)

    origins = pd.to_numeric(template[origin_col], errors="coerce")
    dests = pd.to_numeric(template[dest_col], errors="coerce")
    agent_od = np.full(n_agents, -1, dtype=int)
    no_od = 0
    for i in range(n_agents):
        if pd.isna(origins[i]) or pd.isna(dests[i]):
            no_od += 1
            continue
        od_idx = od_lookup.get((int(origins[i]), int(dests[i])), -1)
        if od_idx < 0:
            no_od += 1
        else:
            agent_od[i] = od_idx
    switchable = agent_od >= 0
    n_switch = int(switchable.sum())
    log("agents with a choiceset OD: %d, without: %d (kept on template path)"
        % (n_switch, no_od))
    # The choice set holds no intrazonal (i, i) routes; flag the agents whose
    # demand is intrazonal so they are not silently excluded from the UE.
    n_intrazonal = sum(
        1 for i in range(n_agents)
        if agent_od[i] < 0
        and not pd.isna(origins[i]) and not pd.isna(dests[i])
        and int(origins[i]) == int(dests[i]))
    if n_intrazonal:
        log("warning: %d agents have intrazonal (i,i) demand, which the choice "
            "set does not cover; they stay on their template path" % n_intrazonal)
    if n_switch == 0:
        raise UEError("no agent maps to a choiceset OD pair")

    # A NaN template path would become the literal string "nan"; write it as
    # an empty path instead.
    template_paths = [("" if pd.isna(p) else str(p))
                      for p in template[path_col].tolist()]
    current_route = np.full(n_agents, -1, dtype=int)
    link_time = dict(freeflow)

    # Step 0: all-or-nothing on free-flow (historical) travel times.
    for i in np.flatnonzero(switchable):
        od = agent_od[i]
        best, best_tt = 0, None
        for r, pairs in enumerate(od_linkpairs[od]):
            tt = route_time(pairs, link_time, freeflow, log)
            if best_tt is None or tt < best_tt:
                best, best_tt = r, tt
        current_route[i] = best
    log("Step 0: all agents assigned shortest free-flow paths")

    def path_strings():
        out = []
        for i in range(n_agents):
            if switchable[i]:
                nodes = od_routes[agent_od[i]][current_route[i]]
                out.append(";".join(str(x) for x in nodes) + ";")
            else:
                out.append(template_paths[i])
        return out

    def write_input():
        # The demand template can carry helper columns (e.g. pandas "Unnamed:"
        # columns) that must not leak into DTALite's input_agent.csv.
        df = template.drop(columns=[c for c in template.columns
                                    if str(c).startswith("Unnamed:")],
                           errors="ignore").copy()
        df[path_col] = pd.Series(path_strings(), index=df.index)
        df.to_csv(data_dir / "input_agent.csv", index=False, lineterminator="\r\n")

    write_input()
    refreshed = run_dtalite(data_dir, args.exe, args.timeout, log)
    link_time = load_link_times(data_dir, refreshed, freeflow, log)

    def shortest_routes():
        aux = np.zeros(n_agents, dtype=int)
        aux_tt = np.zeros(n_agents)
        for i in np.flatnonzero(switchable):
            od = agent_od[i]
            best, best_tt = 0, None
            for r, pairs in enumerate(od_linkpairs[od]):
                tt = route_time(pairs, link_time, freeflow, log)
                if best_tt is None or tt < best_tt:
                    best, best_tt = r, tt
            aux[i] = best
            aux_tt[i] = best_tt
        return aux, aux_tt

    def relative_gap():
        cur_sum, sp_sum = 0.0, 0.0
        for i in np.flatnonzero(switchable):
            od = agent_od[i]
            cur_sum += route_time(od_linkpairs[od][current_route[i]], link_time, freeflow, log)
            best_tt = None
            for pairs in od_linkpairs[od]:
                tt = route_time(pairs, link_time, freeflow, log)
                if best_tt is None or tt < best_tt:
                    best_tt = tt
            sp_sum += best_tt
        if sp_sum <= 0:
            return float("inf")
        return (cur_sum - sp_sum) / sp_sum

    gap = relative_gap()
    log("initial gap after Step 0: %.6f" % gap)

    rng = np.random.default_rng(args.seed)
    order = np.flatnonzero(switchable)
    # Spec-faithful loop guard (from build B): with n = 1 the first move
    # would use alpha = 1.0; if Step 0 already converged, skip the loop.
    n = 1
    iters_done = 0
    converged = gap < args.gap_threshold
    while n <= args.max_iters and not converged:
        aux, _ = shortest_routes()          # Step 2: auxiliary pattern
        alpha = 1.0 / n                     # Step 3: move, agent-level MSA
        draw = rng.random(n_switch)
        move = draw < alpha
        idx = order[move]
        current_route[idx] = aux[idx]
        write_input()
        refreshed = run_dtalite(data_dir, args.exe, args.timeout, log)
        link_time = load_link_times(data_dir, refreshed, freeflow, log)
        gap = relative_gap()                # Step 4: gap on FRESH link times
        iters_done = n
        log("iter %2d (n=%d, alpha=%.4f): switched %5d agents, gap=%.6f"
            % (n, n, alpha, int(move.sum()), gap))
        if gap < args.gap_threshold:
            converged = True
            break
        n += 1

    if converged:
        log("converged: relative gap %.6f < %.6f after %d iteration(s)"
            % (gap, args.gap_threshold, iters_done))
    else:
        log("WARNING: gap %.6f still above %.6f after %d iteration(s) (cap reached)"
            % (gap, args.gap_threshold, iters_done))

    # Test 6 (Future Traffic Conditions Estimation): save the final MSA
    # iteration link volumes as the "predicted" UE condition.  The reaction
    # phase (preference_reaction.py) will generate the "actual" volumes.
    # Copy before anything else touches data_dir.
    for _link_file in ("output_LinkMOE.csv", "output_link.csv",
                       "output_LinkTDMOE.csv"):
        _src = data_dir / _link_file
        if _src.is_file():
            shutil.copy(str(_src), str(result_dir / "predicted_LinkMOE.csv"))
            log("saved predicted link volumes to %s"
                % (result_dir / "predicted_LinkMOE.csv"))
            break
    else:
        log("warning: no link volume file found after MSA; "
            "predicted volumes not saved for Test 6")

    veh_hours, veh_note = vehicle_hours(data_dir, refreshed, log)
    routes_path = result_dir / "ue_routes.csv"
    backup_existing(routes_path)
    pd.DataFrame({agent_col: agent_ids,
                  path_col: path_strings()}).to_csv(routes_path, index=False)
    log("wrote %s (%d agents)" % (routes_path, n_agents))

    summary_path = result_dir / "ue_summary.txt"
    backup_existing(summary_path)
    with open(summary_path, "w") as fh:
        fh.write("Test 4: UE information baseline (MSA, Sheffi 1985)\n")
        fh.write("seed=%d, max_iters=%d, gap_threshold=%.6f, timeout=%ds\n"
                 % (args.seed, args.max_iters, args.gap_threshold, args.timeout))
        fh.write("agents: %d routed, %d kept on template path\n" % (n_switch, no_od))
        fh.write("iterations: %d\n" % iters_done)
        fh.write("final relative gap: %.6f\n" % gap)
        fh.write("converged: %s\n" % converged)
        if veh_hours is None:
            fh.write("vehicle-hours: n/a (%s)\n" % veh_note)
        else:
            fh.write("vehicle-hours: %.2f (%.4f per agent; %s)\n"
                     % (veh_hours, veh_hours / n_agents, veh_note))
        fh.write("elapsed: %.1f min\n" % ((time.time() - t_start) / 60.0))
        fh.write("---- console log ----\n")
        fh.write("\n".join(log.lines) + "\n")
    log("wrote %s" % summary_path)
    if veh_hours is None:
        log("vehicle-hours: n/a (%s)" % veh_note)
    else:
        log("vehicle-hours: %.2f" % veh_hours)


if __name__ == "__main__":
    sys.exit(main())
