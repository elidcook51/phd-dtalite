"""preference_reaction.py (merged M1)
==================================
Phase 2 ("Reaction") for Bingrong Sun (2018) Ch.4 baselines, Tests 2-4.

Phase 1 (guidance) is done by scenarios_ue.py / scenarios_realtime.py and
produces ue_routes.csv / realtime_routes.csv -- the recommended routes.
Here, users see that guidance and pick routes according to their TRUE
individual preferences, via the pairwise utility tournament ported from
comrealtimeassignment() in v9's realtimeassignment.py (lines ~269-334),
itself a port of Bingrong's MATLAB comrealtimeassignment.m.

Key modeling choices (documented so they are never ambiguous):
  * The tournament uses realweights (SiouxFalls_net.xlsx, 'weights' sheet,
    cols J:O) -- the TRUE preference weights -- not the estimated weights
    (cols A:F) used in the predict phase. The reaction measures what users
    would actually do given the guidance, so the true weights are right.
    (v9 passes realweights into comassignment as its `weights` argument,
    msa.py line 1129.)
  * Route attributes are [length, tt, pltt, fuelcost, nc]. tt is the sum of
    per-link mean travel times from the GUIDANCE run's output_LinkTDMOE.csv
    (free-flow fallback for unobserved links). v9's realtime loop uses the
    TD-link table at a single timestamp; the reaction is a single standalone
    pass, so the mean over timestamps is the documented analog.
  * pltt (planning time) = tt. v9 builds pltt from observed per-path agent
    times when they exist and sets pltt = tt when they do not (its itr == 1
    branch). The reaction has no per-path observations, so the no-observation
    branch applies.
  * fuelcost mirrors v9's traveltimecal mpg bands on realized link speed
    (>40 mph -> 32, <25 mph -> 22, else 27 mpg; gas = $3.0/gal).
  * output_LinkTDMOE.csv is parsed with the csv module, NOT pandas:
    DTALite writes a trailing comma (25 data fields vs 24 header columns),
    which silently shifts every pandas column left by one. (The repo's
    scenarios_ue.py already carries a ragged-CSV reader for this reason.)

Pipeline:
  1. Load guidance CSV (ue_routes.csv or realtime_routes.csv).
  2. Load demand (input_agent_initial.csv), choice set (.mat), user
     preferences and true weights (SiouxFalls_net.xlsx).
  3. Load per-link travel times from the GUIDANCE run's
     output_LinkTDMOE.csv (mean over timestamps; free-flow fallback).
     This MUST be read before the reaction DTALite run overwrites it.
  4. Per agent: pairwise tournament over the choice-set routes for the
     agent's OD -> actual choice.
  5. Write input_agent.csv (byte-identical to the template except the path
     column) and run DTALite.exe.
  6. Satisfaction = mean(actual == guidance, normalized) over agents with
     valid guidance; vehicle-hours from the actual run's output_agent.csv.
  7. Write actual_routes.csv and reaction_summary.txt.

Bingrong's satisfaction targets (Sun 2018, Ch.4):
  Test 4 (UE info)      66.54%
  Test 2 (RT, 1-min)    70.49%
  Test 3 (RT, 10-min)   70.19%

Usage:
  python preference_reaction.py --guidance-csv results_ue/ue_routes.csv \
      --data-dir <sandbox> --result-dir results_reaction_ue \
      --test-name "Test 4 (UE)" --target-satisfaction 66.54

Python 3.9, stdlib + numpy + pandas + scipy (+ openpyxl for xlsx).
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat

VERSION = "M1"

GAS_PRICE = 3.0          # $/gallon, matches v9's traveltimecal
MPG_FAST, MPG_MID, MPG_SLOW = 32.0, 27.0, 22.0   # speed bands 40 / 25 mph
DATASET2_CUTOFF = 16650  # pref >= this -> Dataset 2 (standardized diffs)

CHOICE_SET_MAT = "choice set no overlap try.mat"
NETWORK_XLSX = "SiouxFalls_net.xlsx"
AGENT_TEMPLATE = "input_agent_initial.csv"
LINK_TDMOE = "output_LinkTDMOE.csv"
AGENT_OUTPUT = "output_agent.csv"
AGENT_INPUT = "input_agent.csv"

# Outputs DTALite refreshes each run; deleted beforehand so a partial/failed
# run can never be mistaken for fresh results (discipline from scenarios_ue).
STALE_OUTPUTS = ("output_agent.csv", "output_link.csv",
                 "output_LinkTDMOE.csv", "Each iteration.csv")


class ReactionError(Exception):
    """Fail-fast error with a message the operator can act on."""


def fail(msg):
    raise ReactionError(msg)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def read_network(xlsx_path):
    """Link table from sheet 1, B90:E165 -> {(from, to): (length, freeflow)}.

    Same read as v9 (msa.py): sheet 0, cols B:E, skiprows=89, 76 rows ->
    (from_node, to_node, length, free-flow speed). freeflow is
    length / speed * 60 minutes; NaN when the speed is unusable (the
    attribute cache falls back to 1.0, mirroring traveltimecal).
    """
    if not os.path.isfile(xlsx_path):
        fail("network workbook not found: %s" % xlsx_path)
    try:
        tbl = pd.read_excel(
            xlsx_path, sheet_name=0, usecols="B:E",
            skiprows=89, nrows=76, header=None,
        ).to_numpy(dtype=float)
    except Exception as exc:
        fail("cannot read network sheet of %s: %s" % (xlsx_path, exc))
    links = {}
    for row in tbl:
        if not np.all(np.isfinite(row[:4])):
            continue
        a, b = int(row[0]), int(row[1])
        length, speed = float(row[2]), float(row[3])
        ff = length / speed * 60.0 if speed > 0 else float("nan")
        links[(a, b)] = (length, ff)
    if not links:
        fail("no usable links parsed from %s (B90:E165)" % xlsx_path)
    return links


def read_users(xlsx_path):
    """Sheet 'users' cols A:B -> {agent_id: pref}.

    Mirrors v9 exactly (msa.py line 397): the sheet HAS a header row
    ('No', 'Preference'), which pandas consumes with its default header,
    so data row i is agent i's row and user[agent_id, 1] is that agent's
    1-based preference id.
    """
    if not os.path.isfile(xlsx_path):
        fail("network workbook not found: %s" % xlsx_path)
    try:
        df = pd.read_excel(xlsx_path, sheet_name="users", usecols="A:B")
    except Exception as exc:
        fail("cannot read 'users' sheet of %s: %s" % (xlsx_path, exc))
    if df.shape[1] < 2:
        fail("sheet 'users' in %s has fewer than 2 columns" % xlsx_path)
    prefs = {}
    for i, pref in enumerate(df.iloc[:, 1]):
        try:
            p = int(float(pref))
        except (TypeError, ValueError):
            fail("bad preference %r on users row %d in %s"
                 % (pref, i + 2, xlsx_path))
        if p < 1:
            fail("preference must be 1-based, got %d for agent %d" % (p, i))
        prefs[i] = p
    if not prefs:
        fail("no agents in sheet 'users' of %s" % xlsx_path)
    return prefs


def read_realweights(xlsx_path):
    """TRUE preference weights: sheet 'weights' cols J:O, skiprows=5.

    The estimated weights live in cols A:F; the reaction uses the true
    ones. Returns an (n_pref, 6) float array; only the first 5 columns feed
    the tournament (real weights carry no intercept term). The sheet pads
    with extra rows / a NaN column O beyond the 16,650 preference types,
    so NaN is validated only on cols J:N, which the tournament reads.
    """
    try:
        arr = pd.read_excel(
            xlsx_path, sheet_name="weights", usecols="J:O",
            skiprows=5, header=None,
        ).to_numpy(dtype=float)
    except Exception as exc:
        fail("cannot read realweights (cols J:O) of %s: %s"
             % (xlsx_path, exc))
    if arr.shape[1] != 6:
        fail("expected 6 columns (J:O) for realweights, got %d"
             % arr.shape[1])
    if arr.size == 0 or np.isnan(arr[:, :5]).any():
        fail("NaN found in realweights data (sheet 'weights' cols J:N)")
    return arr


def read_meanstd2(xlsx_path):
    """Dataset-2 standardization: sheet 'weights' cols H:L, rows 2-3.

    Returns (2, 5): row 0 = means, row 1 = stds. Mirrors msa.py exactly;
    the zero std at index 3 divides to inf, the same way MATLAB's silent
    division does (the tournament suppresses the warning locally).
    """
    try:
        raw = pd.read_excel(
            xlsx_path, sheet_name="weights", usecols="H:L",
            skiprows=0, nrows=3, header=None,
        ).values
    except Exception as exc:
        fail("cannot read meanstd2 (cols H:L) of %s: %s" % (xlsx_path, exc))
    ms = raw[1:3].astype(float)
    if ms.shape != (2, 5) or np.isnan(ms).any():
        fail("could not parse meanstd2 from sheet 'weights' cols H:L rows 2-3")
    return ms


def parse_nodes(path):
    """Path string -> tuple of node ints, or None if unusable."""
    if path is None:
        return None
    if isinstance(path, np.ndarray):
        if path.size != 1:
            return None
        path = path.item()
    if not isinstance(path, str):
        return None
    try:
        nodes = tuple(int(float(p.strip())) for p in path.split(";")
                      if p.strip())
    except (TypeError, ValueError):
        return None
    return nodes if len(nodes) >= 2 else None


def normalize_path(nodes):
    return ";".join(str(n) for n in nodes) + ";"


def read_choice_set(data_dir):
    """Choice set .mat -> (routelocation, od_routes, od_of).

    Faithful port of v9's msa.py lines 339-373: the transposed finallist
    is scanned cell by cell; each valid route appends a routelocation row
    [origin, dest, 1, route_idx, od_idx]. Agents match by (origin, dest)
    against routelocation columns 0-1, take od_idx from column 4, and
    tournament over that OD's sorted route indices from column 3.
    """
    path = os.path.join(data_dir, CHOICE_SET_MAT)
    if not os.path.isfile(path):
        fail("choice-set file not found: %s" % path)
    try:
        mat = loadmat(path, simplify_cells=True)
    except Exception as exc:
        fail("cannot read %s: %s" % (path, exc))
    if "finallist" not in mat:
        fail("variable 'finallist' not found in %s" % path)
    choiceset = np.asarray(mat["finallist"], dtype=object).T
    if choiceset.ndim != 2:
        fail("'finallist' in %s is not 2-D" % path)
    n_od, n_routes = choiceset.shape
    rl_rows = []
    od_routes = {}   # od_idx -> {route_idx: nodes}
    od_of = {}       # od_idx -> (origin, dest)
    for od_idx in range(n_od):
        routes = {}
        for r in range(n_routes):
            nodes = parse_nodes(choiceset[od_idx, r])
            if nodes is None:
                continue
            routes[r] = nodes
            rl_rows.append([float(nodes[0]), float(nodes[-1]), 1.0,
                            float(r), float(od_idx)])
        if routes:
            od_routes[od_idx] = routes
            first = routes[sorted(routes)[0]]
            od_of[od_idx] = (first[0], first[-1])
    if not rl_rows:
        fail("no valid routes parsed from %s" % path)
    return np.array(rl_rows, dtype=float), od_routes, od_of


def read_link_times(data_dir):
    """Mean travel_time_in_min per directed link over all timestamps.

    Parsed with the csv module (not pandas): DTALite's LinkTDMOE rows
    carry a trailing comma -- 25 fields against a 24-column header --
    which silently shifts pandas' column parser by one. Column positions
    come from the header names, so the trailing empty field is ignored.
    """
    path = os.path.join(data_dir, LINK_TDMOE)
    if not os.path.isfile(path):
        fail("%s not found in %s -- run the guidance phase (DTALite) first"
             % (LINK_TDMOE, data_dir))
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        fail("no data rows in %s" % path)
    names = [h.strip() for h in rows[0]]
    for col in ("from_node_id", "to_node_id", "travel_time_in_min"):
        if col not in names:
            fail("column '%s' missing from %s" % (col, path))
    i_from = names.index("from_node_id")
    i_to = names.index("to_node_id")
    i_tt = names.index("travel_time_in_min")
    sums, counts = {}, {}
    for row in rows[1:]:
        try:
            a = int(row[i_from].strip())
            b = int(row[i_to].strip())
            tt = float(row[i_tt].strip())
        except (IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(tt):
            continue
        key = (a, b)
        sums[key] = sums.get(key, 0.0) + tt
        counts[key] = counts.get(key, 0) + 1
    if not sums:
        fail("no usable travel times in %s" % path)
    return {k: sums[k] / counts[k] for k in sums}


def read_guidance(guidance_csv):
    """{agent_id: normalized guidance path} from ue/realtime_routes.csv.

    Only agent_id and path_node_sequence are used, so both the 2-column
    ue format and the 4-column realtime format are accepted.
    """
    if not os.path.isfile(guidance_csv):
        fail("guidance csv not found: %s" % guidance_csv)
    try:
        df = pd.read_csv(guidance_csv)
    except Exception as exc:
        fail("cannot read %s: %s" % (guidance_csv, exc))
    df.columns = df.columns.str.strip()
    for col in ("agent_id", "path_node_sequence"):
        if col not in df.columns:
            fail("column '%s' missing from %s" % (col, guidance_csv))
    guidance = {}
    for _, row in df.iterrows():
        try:
            aid = int(float(str(row["agent_id"]).strip()))
        except (TypeError, ValueError):
            fail("bad agent_id %r in %s" % (row["agent_id"], guidance_csv))
        if aid in guidance:
            fail("duplicate agent_id %d in %s" % (aid, guidance_csv))
        nodes = parse_nodes(str(row["path_node_sequence"]))
        if nodes is None:
            fail("unparseable guidance path for agent %d in %s"
                 % (aid, guidance_csv))
        guidance[aid] = normalize_path(nodes)
    if not guidance:
        fail("no guidance routes parsed from %s" % guidance_csv)
    return guidance


def read_agent_template(data_dir):
    """Demand template -> ({agent_id: (o, d)}, {agent_id: raw path}, path).

    OD columns are looked up by name but asserted to sit at positions 4
    and 5, v9's positional convention (comrealtimeassignment reads
    S.iloc[:, [4, 5]]).
    """
    path = os.path.join(data_dir, AGENT_TEMPLATE)
    if not os.path.isfile(path):
        fail("agent template not found: %s" % path)
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        fail("no data rows in %s" % path)
    names = [h.strip() for h in rows[0]]
    for col in ("agent_id", "from_origin_node_id", "to_destination_node_id",
                "path_node_sequence"):
        if col not in names:
            fail("column '%s' missing from %s" % (col, path))
    i_from = names.index("from_origin_node_id")
    i_to = names.index("to_destination_node_id")
    if (i_from, i_to) != (4, 5):
        fail("OD columns of %s are at positions %d,%d, not v9's 4,5"
             % (path, i_from, i_to))
    id_col = names.index("agent_id")
    path_col = names.index("path_node_sequence")
    ods, template_paths = {}, {}
    for row in rows[1:]:
        try:
            aid = int(float(row[id_col].strip()))
            o = int(float(row[i_from].strip()))
            d = int(float(row[i_to].strip()))
        except (IndexError, TypeError, ValueError):
            fail("bad agent/OD row in %s" % path)
        if aid in ods:
            fail("duplicate agent_id %d in %s" % (aid, path))
        ods[aid] = (o, d)
        template_paths[aid] = row[path_col] if path_col < len(row) else ""
    if not ods:
        fail("no agents in %s" % path)
    return ods, template_paths, path


# ---------------------------------------------------------------------------
# Route attributes and the pairwise tournament
# ---------------------------------------------------------------------------

class AttributeCache:
    """Per-(od_idx, route) attribute vectors [length, tt, pltt, fuelcost, nc].

    length = sum of link lengths from the network table.
    tt     = sum of per-link mean times from the guidance DTALite run,
             falling back to free-flow (length/speed*60), then to 1.0 --
             traveltimecal's last-resort fallback.
    pltt   = tt (no per-path observations in a standalone pass; v9's
             no-observation branch).
    fuelcost mirrors traveltimecal: mpg picked from the realized link
             speed (length/tt*60): >40 -> 32, <25 -> 22, else 27.
    nc     = number of links.
    """

    def __init__(self, link_info, link_tt):
        self.link_info = link_info    # (a, b) -> (length, freeflow_tt)
        self.link_tt = link_tt        # (a, b) -> mean observed tt
        self.cache = {}

    def link_time(self, a, b):
        tt = self.link_tt.get((a, b))
        if tt is not None:
            return tt
        info = self.link_info.get((a, b))
        if info is not None and np.isfinite(info[1]):
            return info[1]
        return 1.0

    def get(self, od_idx, r, nodes):
        key = (od_idx, r)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        length = 0.0
        tt = 0.0
        fuel = 0.0
        links = list(zip(nodes[:-1], nodes[1:]))
        for a, b in links:
            info = self.link_info.get((a, b))
            seg_len = info[0] if info is not None else 0.0
            seg_tt = self.link_time(a, b)
            length += seg_len
            tt += seg_tt
            if seg_len > 0 and seg_tt > 0:
                speed = seg_len / seg_tt * 60.0
                if speed > 40:
                    mpg = MPG_FAST
                elif speed < 25:
                    mpg = MPG_SLOW
                else:
                    mpg = MPG_MID
                fuel += seg_len / mpg * GAS_PRICE
        vec = np.array([length, tt, tt, fuel, float(len(links))])
        # vec[2] is pltt = tt: see the class docstring.
        self.cache[key] = vec
        return vec


def tournament(od_idx, route_ids, wrow, dataset2, attr_cache, od_routes,
               meanstd2):
    """Pairwise utility tournament; returns the winning route index.

    Exact port of the tournament loop in v9's comrealtimeassignment
    (realtimeassignment.py lines 290-315): the champion starts as the
    first (lowest-index) route; each challenger replaces it when
    choice == 0, i.e. abs(ojvalue-1) > abs(ojvalue+1). diff = b1 - b0
    (champion minus challenger); dataset-2 agents (pref >= 16650)
    standardize the differences with meanstd2; ojvalue = sum(bscale*wrow)
    with no intercept ("real weights don't have intercept"). The
    onlylike0/onlylike1 debug lists in v9 are empty and are dropped here,
    as is the parallel minimum-travel-time (pre_choice) tournament.
    """
    champion = route_ids[0]
    b1 = attr_cache.get(od_idx, champion, od_routes[champion])
    for r in route_ids[1:]:
        b0 = attr_cache.get(od_idx, r, od_routes[r])
        diff = b1 - b0
        if dataset2:
            # Matches v9/MATLAB: the zero std at index 3 divides to inf
            # (0/0 to nan); the warning is suppressed exactly as v9 does.
            with np.errstate(divide="ignore", invalid="ignore"):
                bscale = (diff - meanstd2[0, :]) / meanstd2[1, :]
        else:
            bscale = diff
        ojvalue = float(np.sum(bscale * wrow))
        choice = 0 if abs(ojvalue - 1) > abs(ojvalue + 1) else 1
        if choice != 1:
            champion = r
            b1 = b0
    return champion


# ---------------------------------------------------------------------------
# input_agent.csv writer, DTALite runner, vehicle-hours
# ---------------------------------------------------------------------------

def write_input_agent(template_path, out_path, actual):
    """Rewrite the demand template with per-agent actual paths.

    Byte-identical to the template except for the path column (CRLF).
    actual maps agent_id -> node tuple. Fails fast if any template agent
    has no actual path.
    """
    with open(template_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        fail("empty demand template: %s" % template_path)
    header = rows[0]
    names = [h.strip() for h in header]
    if "agent_id" not in names:
        fail("column 'agent_id' not found in %s" % template_path)
    if "path_node_sequence" not in names:
        fail("column 'path_node_sequence' not found in %s" % template_path)
    id_col = names.index("agent_id")
    path_col = names.index("path_node_sequence")
    width = len(header)

    out_rows = []
    for row in rows[1:]:
        if len(row) < width:
            row = row + [""] * (width - len(row))
        try:
            aid = int(float(row[id_col].strip()))
        except (TypeError, ValueError):
            fail("bad agent_id %r in %s" % (row[id_col], template_path))
        if aid not in actual:
            fail("no actual route for agent_id %d" % aid)
        row[path_col] = normalize_path(actual[aid])
        out_rows.append(row[:width])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(header)
        writer.writerows(out_rows)


def dtalite_command(data_dir, exe_override):
    """Cross-platform DTALite launch (same discipline as scenarios_ue)."""
    if exe_override:
        if not os.path.isfile(exe_override):
            fail("DTALite executable not found: %s" % exe_override)
        return [exe_override]
    if sys.platform.startswith("win"):
        exe = os.path.join(data_dir, "DTALite.exe")
        if not os.path.isfile(exe):
            fail("DTALite.exe not found in %s" % data_dir)
        return [exe]
    native = os.path.join(data_dir, "DTALite")
    if os.path.isfile(native) and os.access(native, os.X_OK):
        return [native]
    exe = os.path.join(data_dir, "DTALite.exe")
    if os.path.isfile(exe) and shutil.which("wine"):
        return ["wine", exe]
    fail("no runnable DTALite found in %s (need ./DTALite, or DTALite.exe "
         "plus wine)" % data_dir)


def run_dtalite(data_dir, exe_override, timeout):
    """Run DTALite once; fail fast with a clear error on any failure."""
    for name in STALE_OUTPUTS:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            os.remove(p)
            print("removed stale %s" % name)
    cmd = dtalite_command(data_dir, exe_override)
    print("running DTALite (%s) ..." % " ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=data_dir, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        fail("DTALite timed out after %d s" % timeout)
    except OSError as exc:
        fail("could not launch DTALite (%s)" % exc)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else "no output captured"
        fail("DTALite exited with code %d: %s" % (proc.returncode, detail))
    print("DTALite finished in %.0f s" % (time.time() - t0))


def vehicle_hours(data_dir):
    """Total vehicle-hours from the actual run's output_agent.csv.

    Strict like scenarios_ue.py: any non-numeric travel time is a data
    problem and fails fast instead of silently undercounting. Guards the
    ragged-row case (data rows must match the header width) before the
    pandas read, since DTALite's link/agent CSVs are known to carry
    trailing commas.
    """
    path = os.path.join(data_dir, AGENT_OUTPUT)
    if not os.path.isfile(path):
        fail("%s not found after DTALite run" % path)
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        fail("no data rows in %s" % path)
    width = len(rows[0])
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != width:
            fail("%s line %d has %d fields, header has %d"
                 % (path, i, len(row), width))
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "travel_time_in_min" not in df.columns:
        fail("column 'travel_time_in_min' missing from %s" % path)
    minutes = pd.to_numeric(df["travel_time_in_min"], errors="coerce")
    n_bad = int(minutes.isna().sum())
    if n_bad:
        fail("%d non-numeric travel times in %s" % (n_bad, path))
    total_min = float(minutes.sum())
    n = len(df)
    return total_min / 60.0, total_min / n, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Preference-reaction phase: agents re-choose routes by a "
                    "pairwise utility tournament on TRUE preferences.")
    p.add_argument("--guidance-csv", required=True,
                   help="ue_routes.csv or realtime_routes.csv "
                        "(agent_id, path_node_sequence[, ...])")
    # DTALite_Files/ is read-only: every run must point --data-dir at a
    # writable sandbox copy. No default on purpose -- see the guard below.
    p.add_argument("--data-dir", default=None,
                   help="writable sandbox copy of the DTALite inputs; REQUIRED")
    p.add_argument("--result-dir", required=True,
                   help="output directory for actual_routes.csv and "
                        "reaction_summary.txt")
    p.add_argument("--test-name", default="reaction",
                   help="label used in the summary (e.g. 'Test 4 (UE)')")
    p.add_argument("--target-satisfaction", type=float, default=None,
                   help="Bingrong target %% for the summary, e.g. 66.54")
    p.add_argument("--exe", default=None,
                   help="DTALite executable override "
                        "(default: auto-detect in <data-dir>)")
    p.add_argument("--timeout", type=int, default=3600,
                   help="seconds to wait for DTALite (default 3600)")
    args = p.parse_args(argv)
    if not args.data_dir:
        p.error("--data-dir is required: pass a writable sandbox copy of the DTALite inputs (DTALite_Files/ is read-only)")
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    t_start = time.time()
    data_dir = args.data_dir
    result_dir = args.result_dir
    if not os.path.isdir(data_dir):
        fail("data directory not found: %s" % data_dir)
    os.makedirs(result_dir, exist_ok=True)

    log_lines = []
    log_lines.append("%s -- preference reaction (%s)"
                     % (args.test_name, VERSION))
    log_lines.append("guidance : %s" % args.guidance_csv)
    log_lines.append("data dir : %s" % data_dir)

    xlsx = os.path.join(data_dir, NETWORK_XLSX)
    print("loading network, users, weights ...")
    link_info = read_network(xlsx)
    prefs = read_users(xlsx)
    realweights = read_realweights(xlsx)
    meanstd2 = read_meanstd2(xlsx)
    print("loading choice set ...")
    routelocation, od_routes, od_of = read_choice_set(data_dir)
    print("loading link travel times ...")
    # From the GUIDANCE run's outputs: read before the reaction run
    # overwrites the DTALite output files.
    link_tt = read_link_times(data_dir)
    print("loading guidance and agent ODs ...")
    guidance = read_guidance(args.guidance_csv)
    agent_ods, template_paths, template_path = read_agent_template(data_dir)

    n_agents = len(agent_ods)
    if len(prefs) < n_agents:
        fail("sheet 'users' has %d rows but demand has %d agents"
             % (len(prefs), n_agents))

    log_lines.append("agents in demand file : %d" % n_agents)
    log_lines.append("agents with guidance   : %d" % len(guidance))
    log_lines.append("choice set             : %d OD cells, %d routelocation "
                     "rows" % (len(od_routes), routelocation.shape[0]))
    log_lines.append("link times             : %d links from guidance "
                     "output_LinkTDMOE.csv (mean over timestamps)"
                     % len(link_tt))
    log_lines.append("weights                : realweights (TRUE preferences, "
                     "cols J:O); predict phase used estimated cols A:F")

    cache = AttributeCache(link_info, link_tt)

    actual = {}          # agent_id -> chosen node tuple
    n_tournament = 0
    n_no_choice_set = 0
    for aid in sorted(agent_ods):
        if aid not in prefs:
            fail("agent_id %d missing from sheet 'users'" % aid)
        pref = prefs[aid]
        if pref < 1 or pref - 1 >= len(realweights):
            fail("pref %d for agent %d outside realweights (%d rows)"
                 % (pref, aid, len(realweights)))
        wrow = realweights[pref - 1, :5]
        dataset2 = pref >= DATASET2_CUTOFF
        o, d = agent_ods[aid]
        matches = np.where((routelocation[:, 0] == o)
                           & (routelocation[:, 1] == d))[0]
        routes = None
        if len(matches):
            f = int(routelocation[matches[0], 4])
            routes = od_routes.get(f)
        if not routes:
            # No choice-set routes for this OD: the agent keeps the guided
            # path, else the template's own path as a last resort.
            n_no_choice_set += 1
            g = guidance.get(aid)
            nodes = (parse_nodes(g) if g is not None
                     else parse_nodes(template_paths[aid]))
            if nodes is None:
                fail("agent %d has no choice set and no fallback path" % aid)
            actual[aid] = nodes
            continue
        route_ids = sorted(routes.keys())
        winner = tournament(f, route_ids, wrow, dataset2, cache, routes,
                            meanstd2)
        actual[aid] = routes[winner]
        n_tournament += 1

    # ---- satisfaction (actual vs guidance, normalized both sides) ----
    n_guided = 0
    n_satisfied = 0
    rows = []
    for aid in sorted(agent_ods):
        actual_path = normalize_path(actual[aid])
        g = guidance.get(aid)
        if g is None:
            rows.append((aid, "", actual_path, ""))
        else:
            sat = 1 if actual_path == g else 0
            n_guided += 1
            n_satisfied += sat
            rows.append((aid, g, actual_path, str(sat)))
    satisfaction = 100.0 * n_satisfied / n_guided if n_guided else 0.0
    log_lines.append("tournament run for     : %d agents" % n_tournament)
    log_lines.append("no choice-set OD       : %d agents (kept guidance path)"
                     % n_no_choice_set)

    print("writing %s ..." % AGENT_INPUT)
    write_input_agent(template_path,
                      os.path.join(data_dir, AGENT_INPUT), actual)
    log_lines.append("wrote %s (%d agents)"
                     % (os.path.join(data_dir, AGENT_INPUT), n_agents))

    print("running DTALite ...")
    run_dtalite(data_dir, args.exe, args.timeout)
    log_lines.append("DTALite finished (rc=0)")

    vh, per_agent_min, n_out = vehicle_hours(data_dir)

    os.makedirs(result_dir, exist_ok=True)
    actual_csv = os.path.join(result_dir, "actual_routes.csv")
    with open(actual_csv, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(
            ["agent_id", "guidance_route", "actual_route", "satisfied"])
        writer.writerows(rows)
    log_lines.append("wrote %s (%d agents)" % (actual_csv, len(rows)))

    summary_path = os.path.join(result_dir, "reaction_summary.txt")
    lines = list(log_lines)
    lines.append("")
    lines.append("satisfaction: %d / %d = %.2f%%"
                 % (n_satisfied, n_guided, satisfaction))
    if args.target_satisfaction is not None:
        lines.append("Bingrong target: %.2f%%  (diff %+.2f pp)"
                     % (args.target_satisfaction,
                        satisfaction - args.target_satisfaction))
    lines.append("vehicle-hours (actual run): %.2f" % vh)
    lines.append("per-agent travel time     : %.2f min/agent (n=%d)"
                 % (per_agent_min, n_out))
    lines.append("elapsed: %.1f min" % ((time.time() - t_start) / 60.0))
    with open(summary_path, "w", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print("")
    print("\n".join(lines))
    print("wrote %s" % actual_csv)
    print("wrote %s" % summary_path)


if __name__ == "__main__":
    try:
        main()
    except ReactionError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        sys.exit(1)
