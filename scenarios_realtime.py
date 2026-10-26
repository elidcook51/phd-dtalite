"""Travel-time-based real-time route guidance baselines (Tests 2 and 3).

Two-shot (collapsed) per-interval X_{m-1} implementation of the real-time
baseline experiment from Bingrong Sun's 2018 UVA PhD dissertation, Chapter 4
("travel time based real-time route guidance"): the evaluation period is
split into intervals of sigma minutes, and every agent departing in
interval m is assigned the shortest-travel-time route from its choice set
under X_{m-1}, the link travel times observed during the previous interval.
X_{-1} does not exist, so agents departing in interval 0 fall back to
free-flow times (the dissertation's "historical traffic conditions").

Deliberate approximations (not an exact rebuild of the MATLAB experiment)
------------------------------------------------------------------------
The MATLAB reference (realtimeassignment.m) runs DTALite ONCE PER PERIOD
(up to 60 runs for sigma = 1), reading each period's X_{m-1} from the
previous period's outputs, in which earlier periods' agents already follow
guidance -- so guidance feeds back into the information.  This script runs
DTALite TWICE TOTAL: once under the free-flow assignment, once under the
reactive assignment.  All X_{m-1} tables derive from that single
counterfactual all-free-flow run; guidance does not feed back into the
information tables.  A faithful rolling loop is ~60 DTALite runs for
sigma = 1, which was judged prohibitive -- the two-shot collapse is a
deliberate, documented approximation (a per-period rolling loop remains a
possible future change).

Related accepted approximations:
- X_{m-1} here is link-level interval-means summed statically over a route;
  MATLAB prefers route-level experienced means at the point timestamp
  359 + m*sigma, falling back to a link-sum with a forward time-walk per
  traveltimecal.m.  For sigma = 10 MATLAB uses a single point timestamp,
  not a 10-minute mean.
- Route evaluation is static-bin: every link of a route is looked up in the
  single previous-interval table X_{m-1}, while MATLAB's traveltimecal.m
  walks the links in order, looking up each link at
  floor(timestamp + cumulative time) (dissertation Eq. 21).  The Eq. 25
  prose is ambiguous -- this script follows the prose's static-bin reading,
  which diverges from the MATLAB code.  Noted as accepted; not reimplemented.
- Period-1 (interval 0) agents use free-flow times; MATLAB sources period-1
  information from the UO search's predicted congested conditions
  (Each iteration.csv), measured at 1.07x-1.95x free-flow at ts = 360, which
  can flip first-interval route rankings (~600 agents at sigma = 1).
  Rewiring period-1 sourcing was judged not worth the extra cost; see
  assign_paths.
- Agent-experienced route times are not used: MATLAB sets tt{i,h} to the
  mean of agents' experienced travel times for observed routes; this script
  always sums per-link means.  Close but not identical.
- Interval boundaries are left-closed/right-open (m = floor(t / sigma));
  MATLAB phases are left-open/right-closed, so t = 360.0 exactly falls in no
  MATLAB phase.  419/36,030 agents (1.16%) depart at exact integer minutes;
  accepted as negligible.

Procedure
---------
1. Assign every agent its free-flow shortest choice-set route, write
   input_agent.csv, and run DTALite once to collect output_LinkTDMOE.csv.
2. For each interval m >= 1, X_{m-1} maps every directed link to the mean
   of travel_time_in_min over rows with
   (m-1)*sigma <= timestamp_in_min < m*sigma  (left-closed, right-open).
3. Re-assign: an agent departing at time t falls in interval
   m = floor(t / sigma) and takes the shortest route under X_{m-1}
   (free-flow times when m == 0, and free-flow for any single link with
   no observations in the bin).
4. Write the reactive plan to input_agent.csv, run DTALite once more, and
   report total vehicle-hours from output_agent.csv.

This file merges two independently built implementations (builder A and
builder B, cross-validated to produce bit-identical assignments on 3,000
agents for sigma = 1 and sigma = 10): it keeps A's input parsing
(fuzzy column detection, strict ragged-CSV reader, robust choice-set
loading), multi-candidate DTALite launcher, per-(interval, OD) route
caching, and failure diagnostics, and takes B's byte-identical template
writer for input_agent.csv plus its fail-fast checks (duplicate agent
ids, empty LinkTDMOE).

Targets (Bingrong 2018, 54,045 agents): sigma=1 -> 26,983.1 vehicle-hours,
sigma=10 -> 28,039.6 vehicle-hours.  This repo's demand has ~36k agents,
so the summary compares per-agent travel times.  Note the Table 4 targets
are actual-traffic vehicle-hours under true-preference reactions
(Table 3), while this script measures predicted traffic under
shortest-route recommendations -- the "% vs target" line is a rough
comparison only; the valid comparison comes after the
preference-reaction step.

Usage:
    python scenarios_realtime.py --sigma 1 --data-dir <sandbox> \
        --result-dir results/realtime_1
    python scenarios_realtime.py --sigma 10 --data-dir <sandbox>
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys

import netconfig

import engine as _engine_mod
import numpy as np
import pandas as pd
import openpyxl
from scipy.io import loadmat


# Bingrong (2018) Table 4 totals, measured over 54,045 agents.
TARGET_VEHICLE_HOURS = {1: 26983.1, 10: 28039.6}
TARGET_N_AGENTS = 54045

DTALITE_TIMEOUT_S = 5400


class ScenarioError(RuntimeError):
    """Fail-fast error with a message suitable for printing to the user."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def detect_column(df, candidates):
    """Find a DataFrame column by exact, case-insensitive, then substring match."""
    cols = list(df.columns)
    lowered = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for cand in candidates:
        for c in cols:
            if cand in str(c).strip().lower():
                return c
    raise ScenarioError(
        "could not find any of %s among columns %s" % (candidates, cols)
    )


def read_ragged_csv(path, required_columns):
    """Read a CSV that may carry DTALite's stray trailing commas.

    Rows longer than the header are trimmed (only when the extra fields
    are empty; non-empty extras raise); shorter rows are padded.
    Quoting-aware via the csv module.
    """
    if not os.path.isfile(path):
        raise ScenarioError("required file not found: %s" % path)
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ScenarioError("empty CSV file: %s" % path)
    header = [c.strip() for c in rows[0]]
    width = len(header)
    records = []
    for lineno, row in enumerate(rows[1:], start=2):
        if len(row) > width:
            extra = row[width:]
            if any(cell.strip() != "" for cell in extra):
                raise ScenarioError(
                    "%s line %d: non-empty fields beyond the %d-column header"
                    % (path, lineno, width)
                )
            row = row[:width]
        elif len(row) < width:
            row = row + [""] * (width - len(row))
        records.append(row)
    df = pd.DataFrame(records, columns=header)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ScenarioError(
            "%s is missing required columns %s (found %s)"
            % (path, missing, list(df.columns))
        )
    return df


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def read_free_flow_times_baltimore(csv_path):
    """Free-flow minutes per directed link from Baltimore input_link.csv.

    Returns {(from_node, to_node): free_flow_minutes}.
    Uses free_flow_time column if present, else length/free_speed*60.
    """
    import csv as _csv
    if not os.path.isfile(csv_path):
        raise ScenarioError("network file not found: %s" % csv_path)
    out = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in _csv.DictReader(fh):
            try:
                u = int(float(row["from_node_id"]))
                v = int(float(row["to_node_id"]))
                fft_raw = row.get("free_flow_time")
                if fft_raw:
                    fft = float(fft_raw)
                else:
                    fft = float(row["length"]) / float(row["free_speed"]) * 60.0
            except (ValueError, TypeError, KeyError):
                continue
            if fft > 0 and (u, v) not in out:
                out[(u, v)] = fft
    return out


def read_choice_set_baltimore(data_dir):
    """Choice set as dict OD (oz, dz) -> list of (nodes, links) routes.

    Loads choice_sets.pkl {(oz,dz): [[nodes], ...]} and converts each
    path to (nodes_tuple, [(n1,n2), ...]) to match read_choice_set format.
    """
    import pickle as _pickle
    pkl_path = os.path.join(data_dir, "choice_sets.pkl")
    if not os.path.isfile(pkl_path):
        raise ScenarioError("choice-set file not found: %s" % pkl_path)
    with open(pkl_path, "rb") as fh:
        d = _pickle.load(fh)
    out = {}
    for (oz, dz), paths in d.items():
        for p in paths:
            if p is None:
                continue
            nodes = tuple(int(x) for x in p)
            if len(nodes) < 2:
                continue
            links = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
            # Key by zone OD (the pkl keys), matching the zone-based
            # agent lookup in read_demand. Keying by path node
            # endpoints merged different zone ODs that share
            # endpoints and missed agents whose origin/destination
            # node is not a path endpoint.
            key = (float(oz), float(dz))
            out.setdefault(key, [])
            # Avoid duplicate routes for the same zone OD
            if (nodes, links) not in out[key]:
                out[key].append((nodes, links))
    return out


def read_free_flow_times(xlsx_path):
    """Free-flow minutes per directed link from SiouxFalls_net.xlsx.

    Classic TNTP layout: a '~' header row, then (init node, term node,
    length, speed) rows.  Free-flow time = length / speed * 60.
    """
    if not os.path.isfile(xlsx_path):
        raise ScenarioError("network file not found: %s" % xlsx_path)
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["SiouxFalls_net"] if "SiouxFalls_net" in wb.sheetnames else wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    header_row = None
    for i, row in enumerate(rows):
        if row and str(row[0]).strip() == "~":
            header_row = i
            break
    if header_row is None:
        raise ScenarioError("could not locate the '~' link table header in %s" % xlsx_path)

    free_flow = {}
    # The intended table is the 'link length information' block: rows of
    # (init node, term node, length, speed) with free-flow time computed as
    # length / speed * 60. The '~'-headed block above it carries
    # (capacity, length, ...) in the same columns, so it must NOT be read
    # with this column mapping (the old last-wins loop was correct only by
    # row-order accident: block 2 overwrote block 1's misread values).
    marker = None
    # Search only below the '~' link-table header: a matching phrase in a
    # note/description row above the real block must not redirect the read
    # into the capacity block with the (length, speed) column mapping.
    for i in range(header_row + 1, len(rows)):
        row = rows[i]
        if row and any("link length information" in str(c).lower()
                       for c in row if c is not None):
            marker = i
            break
    if marker is None:
        raise ScenarioError(
            "could not locate the 'link length information' block in %s; "
            "free-flow times are computed as length / speed * 60 from that "
            "block" % xlsx_path)
    n_skipped = 0
    for row in rows[marker + 1:]:
        if len(row) < 5:
            if free_flow:
                break  # end of the link-length block
            continue   # blank rows between the marker and the data
        f, t, length, speed = row[1], row[2], row[3], row[4]
        if (not isinstance(f, (int, float)) or isinstance(f, bool)
                or not isinstance(t, (int, float)) or isinstance(t, bool)
                or pd.isna(f) or pd.isna(t)):
            if free_flow:
                break  # label/footer row ends the block
            continue   # junk rows above the block
        try:
            length, speed = float(length), float(speed)
        except (TypeError, ValueError):
            n_skipped += 1
            continue  # malformed link row: skip, never ends the block
        if pd.isna(length) or pd.isna(speed):
            n_skipped += 1
            continue  # never overwrite a good value with NaN
        if speed <= 0:
            raise ScenarioError("non-positive speed for link %s -> %s" % (f, t))
        key = (int(f), int(t))
        if key in free_flow:
            n_skipped += 1
            continue  # keep first; never overwrite a good value
        free_flow[key] = length / speed * 60.0
    if n_skipped:
        print("WARNING: skipped %d malformed/duplicate rows in the "
              "'link length information' block of %s"
              % (n_skipped, xlsx_path), file=sys.stderr)
    if not free_flow:
        raise ScenarioError("no links parsed from %s" % xlsx_path)
    return free_flow


def read_demand(csv_path):
    """Read the agent template; fail fast on NaN agent ids / departures."""
    df = pd.read_csv(csv_path)
    agent_col = detect_column(df, ["agent_id"])
    if netconfig.is_baltimore():
        # Baltimore choice sets are keyed by zone OD (see
        # read_choice_set_baltimore), so the agent OD is the zone pair.
        origin_col = detect_column(df, ["from_zone_id"])
        dest_col = detect_column(df, ["to_zone_id"])
    else:
        origin_col = detect_column(df, ["from_origin_node_id", "origin_node_id"])
        dest_col = detect_column(df, ["to_destination_node_id", "destination_node_id"])
    dep_col = detect_column(df, ["departure_time_in_min", "departure_time"])
    path_col = detect_column(df, ["path_node_sequence"])

    df["_agent_id"] = pd.to_numeric(df[agent_col], errors="coerce")
    df["_departure"] = pd.to_numeric(df[dep_col], errors="coerce")
    if df["_agent_id"].isna().any():
        raise ScenarioError("NaN agent_id in %s" % csv_path)
    if df["_departure"].isna().any():
        bad = df.loc[df["_departure"].isna(), agent_col].tolist()[:5]
        raise ScenarioError(
            "NaN departure_time_in_min for agent(s) %s in %s" % (bad, csv_path)
        )
    df["_origin"] = pd.to_numeric(df[origin_col], errors="coerce")
    df["_dest"] = pd.to_numeric(df[dest_col], errors="coerce")
    df["_od"] = list(zip(df["_origin"].to_numpy(), df["_dest"].to_numpy()))
    meta = {"agent_col": agent_col, "path_col": path_col}
    return df, meta


def parse_route_string(text):
    """'1;2;3;' -> [1, 2, 3]; returns [] when unparseable."""
    nodes = []
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            nodes.append(int(float(part)))
        except (TypeError, ValueError):
            return []
    return nodes


def read_choice_set(mat_path):
    """Choice set as dict OD (float, float) -> list of (nodes, links) routes.

    Loads mat['finallist'].T -> (n_ods, n_routes); each cell holds a
    ';'-separated node string.  A row's OD is read from the endpoints of
    its first route: origin = float(nodes[0]), dest = float(nodes[-1]).
    Duplicate routes within a row are kept once.
    """
    if not os.path.isfile(mat_path):
        raise ScenarioError("choice-set file not found: %s" % mat_path)
    mat = loadmat(mat_path)
    if "finallist" not in mat:
        raise ScenarioError("finallist not found in %s" % mat_path)
    table = mat["finallist"].T
    od_routes = {}
    for r in range(table.shape[0]):
        routes = []
        seen = set()
        for c in range(table.shape[1]):
            cell = np.asarray(table[r, c]).ravel()
            if cell.size == 0:
                continue
            nodes = parse_route_string(cell[0])
            if len(nodes) < 2 or tuple(nodes) in seen:
                continue
            seen.add(tuple(nodes))
            links = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
            routes.append((nodes, links))
        if not routes:
            continue
        od = (float(routes[0][0][0]), float(routes[0][0][-1]))
        od_routes[od] = routes
    if not od_routes:
        raise ScenarioError("no routes parsed from %s" % mat_path)
    return od_routes


# ---------------------------------------------------------------------------
# DTALite
# ---------------------------------------------------------------------------

def run_dtalite(data_dir, timeout=DTALITE_TIMEOUT_S, engine=None):
    """Run DTALite once; fail fast with a clean error on any failure.

    The executable is verified BEFORE any stale output_* files are deleted,
    so a missing-exe run cannot destroy prior outputs.  Candidates that
    exit non-zero fall through to the next one (e.g. a non-zero wine exit
    does not preclude trying the native binary).
    """
    _engine = _engine_mod.resolve(engine)
    if _engine == "pysim":
        # Verify the PySim runner exists BEFORE deleting stale outputs,
        # mirroring the executable check below for the DTALite path.
        try:
            _engine_mod.pysim_command(data_dir)
        except RuntimeError as exc:
            raise ScenarioError(str(exc))
        for name in sorted(os.listdir(data_dir)):
            if name.startswith("output_"):
                path = os.path.join(data_dir, name)
                if os.path.isfile(path):  # never os.remove a directory
                    os.remove(path)
        try:
            _engine_mod.run_pysim(data_dir, timeout, "scenarios_realtime")
        except RuntimeError as exc:
            raise ScenarioError(str(exc))
        return

    # Verify an executable exists first; a missing-exe run must not
    # delete output_* files and then raise.
    exe_path = os.path.join(data_dir, "DTALite.exe")
    commands = []
    if os.path.isfile(exe_path):
        commands.append([exe_path])  # native launch (Windows)
        if shutil.which("wine") is not None:
            commands.append(["wine", exe_path])  # wine fallback (Linux)
    alt_path = os.path.join(data_dir, "DTALite")
    if os.path.isfile(alt_path) and os.access(alt_path, os.X_OK):
        commands.append([alt_path])
    if not commands:
        raise ScenarioError(
            "DTALite executable not found in %s "
            "(looked for DTALite.exe and DTALite)" % data_dir
        )

    for name in sorted(os.listdir(data_dir)):
        if name.startswith("output_"):
            path = os.path.join(data_dir, name)
            if os.path.isfile(path):  # never os.remove a directory
                os.remove(path)

    last_error = None
    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                cwd=data_dir,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            # e.g. Exec format error on Linux -> try the next candidate.
            last_error = exc
            continue
        except subprocess.TimeoutExpired:
            raise ScenarioError("DTALite timed out after %d seconds" % timeout)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            last_error = ScenarioError(
                "DTALite failed (exit code %d): %s" % (proc.returncode, detail[-2000:])
            )
            continue  # fall through to the next candidate
        return
    raise ScenarioError("could not launch DTALite in %s: %s" % (data_dir, last_error))


def write_input_agent(template_path, out_path, agent_col, path_col, paths):
    """Rewrite the demand template with per-agent paths, byte-identical otherwise.

    The template is re-read with the csv module (not the parsed DataFrame),
    so the header and row order are preserved exactly; only the path column
    changes.  paths maps agent_id -> node list; None keeps the template's
    own path cell untouched.  Fails fast on duplicate agent ids in the
    template or an agent with no assigned path.

    Note: rows longer than the header are silently trimmed to the header
    width, while read_ragged_csv raises on non-empty extras -- the
    strictness is asymmetric by design (the writer preserves template
    shape; the reader guards input data).
    """
    if not os.path.isfile(template_path):
        raise ScenarioError("demand template not found: %s" % template_path)
    with open(template_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ScenarioError("empty demand template: %s" % template_path)
    header = rows[0]
    names = [h.strip() for h in header]
    if agent_col.strip() not in names:
        raise ScenarioError(
            "column '%s' not found in %s" % (agent_col, template_path)
        )
    if path_col.strip() not in names:
        raise ScenarioError(
            "column '%s' not found in %s" % (path_col, template_path)
        )
    id_col = names.index(agent_col.strip())
    path_idx = names.index(path_col.strip())
    width = len(header)

    seen = set()
    out_rows = []
    for row in rows[1:]:
        if len(row) < width:
            row = row + [""] * (width - len(row))
        try:
            aid = int(float(row[id_col].strip()))
        except (TypeError, ValueError):
            raise ScenarioError(
                "bad agent_id %r in %s" % (row[id_col], template_path)
            )
        if aid in seen:
            raise ScenarioError(
                "duplicate agent_id %d in %s" % (aid, template_path)
            )
        seen.add(aid)
        if aid not in paths:
            raise ScenarioError("no route assigned for agent_id %d" % aid)
        nodes = paths[aid]
        if nodes is not None:
            row[path_idx] = ";".join(str(n) for n in nodes) + ";"
        out_rows.append(row[:width])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(header)
        writer.writerows(out_rows)


# ---------------------------------------------------------------------------
# X_{m-1} construction and reactive assignment
# ---------------------------------------------------------------------------

def build_interval_link_times(tdmoe_df, sigma):
    """Mean travel time per directed link per departure-time interval.

    Returns dict interval_key -> {(from, to): mean minutes}, where
    interval_key = floor(timestamp_in_min / sigma).  X_{m-1} for agents in
    interval m is table[m - 1], i.e. rows with
    (m-1)*sigma <= timestamp < m*sigma (left-closed, right-open).

    Note: each bin entry is an unweighted mean over the observed minutes --
    a minute with little traffic counts as much as a busy one.
    """
    df = tdmoe_df.copy()
    df["from_node_id"] = pd.to_numeric(df["from_node_id"], errors="coerce")
    df["to_node_id"] = pd.to_numeric(df["to_node_id"], errors="coerce")
    df["timestamp_in_min"] = pd.to_numeric(df["timestamp_in_min"], errors="coerce")
    df["travel_time_in_min"] = pd.to_numeric(df["travel_time_in_min"], errors="coerce")
    df = df.dropna(
        subset=["from_node_id", "to_node_id", "timestamp_in_min", "travel_time_in_min"]
    )
    if df.empty:
        return {}
    df["from_node_id"] = df["from_node_id"].astype(int)
    df["to_node_id"] = df["to_node_id"].astype(int)
    df["interval"] = np.floor(df["timestamp_in_min"].to_numpy() / sigma).astype(int)
    grouped = df.groupby(["interval", "from_node_id", "to_node_id"])[
        "travel_time_in_min"
    ].mean()
    table = {}
    for (m, f, t), value in grouped.items():
        table.setdefault(int(m), {})[(int(f), int(t))] = float(value)
    return table


def route_travel_time(links, link_times, free_flow):
    """Sum of link minutes; links missing from the bin fall back to free-flow.

    Accepted approximation: MATLAB sets tt{i,h} to the mean of agents'
    experienced travel times for observed routes, while this always sums
    per-link means.  Close but not identical.
    """
    total = 0.0
    for key in links:
        if key in link_times:
            total += link_times[key]
        elif key in free_flow:
            total += free_flow[key]
        else:
            raise ScenarioError(
                "route uses link %d -> %d which is not in the network" % key
            )
    return total


def assign_paths(agents, od_routes, free_flow, interval_table, sigma):
    """Assign each agent the shortest choice-set route under X_{m-1}.

    agents carries _agent_id, _od, _departure, interval_m = floor(t / sigma).
    Returns (paths, stats); paths maps agent_id -> (nodes|None, m), where
    None keeps the template path and is counted as a missing OD.  The best
    route is cached per (interval, OD), so each OD is evaluated once per
    interval rather than once per agent.  On exact travel-time ties the
    LATER route wins, matching MATLAB's pre_choice convention (not Python
    min()'s keep-the-first behavior).
    """
    paths = {}
    n_missing_od = 0
    n_interval_zero = 0
    n_empty_bin = 0
    for m, group in agents.groupby("interval_m", sort=True):
        m = int(m)
        if m == 0:
            # X_{-1} does not exist: use free-flow (historical) times.
            # Period-1 substitution (M-RT1): MATLAB sources period-1
            # information from the UO search's predicted congested
            # conditions (Each iteration.csv), not free-flow -- measured at
            # 1.07x-1.95x free-flow at ts = 360, which can flip first-
            # interval route rankings (~600 agents at sigma = 1).  Rewiring
            # period-1 sourcing was judged not worth the extra cost; the
            # free-flow substitution stands.  (Also note this branch is dead
            # code in practice: departures are 360-420 min, so
            # floor(t / sigma) >= 36 always.)
            link_times = {}
            n_interval_zero += len(group)
        else:
            link_times = interval_table.get(m - 1, {})
            if not link_times:
                n_empty_bin += len(group)
        best_for_od = {}
        for od in group["_od"].unique():
            routes = od_routes.get(tuple(od))
            if not routes:
                best_for_od[tuple(od)] = None
                continue
            best_nodes = None
            best_time = None
            for route in routes:
                t = route_travel_time(route[1], link_times, free_flow)
                # On exact ties keep the LATER route, matching MATLAB's
                # pre_choice convention (keep the old best only on a
                # strictly smaller new time; take the new route otherwise).
                if best_time is None or t <= best_time:
                    best_nodes = route[0]
                    best_time = t
            best_for_od[tuple(od)] = best_nodes
        for _, row in group.iterrows():
            nodes = best_for_od.get(tuple(row["_od"]))
            if nodes is None:
                n_missing_od += 1
            paths[int(row["_agent_id"])] = (nodes, m)
    stats = {
        "n_missing_od": n_missing_od,
        "n_interval_zero": n_interval_zero,
        "n_empty_bin": n_empty_bin,
    }
    return paths, stats


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def read_vehicle_hours(output_agent_path):
    """Total vehicle-hours from output_agent.csv (sum of travel_time / 60)."""
    df = read_ragged_csv(output_agent_path, [])
    tt_col = detect_column(df, ["travel_time_in_min", "travel_time"])
    minutes = pd.to_numeric(df[tt_col], errors="coerce")
    if minutes.isna().any():
        raise ScenarioError("NaN travel times in %s" % output_agent_path)
    return float(minutes.sum()) / 60.0, len(df)


def write_outputs(result_dir, sigma, agents, paths, template_df, meta,
                  vehicle_hours, n_simulated, stats, seed):
    """Write realtime_routes.csv and realtime_summary.txt; return key numbers."""
    os.makedirs(result_dir, exist_ok=True)

    template_paths = template_df.set_index(
        pd.to_numeric(template_df[meta["agent_col"]]).to_numpy()
    )[meta["path_col"]]
    path_strings = {}
    for agent_id, (nodes, m) in paths.items():
        if nodes is None:
            path_strings[agent_id] = str(template_paths.loc[float(agent_id)])
        else:
            path_strings[agent_id] = ";".join(str(n) for n in nodes) + ";"

    routes_path = os.path.join(result_dir, "realtime_routes.csv")
    with open(routes_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "path_node_sequence", "sigma_min", "interval_m"])
        for agent_id in sorted(path_strings):
            writer.writerow(
                [agent_id, path_strings[agent_id], sigma, paths[agent_id][1]]
            )

    target = TARGET_VEHICLE_HOURS[sigma]
    target_per_agent_min = target / TARGET_N_AGENTS * 60.0
    mine_per_agent_min = vehicle_hours / n_simulated * 60.0
    pct_diff = (
        (mine_per_agent_min - target_per_agent_min) / target_per_agent_min * 100.0
    )

    summary_path = os.path.join(result_dir, "realtime_summary.txt")
    with open(summary_path, "w", newline="") as f:
        f.write("Travel-time-based real-time route guidance baseline\n")
        f.write("Bingrong Sun (2018), PhD dissertation, Chapter 4\n")
        f.write("\n")
        f.write("sigma (information update interval): %d minute(s)\n" % sigma)
        f.write("seed: %d (deterministic script; accepted for API parity)\n" % seed)
        f.write("agents in demand file: %d\n" % len(agents))
        f.write("agents simulated by DTALite: %d\n" % n_simulated)
        f.write(
            "agents in interval m = 0 (free-flow fallback; always 0 here, "
            "departures start at t = 360 min): %d\n"
            % stats["n_interval_zero"]
        )
        f.write(
            "agents with empty X_{m-1} bin (fell back to free-flow): %d\n"
            % stats["n_empty_bin"]
        )
        f.write(
            "agents with no valid OD in choice set (kept template path): %d\n"
            % stats["n_missing_od"]
        )
        f.write("\n")
        f.write("total travel time: %.2f vehicle-hours\n" % vehicle_hours)
        f.write("per-agent travel time: %.2f min/agent\n" % mine_per_agent_min)
        f.write("\n")
        f.write(
            "Bingrong (2018) Table 4 target: %.1f vehicle-hours over %d "
            "agents (actual traffic under true-preference reactions; rough "
            "comparison -- this run measures predicted traffic under "
            "shortest-route recommendations)\n" % (target, TARGET_N_AGENTS)
        )
        f.write("target per-agent travel time: %.2f min/agent\n" % target_per_agent_min)
        f.write(
            "difference vs Table-4 actual-traffic target "
            "(rough; predicted traffic only): %+.1f%%\n" % pct_diff
        )
    return routes_path, summary_path, mine_per_agent_min, pct_diff


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Real-time route guidance baselines (sigma = 1 or 10 min)."
    )
    parser.add_argument(
        "--sigma",
        type=int,
        required=True,
        choices=(1, 10),
        help="information update interval in minutes (Test 2 = 1, Test 3 = 10)",
    )
    # DTALite_Files/ is read-only: every run must point --data-dir at a
    # writable sandbox copy. No default on purpose -- see the guard below.
    parser.add_argument(
        "--data-dir",
        default=None,
        help="writable sandbox copy of the DTALite inputs; REQUIRED",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="output directory (default: ./results_realtime_<sigma>min)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (accepted for API parity with scenarios_ue.py; "
             "this script is fully deterministic and ignores it)",
    )
    parser.add_argument(
        "--network",
        default="siouxfalls",
        choices=["siouxfalls", "baltimore"],
        help="network: siouxfalls (default) or baltimore",
    )
    parser.add_argument(
        "--engine",
        default="pysim",
        choices=["pysim", "dtalite"],
        help="simulation engine: pysim (default) or dtalite",
    )
    args = parser.parse_args(argv)
    if not args.data_dir:
        parser.error("--data-dir is required: pass a writable sandbox copy of the DTALite inputs (DTALite_Files/ is read-only)")
    return args


def main(argv=None):
    args = parse_args(argv)
    netconfig.configure(args.network)
    sigma = args.sigma
    data_dir = args.data_dir
    result_dir = args.result_dir or "results_realtime_%dmin" % sigma

    if not os.path.isdir(data_dir):
        raise ScenarioError("data directory not found: %s" % data_dir)

    print("sigma = %d min; data dir: %s" % (sigma, data_dir))
    print("seed = %d (deterministic script; accepted for API parity)"
          % args.seed)

    if netconfig.is_baltimore():
        free_flow = read_free_flow_times_baltimore(
            os.path.join(data_dir, "input_link.csv"))
    else:
        free_flow = read_free_flow_times(os.path.join(data_dir, "SiouxFalls_net.xlsx"))
    print("network: %d directed links" % len(free_flow))

    template_path = os.path.join(data_dir, "input_agent_initial.csv")
    template_df, meta = read_demand(template_path)
    print("demand: %d agents" % len(template_df))

    if netconfig.is_baltimore():
        od_routes = read_choice_set_baltimore(data_dir)
    else:
        od_routes = read_choice_set(
            os.path.join(data_dir, "choice set no overlap try.mat")
        )
    print("choice set: %d ODs" % len(od_routes))

    agents = template_df.sort_values("_departure").reset_index(drop=True)
    # Left-closed/right-open binning; MATLAB phases are left-open/right-closed,
    # so t = 360.0 exactly falls in no MATLAB phase.  419/36,030 agents (1.16%)
    # depart at exact integer minutes -- accepted as negligible.
    agents["interval_m"] = np.floor(
        agents["_departure"].to_numpy() / sigma
    ).astype(int)

    input_agent_path = os.path.join(data_dir, "input_agent.csv")

    # Step 1: free-flow assignment -> DTALite -> observed link times.
    ff_paths, ff_stats = assign_paths(agents, od_routes, free_flow, {}, sigma)
    print(
        "step 1: free-flow assignment "
        "(%d agents without a valid OD)" % ff_stats["n_missing_od"]
    )
    write_input_agent(
        template_path, input_agent_path, meta["agent_col"], meta["path_col"],
        {a: p[0] for a, p in ff_paths.items()},
    )
    print("step 1: running DTALite ...")
    run_dtalite(data_dir, engine=args.engine)

    tdmoe_path = os.path.join(data_dir, "output_LinkTDMOE.csv")
    tdmoe = read_ragged_csv(
        tdmoe_path,
        ["from_node_id", "to_node_id", "timestamp_in_min", "travel_time_in_min"],
    )
    if len(tdmoe) == 0:
        raise ScenarioError(
            "no link-time rows in %s; DTALite produced no observations" % tdmoe_path
        )
    interval_table = build_interval_link_times(tdmoe, sigma)
    print(
        "step 1: LinkTDMOE parsed (%d rows, %d intervals with data)"
        % (len(tdmoe), len(interval_table))
    )

    # Test 6 (Future Traffic Conditions Estimation): save the Step 1 link
    # volumes as the "predicted" baseline before Step 2's run_dtalite()
    # deletes all output_* files.  Step 1's volumes are the historical
    # conditions the RT system uses to build X_{m-1} guidance.
    os.makedirs(result_dir, exist_ok=True)
    predicted_src = os.path.join(data_dir, "output_LinkMOE.csv")
    if os.path.isfile(predicted_src):
        shutil.copy(predicted_src,
                    os.path.join(result_dir, "predicted_LinkMOE.csv"))
        print("saved predicted link volumes to %s"
              % os.path.join(result_dir, "predicted_LinkMOE.csv"))
    else:
        print("warning: %s not found after Step 1; "
              "predicted link volumes not saved for Test 6" % predicted_src)

    # Step 2: reactive assignment under X_{m-1} -> DTALite once.
    rt_paths, rt_stats = assign_paths(
        agents, od_routes, free_flow, interval_table, sigma
    )
    print(
        "step 2: reactive assignment (m=0: %d, empty X bin: %d, no valid OD: %d)"
        % (
            rt_stats["n_interval_zero"],
            rt_stats["n_empty_bin"],
            rt_stats["n_missing_od"],
        )
    )
    write_input_agent(
        template_path, input_agent_path, meta["agent_col"], meta["path_col"],
        {a: p[0] for a, p in rt_paths.items()},
    )
    print("step 2: running DTALite ...")
    run_dtalite(data_dir, engine=args.engine)

    vehicle_hours, n_simulated = read_vehicle_hours(
        os.path.join(data_dir, "output_agent.csv")
    )
    routes_path, summary_path, per_agent_min, pct_diff = write_outputs(
        result_dir, sigma, agents, rt_paths, template_df, meta,
        vehicle_hours, n_simulated, rt_stats, args.seed,
    )
    print(
        "total travel time: %.2f vehicle-hours (%.2f min/agent)"
        % (vehicle_hours, per_agent_min)
    )
    print("vs Bingrong Table-4 actual-traffic target (rough; predicted traffic only): %+.1f%%" % pct_diff)
    print("wrote %s" % routes_path)
    print("wrote %s" % summary_path)


if __name__ == "__main__":
    try:
        main()
    except ScenarioError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        sys.exit(1)
