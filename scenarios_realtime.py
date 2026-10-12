#!/usr/bin/env python3
"""Real-time route-guidance baselines (Tests 2 & 3 of Bingrong Sun, 2018, Ch. 4).

Reactive guidance in the style of Google Maps: every agent departs with the
shortest-travel-time route under the real-time conditions known at departure.
Time is discretized into intervals of length sigma (1 min = Test 2,
10 min = Test 3); an agent departing in interval m sees conditions X_{m-1}.

X_{m-1} APPROXIMATION (agreed framework): this is a single pass over
free-flow link times. X_0 is the free-flow network and X_{m-1} is taken as
free-flow for every interval, so each agent is assigned the free-flow
shortest route from its choice-set row. Building true per-interval X_{m-1}
from repeated DTALite runs is a documented future step. The system assumes
every user minimizes travel time only; true (preference-based) choices are
handled elsewhere. sigma_min is recorded per agent in realtime_routes.csv so
the analysis layer can distinguish the sigma=1 and sigma=10 runs.

Outputs (all under --result-dir unless noted):
  input_agent.csv      - demand template with assigned paths (written into the
                         DTALite run folder = data dir; DTALite.exe is run
                         with cwd=data dir, so it reads it there)
  realtime_routes.csv  - agent_id, path_node_sequence, sigma_min
  realtime_summary.txt - vehicle-hours from the DTALite run vs the reported
                         target (26983.1 for sigma=1, 28039.6 for sigma=10)

MERGE NOTE: this file is the validated winner of two independent
implementations (build C and build D). Both produced byte-identical route
assignments on all 36,030 demand agents; they differed only in plumbing.
Kept from D: grid OD lookup with endpoint validation, direct "Free flow
time" column read, --timeout, travel-time column heuristic, manual
result-dir default. Kept from C: strict directed link lookup (no undirected
tolerance), strip-based path-column matching, demand-column validation,
data-dir validation, --exe override, fail-fast DTALite error handling,
pandas output writing. Dropped from D: the Dijkstra fallback for
choice-set-less ODs (no demand agent hits it; keep-template is the
framework-faithful behavior) and the undirected-link tolerance. Fixed from
C: DTALite is now run with cwd=data_dir (C ran it from the parent dir while
writing input_agent.csv into data_dir).

Final review (2026-10-11): fail fast on missing agent_id / departure times
(used to crash deep in the assignment loop); agents with a missing origin
or destination now keep the template path instead of crashing.

Requires: Python 3.9+, numpy, pandas, scipy, openpyxl.
"""

import argparse
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy.io

N_ZONES = 24
TARGET_VH = {1: 26983.1, 10: 28039.6}


def log(msg):
    print("[realtime] %s" % msg, flush=True)


def fail(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def find_data_dir(given):
    """Locate the folder holding DTALite_Files contents (network, demand, choice set)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if given:
        candidates.append(given)
    candidates += [
        os.path.join(here, "repo", "DTALite_Files"),
        os.path.join(here, "DTALite_Files"),
        os.path.join(os.getcwd(), "DTALite_Files"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "input_agent_initial.csv")):
            return os.path.abspath(c)
    fail(
        "could not find input_agent_initial.csv; tried:\n  "
        + "\n  ".join(candidates)
        + "\npass --data-dir explicitly"
    )


def load_freeflow_times(xlsx_path):
    """Read (from_node, to_node) -> free-flow travel time in minutes.

    SiouxFalls_net.xlsx, sheet 'SiouxFalls_net', is the classic
    transportation-network text format: a header row starting with '~'
    followed by rows (init node, term node, capacity, length,
    free flow time, ...). Reads the 'Free flow time' column directly
    (column F); verified identical to length/speed*60 on this network.
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    if "SiouxFalls_net" not in wb.sheetnames:
        fail("sheet 'SiouxFalls_net' not in %s" % xlsx_path)
    rows = list(wb["SiouxFalls_net"].iter_rows(values_only=True))
    wb.close()
    data = False
    times = {}
    for r in rows:
        if not data:
            if r and str(r[0]).strip() == "~":
                data = True
            continue
        if len(r) < 6 or r[1] is None or r[2] is None or r[5] is None:
            continue
        times[(int(r[1]), int(r[2]))] = float(r[5])
    if not times:
        fail("no link rows parsed from %s" % xlsx_path)
    log("network: %d directed links with free-flow times" % len(times))
    return times


def parse_path_string(s):
    """'16;10;' -> [16, 10]; None for empty/invalid."""
    if s is None:
        return None
    try:
        if pd.isna(s):
            return None
    except (TypeError, ValueError):
        pass
    if not isinstance(s, str):
        return None
    try:
        nodes = [int(float(x)) for x in s.split(";") if x.strip()]
    except (TypeError, ValueError):
        return None
    return nodes if len(nodes) >= 2 else None


def cell_to_string(cell):
    """Extract the path string from a choice-set .mat cell."""
    if isinstance(cell, np.ndarray):
        if cell.size == 0:
            return None
        if cell.size == 1:
            cell = cell.item()
        else:
            return None
    if isinstance(cell, str):
        return cell
    if isinstance(cell, np.ndarray) and cell.size > 0:
        v = cell.flat[0]
        if isinstance(v, str):
            return v
    return None


def load_choice_set(mat_path):
    """finallist (10, 576) -> routes[od_idx] = list of node lists (<=10 routes)."""
    raw = scipy.io.loadmat(mat_path)["finallist"].T  # (576, 10) = (n_ods, n_routes)
    n_ods, n_routes = raw.shape
    log("choiceset: %d OD rows x %d routes" % (n_ods, n_routes))
    routes = {}
    for od in range(n_ods):
        od_routes = []
        for c in range(n_routes):
            nodes = parse_path_string(cell_to_string(raw[od, c]))
            if nodes:
                od_routes.append(nodes)
        routes[od] = od_routes
    usable = sum(1 for r in routes.values() if r)
    log("usable OD rows: %d (empty: %d)" % (usable, n_ods - usable))
    return routes


def od_index(origin, dest):
    """Choice-set row for an (origin, dest) pair: 24x24 zone grid, row-major."""
    return (origin - 1) * N_ZONES + (dest - 1)


def validate_od_grid(routes):
    """Check the grid assumption against every row's first valid route."""
    bad = 0
    for od, od_routes in routes.items():
        if not od_routes:
            continue
        o, d = od // N_ZONES + 1, od % N_ZONES + 1
        nodes = od_routes[0]
        if nodes[0] != o or nodes[-1] != d:
            bad += 1
    return bad


def route_cost(nodes, times):
    """Sum of free-flow link minutes; None if any link is missing."""
    total = 0.0
    for a, b in zip(nodes[:-1], nodes[1:]):
        t = times.get((a, b))
        if t is None:
            return None
        total += t
    return total


def format_path(nodes):
    return ";".join(str(n) for n in nodes) + ";"


def assign_routes(demand, routes, times, sigma, path_col):
    """Assign one route per agent, in departure order.

    Returns dict agent_id -> (path string, method) where method is
    'choiceset' or 'template' (template path kept when the OD is missing, has
    no choice-set row, or has no valid route).
    """
    order = demand.sort_values("departure_time_in_min", kind="mergesort")
    assigned = {}
    counts = {"choiceset": 0, "template": 0}
    skipped = 0
    for i in order.index:
        dep = float(demand.at[i, "departure_time_in_min"])
        m = int(math.floor(dep / sigma))  # departure interval (X_{m-1} ~= X_0)
        _ = m
        o_raw = demand.at[i, "from_origin_node_id"]
        d_raw = demand.at[i, "to_destination_node_id"]
        best, method = None, "template"
        if not pd.isna(o_raw) and not pd.isna(d_raw):
            origin, dest = int(o_raw), int(d_raw)
            if 1 <= origin <= N_ZONES and 1 <= dest <= N_ZONES:
                best_cost = None
                for nodes in routes.get(od_index(origin, dest), []):
                    c = route_cost(nodes, times)
                    if c is not None and (best_cost is None or c < best_cost):
                        best_cost, best = c, nodes
                if best is not None:
                    method = "choiceset"
        if best is None:  # no choice-set route: keep the template's path
            best = parse_path_string(demand.at[i, path_col])
            if best is None:
                skipped += 1
                continue
        assigned[int(demand.at[i, "agent_id"])] = (format_path(best), method)
        counts[method] += 1
    return assigned, counts, skipped


def run_dtalite(run_dir, exe, timeout_s):
    log("running DTALite: %s (timeout %ds)" % (exe, timeout_s))
    try:
        proc = subprocess.run(
            [exe], cwd=run_dir, timeout=timeout_s,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except subprocess.TimeoutExpired:
        fail("DTALite.exe timed out after %d s" % timeout_s)
    except OSError as e:
        fail("could not launch DTALite.exe (%s): %s" % (exe, e))
    if proc.returncode != 0:
        tail = (proc.stdout or "").strip().splitlines()[-5:]
        fail("DTALite.exe exited with code %d\n%s" % (
            proc.returncode, "\n".join(tail)))
    log("DTALite finished (exit 0)")
    out_path = os.path.join(run_dir, "output_agent.csv")
    if not os.path.isfile(out_path):
        fail("DTALite did not produce output_agent.csv")
    return out_path


def vehicle_hours(output_agent_path):
    """Sum of experienced travel time / 60 from DTALite's output_agent.csv."""
    out = pd.read_csv(output_agent_path)
    tt_col = next(
        (c for c in out.columns
         if "travel_time" in str(c).strip().lower() and "min" in str(c).lower()),
        None)
    if tt_col is None:
        if len(out.columns) > 12:
            tt_col = out.columns[12]
        else:
            fail("could not identify the travel-time column in output_agent.csv")
    tt = pd.to_numeric(out[tt_col], errors="coerce").dropna()
    return float(tt.sum()) / 60.0, len(tt)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Real-time guidance baselines (Tests 2/3): shortest-travel-time "
                    "routing on free-flow conditions, single pass, one DTALite run.")
    ap.add_argument("--sigma", type=int, choices=[1, 10], required=True,
                    help="update interval in minutes: 1 (Test 2) or 10 (Test 3)")
    ap.add_argument("--result-dir", default=None,
                    help="output folder (default results/realtime_<sigma>)")
    ap.add_argument("--data-dir", default=None,
                    help="folder holding DTALite_Files contents (default: auto-detect)")
    ap.add_argument("--exe", default=None,
                    help="DTALite executable (default: <data-dir>/DTALite.exe)")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="DTALite.exe timeout in seconds (default 3600)")
    args = ap.parse_args(argv)

    sigma = args.sigma
    data_dir = find_data_dir(args.data_dir)
    result_dir = (os.path.abspath(args.result_dir) if args.result_dir
                  else os.path.abspath(
                      os.path.join("results", "realtime_%d" % sigma)))
    os.makedirs(result_dir, exist_ok=True)
    log("sigma=%d min, data dir=%s, result dir=%s" % (sigma, data_dir, result_dir))

    mat_path = os.path.join(data_dir, "choice set no overlap try.mat")
    if not os.path.isfile(mat_path):
        fail("choice set not found: %s" % mat_path)
    routes = load_choice_set(mat_path)
    grid_bad = validate_od_grid(routes)
    if grid_bad:
        fail("%d choice-set rows disagree with the OD grid" % grid_bad)

    xlsx_path = os.path.join(data_dir, "SiouxFalls_net.xlsx")
    if not os.path.isfile(xlsx_path):
        fail("network file not found: %s" % xlsx_path)
    times = load_freeflow_times(xlsx_path)

    template = os.path.join(data_dir, "input_agent_initial.csv")
    if not os.path.isfile(template):
        fail("demand template not found: %s" % template)
    demand = pd.read_csv(template)
    required = {"agent_id", "from_origin_node_id",
                "to_destination_node_id", "departure_time_in_min"}
    missing = required - set(demand.columns)
    if missing:
        fail("demand file missing columns: %s" % sorted(missing))
    for col in ("agent_id", "departure_time_in_min"):
        if demand[col].isna().any():
            fail("demand file has missing values in column '%s'" % col)
    path_col = next((c for c in demand.columns
                     if str(c).strip() == "path_node_sequence"), None)
    if path_col is None:
        fail("template has no path_node_sequence column")
    log("demand: %d agents, departures %.1f-%.1f min" % (
        len(demand), demand["departure_time_in_min"].min(),
        demand["departure_time_in_min"].max()))

    log("assigning routes (free-flow X_0 for every interval) ...")
    assigned, counts, skipped = assign_routes(demand, routes, times, sigma, path_col)
    log("assigned: %d via choice set, %d kept template path (skipped, no valid path: %d)"
        % (counts["choiceset"], counts["template"], skipped))

    # --- input_agent.csv: template columns preserved, row order kept ---
    paths = demand["agent_id"].map(
        lambda a: assigned.get(int(a), (None, None))[0])
    demand[path_col] = paths.fillna(demand[path_col])
    agent_csv = os.path.join(data_dir, "input_agent.csv")
    demand.to_csv(agent_csv, index=False, lineterminator="\r\n")
    log("wrote %s (%d rows)" % (agent_csv, len(demand)))

    # --- run DTALite once (cwd = data dir: DTALite reads input_agent.csv there) ---
    exe = args.exe if args.exe else os.path.join(data_dir, "DTALite.exe")
    output_agent_path = run_dtalite(data_dir, exe, args.timeout)

    # --- realtime_routes.csv: agent_id, path_node_sequence, sigma_min ---
    routes_rows = [
        {"agent_id": agent_id,
         "path_node_sequence": path,
         "sigma_min": sigma}
        for agent_id, (path, _) in sorted(assigned.items())
    ]
    routes_path = os.path.join(result_dir, "realtime_routes.csv")
    pd.DataFrame(routes_rows,
                 columns=["agent_id", "path_node_sequence", "sigma_min"]
                 ).to_csv(routes_path, index=False, lineterminator="\r\n")
    log("wrote %s (%d rows)" % (routes_path, len(routes_rows)))

    # --- vehicle-hours vs Bingrong's target ---
    vh, n_tt = vehicle_hours(output_agent_path)
    target = TARGET_VH[sigma]
    diff = 100.0 * (vh - target) / target
    summary_lines = [
        "Real-time guidance baseline (Test %s), sigma = %d min"
        % ("2" if sigma == 1 else "3", sigma),
        "agents assigned: %d (choice set %d, template-path fallback %d; skipped %d)"
        % (len(assigned), counts["choiceset"], counts["template"], skipped),
        "DTALite run: single pass, input_agent.csv (%d rows)" % len(demand),
        "",
        "vehicle-hours (sum of agent travel times / 60): %.1f (%d agents)"
        % (vh, n_tt),
        "target (Sun 2018, Ch.4): %.1f" % target,
        "difference: %+.1f%%" % diff,
        "",
        "single-pass reactive assignment on free-flow X_0 for all intervals;",
        "shortest travel time route per agent; sigma_min recorded per agent in",
        "realtime_routes.csv (agent_id, path_node_sequence, sigma_min).",
    ]
    summary_path = os.path.join(result_dir, "realtime_summary.txt")
    with open(summary_path, "w", newline="") as f:
        f.write("\r\n".join(summary_lines) + "\r\n")
    log("wrote %s" % summary_path)
    log("vehicle-hours: %.1f vs target %.1f (%+.1f%%)" % (vh, target, diff))
    log("done")


if __name__ == "__main__":
    main()
