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
  * Route attributes are [length, tt, pltt, fuelcost, nc], evaluated at the
    agent's own departure timestamp ts = floor(departure) (MATLAB:
    po = floor(agent(i,2)) - 359, then tt{1,f}(po,1)). For a route/timestamp
    observed in the guidance run, tt = mean of the agents' experienced
    travel times at that timestamp and pltt = max -- v9's experienced
    mean/max construction (realtimeassignment.py lines ~180-215), itself a
    port of the UE reference comassignment.m lines ~108-160. Unobserved
    route/timestamps fall back to the TD-link table at that timestamp (the
    traveltimecal analog), with pltt = tt; links missing from the table
    fall back to free-flow, then to 1.0.
  * The tournament standardizes attribute diffs for dataset-28 agents
    (pref < 16650) with meanstd28 (SiouxFalls_net.xlsx 'weights' sheet,
    cols A:E, rows 2-3): bscale = (diff - means) / stds. This is the UE
    folder's convention (UE as information new equilibrium/
    comassignment.m:233-236), the code behind Test 4's 66.54% target.
    Pass --no-standardize to reproduce the Proposed folder's convention
    (raw diffs; matlab_fixed/comassignment.m:247-252, commented out there)
    for the direction-of-effect experiment. Dataset-2 agents (pref >=
    16650) always standardize with meanstd2 -- both MATLAB folders agree.
  * fuelcost mirrors v9's traveltimecal mpg bands on realized link speed
    (>40 mph -> 32, <25 mph -> 22, else 27 mpg; gas = $3.0/gal), evaluated
    at the agent's departure timestamp.
  * output_LinkTDMOE.csv is parsed with the csv module, NOT pandas:
    DTALite writes a trailing comma (25 data fields vs 24 header columns),
    which silently shifts every pandas column left by one. (The repo's
    scenarios_ue.py already carries a ragged-CSV reader for this reason.)

Pipeline:
  1. Load guidance CSV (ue_routes.csv or realtime_routes.csv).
  2. Load demand (input_agent_initial.csv), choice set (.mat), user
     preferences and true weights (SiouxFalls_net.xlsx).
  3. Load per-timestamp per-link travel times from the GUIDANCE run's
     output_LinkTDMOE.csv, plus the guidance run's agent experienced times
     (output_agent.csv) for the observed mean/max attributes. Both MUST be
     read before the reaction run overwrites the DTALite output files.
  4. Per agent: pairwise tournament over the choice-set routes for the
     agent's OD at the agent's departure timestamp -> actual choice. The
     standardization convention is selected by --standardize (default,
     UE folder) / --no-standardize (Proposed folder).
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
import netconfig

import engine as _engine_mod
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

def read_network_baltimore(csv_path):
    """Link table from Baltimore input_link.csv -> {(from, to): (length, freeflow)}.

    freeflow = length / free_speed * 60 minutes (or free_flow_time if present).
    """
    import csv as _csv
    if not os.path.isfile(csv_path):
        fail("network csv not found: %s" % csv_path)
    out = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in _csv.DictReader(fh):
            try:
                u = int(float(row["from_node_id"]))
                v = int(float(row["to_node_id"]))
                length = float(row["length"])
                fft_raw = row.get("free_flow_time")
                if fft_raw:
                    fft = float(fft_raw)
                else:
                    fft = length / float(row["free_speed"]) * 60.0
            except (ValueError, TypeError, KeyError):
                continue
            if (u, v) not in out:
                out[(u, v)] = (length, fft)
    return out


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


def read_users_csv(data_dir):
    """Baltimore: agent_weights.csv -> {agent_id: pref}.

    Same contract as read_users: 1-based preference id per agent.
    """
    import csv as _csv
    path = os.path.join(data_dir, "agent_weights.csv")
    if not os.path.isfile(path):
        fail("agent_weights.csv not found: %s" % path)
    prefs = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in _csv.DictReader(fh):
            try:
                prefs[int(float(row["agent_id"]))] = int(float(row["pref"]))
            except (TypeError, ValueError, KeyError):
                fail("bad row in agent_weights.csv: %r" % (row,))
    if not prefs:
        fail("no agents in agent_weights.csv")
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


def read_meanstd28(xlsx_path):
    """Dataset-28 standardization: sheet 'weights' cols A:E, rows 2-3.

    Returns (2, 5): row 0 = means, row 1 = stds. This is MATLAB's meanstd28
    (UE msa.m:37 reads sheet 6, A2:E3 -- 'weights' is the 6th sheet), used
    by the UE folder's reaction tournament (comassignment.m:233-236, the
    code behind Test 4's 66.54%): bscale = (b - meanstd28(1,:)) ./ stds.
    The Proposed folder's copy has this standardization commented out
    (matlab_fixed/comassignment.m:247-252); pass --no-standardize to
    reproduce that convention. Values in the current workbook: means
    [-2.73, -2.68, 2.05, -0.31, 6.0], stds [1.59, 3.72, 3.07, 0.37, 6.46].
    """
    try:
        raw = pd.read_excel(
            xlsx_path, sheet_name="weights", usecols="A:E",
            skiprows=0, nrows=3, header=None,
        ).values
    except Exception as exc:
        fail("cannot read meanstd28 (cols A:E) of %s: %s" % (xlsx_path, exc))
    ms = raw[1:3].astype(float)
    if ms.shape != (2, 5) or np.isnan(ms).any():
        fail("could not parse meanstd28 from sheet 'weights' cols A:E "
             "rows 2-3")
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


def read_choice_set_baltimore(data_dir):
    """Baltimore: choice_sets.pkl -> (routelocation, od_routes).

    Same contract as read_choice_set, but ODs are zones (mirrors the
    MATLAB od_path_map keyed by (origin_zone, dest_zone)).
    """
    import pickle as _pickle
    path = os.path.join(data_dir, "choice_sets.pkl")
    if not os.path.isfile(path):
        fail("choice-set file not found: %s" % path)
    with open(path, "rb") as fh:
        choice_dict = _pickle.load(fh)
    rl_rows = []
    od_routes = {}
    for od_idx, (oz, dz) in enumerate(sorted(choice_dict.keys())):
        routes = {}
        for r, nodes in enumerate(choice_dict[(oz, dz)]):
            nodes = [int(n) for n in nodes]
            if not nodes:
                continue
            routes[r] = nodes
            rl_rows.append([float(oz), float(dz), 1.0, float(r),
                            float(od_idx)])
        if routes:
            od_routes[od_idx] = routes
    if not rl_rows:
        fail("no routes in %s" % path)
    return np.asarray(rl_rows, dtype=float), od_routes


def read_choice_set(data_dir):
    """Choice set .mat -> (routelocation, od_routes).

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
    if not rl_rows:
        fail("no valid routes parsed from %s" % path)
    return np.array(rl_rows, dtype=float), od_routes


def read_link_times(data_dir):
    """Travel time per directed link per timestamp, from the GUIDANCE run.

    Returns {(from, to, ts): travel_time_in_min} from output_LinkTDMOE.csv,
    keyed by integer timestamp (MATLAB's TDlink(:,3)). Parsed with the csv
    module (not pandas): DTALite's LinkTDMOE rows can carry a trailing
    comma -- 25 fields against a 24-column header -- which silently shifts
    pandas' column parser by one. (The repo's scenarios_ue.py already
    carries a ragged-CSV reader for this reason.) Duplicate (link, ts)
    rows keep the last value.
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
    for col in ("from_node_id", "to_node_id", "timestamp_in_min",
                "travel_time_in_min"):
        if col not in names:
            fail("column '%s' missing from %s" % (col, path))
    i_from = names.index("from_node_id")
    i_to = names.index("to_node_id")
    i_ts = names.index("timestamp_in_min")
    i_tt = names.index("travel_time_in_min")
    tt = {}
    for row in rows[1:]:
        try:
            a = int(row[i_from].strip())
            b = int(row[i_to].strip())
            ts = int(float(row[i_ts].strip()))
            val = float(row[i_tt].strip())
        except (IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(val):
            continue
        tt[(a, b, ts)] = val
    if not tt:
        fail("no usable travel times in %s" % path)
    return tt


def read_agent_observations(data_dir, od_routes):
    """Experienced travel times from the GUIDANCE run's output_agent.csv.

    Returns (observed, n_rows, n_unmatched) where observed maps
    (od_idx, route_idx, ts) -> (tt_mean, pltt_max): tt is the mean of the
    agents' experienced travel times on that route at departure timestamp
    ts = floor(departure), pltt the max -- the ttloc branch of the UE
    reference comassignment.m (~108-160), and v9's realtimeassignment.py
    lines ~180-215. Routes are matched to the choice set by normalized
    path string (MATLAB: find(strcmp(path, agentpath))); rows whose path
    is not a choice-set route are skipped and counted. Must be read BEFORE
    the reaction DTALite run overwrites the guidance outputs (same
    stale-output discipline as read_link_times).
    """
    path = os.path.join(data_dir, AGENT_OUTPUT)
    if not os.path.isfile(path):
        fail("%s not found in %s -- run the guidance phase (DTALite) first"
             % (AGENT_OUTPUT, data_dir))
    path_lookup = {}
    for od_idx, routes in od_routes.items():
        for r, nodes in routes.items():
            path_lookup[normalize_path(nodes)] = (od_idx, r)
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        fail("no data rows in %s" % path)
    names = [h.strip() for h in rows[0]]
    for col in ("departure_time_in_min", "travel_time_in_min",
                "path_node_sequence"):
        if col not in names:
            fail("column '%s' missing from %s" % (col, path))
    i_dep = names.index("departure_time_in_min")
    i_tt = names.index("travel_time_in_min")
    i_path = names.index("path_node_sequence")
    buckets = {}
    n_rows = 0
    n_unmatched = 0
    for row in rows[1:]:
        if max(i_dep, i_tt, i_path) >= len(row):
            fail("short row in %s" % path)
        try:
            dep = float(row[i_dep].strip())
            tt = float(row[i_tt].strip())
        except (TypeError, ValueError):
            fail("non-numeric departure/travel time in %s" % path)
        if not (np.isfinite(dep) and np.isfinite(tt)):
            fail("non-finite departure/travel time in %s" % path)
        ts = int(np.floor(dep))
        win_lo = netconfig.TIME_START
        win_hi = win_lo + netconfig.N_PERIODS - 1
        if ts < win_lo or ts > win_hi:
            fail("departure %.2f min in %s outside the %d-%d window"
                 % (dep, path, win_lo, win_hi))
        nodes = parse_nodes(row[i_path])
        key = path_lookup.get(normalize_path(nodes)) if nodes else None
        if key is None:
            n_unmatched += 1
            continue
        buckets.setdefault((key[0], key[1], ts), []).append(tt)
        n_rows += 1
    if not buckets:
        fail("no usable agent observations in %s" % path)
    observed = {k: (float(np.mean(v)), float(np.max(v)))
                for k, v in buckets.items()}
    return observed, n_rows, n_unmatched


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
    """Demand template -> ({agent_id: (o, d)}, {agent_id: raw path},
    {agent_id: departure_min}, template path).

    OD columns are looked up by name but asserted to sit at positions 4
    and 5, v9's positional convention (comrealtimeassignment reads
    S.iloc[:, [4, 5]]). Departures feed the per-timestamp attribute lookup
    (ts = floor(departure), MATLAB's po = floor(agent(i,2)) - 359).
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
                "departure_time_in_min", "path_node_sequence"):
        if col not in names:
            fail("column '%s' missing from %s" % (col, path))
    if netconfig.is_baltimore():
        i_from = names.index("from_zone_id")
        i_to = names.index("to_zone_id")
    else:
        i_from = names.index("from_origin_node_id")
        i_to = names.index("to_destination_node_id")
        if (i_from, i_to) != (4, 5):
            fail("OD columns of %s are at positions %d,%d, not v9's 4,5"
                 % (path, i_from, i_to))
    id_col = names.index("agent_id")
    dep_col = names.index("departure_time_in_min")
    path_col = names.index("path_node_sequence")
    ods, template_paths, departures = {}, {}, {}
    for row in rows[1:]:
        try:
            aid = int(float(row[id_col].strip()))
            o = int(float(row[i_from].strip()))
            d = int(float(row[i_to].strip()))
            dep = float(row[dep_col].strip())
        except (IndexError, TypeError, ValueError):
            fail("bad agent/OD/departure row in %s" % path)
        if not np.isfinite(dep):
            fail("non-finite departure time for agent %d in %s" % (aid, path))
        if aid in ods:
            fail("duplicate agent_id %d in %s" % (aid, path))
        ods[aid] = (o, d)
        departures[aid] = dep
        template_paths[aid] = row[path_col] if path_col < len(row) else ""
    if not ods:
        fail("no agents in %s" % path)
    return ods, template_paths, departures, path


# ---------------------------------------------------------------------------
# Route attributes and the pairwise tournament
# ---------------------------------------------------------------------------

class AttributeCache:
    """Per-(od_idx, route, timestamp) attribute vectors
    [length, tt, pltt, fuelcost, nc].

    tt/pltt follow the UE reference (comassignment.m ~108-160): for a
    route/timestamp observed in the guidance run, tt = mean of the agents'
    experienced travel times at that departure timestamp and pltt = max;
    unobserved route/timestamps fall back to the sum of the TD-link travel
    times at that timestamp (pltt = tt). length and nc are
    time-independent. fuelcost mirrors traveltimecal's mpg bands on the
    realized per-link speed at that timestamp (like v9, which always calls
    traveltimecal for fuel even on observed routes).
    """

    def __init__(self, link_info, link_tt_ts, observed):
        self.link_info = link_info      # (a, b) -> (length, freeflow_tt)
        self.link_tt_ts = link_tt_ts    # (a, b, ts) -> observed tt
        self.observed = observed        # (od, r, ts) -> (tt_mean, pltt_max)
        self.cache = {}

    def link_time(self, a, b, ts):
        tt = self.link_tt_ts.get((a, b, ts))
        if tt is not None:
            return tt
        info = self.link_info.get((a, b))
        if info is not None and np.isfinite(info[1]):
            return info[1]
        return 1.0

    def get(self, od_idx, r, nodes, ts):
        key = (od_idx, r, ts)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        length = 0.0
        tt_sum = 0.0
        fuel = 0.0
        links = list(zip(nodes[:-1], nodes[1:]))
        for a, b in links:
            info = self.link_info.get((a, b))
            seg_len = info[0] if info is not None else 0.0
            seg_tt = self.link_time(a, b, ts)
            length += seg_len
            tt_sum += seg_tt
            if seg_len > 0 and seg_tt > 0:
                speed = seg_len / seg_tt * 60.0
                if speed > 40:
                    mpg = MPG_FAST
                elif speed < 25:
                    mpg = MPG_SLOW
                else:
                    mpg = MPG_MID
                fuel += seg_len / mpg * GAS_PRICE
        obs = self.observed.get(key)
        if obs is not None:
            tt_val, pltt_val = obs
        else:
            tt_val, pltt_val = tt_sum, tt_sum
        vec = np.array([length, tt_val, pltt_val, fuel, float(len(links))])
        self.cache[key] = vec
        return vec


def tournament(od_idx, route_ids, wrow, dataset2, attr_cache, od_routes,
               meanstd2, ts, standardize=True, meanstd28=None):
    """Pairwise utility tournament; returns the winning route index.

    Exact port of the tournament loop in v9's comrealtimeassignment
    (realtimeassignment.py lines 290-315): the champion starts as the
    first (lowest-index) route; each challenger replaces it when
    choice == 0, i.e. abs(ojvalue-1) > abs(ojvalue+1). diff = b1 - b0
    (champion minus challenger); ojvalue = sum(bscale*wrow) with no
    intercept ("real weights don't have intercept"). The onlylike0/
    onlylike1 debug lists in v9 are empty and are dropped here, as is the
    parallel minimum-travel-time (pre_choice) tournament.

    Route attributes are evaluated at the agent's departure timestamp ts
    (floor of departure minutes; MATLAB's po = floor(agent(i,2)) - 359).

    standardize selects the dataset-28 (pref < 16650) convention:
      True  (default) -- the UE folder's convention (UE as information new
              equilibrium/comassignment.m:233-236), behind Test 4's 66.54%
              target: bscale = (diff - meanstd28_means) / meanstd28_stds.
      False -- the Proposed folder's convention
              (matlab_fixed/comassignment.m:247-252, commented out there):
              raw diffs. Use for the direction-of-effect experiment.
    Dataset-2 agents (pref >= 16650) always standardize with meanstd2 --
    both MATLAB folders agree there, so the flag does not touch them.
    """
    champion = route_ids[0]
    b1 = attr_cache.get(od_idx, champion, od_routes[champion], ts)
    for r in route_ids[1:]:
        b0 = attr_cache.get(od_idx, r, od_routes[r], ts)
        diff = b1 - b0
        if dataset2:
            # Matches v9/MATLAB: the zero std at index 3 divides to inf
            # (0/0 to nan); the warning is suppressed exactly as v9 does.
            with np.errstate(divide="ignore", invalid="ignore"):
                bscale = (diff - meanstd2[0, :]) / meanstd2[1, :]
        elif standardize:
            if meanstd28 is None:
                fail("standardize=True requires meanstd28 "
                     "(read from the xlsx 'weights' sheet)")
            # UE folder convention: MATLAB's silent per-element division.
            with np.errstate(divide="ignore", invalid="ignore"):
                bscale = (diff - meanstd28[0, :]) / meanstd28[1, :]
        else:
            # Proposed folder convention: raw diffs (the standardization
            # lines are commented out in matlab_fixed/comassignment.m).
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


def run_dtalite(data_dir, exe_override, timeout, engine=None):
    """Run DTALite once; fail fast with a clear error on any failure."""
    if _engine_mod.resolve(engine) == "pysim":
        try:
            _engine_mod.pysim_command(data_dir)
        except RuntimeError as exc:
            fail(str(exc))
        if exe_override:
            fail("--exe is a DTALite executable override and cannot be "
                 "combined with --engine pysim")
    for name in STALE_OUTPUTS:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            os.remove(p)
            print("removed stale %s" % name)
    _engine = _engine_mod.resolve(engine)
    if _engine == "pysim":
        print("running PySim (%s) ..." % " ".join(
            _engine_mod.pysim_command(data_dir)))
        t0 = time.time()
        try:
            _engine_mod.run_pysim(data_dir, timeout, "preference_reaction")
        except RuntimeError as exc:
            fail(str(exc))
        print("PySim finished in %.0f s" % (time.time() - t0))
        return
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
    p.add_argument("--network", default="siouxfalls",
                   choices=["siouxfalls", "baltimore"],
                   help="network: siouxfalls (default) or baltimore")
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
    p.add_argument("--engine", default="pysim",
                   choices=["pysim", "dtalite"],
                   help="simulation engine: pysim (default) or dtalite")
    p.add_argument("--timeout", type=int, default=3600,
                   help="seconds to wait for DTALite (default 3600)")
    p.add_argument("--standardize", dest="standardize", action="store_true",
                   default=True,
                   help="standardize dataset-28 tournament diffs with "
                        "meanstd28 (UE-folder convention; default -- the "
                        "convention behind Test 4's 66.54%% target)")
    p.add_argument("--no-standardize", dest="standardize",
                   action="store_false",
                   help="use raw diffs for dataset-28 agents (Proposed-folder "
                        "convention; for the direction-of-effect experiment)")
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

    import netconfig as _nc
    _nc.configure(args.network)
    xlsx = os.path.join(data_dir, NETWORK_XLSX)
    print("loading network, users, weights ...")
    if _nc.is_baltimore():
        link_info = read_network_baltimore(
            os.path.join(data_dir, "input_link.csv"))
    else:
        link_info = read_network(xlsx)
    prefs = (read_users_csv(data_dir) if _nc.is_baltimore()
             else read_users(xlsx))
    realweights = read_realweights(xlsx)
    meanstd2 = read_meanstd2(xlsx)
    meanstd28 = read_meanstd28(xlsx)
    print("loading choice set ...")
    routelocation, od_routes = (read_choice_set_baltimore(data_dir)
                                 if _nc.is_baltimore()
                                 else read_choice_set(data_dir))
    print("loading link travel times ...")
    # From the GUIDANCE run's outputs: read before the reaction run
    # overwrites the DTALite output files.
    link_tt = read_link_times(data_dir)
    print("loading guidance-run agent observations ...")
    observed, n_obs_rows, n_unmatched = read_agent_observations(
        data_dir, od_routes)
    print("loading guidance and agent ODs ...")
    guidance = read_guidance(args.guidance_csv)
    agent_ods, template_paths, departures, template_path = \
        read_agent_template(data_dir)

    n_agents = len(agent_ods)
    if len(prefs) < n_agents:
        fail("sheet 'users' has %d rows but demand has %d agents"
             % (len(prefs), n_agents))

    log_lines.append("agents in demand file : %d" % n_agents)
    log_lines.append("agents with guidance   : %d" % len(guidance))
    log_lines.append("choice set             : %d OD cells, %d routelocation "
                     "rows" % (len(od_routes), routelocation.shape[0]))
    log_lines.append("link times             : %d (link, timestamp) cells "
                     "from guidance output_LinkTDMOE.csv" % len(link_tt))
    log_lines.append("observed tt/pltt       : %d (route, timestamp) cells "
                     "from %d guidance agents (%d unmatched paths)"
                     % (len(observed), n_obs_rows, n_unmatched))
    log_lines.append("standardization        : %s"
                     % ("meanstd28 (UE-folder convention)"
                        if args.standardize
                        else "raw diffs (Proposed-folder convention)"))
    log_lines.append("weights                : realweights (TRUE preferences, "
                     "cols J:O); predict phase used estimated cols A:F")

    cache = AttributeCache(link_info, link_tt, observed)

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
            # path, else the template's own path as a last resort. The kept
            # guidance path is auto-satisfied by construction, so this branch
            # is structural (n_no_choice_set = 0 in the actual run), not a
            # result contributor.
            n_no_choice_set += 1
            g = guidance.get(aid)
            nodes = (parse_nodes(g) if g is not None
                     else parse_nodes(template_paths[aid]))
            if nodes is None:
                fail("agent %d has no choice set and no fallback path" % aid)
            actual[aid] = nodes
            continue
        dep = departures[aid]
        ts = int(np.floor(dep))
        win_lo = netconfig.TIME_START
        win_hi = win_lo + netconfig.N_PERIODS - 1
        if ts < win_lo or ts > win_hi:
            fail("agent %d departs at %.2f min, outside the %d-%d "
                 "information window" % (aid, dep, win_lo, win_hi))
        route_ids = sorted(routes.keys())
        winner = tournament(f, route_ids, wrow, dataset2, cache, routes,
                            meanstd2, ts,
                            standardize=args.standardize,
                            meanstd28=meanstd28)
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
    run_dtalite(data_dir, args.exe, args.timeout, args.engine)
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
