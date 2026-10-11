"""
runner_bingrong.py (v2)
=======================
Analyze a completed MSA run against Bingrong Sun's 2018 dissertation
Chapter 4 results (Tables 8 & 9) -- the "Proposed" scenario
(proactive user-optimum-oriented guidance at 100% MPR).

What it computes, and the exact definitions behind each number:

1. Satisfaction rate (primary, per dissertation text):
   Sun (2018), Ch.4: "the number of agents whose truly preferred route
   match the recommended routes is used as measurement of users'
   satisfaction" and "users' actual route choices are compared to the
   recommended routes in the generated guidance information."
   - Recommended routes = UO guidance = {N}UOinfor_agent.csv
     (UO phase runs the pairwise tournament with ESTIMATED weights;
     para 596: "routes are recommended based on estimated route
     choice preferences").
   - Actual choices = {N}actual_agent.csv (comassignment runs the
     tournament with TRUE weights -- realweights from SiouxFalls_net
     -- fed the guidance info, per the evaluation flowchart).
   - satisfied(a) = norm(path_actual[a]) == norm(path_uo[a]),
     indexed by agent_id (never row order), path_node_sequence
     normalized identically on both sides (strip, split ';',
     drop empties, rejoin). Uses the LAST bigloop's files.
   - Cross-check: the same comparison against {N}predict_agent.csv.
     At 100% MPR every agent is a member (no realtime users), so
     predict == UO for all agents and the two numbers must agree;
     any disagreement is flagged.

2. Total travel time (vehicle-hours):
   "system total travel time is defined as the summation of all
   agents' experienced travel time" -- sum of travel_time_in_min
   over {N}actual_agent.csv, /60. Bingrong's demand was 54,045
   agents; this run is ~36k, so the per-agent figure
   (min/agent, target 27.44) is the fair comparison.

3. Total delay (vehicle-hours):
   "calculated by summing the multiplication of link delay and the
   link volume on all links" -- from {N}actual_LinkMOE.csv:
   sum((travel_time - free-flow_travel_time) * total_link_volume)/60.
   Assumes LinkMOE travel times are in minutes.

4. Energy / emissions: column sums from {N}actual_agent.csv --
   TotalEnergy_(KJ)/1e9 (target 2.65), CO2_(g)/1e6 tons (190.8),
   NOX_(g)/1e3 kg (141.3), CO_(g)/1e3 kg (1525.3), HC_(g)/1e3 kg (47.8).

Bingrong's reported "Proposed" results @ 100% MPR:
    Satisfaction: 88.64% (47,908 / 54,045)
    Vehicle-hours: 24,719.3   (0.45738 h/agent = 27.44 min/agent)
    Delay:         7,860.4    (0.14544 h/agent)
    Energy: 2.65e9 kJ | CO2 190.8 t | NOX 141.3 kg | CO 1525.3 kg | HC 47.8 kg

Usage (run from the project root or point at DTALite_Files):
    python runner_bingrong.py --result-dir C:\\path\\to\\DTALite_Files
    python runner_bingrong.py --result-dir . --bigloop 10

Stdlib only -- runs on the same Python 3.9 stack as the pipeline.
"""

import argparse
import csv
import os
import re
import sys

VERSION = "v2"

# ----------------------------------------------------------------------
# Bingrong's reported results @ 100% MPR (Sun 2018, Ch.4 Tables 8 & 9)
# ----------------------------------------------------------------------
BINGRONG_100MPR = {
    "proposed": {
        "description": "Proactive user-optimum-oriented guidance",
        "satisfaction_rate": 0.8864,
        "n_satisfied": 47908,
        "vehicle_hours": 24719.3,
        "vh_per_agent_min": 27.44,
        "delay_hours": 7860.4,
        "delay_per_agent_h": 0.14544,
        "energy_1e9kj": 2.65,
        "co2_ton": 190.8,
        "nox_kg": 141.3,
        "co_kg": 1525.3,
        "hc_kg": 47.8,
    },
    "realtime_1min": {
        "description": "Real-time guidance, 1-min updates",
        "satisfaction_rate": 0.7049,
        "n_satisfied": 38096,
        "vehicle_hours": 26983.1,
        "vh_per_agent_min": 29.96,
        "delay_hours": 9831.9,
    },
    "realtime_10min": {
        "description": "Real-time guidance, 10-min updates",
        "satisfaction_rate": 0.7019,
        "n_satisfied": 37935,
        "vehicle_hours": 28039.6,
        "vh_per_agent_min": 31.13,
        "delay_hours": 10809.5,
    },
    "ue": {
        "description": "UE as information (DUE via MSA)",
        "satisfaction_rate": 0.6654,
        "n_satisfied": 35963,
        "vehicle_hours": 30760.5,
        "vh_per_agent_min": 34.15,
        "delay_hours": 13857.4,
    },
}

TOTAL_DEMAND = 54045
TOLERANCE = 0.05  # 5% relative tolerance for MATCH verdicts

TAG_RE = re.compile(r"^(\d+)(UOinfor|predict|actual)_agent\.csv$")


# ----------------------------------------------------------------------
# File discovery
# ----------------------------------------------------------------------
def find_result_dir(given):
    """Accept the project root or DTALite_Files itself."""
    cands = [given, os.path.join(given, "DTALite_Files")]
    for c in cands:
        if os.path.isdir(c):
            return c
    return given


def find_bigloops(d):
    """Map bigloop number -> set of tagged agent files present."""
    found = {}
    try:
        names = os.listdir(d)
    except OSError:
        return found
    for name in names:
        m = TAG_RE.match(name)
        if m:
            found.setdefault(int(m.group(1)), set()).add(m.group(2))
    return found


def pick_bigloop(d, requested):
    found = find_bigloops(d)
    if not found:
        return None, {}
    if requested is not None:
        if requested not in found:
            return None, found
        return requested, found
    # last bigloop that has both actual and UOinfor
    cands = [n for n, tags in found.items()
             if "actual" in tags and "UOinfor" in tags]
    if not cands:
        return None, found
    return max(cands), found


# ----------------------------------------------------------------------
# CSV helpers (DTALite headers carry stray leading spaces)
# ----------------------------------------------------------------------
def norm_headers(fieldnames):
    return {c.strip().lower(): c for c in fieldnames if c and c.strip()}


def read_agent_paths(path):
    """agent_id -> normalized path_node_sequence."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = norm_headers(reader.fieldnames or [])
        if "agent_id" not in cols or "path_node_sequence" not in cols:
            raise ValueError(
                "missing agent_id/path_node_sequence in %s" % path)
        aid_c = cols["agent_id"]
        path_c = cols["path_node_sequence"]
        for row in reader:
            aid = (row.get(aid_c) or "").strip()
            if aid:
                out[aid] = norm_path(row.get(path_c) or "")
    return out


def norm_path(s):
    """Identical normalization on both sides: strip, split ';',
    drop empty segments, rejoin."""
    return ";".join(seg.strip() for seg in s.strip().split(";") if seg.strip())


def find_tt_col(cols):
    for key, orig in cols.items():
        if "travel_time" in key and "min" in key:
            return orig
    return None


def sum_travel_minutes(path):
    total = 0.0
    n = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = norm_headers(reader.fieldnames or [])
        tt_c = find_tt_col(cols)
        if tt_c is None:
            raise ValueError("no travel-time column in %s" % path)
        for row in reader:
            try:
                v = float(row[tt_c])
            except (TypeError, ValueError):
                continue
            if v == v:  # not NaN
                total += v
                n += 1
    return total, n


def sum_emissions(path):
    """Column sums for energy/emissions; missing columns -> None."""
    want = {
        "energy_1e9kj": ("totalenergy_(kj)", 1e9),
        "co2_ton": ("co2_(g)", 1e6),
        "nox_kg": ("nox_(g)", 1e3),
        "co_kg": ("co_(g)", 1e3),
        "hc_kg": ("hc_(g)", 1e3),
    }
    sums = {k: 0.0 for k in want}
    seen = {k: False for k in want}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = norm_headers(reader.fieldnames or [])
        # exact normalized-name match (avoids 'travel_time' style
        # substring collisions)
        colmap = {}
        for k, (lname, _div) in want.items():
            if lname in cols:
                colmap[k] = cols[lname]
        if not colmap:
            return {}
        for row in reader:
            for k, c in colmap.items():
                try:
                    v = float(row[c])
                except (TypeError, ValueError):
                    continue
                if v == v:
                    sums[k] += v
                    seen[k] = True
    return {k: sums[k] / want[k][1] for k in colmap if seen[k]}


def compute_delay(linkmoe_path):
    """sum((travel_time - free-flow_travel_time) * total_link_volume)/60.

    Assumes LinkMOE travel times are in minutes. Returns None when the
    file or the needed columns are absent.
    """
    if not os.path.exists(linkmoe_path):
        return None
    total = 0.0
    n = 0
    with open(linkmoe_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = norm_headers(reader.fieldnames or [])
        need = ("travel_time", "free-flow_travel_time", "total_link_volume")
        if any(k not in cols for k in need):
            return None
        tt_c, fft_c, vol_c = (cols[k] for k in need)
        for row in reader:
            try:
                tt = float(row[tt_c])
                fft = float(row[fft_c])
                vol = float(row[vol_c])
            except (TypeError, ValueError):
                continue
            if tt == tt and fft == fft and vol == vol:
                total += (tt - fft) * vol
                n += 1
    if n == 0:
        return None
    return total / 60.0


# ----------------------------------------------------------------------
# Satisfaction
# ----------------------------------------------------------------------
def satisfaction(actual_paths, ref_paths):
    """(n_satisfied, n_common, rate). Comparison is on the intersection
    of agent_ids present in both files."""
    common = set(actual_paths) & set(ref_paths)
    n_sat = sum(1 for a in common if actual_paths[a] == ref_paths[a])
    rate = (n_sat / len(common)) if common else 0.0
    return n_sat, len(common), rate


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def verdict(diff_pct):
    return "MATCH" if abs(diff_pct) <= TOLERANCE * 100 else "DIFFER"


def pct_line(label, ours, target):
    d = ours - target
    dpct = 100.0 * d / target if target else float("nan")
    return ("  %-28s ours=%12.2f  target=%12.2f  diff=%+9.2f (%+.1f%%)  %s"
            % (label, ours, target, d, dpct, verdict(dpct)))


def analyze(result_dir, bigloop_req):
    d = find_result_dir(result_dir)
    n, found = pick_bigloop(d, bigloop_req)
    if n is None:
        print("ERROR: no usable bigloop files in %s" % d)
        if found:
            print("  found bigloops: %s"
                  % sorted("%d:%s" % (k, ",".join(sorted(v)))
                            for k, v in found.items()))
        else:
            print("  (no {N}UOinfor/predict/actual_agent.csv files at all)")
            print("  hint: point --result-dir at the folder holding "
                  "DTALite_Files, or at DTALite_Files itself")
        return 2

    tags = found[n]
    info = BINGRONG_100MPR["proposed"]
    f_uo = os.path.join(d, "%dUOinfor_agent.csv" % n)
    f_pr = os.path.join(d, "%dpredict_agent.csv" % n)
    f_ac = os.path.join(d, "%dactual_agent.csv" % n)
    f_lm = os.path.join(d, "%dactual_LinkMOE.csv" % n)

    print("=" * 70)
    print("Bingrong Sun (2018) Ch.4 replication check -- %s" % VERSION)
    print("=" * 70)
    print("result dir : %s" % d)
    print("bigloop    : %d  (last complete; files: %s)"
          % (n, ", ".join(sorted(tags))))

    # ---- satisfaction ----
    print("\n-- Satisfaction: actual choices vs recommended (UO) routes --")
    actual = read_agent_paths(f_ac)
    uo = read_agent_paths(f_uo)
    n_sat, n_common, rate = satisfaction(actual, uo)
    tgt = info["satisfaction_rate"]
    pp = 100.0 * (rate - tgt)
    print("  actual file : %d agents" % len(actual))
    print("  UO file     : %d agents" % len(uo))
    if n_common < min(len(actual), len(uo)):
        print("  WARNING: only %d agent_ids in common -- compared on "
              "intersection" % n_common)
    print("  satisfied   : %d / %d = %.2f%%" % (n_sat, n_common, 100 * rate))
    print("  Bingrong    : %.2f%% (%d / %d)" % (100 * tgt, info["n_satisfied"],
                                                TOTAL_DEMAND))
    print("  diff        : %+.2f pp  %s" % (pp, verdict(100 * pp / (100 * tgt))
                                            if tgt else "n/a"))

    if "predict" in tags:
        pr = read_agent_paths(f_pr)
        n2, c2, r2 = satisfaction(actual, pr)
        print("\n-- Cross-check: actual vs predict-phase routes --")
        print("  satisfied   : %d / %d = %.2f%%" % (n2, c2, 100 * r2))
        if abs(r2 - rate) < 1e-9:
            print("  agrees with primary -- as expected at 100%% MPR "
                  "(predict == UO when no realtime users)")
        else:
            print("  DIFFERS from primary by %.2f pp -- investigate "
                  "realtime_user composition" % (100 * (r2 - rate)))
    else:
        print("\n-- Cross-check skipped: no %dpredict_agent.csv --" % n)

    # ---- vehicle-hours ----
    print("\n-- System travel time (vehicle-hours, from actual file) --")
    tot_min, nn = sum_travel_minutes(f_ac)
    vh = tot_min / 60.0
    per_agent = tot_min / nn if nn else float("nan")
    print("  experienced travel time summed over %d agents" % nn)
    print(pct_line("total vehicle-hours", vh, info["vehicle_hours"]))
    print(pct_line("min per agent", per_agent, info["vh_per_agent_min"]))
    print("  (per-agent is the fair comparison: Bingrong ran 54,045 "
          "agents)")

    # ---- delay ----
    print("\n-- Total delay (vehicle-hours, from actual_LinkMOE) --")
    delay = compute_delay(f_lm)
    if delay is None:
        print("  [SKIP] %dactual_LinkMOE.csv missing or lacks the needed "
              "columns" % n)
    else:
        print(pct_line("total delay", delay, info["delay_hours"]))
        print(pct_line("delay per agent (h)",
                       delay / nn if nn else float("nan"),
                       info["delay_per_agent_h"]))
        print("  formula: sum((travel_time - free-flow_travel_time) * "
              "total_link_volume)/60")

    # ---- energy / emissions ----
    print("\n-- Energy / emissions (column sums, actual file) --")
    em = sum_emissions(f_ac)
    labels = (("energy_1e9kj", "energy (1e9 kJ)"),
              ("co2_ton", "CO2 (t)"),
              ("nox_kg", "NOX (kg)"),
              ("co_kg", "CO (kg)"),
              ("hc_kg", "HC (kg)"))
    any_em = False
    for key, label in labels:
        if key in em:
            any_em = True
            print(pct_line(label, em[key], info[key]))
    if not any_em:
        print("  [SKIP] no energy/emissions columns found")

    # ---- summary ----
    print("\n" + "=" * 70)
    print("SUMMARY vs Bingrong 'Proposed' @ 100%% MPR")
    print("=" * 70)
    print("  satisfaction : %6.2f%%  (target %6.2f%%, %+.2f pp)"
          % (100 * rate, 100 * tgt, pp))
    print("  veh-hours    : %9.1f  (target %9.1f)" % (vh, info["vehicle_hours"]))
    print("  min/agent    : %9.2f  (target %9.2f)"
          % (per_agent, info["vh_per_agent_min"]))
    if delay is not None:
        print("  delay (vh)   : %9.1f  (target %9.1f)"
              % (delay, info["delay_hours"]))
    print("\nTolerances: MATCH = within %.0f%% relative. "
          "A DIFFER verdict is a signal to investigate, not a failure --"
          % (TOLERANCE * 100))
    print("Bingrong's MATLAB had its own quirks (e.g. tt=0 on TDlink miss);")
    print("document the residual gap for the methods section either way.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Check a completed MSA run vs Bingrong Sun (2018) "
                    "Ch.4 'Proposed' results.")
    ap.add_argument("--result-dir", default=".",
                    help="project root or DTALite_Files dir")
    ap.add_argument("--bigloop", type=int, default=None,
                    help="bigloop to analyze (default: last complete)")
    ap.add_argument("--scenario", default="proposed",
                    choices=list(BINGRONG_100MPR.keys()),
                    help="reference scenario (only 'proposed' is produced "
                         "by this pipeline; others shown for reference)")
    args = ap.parse_args(argv)

    if args.scenario != "proposed":
        i = BINGRONG_100MPR[args.scenario]
        print("Reference targets for '%s' (%s):"
              % (args.scenario, i["description"]))
        print("  satisfaction %.2f%%, vehicle-hours %,.1f"
              % (100 * i["satisfaction_rate"], i["vehicle_hours"]))
        print("NOTE: this pipeline only produces the 'proposed' scenario; "
              "run the matching scenario first, then re-run with "
              "--scenario proposed for the file check.")
        return 0

    return analyze(args.result_dir, args.bigloop)


if __name__ == "__main__":
    sys.exit(main())
