#!/usr/bin/env python3
"""
mpr_aggregate.py -- Aggregate the MPR sweep (Test 5) results into
Bingrong's dissertation Table 5 (unsatisfied guidance users by MPR) and
Table 6 (sustainability metrics by MPR), plus a human-readable
mpr_summary.txt.

Input layout (per-bigloop results):

    sweep_results/
        bigloop_0/
            0UOinfor_agent.csv, 0actual_agent.csv, 0predict_agent.csv, ...
        bigloop_1/
            ...
        ...
        bigloop_10/
            ...

A flat layout (all "<n>UOinfor_agent.csv" etc. directly in --input-dir)
is also accepted.

Usage:
    python mpr_aggregate.py --input-dir sweep_results/ --output-dir mpr_tables/
    python mpr_aggregate.py --input-dir sweep_results/ --ue-dir ue_sweep/ \
        --output-dir mpr_tables/

Outputs (all CRLF):
    mpr_table5.csv   MPR, Guidance Users, Proposed/UE unsatisfied counts and %
    mpr_table6.csv   MPR, Proposed/UE Energy (10^9 KJ), CO2 (ton),
                     NOX (kg), CO (kg), HC (kg)
    mpr_summary.txt  per-bigloop detail, comparison vs Bingrong's targets,
                     and notes on any missing data

The satisfaction logic (norm_path, read_agent_paths, satisfaction,
find_tt_col, sum_travel_minutes) is copied from wholerunner.py rather
than imported, so this script stays standalone. The MPR schedule matches
parallel_msa.runner_schedule(n=11): bigloop 0 is the baseline
(p=0.001, p_fix=0.6, p_realtime=0.4); otherwise p=0.1*i,
p_fix=(1-p)*0.6, p_realtime=(1-p)*0.4.

Guidance users per bigloop = round(p*N) + round(p_realtime*N), the same
int(round(...)) split msa.dividerand uses (members + real-time users;
fixed/habitual users excluded). Satisfaction is computed over the guided
agents selected by a dividerand replication using --seed (default 42).
Guidance user counts use int(round(p*N)) matching msa.dividerand. Note:
Python's round-half-even differs from MATLAB's round-half-away:
10% -> 24,860 (Python) vs 24,861 (Bingrong/MATLAB); 20% -> 28,103;
100% -> 54,045 (N=54,045). The 1-agent difference at 10% is a known
MATLAB/Python rounding divergence.

Sustainability comes from the per-agent columns DTALite already writes
into <n>actual_agent.csv: TotalEnergy_(KJ), CO2_(g), NOX_(g), CO_(g),
HC_(g). Columns are matched by name (punctuation/case-insensitive); any
metric whose column is absent is reported as N/A and noted in the
summary. Energy is reported in 10^9 KJ, CO2 in metric tons, NOX/CO/HC
in kg.

Preparing sweep inputs (wholerunner.py collect_1 change):
    collect_1() currently keeps only the LAST bigloop (find_last_bigloop).
    For sweep mode it should instead loop over every bigloop that has both
    <n>actual_agent.csv and <n>UOinfor_agent.csv and copy each bigloop's
    9 CSVs into results/proposed/sweep/bigloop_<n>/, then aggregate with:
        python mpr_aggregate.py --input-dir results/proposed/sweep \
            --output-dir results/proposed/mpr_tables

Python 3.9 compatible, stdlib only (csv, os, re, argparse, glob, sys);
numpy is used only if already installed (never required).
"""

import argparse
import csv
import datetime
import glob
import os
import re
import sys

NA = "N/A"


# ----------------------------------------------------------------------
# Satisfaction logic -- copied verbatim from wholerunner.py (do not import)
# ----------------------------------------------------------------------
def norm_headers(fieldnames):
    return {c.strip().lower(): c for c in fieldnames if c and c.strip()}


def norm_path(s):
    """Identical normalization on both sides: strip, split ';', drop
    empty segments, rejoin."""
    return ";".join(seg.strip() for seg in s.strip().split(";")
                    if seg.strip())


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


def satisfaction(actual_paths, ref_paths):
    """(n_satisfied, n_common, rate) on the intersection of agent_ids."""
    common = set(actual_paths) & set(ref_paths)
    n_sat = sum(1 for a in common if actual_paths[a] == ref_paths[a])
    rate = (n_sat / len(common)) if common else 0.0
    return n_sat, len(common), rate


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


# ----------------------------------------------------------------------
# MPR schedule -- mirrors parallel_msa.runner_schedule(n=11)
# ----------------------------------------------------------------------
def mpr_schedule_entry(i):
    """(p, p_fix, p_realtime) for bigloop i."""
    if i == 0:
        return 0.001, 0.6, 0.4
    p = 0.1 * i
    return p, (1 - p) * 0.6, (1 - p) * 0.4


def guidance_users(bigloop, n_agents):
    """Members + real-time users (fixed/habitual users excluded).

    Replicates msa.dividerand exactly (int(round(...)) on the same
    p/p_realtime). Note: Python round() is round-half-even, so at
    10% MPR / 54,045 agents this gives 5,404 + 19,456 = 24,860 --
    one agent fewer than Bingrong's 24,861 (MATLAB rounds half away
    from zero). The summary flags the difference where it matters.
    """
    p, _p_fix, p_realtime = mpr_schedule_entry(bigloop)
    return int(round(p * n_agents)) + int(round(p_realtime * n_agents))


def mpr_label(bigloop):
    if bigloop == 0:
        return "0%"
    return "%d%%" % int(round(100 * mpr_schedule_entry(bigloop)[0]))


def dividerand(driver_ids, p, p_realtime, p_fix, seed=1):
    """Replicates msa.py dividerand. Uses numpy if available, else
    Python random.

    NOTE: Exact agent IDs may differ from MATLAB/Bingrong's due to RNG
    differences, but the COUNT of guided agents matches
    (int(round(p*N))).
    """
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        rng = np.random.default_rng(seed)
        driver_ids = np.array(driver_ids)
        rng.shuffle(driver_ids)
        n = len(driver_ids)
        n_member = int(round(p * n))
        n_realtime = int(round(p_realtime * n))
        n_fix = int(round(p_fix * n))
        member = driver_ids[:n_member]
        realtime_user = driver_ids[n_member:n_member + n_realtime]
        fix_user = driver_ids[n_member + n_realtime:
                              n_member + n_realtime + n_fix]
        return (set(int(x) for x in member),
                set(int(x) for x in realtime_user),
                set(int(x) for x in fix_user))
    # Fallback: Python random (counts match, IDs may differ from numpy)
    import random
    rng = random.Random(seed)
    ids = list(driver_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_member = int(round(p * n))
    n_realtime = int(round(p_realtime * n))
    return (set(ids[:n_member]),
            set(ids[n_member:n_member + n_realtime]),
            set())


def mpr_schedule(bigloop):
    """Returns (p, p_realtime, p_fix) for a bigloop, matching
    parallel_msa.runner_schedule."""
    p, p_fix, p_realtime = mpr_schedule_entry(bigloop)
    return p, p_realtime, p_fix


def guided_agent_ids(bigloop, n_agents, seed):
    """Returns set of guided agent IDs (member union realtime_user)
    for a bigloop."""
    p, p_realtime, p_fix = mpr_schedule(bigloop)
    # Agent IDs are 0..n_agents-1 (matching msa.py driver_ids =
    # np.arange(dsize))
    member, realtime, _fix = dividerand(range(n_agents), p, p_realtime,
                                        p_fix, seed=seed)
    return member | realtime


# ----------------------------------------------------------------------
# Sustainability -- per-agent columns DTALite writes into actual_agent.csv
# ----------------------------------------------------------------------
def _canon(col):
    return re.sub(r"[^a-z0-9]", "", col.strip().lower())


# (metric, candidate canonical column names, divisor to table unit, unit)
SUST_SPECS = [
    ("energy", ("totalenergykj", "energykj", "energyinkj"), 1e9, "10^9 KJ"),
    ("co2",    ("co2g",),  1e6, "ton"),
    ("nox",    ("noxg",),  1e3, "kg"),
    ("co",     ("cog",),   1e3, "kg"),
    ("hc",     ("hcg",),   1e3, "kg"),
]


def sum_sustainability(path):
    """Sum per-agent sustainability columns.

    Returns (totals, cols_used): totals maps metric -> float in table
    units, or None when the column is absent; cols_used maps metric ->
    the actual CSV column name that was summed.
    """
    totals = {m: None for m, _, _, _ in SUST_SPECS}
    cols_used = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        canon = {_canon(c): c for c in (reader.fieldnames or []) if c}
        for metric, cands, _div, _unit in SUST_SPECS:
            hit = next((canon[c] for c in cands if c in canon), None)
            if hit is not None:
                cols_used[metric] = hit
                totals[metric] = 0.0
        if not cols_used:
            return totals, cols_used
        for row in reader:
            for metric, _c, divisor, _u in SUST_SPECS:
                col = cols_used.get(metric)
                if col is None:
                    continue
                try:
                    v = float(row[col])
                except (TypeError, ValueError):
                    continue
                if v == v:  # not NaN
                    totals[metric] += v
        for metric, _c, divisor, _u in SUST_SPECS:
            if totals[metric] is not None:
                totals[metric] = totals[metric] / divisor
    return totals, cols_used


# ----------------------------------------------------------------------
# Bingrong's published targets (for the summary comparison)
# ----------------------------------------------------------------------
# Table 5 (unsatisfied %): only the 10% and 100% endpoint rows are on
# record so far; mid-MPR rows are TBD from the dissertation.
BINGRONG_TABLE5 = {
    10:  {"users": 24861, "prop_unsat": 4829, "prop_pct": 19.4,
          "ue_unsat": 5994, "ue_pct": 24.1},
    100: {"users": 54045, "prop_unsat": 6137, "prop_pct": 11.4,
          "ue_unsat": 18082, "ue_pct": 33.5},
}

# Table 6 endpoint rows: (proposed, UE) for energy, CO2, NOX, CO, HC.
BINGRONG_TABLE6 = {
    0:   {"energy": (2.72, 2.72), "co2": (195.0, 195.0),
          "nox": (143.0, 143.0), "co": (1550.0, 1550.0),
          "hc": (49.0, 49.0)},
    10:  {"energy": (2.71, 2.74), "co2": (194.5, 196.7),
          "nox": (142.5, 143.4), "co": (1543.8, 1556.5),
          "hc": (48.9, 49.6)},
    100: {"energy": (2.65, 2.97), "co2": (190.8, 213.5),
          "nox": (141.3, 148.0), "co": (1525.3, 1628.2),
          "hc": (47.8, 55.4)},
}


# ----------------------------------------------------------------------
# Per-bigloop analysis
# ----------------------------------------------------------------------
def warn(msg):
    sys.stderr.write("warning: %s\n" % msg)


def discover_bigloop_dirs(root):
    """bigloop -> directory holding its CSVs.

    Accepts bigloop_<n>/ subdirectories, or a flat layout with
    <n>UOinfor/predict/actual_agent.csv directly in root.
    """
    out = {}
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in sorted(names):
        m = re.match(r"^bigloop_(\d+)$", name)
        if m and os.path.isdir(os.path.join(root, name)):
            out[int(m.group(1))] = os.path.join(root, name)
    if not out:
        for name in sorted(names):
            m = re.match(r"^(\d+)(?:UOinfor|predict|actual)_agent\.csv$", name)
            if m:
                out.setdefault(int(m.group(1)), root)
    return out


def analyze_bigloop(d, bigloop, seed=42):
    """Analyze one bigloop. Returns a result dict, or None (warning).

    Satisfaction is restricted to the guided agents (members +
    real-time users) selected by the dividerand replication with the
    given seed, matching the Table 5 denominator.
    """
    f_uo = os.path.join(d, "%dUOinfor_agent.csv" % bigloop)
    f_ac = os.path.join(d, "%dactual_agent.csv" % bigloop)
    missing = [p for p in (f_uo, f_ac) if not os.path.isfile(p)]
    if missing:
        warn("bigloop %d skipped: missing %s"
             % (bigloop, ", ".join(os.path.basename(p) for p in missing)))
        return None
    try:
        actual = read_agent_paths(f_ac)
        uo = read_agent_paths(f_uo)
    except (OSError, ValueError, csv.Error) as e:
        warn("bigloop %d skipped: %s" % (bigloop, e))
        return None
    n_agents = len(actual)
    guided = guided_agent_ids(bigloop, n_agents, seed)
    # actual/uo keys are strings; guided are ints
    guided_str = set(str(g) for g in guided)
    actual_g = {k: v for k, v in actual.items() if k in guided_str}
    uo_g = {k: v for k, v in uo.items() if k in guided_str}
    n_sat, n_common, rate = satisfaction(actual_g, uo_g)
    n_unsat = n_common - n_sat
    unsat_pct = 100.0 * n_unsat / n_common if n_common else 0.0

    try:
        tot_min, n_tt = sum_travel_minutes(f_ac)
        veh_hrs = tot_min / 60.0
        min_per_agent = tot_min / n_tt if n_tt else None
    except (OSError, ValueError, csv.Error) as e:
        warn("bigloop %d: vehicle-hours unavailable (%s)" % (bigloop, e))
        veh_hrs, min_per_agent = None, None

    try:
        sust, sust_cols = sum_sustainability(f_ac)
    except (OSError, csv.Error) as e:
        warn("bigloop %d: sustainability unavailable (%s)" % (bigloop, e))
        sust = {m: None for m, _, _, _ in SUST_SPECS}
        sust_cols = {}

    return {
        "bigloop": bigloop,
        "mpr": mpr_label(bigloop),
        "n_agents": n_agents,
        "guidance_users": guidance_users(bigloop, n_agents),
        "n_sat": n_sat,
        "n_common": n_common,
        "n_guided": n_common,
        "n_unsat": n_unsat,
        "unsat_pct": unsat_pct,
        "veh_hrs": veh_hrs,
        "min_per_agent": min_per_agent,
        "sust": sust,
        "sust_cols": sust_cols,
    }


def analyze_sweep(root, seed=42):
    """bigloop -> result dict for every analyzable bigloop under root."""
    results = {}
    dirs = discover_bigloop_dirs(root)
    for b in sorted(dirs):
        r = analyze_bigloop(dirs[b], b, seed=seed)
        if r is not None:
            results[b] = r
    return results


# ----------------------------------------------------------------------
# Output writers (CRLF)
# ----------------------------------------------------------------------
TABLE5_HEADER = ["MPR", "Guidance Users", "Proposed Unsatisfied",
                 "Proposed Unsatisfied %", "UE Unsatisfied",
                 "UE Unsatisfied %"]

TABLE6_HEADER = ["MPR", "Proposed Energy (10^9 KJ)", "UE Energy (10^9 KJ)",
                 "Proposed CO2 (ton)", "UE CO2 (ton)",
                 "Proposed NOX (kg)", "UE NOX (kg)",
                 "Proposed CO (kg)", "UE CO (kg)",
                 "Proposed HC (kg)", "UE HC (kg)"]

# decimals per Table 6 metric
SUST_DECIMALS = {"energy": 2, "co2": 1, "nox": 1, "co": 1, "hc": 1}


def _fmt(v, nd):
    return NA if v is None else ("%.*f" % (nd, v))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:  # csv emits \r\n -> CRLF
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_crlf_text(path, text):
    """Write text with CRLF line endings (one normalization pass)."""
    data = text.replace("\r\n", "\n").replace("\n", "\r\n")
    with open(path, "w", newline="") as f:
        f.write(data)


def build_table5(prop, ue):
    """Rows for bigloops 1..10 (bigloop 0 excluded, per Table 5)."""
    rows = []
    for b in sorted(prop):
        if b == 0:
            continue
        r = prop[b]
        row = [r["mpr"], r["guidance_users"], r["n_unsat"],
               "%.1f%%" % r["unsat_pct"]]
        u = ue.get(b) if ue else None
        if u is not None:
            row += [u["n_unsat"], "%.1f%%" % u["unsat_pct"]]
        else:
            row += [NA, NA]
        rows.append(row)
    return rows


def build_table6(prop, ue, ue_provided):
    """Rows for bigloops 0..10. UE columns are N/A wherever no UE data
    is provided (including the 0% baseline row)."""
    rows = []
    for b in sorted(prop):
        r = prop[b]
        u = ue.get(b) if ue else None
        row = [r["mpr"]]
        for metric, _c, _d, _unit in SUST_SPECS:
            nd = SUST_DECIMALS[metric]
            p_val = r["sust"][metric]
            if u is not None:
                u_val = u["sust"][metric]
            else:
                u_val = None  # N/A when no UE data (incl. the 0% row)
            row += [_fmt(p_val, nd), _fmt(u_val, nd)]
        rows.append(row)
    return rows


def build_summary(prop, ue, ue_provided, input_dir, ue_dir, out_dir):
    L = []
    L.append("MPR sweep aggregation -- mpr_summary.txt")
    L.append("generated: %s" % datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M"))
    L.append("input: %s (%d bigloops analyzed: %s)"
             % (input_dir, len(prop),
                ", ".join(str(b) for b in sorted(prop)) or "none"))
    if ue_provided:
        L.append("ue input: %s (%d bigloops analyzed: %s)"
                 % (ue_dir, len(ue),
                    ", ".join(str(b) for b in sorted(ue)) or "none"))
    else:
        L.append("ue input: not provided -- all UE columns are N/A")
    L.append("")

    L.append("Per-bigloop results (proposed)")
    L.append("-" * 78)
    L.append("%-8s %-4s %-8s %-9s %-8s %-7s %-8s %-9s %-10s" %
             ("bigloop", "MPR", "agents", "guidance", "sat%", "unsat",
              "unsat%", "veh-hrs", "min/agent"))
    for b in sorted(prop):
        r = prop[b]
        sat_pct = 100.0 - r["unsat_pct"]
        L.append("%-8d %-4s %-8d %-9d %-8.2f %-7d %-8.1f %-9s %-10s" % (
            b, r["mpr"], r["n_agents"], r["guidance_users"], sat_pct,
            r["n_unsat"], r["unsat_pct"],
            _fmt(r["veh_hrs"], 1),
            _fmt(r["min_per_agent"], 2)))
    L.append("")

    # Sustainability columns actually used (first bigloop with any).
    used = {}
    for b in sorted(prop):
        for m, col in prop[b]["sust_cols"].items():
            used.setdefault(m, col)
    L.append("Sustainability columns summed from <n>actual_agent.csv:")
    for metric, _c, _div, unit in SUST_SPECS:
        if metric in used:
            L.append("  %-6s <- %s  (reported in %s)" % (metric, used[metric],
                                                        unit))
        else:
            L.append("  %-6s <- column not found -> N/A" % metric)
    L.append("")

    # Comparison vs Bingrong's Table 5 endpoints.
    L.append("Comparison vs Bingrong dissertation Table 5 (unsatisfied %)")
    L.append("-" * 78)
    for mpr_int in (10, 100):
        t = BINGRONG_TABLE5[mpr_int]
        b = mpr_int // 10
        if b in prop:
            r = prop[b]
            d = r["unsat_pct"] - t["prop_pct"]
            L.append("%3d%% MPR: proposed unsat %.1f%% vs Bingrong %.1f%% "
                     "(diff %+.1f pp; guidance users %d vs %d)" %
                     (mpr_int, r["unsat_pct"], t["prop_pct"], d,
                      r["guidance_users"], t["users"]))
            u = ue.get(b) if ue else None
            if u is not None:
                du = u["unsat_pct"] - t["ue_pct"]
                L.append("%3d%% MPR: UE unsat %.1f%% vs Bingrong %.1f%% "
                         "(diff %+.1f pp)" %
                         (mpr_int, u["unsat_pct"], t["ue_pct"], du))
            elif ue_provided:
                L.append("%3d%% MPR: UE bigloop missing -> no comparison"
                         % mpr_int)
        else:
            L.append("%3d%% MPR: proposed bigloop missing -> no comparison"
                     % mpr_int)
    L.append("")

    # Comparison vs Bingrong's Table 6 endpoints.
    L.append("Comparison vs Bingrong dissertation Table 6 (sustainability)")
    L.append("-" * 78)
    for mpr_int in (0, 10, 100):
        t = BINGRONG_TABLE6[mpr_int]
        b = mpr_int // 10
        if b in prop:
            r = prop[b]
            bits = []
            for metric, _c, _d, unit in SUST_SPECS:
                tgt_p, _tgt_u = t[metric]
                v = r["sust"][metric]
                if v is None:
                    bits.append("%s: N/A (Bingrong %.1f %s)" %
                                (metric, tgt_p, unit))
                else:
                    bits.append("%s: %s vs %.1f %s" %
                                (metric, _fmt(v, SUST_DECIMALS[metric]),
                                 tgt_p, unit))
            L.append("%3d%% MPR proposed: %s" % (mpr_int, "; ".join(bits)))
        else:
            L.append("%3d%% MPR: proposed bigloop missing -> no comparison"
                     % mpr_int)
    L.append("")

    L.append("Notes")
    L.append("-" * 78)
    if not ue_provided:
        L.append("- UE-guidance MPR variant not provided: all UE columns "
                 "in mpr_table5.csv and mpr_table6.csv are N/A.")
    missing_ue = [b for b in sorted(prop) if b != 0 and
                  (not ue or b not in ue)]
    if ue_provided and missing_ue:
        L.append("- UE bigloops missing (N/A in tables): %s"
                 % ", ".join(str(b) for b in missing_ue))
    if any(prop[b]["sust"][m] is None
           for b in prop for m, _, _, _ in SUST_SPECS):
        L.append("- Some sustainability metrics are N/A: the corresponding "
                 "columns were not found in <n>actual_agent.csv.")
    L.append("- Table 5 unsatisfied % uses the guided-agent population in "
             "both files (n_common) as the denominator, matching Bingrong's "
             "Table 5 (e.g. 4,829/24,861 = 19.4% at 10% MPR).")
    L.append("- Guidance users = round(p*N) + round(p_realtime*N), the "
             "dividerand split (members + real-time users); --seed "
             "controls which agents are guided (default 42).")
    L.append("- Rounding: Python round() is round-half-even, so at 10% MPR / "
             "54,045 agents the pipeline assigns 24,860 guidance users, "
             "one fewer than Bingrong's 24,861 (MATLAB round-half-away).")
    L.append("")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Aggregate MPR sweep per-bigloop results into "
                    "Bingrong's Tables 5 & 6 (mpr_table5.csv, "
                    "mpr_table6.csv, mpr_summary.txt).")
    ap.add_argument("--input-dir", required=True,
                    help="directory with bigloop_<n>/ per-bigloop outputs "
                         "(flat <n>*.csv layout also accepted)")
    ap.add_argument("--ue-dir", default=None,
                    help="optional directory with the UE-guidance MPR "
                         "variant per-bigloop outputs (same layout)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for dividerand to compute guided agent IDs. "
                         "Default: 42.")
    ap.add_argument("--output-dir", required=True,
                    help="directory for mpr_table5.csv, mpr_table6.csv, "
                         "mpr_summary.txt")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isdir(args.input_dir):
        sys.stderr.write("error: input dir not found: %s\n"
                         % args.input_dir)
        return 1
    ue_provided = args.ue_dir is not None
    if ue_provided and not os.path.isdir(args.ue_dir):
        sys.stderr.write("error: ue dir not found: %s\n" % args.ue_dir)
        return 1

    prop = analyze_sweep(args.input_dir, seed=args.seed)
    if not prop:
        sys.stderr.write("error: no analyzable bigloops in %s\n"
                         % args.input_dir)
        return 1
    ue = analyze_sweep(args.ue_dir, seed=args.seed) if ue_provided else {}

    os.makedirs(args.output_dir, exist_ok=True)
    p5 = os.path.join(args.output_dir, "mpr_table5.csv")
    p6 = os.path.join(args.output_dir, "mpr_table6.csv")
    ps = os.path.join(args.output_dir, "mpr_summary.txt")
    write_csv(p5, TABLE5_HEADER, build_table5(prop, ue))
    write_csv(p6, TABLE6_HEADER, build_table6(prop, ue, ue_provided))
    write_crlf_text(ps, build_summary(prop, ue, ue_provided,
                                     os.path.abspath(args.input_dir),
                                     os.path.abspath(args.ue_dir)
                                     if ue_provided else None,
                                     os.path.abspath(args.output_dir)))
    print("wrote %s (%d data rows)" % (p5, len(build_table5(prop, ue))))
    print("wrote %s (%d data rows)" % (p6, len(build_table6(prop, ue,
                                                           ue_provided))))
    print("wrote %s" % ps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
