#!/usr/bin/env python3
"""Total UE vehicle-hours per MPR for the Test 5 table (Track A paper).

Mirrors mpr_aggregate.sum_travel_minutes: sums the travel-time-minutes
column over ALL rows of each UE sweep actual agent file, divided by 60.
Vehicle-hours here are totals over all agents in the UE sweep's reaction
run, matching the proposed vehicle-hours convention (not restricted to
the guided population).

Input: results/test5/ue_sweep/ with bigloop_<N>/<N>actual_agent.csv.
Output: prints MPR / UE vehicle-hours / agents, and writes
ue_vehhrs.csv next to this script.

Usage (Windows):
    python ue_vehhrs.py C:\\Users\\parkl\\Desktop\\DTALite_Analysis_Eli\\phd-dtalite\\results\\test5\\ue_sweep

Stdlib only. Read-only on inputs.
"""

import csv
import glob
import os
import re
import sys

BIGLOOP_DIR_RE = re.compile(r"^bigloop_(\d+)$")


def norm_headers(fieldnames):
    return {c.strip().lower(): c for c in fieldnames if c and c.strip()}


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


def find_actual(d, bigloop):
    p = os.path.join(d, "%dactual_agent.csv" % bigloop)
    if os.path.isfile(p):
        return p
    hits = sorted(glob.glob(os.path.join(d, "*actual_agent.csv")))
    return hits[0] if hits else None


def main(argv):
    if len(argv) < 2:
        print("Usage: python ue_vehhrs.py <ue_sweep_root>")
        return 2
    root = argv[1]
    if not os.path.isdir(root):
        print("ERROR: ue_sweep root not found: %s" % root)
        return 1
    rows = []
    for name in sorted(os.listdir(root)):
        m = BIGLOOP_DIR_RE.match(name)
        if not m:
            continue
        bl = int(m.group(1))
        d = os.path.join(root, name)
        p = find_actual(d, bl)
        if p is None:
            print("bigloop %d: NO actual agent file" % bl)
            continue
        try:
            tot_min, n = sum_travel_minutes(p)
        except ValueError as e:
            print("bigloop %d: %s" % (bl, e))
            continue
        mpr = "%d%%" % (bl * 10)
        veh_hrs = tot_min / 60.0
        rows.append((mpr, veh_hrs, n))
        print("%-6s UE veh-hrs %12.1f  (%d agents)" % (mpr, veh_hrs, n))
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ue_vehhrs.csv")
    with open(out_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("MPR,UE vehicle-hours,Agents\n")
        for mpr, veh_hrs, n in rows:
            f.write("%s,%.1f,%d\n" % (mpr, veh_hrs, n))
    print("Wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
