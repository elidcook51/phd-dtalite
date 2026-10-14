#!/usr/bin/env python3
"""Manual Test 5 collection - replicates wholerunner.collect_5 for hung runs.

Usage (from repo root):
    python collect_test5_manual.py --run-dir temp/run_38356 --seed 123

Copies per-bigloop outputs from the temp sandboxes to results/test5/,
then runs mpr_aggregate.py.
"""
import argparse
import os
import shutil
import subprocess
import sys

T1_KINDS = ("agent", "LinkMOE", "LinkTDMOE")
T1_TAGS = ("UOinfor", "predict", "actual")


def collect_5_manual(sub, sandbox, results_root):
    """Replicates wholerunner.collect_5."""
    sweep_name = "proposed_sweep" if sub == "P" else "ue_sweep"
    lo = 0 if sub == "P" else 1
    data = os.path.join(sandbox, "DTALite_Files")
    collected = []
    for n in range(lo, 11):
        f_act = os.path.join(data, "%dactual_agent.csv" % n)
        f_uo = os.path.join(data, "%dUOinfor_agent.csv" % n)
        if not (os.path.isfile(f_act) and os.path.isfile(f_uo)):
            print("  WARNING: bigloop %d missing actual/UOinfor -- skipped" % n)
            continue
        out = os.path.join(results_root, "test5", sweep_name, "bigloop_%d" % n)
        os.makedirs(out, exist_ok=True)
        copied = 0
        for tag in T1_TAGS:
            for kind in T1_KINDS:
                fn = "%d%s_%s.csv" % (n, tag, kind)
                src = os.path.join(data, fn)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(out, fn))
                    copied += 1
        print("  bigloop %d -> %s (%d files)" % (n, out, copied))
        collected.append(n)
    print("Collected %d/%d bigloops for 5%s" % (len(collected), 11 - lo, sub))
    return collected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="Temp run dir (e.g. temp/run_38356)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--results-root", default="results")
    args = ap.parse_args()

    run_dir = args.run_dir
    results_root = args.results_root

    # Collect 5P
    print("Collecting 5P (proposed sweep)...")
    sb_p = os.path.join(run_dir, "test_5P")
    if not os.path.isdir(sb_p):
        print("ERROR: %s not found" % sb_p, file=sys.stderr)
        sys.exit(1)
    collected_p = collect_5_manual("P", sb_p, results_root)

    # Collect 5U
    print("\nCollecting 5U (UE sweep)...")
    sb_u = os.path.join(run_dir, "test_5U")
    if not os.path.isdir(sb_u):
        print("ERROR: %s not found" % sb_u, file=sys.stderr)
        sys.exit(1)
    collected_u = collect_5_manual("U", sb_u, results_root)

    if not collected_p or not collected_u:
        print("\nERROR: One or both sweeps have no collected bigloops. "
              "Cannot aggregate.", file=sys.stderr)
        sys.exit(1)

    # Run aggregation
    print("\nRunning mpr_aggregate.py...")
    cmd = [sys.executable, "mpr_aggregate.py",
           "--input-dir", os.path.join(results_root, "test5", "proposed_sweep"),
           "--ue-dir", os.path.join(results_root, "test5", "ue_sweep"),
           "--output-dir", os.path.join(results_root, "test5"),
           "--seed", str(args.seed)]
    # mpr_aggregate writes to results/test5/ by default; check its --output-dir
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print("Aggregation FAILED (rc=%d)" % result.returncode, file=sys.stderr)
        sys.exit(1)

    print("\nDone. Check results/test5/mpr_table5.csv and mpr_table6.csv")


if __name__ == "__main__":
    main()
