#!/usr/bin/env python3
"""
wholerunner.py — Run all 4 Bingrong Ch.4 test cases sequentially.

Bingrong Sun (2018) Chapter 4 test cases at 100% MPR:
  Test 1 — Proposed (proactive preference-aware guidance) ... via v9 pipeline
  Test 2 — Real-time guidance, sigma=1 min ................. via scenarios_realtime.py
  Test 3 — Real-time guidance, sigma=10 min ................ via scenarios_realtime.py
  Test 4 — UE as information (MSA to 0.01% gap) ........... via scenarios_ue.py

Each baseline (Tests 2-4) is followed by preference_reaction.py which
computes satisfaction vs Bingrong's targets.

Tests MUST run sequentially: all scripts share DTALite_Files/input_agent.csv
and DTALite_Files/output_*.csv. Parallel execution would clobber shared state.

Usage:
  # Baselines only (Tests 2-4), the common case:
  python wholerunner.py --data-dir DTALite_Files --results-dir results

  # Include Test 1 (v9 pipeline — long run, 18 bigloops):
  python wholerunner.py --data-dir DTALite_Files --results-dir results --include-test1

  # Only specific tests:
  python wholerunner.py --data-dir DTALite_Files --results-dir results --tests 2,4

  # 54k demand (Bingrong's scale):
  python wholerunner.py --data-dir DTALite_Files --results-dir results_54k --tests 2,3,4

Outputs per test under <results-dir>/:
  realtime_1/   realtime_routes.csv, realtime_summary.txt, actual_routes.csv, reaction_summary.txt
  realtime_10/  (same)
  ue_baseline/  ue_routes.csv, ue_summary.txt, actual_routes.csv, reaction_summary.txt

Requires: Python 3.9+, scenarios_realtime.py, scenarios_ue.py,
          preference_reaction.py in the same directory as this file.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Bingrong's targets for satisfaction
TARGETS = {
    2: 70.49,  # real-time 1min
    3: 70.19,  # real-time 10min
    4: 66.54,  # UE as information
}

# Test 1 target (for reference)
TARGET_T1_SAT = 88.64


def log(msg):
    print("[wholerunner] %s" % msg, flush=True)


def run_cmd(cmd, desc):
    """Run a command, fail fast on error."""
    log("%s ..." % desc)
    log("  $ %s" % " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    dt = time.time() - t0
    if r.returncode != 0:
        print("ERROR: %s failed (rc=%d)" % (desc, r.returncode),
              file=sys.stderr)
        sys.exit(1)
    log("%s done (%.1f min)" % (desc, dt / 60.0))


def run_test2(data_dir, results_dir):
    out = os.path.join(results_dir, "realtime_1")
    run_cmd(
        [sys.executable, os.path.join(HERE, "scenarios_realtime.py"),
         "--sigma", "1", "--data-dir", data_dir, "--result-dir", out],
        "Test 2: real-time sigma=1 (guidance)")
    run_cmd(
        [sys.executable, os.path.join(HERE, "preference_reaction.py"),
         "--guidance-csv", os.path.join(out, "realtime_routes.csv"),
         "--result-dir", out,
         "--target-satisfaction", str(TARGETS[2])],
        "Test 2: preference reaction")


def run_test3(data_dir, results_dir):
    out = os.path.join(results_dir, "realtime_10")
    run_cmd(
        [sys.executable, os.path.join(HERE, "scenarios_realtime.py"),
         "--sigma", "10", "--data-dir", data_dir, "--result-dir", out],
        "Test 3: real-time sigma=10 (guidance)")
    run_cmd(
        [sys.executable, os.path.join(HERE, "preference_reaction.py"),
         "--guidance-csv", os.path.join(out, "realtime_routes.csv"),
         "--result-dir", out,
         "--target-satisfaction", str(TARGETS[3])],
        "Test 3: preference reaction")


def run_test4(data_dir, results_dir):
    out = os.path.join(results_dir, "ue_baseline")
    run_cmd(
        [sys.executable, os.path.join(HERE, "scenarios_ue.py"),
         "--data-dir", data_dir, "--result-dir", out],
        "Test 4: UE baseline (MSA guidance)")
    run_cmd(
        [sys.executable, os.path.join(HERE, "preference_reaction.py"),
         "--guidance-csv", os.path.join(out, "ue_routes.csv"),
         "--result-dir", out,
         "--target-satisfaction", str(TARGETS[4])],
        "Test 4: preference reaction")


def run_test1(data_dir, results_dir):
    """Test 1 via the v9 pipeline (runner_parallel.py). Long run."""
    log("Test 1: launching v9 pipeline (runner_parallel.py) ...")
    log("  This runs 18 bigloops and may take 30+ minutes.")
    run_cmd(
        [sys.executable, os.path.join(HERE, "runner_parallel.py")],
        "Test 1: v9 Proposed pipeline")
    log("Test 1 complete. Analyze with runner_bingrong.py.")


def print_summary(results_dir, tests):
    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for t in tests:
        if t == 2:
            p = os.path.join(results_dir, "realtime_1", "reaction_summary.txt")
        elif t == 3:
            p = os.path.join(results_dir, "realtime_10", "reaction_summary.txt")
        elif t == 4:
            p = os.path.join(results_dir, "ue_baseline", "reaction_summary.txt")
        else:
            continue
        if os.path.isfile(p):
            log("--- Test %d ---" % t)
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("satisfaction:") or \
                       line.startswith("vehicle-hours") or \
                       line.startswith("per-agent"):
                        log("  " + line)
        else:
            log("Test %d: no reaction_summary.txt found" % t)
    log("=" * 60)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run all Bingrong Ch.4 baseline tests sequentially.")
    ap.add_argument("--data-dir", default="DTALite_Files",
                    help="DTALite_Files directory")
    ap.add_argument("--results-dir", default="results",
                    help="output root directory")
    ap.add_argument("--tests", default="2,3,4",
                    help="comma-separated tests to run (default: 2,3,4)")
    ap.add_argument("--include-test1", action="store_true",
                    help="also run Test 1 via v9 pipeline (long)")
    args = ap.parse_args(argv)

    tests = [int(x.strip()) for x in args.tests.split(",") if x.strip()]
    if args.include_test1 and 1 not in tests:
        tests = [1] + tests

    # Verify required scripts exist
    for fn in ["scenarios_realtime.py", "scenarios_ue.py",
               "preference_reaction.py"]:
        if not os.path.isfile(os.path.join(HERE, fn)):
            print("ERROR: %s not found in %s" % (fn, HERE), file=sys.stderr)
            sys.exit(1)

    log("Tests to run: %s" % tests)
    log("Data dir: %s" % args.data_dir)
    log("Results dir: %s" % args.results_dir)
    os.makedirs(args.results_dir, exist_ok=True)

    t_start = time.time()
    for t in tests:
        if t == 1:
            run_test1(args.data_dir, args.results_dir)
        elif t == 2:
            run_test2(args.data_dir, args.results_dir)
        elif t == 3:
            run_test3(args.data_dir, args.results_dir)
        elif t == 4:
            run_test4(args.data_dir, args.results_dir)
        else:
            print("ERROR: unknown test %d (valid: 1,2,3,4)" % t,
                  file=sys.stderr)
            sys.exit(1)

    print_summary(args.results_dir, tests)
    log("All done (%.1f min total)" % ((time.time() - t_start) / 60.0))


if __name__ == "__main__":
    main()
