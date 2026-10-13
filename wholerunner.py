#!/usr/bin/env python3
"""
wholerunner.py -- Run all 4 Bingrong Ch.4 test cases in parallel sandboxes.

Bingrong Sun (2018) Chapter 4 test cases at 100% MPR:
  Test 1 -- Proposed (proactive preference-aware guidance) ... via v9 pipeline
  Test 2 -- Real-time guidance, sigma=1 min ................. via scenarios_realtime.py
  Test 3 -- Real-time guidance, sigma=10 min ................ via scenarios_realtime.py
  Test 4 -- UE as information (MSA to 0.01% gap) ............ via scenarios_ue.py

Each test gets its own sandbox under temp/run_<pid>/test_<N>/: a private copy of
DTALite_Files/ (the real one stays read-only) plus the scripts that test
needs. Tests run concurrently as separate processes -- no shared writable
state -- and all outputs are collected under <results-dir>/.

Usage:
  # All four tests in parallel (Test 1 is a long v9 run):
  python wholerunner.py --data-dir DTALite_Files --results-dir results

  # Baselines only (Tests 2-4), the common case:
  python wholerunner.py --data-dir DTALite_Files --results-dir results --skip-test1

  # Only specific tests:
  python wholerunner.py --data-dir DTALite_Files --results-dir results --tests 2,3

Outputs per test under <results-dir>/:
  proposed/     proposed_summary.txt, full_output/ (bigloop result CSVs)
  realtime_1/   realtime_routes.csv, realtime_summary.txt, actual_routes.csv,
                reaction_summary.txt, full_output/
  realtime_10/  (same)
  ue_baseline/  ue_routes.csv, ue_summary.txt, actual_routes.csv,
                reaction_summary.txt, full_output/

Sandboxes live in temp/run_<pid>/test_<N>/ and are deleted on success unless
--keep-temp is given. On failure the sandbox is kept and its path is
printed for debugging; other tests keep running.

Python 3.9 compatible, stdlib only.
"""

import argparse
import concurrent.futures
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# Bingrong's satisfaction targets (100% MPR), used by preference_reaction
TARGETS = {
    2: 70.49,  # real-time 1min
    3: 70.19,  # real-time 10min
    4: 66.54,  # UE as information
}

# Test 1 (Proposed) reference targets, Sun (2018) Ch.4 Tables 8 & 9
T1_SAT_RATE = 88.64      # % of agents whose actual choice matched UO guidance
T1_VEH_HRS = 24719.3     # system vehicle-hours (54,045 agents)
T1_MIN_PER_AGENT = 27.44

# Per-test configuration: which scripts each sandbox needs and what to run.
TESTS = {
    1: {
        "label": "Test 1: Proposed -- v9 pipeline",
        "scripts": ["runner_parallel.py", "parallel_msa.py", "msa.py",
                    "assignment.py", "realtimeassignment.py",
                    "traveltimecal.py", "progress.py", "choiceset.py"],
    },
    2: {
        "label": "Test 2: real-time guidance, sigma=1 min",
        "scripts": ["scenarios_realtime.py", "preference_reaction.py"],
        "sigma": 1,
        "target": TARGETS[2],
    },
    3: {
        "label": "Test 3: real-time guidance, sigma=10 min",
        "scripts": ["scenarios_realtime.py", "preference_reaction.py"],
        "sigma": 10,
        "target": TARGETS[3],
    },
    4: {
        "label": "Test 4: UE as information (MSA baseline)",
        "scripts": ["scenarios_ue.py", "preference_reaction.py"],
        "target": TARGETS[4],
    },
}

# Results subdirectory per test
RESULT_NAMES = {1: "proposed", 2: "realtime_1", 3: "realtime_10",
                4: "ue_baseline"}

# Bigloop result files collected for Test 1: the 9 result CSVs from the
# parity checker ({b}UOinfor/predict/actual x agent/LinkMOE/LinkTDMOE).
T1_KINDS = ("agent", "LinkMOE", "LinkTDMOE")
T1_TAGS = ("UOinfor", "predict", "actual")
TAG_RE = re.compile(r"^(\d+)(UOinfor|predict|actual)_agent\.csv$")


def log(msg):
    print("[wholerunner] %s" % msg, flush=True)


def fmt_min(sec):
    return "%.1f min" % (sec / 60.0)


# ----------------------------------------------------------------------
# Sandbox setup (fail-fast: any error here aborts before anything runs)
# ----------------------------------------------------------------------
def setup_sandbox(test, data_dir, temp_root):
    """Create temp/run_<pid>/test_<N>/ with a private DTALite_Files/ copy and the
    test's scripts. Raises on any error."""
    cfg = TESTS[test]
    sandbox = os.path.join(temp_root, "test_%d" % test)
    if os.path.isdir(sandbox):
        log("removing stale sandbox %s" % sandbox)
        shutil.rmtree(sandbox)
    os.makedirs(sandbox)

    sandbox_data = os.path.join(sandbox, "DTALite_Files")
    log("test %d: copying DTALite_Files -> sandbox ..." % test)
    shutil.copytree(data_dir, sandbox_data)

    for fn in cfg["scripts"]:
        src = os.path.join(HERE, fn)
        if not os.path.isfile(src):
            raise RuntimeError(
                "required script not found next to wholerunner.py: %s" % fn)
        shutil.copy2(src, sandbox)
    log("test %d: sandbox ready (%s)" % (test, sandbox))
    return sandbox


# ----------------------------------------------------------------------
# Per-test command lines
# ----------------------------------------------------------------------
def test1_commands(sandbox):
    return [
        ([PY, os.path.join(sandbox, "runner_parallel.py")],
         "v9 pipeline (runner_parallel.py)"),
    ]


def test24_commands(test, sandbox):
    """Guidance scenario followed by the preference-reaction step."""
    staging = os.path.join(sandbox, "staging")
    os.makedirs(staging, exist_ok=True)
    data = os.path.join(sandbox, "DTALite_Files")
    cfg = TESTS[test]

    if test in (2, 3):
        scenario = [PY, os.path.join(sandbox, "scenarios_realtime.py"),
                    "--sigma", str(cfg["sigma"]),
                    "--data-dir", data, "--result-dir", staging]
        scenario_desc = "real-time sigma=%d (guidance)" % cfg["sigma"]
        guidance_csv = os.path.join(staging, "realtime_routes.csv")
    else:  # test 4
        scenario = [PY, os.path.join(sandbox, "scenarios_ue.py"),
                    "--data-dir", data, "--result-dir", staging]
        scenario_desc = "UE baseline (MSA guidance)"
        guidance_csv = os.path.join(staging, "ue_routes.csv")

    reaction = [PY, os.path.join(sandbox, "preference_reaction.py"),
                "--guidance-csv", guidance_csv,
                "--data-dir", data,
                "--result-dir", staging,
                "--target-satisfaction", str(cfg["target"])]
    return [(scenario, scenario_desc),
            (reaction, "preference reaction")]


# ----------------------------------------------------------------------
# Test execution (one thread per test; each launches its own processes)
# ----------------------------------------------------------------------
def run_test(test, sandbox):
    """Run one test's command sequence inside its sandbox. All output
    goes to run.log. Returns (ok, elapsed)."""
    cfg = TESTS[test]
    log_path = os.path.join(sandbox, "run.log")
    if test == 1:
        cmds = test1_commands(sandbox)
    else:
        cmds = test24_commands(test, sandbox)

    t0 = time.time()
    log("%s: started (log: %s)" % (cfg["label"], log_path))
    try:
        with open(log_path, "wb") as logfh:
            for cmd, desc in cmds:
                step0 = time.time()
                log("%s: %s ..." % (cfg["label"], desc))
                proc = subprocess.Popen(cmd, cwd=sandbox,
                                        stdout=logfh, stderr=subprocess.STDOUT)
                last_note = time.time()
                while proc.poll() is None:
                    time.sleep(10)
                    if time.time() - last_note >= 600:
                        log("%s: %s still running (%s elapsed)" %
                            (cfg["label"], desc, fmt_min(time.time() - t0)))
                        last_note = time.time()
                if proc.returncode != 0:
                    log("%s: FAILED at '%s' (rc=%d); sandbox kept: %s" %
                        (cfg["label"], desc, proc.returncode, sandbox))
                    return False, time.time() - t0
                log("%s: %s done (%s)" %
                    (cfg["label"], desc, fmt_min(time.time() - step0)))
    except OSError as e:
        log("%s: could not launch process: %s" % (cfg["label"], e))
        return False, time.time() - t0
    dt = time.time() - t0
    log("%s: finished (%s)" % (cfg["label"], fmt_min(dt)))
    return True, dt


# ----------------------------------------------------------------------
# Result collection
# ----------------------------------------------------------------------
def collect_234(test, sandbox, results_root):
    """Copy staging summaries/routes and output_*.csv for Tests 2-4."""
    name = RESULT_NAMES[test]
    out = os.path.join(results_root, name)
    staging = os.path.join(sandbox, "staging")
    data = os.path.join(sandbox, "DTALite_Files")

    os.makedirs(out, exist_ok=True)
    copied = []
    for pat in ("*_summary.txt", "*_routes.csv"):
        for src in sorted(glob.glob(os.path.join(staging, pat))):
            dst = os.path.join(out, os.path.basename(src))
            shutil.copy2(src, dst)
            copied.append(os.path.basename(src))

    full = os.path.join(out, "full_output")
    os.makedirs(full, exist_ok=True)
    n_csv = 0
    for src in sorted(glob.glob(os.path.join(data, "output_*.csv"))):
        shutil.copy2(src, os.path.join(full, os.path.basename(src)))
        n_csv += 1
    log("test %d: collected %d files -> %s, %d output_*.csv -> full_output/"
        % (test, len(copied), out, n_csv))


def collect_1(sandbox, results_root):
    """Copy Test 1 bigloop result CSVs and generate proposed_summary.txt."""
    out = os.path.join(results_root, "proposed")
    full = os.path.join(out, "full_output")
    os.makedirs(full, exist_ok=True)
    data = os.path.join(sandbox, "DTALite_Files")

    n = find_last_bigloop(data)
    if n is None:
        raise RuntimeError(
            "test 1: no completed bigloop files found in %s" % data)
    copied = []
    for tag in T1_TAGS:
        for kind in T1_KINDS:
            fn = "%d%s_%s.csv" % (n, tag, kind)
            src = os.path.join(data, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(full, fn))
                copied.append(fn)
            else:
                log("test 1: warning: expected file missing: %s" % fn)
    log("test 1: collected %d bigloop-%d CSVs -> %s" % (len(copied), n, full))

    summary_path = os.path.join(out, "proposed_summary.txt")
    generate_proposed_summary(n, full, summary_path)
    log("test 1: wrote %s" % summary_path)


# ----------------------------------------------------------------------
# Test 1 summary (analysis logic carried over from runner_bingrong.py)
# ----------------------------------------------------------------------
def find_last_bigloop(d):
    """Last bigloop number with both actual and UOinfor agent files."""
    found = {}
    try:
        names = os.listdir(d)
    except OSError:
        return None
    for name in names:
        m = TAG_RE.match(name)
        if m:
            found.setdefault(int(m.group(1)), set()).add(m.group(2))
    cands = [b for b, tags in found.items()
             if "actual" in tags and "UOinfor" in tags]
    return max(cands) if cands else None


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


def generate_proposed_summary(bigloop, full_output_dir, summary_path):
    """Write proposed_summary.txt mirroring the baseline format:
    satisfaction vs Bingrong's 88.64% and vehicle-hours vs 24,719.3."""
    f_uo = os.path.join(full_output_dir, "%dUOinfor_agent.csv" % bigloop)
    f_ac = os.path.join(full_output_dir, "%dactual_agent.csv" % bigloop)
    if not os.path.isfile(f_uo):
        raise RuntimeError("missing %s" % f_uo)
    if not os.path.isfile(f_ac):
        raise RuntimeError("missing %s" % f_ac)

    actual = read_agent_paths(f_ac)
    uo = read_agent_paths(f_uo)
    n_sat, n_common, rate = satisfaction(actual, uo)

    tot_min, n_tt = sum_travel_minutes(f_ac)
    veh_hrs = tot_min / 60.0
    per_agent = tot_min / n_tt if n_tt else float("nan")

    rate_pct = 100.0 * rate
    pp = rate_pct - T1_SAT_RATE

    with open(summary_path, "w", newline="") as f:
        f.write("Proposed system (Test 1) -- v9 pipeline\n")
        f.write("bigloop: %d\n" % bigloop)
        f.write("agents: %d\n" % n_common)
        f.write("\n")
        f.write("satisfaction: %d / %d = %.2f%%\n"
                % (n_sat, n_common, rate_pct))
        f.write("Bingrong target: %.2f%% (diff %+.2f pp)\n"
                % (T1_SAT_RATE, pp))
        f.write("\n")
        f.write("vehicle-hours: %.2f\n" % veh_hrs)
        f.write("per-agent: %.2f min/agent\n" % per_agent)
        f.write("Bingrong target: %.1f veh-hrs (%.2f min/agent)\n"
                % (T1_VEH_HRS, T1_MIN_PER_AGENT))


# ----------------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------------
def print_summary(results_root, tests, outcomes):
    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for t in tests:
        name = RESULT_NAMES[t]
        ok = outcomes.get(t)
        status = "OK" if ok else "FAILED"
        log("Test %d (%s): %s" % (t, name, status))
        if t == 1 and ok:
            p = os.path.join(results_root, name, "proposed_summary.txt")
            if os.path.isfile(p):
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("satisfaction:") or \
                           line.startswith("vehicle-hours") or \
                           line.startswith("per-agent"):
                            log("  " + line)
        elif ok:
            p = os.path.join(results_root, name, "reaction_summary.txt")
            if os.path.isfile(p):
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("satisfaction:") or \
                           line.startswith("vehicle-hours") or \
                           line.startswith("per-agent"):
                            log("  " + line)
            else:
                log("  (no reaction_summary.txt)")
    log("=" * 60)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Run Bingrong Ch.4 Tests 1-4 in parallel sandboxes.")
    ap.add_argument("--tests", default="1,2,3,4",
                    help="comma-separated tests to run (default: 1,2,3,4)")
    ap.add_argument("--skip-test1", action="store_true",
                    help="skip the long v9 run (same as --tests 2,3,4)")
    ap.add_argument("--data-dir", default="DTALite_Files",
                    help="read-only DTALite_Files source (default: "
                         "DTALite_Files)")
    ap.add_argument("--results-dir", default="results",
                    help="output root (default: results)")
    ap.add_argument("--keep-temp", action="store_true",
                    help="keep temp/run_<pid>/test_<N>/ sandboxes after success")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="parallel workers (default: min(4, tests selected))")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.skip_test1:
        tests = [2, 3, 4]
    else:
        try:
            tests = [int(x.strip()) for x in args.tests.split(",")
                     if x.strip()]
        except ValueError:
            print("ERROR: --tests must be comma-separated integers "
                  "(e.g. 1,2,3,4)", file=sys.stderr)
            sys.exit(2)
    for t in tests:
        if t not in TESTS:
            print("ERROR: unknown test %d (valid: 1,2,3,4)" % t,
                  file=sys.stderr)
            sys.exit(2)
    tests = sorted(set(tests))
    if not tests:
        print("ERROR: --tests selected no tests", file=sys.stderr)
        sys.exit(2)

    data_dir = os.path.abspath(args.data_dir)
    results_root = os.path.abspath(args.results_dir)
    temp_root = os.path.abspath(os.path.join(os.getcwd(), "temp", "run_%d" % os.getpid()))
    if not os.path.isdir(data_dir):
        print("ERROR: data dir not found: %s" % data_dir, file=sys.stderr)
        sys.exit(2)
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(temp_root, exist_ok=True)

    if args.max_workers is not None and args.max_workers < 1:
        print("ERROR: --max-workers must be >= 1", file=sys.stderr)
        sys.exit(2)
    max_workers = args.max_workers or min(4, len(tests))
    log("Tests: %s (max_workers=%d)" % (tests, max_workers))
    log("Data dir: %s (read-only)" % data_dir)
    log("Results dir: %s" % results_root)
    log("Temp root: %s" % temp_root)

    # Sandbox setup -- fail fast: any setup error aborts before anything runs.
    sandboxes = {}
    for t in tests:
        try:
            sandboxes[t] = setup_sandbox(t, data_dir, temp_root)
        except Exception as e:
            print("ERROR: sandbox setup failed for test %d: %s"
                  % (t, e), file=sys.stderr)
            sys.exit(1)

    t_start = time.time()
    outcomes = {}
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as ex:
        futures = {ex.submit(run_test, t, sandboxes[t]): t for t in tests}
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                ok, dt = fut.result()
            except Exception as e:  # pragma: no cover - defensive
                log("test %d: unexpected runner error: %s" % (t, e))
                ok, dt = False, 0.0
            outcomes[t] = ok

    # Collect results for the tests that succeeded; keep sandboxes of
    # tests that failed for debugging.
    for t in tests:
        sandbox = sandboxes[t]
        if not outcomes[t]:
            print("Test %d FAILED -- sandbox kept for debugging: %s"
                  % (t, sandbox))
            continue
        try:
            if t == 1:
                collect_1(sandbox, results_root)
            else:
                collect_234(t, sandbox, results_root)
        except Exception as e:
            print("ERROR: result collection failed for test %d: %s"
                  % (t, e), file=sys.stderr)
            outcomes[t] = False
            continue
        if not args.keep_temp:
            shutil.rmtree(sandbox)
            log("test %d: sandbox removed" % t)
        else:
            log("test %d: sandbox kept (--keep-temp): %s" % (t, sandbox))

    print_summary(results_root, tests, outcomes)
    log("All done (%s total)" % fmt_min(time.time() - t_start))
    if not all(outcomes.values()):
        failed = [t for t in tests if not outcomes[t]]
        print("FAILED tests: %s" % failed, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
