#!/usr/bin/env python3
"""
wholerunner.py -- Run Bingrong Ch.4 Tests 1-6 in parallel sandboxes.

Tests at 100% MPR (Sioux Falls, 54k demand):
  Test 1 -- Proposed (proactive preference-aware guidance) ... via v9 pipeline
  Test 2 -- Real-time guidance, sigma=1 min ................. via scenarios_realtime.py
  Test 3 -- Real-time guidance, sigma=10 min ................ via scenarios_realtime.py
  Test 4 -- UE as information (MSA baseline) ................ via scenarios_ue.py
  Test 5 -- MPR sweep: Proposed bigloops 0-10 plus UE-guidance bigloops 1-10,
            then mpr_aggregate.py builds Tables 5 & 6
  Test 6 -- Future-condition volume estimation (Figure 8) ... via volume_estimation.py
            (smart mode: runs prerequisite Tests 1-4 first unless their
            results already exist; --test6-force-rerun rebuilds them anyway)

Each test/sweep gets its own sandbox under temp/run_<pid>/<name>/: a private
copy of DTALite_Files/ (the real one stays read-only) plus the scripts that
job needs. Phase-A jobs (Tests 1-4 and Test 5's proposed sweep) run
concurrently as separate processes -- no shared writable state -- and all
outputs are collected under <results-dir>/.

While the jobs run, a live progress display shows one bar per job, refreshed
every 2 seconds (main thread). Subprocess stdout is captured via PIPE, parsed
for progress markers, and also written to each sandbox's run.log. Use
--no-progress to disable the display (e.g. when redirecting output to a file).

Usage:
  python wholerunner.py --tests 1                    # Test 1 only (bigloop 10)
  python wholerunner.py --tests 1,2,3,4              # as before
  python wholerunner.py --tests 5                    # Test 5: both MPR sweeps + aggregation
  python wholerunner.py --tests 6                    # Test 6 smart mode (uses existing results if usable)
  python wholerunner.py --tests 1-6                  # everything
  python wholerunner.py --tests 1-4,6                # ranges + singles
  python wholerunner.py --tests 5 --skip-aggregation # sweeps only, aggregate later by hand
  python wholerunner.py --data-dir DTALite_Files --results-dir results

Outputs per test under <results-dir>/:
  test1/    proposed_summary.txt, full_results/ (bigloop result CSVs)
  test2/    realtime_routes.csv, realtime_summary.txt, actual_routes.csv,
            reaction_summary.txt, predicted_LinkMOE.csv, full_results/
  test3/    (same as test2)
  test4/    ue_routes.csv, ue_summary.txt, actual_routes.csv,
            reaction_summary.txt, predicted_LinkMOE.csv, full_results/
  test5/    proposed_sweep/bigloop_0..10/, ue_sweep/bigloop_1..10/,
            mpr_table5.csv, mpr_table6.csv, mpr_summary.txt
  test6/    volume_estimation.csv, volume_estimation_summary.txt
  (root)    table4_emissions.csv, table4_emissions_summary.txt
            (Tests 1-4 per-scenario emissions, via table4_emissions.py)

Sandboxes live in temp/run_<pid>/<name>/ and are deleted on success unless
--keep-temp is given. On failure the sandbox is kept and its path is
printed for debugging; other jobs keep running.

Test 5 needs the UE-guidance trio (runner_parallel_ue.py, parallel_msa_ue.py,
msa_ue.py) next to this file, plus the v9 support scripts.

Python 3.9 compatible, stdlib only.
"""

import argparse
import concurrent.futures
import csv
import glob
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings

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

# v9 pipeline support scripts (shared by Test 1 and Test 5's proposed sweep)
V9_SCRIPTS = ["runner_parallel.py", "parallel_msa.py", "msa.py",
              "assignment.py", "realtimeassignment.py",
              "traveltimecal.py", "progress.py", "choiceset.py",
              "netconfig.py"]

# Simulation-engine files staged into EVERY sandbox alongside the scripts
# above (setup_sandbox copies them for all tests, so the engine is
# available wherever a script runs -- including inside RunX/ workers,
# which resolve run_pysim.py and pysim/ from engine.py's own directory).
# These are NOT part of the per-test script lists: they are engine
# plumbing, not test logic, and every test needs them.
ENGINE_FILES = ["engine.py", "run_pysim.py"]
ENGINE_DIRS = ["pysim"]

# Test registry. "kind" tells main() how to run the entry:
#   "v9" / "scenario" -- one sandbox, one command sequence (run_test)
#   "dual-sweep"      -- two parallel sandboxes + aggregation (Test 5)
#   "analysis"        -- in-place, no sandbox (Test 6)
TESTS = {
    1: {"label": "Test 1: Proposed -- v9 pipeline",
        "kind": "v9",
        "scripts": V9_SCRIPTS},
    2: {"label": "Test 2: real-time guidance, sigma=1 min",
        "kind": "scenario",
        "scripts": ["scenarios_realtime.py", "preference_reaction.py",
                    "netconfig.py"],
        "sigma": 1, "target": TARGETS[2]},
    3: {"label": "Test 3: real-time guidance, sigma=10 min",
        "kind": "scenario",
        "scripts": ["scenarios_realtime.py", "preference_reaction.py",
                    "netconfig.py"],
        "sigma": 10, "target": TARGETS[3]},
    4: {"label": "Test 4: UE as information (MSA baseline)",
        "kind": "scenario",
        "scripts": ["scenarios_ue.py", "preference_reaction.py",
                    "netconfig.py"],
        "target": TARGETS[4]},
    5: {"label": "Test 5: MPR sweep (Proposed + UE-guidance)",
        "kind": "dual-sweep",
        "subs": {
            "P": {"label": "proposed sweep",
                  "scripts": V9_SCRIPTS, "script_dir": None,
                  "bigloops": "0-10", "n_bigloops": 11},
            "U": {"label": "UE-guidance sweep",
                  "scripts": ["runner_parallel_ue.py", "parallel_msa_ue.py",
                              "msa_ue.py", "assignment.py",
                              "realtimeassignment.py", "traveltimecal.py",
                              "progress.py", "choiceset.py", "netconfig.py"],
                  # All 9 scripts (the UE-guidance trio, the 5 v9 support
                  # scripts, and netconfig.py) live next to wholerunner.py.
                  "script_dir": None,
                  "bigloops": "1-10", "n_bigloops": 10},
        }},
    6: {"label": "Test 6: future-condition volume estimation",
        "kind": "analysis"},
}

# Results subdirectory per test
RESULT_DIR_NAMES = {1: "test1", 2: "test2", 3: "test3",
                    4: "test4", 5: "test5", 6: "test6"}

# Short labels for the progress display
DISPLAY_LABELS = {1: "Test 1 (Proposed)",
                  2: "Test 2 (RT-1min)",
                  3: "Test 3 (RT-10min)",
                  4: "Test 4 (UE)",
                  "5P": "Test 5 Proposed sweep",
                  "5U": "Test 5 UE sweep",
                  "5A": "Test 5 (aggregate)",
                  6: "Test 6 (volume est.)"}

# Bigloop result files collected for Test 1: the 9 result CSVs from the
# parity checker ({b}UOinfor/predict/actual x agent/LinkMOE/LinkTDMOE).
T1_KINDS = ("agent", "LinkMOE", "LinkTDMOE")
T1_TAGS = ("UOinfor", "predict", "actual")
TAG_RE = re.compile(r"^(\d+)(UOinfor|predict|actual)_agent\.csv$")

# Test 1: the v9 pipeline runs N bigloops in parallel (default 11),
# each doing M MSA iterations (default 18). The progress parser
# tracks total iterations completed across all bigloops (N*M).
# Test 4 MSA iteration cap (matches scenarios_ue.py --max-iters default).
T4_MAX_ITERS = 50

# LinkMOE filename suffixes (mirrored from volume_estimation.py so the
# Test 6 "results exist" check matches what it pairs up).
T6_PREDICT_SUFFIX = "predict_LinkMOE.csv"
T6_ACTUAL_SUFFIX = "actual_LinkMOE.csv"


def log(msg):
    print("[wholerunner] %s" % msg, flush=True)


def fmt_min(sec):
    return "%.1f min" % (sec / 60.0)


# ----------------------------------------------------------------------
# --tests spec parsing
# ----------------------------------------------------------------------
def parse_tests_spec(spec):
    """'1-4,6' -> [1, 2, 3, 4, 6]. Raises ValueError on bad input or
    unknown test numbers."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            try:
                lo, hi = int(lo_s.strip()), int(hi_s.strip())
            except ValueError:
                raise ValueError("invalid range %r in --tests" % part)
            if lo > hi:
                raise ValueError("invalid range %r in --tests" % part)
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(part))
            except ValueError:
                raise ValueError("invalid test %r in --tests" % part)
    tests = sorted(set(out))
    bad = [t for t in tests if t not in TESTS]
    if bad:
        raise ValueError("unknown test(s) %s (valid: 1-6)" % bad)
    if not tests:
        raise ValueError("--tests selected no tests")
    return tests


def progress_keys(selected):
    """Ordered tracker keys for the whole run. Only tests in the
    --tests selection get a bar (no phantom entries for dependencies)."""
    keys = []
    for t in selected:
        if t == 5:
            keys += ["5P", "5U", "5A"]
        else:
            keys.append(t)
    return keys


# ----------------------------------------------------------------------
# Progress tracking (thread-safe; one entry per test)
# ----------------------------------------------------------------------
class ProgressTracker:
    """Thread-safe per-test progress: fraction 0.0-1.0 plus a short
    status string. Fractions are monotonic (never move backwards)."""

    def __init__(self, tests):
        self._lock = threading.Lock()
        self._frac = {t: 0.0 for t in tests}
        self._status = {t: "starting..." for t in tests}
        self._done = {t: False for t in tests}
        self._t0 = {t: None for t in tests}

    def start(self, test):
        with self._lock:
            self._t0[test] = time.time()
            self._status[test] = "starting..."

    def update(self, test, frac, status=None):
        """frac=None keeps the current fraction (status-only update)."""
        with self._lock:
            if frac is not None and frac > self._frac[test]:
                self._frac[test] = min(frac, 1.0)
            if status is not None:
                self._status[test] = status

    def finish(self, test, ok, status=None):
        """status overrides the default "done"/"FAILED" text (e.g. "skipped")."""
        with self._lock:
            self._frac[test] = 1.0
            self._done[test] = True
            self._status[test] = (status if status is not None
                                  else ("done" if ok else "FAILED"))

    def snapshot(self):
        with self._lock:
            return {t: (self._frac[t], self._status[t],
                       self._done[t], self._t0[t])
                    for t in self._frac}


# ----------------------------------------------------------------------
# Progress-line parsers (fuzzy: each returns True if it handled the line)
# ----------------------------------------------------------------------
_BIGLOOP_RE = re.compile(r"bigloop\s+(\d+)", re.IGNORECASE)
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_ITER_RE = re.compile(r"\biter\s+(\d+)", re.IGNORECASE)
_GAP_RE = re.compile(r"gap=([0-9.eE+-]+)")
# v9 sweep launch: "Launching 11 MSA iterations in parallel ..." and the
# UE-guidance variant "Launching 10 UE-guidance MSA iterations in parallel".
_T1_LAUNCH_RE = re.compile(r"launching\s+(\d+)\s+.*?msa\s+iterations",
                           re.IGNORECASE)
# v9 sweep per-bigloop tqdm lines, e.g.
# "Bigloop 3 itr 12/18: :  60%|####    | 12/20 [05:23<03:41]".
_T1_ITR_RE = re.compile(r"bigloop\s+(\d+)\s+itr\s+(\d+)\s*/\s*(\d+)",
                        re.IGNORECASE)
# v9 sweep per-bigloop completion, e.g. "Bigloop 10 done: : 100%|...".
_T1_DONE_RE = re.compile(r"bigloop\s+(\d+)\s+done", re.IGNORECASE)


class SweepProgressState:
    """Per-sweep progress state for the v9 sweep line parser: per-bigloop
    high-water iteration marks plus the run dimensions parsed from the log.
    Only one thread ever touches a given state object, so no lock needed."""

    def __init__(self, n_bigloops=11, iters_per=18):
        self.iters = {}
        self.n_bigloops = max(1, n_bigloops)
        self.iters_per = iters_per

    def reset(self, n_bigloops):
        self.iters.clear()
        self.n_bigloops = max(1, n_bigloops)
        self.iters_per = 18


def parse_v9_sweep_line(line, tracker, key, state):
    """Generalized Test-1 line parser. `key` is the tracker key
    (1, "5P", "5U"); `state` is a SweepProgressState. Returns True if the
    line was handled."""
    m = _T1_LAUNCH_RE.search(line)
    if m:
        try:
            state.reset(int(m.group(1)))
        except ValueError:
            state.reset(11)
        tracker.update(key, None,
                       "launching %d bigloops" % state.n_bigloops)
        return True
    m = _T1_ITR_RE.search(line)
    if m:
        try:
            b = int(m.group(1))
            it = int(m.group(2))
            per = int(m.group(3))
        except ValueError:
            return False
        if per > 0:
            state.iters_per = per
        cur = state.iters.get(b, 0)
        if it > cur:
            state.iters[b] = it
            cur = it
        total = sum(state.iters.values())
        grand = state.n_bigloops * state.iters_per
        frac = (total / grand) if grand > 0 else 0.0
        tracker.update(key, frac,
                       "iter %d/%d (bigloop %d: %d/%d)"
                       % (total, grand, b, cur, state.iters_per))
        return True
    m = _T1_DONE_RE.search(line)
    if m:
        try:
            b = int(m.group(1))
        except ValueError:
            return False
        # Mark this bigloop as complete (all iterations done).
        if state.iters.get(b, 0) < state.iters_per:
            state.iters[b] = state.iters_per
        total = sum(state.iters.values())
        grand = state.n_bigloops * state.iters_per
        frac = (total / grand) if grand > 0 else 0.0
        tracker.update(key, frac,
                       "iter %d/%d (bigloop %d done)"
                       % (total, grand, b))
        return True
    # Legacy fallback: a bare "bigloop N" with no iteration info.
    # Only use if we haven't seen any itr data yet AND N is a valid
    # 0-based bigloop index. Otherwise ignore -- the itr/done patterns
    # are authoritative, and N may be a bigloop ID (e.g. 10) rather
    # than a progress count.
    m = _BIGLOOP_RE.search(line)
    if m:
        try:
            n = int(m.group(1))
        except ValueError:
            return False
        if state.iters or n >= state.n_bigloops:
            return True  # consume; don't guess progress
        tracker.update(key, n / state.n_bigloops,
                       "bigloop %d/%d" % (n, state.n_bigloops))
        return True
    m = _PCT_RE.search(line)
    if m:
        try:
            tracker.update(key, float(m.group(1)) / 100.0,
                           line.strip()[:44])
            return True
        except ValueError:
            pass
    return False


# Per-sweep parser state: Test 1 keeps its default 11-bigloop shape,
# Test 5's sweeps are initialized to theirs (the launch line corrects
# any drift when the run starts).
_T1_STATE = SweepProgressState(11)
_SWEEP_STATES = {"5P": SweepProgressState(11), "5U": SweepProgressState(10)}


def parse_test1_line(line, tracker):
    """Test 1 keeps byte-identical behavior via the shared parser."""
    return parse_v9_sweep_line(line, tracker, 1, _T1_STATE)


def _parse_reaction_line(test, line, tracker, base):
    """Shared preference_reaction.py markers. base is where the reaction
    phase starts (0.50 for tests 2/3, 0.80 for test 4); spans to 1.0."""
    low = line.lower()
    span = 1.0 - base
    if "loading network" in low:
        tracker.update(test, base + span * 0.10, "reaction: loading data")
        return True
    if "tournament run for" in low:
        tracker.update(test, base + span * 0.40, "reaction: tournament done")
        return True
    if "writing" in low and "input_agent" in low:
        tracker.update(test, base + span * 0.50, "reaction: writing input")
        return True
    if "running dtalite" in low or "running pysim" in low:
        tracker.update(test, base + span * 0.70, "reaction: engine")
        return True
    if "actual_routes" in low and "wrote" in low:
        tracker.update(test, base + span * 0.90, "reaction: routes written")
        return True
    if "reaction_summary" in low or ("wrote" in low and "summary" in low):
        tracker.update(test, 1.0, "done")
        return True
    if low.strip().startswith("satisfaction:"):
        tracker.update(test, 1.0, line.strip()[:44])
        return True
    return False


def parse_test23_line(cmd_idx, line, tracker, test):
    low = line.lower()
    if cmd_idx == 0:  # scenarios_realtime.py (guidance), spans 0.0-0.50
        if "step 1:" in low:
            if "running dtalite" in low:
                tracker.update(test, 0.20, "step 1: DTALite baseline")
            else:
                tracker.update(test, 0.10, "step 1: free-flow assignment")
            return True
        if "step 2:" in low:
            if "running dtalite" in low:
                tracker.update(test, 0.40, "step 2: DTALite reactive")
            else:
                tracker.update(test, 0.30, "step 2: reactive assignment")
            return True
        if "realtime_routes" in low and "wrote" in low:
            tracker.update(test, 0.50, "guidance complete")
            return True
        return False
    # cmd_idx == 1: preference_reaction.py, spans 0.50-1.0
    return _parse_reaction_line(test, line, tracker, 0.50)


def parse_test4_line(cmd_idx, line, tracker):
    low = line.lower()
    if cmd_idx == 0:  # scenarios_ue.py (guidance), spans 0.0-0.80
        m = _ITER_RE.search(low)
        if m:
            n = int(m.group(1))
            frac = 0.05 + (n / T4_MAX_ITERS) * 0.75
            status = "iter %d/%d" % (n, T4_MAX_ITERS)
            gm = _GAP_RE.search(low)
            if gm:
                status += " gap=%s" % gm.group(1)
            tracker.update(4, frac, status)
            return True
        if "initial gap" in low:
            tracker.update(4, 0.05, "MSA starting")
            return True
        if "converged" in low:
            tracker.update(4, 0.80, "MSA converged")
            return True
        if "ue_routes" in low and "wrote" in low:
            tracker.update(4, 0.80, "guidance complete")
            return True
        return False
    # cmd_idx == 1: preference_reaction.py, spans 0.80-1.0
    return _parse_reaction_line(4, line, tracker, 0.80)


def parse_progress_line(key, cmd_idx, line, tracker):
    """Dispatch one captured stdout line to the right parser. `key` is
    the tracker key (1..6, "5P", "5U"). Keys 6 and "5A" are status-only."""
    if tracker is None or not line or not line.strip():
        return
    try:
        if key == 1:
            parse_test1_line(line, tracker)
        elif key in ("5P", "5U"):
            parse_v9_sweep_line(line, tracker, key, _SWEEP_STATES[key])
        elif key in (2, 3):
            parse_test23_line(cmd_idx, line, tracker, key)
        elif key == 4:
            parse_test4_line(cmd_idx, line, tracker)
    except Exception:
        # Parsing must never break a run; markers are best-effort.
        pass


# ----------------------------------------------------------------------
# Progress display (main thread; redraws every 2 seconds)
# ----------------------------------------------------------------------
def _bar_chars():
    """Block characters if the console can encode them, else ASCII."""
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
    try:
        ("\u2588\u2591").encode(enc)
        return "\u2588", "\u2591"
    except (UnicodeEncodeError, LookupError):
        return "#", "-"


_FULL_CH, _EMPTY_CH = _bar_chars()


def render_bars(tracker, keys):
    snap = tracker.snapshot()
    lines = []
    for t in keys:
        frac, status, done, t0 = snap[t]
        pct = int(frac * 100)
        filled = int(frac * 20)
        bar = _FULL_CH * filled + _EMPTY_CH * (20 - filled)
        if not done and t0 is not None:
            status = "%s (%s)" % (status, fmt_min(time.time() - t0))
        lines.append("%-22s [%s] %3d%%  %s"
                     % (DISPLAY_LABELS.get(t, str(t)),
                        bar, pct, status))
    return lines


def display_loop(tracker, keys, stop_event, interval=2.0):
    """Redraw the progress block until stop_event is set."""
    n_lines = len(keys) + 1  # bars plus trailing blank line
    first = True
    try:
        while not stop_event.is_set():
            if not first:
                sys.stdout.write("\033[%dA" % n_lines)
            for line in render_bars(tracker, keys):
                sys.stdout.write("\r\033[K" + line + "\n")
            sys.stdout.write("\r\033[K\n")
            sys.stdout.flush()
            first = False
            if stop_event.wait(interval):
                break
    except Exception:
        # Display must never break a run (e.g. dumb terminal).
        pass
    # Final render so the terminal is left in a clean state.
    try:
        if not first:
            sys.stdout.write("\033[%dA" % n_lines)
        for line in render_bars(tracker, keys):
            sys.stdout.write("\r\033[K" + line + "\n")
        sys.stdout.write("\r\033[K\n")
        sys.stdout.flush()
    except Exception:
        pass


# ----------------------------------------------------------------------
# Sandbox setup (fail-fast: any error here aborts before anything runs)
# ----------------------------------------------------------------------
def setup_sandbox(name, data_dir, temp_root, scripts):
    """Create temp/run_<pid>/<name>/ with a private DTALite_Files/ copy and
    the job's scripts. `scripts` is a list of (src_dir, filename) tuples;
    src_dir None means next to wholerunner.py, otherwise a subdir of HERE.
    Raises on any error."""
    sandbox = os.path.join(temp_root, name)
    if os.path.isdir(sandbox):
        log("removing stale sandbox %s" % sandbox)
        shutil.rmtree(sandbox)
    os.makedirs(sandbox)

    sandbox_data = os.path.join(sandbox, "DTALite_Files")
    log("%s: copying DTALite_Files -> sandbox ..." % name)
    shutil.copytree(data_dir, sandbox_data)

    for src_dir, fn in scripts:
        if src_dir:
            src = os.path.join(HERE, src_dir, fn)
        else:
            src = os.path.join(HERE, fn)
        if not os.path.isfile(src):
            raise RuntimeError("required script not found: %s" % src)
        shutil.copy2(src, sandbox)
    for fn in ENGINE_FILES:
        src = os.path.join(HERE, fn)
        if not os.path.isfile(src):
            raise RuntimeError("required engine file not found: %s" % src)
        shutil.copy2(src, sandbox)
    for dirname in ENGINE_DIRS:
        src = os.path.join(HERE, dirname)
        if not os.path.isdir(src):
            raise RuntimeError("required engine dir not found: %s" % src)
        shutil.copytree(src, os.path.join(sandbox, dirname))
    log("%s: sandbox ready (%s)" % (name, sandbox))
    return sandbox


def scripts_for_test(test):
    """[(src_dir, filename)] for tests 1-4 from the TESTS registry."""
    return [(None, fn) for fn in TESTS[test]["scripts"]]


def scripts_for_sweep(sub):
    """[(src_dir, filename)] for Test 5 sub "P"/"U" from its registry entry.

    The optional "script_dirs" map overrides the sub's "script_dir" for
    individual scripts (None = next to wholerunner.py)."""
    cfg = TESTS[5]["subs"][sub]
    default_dir = cfg["script_dir"]
    per_script = cfg.get("script_dirs", {})
    return [(per_script.get(fn, default_dir), fn) for fn in cfg["scripts"]]


# ----------------------------------------------------------------------
# Per-test command lines
# ----------------------------------------------------------------------
def test1_commands(sandbox, seed, test1_mode="single", network="siouxfalls",
                    engine="pysim"):
    """Test 1 command lines.

    test1_mode="single": bigloop 10 only (100% MPR, the Test 1 case).
    test1_mode="sweep":  all 11 bigloops (the legacy MPR sweep spelling;
                         new runs should use --tests 5).
    """
    cmd = [PY, os.path.join(sandbox, "runner_parallel.py"),
           "--seed", str(seed), "--network", network, "--engine", engine]
    if test1_mode == "single":
        cmd += ["--bigloops", "10"]
        desc = "v9 pipeline (runner_parallel.py --bigloops 10)"
    else:
        cmd += ["--bigloops", "0-10"]
        desc = "v9 pipeline (runner_parallel.py --bigloops 0-10, MPR sweep)"
    return [(cmd, desc)]


def test24_commands(test, sandbox, seed, network="siouxfalls", gap_threshold=0.001,
                      engine="pysim"):
    """Guidance scenario followed by the preference-reaction step."""
    staging = os.path.join(sandbox, "staging")
    os.makedirs(staging, exist_ok=True)
    data = os.path.join(sandbox, "DTALite_Files")
    cfg = TESTS[test]

    if test in (2, 3):
        scenario = [PY, os.path.join(sandbox, "scenarios_realtime.py"),
                    "--sigma", str(cfg["sigma"]),
                    "--data-dir", data, "--result-dir", staging,
                    "--seed", str(seed), "--network", network,
                    "--engine", engine]
        scenario_desc = "real-time sigma=%d (guidance)" % cfg["sigma"]
        guidance_csv = os.path.join(staging, "realtime_routes.csv")
    else:  # test 4
        scenario = [PY, os.path.join(sandbox, "scenarios_ue.py"),
                    "--data-dir", data, "--result-dir", staging,
                    "--seed", str(seed), "--network", network,
                    "--engine", engine,
                    "--gap-threshold", str(gap_threshold)]
        scenario_desc = "UE baseline (MSA guidance)"
        guidance_csv = os.path.join(staging, "ue_routes.csv")

    reaction = [PY, os.path.join(sandbox, "preference_reaction.py"),
                "--guidance-csv", guidance_csv,
                "--data-dir", data,
                "--result-dir", staging,
                "--target-satisfaction", str(cfg["target"]),
                "--network", network, "--engine", engine]
    return [(scenario, scenario_desc),
            (reaction, "preference reaction")]


def test5_commands(sub, sandbox, seed, ue_routes_path=None, network="siouxfalls",
                   engine="pysim"):
    """One [(cmd, desc)] entry per Test 5 sweep. sub is "P" or "U"."""
    if sub == "P":
        cmd = [PY, os.path.join(sandbox, "runner_parallel.py"),
               "--bigloops", "0-10", "--seed", str(seed),
               "--network", network, "--engine", engine]
        desc = "v9 pipeline (runner_parallel.py --bigloops 0-10, proposed MPR sweep)"
    else:
        if not ue_routes_path:
            raise ValueError("ue_routes_path is required for the UE-guidance sweep")
        cmd = [PY, os.path.join(sandbox, "runner_parallel_ue.py"),
               "--ue-routes", ue_routes_path,
               "--bigloops", "1-10", "--seed", str(seed),
               "--network", network, "--engine", engine]
        desc = "UE-guidance MPR sweep (runner_parallel_ue.py --bigloops 1-10)"
    return [(cmd, desc)]


# ----------------------------------------------------------------------
# Execution (shared Popen engine; one thread per job)
# ----------------------------------------------------------------------
# Subprocess I/O strategy: the child writes DIRECTLY to run.log (a file);
# the parent polls the file for new lines. Pipes are deliberately NOT used.
# On Windows, if the child exits while one of its own descendants still
# holds the pipe open, the reader never sees EOF; worse, if the child
# itself hangs (e.g. ProcessPoolExecutor.shutdown(wait=True) waiting on a
# stuck worker), proc.poll() stays None forever, so no pipe-EOF grace
# period can ever fire. A file cannot deadlock in either case.
#
# Two timeouts guarantee forward progress:
#   _RUN_STALL_S: no new output for this long -> kill the process tree and
#                 mark the job failed. (Tests 1-5 all print tqdm/line
#                 progress every few seconds in normal operation.)
#   _RUN_ABS_S:   absolute backstop per _run_commands call.
_RUN_POLL_S = 2.0
_RUN_STALL_S = 30 * 60  # overridden by --stall-timeout-min in main()
_RUN_ABS_S = 18 * 3600


def _kill_proc_tree(proc):
    """Best-effort kill of a subprocess and its children."""
    if sys.platform == "win32":
        # /T kills the whole tree (ProcessPoolExecutor workers too).
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=30)
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def _poll_process(label, key, cmd_idx, proc, log_path, parse_line,
                tracker=None):
    """Poll log_path for new output lines and proc for exit.

    Returns (returncode, killed, kill_reason). `killed` is True when the
    process had to be killed (stall or absolute timeout); otherwise the
    process exited on its own.

    The stall detector is progress-aware: last_progress is reset only when
    the tracker's fraction for `key` actually advances. A chatty-but-stuck
    child that emits bytes without progress will still be killed.
    """
    pos = 0  # header lines ("### ...") match no progress regex; harmless
    last_out = time.monotonic()
    last_progress = time.monotonic()
    # Track the fraction to detect actual progress (not just bytes)
    def _get_frac():
        if tracker is None:
            return None
        try:
            snap = tracker.snapshot()
            if key in snap:
                return snap[key][0]
        except Exception:
            pass
        return None
    last_frac = _get_frac()
    abs_deadline = last_out + _RUN_ABS_S
    killed = False
    kill_why = ""
    rc = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as rf:
        while True:
            rf.seek(pos)
            try:
                chunk = rf.read()
            except Exception:
                chunk = ""
            if chunk:
                pos = rf.tell()
                # tqdm writes \r-separated bar updates with no \n; split
                # on both so progress markers are seen promptly.
                for piece in re.split(r"[\r\n]+", chunk):
                    if piece.strip():
                        parse_line(cmd_idx, piece)
                # Progress-aware: only reset timers if fraction advanced.
                # Bytes without progress do not reset the stall detector.
                new_frac = _get_frac()
                if new_frac is not None and last_frac is not None:
                    if new_frac > last_frac:
                        last_progress = time.monotonic()
                        last_frac = new_frac
                # If no tracker, fall back to byte-based (original behavior)
                elif new_frac is None and last_frac is None:
                    last_out = time.monotonic()
                    last_progress = time.monotonic()
            rc = proc.poll()
            if rc is not None:
                # Final drain: the child may have written between our
                # last read and its exit.
                rf.seek(pos)
                try:
                    chunk = rf.read()
                except Exception:
                    chunk = ""
                if chunk:
                    for piece in re.split(r"[\r\n]+", chunk):
                        if piece.strip():
                            parse_line(cmd_idx, piece)
                break
            now = time.monotonic()
            if now - last_progress > _RUN_STALL_S:
                kill_why = ("no progress for %s; killed stuck process"
                            % fmt_min(_RUN_STALL_S))
                log("ERROR: %s: %s" % (label, kill_why))
                _kill_proc_tree(proc)
                killed = True
                rc = proc.poll()
                break
            if now > abs_deadline:
                kill_why = ("exceeded absolute timeout %s; killed process"
                            % fmt_min(_RUN_ABS_S))
                log("ERROR: %s: %s" % (label, kill_why))
                _kill_proc_tree(proc)
                killed = True
                rc = proc.poll()
                break
            time.sleep(_RUN_POLL_S)
    if rc is None:
        # Kill did not yield a returncode; treat as failure regardless.
        rc = -1
    return rc, killed, kill_why


def _run_commands(label, sandbox, cmds, tracker, key, parse_line):
    """Run cmds sequentially in sandbox; new run.log lines are passed to
    parse_line(cmd_idx, line). Returns (ok, elapsed).

    Never hangs indefinitely: a process that produces no output for
    _RUN_STALL_S is killed, and _RUN_ABS_S bounds the whole call.
    """
    log_path = os.path.join(sandbox, "run.log")
    quiet = tracker is not None
    if tracker:
        tracker.start(key)
    t0 = time.time()
    if not quiet:
        log("%s: started (log: %s)" % (label, log_path))
    else:
        tracker.update(key, 0.0, "starting...")
    try:
        for cmd_idx, (cmd, desc) in enumerate(cmds):
            step0 = time.time()
            if not quiet:
                log("%s: %s ..." % (label, desc))
            else:
                tracker.update(key, None, desc)
            # The child writes straight to the log file. No pipe, so no
            # pipe-EOF deadlock is possible, ever.
            with open(log_path, "w", encoding="utf-8",
                       errors="replace") as logfh:
                logfh.write("### %s\n### cmd: %s\n"
                            % (desc, " ".join(cmd)))
                logfh.flush()
                try:
                    proc = subprocess.Popen(
                        cmd, cwd=sandbox,
                        stdout=logfh, stderr=subprocess.STDOUT)
                except OSError as e:
                    if tracker:
                        tracker.finish(key, False)
                    log("%s: could not launch process: %s" % (label, e))
                    return False, time.time() - t0
            # Parent's handle is closed; the child holds its own duplicate.
            # Poll the file for progress and the process for exit.
            rc, killed, kill_why = _poll_process(
                label, key, cmd_idx, proc, log_path, parse_line,
                tracker=tracker)
            if killed or rc != 0:
                if tracker:
                    tracker.finish(key, False)
                log("%s: FAILED at '%s' (rc=%s%s); sandbox kept: %s"
                    % (label, desc, rc,
                       "; %s" % kill_why if killed else "", sandbox))
                return False, time.time() - t0
            if not quiet:
                log("%s: %s done (%s)"
                    % (label, desc, fmt_min(time.time() - step0)))
        # All commands succeeded; make sure the bar reaches 100%.
        if tracker:
            tracker.finish(key, True)
    except OSError as e:
        if tracker:
            tracker.finish(key, False)
        log("%s: could not launch process: %s" % (label, e))
        return False, time.time() - t0
    dt = time.time() - t0
    if not quiet:
        log("%s: finished (%s)" % (label, fmt_min(dt)))
    return True, dt

def run_test(test, sandbox, tracker=None, seed=42, test1_mode="single",
           network="siouxfalls", gap_threshold=0.001, engine="pysim"):
    """Run one test's command sequence inside its sandbox. Returns
    (ok, elapsed)."""
    cfg = TESTS[test]
    if test == 1:
        cmds = test1_commands(sandbox, seed, test1_mode, network, engine)
    else:
        cmds = test24_commands(test, sandbox, seed, network, gap_threshold,
                               engine)
    return _run_commands(cfg["label"], sandbox, cmds, tracker, test,
                         lambda i, line: parse_progress_line(test, i, line, tracker))


def run_sweep(sub, sandbox, tracker, seed, ue_routes_path=None,
            network="siouxfalls", engine="pysim"):
    """Run one Test 5 sweep (sub "P"/"U"). Returns (ok, elapsed)."""
    key = "5" + sub
    label = "Test 5 (%s)" % TESTS[5]["subs"][sub]["label"]
    cmds = test5_commands(sub, sandbox, seed, ue_routes_path, network,
                          engine)
    return _run_commands(label, sandbox, cmds, tracker, key,
                         lambda i, line: parse_progress_line(key, i, line, tracker))


# ----------------------------------------------------------------------
# Result collection
# ----------------------------------------------------------------------
def _collect_engine_log(sandbox, out_dir):
    """Copy the sandbox's engine_calls.log (PySim audit trail) into the
    results, when one exists. Under --engine dtalite there is no such
    file and this is a no-op."""
    src = os.path.join(sandbox, "engine_calls.log")
    if os.path.isfile(src):
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(out_dir, "engine_calls.log"))


def collect_234(test, sandbox, results_root):
    """Copy staging summaries/routes, predicted_LinkMOE.csv and
    output_*.csv for Tests 2-4."""
    name = RESULT_DIR_NAMES[test]
    out = os.path.join(results_root, name)
    staging = os.path.join(sandbox, "staging")
    data = os.path.join(sandbox, "DTALite_Files")

    os.makedirs(out, exist_ok=True)
    copied = []
    # predicted_LinkMOE.csv is written into staging by scenarios_realtime.py
    # and scenarios_ue.py; it feeds Test 6.
    for pat in ("*_summary.txt", "*_routes.csv", "predicted_LinkMOE.csv"):
        for src in sorted(glob.glob(os.path.join(staging, pat))):
            dst = os.path.join(out, os.path.basename(src))
            shutil.copy2(src, dst)
            copied.append(os.path.basename(src))

    full = os.path.join(out, "full_results")
    os.makedirs(full, exist_ok=True)
    n_csv = 0
    for src in sorted(glob.glob(os.path.join(data, "output_*.csv"))):
        shutil.copy2(src, os.path.join(full, os.path.basename(src)))
        n_csv += 1
    _collect_engine_log(sandbox, out)
    log("test %d: collected %d files -> %s, %d output_*.csv -> full_results/"
        % (test, len(copied), out, n_csv))


def collect_1(sandbox, results_root):
    """Copy Test 1 bigloop result CSVs and generate proposed_summary.txt.
    In sweep mode the same "last bigloop only" behavior is kept here; the
    full per-bigloop sweep collection is collect_5's job."""
    out = os.path.join(results_root, "test1")
    full = os.path.join(out, "full_results")
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
    _collect_engine_log(sandbox, out)
    log("test 1: wrote %s" % summary_path)


def collect_5(sub, sandbox, results_root):
    """Copy per-bigloop result CSVs for a Test 5 sweep.

    sub="P": bigloops 0..10 -> results/test5/proposed_sweep/bigloop_{n}/
    sub="U": bigloops 1..10 -> results/test5/ue_sweep/bigloop_{n}/
    For each bigloop with BOTH {n}actual_agent.csv and {n}UOinfor_agent.csv
    present, copy the 9 CSVs {n}{UOinfor,predict,actual}_{agent,LinkMOE,
    LinkTDMOE}.csv. Missing/incomplete bigloops are skipped with a warning
    (mpr_aggregate.py notes them in mpr_summary.txt). Returns the list of
    bigloops actually collected.
    """
    sweep_name = "proposed_sweep" if sub == "P" else "ue_sweep"
    lo = 0 if sub == "P" else 1
    data = os.path.join(sandbox, "DTALite_Files")
    collected = []
    for n in range(lo, 11):
        f_act = os.path.join(data, "%dactual_agent.csv" % n)
        f_uo = os.path.join(data, "%dUOinfor_agent.csv" % n)
        if not (os.path.isfile(f_act) and os.path.isfile(f_uo)):
            log("test 5%s: warning: bigloop %d missing actual/UOinfor agent "
                "files -- skipped" % (sub, n))
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
                else:
                    log("test 5%s: warning: expected file missing: %s"
                        % (sub, fn))
        log("test 5%s: bigloop %d -> %s (%d files)"
            % (sub, n, out, copied))
        collected.append(n)
    log("test 5%s: collected %d/%d bigloops"
        % (sub, len(collected), 11 - lo))
    _collect_engine_log(
        sandbox, os.path.join(results_root, "test5", sweep_name))
    return collected


def run_aggregation(results_root, seed, tracker=None):
    """Run mpr_aggregate.py in-place on the collected sweeps (no sandbox).

    NOTE the flag is --input-dir (not --proposed-dir). Writes mpr_table5.csv,
    mpr_table6.csv and mpr_summary.txt into results/test5/. Returns
    (ok, elapsed). Raises RuntimeError if there is nothing to aggregate.
    """
    sweep_dir = os.path.join(results_root, "test5", "proposed_sweep")
    ue_dir = os.path.join(results_root, "test5", "ue_sweep")
    out_dir = os.path.join(results_root, "test5")
    os.makedirs(out_dir, exist_ok=True)
    if not glob.glob(os.path.join(sweep_dir, "bigloop_*")):
        raise RuntimeError(
            "Test 5: nothing to aggregate (no bigloop dirs in %s)"
            % sweep_dir)
    cmd = [PY, os.path.join(HERE, "mpr_aggregate.py"),
           "--input-dir", sweep_dir,
           "--ue-dir", ue_dir,
           "--seed", str(seed),
           "--output-dir", out_dir]
    if tracker:
        tracker.start("5A")
        tracker.update("5A", None, "aggregating...")
    log("Test 5: aggregating MPR sweep (mpr_aggregate.py) ...")
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, timeout=4*3600).returncode
    except subprocess.TimeoutExpired:
        log("ERROR: mpr_aggregate.py timed out after 4h; marking aggregation failed")
        if tracker:
            tracker.finish("5A", False)
        return False, time.time() - t0
    dt = time.time() - t0
    ok = rc == 0
    if tracker:
        tracker.finish("5A", ok)
    log("Test 5: aggregation %s (%s)" % ("done" if ok else "FAILED",
                                        fmt_min(dt)))
    return ok, dt


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
# Test 5 ue_routes dependency
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Test 6: future-condition volume estimation (smart mode)
# ----------------------------------------------------------------------
def _proposed_pair_exists(d):
    """True if some filename prefix in d has BOTH *predict_LinkMOE.csv and
    *actual_LinkMOE.csv. Mirrors volume_estimation.find_proposed_pair."""
    preds = set()
    for path in glob.glob(os.path.join(d, "*" + T6_PREDICT_SUFFIX)):
        preds.add(os.path.basename(path)[:-len(T6_PREDICT_SUFFIX)])
    acts = set()
    for path in glob.glob(os.path.join(d, "*" + T6_ACTUAL_SUFFIX)):
        acts.add(os.path.basename(path)[:-len(T6_ACTUAL_SUFFIX)])
    return bool(preds & acts)


def _scenario_actual_path(t, results_root):
    """First existing actual-LinkMOE candidate for Test t (2-4), or None.

    Mirrors volume_estimation.SCENARIO_ACTUAL_CANDIDATES: collect_234()
    copies the reaction run's output_LinkMOE.csv into
    testN/full_results/; the bare testN/output_LinkMOE.csv fallback
    covers hand-assembled results dirs."""
    test_dir = os.path.join(results_root, "test%d" % t)
    for cand in ("full_results/output_LinkMOE.csv", "output_LinkMOE.csv"):
        p = os.path.join(test_dir, cand)
        if os.path.isfile(p):
            return p
    return None


def results_exist(t, results_root):
    """True if Test t's results are present and usable by
    volume_estimation. Test 1 needs a predict/actual LinkMOE pair (the
    hard requirement for Figure 8); tests 2-4 each need BOTH
    predicted_LinkMOE.csv AND the reaction run's actual LinkMOE file.
    A test with only the predicted file counts as missing here, so
    Test 6 smart mode reruns it as a prerequisite instead of passing a
    half-usable system to volume_estimation (which would skip it)."""
    if t == 1:
        d = os.path.join(results_root, "test1", "full_results")
        if not os.path.isdir(d):
            return False
        return _proposed_pair_exists(d)
    if t in (2, 3, 4):
        return (os.path.isfile(os.path.join(
                    results_root, "test%d" % t, "predicted_LinkMOE.csv"))
                and _scenario_actual_path(t, results_root) is not None)
    return False


def run_test6(results_root, tracker=None):
    """Run volume_estimation.py in-place (no sandbox; read-only analysis).
    Analyzes whichever of Tests 1-4 have complete predicted/actual
    pairs (see results_exist). Returns (ok, elapsed). Raises
    RuntimeError if none do."""
    systems = ["test%d" % t for t in (1, 2, 3, 4)
               if results_exist(t, results_root)]
    if not systems:
        raise RuntimeError("Test 6: no usable results for tests 1-4 under %s"
                           % results_root)
    cmd = [PY, os.path.join(HERE, "volume_estimation.py"),
           "--results-dir", results_root,
           "--output-dir", os.path.join(results_root, "test6"),
           "--systems", ",".join(systems)]
    if tracker:
        tracker.start(6)
        tracker.update(6, None, "running volume estimation...")
    log("Test 6: estimating future-condition volumes (%s) ..."
        % ",".join(systems))
    t0 = time.time()
    # Fast and deterministic; no progress markers to parse, so inherit
    # the console instead of capturing.
    try:
        rc = subprocess.run(cmd, timeout=4*3600).returncode
    except subprocess.TimeoutExpired:
        log("ERROR: volume_estimation.py (Test 6) timed out after 4h")
        if tracker:
            tracker.finish(6, False)
        return False, time.time() - t0
    dt = time.time() - t0
    ok = rc == 0
    if tracker:
        tracker.finish(6, ok)
    log("Test 6: %s (%s)" % ("done" if ok else "FAILED", fmt_min(dt)))
    return ok, dt


# ----------------------------------------------------------------------
# Run manifest (big-run package, Worker 4)
# ----------------------------------------------------------------------
# results/manifest.txt: methods setup facts for the paper, read at
# runtime from the actual input files -- never hardcoded Sioux Falls
# numbers, so the same code serves Baltimore (--network baltimore).
# Written once at run start (status: started) and rewritten at run end
# (status: completed/failed) with outcomes. A missing or unreadable
# input file is recorded as "unknown", never fatal: the manifest must
# not kill a run. Engine-independent: it only records args.engine, it
# never branches on it, so it works under --engine dtalite and pysim.
# NOTE: the file lives at the results ROOT (results/manifest.txt), not
# inside a per-test dir, so the auto-clear of per-test dirs (which runs
# before the start-manifest write) can never delete it.
MANIFEST_NAME = "manifest.txt"
_MANIFEST_UNKNOWN = "unknown"


def _manifest_ts(t):
    if t is None:
        return _MANIFEST_UNKNOWN
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(t))
    except Exception:
        return _MANIFEST_UNKNOWN


def _manifest_csv_rows(path):
    """Data-row count of a CSV file, or None if it cannot be read."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return 0
        return max(0, len(rows) - 1)
    except Exception:
        return None


def _manifest_agent_stats(data_dir):
    """(agents, dep_min, dep_max, od_cells) from input_agent_initial.csv.

    Any element is None when the file is missing or unreadable. OD
    cells are distinct (from_zone_id, to_zone_id) pairs in the agent
    file, i.e. the ODs actually travelled, as opposed to the demand
    matrix below (which may list zero-demand cells as well).
    """
    path = os.path.join(data_dir, "input_agent_initial.csv")
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        cols = {}
        for i, name in enumerate(header):
            cols[name.strip()] = i
        need = ("from_zone_id", "to_zone_id", "departure_time_in_min")
        if any(c not in cols for c in need):
            return None, None, None, None
        agents = 0
        dep_min = None
        dep_max = None
        ods = set()
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) <= max(cols[c] for c in need):
                    continue
                agents += 1
                try:
                    dep = float(row[cols["departure_time_in_min"]])
                except (ValueError, TypeError):
                    dep = None
                if dep is not None:
                    dep_min = dep if dep_min is None else min(dep_min, dep)
                    dep_max = dep if dep_max is None else max(dep_max, dep)
                ods.add((row[cols["from_zone_id"]].strip(),
                         row[cols["to_zone_id"]].strip()))
        return agents, dep_min, dep_max, len(ods)
    except Exception:
        return None, None, None, None


def _manifest_demand_od_cells(data_dir):
    """Nonzero cells of input_demand.csv (OD matrix), or None."""
    path = os.path.join(data_dir, "input_demand.csv")
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return None
        count = 0
        for row in rows[1:]:
            for val in row[1:]:
                try:
                    if float(val) != 0:
                        count += 1
                except (ValueError, TypeError):
                    continue
        return count
    except Exception:
        return None


def _manifest_choice_set(data_dir):
    """(source_file, od_cells, total_routes) for the choice set.

    Baltimore path: choice_sets.pkl ({(oz, dz): [paths]}), counted
    directly. Sioux Falls path: a MATLAB .mat file, counted via
    scipy when available (finallist cells); without scipy the file is
    still named but the counts stay unknown rather than guessed.
    """
    pkl = os.path.join(data_dir, "choice_sets.pkl")
    if os.path.isfile(pkl):
        try:
            import pickle
            with open(pkl, "rb") as f:
                choice_dict = pickle.load(f)
            return ("choice_sets.pkl", len(choice_dict),
                    sum(len(v) for v in choice_dict.values()))
        except Exception:
            return ("choice_sets.pkl", None, None)
    mats = sorted(glob.glob(os.path.join(data_dir, "choice*.mat")))
    if mats:
        src = os.path.basename(mats[0])
        try:
            from scipy.io import loadmat
            mat = loadmat(mats[0])
            final = mat.get("finallist")
            if final is None:
                return (src, None, None)
            # The pipeline transposes finallist (msa.py: choiceset =
            # mat['finallist'].T), so OD cells are shape[1] after that
            # transpose (576 ODs x up to 10 routes), not shape[0].
            ods = final.shape[1] if len(final.shape) == 2 else final.shape[0]
            routes = 0
            for cell in final.flat:
                try:
                    if cell is not None and len(cell):
                        routes += 1
                except TypeError:
                    routes += 1
            return (src, ods, routes)
        except Exception:
            return (src, None, None)
    return (None, None, None)


def _manifest_test6_systems(results_root):
    """Tests 1-4 with a results dir on disk (Test 6's actual systems)."""
    return ["test%d" % t for t in (1, 2, 3, 4)
            if os.path.isdir(os.path.join(results_root,
                                          RESULT_DIR_NAMES[t]))]


def _manifest_val(value):
    return _MANIFEST_UNKNOWN if value is None else str(value)


def write_run_manifest(results_root, args, tests, data_dir, status,
                       outcomes=None, sweep_outcomes=None,
                       agg_status=None, started=None, ended=None,
                       cmdline=None):
    """Write (or rewrite) results/manifest.txt. Never raises."""
    try:
        agents, dep_min, dep_max, agent_ods = _manifest_agent_stats(data_dir)
        demand_ods = _manifest_demand_od_cells(data_dir)
        links = _manifest_csv_rows(os.path.join(data_dir, "input_link.csv"))
        nodes = _manifest_csv_rows(os.path.join(data_dir, "input_node.csv"))
        choice_src, choice_ods, choice_routes = _manifest_choice_set(data_dir)
        lines = []
        lines.append("Run manifest -- Bingrong Ch.4 Tests 1-6 (wholerunner.py)")
        lines.append("status: %s" % status)
        lines.append("timestamp_start: %s" % _manifest_ts(started))
        lines.append("timestamp_end: %s" % _manifest_ts(ended))
        lines.append("command_line: %s" % (cmdline or _MANIFEST_UNKNOWN))
        lines.append("network: %s" % args.network)
        lines.append("data_dir: %s" % data_dir)
        lines.append("results_dir: %s" % results_root)
        lines.append("engine: %s" % args.engine)
        lines.append("seed: %s" % args.seed)
        lines.append("")
        lines.append("Demand and network (read from the input files):")
        lines.append("agents: %s" % _manifest_val(agents))
        lines.append("od_cells_agent_file: %s" % _manifest_val(agent_ods))
        lines.append("od_cells_demand_file: %s" % _manifest_val(demand_ods))
        lines.append("choice_set_file: %s" % _manifest_val(choice_src))
        lines.append("choice_set_od_cells: %s" % _manifest_val(choice_ods))
        lines.append("choice_set_routes_total: %s"
                     % _manifest_val(choice_routes))
        lines.append("links: %s" % _manifest_val(links))
        lines.append("nodes: %s" % _manifest_val(nodes))
        if dep_min is not None and dep_max is not None:
            lines.append("departure_window_min: %s (minutes)" % dep_min)
            lines.append("departure_window_max: %s (minutes)" % dep_max)
        else:
            lines.append("departure_window_min: %s" % _MANIFEST_UNKNOWN)
            lines.append("departure_window_max: %s" % _MANIFEST_UNKNOWN)
        lines.append("")
        lines.append("Methods setup:")
        lines.append("standardization: meanstd28 "
                     "(weights sheet cols A:E, rows 2-3; "
                     "UE standardization per msa.m / msa.py)")
        lines.append("preference_weights_predict: estimated weights, "
                     "cols A:F of the weights sheet")
        lines.append("preference_weights_reaction: true weights, "
                     "cols J:O of the weights sheet")
        lines.append("ue_gap_threshold: %s" % args.gap_threshold)
        lines.append("ue_max_iters: %s" % T4_MAX_ITERS)
        lines.append("")
        lines.append("Per-test configuration (selected tests only):")
        lines.append("tests_selected: %s"
                     % ", ".join(str(t) for t in tests))
        if 1 in tests:
            lines.append("test1_bigloops: %s"
                         % ("10 only (single mode)"
                            if args.test1_mode == "single"
                            else "0-10 (sweep mode)"))
        if 2 in tests:
            lines.append("test2_sigma_min: %s" % TESTS[2]["sigma"])
        if 3 in tests:
            lines.append("test3_sigma_min: %s" % TESTS[3]["sigma"])
        if 4 in tests:
            lines.append("test4_max_iters: %s" % T4_MAX_ITERS)
            lines.append("test4_gap_threshold: %s" % args.gap_threshold)
        if 5 in tests:
            lines.append("test5_proposed_bigloops: %s"
                         % TESTS[5]["subs"]["P"]["bigloops"])
            lines.append("test5_ue_bigloops: %s"
                         % TESTS[5]["subs"]["U"]["bigloops"])
        if 6 in tests:
            if status == "started":
                lines.append("test6_systems: test1,test2,test3,test4 "
                             "(planned; actual systems are those with "
                             "results dirs when Test 6 runs)")
            else:
                lines.append("test6_systems: %s"
                             % (",".join(_manifest_test6_systems(results_root))
                                or _MANIFEST_UNKNOWN))
        if outcomes is not None:
            lines.append("")
            lines.append("Outcomes:")
            for t in tests:
                if t == 5:
                    lines.append("test5_proposed_sweep: %s"
                                 % ("OK" if sweep_outcomes.get("P")
                                    else "FAILED"))
                    lines.append("test5_ue_sweep: %s"
                                 % ("OK" if sweep_outcomes.get("U")
                                    else "FAILED"))
                    lines.append("test5_aggregation: %s"
                                 % (agg_status or _MANIFEST_UNKNOWN))
                else:
                    lines.append("test%d: %s"
                                 % (t, "OK" if outcomes.get(t) else "FAILED"))
        path = os.path.join(results_root, MANIFEST_NAME)
        with open(path, "w", newline="") as f:
            f.write("\r\n".join(lines) + "\r\n")
        log("Run manifest: %s" % path)
    except Exception as e:
        log("WARNING: could not write run manifest: %s" % e)


# ----------------------------------------------------------------------
# Table 4 emissions (Tests 1-4)
# ----------------------------------------------------------------------
def run_table4_emissions(results_root):
    """Compute Table 4 emissions from the collected Tests 1-4 results.

    Thin wrapper over table4_emissions.py (same directory): writes
    table4_emissions.csv + table4_emissions_summary.txt into
    results_root and appends a Table 4 block to each available per-test
    summary file. Returns True if at least one scenario row was
    written, False otherwise. Never raises for missing data -- a
    partial run writes the rows it can and notes the rest -- but an
    import failure or an output-write error propagates to the caller,
    which logs it as a warning so the overall run is never failed by
    this step alone. Deliberately independent of run_test6() and of
    any manifest/threshold/auto-clear logic.
    """
    try:
        import table4_emissions
    except ImportError:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import table4_emissions
    rows, notes = table4_emissions.build_table4_emissions(results_root)
    for note in notes:
        log("Table 4: %s" % note)
    if rows:
        log("Table 4: wrote table4_emissions.csv (%d scenario rows)"
            % len(rows))
        return True
    log("Table 4: WARNING: no scenario rows written "
        "(no collected agent files found for Tests 1-4)")
    return False


# ----------------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------------
def print_summary(results_root, tests, outcomes, sweep_outcomes, agg_status):
    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for t in tests:
        if t == 5:
            p5 = "OK" if sweep_outcomes.get("P") else "FAILED"
            u5 = "OK" if sweep_outcomes.get("U") else "FAILED"
            log("Test 5 (MPR sweep): proposed=%s ue=%s aggregation=%s"
                % (p5, u5, agg_status))
            for fn in ("mpr_table5.csv", "mpr_table6.csv", "mpr_summary.txt"):
                p = os.path.join(results_root, "test5", fn)
                log("  %s: %s" % (fn, "present" if os.path.isfile(p)
                                  else "MISSING"))
            continue
        if t == 6:
            ok = outcomes.get(6)
            log("Test 6 (volume est.): %s" % ("OK" if ok else "FAILED"))
            p = os.path.join(results_root, "test6",
                             "volume_estimation_summary.txt")
            log("  volume_estimation_summary.txt: %s"
                % ("present" if os.path.isfile(p) else "MISSING"))
            continue
        name = RESULT_DIR_NAMES[t]
        ok = outcomes.get(t)
        log("Test %d (%s): %s" % (t, name, "OK" if ok else "FAILED"))
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
    t4 = os.path.join(results_root, "table4_emissions.csv")
    log("Table 4 emissions: %s"
        % ("present" if os.path.isfile(t4) else "MISSING"))
    log("=" * 60)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def auto_clear_results(results_root, tests):
    """Delete and recreate the results subdirectories for the tests in
    `tests` -- per-test scope only, never the results root itself.

    Why: the collect_* functions only overwrite files, never delete
    them, so a stale file from a partial/failed run would otherwise
    survive into the new run's tables (the stale Test 5 bigloop_1
    class of bug). Clearing before a test runs makes every run start
    from an empty per-test dir.

    Scope rules (enforced by main()'s call site, after the Test 6
    prereq fold):
      * Only tests in `tests` are cleared. On a bare `--tests 6` run
        that means test6/ only: Tests 1-4 results are Test 6's inputs
        and must be left alone.
      * A `--tests 6` run that folds in a missing prerequisite clears
        that prerequisite's dir too, since it is about to be rerun;
        usable prerequisites that were NOT folded in are untouched.
      * A Test 4 run launched only as a Test 5 dependency (4 not in
        `tests`) is NOT cleared: its existing ue_routes.csv is the
        input the 5U sweep reuses.
    A removal failure is fatal (sys.exit(1)): proceeding with a
    half-cleared dir would silently mix stale and fresh results, which
    is exactly what auto-clear exists to prevent. The usual cause on
    Windows is a locked file (stale python.exe / open file); unlock
    and rerun. Engine-agnostic: this only touches result dirs, never
    engine outputs in sandboxes.
    """
    for t in sorted(set(tests)):
        name = RESULT_DIR_NAMES.get(t)
        if name is None:
            continue
        out = os.path.join(results_root, name)
        # Safety: delete only a direct child of the results root, and
        # never the root itself, whatever RESULT_DIR_NAMES holds.
        if (os.path.abspath(out) == os.path.abspath(results_root)
                or os.path.dirname(os.path.abspath(out))
                != os.path.abspath(results_root)):
            continue
        if os.path.isdir(out):
            try:
                shutil.rmtree(out)
            except OSError as e:
                print("ERROR: Auto-clear failed to remove %s: %s"
                      % (out, e), file=sys.stderr)
                sys.exit(1)
            log("Auto-clear: removed %s" % out)
        elif os.path.lexists(out):
            try:
                os.remove(out)
            except OSError as e:
                print("ERROR: Auto-clear failed to remove %s: %s"
                      % (out, e), file=sys.stderr)
                sys.exit(1)
            log("Auto-clear: removed file %s" % out)
        else:
            log("Auto-clear: %s did not exist, nothing to remove" % out)
        os.makedirs(out, exist_ok=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Run Bingrong Ch.4 Tests 1-6 in parallel sandboxes.")
    ap.add_argument("--tests", default=None,
                    help="tests to run: comma-separated ints and ranges, e.g. "
                         "1 | 1,2,3 | 1-4 | 1,5,6 | 1-4,6 "
                         "(default: 1,2,3,4)")
    ap.add_argument("--skip-test1", action="store_true",
                    help="DEPRECATED: same as --tests 2,3,4")
    ap.add_argument("--data-dir", default="DTALite_Files",
                    help="read-only DTALite_Files source (default: "
                         "DTALite_Files)")
    ap.add_argument("--gap-threshold", type=float, default=0.001,
                    help="UE relative gap convergence threshold (Test 4; "
                         "the Test 5 UE sweep reuses Test 4's ue_routes.csv "
                         "and so inherits this threshold; default: 0.001)")
    ap.add_argument("--network", default="siouxfalls",
                    choices=["siouxfalls", "baltimore"],
                    help="network: siouxfalls (default) or baltimore "
                         "(use with --data-dir baltimore_files)")
    ap.add_argument("--engine", default="pysim",
                    choices=["pysim", "dtalite"],
                    help="simulation engine: pysim (default) or dtalite "
                         "(dtalite runs the DTALite executable exactly as "
                         "before; pysim runs the Python port)")
    ap.add_argument("--results-dir", default="results",
                    help="output root (default: results)")
    ap.add_argument("--keep-temp", action="store_true",
                    help="keep temp/run_<pid>/ sandboxes after success")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="parallel workers (default: min(4, phase-A jobs))")
    ap.add_argument("--stall-timeout-min", type=float, default=30,
                    help="kill a test if no progress for this many minutes "
                         "(default: 30; 0 disables)")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable the live progress display")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for all tests (Tests 2-4 and the v9 "
                         "pipeline; default: 42)")
    ap.add_argument("--test1-mode", choices=["single", "sweep"],
                    default="single",
                    help="Test 1 bigloops: single = bigloop 10 only "
                         "(100pct MPR, Test 1); sweep = all 11 bigloops "
                         "(legacy MPR sweep spelling -- new runs should use "
                         "--tests 5). Default: single.")
    ap.add_argument("--test6-force-rerun", action="store_true",
                    help="Test 6: ignore existing results/testN/ outputs and "
                         "rerun prerequisite Tests 1-4 before estimating")
    ap.add_argument("--skip-aggregation", action="store_true",
                    help="Test 5: run both MPR sweeps but skip the "
                         "mpr_aggregate.py step")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    global _RUN_STALL_S
    if args.stall_timeout_min <= 0:
        _RUN_STALL_S = float('inf')
    else:
        _RUN_STALL_S = args.stall_timeout_min * 60

    # Couple --network to data/results dirs: --network baltimore implies
    # --data-dir baltimore_files and --results-dir results_baltimore,
    # unless the user explicitly overrode them.
    if args.network == "baltimore":
        import sys as _sys
        _argv = _sys.argv[1:] if argv is None else argv
        _argv_str = " ".join(_argv) if _argv else ""
        if "--data-dir" not in _argv_str and args.data_dir == "DTALite_Files":
            args.data_dir = "baltimore_files"
        if "--results-dir" not in _argv_str and args.results_dir == "results":
            args.results_dir = "results_baltimore"

    if args.skip_test1:
        warnings.warn("--skip-test1 is deprecated; use --tests 2,3,4",
                      DeprecationWarning, stacklevel=2)
        log("WARNING: --skip-test1 is deprecated; use --tests 2,3,4")
        if args.tests is not None:
            print("ERROR: --skip-test1 cannot be combined with --tests",
                  file=sys.stderr)
            sys.exit(2)
        tests = [2, 3, 4]
    else:
        try:
            tests = parse_tests_spec(
                args.tests if args.tests is not None else "1,2,3,4")
        except ValueError as e:
            print("ERROR: %s" % e, file=sys.stderr)
            sys.exit(2)
    tests = sorted(set(tests))

    data_dir = os.path.abspath(args.data_dir)
    results_root = os.path.abspath(args.results_dir)
    temp_root = os.path.abspath(os.path.join(os.getcwd(), "temp", "run_%d" % os.getpid()))
    if not os.path.isdir(data_dir):
        print("ERROR: data dir not found: %s" % data_dir, file=sys.stderr)
        sys.exit(2)
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(temp_root, exist_ok=True)

    # Test 6 smart mode: fold missing prerequisites into the selection
    # BEFORE setup, so they run in Phase A like any other selected test.
    if 6 in tests:
        prereqs = [t for t in (1, 2, 3, 4)
                   if t not in tests
                   and not (results_exist(t, results_root)
                            and not args.test6_force_rerun)]
        if prereqs:
            log("Test 6: no usable results for test(s) %s -- running them "
                "as prerequisites" % prereqs)
            tests = sorted(set(tests) | set(prereqs))

    # Auto-clear: delete + recreate ONLY the results dirs of the tests
    # about to run, so stale files from a partial/failed run can never
    # survive into this run's tables. Placed AFTER the Test 6 prereq
    # fold on purpose: clearing before the fold would make
    # results_exist() false for every test and turn a bare --tests 6
    # analysis run into a full rerun. Tests 1-4 are therefore cleared
    # only when they are in `tests` (selected, or folded in as a
    # missing prerequisite); a bare --tests 6 run clears test6/ only.
    # Test 4 as a Test 5-only dependency (4 not in `tests`) is not
    # cleared: its existing ue_routes.csv is the 5U sweep's input.
    auto_clear_results(results_root, tests)

    # Run manifest (start): rewritten at run end with outcomes.
    # Placed AFTER auto-clear on purpose: the manifest lives at the
    # results root, which auto-clear never touches, and the start
    # write must not precede the Test 6 prereq fold above (so the
    # selected-tests list it records is final).
    _run_started = time.time()
    _run_cmdline = (" ".join(sys.argv) if argv is None
                    else " ".join([sys.argv[0]] + list(argv)))
    write_run_manifest(results_root, args, tests, data_dir, "started",
                       None, None, None, _run_started, None, _run_cmdline)

    # Test 5 UE sweep needs its 8 scripts next to this file (the
    # UE-guidance trio plus the 5 v9 support scripts). Fail fast if
    # any are missing. Resolve through scripts_for_sweep so this
    # check can never drift from the sandbox spec.
    if 5 in tests:
        missing = []
        for src_dir, fn in scripts_for_sweep("U"):
            src = (os.path.join(HERE, src_dir, fn) if src_dir
                   else os.path.join(HERE, fn))
            if not os.path.isfile(src):
                missing.append(src)
        if missing:
            print("ERROR: Test 5 UE sweep is missing required script(s): %s"
                  % ", ".join(missing), file=sys.stderr)
            sys.exit(1)

    # If Test 5 is selected without Test 4, resolve ue_routes.csv BEFORE
    # Phase A so the 5U sweep can start immediately in Phase A instead of
    # waiting. (When Test 4 IS selected, 5U starts from inside the Phase A
    # loop as soon as Test 4 is collected.)
    ue_routes_pre = None
    if 5 in tests and 4 not in tests:
        _pre_target = os.path.join(results_root, "test4", "ue_routes.csv")
        if os.path.isfile(_pre_target):
            log("Test 5: using existing results/test4/ue_routes.csv")
            ue_routes_pre = _pre_target
        else:
            log("Test 5: no usable ue_routes.csv -- running Test 4 as a "
                "dependency before Phase A ...")
            try:
                _sb_dep4 = setup_sandbox("test_dep4", data_dir, temp_root,
                                         scripts_for_test(4))
            except Exception as e:
                print("ERROR: sandbox setup failed for Test 4 dependency: %s"
                      % e, file=sys.stderr)
                ue_routes_pre = None
            else:
                _ok_dep4, _dt_dep4 = run_test(4, _sb_dep4, None, args.seed,
                                              "single", args.network,
                                              args.gap_threshold, args.engine)
                if _ok_dep4:
                    try:
                        collect_234(4, _sb_dep4, results_root)
                    except Exception as e:
                        print("ERROR: dependency Test 4 collection failed: %s"
                              % e, file=sys.stderr)
                        _ok_dep4 = False
                if not args.keep_temp:
                    try:
                        shutil.rmtree(_sb_dep4)
                    except Exception:
                        pass
                if _ok_dep4 and os.path.isfile(_pre_target):
                    log("Test 5: dependency Test 4 produced ue_routes.csv")
                    ue_routes_pre = _pre_target
                else:
                    print("ERROR: Test 5 UE sweep needs ue_routes.csv: "
                          "dependency Test 4 failed and no existing "
                          "results/test4/ue_routes.csv", file=sys.stderr)
                    ue_routes_pre = None

    if args.max_workers is not None and args.max_workers < 1:
        print("ERROR: --max-workers must be >= 1", file=sys.stderr)
        sys.exit(2)

    keys = progress_keys(tests)

    # Live progress display unless disabled or stdout is not a terminal.
    use_progress = (not args.no_progress) and sys.stdout.isatty()
    tracker = ProgressTracker(keys) if use_progress else None
    stop_event = threading.Event()
    display_thread = None

    log("Engine: %s" % args.engine)
    log("Tests: %s (keys: %s)" % (tests, keys))
    log("Seed: %d (all tests)" % args.seed)
    if 1 in tests:
        log("Test 1 mode: %s (%s)" % (
            args.test1_mode,
            "bigloop 10 only" if args.test1_mode == "single"
            else "all 11 bigloops (MPR sweep; consider --tests 5)"))
    log("Data dir: %s (read-only)" % data_dir)
    log("Results dir: %s" % results_root)
    log("Temp root: %s" % temp_root)
    if use_progress:
        log("Live progress display enabled (refresh every 2s)")
        # Print a blank block so the first redraw has room to move up.
        display_thread = threading.Thread(
            target=display_loop, args=(tracker, keys, stop_event),
            daemon=True)
        display_thread.start()
    elif not args.no_progress:
        log("Progress display disabled (stdout is not a terminal)")

    # ---- Phase A setup (fail-fast) ----
    sandboxes = {}
    try:
        for t in tests:
            if t in (1, 2, 3, 4):
                sandboxes[t] = setup_sandbox(
                    "test_%d" % t, data_dir, temp_root, scripts_for_test(t))
        if 5 in tests:
            sandboxes["5P"] = setup_sandbox(
                "test_5P", data_dir, temp_root, scripts_for_sweep("P"))
            # Set up 5U sandbox here too (not inside the executor) so a
            # startup failure is caught by the fail-fast handler above.
            # The 5U sweep itself is submitted in the executor below.
            if 4 not in tests and ue_routes_pre and os.path.isfile(ue_routes_pre):
                _sb5u_pre = setup_sandbox(
                    "test_5U", data_dir, temp_root, scripts_for_sweep("U"))
                shutil.copy2(ue_routes_pre,
                             os.path.join(_sb5u_pre, "ue_routes.csv"))
                sandboxes["5U"] = _sb5u_pre
                log("Test 5: 5U sandbox ready (ue_routes.csv pre-resolved)")
    except Exception as e:
        print("ERROR: sandbox setup failed: %s" % e, file=sys.stderr)
        if display_thread:
            stop_event.set()
            display_thread.join(timeout=5)
        sys.exit(1)

    # ---- Phase A run (Tests 1-4 plus the proposed sweep, in parallel) ----
    # The proposed sweep (5P) is independent of everything, so it starts
    # in Phase A rather than being serialized behind Tests 1-4.
    t_start = time.time()
    outcomes = {t: False for t in tests}
    sweep_outcomes = {"P": False, "U": False}
    agg_status = "not-run"
    n_jobs = len(sandboxes)
    max_workers = args.max_workers or min(4, max(1, n_jobs))

    if n_jobs:
        # Track whether Test 6 has been triggered (it runs as soon as
        # Tests 1-4 are all collected, without waiting for the 5P sweep).
        test6_done = False
        tests_14 = [t for t in tests if t in (1, 2, 3, 4)]

        # The 5U sweep starts as soon as Test 4's ue_routes.csv is
        # available (right after Test 4 is collected), instead of waiting
        # for the whole of Phase A (including the 5P sweep).
        # sweep_5u_started prevents double-start.
        sweep_5u_started = False
        test4_completed = False

        # Single futures dict for ALL Phase A jobs. 5U goes here too
        # (no separate fut_5u_box indirection) so the polling loop below
        # handles every job uniformly. Values are ("test", t) or
        # ("sweep", "5P"/"5U").
        futures = {}

        def _maybe_start_5u():
            # Start the 5U sweep if its ue_routes.csv prerequisite is met.
            # Called after each Phase A job completes. When Test 4 is
            # selected, waits for this run's Test 4 to succeed+collect.
            # When Test 4 is not selected, uses the pre-resolved path
            # (ue_routes_pre) immediately. Submits directly into the
            # shared futures dict (no fut_5u_box).
            nonlocal sweep_5u_started, test4_completed
            if sweep_5u_started or 5 not in tests:
                return
            ue_routes = None
            if 4 in tests:
                if not test4_completed:
                    return  # Test 4 hasn't finished yet
                if not outcomes.get(4):
                    # Test 4 failed -- 5U cannot run; fail it loudly.
                    log("Test 5: Test 4 failed -- UE sweep (5U) cannot run")
                    sweep_outcomes["U"] = False
                    sweep_5u_started = True
                    if tracker:
                        tracker.finish("5U", False)
                    return
                candidate = os.path.join(results_root, "test4",
                                         "ue_routes.csv")
                if os.path.isfile(candidate):
                    ue_routes = candidate
                    log("Test 5: using ue_routes.csv from this run's Test 4")
                else:
                    log("Test 5: Test 4 succeeded but %s is missing -- "
                        "UE sweep (5U) cannot run" % candidate)
                    sweep_outcomes["U"] = False
                    sweep_5u_started = True
                    if tracker:
                        tracker.finish("5U", False)
                    return
            else:
                # Test 4 not selected: use the pre-resolved path (may be
                # None if the dependency run failed -- fail 5U loudly).
                if ue_routes_pre and os.path.isfile(ue_routes_pre):
                    ue_routes = ue_routes_pre
                else:
                    log("Test 5: no ue_routes.csv available -- "
                        "UE sweep (5U) cannot run")
                    sweep_outcomes["U"] = False
                    sweep_5u_started = True
                    if tracker:
                        tracker.finish("5U", False)
                    return
            # Start the 5U sweep in the (still-open) executor, adding it
            # to the shared futures dict.
            sweep_5u_started = True
            try:
                sb5u = setup_sandbox("test_5U", data_dir, temp_root,
                                     scripts_for_sweep("U"))
            except Exception as e:
                print("ERROR: sandbox setup failed for Test 5 UE sweep: %s"
                      % e, file=sys.stderr)
                sweep_outcomes["U"] = False
                if tracker:
                    tracker.finish("5U", False)
                return
            sandboxes["5U"] = sb5u
            shutil.copy2(ue_routes, os.path.join(sb5u, "ue_routes.csv"))
            log("Test 5: starting UE sweep (5U) -- ue_routes.csv ready")
            _f5u = ex.submit(
                run_sweep, "U", sb5u, tracker, args.seed,
                os.path.join(sb5u, "ue_routes.csv"), args.network,
                args.engine)
            futures[_f5u] = ("sweep", "5U")
            log("Test 5: 5U sweep submitted to executor")

        def _maybe_run_test6():
            """Run Test 6 if all selected Tests 1-4 are done+collected."""
            nonlocal test6_done
            if test6_done or 6 not in tests:
                return
            if not tests_14:
                return
            if not all(outcomes.get(t) for t in tests_14):
                return
            test6_done = True
            try:
                ok6, dt6 = run_test6(results_root, tracker)
                outcomes[6] = ok6
            except RuntimeError as e:
                log("Test 6: %s" % e)
                outcomes[6] = False
            except Exception as e:  # pragma: no cover - defensive
                log("Test 6: unexpected error: %s" % e)
                outcomes[6] = False

        def _collect_now(kind, key):
            """Collect results immediately for a finished job.

            Returns True on success; False if collection itself failed
            (which marks the job as failed). Never hangs: pure file
            copies with per-call exception handling.
            """
            if kind == "test":
                t = key
                sandbox = sandboxes[t]
                label = "Test %d" % t
                try:
                    if t == 1:
                        collect_1(sandbox, results_root)
                    else:
                        collect_234(t, sandbox, results_root)
                except Exception as e:
                    print("ERROR: result collection failed for test %d: %s"
                          % (t, e), file=sys.stderr)
                    return False
            else:  # sweep 5P or 5U
                sub = "U" if key == "5U" else "P"
                sandbox = sandboxes[key]
                label = "Test 5 (%s sweep)" % ("UE" if sub == "U" else "proposed")
                try:
                    collect_5(sub, sandbox, results_root)
                except Exception as e:
                    print("ERROR: result collection failed for Test 5 "
                          "%s sweep: %s" % (sub, e), file=sys.stderr)
                    return False
            if not args.keep_temp:
                try:
                    shutil.rmtree(sandbox)
                except Exception as e:
                    log("WARNING: could not remove sandbox %s: %s"
                        % (sandbox, e))
                else:
                    log("%s: sandbox removed" % label)
            else:
                log("%s: sandbox kept (--keep-temp): %s" % (label, sandbox))
            return True

        # Overall deadline for Phase A. Test 5's two sweeps are the long
        # pole (~45 min each, in parallel); 18h is generous but guarantees
        # the process ALWAYS reaches the summary instead of hanging
        # silently at 100%.
        _PHASE_A_TIMEOUT_S = 18 * 3600
        _POLL_INTERVAL_S = 10.0

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers) as ex:
            # Submit initial Phase A jobs.
            for t in tests:
                if t in (1, 2, 3, 4):
                    fut = ex.submit(run_test, t, sandboxes[t], tracker,
                                    args.seed, args.test1_mode, args.network,
                                    args.gap_threshold, args.engine)
                    futures[fut] = ("test", t)
            if 5 in tests:
                fut = ex.submit(run_sweep, "P", sandboxes["5P"], tracker,
                                args.seed, None, args.network, args.engine)
                futures[fut] = ("sweep", "5P")
            # 5U sandbox was set up in Phase A setup (if ue_routes.csv
            # was pre-resolved). Submit it here, into the shared dict.
            if "5U" in sandboxes:
                log("Test 5: submitting UE sweep (5U) to executor")
                sweep_5u_started = True
                _fut5u = ex.submit(
                    run_sweep, "U", sandboxes["5U"], tracker, args.seed,
                    os.path.join(sandboxes["5U"], "ue_routes.csv"),
                    args.network, args.engine)
                futures[_fut5u] = ("sweep", "5U")
                log("Test 5: 5U sweep submitted to executor")

            # Polling loop (replaces as_completed). as_completed() blocks
            # until a future completes; if a worker thread never releases
            # its future (e.g. a stuck pipe reader), it hangs forever.
            # Polling fut.done() with a hard deadline guarantees forward
            # progress no matter what the workers do.
            _deadline = time.monotonic() + _PHASE_A_TIMEOUT_S
            _processed_keys = set()
            while True:
                # Process every completed-but-unprocessed future. New
                # futures (5U via _maybe_start_5u) may be added to the
                # dict while processing; list() snapshots the dict so
                # the mutation is safe, and the new future is picked up
                # on the next pass.
                for fut, (kind, key) in list(futures.items()):
                    if key in _processed_keys:
                        continue
                    if not fut.done():
                        continue
                    _processed_keys.add(key)
                    try:
                        # fut is done, so result() returns immediately;
                        # the timeout is a safety net only.
                        ok, dt = fut.result(timeout=120)
                    except Exception as e:  # pragma: no cover - defensive
                        log("job %s: unexpected runner error: %s" % (key, e))
                        if tracker:
                            tracker.finish(key, False)
                        ok, dt = False, 0.0
                    if ok:
                        # Collect immediately; a collection failure marks
                        # the job as failed (never hangs: file copies).
                        if not _collect_now(kind, key):
                            ok = False
                        elif tracker:
                            tracker.update(key, None, "done (collected)")
                    else:
                        sb = sandboxes.get(key, "?")
                        print("%s FAILED -- sandbox kept for debugging: %s"
                              % (key, sb))
                    if kind == "test":
                        outcomes[key] = ok
                        if key == 4:
                            test4_completed = True
                    elif key == "5P":
                        sweep_outcomes["P"] = ok
                    elif key == "5U":
                        sweep_outcomes["U"] = ok
                    # Test 6 triggers as soon as Tests 1-4 are collected,
                    # without waiting for the 5P sweep to finish.
                    _maybe_run_test6()
                    # 5U starts as soon as Test 4's ue_routes.csv is ready
                    # (may add a new future to the dict).
                    _maybe_start_5u()

                # Done when every future currently in the dict has been
                # processed. If _maybe_start_5u just added 5U, it is not
                # in _processed_keys, so we loop again.
                _all_keys = set(key for (kind, key) in futures.values())
                if _all_keys and _all_keys <= _processed_keys:
                    break

                # Hard deadline: never hang silently at 100%.
                if time.monotonic() > _deadline:
                    _stuck = sorted(
                        str(key) for (kind, key) in futures.values()
                        if key not in _processed_keys)
                    log("ERROR: Phase A timed out after %s; stuck jobs: %s"
                        % (fmt_min(_PHASE_A_TIMEOUT_S), _stuck))
                    print("ERROR: Phase A timed out; stuck jobs: %s"
                          % _stuck, file=sys.stderr)
                    for (kind, key) in futures.values():
                        if key in _processed_keys:
                            continue
                        if tracker:
                            tracker.finish(key, False)
                        if kind == "test":
                            outcomes[key] = False
                        elif key == "5P":
                            sweep_outcomes["P"] = False
                        elif key == "5U":
                            sweep_outcomes["U"] = False
                    # Don't let the `with` block's shutdown(wait=True)
                    # hang on stuck workers.
                    ex.shutdown(wait=False, cancel_futures=True)
                    break

                time.sleep(_POLL_INTERVAL_S)
    else:
        log("Phase A: no jobs to run")
        test6_done = False

    # Test 6 fallback: if it wasn't triggered inside the loop (e.g. no
    # Phase A jobs ran, or Tests 1-4 were not all selected), try it now
    # against whatever results exist.
    if 6 in tests and not (n_jobs and test6_done):
        try:
            ok6, dt6 = run_test6(results_root, tracker)
            outcomes[6] = ok6
        except RuntimeError as e:
            log("Test 6: %s" % e)
            outcomes[6] = False
        except Exception as e:  # pragma: no cover - defensive
            log("Test 6: unexpected error: %s" % e)
            outcomes[6] = False

    # ---- Test 5 aggregation (barrier on collected sweep OUTPUTS) ----
    # Filesystem-based: verifies the per-bigloop dirs actually exist under
    # results/test5/, instead of trusting the in-memory sweep_outcomes
    # dict. A sweep that "succeeded" but collected nothing fails the
    # barrier loudly with a clear message naming the missing sweep.
    if 5 in tests:
        if args.skip_aggregation:
            log("Test 5: aggregation skipped (--skip-aggregation)")
            agg_status = "skipped"
            if tracker:
                tracker.finish("5A", True, "skipped")
        else:
            _p_dirs = sorted(glob.glob(os.path.join(
                results_root, "test5", "proposed_sweep", "bigloop_*")))
            _u_dirs = sorted(glob.glob(os.path.join(
                results_root, "test5", "ue_sweep", "bigloop_*")))
            if not _p_dirs:
                print("ERROR: Cannot aggregate: "
                      "results/test5/proposed_sweep/ has no bigloop outputs "
                      "(5P sweep failed or was not collected)",
                      file=sys.stderr)
                log("Test 5: aggregation not run "
                    "(proposed_sweep outputs missing)")
                agg_status = "FAILED"
                if tracker:
                    tracker.finish("5A", False)
            elif not _u_dirs:
                print("ERROR: Cannot aggregate: "
                      "results/test5/ue_sweep/ has no bigloop outputs "
                      "(5U sweep failed or was not collected)",
                      file=sys.stderr)
                log("Test 5: aggregation not run (ue_sweep outputs missing)")
                agg_status = "FAILED"
                if tracker:
                    tracker.finish("5A", False)
            else:
                log("Test 5: aggregating %d proposed + %d UE bigloops ..."
                    % (len(_p_dirs), len(_u_dirs)))
                try:
                    ok_agg, _ = run_aggregation(results_root, args.seed,
                                                tracker)
                    agg_status = "ok" if ok_agg else "FAILED"
                except RuntimeError as e:
                    print("ERROR: %s" % e, file=sys.stderr)
                    agg_status = "FAILED"
                    if tracker:
                        tracker.finish("5A", False)

    # ---- Summary (display stays live until here) ----
    if display_thread:
        stop_event.set()
        display_thread.join(timeout=10)

    # ---- Table 4 emissions (Tests 1-4) ----
    # Collected-results only (no sandbox, no engine): runs once,
    # after every collection and after Test 5 aggregation, so a
    # full run always produces it. Partial runs write the rows
    # whose source files exist and note the rest as missing.
    if any(t in tests for t in (1, 2, 3, 4)):
        try:
            run_table4_emissions(results_root)
        except Exception as e:
            log("WARNING: Table 4 emissions failed: %s "
                "(other results unaffected)" % e)
    print_summary(results_root, tests, outcomes, sweep_outcomes, agg_status)
    log("All done (%s total)" % fmt_min(time.time() - t_start))

    failed = []
    for t in tests:
        if t == 5:
            if not (sweep_outcomes["P"] and sweep_outcomes["U"]
                    and agg_status in ("ok", "skipped")):
                failed.append(t)
        elif not outcomes.get(t):
            failed.append(t)
    _run_status = "failed" if failed else "completed"
    write_run_manifest(results_root, args, tests, data_dir, _run_status,
                       outcomes, sweep_outcomes, agg_status,
                       _run_started, time.time(), _run_cmdline)
    if failed:
        print("FAILED tests: %s" % failed, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
