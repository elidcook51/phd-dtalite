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

Sandboxes live in temp/run_<pid>/<name>/ and are deleted on success unless
--keep-temp is given. On failure the sandbox is kept and its path is
printed for debugging; other jobs keep running.

Test 5 needs the UE-guidance trio (runner_parallel_ue.py, parallel_msa_ue.py,
msa_ue.py) in ue_guidance/ next to this file, plus the v9 support scripts.

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
              "traveltimecal.py", "progress.py", "choiceset.py"]

# UE-guidance trio, extracted from ue_guidance_mpr_v1.zip, next to this file
UE_GUIDANCE_DIR = "ue_guidance"

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
        "scripts": ["scenarios_realtime.py", "preference_reaction.py"],
        "sigma": 1, "target": TARGETS[2]},
    3: {"label": "Test 3: real-time guidance, sigma=10 min",
        "kind": "scenario",
        "scripts": ["scenarios_realtime.py", "preference_reaction.py"],
        "sigma": 10, "target": TARGETS[3]},
    4: {"label": "Test 4: UE as information (MSA baseline)",
        "kind": "scenario",
        "scripts": ["scenarios_ue.py", "preference_reaction.py"],
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
                              "progress.py", "choiceset.py"],
                  "script_dir": UE_GUIDANCE_DIR,
                  # Per-script source dirs: the trio lives in ue_guidance/,
                  # the 5 support scripts live next to wholerunner.py (None),
                  # so both sweeps run byte-identical support code.
                  "script_dirs": {"runner_parallel_ue.py": UE_GUIDANCE_DIR,
                                  "parallel_msa_ue.py": UE_GUIDANCE_DIR,
                                  "msa_ue.py": UE_GUIDANCE_DIR,
                                  "assignment.py": None,
                                  "realtimeassignment.py": None,
                                  "traveltimecal.py": None,
                                  "progress.py": None,
                                  "choiceset.py": None},
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
    """Ordered tracker keys for the whole run. Key 4 is added whenever
    Test 5 is selected: a dependency Test 4 run may need a bar."""
    keys = []
    for t in selected:
        if t == 5:
            keys += ["5P", "5U", "5A"]
        else:
            keys.append(t)
    if 5 in selected and 4 not in keys:
        keys.append(4)
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
    if "running dtalite" in low:
        tracker.update(test, base + span * 0.70, "reaction: DTALite")
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
def test1_commands(sandbox, seed, test1_mode="single"):
    """Test 1 command lines.

    test1_mode="single": bigloop 10 only (100% MPR, the Test 1 case).
    test1_mode="sweep":  all 11 bigloops (the legacy MPR sweep spelling;
                         new runs should use --tests 5).
    """
    cmd = [PY, os.path.join(sandbox, "runner_parallel.py"),
           "--seed", str(seed)]
    if test1_mode == "single":
        cmd += ["--bigloops", "10"]
        desc = "v9 pipeline (runner_parallel.py --bigloops 10)"
    else:
        cmd += ["--bigloops", "0-10"]
        desc = "v9 pipeline (runner_parallel.py --bigloops 0-10, MPR sweep)"
    return [(cmd, desc)]


def test24_commands(test, sandbox, seed):
    """Guidance scenario followed by the preference-reaction step."""
    staging = os.path.join(sandbox, "staging")
    os.makedirs(staging, exist_ok=True)
    data = os.path.join(sandbox, "DTALite_Files")
    cfg = TESTS[test]

    if test in (2, 3):
        scenario = [PY, os.path.join(sandbox, "scenarios_realtime.py"),
                    "--sigma", str(cfg["sigma"]),
                    "--data-dir", data, "--result-dir", staging,
                    "--seed", str(seed)]
        scenario_desc = "real-time sigma=%d (guidance)" % cfg["sigma"]
        guidance_csv = os.path.join(staging, "realtime_routes.csv")
    else:  # test 4
        scenario = [PY, os.path.join(sandbox, "scenarios_ue.py"),
                    "--data-dir", data, "--result-dir", staging,
                    "--seed", str(seed)]
        scenario_desc = "UE baseline (MSA guidance)"
        guidance_csv = os.path.join(staging, "ue_routes.csv")

    reaction = [PY, os.path.join(sandbox, "preference_reaction.py"),
                "--guidance-csv", guidance_csv,
                "--data-dir", data,
                "--result-dir", staging,
                "--target-satisfaction", str(cfg["target"])]
    return [(scenario, scenario_desc),
            (reaction, "preference reaction")]


def test5_commands(sub, sandbox, seed, ue_routes_path=None):
    """One [(cmd, desc)] entry per Test 5 sweep. sub is "P" or "U"."""
    if sub == "P":
        cmd = [PY, os.path.join(sandbox, "runner_parallel.py"),
               "--bigloops", "0-10", "--seed", str(seed)]
        desc = "v9 pipeline (runner_parallel.py --bigloops 0-10, proposed MPR sweep)"
    else:
        if not ue_routes_path:
            raise ValueError("ue_routes_path is required for the UE-guidance sweep")
        cmd = [PY, os.path.join(sandbox, "runner_parallel_ue.py"),
               "--ue-routes", ue_routes_path,
               "--bigloops", "1-10", "--seed", str(seed)]
        desc = "UE-guidance MPR sweep (runner_parallel_ue.py --bigloops 1-10)"
    return [(cmd, desc)]


# ----------------------------------------------------------------------
# Execution (shared Popen engine; one thread per job)
# ----------------------------------------------------------------------
def _run_commands(label, sandbox, cmds, tracker, key, parse_line):
    """Run cmds sequentially in sandbox; each stdout line is appended to
    run.log and passed to parse_line(cmd_idx, line). Returns (ok, elapsed)."""
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
        with open(log_path, "w", encoding="utf-8", errors="replace") as logfh:
            for cmd_idx, (cmd, desc) in enumerate(cmds):
                step0 = time.time()
                if not quiet:
                    log("%s: %s ..." % (label, desc))
                else:
                    tracker.update(key, None, desc)
                proc = subprocess.Popen(
                    cmd, cwd=sandbox,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, errors="replace")
                while True:
                    line = proc.stdout.readline()
                    if line:
                        logfh.write(line)
                        parse_line(cmd_idx, line)
                    elif proc.poll() is not None:
                        break
                    else:
                        time.sleep(0.05)
                # Drain anything left in the pipe after exit.
                try:
                    rest = proc.stdout.read()
                except Exception:
                    rest = ""
                if rest:
                    logfh.write(rest)
                    for rline in rest.splitlines():
                        parse_line(cmd_idx, rline)
                logfh.flush()
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                if proc.returncode != 0:
                    if tracker:
                        tracker.finish(key, False)
                    log("%s: FAILED at '%s' (rc=%d); sandbox kept: %s" %
                        (label, desc, proc.returncode, sandbox))
                    return False, time.time() - t0
                if not quiet:
                    log("%s: %s done (%s)" %
                        (label, desc, fmt_min(time.time() - step0)))
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


def run_test(test, sandbox, tracker=None, seed=42, test1_mode="single"):
    """Run one test's command sequence inside its sandbox. Returns
    (ok, elapsed)."""
    cfg = TESTS[test]
    if test == 1:
        cmds = test1_commands(sandbox, seed, test1_mode)
    else:
        cmds = test24_commands(test, sandbox, seed)
    return _run_commands(cfg["label"], sandbox, cmds, tracker, test,
                         lambda i, line: parse_progress_line(test, i, line, tracker))


def run_sweep(sub, sandbox, tracker, seed, ue_routes_path=None):
    """Run one Test 5 sweep (sub "P"/"U"). Returns (ok, elapsed)."""
    key = "5" + sub
    label = "Test 5 (%s)" % TESTS[5]["subs"][sub]["label"]
    cmds = test5_commands(sub, sandbox, seed, ue_routes_path)
    return _run_commands(label, sandbox, cmds, tracker, key,
                         lambda i, line: parse_progress_line(key, i, line, tracker))


# ----------------------------------------------------------------------
# Result collection
# ----------------------------------------------------------------------
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
    rc = subprocess.run(cmd).returncode
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
def resolve_ue_routes(selected, outcomes, results_root, data_dir, temp_root,
                      tracker, args):
    """Return the absolute path of a usable ue_routes.csv for the Test 5
    UE sweep.

    Priority:
      1. This run's Test 4 output (4 in selected and it succeeded) --
         read from results/test4/ue_routes.csv (NOT the sandbox, which may
         be deleted already).
      2. Pre-existing results/test4/ue_routes.csv on disk.
      3. Run Test 4 now as a dependency, collect it, and use its output.
    Raises RuntimeError if all three fail.
    """
    target = os.path.join(results_root, "test4", "ue_routes.csv")
    if 4 in selected and outcomes.get(4):
        if os.path.isfile(target):
            log("Test 5: using ue_routes.csv from this run's Test 4")
            return target
        log("Test 5: Test 4 succeeded but %s is missing -- checking disk"
            % target)
    if os.path.isfile(target):
        log("Test 5: using existing results/test4/ue_routes.csv")
        return target
    log("Test 5: no usable ue_routes.csv -- running Test 4 as a dependency ...")
    sb = setup_sandbox("test_dep4", data_dir, temp_root, scripts_for_test(4))
    ok, dt = run_test(4, sb, tracker, args.seed, "single")
    if not ok:
        raise RuntimeError("Test 5 UE sweep needs ue_routes.csv: "
                           "dependency Test 4 failed and no existing "
                           "results/test4/ue_routes.csv")
    try:
        collect_234(4, sb, results_root)
    except Exception as e:
        raise RuntimeError("Test 5 UE sweep needs ue_routes.csv: "
                           "dependency Test 4 collection failed: %s" % e)
    if not args.keep_temp:
        shutil.rmtree(sb)
        log("test dep4: sandbox removed")
    if not os.path.isfile(target):
        raise RuntimeError("Test 5 UE sweep needs ue_routes.csv: "
                           "dependency Test 4 ran but produced no ue_routes.csv")
    log("Test 5: dependency Test 4 produced ue_routes.csv (%s)" % fmt_min(dt))
    return target


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


def results_exist(t, results_root):
    """True if Test t's results are present and usable by
    volume_estimation. Test 1 needs a predict/actual LinkMOE pair (the
    hard requirement for Figure 8); tests 2-4 each need
    predicted_LinkMOE.csv."""
    if t == 1:
        d = os.path.join(results_root, "test1", "full_results")
        if not os.path.isdir(d):
            return False
        return _proposed_pair_exists(d)
    if t in (2, 3, 4):
        return os.path.isfile(os.path.join(
            results_root, "test%d" % t, "predicted_LinkMOE.csv"))
    return False


def run_test6(results_root, tracker=None):
    """Run volume_estimation.py in-place (no sandbox; read-only analysis).
    Analyzes whichever of Tests 1-4 have usable results. Returns
    (ok, elapsed). Raises RuntimeError if none do."""
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
    rc = subprocess.run(cmd).returncode
    dt = time.time() - t0
    ok = rc == 0
    if tracker:
        tracker.finish(6, ok)
    log("Test 6: %s (%s)" % ("done" if ok else "FAILED", fmt_min(dt)))
    return ok, dt


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
    log("=" * 60)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
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
    ap.add_argument("--results-dir", default="results",
                    help="output root (default: results)")
    ap.add_argument("--keep-temp", action="store_true",
                    help="keep temp/run_<pid>/ sandboxes after success")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="parallel workers (default: min(4, phase-A jobs))")
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

    # Test 5 Phase B needs all 8 scripts the UE-sweep sandbox copies: the
    # UE-guidance trio in ue_guidance/ plus the 5 support scripts next to
    # this file. Auto-create ue_guidance/ from the trio in the repo root
    # on first run; fail fast only if they are missing from both places.
    # Resolve through scripts_for_sweep so this check can never drift from
    # the sandbox spec.
    if 5 in tests:
        ue_dir = os.path.join(HERE, UE_GUIDANCE_DIR)
        ue_trio = ["runner_parallel_ue.py", "parallel_msa_ue.py", "msa_ue.py"]
        trio_in_dir = all(os.path.isfile(os.path.join(ue_dir, f))
                          for f in ue_trio)
        if not trio_in_dir:
            trio_in_root = [f for f in ue_trio
                            if os.path.isfile(os.path.join(HERE, f))]
            if len(trio_in_root) == len(ue_trio):
                os.makedirs(ue_dir, exist_ok=True)
                for f in ue_trio:
                    shutil.copy(os.path.join(HERE, f),
                                os.path.join(ue_dir, f))
                print("[wholerunner] Auto-created %s/ from the trio in the "
                      "repo root" % UE_GUIDANCE_DIR)
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

    if args.max_workers is not None and args.max_workers < 1:
        print("ERROR: --max-workers must be >= 1", file=sys.stderr)
        sys.exit(2)

    keys = progress_keys(tests)

    # Live progress display unless disabled or stdout is not a terminal.
    use_progress = (not args.no_progress) and sys.stdout.isatty()
    tracker = ProgressTracker(keys) if use_progress else None
    stop_event = threading.Event()
    display_thread = None

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
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers) as ex:
            futures = {}
            for t in tests:
                if t in (1, 2, 3, 4):
                    fut = ex.submit(run_test, t, sandboxes[t], tracker,
                                    args.seed, args.test1_mode)
                    futures[fut] = ("test", t)
            if 5 in tests:
                fut = ex.submit(run_sweep, "P", sandboxes["5P"], tracker,
                                args.seed)
                futures[fut] = ("sweep", "5P")
            for fut in concurrent.futures.as_completed(futures):
                kind, key = futures[fut]
                try:
                    ok, dt = fut.result()
                except Exception as e:  # pragma: no cover - defensive
                    log("job %s: unexpected runner error: %s" % (key, e))
                    if tracker:
                        tracker.finish(key, False)
                    ok, dt = False, 0.0
                if kind == "test":
                    outcomes[key] = ok
                else:
                    sweep_outcomes["P"] = ok
    else:
        log("Phase A: no jobs to run")

    # ---- Phase A collection (in test order; keep failed sandboxes) ----
    for t in tests:
        if t not in (1, 2, 3, 4):
            continue
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

    if 5 in tests:
        sandbox = sandboxes["5P"]
        if not sweep_outcomes["P"]:
            print("Test 5 (proposed sweep) FAILED -- sandbox kept for "
                  "debugging: %s" % sandbox)
        else:
            try:
                collect_5("P", sandbox, results_root)
            except Exception as e:
                print("ERROR: result collection failed for Test 5 proposed "
                      "sweep: %s" % e, file=sys.stderr)
                sweep_outcomes["P"] = False
            else:
                if not args.keep_temp:
                    shutil.rmtree(sandbox)
                    log("test 5P: sandbox removed")
                else:
                    log("test 5P: sandbox kept (--keep-temp): %s" % sandbox)

    # ---- Test 6 (after Phase A collection; independent of Test 5) ----
    # Runs in the main thread: the display thread keeps rendering.
    if 6 in tests:
        try:
            ok6, dt6 = run_test6(results_root, tracker)
            outcomes[6] = ok6
        except RuntimeError as e:
            log("Test 6: %s" % e)
            outcomes[6] = False
        except Exception as e:  # pragma: no cover - defensive
            log("Test 6: unexpected error: %s" % e)
            outcomes[6] = False

    # ---- Test 5 Phase B (UE sweep, then aggregation) ----
    if 5 in tests:
        try:
            ue_routes = resolve_ue_routes(tests, outcomes, results_root,
                                          data_dir, temp_root, tracker, args)
        except RuntimeError as e:
            print("ERROR: %s" % e, file=sys.stderr)
            ue_routes = None
        if ue_routes is not None:
            try:
                sb5u = setup_sandbox("test_5U", data_dir, temp_root,
                                     scripts_for_sweep("U"))
            except Exception as e:
                print("ERROR: sandbox setup failed for Test 5 UE sweep: %s"
                      % e, file=sys.stderr)
                if display_thread:
                    stop_event.set()
                    display_thread.join(timeout=5)
                sys.exit(1)
            shutil.copy2(ue_routes, os.path.join(sb5u, "ue_routes.csv"))
            ok5u, _ = run_sweep("U", sb5u, tracker, args.seed,
                                os.path.join(sb5u, "ue_routes.csv"))
            sweep_outcomes["U"] = ok5u
            if ok5u:
                try:
                    collect_5("U", sb5u, results_root)
                except Exception as e:
                    print("ERROR: result collection failed for Test 5 "
                          "UE sweep: %s" % e, file=sys.stderr)
                    sweep_outcomes["U"] = False
            if not sweep_outcomes["U"]:
                print("Test 5 (UE sweep) FAILED -- sandbox kept for "
                      "debugging: %s" % sb5u)
            elif not args.keep_temp:
                shutil.rmtree(sb5u)
                log("test 5U: sandbox removed")
            else:
                log("test 5U: sandbox kept (--keep-temp): %s" % sb5u)

        # Aggregation is a hard barrier on BOTH sweeps' collected outputs.
        if args.skip_aggregation:
            log("Test 5: aggregation skipped (--skip-aggregation)")
            agg_status = "skipped"
            if tracker:
                tracker.finish("5A", True, "skipped")
        elif sweep_outcomes["P"] and sweep_outcomes["U"]:
            try:
                ok_agg, _ = run_aggregation(results_root, args.seed, tracker)
                agg_status = "ok" if ok_agg else "FAILED"
            except RuntimeError as e:
                print("ERROR: %s" % e, file=sys.stderr)
                agg_status = "FAILED"
                if tracker:
                    tracker.finish("5A", False)
        else:
            log("Test 5: aggregation not run (a sweep failed)")
            agg_status = "not-run"
            if tracker:
                tracker.update("5A", None, "not run")
                tracker.finish("5A", False)

    # ---- Summary (display stays live until here) ----
    if display_thread:
        stop_event.set()
        display_thread.join(timeout=10)

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
    if failed:
        print("FAILED tests: %s" % failed, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
