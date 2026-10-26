"""
parallel_msa_ue.py
==================

UE-guidance variant of ``parallel_msa.py``.

Identical sandboxing machinery (``RunX/`` per iteration), but every worker
runs ``msa_ue.msa_ue`` instead of ``msa.msa``: guided members receive their
routes from a UE reference solution (``ue_routes.csv``) rather than the
tournament-selected UO route.

The parallel-safe entry points are:

- ``msa_parallel_ue``  -> ``msa_ue.msa_ue`` (takes the extra ``ue_routes_path``
  argument, threaded through from the orchestrator)
- ``run_parallel_msa_ue`` -> orchestrator over a ``(bigloop, p, p_fix,
  p_realtime)`` schedule, threading ``ue_routes_path`` into every worker task

All other helpers (``assignment_parallel``, ``comassignment_parallel``,
``fixedcomassignment_parallel``, ``comrealtimeassignment_parallel``,
``realtimeassignment_fast_parallel``, ``runner_schedule``) are unchanged
from ``parallel_msa.py``.
"""

import concurrent.futures
import contextlib
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings

# Import the *modules* (not the functions) so the underlying implementations
# stay patchable / mockable in tests: e.g. ``msa_ue.msa_ue = fake`` works because
# the worker resolves the attribute at call time.
import msa_ue as _msa_mod
import assignment as _assignment_mod
import realtimeassignment as _realtime_mod
import engine as _engine_mod


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Name of the shared DTALite working folder (relative to the repo root).
SOURCE_DIR = "DTALite_Files"
NETWORK_SOURCE_DIRS = {
    "siouxfalls": "DTALite_Files",
    "baltimore": "baltimore_files",
}

#: Sandbox folders are named ``Run0``, ``Run1``, ... under the repo root.
SANDBOX_PREFIX = "Run"


# A single bigloop that produces no result within this long is treated
# as hung: fail with a clear error instead of blocking forever (which
# used to hang runner_parallel.py -- and therefore wholerunner --
# indefinitely at 100%%).
BIGLOOP_TIMEOUT_S = 3600

#: Per-bigloop wall-clock budget, measured from batch submission. A bigloop
#: still pending after this long is killed and marked failed (then retried
#: once) instead of holding the whole sweep hostage.
PER_BIGLOOP_TIMEOUT_S = 3000

#: Worker self-watchdog (see parallel_msa.py): os._exit(2) if the worker is
#: still alive after this long. Backstop for worker-side hangs.
WORKER_WATCHDOG_S = 3300

#: Retries for a failed/timed-out bigloop before permanent failure.
MAX_RETRIES = 1


def _timeouts_for_network(network):
    """Wall-clock budgets (per-bigloop, watchdog, absolute) in seconds.

    Mirrors parallel_msa.py (v34): the constants above are sized for
    Sioux Falls; a Baltimore bigloop runs ~18 DTALite calls at
    ~20 min each (~6 h) and needs much larger budgets.
    """
    if network == 'baltimore':
        return 43200.0, 46800.0, 50400.0
    return (float(PER_BIGLOOP_TIMEOUT_S), float(WORKER_WATCHDOG_S),
            float(BIGLOOP_TIMEOUT_S))


# Budgets for the batch currently running; run_parallel_msa_ue() sets
# this from the network before each batch.
_ACTIVE_TIMEOUTS = _timeouts_for_network('siouxfalls')



def _result_names(bigloop):
    """Per-iteration result files produced for a given ``bigloop``."""
    i = bigloop
    return [
    # v41: only the four paper-metric files are harvested (see README).
        f"{i}UOinfor_agent.csv",
        f"{i}predict_LinkMOE.csv",
        f"{i}actual_agent.csv",
        f"{i}actual_LinkMOE.csv",
    ]


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------

def sandbox_path(run_id, repo_root=None):
    """Absolute path of the ``RunX`` sandbox directory."""
    repo_root = os.path.abspath(repo_root or os.getcwd())
    return os.path.join(repo_root, f"{SANDBOX_PREFIX}{run_id}")


def stage_run_dir(run_id, source_dir=SOURCE_DIR, repo_root=None):
    """
    Create ``Run{run_id}/`` and copy the DTALite inputs into
    ``Run{run_id}/DTALite_Files/``.

    Any stale sandbox left by a crashed run is removed first.  Returns the
    absolute sandbox path.
    """
    repo_root = os.path.abspath(repo_root or os.getcwd())
    run_dir = sandbox_path(run_id, repo_root)
    src = os.path.join(repo_root, source_dir)
    dst = os.path.join(run_dir, SOURCE_DIR)

    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"Source DTALite folder not found: {src}"
        )

    if os.path.exists(run_dir):
        warnings.warn(f"Removing stale sandbox: {run_dir}")
        shutil.rmtree(run_dir, ignore_errors=True)

    shutil.copytree(src, dst)
    return run_dir


def collect_results(run_dir, bigloop, dest_dir=SOURCE_DIR):
    """
    Copy one iteration's result files from ``RunX/DTALite_Files/`` back to
    the real ``DTALite_Files/`` folder.

    Files that were not produced are skipped with a warning.  Returns the list
    of files actually copied.
    """
    src_dir = os.path.join(os.path.abspath(run_dir), SOURCE_DIR)
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    copied = []
    for name in _result_names(bigloop):
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(dest_dir, name))
            copied.append(name)
        else:
            warnings.warn(
                f"Expected result file not produced in {run_dir}: {name}"
            )
    return copied



def cleanup_run_dir(run_dir):
    """Delete the ``RunX`` sandbox directory to reclaim disk space."""
    shutil.rmtree(os.path.abspath(run_dir), ignore_errors=True)


@contextlib.contextmanager
def _sandbox_cwd(run_dir):
    """
    Temporarily ``chdir`` into the sandbox so every relative
    ``DTALite_Files/...`` path used by the existing code (including the
    ``DTALite.exe`` subprocess calls) resolves inside ``RunX/``.
    """
    prev = os.getcwd()
    os.chdir(os.path.abspath(run_dir))
    try:
        yield
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------------
# Parallel-safe versions of the pipeline functions
# ---------------------------------------------------------------------------

def msa_parallel_ue(bigloop, p, p_fix, p_realtime, ue_routes_path,
                    run_id=None, repo_root=None, keep_on_failure=True,
                    cleanup=True, msa_fn=None, seed=1,
                    network='siouxfalls', engine=None):
    """
    Parallel-safe version of ``msa_ue.msa_ue`` (UE-guidance MSA).

    Stages ``RunX/``, runs the full MSA pipeline inside it (including the
    ``DTALite.exe`` calls), copies that iteration's results back to the real
    ``DTALite_Files/`` and deletes the sandbox.

    Parameters
    ----------
    bigloop, p, p_fix, p_realtime : as in ``msa_ue.msa_ue``.
    ue_routes_path : path to ``ue_routes.csv`` (required; see
        ``msa_ue.msa_ue`` -- the CSV must be generated from the SAME demand
        file, with ``agent_id`` values 0..dsize-1).  Relative paths resolve
        against the repo root, not the sandbox: it is resolved to absolute
        before workers ``chdir`` into ``RunX/``.
    run_id : sandbox number; defaults to ``bigloop`` (``Run{bigloop}/``).
    repo_root : repo root containing ``DTALite_Files/``; defaults to cwd.
    keep_on_failure : if True, a failed run keeps its ``RunX/`` folder on
        disk for inspection instead of deleting it.
    cleanup : if False, the sandbox is left on disk even on success (used
        by ``run_parallel_msa_ue`` to harvest shared result files afterwards).
    msa_fn : optional
        ``(bigloop, p, p_fix, p_realtime, ue_routes_path)`` callable used
        instead of ``msa_ue.msa_ue``.  Testing hook: on Windows (spawn start
        method) monkeypatching ``msa_ue.msa_ue`` does not propagate into worker
        processes, so pass the fake explicitly.  Must be a top-level
        function to survive pickling to the workers.
    seed : RNG seed for ``dividerand`` (user-type assignment). Default: 1.
    """
    run_id = bigloop if run_id is None else run_id
    repo_root = os.path.abspath(repo_root or os.getcwd())
    _engine = _engine_mod.resolve(engine)
    source_dir = NETWORK_SOURCE_DIRS.get(network, SOURCE_DIR)
    dest_dir = os.path.join(repo_root, source_dir)

    run_dir = stage_run_dir(run_id, source_dir=source_dir, repo_root=repo_root)
    try:
        with _sandbox_cwd(run_dir):
            # Mirror runner.py's per-iteration setup inside the sandbox.
            shutil.copyfile(
                os.path.join(SOURCE_DIR, "input_agent_initial.csv"),
                os.path.join(SOURCE_DIR, "input_agent.csv"),
            )
            # Resolve ue_routes_path against the absolute repo_root, so it
            # stays valid inside the RunX/ sandbox after chdir.
            _ue_path = os.path.abspath(os.path.join(repo_root, ue_routes_path))
            if msa_fn is not None:
                msa_fn(bigloop, p, p_fix, p_realtime, _ue_path)
            else:
                _msa_mod.msa_ue(bigloop, p, p_fix, p_realtime, _ue_path,
                                seed=seed, network=network, engine=_engine)

        copied = collect_results(run_dir, bigloop, dest_dir=dest_dir)
    except Exception:
        if not keep_on_failure:
            cleanup_run_dir(run_dir)
        else:
            warnings.warn(
                f"Run {run_id} failed; sandbox kept at {run_dir} for inspection."
            )
        raise
    else:
        if cleanup:
            cleanup_run_dir(run_dir)

    return {"bigloop": bigloop, "run_id": run_id,
            "run_dir": run_dir, "results": copied}


def assignment_parallel(run_dir, itr, choiceset, user, routelocation, weights,
                        meanstd28, meanstd2, member, fix_user, realtime_user,
                        a0, inputagent, bigloop, dsize, num_tdlink_rows):
    """
    Parallel-safe version of ``assignment.assignment``.

    ``run_dir`` must be a staged sandbox (see ``stage_run_dir``); the call
    executes with cwd inside it so all ``DTALite_Files/...`` file access is
    isolated.  Signature is otherwise identical to ``assignment.assignment``.
    """
    with _sandbox_cwd(run_dir):
        return _assignment_mod.assignment(
            itr, choiceset, user, routelocation, weights, meanstd28, meanstd2,
            member, fix_user, realtime_user, a0, inputagent, bigloop, dsize,
            num_tdlink_rows,
        )


def comassignment_parallel(run_dir, itr, choiceset, user, routelocation,
                           weights, meanstd28, meanstd2, member, fix_user,
                           realtime_user, a0, inputagent, bigloop, dsize,
                           num_tdlink_rows):
    """
    Parallel-safe version of ``assignment.comassignment``.

    ``run_dir`` must be a staged sandbox (see ``stage_run_dir``); otherwise
    identical to ``assignment.comassignment``.
    """
    with _sandbox_cwd(run_dir):
        return _assignment_mod.comassignment(
            itr, choiceset, user, routelocation, weights, meanstd28, meanstd2,
            member, fix_user, realtime_user, a0, inputagent, bigloop, dsize,
            num_tdlink_rows,
        )


def fixedcomassignment_parallel(run_dir, itr, choiceset, routelocation,
                                nchoice, rposition, weights, meanstd28,
                                meanstd2, user, fix_user, dsize,
                                num_tdlink_rows):
    """
    Parallel-safe version of ``assignment.fixedcomassignment``.

    ``run_dir`` must be a staged sandbox (see ``stage_run_dir``); otherwise
    identical to ``assignment.fixedcomassignment``.
    """
    with _sandbox_cwd(run_dir):
        return _assignment_mod.fixedcomassignment(
            itr, choiceset, routelocation, nchoice, rposition, weights,
            meanstd28, meanstd2, user, fix_user, dsize, num_tdlink_rows,
        )


def comrealtimeassignment_parallel(run_dir, itr, choiceset, routelocation,
                                   phlength, realtime_user, nchoice,
                                   rposition, weights, meanstd28, meanstd2,
                                   user, bigloop, dsize, num_tdlink_rows):
    """
    Parallel-safe version of
    ``realtimeassignment.comrealtimeassignment``.

    ``run_dir`` must be a staged sandbox (see ``stage_run_dir``); otherwise
    identical to ``realtimeassignment.comrealtimeassignment``.
    """
    with _sandbox_cwd(run_dir):
        return _realtime_mod.comrealtimeassignment(
            itr, choiceset, routelocation, phlength, realtime_user, nchoice,
            rposition, weights, meanstd28, meanstd2, user, bigloop, dsize,
            num_tdlink_rows,
        )


def realtimeassignment_fast_parallel(run_dir, itr, choiceset, routelocation,
                                     phlength, realtime_user, nchoice,
                                     rposition, bigloop, dsize,
                                     num_tdlink_rows):
    """
    Parallel-safe version of
    ``realtimeassignment.realtimeassignment_fast``.

    ``run_dir`` must be a staged sandbox (see ``stage_run_dir``); otherwise
    identical to ``realtimeassignment.realtimeassignment_fast``.
    """
    with _sandbox_cwd(run_dir):
        return _realtime_mod.realtimeassignment_fast(
            itr, choiceset, routelocation, phlength, realtime_user, nchoice,
            rposition, bigloop, dsize, num_tdlink_rows,
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def runner_schedule(n=11):
    """
    Reproduce ``runner.py``'s (bigloop, p, p_fix, p_realtime) schedule for
    ``n`` iterations.
    """
    schedule = []
    for i in range(n):
        if i == 0:
            p, p_realtime, p_fix = 0.001, 0.4, 0.6
        else:
            p = 0.1 * i
            p_fix = (1 - p) * 0.6
            p_realtime = (1 - p) * 0.4
        schedule.append((i, p, p_fix, p_realtime))
    return schedule


def _worker_initializer():
    """Executor initializer: arm watchdog once per worker process.

    Called once when each pool worker process starts (via the executor's
    `initializer` parameter), NOT per-task. The watchdog measures worker
    LIFETIME. Per-task stuckness is handled by the parent's per-bigloop
    timeout.
    """
    def _watchdog():
        # Spawned workers re-import this module; the budget arrives via
        # env (set by run_parallel_msa_ue), not parent globals.
        budget = float(os.environ.get('DTALITE_WORKER_WATCHDOG_S',
                                      WORKER_WATCHDOG_S))
        time.sleep(budget)
        try:
            sys.stderr.write(
                "\nWORKER WATCHDOG: worker still alive after %ds; "
                "it is stuck -- forcing os._exit(2)\n" % budget)
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(2)

    t = threading.Thread(target=_watchdog, daemon=True,
                         name="worker-watchdog")
    t.start()


class SweepResults(list):
    """Per-bigloop result dicts with an explicit failure record.

    Behaves as a plain list of the successful per-bigloop result dicts,
    plus a ``failures`` attribute mapping bigloop -> error string.
    """

    def __init__(self, results=(), failures=None):
        super().__init__(results)
        self.failures = dict(failures or {})


def _msa_ue_worker(task):
    """
    Top-level worker entry point (must stay top-level for pickling with the
    *spawn* start method on Windows).  ``task`` is a plain dict of keyword
    arguments for ``msa_parallel_ue``.

    Writes a start-timestamp file so the parent can measure per-task
    timeout from actual execution start (not submission time).
    """
    try:
        import os
        ts_dir = task.get("_timestamp_dir")
        if ts_dir:
            bl = task.get("bigloop", "?")
            ts_file = os.path.join(ts_dir, f"started_{bl}.ts")
            with open(ts_file, "w") as tf:
                tf.write(str(time.monotonic()))
    except Exception:
        pass
    task = {k: v for k, v in task.items() if not k.startswith("_")}
    return msa_parallel_ue(**task)



def _kill_all_workers():
    """Aggressively terminate all child worker processes.

    Uses taskkill /F /T on Windows (kills entire process tree) and
    os.kill(SIGKILL) on POSIX. Called after the executor shuts down to
    ensure no spinning workers survive as orphans.
    """
    killed = 0
    for child in multiprocessing.active_children():
        try:
            pid = child.pid
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid), "/T"],
                    capture_output=True, timeout=10,
                )
            else:
                os.kill(pid, 9)  # SIGKILL
            killed += 1
        except Exception:
            pass
    for child in multiprocessing.active_children():
        try:
            child.terminate()
        except Exception:
            pass
    return killed


def _run_batch(task_list, max_workers, repo_root, msa_fn, seed,
               worker_fn):
    """Run one batch of bigloop tasks to completion-or-failure.

    Returns ``(ok, failed)`` where ``ok`` maps bigloop -> result dict and
    ``failed`` maps bigloop -> human-readable error string. Per-bigloop
    failures never raise; the batch always returns. A per-bigloop timeout
    (``PER_BIGLOOP_TIMEOUT_S``, dynamically tightened to 2x the median peer
    duration once >=3 finish) kills stuck workers. Sandboxes are left
    untouched; the caller decides cleanup.
    """
    ok = {}
    failed = {}
    if not task_list:
        return ok, failed

    futures = {}
    submit_time = time.monotonic()
    completed_durations = []
    # Timestamp dir for per-task start times (Fix 6)
    import tempfile
    ts_dir = tempfile.mkdtemp(prefix="bigloop_ts_")
    for task in task_list:
        task["_timestamp_dir"] = ts_dir
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_initializer)
    try:
        futures = {executor.submit(worker_fn, task): task
                   for task in task_list}
        pending = set(futures)
        processed = set()
        while pending:
            done, still_pending = concurrent.futures.wait(
                pending, timeout=5.0,
                return_when=concurrent.futures.FIRST_COMPLETED)
            now = time.monotonic()
            for future in done:
                if future in processed:
                    continue
                processed.add(future)
                task = futures[future]
                bl = task["bigloop"]
                try:
                    result = future.result()
                except Exception as exc:
                    failed[bl] = "%s: %s" % (type(exc).__name__, exc)
                else:
                    ok[bl] = result
                    completed_durations.append(now - submit_time)
            pending = still_pending - processed

            if pending:
                # Per-task timeout (Fix 6): measure from actual task start
                def _task_elapsed(task):
                    bl = task["bigloop"]
                    ts_file = os.path.join(ts_dir, f"started_{bl}.ts")
                    try:
                        with open(ts_file, "r") as tf:
                            start = float(tf.read().strip())
                        return now - start
                    except Exception:
                        return 0.0
                elapsed = now - submit_time  # batch-level backstop
                if len(completed_durations) >= 3:
                    median = sorted(completed_durations)[
                        len(completed_durations) // 2]
                    dyn_timeout = max(1800.0, 2.0 * median)
                else:
                    dyn_timeout = float("inf")
                timeout = min(_ACTIVE_TIMEOUTS[0],
                              dyn_timeout,
                              _ACTIVE_TIMEOUTS[2])
                if elapsed > timeout:
                    for future in pending:
                        task = futures[future]
                        bl = task["bigloop"]
                        if bl not in ok and bl not in failed:
                            failed[bl] = (
                                "timed out after %.0fs (budget %.0fs); "
                                "worker killed" % (elapsed, timeout))
                    try:
                        executor.shutdown(wait=False,
                                          cancel_futures=True)
                    except Exception:
                        pass
                    _kill_all_workers()
                    pending = set()
            if pending:
                time.sleep(1.0)
    finally:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _kill_all_workers()
        time.sleep(2.0)
        _kill_all_workers()
    return ok, failed


def run_parallel_msa_ue(schedule, ue_routes_path, max_workers=None,
                        repo_root=None, keep_on_failure=True, msa_fn=None,
                        seed=1, network='siouxfalls', engine=None,
                        max_retries=MAX_RETRIES):
    """
    Run many UE-guidance MSA iterations concurrently, one sandbox
    (``RunX/``) per iteration.

    Parameters
    ----------
    schedule : list of ``(bigloop, p, p_fix, p_realtime)`` tuples, e.g. from
        ``runner_schedule()``.
    ue_routes_path : path to ``ue_routes.csv`` (required; see
        ``msa_ue.msa_ue``).  Threaded into every worker task; relative paths
        resolve against ``repo_root``.
    max_workers : number of parallel worker processes; defaults to
        ``min(len(schedule), os.cpu_count())``.  Each worker runs its own
        copy of ``DTALite.exe``, so size this to your CPU/RAM.
    repo_root : repo root containing ``DTALite_Files/``; defaults to cwd.
    keep_on_failure : keep failed runs' ``RunX/`` folders for inspection.
        (v3: on any failure ALL sandboxes are kept regardless.)
    msa_fn : optional
        ``(bigloop, p, p_fix, p_realtime, ue_routes_path)`` callable used
        instead of ``msa_ue.msa_ue`` in every worker (testing hook; must be a
        top-level function to survive pickling).
    seed : RNG seed for ``dividerand`` (user-type assignment). Default: 1.
    max_retries : retries for a failed/timed-out bigloop before permanent
        failure. Default: 1.

    Returns
    -------
    SweepResults
        List of per-iteration result dicts for succeeded bigloops, with a
        ``.failures`` dict mapping bigloop -> error string. Non-empty
        ``.failures`` means the sweep was partial.

    Failure handling (v3): per-bigloop failures (exception, worker death,
    timeout) are recorded as warnings and retried once; the sweep continues
    with the remaining bigloops. Each worker arms a watchdog thread that
    calls ``os._exit(2)`` if stuck, converting hangs into loud failures.
    Sandboxes are deleted ONLY on full success; on any failure everything
    is kept. Shared result files come from the highest *successful* bigloop.
    Raises ``RuntimeError`` only if ALL bigloops failed.
    """
    global _ACTIVE_TIMEOUTS
    _ACTIVE_TIMEOUTS = _timeouts_for_network(network)
    os.environ['DTALITE_WORKER_WATCHDOG_S'] = str(int(_ACTIVE_TIMEOUTS[1]))
    repo_root = os.path.abspath(repo_root or os.getcwd())
    if not schedule:
        return SweepResults()
    if max_workers is None:
        max_workers = min(len(schedule), os.cpu_count() or 1)
    max_workers = max(1, max_workers)

    def _make_task(bigloop, p, p_fix, p_realtime):
        return {
            "bigloop": bigloop,
            "p": p,
            "p_fix": p_fix,
            "p_realtime": p_realtime,
            "ue_routes_path": ue_routes_path,
            "run_id": bigloop,
            "repo_root": repo_root,
            "keep_on_failure": keep_on_failure,
            "cleanup": False,  # orchestrator deletes only on full success
            "msa_fn": msa_fn,
            "seed": seed,
            "network": network,
            "engine": engine,
        }

    schedule_tuples = list(schedule)
    all_ok = {}
    all_failed = {}
    remaining = list(schedule_tuples)
    attempt = 0
    while remaining:
        attempt += 1
        task_list = [_make_task(*s) for s in remaining]
        ok, failed = _run_batch(task_list, max_workers, repo_root,
                                msa_fn, seed, _msa_ue_worker)
        all_ok.update(ok)
        # Resilient: record failures, continue; retry failed bigloops.
        for bl, err in sorted(failed.items()):
            warnings.warn(
                "UE-guidance MSA run failed for bigloop=%s (attempt %d): %s"
                % (bl, attempt, err))
        if not failed:
            break
        if attempt > max_retries:
            all_failed.update(failed)
            break
        warnings.warn(
            "run_parallel_msa_ue: retrying %d failed bigloop(s) with fresh "
            "workers." % len(failed))
        remaining = [s for s in schedule_tuples if s[0] in failed]


    # Cleanup: delete sandboxes ONLY on full success.
    if not all_failed:
        for bl in sorted(all_ok):
            run_dir = sandbox_path(all_ok[bl]["run_id"], repo_root)
            if os.path.exists(run_dir):
                cleanup_run_dir(run_dir)
    else:
        warnings.warn(
            "run_parallel_msa_ue: sweep partial: %d/%d bigloops succeeded; "
            "%d failed %s. Keeping ALL sandboxes for inspection." % (
                len(all_ok), len(schedule_tuples), len(all_failed),
                sorted(all_failed)))

    if all_failed and not all_ok:
        raise RuntimeError(
            "All %d UE-guidance MSA runs failed (bigloops=%s). Sandboxes "
            "kept for inspection." % (len(all_failed), sorted(all_failed)))
    if all_failed:
        warnings.warn(
            "UE-guidance MSA sweep partially completed: %d succeeded, %d "
            "failed (bigloops=%s). Failed sandboxes kept." % (
                len(all_ok), len(all_failed), sorted(all_failed)))
    return SweepResults([all_ok[bl] for bl in sorted(all_ok)],
                        failures=all_failed)

