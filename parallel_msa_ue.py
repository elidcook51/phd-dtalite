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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Name of the shared DTALite working folder (relative to the repo root).
SOURCE_DIR = "DTALite_Files"

#: Sandbox folders are named ``Run0``, ``Run1``, ... under the repo root.
SANDBOX_PREFIX = "Run"

#: Files written by the pipeline that carry no per-iteration tag.  In the
#: serial runner the final on-disk state of these is whatever the *last*
#: iteration wrote, so the parallel orchestrator reproduces that by copying
#: them back from the highest ``bigloop`` run only.
SHARED_RESULT_NAMES = [
    "UOinfor_agent.csv",
    "UOinfor_LinkMOE.csv",
    "UOinfor_LinkTDMOE.csv",
]

# A single bigloop that produces no result within this long is treated
# as hung: fail with a clear error instead of blocking forever (which
# used to hang runner_parallel.py -- and therefore wholerunner --
# indefinitely at 100%%).
BIGLOOP_TIMEOUT_S = 3600



def _result_names(bigloop):
    """Per-iteration result files produced for a given ``bigloop``."""
    i = bigloop
    return [
        f"{i}UOinfor_agent.csv",
        f"{i}UOinfor_LinkMOE.csv",
        f"{i}UOinfor_LinkTDMOE.csv",
        f"{i}predict_agent.csv",
        f"{i}predict_LinkMOE.csv",
        f"{i}predict_LinkTDMOE.csv",
        f"{i}actual_agent.csv",
        f"{i}actual_LinkMOE.csv",
        f"{i}actual_LinkTDMOE.csv",
        f"myfile{i}.mat",
        f"pathinfo_ass{i}.mat",
        f"pathinfo_comass{i}.mat",
        f"pathinfo_comrealass{i}.mat",
        f"pathinfo_realass{i}.mat",
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
    dst = os.path.join(run_dir, source_dir)

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

    Files that were not produced (e.g. ``pathinfo_ass{i}.mat`` is only
    written for ``itr == 18``) are skipped with a warning.  Returns the list
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


def collect_shared_results(run_dir, dest_dir=SOURCE_DIR):
    """
    Copy the shared (non-iteration-tagged) result files back.  The
    orchestrator calls this once, from the highest-``bigloop`` run, so the
    final on-disk state matches a serial run.
    """
    src_dir = os.path.join(os.path.abspath(run_dir), SOURCE_DIR)
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    copied = []
    for name in SHARED_RESULT_NAMES:
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(dest_dir, name))
            copied.append(name)
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
                    cleanup=True, msa_fn=None, seed=1):
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
    dest_dir = os.path.join(repo_root, SOURCE_DIR)

    run_dir = stage_run_dir(run_id, repo_root=repo_root)
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
                                seed=seed)

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


def _msa_ue_worker(task):
    """
    Top-level worker entry point (must stay top-level for pickling with the
    *spawn* start method on Windows).  ``task`` is a plain dict of keyword
    arguments for ``msa_parallel_ue``.
    """
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


def run_parallel_msa_ue(schedule, ue_routes_path, max_workers=None,
                        repo_root=None, keep_on_failure=True, msa_fn=None,
                        seed=1):
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
    msa_fn : optional
        ``(bigloop, p, p_fix, p_realtime, ue_routes_path)`` callable used
        instead of ``msa_ue.msa_ue`` in every worker (testing hook; must be a
        top-level function to survive pickling).
    seed : RNG seed for ``dividerand`` (user-type assignment). Default: 1.

    Returns
    -------
    list of per-iteration result dicts (in completion order).

    The shared (non-iteration-tagged) result files are copied back once,
    from the highest-``bigloop`` run, so the final on-disk state matches a
    serial run.  Every sandbox is deleted afterwards (unless a
    run failed and ``keep_on_failure`` is True).
    """
    repo_root = os.path.abspath(repo_root or os.getcwd())
    if not schedule:
        return []
    if max_workers is None:
        max_workers = min(len(schedule), os.cpu_count() or 1)
    max_workers = max(1, max_workers)

    tasks = [
        {
            "bigloop": bigloop,
            "p": p,
            "p_fix": p_fix,
            "p_realtime": p_realtime,
            "ue_routes_path": ue_routes_path,
            "run_id": bigloop,
            "repo_root": repo_root,
            "keep_on_failure": keep_on_failure,
            "cleanup": False,  # orchestrator harvests shared files first
            "msa_fn": msa_fn,
            "seed": seed,
        }
        for (bigloop, p, p_fix, p_realtime) in schedule
    ]

    results = []
    failures = []
    futures = {}
    try:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        )
        try:
            futures = {
                executor.submit(_msa_ue_worker, task): task for task in tasks
            }
            pending = set(futures)
            _POLL_INTERVAL_S = 5.0  # never busy-spin: sleep between polls
            _t_start = time.monotonic()
            _processed = set()
            while pending:
                # Poll with a short timeout so we can detect completion
                # promptly without busy-spinning.
                done, still_pending = concurrent.futures.wait(
                    pending, timeout=_POLL_INTERVAL_S,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    if future in _processed:
                        continue
                    _processed.add(future)
                    task = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        # Resilient: record the failure and continue with
                        # remaining bigloops instead of aborting the sweep.
                        # The failed sandbox is kept for inspection (see
                        # finally block below).
                        failures.append(task['bigloop'])
                        warnings.warn(
                            f"UE-guidance MSA run failed for bigloop={task['bigloop']} "
                            f"(sandbox Run{task['run_id']} kept for inspection): {exc}"
                        )
                pending = still_pending - _processed
                # Absolute timeout: record stuck bigloops as failures
                _elapsed = time.monotonic() - _t_start
                if pending and _elapsed > BIGLOOP_TIMEOUT_S:
                    for future in pending:
                        task = futures[future]
                        failures.append(task['bigloop'])
                        warnings.warn(
                            f"UE-guidance MSA run timed out after {BIGLOOP_TIMEOUT_S}s "
                            f"for bigloop={task['bigloop']} "
                            f"(sandbox Run{task['run_id']} kept for inspection)."
                        )
                    pending = set()
                if pending:
                    time.sleep(1.0)  # never busy-spin
        finally:
            # Never block forever on stuck workers: cancel what has not
            # started and do NOT wait for running workers. Then
            # aggressively kill any workers still alive so they cannot
            # linger as spinning orphans after this process exits.
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            _kill_all_workers()
            time.sleep(2.0)
            _kill_all_workers()

        # Mimic serial semantics: shared result files come from the last
        # (highest bigloop) iteration -- use the highest *successful* one.
        if results:
            last_ok = max(results, key=lambda r: r["bigloop"])
            collect_shared_results(
                sandbox_path(last_ok["run_id"], repo_root),
                dest_dir=os.path.join(repo_root, SOURCE_DIR),
            )
        if failures:
            if not results:
                raise RuntimeError(
                    f"All {len(failures)} UE-guidance MSA runs failed "
                    f"(bigloops={sorted(failures)}). Sandboxes kept for inspection."
                )
            warnings.warn(
                f"UE-guidance MSA sweep partially completed: "
                f"{len(results)} succeeded, {len(failures)} failed "
                f"(bigloops={sorted(failures)}). Failed sandboxes kept."
            )
        return results
    finally:
        # The inner finally already shut the executor down without waiting,
        # so some futures may be unfinished/cancelled here; classify each
        # run by its actual outcome rather than by consumed results.
        for fut, task in futures.items():
            try:
                succeeded = (fut.done() and not fut.cancelled()
                             and fut.exception() is None)
            except Exception:
                succeeded = False
            run_dir = sandbox_path(task["run_id"], repo_root)
            if os.path.exists(run_dir):
                if succeeded or not keep_on_failure:
                    cleanup_run_dir(run_dir)
                else:
                    warnings.warn(
                        f"Keeping failed sandbox for inspection: {run_dir}"
                    )
