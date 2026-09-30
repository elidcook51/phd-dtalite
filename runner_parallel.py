"""
runner_parallel.py
==================

Parallel equivalent of ``runner.py``: runs all MSA iterations concurrently,
one sandbox (``RunX/``) per iteration, instead of one after another.

Usage
-----
    python runner_parallel.py            # all 11 iterations, auto workers
    python runner_parallel.py --workers 4
    python runner_parallel.py --iterations 3 --workers 2   # quick smoke test

Each iteration's result files are copied back to ``DTALite_Files/`` and every
``RunX/`` folder is deleted afterwards (failed runs keep their sandbox for
inspection).

The ``if __name__ == "__main__":`` guard is required on Windows, where
multiprocessing uses the *spawn* start method.
"""

import argparse
import os
import sys
import time

from parallel_msa import run_parallel_msa, runner_schedule


def main():
    parser = argparse.ArgumentParser(
        description="Run all MSA iterations in parallel sandboxes."
    )
    parser.add_argument(
        "--iterations", type=int, default=11,
        help="Number of iterations (bigloop = 0 .. iterations-1). Default: 11.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker processes. Default: min(iterations, cpu_count).",
    )
    parser.add_argument(
        "--no-keep-on-failure", action="store_true",
        help="Delete RunX sandboxes even when a run fails.",
    )
    args = parser.parse_args()

    # Run from the repo root so relative paths resolve, wherever the script
    # is launched from.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    schedule = runner_schedule(args.iterations)
    print(f"Launching {len(schedule)} MSA iterations in parallel "
          f"(workers={args.workers or 'auto'}).", flush=True)

    start = time.time()
    results = run_parallel_msa(
        schedule,
        max_workers=args.workers,
        repo_root=repo_root,
        keep_on_failure=not args.no_keep_on_failure,
    )
    elapsed = time.time() - start

    print(f"All {len(results)} iterations finished in {elapsed:.1f}s.", flush=True)
    for r in sorted(results, key=lambda d: d["bigloop"]):
        print(f"  bigloop={r['bigloop']}: "
              f"{len(r['results'])} result files -> DTALite_Files/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
