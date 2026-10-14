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


def parse_bigloops(spec):
    """Parse a bigloop selection spec into a sorted list of ints.

    Accepts comma-separated values and ranges, e.g.:
      "10"     -> [10]
      "0-10"   -> [0, 1, ..., 10]
      "5,10"   -> [5, 10]
      "0-3,10" -> [0, 1, 2, 3, 10]
    """
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s.strip()), int(hi_s.strip())
            if lo > hi:
                raise ValueError("invalid range %r in --bigloops" % part)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    seen = sorted(set(out))
    if not seen:
        raise ValueError("empty --bigloops selection")
    return seen


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
    parser.add_argument(
        "--seed", type=int, default=1,
        help="RNG seed for dividerand (user-type assignment). Default: 1.",
    )
    parser.add_argument(
        "--bigloops", default="0-10",
        help="Bigloops to run: e.g. '10' (Test 1, 100%% MPR only), "
             "'0-10' (full MPR sweep, Test 5), '5,10'. Default: '0-10'.",
    )
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be >= 1 (got %d)" % args.iterations)

    try:
        wanted = parse_bigloops(args.bigloops)
    except ValueError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        sys.exit(2)

    # Run from the repo root so relative paths resolve, wherever the script
    # is launched from.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    full_schedule = runner_schedule(args.iterations)
    schedule = [s for s in full_schedule if s[0] in wanted]
    missing = [b for b in wanted if b not in
               {s[0] for s in full_schedule}]
    if missing:
        print("ERROR: --bigloops selected %s, but --iterations %d only "
              "defines bigloops 0..%d" % (missing, args.iterations,
                                          args.iterations - 1),
              file=sys.stderr)
        sys.exit(2)
    print(f"Launching {len(schedule)} MSA iterations in parallel "
          f"(bigloops={[s[0] for s in schedule]}, "
          f"workers={args.workers or 'auto'}, seed={args.seed}).", flush=True)

    start = time.time()
    results = run_parallel_msa(
        schedule,
        max_workers=args.workers,
        repo_root=repo_root,
        keep_on_failure=not args.no_keep_on_failure,
        seed=args.seed,
    )
    elapsed = time.time() - start

    print(f"All {len(results)} iterations finished in {elapsed:.1f}s.", flush=True)
    for r in sorted(results, key=lambda d: d["bigloop"]):
        print(f"  bigloop={r['bigloop']}: "
              f"{len(r['results'])} result files -> DTALite_Files/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
