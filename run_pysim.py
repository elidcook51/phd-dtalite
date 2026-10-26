#!/usr/bin/env python3
"""run_pysim.py -- one PySim loading pass, as a pipeline engine call.

Usage: python run_pysim.py <data_dir> [--log-dir <dir>]

Runs ``pysim.run.run_loading(data_dir, data_dir)`` -- the same input
file in (``input_agent.csv``) and the same DTALite output files out
(``output_agent.csv``, ``output_LinkTDMOE.csv``, ``output_LinkMOE.csv``,
``output_trip.csv``) -- then appends one JSON record to
``engine_calls.log`` with the input/output SHA-256 prefixes and the
elapsed seconds, so a sweep can be audited load by load.

The log goes to ``--log-dir`` if given, else ``ENGINE_LOG_DIR`` if set,
else the parent of ``data_dir`` (the old engine shim's location). The
pipeline's engine module sets ``ENGINE_LOG_DIR`` to the script directory,
so under wholerunner the log lands in the test sandbox root where it can
be collected with the results.

This is deliberately a thin subprocess entry point, not a compiled
launcher: the pipeline invokes it with an argument list (no shell), so
data directories with spaces in them are safe, and a PySim failure
(unsupported flow model, assignment iterations requested, bad inputs)
propagates as a traceback plus a non-zero exit -- fail loud, never a
partial output mistaken for a completed load. On success this script
prints nothing; the audit record is the only output.
"""

import hashlib
import json
import os
import sys
import time


def _hash(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def _find_pysim():
    """Import run_loading, looking next to this script first.

    The package normally sits beside this file (repo root or sandbox
    root). ``PYSIM_HOME`` -- a directory containing ``pysim/`` -- is the
    escape hatch for layouts where it does not, and a plain sys.path
    import is the last resort. Raises RuntimeError naming every location
    tried, so a staging mistake is diagnosable instead of an opaque
    ImportError.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    tried = [here]
    sys.path.insert(0, here)
    try:
        from pysim.run import run_loading
        return run_loading
    except ImportError:
        pass
    home = os.environ.get("PYSIM_HOME")
    if home:
        tried.append(os.path.abspath(home))
        sys.path.insert(0, os.path.abspath(home))
        try:
            from pysim.run import run_loading
            return run_loading
        except ImportError:
            pass
    raise RuntimeError(
        "pysim package not found (looked for pysim/ next to run_pysim.py "
        "in %s; set PYSIM_HOME to a directory containing pysim/)"
        % ", ".join(tried))


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print("usage: python run_pysim.py <data_dir> [--log-dir <dir>]",
              file=sys.stderr)
        return 2
    data_dir = os.path.abspath(argv[1])
    log_dir = None
    i = 2
    while i < len(argv):
        if argv[i] == "--log-dir" and i + 1 < len(argv):
            log_dir = argv[i + 1]
            i += 2
        else:
            print("unknown argument: %s" % argv[i], file=sys.stderr)
            return 2
    if not os.path.isdir(data_dir):
        print("data directory not found: %s" % data_dir, file=sys.stderr)
        return 2
    run_loading = _find_pysim()
    t0 = time.time()
    inp = _hash(os.path.join(data_dir, "input_agent.csv"))
    run_loading(data_dir, data_dir)
    rec = {
        "cwd": data_dir,
        "input_agent_sha256_16": inp,
        "output_agent_sha256_16": _hash(
            os.path.join(data_dir, "output_agent.csv")),
        "output_LinkTDMOE_sha256_16": _hash(
            os.path.join(data_dir, "output_LinkTDMOE.csv")),
        "output_LinkMOE_sha256_16": _hash(
            os.path.join(data_dir, "output_LinkMOE.csv")),
        "output_trip_sha256_16": _hash(
            os.path.join(data_dir, "output_trip.csv")),
        "seconds": round(time.time() - t0, 2),
    }
    if log_dir is None:
        log_dir = os.environ.get("ENGINE_LOG_DIR")
    if log_dir is None:
        log_dir = os.path.dirname(data_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "engine_calls.log"), "a",
              encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
