"""engine.py -- simulation-engine selection for the DTALite pipeline.

The pipeline can load demand with either of two engines behind the same
file seam: the DTALite executable (``dtalite``) or the pure-Python port
``pysim``. Both read ``input_agent.csv`` out of a DTALite_Files directory
and write the same four output files (``output_agent.csv``,
``output_LinkTDMOE.csv``, ``output_LinkMOE.csv``, ``output_trip.csv``),
so every downstream analysis and aggregation step works unchanged under
either engine.

Selection, in priority order:

1. an explicit ``engine`` argument passed by the caller,
2. the ``DTALITE_ENGINE`` environment variable,
3. ``dtalite`` (the historical behaviour, so code that never mentions an
   engine keeps running exactly as before).

Every command-line entry point in the pipeline (``wholerunner.py``,
``runner_parallel.py``, ``runner_parallel_ue.py``, ``scenarios_*.py``,
``preference_reaction.py``) instead defaults its ``--engine`` flag to
``pysim`` and forwards the value explicitly, so a wholerunner run uses
PySim unless ``--engine dtalite`` is given.

How the PySim call works (and why it is a subprocess, not an import):

Workers run inside ``RunX/`` sandboxes that copy ``DTALite_Files/`` only,
not the repo root, so the pysim package cannot be reached by a relative
path from the sandbox. This module resolves everything from its own file
location instead: ``engine.py`` lives next to ``run_pysim.py`` and the
``pysim/`` package in the script directory (the repo root, or a
wholerunner sandbox root -- the directory the pipeline scripts were
imported from, which survives the worker's chdir into ``RunX/``). The
engine command is a plain argument list -- never a shell string -- so
paths containing spaces (Windows user directories included) are safe,
and the call keeps the pipeline's existing timeout and stdin discipline
(``stdin=DEVNULL`` plus the per-network timeout from ``progress``).

PySim is run through ``run_pysim.py`` rather than imported in-process so
that a failed or hung load fails the same way an exe failure does: a
non-zero exit becomes a loud error, and the subprocess timeout actually
kills the run instead of leaving a stuck worker behind. ``run_pysim.py``
also appends one JSON record per load (input/output hashes, seconds) to
``engine_calls.log`` for auditing, mirroring the old engine shim.

Known PySim limits (fail loud, never silently diverge): ``pysim.run``
refuses flow models 0 (BPR) and 4 (car-following) and any run requesting
assignment iterations, and its emission columns are not the exe's VSP
post-pass (Energy/CO2/NOX/HC near zero for flag=1 inputs, CO on a
different basis). The pipeline never reads emission columns, and the
acceptance bar is the 13 contract fields, not emissions.
"""

import os
import subprocess
import sys

ENGINES = ("pysim", "dtalite")
ENV_VAR = "DTALITE_ENGINE"
DEFAULT_ENGINE = "dtalite"


def resolve(engine=None):
    """Return the engine name for this call: ``pysim`` or ``dtalite``.

    An explicit argument wins, then the ``DTALITE_ENGINE`` environment
    variable, then the historical default (``dtalite``). Anything else
    raises ValueError immediately -- a misspelled engine must never
    silently fall back to the other one.
    """
    if engine is None:
        engine = os.environ.get(ENV_VAR)
    engine = (engine or DEFAULT_ENGINE).strip().lower()
    if engine not in ENGINES:
        raise ValueError(
            "unknown engine %r (expected one of %s; set --engine or %s)"
            % (engine, ", ".join(ENGINES), ENV_VAR))
    return engine


def script_dir():
    """Directory holding this file (repo root, or a sandbox root).

    Every pipeline script lives in this same directory, and it is also
    where ``run_pysim.py`` and the ``pysim/`` package are staged, so
    paths built from here stay valid after a worker chdirs into RunX/.
    """
    return os.path.dirname(os.path.abspath(__file__))


def pysim_command(data_dir):
    """Command list that runs one PySim load over ``data_dir``.

    Raises RuntimeError if ``run_pysim.py`` is not sitting next to this
    module -- checked before any stale outputs are deleted, so a broken
    staging can never destroy a previous run's outputs.
    """
    runner = os.path.join(script_dir(), "run_pysim.py")
    if not os.path.isfile(runner):
        raise RuntimeError(
            "PySim runner not found: %s (engine.py, run_pysim.py and "
            "pysim/ must be staged together)" % runner)
    return [sys.executable, runner, os.path.abspath(data_dir)]


def pysim_env():
    """Environment for the PySim child process.

    Pins the engine for any nested resolver and points the audit log at
    this script's directory, unless the caller already set
    ``ENGINE_LOG_DIR`` (a twin driver or an explicit override wins).
    """
    env = dict(os.environ)
    env[ENV_VAR] = "pysim"
    env.setdefault("ENGINE_LOG_DIR", script_dir())
    return env


def run_pysim(data_dir, timeout, label, stdout=None):
    """Run one PySim load over ``data_dir`` and fail loud on any problem.

    ``timeout`` is the per-call budget in seconds (the same value the
    DTALite call would use); ``label`` names the call in error messages
    (e.g. ``bigloop=10, itr=3``). When ``stdout`` is a file object the
    child's output goes there (the msa/assignment log-file discipline);
    when it is None the output is captured and only the tail is shown on
    failure (the scenarios' discipline). Returns the elapsed seconds.
    Raises RuntimeError on timeout, launch failure, or non-zero exit --
    never a bare CalledProcessError or TimeoutExpired, so callers do not
    need a second error-translation layer.
    """
    import time
    cmd = pysim_command(data_dir)
    t0 = time.time()
    try:
        if stdout is None:
            proc = subprocess.run(
                cmd, cwd=os.path.abspath(data_dir),
                stdin=subprocess.DEVNULL, timeout=timeout,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=pysim_env())
        else:
            proc = subprocess.run(
                cmd, cwd=os.path.abspath(data_dir),
                stdin=subprocess.DEVNULL, timeout=timeout,
                stdout=stdout, stderr=subprocess.STDOUT,
                env=pysim_env())
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "PySim timed out after %ss (%s)" % (timeout, label))
    except OSError as exc:
        raise RuntimeError(
            "could not launch PySim (%s): %s" % (label, exc))
    if proc.returncode != 0:
        if stdout is None:
            tail = (proc.stdout or "").strip().splitlines()
            detail = tail[-1] if tail else "no output captured"
            raise RuntimeError(
                "PySim exited with code %d (%s): %s"
                % (proc.returncode, label, detail))
        raise RuntimeError(
            "PySim exited with code %d (%s); see the call log for detail"
            % (proc.returncode, label))
    return time.time() - t0
