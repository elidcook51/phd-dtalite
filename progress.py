"""progress.py -- quiet DTALite + per-bigloop progress bars for Eli's pipeline.

Usage: workers call ``get_pbar(bigloop, total)`` once, then ``.update(1)``
per unit of work and ``.close()`` at the end. Falls back to plain
carriage-return lines if tqdm is not installed.

Set DTALITE_VERBOSE=1 in the environment to restore DTALite's console
output (by default it is silenced to keep the bars readable).
"""

import os
import sys

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False


class _DummyBar:
    """Minimal tqdm-like fallback: one updating line per worker."""

    def __init__(self, desc, total):
        self.desc = desc
        self.total = total
        self.n = 0
        self._show()

    def _show(self, extra=""):
        sys.stderr.write(
            f"\r{self.desc}: {self.n}/{self.total} {extra}".ljust(80))
        sys.stderr.flush()

    def update(self, n=1):
        self.n += n
        self._show()

    def set_description(self, desc):
        self.desc = desc
        self._show()

    def set_postfix_str(self, s):
        self._show(extra=s)

    def close(self):
        sys.stderr.write("\n")
        sys.stderr.flush()


def get_pbar(bigloop, total=20, position=None):
    """Return a progress bar for one bigloop worker.

    ``position`` stacks the 11 worker bars vertically; defaults to
    ``bigloop`` so bigloops 0..10 each get their own row.
    """
    desc = f"Bigloop {bigloop}"
    if position is None:
        try:
            position = int(bigloop)
        except (TypeError, ValueError):
            position = 0
    if _HAS_TQDM:
        return _tqdm(total=total, desc=desc, position=position,
                     leave=True, dynamic_ncols=True,
                     bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    return _DummyBar(desc, total)


def dtalite_timeout():
    """Per-call DTALite timeout in seconds.

    900 s is ample for Sioux Falls. A Baltimore call re-allocates
    ~47 GB of link MOE arrays on a 430k-link network, and at full
    demand a single call can legitimately exceed 900 s (Halbert's
    full-demand iterations ran ~33 min end to end), so Baltimore
    gets 3600 s. The timeout is only a hang guard; it does not
    change any computation.
    """
    try:
        import netconfig
        if netconfig.is_baltimore():
            return 3600
    except Exception:
        pass
    return 900


def dtalite_kwargs():
    """kwargs for subprocess.run to silence DTALite (unless overridden).

    Includes stdin=DEVNULL (prevents hangs on stdin reads) and a
    timeout (see dtalite_timeout; prevents infinite hangs if DTALite
    stalls). Callers should wrap in try/except
    subprocess.TimeoutExpired for clear error messages.
    """
    import subprocess
    if os.environ.get("DTALITE_VERBOSE") == "1":
        return {"stdin": subprocess.DEVNULL, "timeout": dtalite_timeout()}
    return {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL, "timeout": dtalite_timeout()}


def kill_dtalite_leftovers():
    """Force-kill any lingering DTALite.exe processes.

    Called after a DTALite timeout to prevent orphaned processes from
    holding files or CPU. Uses taskkill on Windows, pkill on POSIX.
    """
    import subprocess
    import sys
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "DTALite.exe", "/T"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
        else:
            subprocess.run(["pkill", "-9", "-f", "DTALite"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
    except Exception:
        pass



def check_dtalite_outputs(label, log_path):
    """Fail loud when DTALite exits 0 but leaves empty outputs.

    DTALite's g_ProgramStop() exits 0 mid-assignment, so a zero exit
    code does NOT mean the run completed: the step-0 truncation leaves
    0-byte outputs and only the MOE writers (which never ran) would
    have filled them. Catch that here, at the call site, instead of
    letting a later pandas read die with a context-free EmptyDataError.
    """
    for _name in ("output_agent.csv", "output_LinkTDMOE.csv"):
        _path = os.path.join("DTALite_Files", _name)
        if (not os.path.exists(_path)) or os.path.getsize(_path) == 0:
            raise RuntimeError(
                "DTALite wrote an empty %s after %s. The exe exits 0 even "
                "when it stops mid-assignment, so this means the run stopped "
                "early -- see %s for the exe's own stop message."
                % (_name, label, log_path)
            )