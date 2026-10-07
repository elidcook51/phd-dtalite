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


def dtalite_kwargs():
    """kwargs for subprocess.run to silence DTALite (unless overridden)."""
    import subprocess
    if os.environ.get("DTALITE_VERBOSE") == "1":
        return {}
    return {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
