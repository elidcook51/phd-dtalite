"""Dual-network configuration for the DTALite pipeline.

The pipeline was built for Sioux Falls and is being extended to the
Baltimore metropolitan network.  Rather than rewriting every hardcoded
path, the run folders are kept interchangeable (the Baltimore folder is
seeded as ``DTALite_Files/`` inside each sandbox), and only the values
that genuinely differ by network live here:

- ``PERIOD_BASE`` / ``TIME_START``: Sioux Falls demand runs 6:00-7:00 AM
  (departures 360-419); Baltimore runs 5:00-6:00 PM (1020-1079).
- ``OD_COLS``: which ``inputagent`` columns identify an agent's OD for
  choice-set matching.  Sioux Falls matches on node IDs (cols 4,5);
  Baltimore matches on zone IDs (cols 2,3), mirroring the MATLAB
  ``od_path_map`` keyed by ``(origin_zone, dest_zone)``.
- ``N_PERIODS``: 60 one-minute periods for both networks.

Usage: call ``configure(network)`` once at the start of ``msa()`` /
``msa_ue()`` (or any standalone entry point).  All other modules read the
module-level values; the Sioux Falls defaults apply when ``configure``
was never called, so existing behavior is unchanged.
"""

NETWORK = 'siouxfalls'
PERIOD_BASE = 359
TIME_START = 360
N_PERIODS = 60
OD_COLS = (4, 5)

_NETWORKS = ('siouxfalls', 'baltimore')


def configure(network='siouxfalls'):
    """Select the active network.  Idempotent."""
    global NETWORK, PERIOD_BASE, TIME_START, OD_COLS
    network = (network or 'siouxfalls').lower()
    if network not in _NETWORKS:
        raise ValueError("unknown network %r (expected one of %s)"
                         % (network, _NETWORKS))
    NETWORK = network
    if network == 'baltimore':
        PERIOD_BASE = 1019
        TIME_START = 1020
        OD_COLS = (2, 3)
    else:
        PERIOD_BASE = 359
        TIME_START = 360
        OD_COLS = (4, 5)
    return network


def is_baltimore():
    return NETWORK == 'baltimore'
