#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply the Python<->MATLAB parity fixes found by the 2026-09-30 audit.
====================================================================

Run ONCE from your repo root, e.g.:

    cd "C:\\Users\\ucg8nb\\Python Projects\\PhD_Work\\DTALite_work"
    python apply_parity_fixes.py

What it does
------------
Applies the two agreed result-changing fixes (3 edits: FIX 1, FIX 2a, FIX 2b)
so the Python code matches the (fixed) MATLAB code in ~/workspace/matlab_fixed/.
(A third, low-impact discrepancy — MATLAB's extra comassignment(2, ..., bigloop+100)
call — is deliberately NOT applied; see parity_fix_summary.md.)

  FIX 1 (assignment.py, function assignment()):
      The 60-period tt/pltt/fuelcost loop omitted the observed travel-time
      branch. MATLAB (assignment.m lines 117-170) builds m/agentn from
      output_agent.csv path matches for itr > 1 and substitutes the
      observed mean travel time (tt) and max travel time (pltt) per
      (route, departure-minute); Python always computed from the link
      tables. The sibling functions fixedcomassignment()/comassignment()
      already had the faithful port - assignment() was simply missing it.

  FIX 2 (msa.py, function msa()):
      cc[itr-1] counted changes in `choice`, which under the transposed
      choiceset convention holds the OD index (never changes per agent),
      so every cc[1..17] was 0. MATLAB's cc block (msa.m) counts agents
      whose *route* index changed. The fix compares `final_rposition`
      (route indices) against a tracked `prev_rposition`.

Target commit: 4d39434 ("fixed small bug") on branch parallel-msa.
After running: py_compile both files, run the test suite, then commit.

Safety properties
-----------------
* Handles CRLF (Windows) files: replacements are done on LF-normalized
  text and the original line-ending style is restored on write.
* For each fix the script asserts the old text occurs EXACTLY once:
    - already applied  -> "skipped (already applied)"
    - not found        -> "skipped (old text not found ...)"
    - found >1 times   -> "FAILED ... aborting this fix" (never guesses)
* Idempotent: running the script twice does not double-apply anything.
"""

import os
import sys


# ---------------------------------------------------------------------------
# Fix definitions: (target_file, name, old_text, new_text, applied_marker)
# ---------------------------------------------------------------------------

FIX1_OLD = """\
            if path is None or path == "":

                continue



            tt[(i, col)] = np.zeros(60)

            pltt[(i, col)] = np.zeros(60)

            fuelcost[(i, col)] = np.zeros(60)



            for k in range(60):

                departure_time = k + 360



                travel_time, fuel = traveltimecal_fastv2(

                    departure_time,

                    TDlink,

                    path,

                    link_lengths,

                    gas,

                    itr,

                    0

                )



                tt[(i,col)][k] = travel_time

                pltt[(i,col)][k] = travel_time

                fuelcost[(i, col)][k] = fuel"""

FIX1_NEW = """\
            if path is None or path == "":

                continue



            # Observed travel-time branch (MATLAB assignment.m lines 117-170).

            # For itr > 1, use the observed mean/max travel times from the

            # previous DTALite run (output_agent.csv) for agents that used

            # this route at this departure minute; otherwise compute from

            # the link tables. (Mirrors fixedcomassignment()/comassignment().)

            m = 1

            agentn = []



            if itr == 1:

                m = 2

            else:

                findagent = [

                    idx

                    for idx, p in enumerate(agentpath)

                    if p == path

                ]



                for idx in findagent:

                    agentn.append([

                        agent[idx, 0],

                        agent[idx, 1],

                        agent[idx, 2]

                    ])



                    m += 1



                agentn = np.array(agentn)



            tt[(i, col)] = np.zeros(60)

            pltt[(i, col)] = np.zeros(60)

            fuelcost[(i, col)] = np.zeros(60)



            for k in range(60):

                dep_time = k + 360  # MATLAB: (k+359) with k = 1..60



                if m == 1:

                    # No agent used this route: compute from link tables.

                    tt_val, fc_val = traveltimecal_fastv2(

                        dep_time,

                        TDlink,

                        path,

                        link_lengths,

                        gas,

                        itr,

                        0

                    )



                    tt[(i, col)][k] = tt_val

                    pltt[(i, col)][k] = tt_val

                    fuelcost[(i, col)][k] = fc_val

                else:

                    if itr == 1:

                        ttloc = []

                    else:

                        ttloc = np.where(

                            np.floor(agentn[:, 1]) == dep_time

                        )[0]



                    if len(ttloc) == 0:

                        # No agent on this route at this minute: compute.

                        tt_val, fc_val = traveltimecal_fastv2(

                            dep_time,

                            TDlink,

                            path,

                            link_lengths,

                            gas,

                            itr,

                            0

                        )



                        tt[(i, col)][k] = tt_val

                        pltt[(i, col)][k] = tt_val

                        fuelcost[(i, col)][k] = fc_val

                    else:

                        # Observed times: mean -> tt, max -> pltt.

                        # Fuel is still computed from the link tables.

                        a = agentn[ttloc, 2]



                        tt[(i, col)][k] = np.mean(a)

                        pltt[(i, col)][k] = np.max(a)



                        _, fc_val = traveltimecal_fastv2(

                            dep_time,

                            TDlink,

                            path,

                            link_lengths,

                            gas,

                            itr,

                            0

                        )



                        fuelcost[(i, col)][k] = fc_val"""

FIX1_MARKER = "# Observed travel-time branch (MATLAB assignment.m lines 117-170)."

FIX2A_OLD = """\
    prev_choice = None"""

FIX2A_NEW = """\
    prev_choice = None
    prev_rposition = None"""

FIX2A_MARKER = "    prev_rposition = None"

FIX2B_OLD = """\
        if itr > 1:

            cc[itr - 1] = np.sum(choice != prev_choice)



        prev_choice = np.array(choice)"""

FIX2B_NEW = """\
        if itr > 1:

            # MATLAB counts agents whose *route* changed (msa.m cc block).

            # Under the transposed choiceset, `choice` holds the OD index

            # (never changes per agent); the route index is final_rposition.

            cc[itr - 1] = np.sum(final_rposition != prev_rposition)



        prev_choice = np.array(choice)
        prev_rposition = np.array(final_rposition)"""

FIX2B_MARKER = "np.sum(final_rposition != prev_rposition)"

# (file, fix name, old, new, applied-marker)
FIXES = [
    ("assignment.py",
     "FIX 1: observed tt/pltt branch in assignment()",
     FIX1_OLD, FIX1_NEW, FIX1_MARKER),
    ("msa.py",
     "FIX 2a: track prev_rposition alongside prev_choice",
     FIX2A_OLD, FIX2A_NEW, FIX2A_MARKER),
    ("msa.py",
     "FIX 2b: cc counts route-index changes, not OD-index changes",
     FIX2B_OLD, FIX2B_NEW, FIX2B_MARKER),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_normalized(path):
    """Read file as bytes; return (LF-normalized text, had_crlf)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    had_crlf = b"\r\n" in raw
    text = raw.replace(b"\r\n", b"\n").decode("utf-8")
    return text, had_crlf


def save_preserving_endings(path, text, had_crlf):
    """Write text back, restoring CRLF if the file originally had it."""
    out = text.replace("\n", "\r\n") if had_crlf else text
    with open(path, "wb") as fh:
        fh.write(out.encode("utf-8"))


def apply_one(path, name, old, new, marker):
    text, had_crlf = load_normalized(path)

    if marker in text:
        return "skipped (already applied)"

    count = text.count(old)
    if count == 1:
        text = text.replace(old, new, 1)
        save_preserving_endings(path, text, had_crlf)
        return "applied"
    if count == 0:
        return ("skipped (old text not found - your file may differ from "
                "commit 4d39434; no changes made)")
    return ("FAILED: old text found %d times - aborting this fix "
            "(will not guess)" % count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Python<->MATLAB parity fixes  (target commit 4d39434, parallel-msa)")
    print("=" * 70)

    missing = [f for f, _, _, _, _ in FIXES
               if not os.path.isfile(os.path.join(os.getcwd(), f))]
    if missing:
        print("ERROR: run this script from your repo root. Missing: %s"
              % ", ".join(sorted(set(missing))))
        sys.exit(1)

    results = []
    for fname, name, old, new, marker in FIXES:
        path = os.path.join(os.getcwd(), fname)
        try:
            status = apply_one(path, name, old, new, marker)
        except Exception as exc:  # never let one fix kill the report
            status = "FAILED with exception: %s" % exc
        results.append((fname, name, status))
        print("[%s] %s\n        -> %s" % (fname, name, status))

    print("=" * 70)
    applied = sum(1 for _, _, s in results if s == "applied")
    failed = sum(1 for _, _, s in results if s.startswith("FAILED"))
    print("Done: %d applied, %d skipped, %d failed."
          % (applied, len(results) - applied - failed, failed))
    if failed:
        print("One or more fixes FAILED - review the messages above before "
              "committing.")
        sys.exit(2)
    print("Next: python -m py_compile assignment.py msa.py")
    print("      run the test suite, then commit.")


if __name__ == "__main__":
    main()
