#!/usr/bin/env python3
"""Debug patch for Test 5 bigloop 1 agent-count anomaly.

This is NOT a fix -- it is a diagnostic instrument. Apply it temporarily
to log the agent count at each stage of bigloop 1's pipeline, run Test 5
with --keep-temp, and inspect the log to find where 54,045 becomes 36,030.

HOW TO APPLY (from the repo root, back up first):
    Copy-Item msa.py msa.py.bak
    Copy-Item msa_ue.py msa_ue.py.bak
    Copy-Item parallel_msa.py parallel_msa.py.bak
    Copy-Item parallel_msa_ue.py parallel_msa_ue.py.bak
    python bigloop1_debug_patch.py

HOW TO REVERT:
    Copy-Item msa.py.bak msa.py
    Copy-Item msa_ue.py.bak msa_ue.py
    Copy-Item parallel_msa.py.bak parallel_msa.py
    Copy-Item parallel_msa_ue.py.bak parallel_msa_ue.py

WHAT IT DOES:
    Inserts a print() statement after the demand read in msa() and
    msa_ue(), logging: bigloop, input file row count (dsize), and the
    input file path. The log line looks like:
        [BIGLOOP1-DEBUG] bigloop=1 dsize=36030 file=DTALite_Files/input_agent_initial.csv

    Run Test 5 with --keep-temp so the Run1 sandbox survives, then check:
    1. The debug log lines for bigloop 1 in each sweep's run.log.
    2. Row counts in the surviving sandboxes (see commands below).
"""

import os
import sys

MARKER = "[BIGLOOP1-DEBUG]"

# The anchor line in msa.py / msa_ue.py, and the debug line to insert after it.
# Both files have the same pattern:
#     temp_agent_data = pd.read_csv('DTALite_Files/input_agent_initial.csv')
#     inputagent = temp_agent_data.values.tolist()
#     dsize = len(inputagent)
ANCHOR = "    dsize = len(inputagent)"
DEBUG_LINE = (
    "    print(f\"%s bigloop=%s dsize=%s file=DTALite_Files/input_agent_initial.csv\" "
    "% (\"" + MARKER + "\", bigloop, dsize), flush=True)"
)


def patch_file(path):
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    if MARKER in text:
        print(f"SKIP (already patched): {path}")
        return False
    if ANCHOR not in text:
        print(f"ERROR: anchor not found in {path}")
        return False
    # Insert debug line after the FIRST occurrence of the anchor.
    # (msa.py and msa_ue.py each have exactly one `dsize = len(inputagent)`.)
    text = text.replace(ANCHOR, ANCHOR + "\n" + DEBUG_LINE, 1)
    # Preserve the file's line endings (CRLF on Windows).
    newline = "\r\n" if "\r\n" in text else "\n"
    with open(path, "w", encoding="utf-8", newline=newline) as f:
        f.write(text)
    print(f"PATCHED: {path}")
    return True


def main():
    targets = ["msa.py", "msa_ue.py"]
    ok = True
    for t in targets:
        if not os.path.isfile(t):
            print(f"ERROR: not found (run from repo root): {t}")
            ok = False
        elif not patch_file(t):
            ok = False
    if ok:
        print()
        print("Patch applied. Now run:")
        print("  python wholerunner.py --tests 5 --seed 42 --engine dtalite --keep-temp")
        print()
        print("Then search the sweep run.logs for BIGLOOP1-DEBUG lines, and run:")
        print("  (Get-Content temp/run_*/test_5P/run.log | Select-String 'BIGLOOP1-DEBUG')")
        print("  (Get-Content temp/run_*/test_5U/run.log | Select-String 'BIGLOOP1-DEBUG')")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
