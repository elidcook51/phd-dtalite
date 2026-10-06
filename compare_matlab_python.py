#!/usr/bin/env python3
"""
compare_matlab_python.py -- parity check between the MATLAB and Python MSA runs.

Compares the per-bigloop result files that both pipelines write into
DTALite_Files/ :

    {b}UOinfor_agent.csv / {b}UOinfor_LinkMOE.csv / {b}UOinfor_LinkTDMOE.csv
    {b}predict_agent.csv / {b}predict_LinkMOE.csv / {b}predict_LinkTDMOE.csv
    {b}actual_agent.csv  / {b}actual_LinkMOE.csv  / {b}actual_LinkTDMOE.csv

plus the final input_agent.csv from each folder, and reports the agent split
sizes from MATLAB's {b}_userassignment.mat for context.

Pass criterion per numeric column (numpy.allclose style):
    |matlab - python| <= atol + rtol * max(|matlab|, |python|)
String columns (e.g. path_node_sequence) must match exactly.
NaN matches NaN.

IMPORTANT for a meaningful comparison: the Python run must have used the
SAME agent split as MATLAB, i.e. msa.py's dividerand_matlab() loading
MATLAB's {b}_userassistant.mat (it does by default -- the pure-Python
dividerand() uses a different RNG stream and will NOT match agent-by-agent).

Usage:
    python compare_matlab_python.py --matlab <matlab DTALite_Files> --python <python DTALite_Files>
    python compare_matlab_python.py --matlab M --python P --bigloops 0 1 2 --rtol 1e-6 --atol 1e-8

Exit code: 0 if every compared file passes, 1 otherwise.
"""

import argparse
import csv
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# CSV reading (tolerant of DTALite's trailing-comma quirk and latin-1 bytes)
# ---------------------------------------------------------------------------

def read_loose_csv(path):
    """Read a CSV into a DataFrame, tolerating ragged rows.

    DTALite appends a stray trailing comma to data rows (24-col header,
    25-col rows). Trailing *empty* fields beyond the header width are
    dropped; nothing else is touched. latin-1 keeps stray bytes readable.
    """
    with open(path, "r", newline="", encoding="latin-1") as f:
        rows = list(csv.reader(f))
    # drop fully-empty rows (e.g. trailing newline artifacts)
    rows = [r for r in rows if any(c.strip() != "" for c in r)]
    if not rows:
        return pd.DataFrame()
    header, data = rows[0], rows[1:]
    ncols = len(header)
    fixed = []
    for r in data:
        while len(r) > ncols and r[-1].strip() == "":
            r = r[:-1]
        fixed.append(r)
    df = pd.DataFrame(fixed, columns=header)
    # strip whitespace that MATLAB/Python formatting can leave behind
    df = df.apply(lambda s: s.str.strip() if s.dtype == object else s)
    return df


def looks_like_row_index(col):
    """True if col is exactly 0,1,2,...,n-1 (pandas' default index column)."""
    try:
        v = pd.to_numeric(col, errors="coerce")
    except Exception:
        return False
    if v.isna().any():
        return False
    v = v.to_numpy(dtype=float)
    return np.array_equal(v, np.arange(len(v), dtype=float))


def coerce_numeric(col):
    """Return (is_numeric, float ndarray with NaN for empties)."""
    s = col.replace("", np.nan)
    v = pd.to_numeric(s, errors="coerce")
    # numeric if every non-empty entry converted
    mask = s.notna()
    if mask.any() and v[mask].notna().all():
        return True, v.to_numpy(dtype=float)
    return False, None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_frames(m_df, p_df, name, rtol, atol, out):
    """Compare two DataFrames. Returns (passed, notes list). Prints details."""
    notes = []
    out(f"\n=== {name} ===")

    # --- index-column artifact: pandas to_csv without index=False adds one
    if m_df.shape[1] != p_df.shape[1]:
        p_wider = p_df.shape[1] > m_df.shape[1]
        wider = p_df if p_wider else m_df
        narrower = m_df if p_wider else p_df
        if wider.shape[1] == narrower.shape[1] + 1 and looks_like_row_index(wider.iloc[:, 0]):
            dropped = "python" if p_wider else "matlab"
            notes.append(f"ignored leading row-index column in {dropped} file (formatting artifact)")
            trimmed = wider.iloc[:, 1:].copy()
            trimmed.columns = narrower.columns  # cosmetic: we compare positionally
            if p_wider:
                p_df = trimmed
            else:
                m_df = trimmed

    if m_df.shape != p_df.shape:
        out(f"  SHAPE MISMATCH: matlab {m_df.shape} vs python {p_df.shape} -> FAIL")
        return False, notes

    if list(m_df.columns) != list(p_df.columns):
        notes.append("column headers differ (comparing positionally): "
                     f"matlab={list(m_df.columns)[:4]}... python={list(p_df.columns)[:4]}...")

    nrows, ncols = m_df.shape
    out(f"  shape {m_df.shape}")
    passed = True
    for j in range(ncols):
        m_col, p_col = m_df.iloc[:, j], p_df.iloc[:, j]
        cname = str(m_df.columns[j])
        m_num_ok, m_num = coerce_numeric(m_col)
        p_num_ok, p_num = coerce_numeric(p_col)
        if m_num_ok and p_num_ok:
            tol = atol + rtol * np.maximum(np.abs(m_num), np.abs(p_num))
            both_nan = np.isnan(m_num) & np.isnan(p_num)
            one_nan = np.isnan(m_num) ^ np.isnan(p_num)
            bad = (np.abs(m_num - p_num) > tol) | one_nan
            bad[both_nan] = False
            nbad = int(bad.sum())
            if nbad:
                passed = False
                with np.errstate(invalid="ignore", divide="ignore"):
                    adiff = np.abs(m_num - p_num)
                    denom = np.maximum(np.abs(m_num), np.abs(p_num))
                    rdiff = np.where(denom > 0, adiff / denom, 0.0)
                adiff[bad == False] = -1  # noqa: E712
                rdiff[bad == False] = -1  # noqa: E712
                out(f"  [numeric] col {j} '{cname}': {nbad}/{nrows} rows differ "
                    f"(max abs diff {adiff.max():.6g}, max rel diff {rdiff.max():.6g})")
                for r in np.where(bad)[0][:5]:
                    out(f"      row {r}: matlab={m_num[r]!r} python={p_num[r]!r}")
        else:
            m_s = m_col.fillna("").astype(str)
            p_s = p_col.fillna("").astype(str)
            bad = (m_s.to_numpy() != p_s.to_numpy())
            nbad = int(bad.sum())
            if nbad:
                passed = False
                out(f"  [string] col {j} '{cname}': {nbad}/{nrows} rows differ")
                for r in np.where(bad)[0][:5]:
                    out(f"      row {r}: matlab='{m_s.iloc[r]}' python='{p_s.iloc[r]}'")
    out(f"  -> {'PASS' if passed else 'FAIL'}")
    return passed, notes


def load_userassignment(matlab_dir, bigloop):
    path = os.path.join(matlab_dir, f"{bigloop}_userassignment.mat")
    if not os.path.isfile(path):
        return None
    d = loadmat(path)
    def size(k):
        return int(d[k].size) if k in d else None
    return {k: size(k) for k in ("member", "realtime_user", "fix_user")}


def discover_bigloops(matlab_dir, python_dir):
    pat = re.compile(r"^(\d+)UOinfor_agent\.csv$")
    m = {pat.match(f).group(1) for f in os.listdir(matlab_dir) if pat.match(f)}
    p = {pat.match(f).group(1) for f in os.listdir(python_dir) if pat.match(f)}
    return sorted(m & p, key=int), sorted(m - p, key=int), sorted(p - m, key=int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RESULT_STEMS = ["UOinfor", "predict", "actual"]
RESULT_KINDS = ["agent", "LinkMOE", "LinkTDMOE"]


def main():
    ap = argparse.ArgumentParser(description="Compare MATLAB vs Python MSA result files.")
    ap.add_argument("--matlab", required=True, help="MATLAB DTALite_Files folder")
    ap.add_argument("--python", required=True, help="Python DTALite_Files folder")
    ap.add_argument("--bigloops", nargs="*", default=None,
                    help="bigloop ids to compare (default: auto-discover common ones)")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--skip-input-agent", action="store_true",
                    help="skip the final input_agent.csv comparison")
    args = ap.parse_args()
    out = print

    for d in (args.matlab, args.python):
        if not os.path.isdir(d):
            out(f"ERROR: not a directory: {d}")
            return 2

    if args.bigloops is None:
        bigloops, only_m, only_p = discover_bigloops(args.matlab, args.python)
        if only_m:
            out(f"NOTE: bigloops only in MATLAB folder (skipped): {only_m}")
        if only_p:
            out(f"NOTE: bigloops only in Python folder (skipped): {only_p}")
    else:
        bigloops = args.bigloops
    if not bigloops:
        out("ERROR: no common bigloops found.")
        return 2
    out(f"Comparing bigloops: {bigloops}  (rtol={args.rtol:g}, atol={args.atol:g})")

    # context: agent split sizes from MATLAB's .mat
    for b in bigloops:
        sizes = load_userassignment(args.matlab, b)
        if sizes:
            out(f"bigloop {b} agent split (from MATLAB .mat): " +
                ", ".join(f"{k}={v}" for k, v in sizes.items() if v is not None))
        else:
            out(f"WARNING: bigloop {b}: no {b}_userassignment.mat in MATLAB folder")

    results = []  # (label, passed)
    for b in bigloops:
        for stem in RESULT_STEMS:
            for kind in RESULT_KINDS:
                fname = f"{b}{stem}_{kind}.csv"
                mp = os.path.join(args.matlab, fname)
                pp = os.path.join(args.python, fname)
                label = f"bigloop {b}: {fname}"
                if not os.path.isfile(mp) or not os.path.isfile(pp):
                    missing = [t for t, p in (("matlab", mp), ("python", pp))
                               if not os.path.isfile(p)]
                    out(f"\n=== {label} ===\n  SKIP (missing in {', '.join(missing)})")
                    continue
                m_df = read_loose_csv(mp)
                p_df = read_loose_csv(pp)
                passed, notes = compare_frames(m_df, p_df, label,
                                               args.rtol, args.atol, out)
                for n in notes:
                    out(f"  note: {n}")
                results.append((label, passed))

    if not args.skip_input_agent:
        label = "final input_agent.csv"
        mp = os.path.join(args.matlab, "input_agent.csv")
        pp = os.path.join(args.python, "input_agent.csv")
        if os.path.isfile(mp) and os.path.isfile(pp):
            passed, notes = compare_frames(read_loose_csv(mp), read_loose_csv(pp),
                                           label, args.rtol, args.atol, out)
            for n in notes:
                out(f"  note: {n}")
            results.append((label, passed))
        else:
            out(f"\n=== {label} ===\n  SKIP (not present in both folders)")

    out("\n" + "=" * 60 + "\nSUMMARY")
    npass = sum(1 for _, p in results if p)
    for label, passed in results:
        out(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    out(f"\n{npass}/{len(results)} files passed")
    if npass == len(results):
        out("OVERALL: PASS -- MATLAB and Python results match within tolerance.")
    else:
        out("OVERALL: FAIL -- see mismatches above.")
    return 0 if npass == len(results) and results else 1


if __name__ == "__main__":
    sys.exit(main())