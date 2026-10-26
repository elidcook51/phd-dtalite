#!/usr/bin/env python3
"""
table4_emissions.py -- Per-scenario emissions for Tests 1-4 (Table 4).

Straightforward column sums over the collected full_results agent files,
using exactly the Table 6 sustainability definitions from
mpr_aggregate.py:

    Energy: sum of TotalEnergy_(KJ) / 1e9  -> 10^9 KJ
    CO2:    sum of CO2_(g) / 1e6            -> ton
    NOX:    sum of NOX_(g) / 1e3            -> kg
    CO:     sum of CO_(g) / 1e3             -> kg
    HC:     sum of HC_(g) / 1e3             -> kg
    (No PM / PM2.5.)

Source files (agent files, NOT routes files -- routes files carry only
route sequences and a satisfaction flag, no emission columns):

    Test 1: results/test1/full_results/{N}actual_agent.csv, where N is
            the last complete bigloop (same rule as wholerunner.py's
            find_last_bigloop: the highest bigloop with both an actual
            and a UOinfor agent file). collect_1() keeps only that
            bigloop, so this is the file it collected.
    Tests 2-4: results/testN/full_results/output_agent.csv. This is the
            agent file the simulation engine writes (both engines:
            DTALite writes it and PySim's run_pysim.py/engine.py write
            the same four output files). preference_reaction.py does NOT
            write an agent file -- it writes actual_routes.csv (routes
            only) and reads output_agent.csv for the actual run's
            vehicle-hours -- so output_agent.csv, left behind by the
            reaction run, is the actual-run agent file for Tests 2-4.
            There is no <n>actual_agent.csv for Tests 2-4.

NO correction factor is applied. The sums reproduce whatever is in the
files, verbatim: if the engine's agent files contain double-counted (or
otherwise inflated) emission values, the Table 4 sums will contain them
too. Any such correction belongs in the engine or in the paper text,
not in this summation.

PySim compatibility: PySim writes the same emission columns (see
pysim/outputs.py AGENT_HEADER); with emissions off it writes 0.0 values.
A present-but-zero column sums to 0.00 and is reported as 0.00, NOT as
N/A -- only a genuinely absent column is N/A. Rows that fail float
conversion and NaN values are skipped, matching mpr_aggregate.py's
sum_sustainability(); a missing source file or a missing column never
crashes the run, it is reported as missing/N/A.

Usage (standalone, e.g. over an existing results dir):

    python table4_emissions.py --results-dir results
    python table4_emissions.py --results-dir results_baltimore

Outputs (written into <results-dir>/):

    table4_emissions.csv          Scenario, Energy (10^9 KJ), CO2 (ton),
                                  NOX (kg), CO (kg), HC (kg)
    table4_emissions_summary.txt  per-scenario source file, row count,
                                  columns used, and any missing data

The per-test summary files are also annotated: a "Table 4 emissions"
block is appended to results/test1/proposed_summary.txt and to
results/testN/reaction_summary.txt (N = 2, 3, 4), once only -- if the
block is already present it is not appended again. A missing per-test
summary file is noted in the summary, not created.

wholerunner.py calls build_table4_emissions() once, after all Phase A
collections, whenever any of Tests 1-4 was selected; a partial run
writes rows for the scenarios whose source files exist and lists the
rest as missing, and never fails the overall run.

Python 3.9 compatible, stdlib only.
"""

import argparse
import csv
import os
import re
import sys

# (metric, candidate canonical column names, divisor to table unit, unit)
# Identical to mpr_aggregate.py's SUST_SPECS, so Table 4 and Table 6
# always use the same definitions.
SUST_SPECS = [
    ("energy", ("totalenergykj", "energykj", "energyinkj"), 1e9, "10^9 KJ"),
    ("co2",    ("co2g",),  1e6, "ton"),
    ("nox",    ("noxg",),  1e3, "kg"),
    ("co",     ("cog",),   1e3, "kg"),
    ("hc",     ("hcg",),   1e3, "kg"),
]

# Decimals per metric, matching mpr_aggregate.py's SUST_DECIMALS.
SUST_DECIMALS = {"energy": 2, "co2": 1, "nox": 1, "co": 1, "hc": 1}

NA = "N/A"

TABLE4_HEADER = ["Scenario", "Energy (10^9 KJ)", "CO2 (ton)",
                 "NOX (kg)", "CO (kg)", "HC (kg)"]

# (test number, scenario label, results subdir, per-test summary file)
TABLE4_SCENARIOS = [
    (1, "Test 1 (Proposed)", "test1", "proposed_summary.txt"),
    (2, "Test 2 (Real-time, sigma=1)", "test2", "reaction_summary.txt"),
    (3, "Test 3 (Real-time, sigma=10)", "test3", "reaction_summary.txt"),
    (4, "Test 4 (UE)", "test4", "reaction_summary.txt"),
]

TAG_RE = re.compile(r"^(\d+)(UOinfor|predict|actual)_agent\.csv$")

MARKER = "Table 4 emissions:"


def _canon(col):
    return re.sub(r"[^a-z0-9]", "", col.strip().lower())


def find_last_bigloop(d):
    """Last bigloop number with both actual and UOinfor agent files.

    Same rule as wholerunner.py's find_last_bigloop, applied here to
    the collected test1/full_results/ directory.
    """
    found = {}
    try:
        names = os.listdir(d)
    except OSError:
        return None
    for name in names:
        m = TAG_RE.match(name)
        if m:
            found.setdefault(int(m.group(1)), set()).add(m.group(2))
    cands = [b for b, tags in found.items()
             if "actual" in tags and "UOinfor" in tags]
    return max(cands) if cands else None


def sum_emissions(path):
    """Sum per-agent emission columns in one agent file.

    Returns (totals, cols_used, n_rows): totals maps metric -> float in
    table units, or None when the column is absent; cols_used maps
    metric -> the actual CSV column name summed; n_rows is the number
    of data rows read. Rows failing float conversion and NaN values
    are skipped, exactly as in mpr_aggregate.sum_sustainability().
    Raises OSError / csv.Error on unreadable files; the caller turns
    those into a missing-data note, never a crash.
    """
    totals = {m: None for m, _, _, _ in SUST_SPECS}
    cols_used = {}
    n_rows = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        canon = {_canon(c): c for c in (reader.fieldnames or []) if c}
        for metric, cands, _div, _unit in SUST_SPECS:
            hit = next((canon[c] for c in cands if c in canon), None)
            if hit is not None:
                cols_used[metric] = hit
                totals[metric] = 0.0
        for row in reader:
            n_rows += 1
            for metric, _c, divisor, _u in SUST_SPECS:
                col = cols_used.get(metric)
                if col is None:
                    continue
                try:
                    v = float(row[col])
                except (TypeError, ValueError):
                    continue
                if v == v:  # not NaN
                    totals[metric] += v
        for metric, _c, divisor, _u in SUST_SPECS:
            if totals[metric] is not None:
                totals[metric] = totals[metric] / divisor
    return totals, cols_used, n_rows


def resolve_source(results_root, test):
    """Collected agent file for one test, or None if it is not there.

    Test 1: test1/full_results/{last bigloop}actual_agent.csv.
    Tests 2-4: testN/full_results/output_agent.csv (the engine's
    actual-run agent file; see the module docstring).
    """
    name = "test%d" % test
    full = os.path.join(results_root, name, "full_results")
    if test == 1:
        n = find_last_bigloop(full)
        if n is None:
            return None
        path = os.path.join(full, "%dactual_agent.csv" % n)
        return path if os.path.isfile(path) else None
    path = os.path.join(full, "output_agent.csv")
    return path if os.path.isfile(path) else None


def _fmt(v, nd):
    return NA if v is None else ("%.*f" % (nd, v))


def _fmt_row(label, totals):
    row = [label]
    for metric, _c, _d, _unit in SUST_SPECS:
        row.append(_fmt(totals[metric], SUST_DECIMALS[metric]))
    return row


def _append_block(path, block):
    """Append block to a per-test summary file, once only.

    Returns True if appended, False if the file is missing or already
    carries the block. The block uses the file's existing line ending
    (CRLF files stay CRLF, LF files stay LF) so no file ends up mixed.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return False
    if MARKER.encode() in data:
        return False
    eol = b"\r\n" if b"\r\n" in data else b"\n"
    text = data.decode("utf-8-sig", errors="replace")
    if text and not text.endswith(("\n", "\r")):
        text += "\n"
    # Normalize the appended block to the file's ending in one pass.
    block_lines = block.replace("\r\n", "\n").split("\n")
    sep = "\r\n" if eol == b"\r\n" else "\n"
    text += sep.join(block_lines)
    if not text.endswith(sep):
        text += sep
    with open(path, "w", newline="") as f:
        f.write(text)
    return True


def build_table4_emissions(results_root):
    """Compute Table 4 emissions for Tests 1-4 from collected results.

    Writes table4_emissions.csv and table4_emissions_summary.txt into
    results_root, appends a Table 4 block to each available per-test
    summary file, and returns (rows, notes): rows is a list of
    (label, totals, source, n_rows) for every scenario whose source
    file was found, notes lists human-readable lines about missing
    scenarios/columns. Never raises for missing data; only genuine
    I/O errors writing the outputs propagate.
    """
    rows = []
    notes = []
    for test, label, subdir, summary_name in TABLE4_SCENARIOS:
        src = resolve_source(results_root, test)
        if src is None:
            notes.append("%s: source agent file not found under "
                         "results/%s/full_results/ -- row omitted"
                         % (label, subdir))
            continue
        try:
            totals, cols_used, n_rows = sum_emissions(src)
        except (OSError, csv.Error) as e:
            notes.append("%s: could not read %s (%s) -- row omitted"
                         % (label, src, e))
            continue
        rows.append((label, totals, src, n_rows, cols_used))
        missing = [m for m, _, _, _ in SUST_SPECS if m not in cols_used]
        if missing:
            notes.append("%s: column(s) not found in %s -> N/A: %s"
                         % (label, os.path.basename(src),
                            ", ".join(missing)))
        # Annotate the per-test summary (once only; missing file noted).
        summary_path = os.path.join(results_root, subdir, summary_name)
        block_lines = ["", MARKER]
        for metric, _c, _d, unit in SUST_SPECS:
            block_lines.append("  %-6s: %s %s"
                               % (metric,
                                  _fmt(totals[metric],
                                       SUST_DECIMALS[metric]),
                                  unit))
        block_lines.append("  source: %s (%d agent rows)"
                           % (os.path.basename(src), n_rows))
        if not _append_block(summary_path, "\n".join(block_lines)):
            try:
                with open(summary_path, "rb") as f:
                    already = MARKER.encode() in f.read()
            except OSError:
                already = False
            if not already:
                notes.append("%s: per-test summary %s not found -- "
                             "Table 4 block not appended"
                             % (label, summary_path))

    csv_path = os.path.join(results_root, "table4_emissions.csv")
    with open(csv_path, "w", newline="") as f:  # csv emits \r\n -> CRLF
        w = csv.writer(f)
        w.writerow(TABLE4_HEADER)
        for label, totals, _src, _n, _cols in rows:
            w.writerow(_fmt_row(label, totals))

    lines = []
    lines.append("Table 4 emissions (Tests 1-4) -- table4_emissions_summary.txt")
    lines.append("source: collected full_results agent files under %s"
                 % os.path.abspath(results_root))
    lines.append("")
    lines.append("Straightforward column sums, same definitions as Table 6 "
                 "(mpr_aggregate.py SUST_SPECS). NO correction factor is "
                 "applied: the sums reproduce whatever is in the files, "
                 "including any engine double-counting, if present.")
    lines.append("A present-but-zero column (e.g. PySim with emissions off) "
                 "is reported as 0.00, not N/A; only an absent column is N/A.")
    lines.append("")
    for label, totals, src, n_rows, cols_used in rows:
        lines.append("%s" % label)
        lines.append("  source: %s (%d agent rows)" % (src, n_rows))
        for metric, _c, _d, unit in SUST_SPECS:
            col = cols_used.get(metric, "column not found")
            lines.append("  %-6s: %s %s  <- %s"
                         % (metric, _fmt(totals[metric],
                                         SUST_DECIMALS[metric]),
                            unit, col))
        lines.append("")
    if notes:
        lines.append("Notes")
        lines.append("-" * 78)
        for note in notes:
            lines.append("- %s" % note)
        lines.append("")
    summary_path = os.path.join(results_root, "table4_emissions_summary.txt")
    data = "\n".join(lines)
    # One normalization pass to CRLF (never chained sed/python replaces).
    data = data.replace("\r\n", "\n").replace("\n", "\r\n")
    with open(summary_path, "w", newline="") as f:
        f.write(data)
    return rows, notes


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Compute Table 4 per-scenario emissions (Tests 1-4) "
                    "from collected wholerunner results.")
    ap.add_argument("--results-dir", default="results",
                    help="results root written by wholerunner.py "
                         "(default: results)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results_root = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_root):
        sys.stderr.write("error: results dir not found: %s\n"
                         % results_root)
        return 1
    rows, notes = build_table4_emissions(results_root)
    print("wrote %s (%d scenario rows)"
          % (os.path.join(results_root, "table4_emissions.csv"),
             len(rows)))
    print("wrote %s"
          % os.path.join(results_root, "table4_emissions_summary.txt"))
    for note in notes:
        print("note: %s" % note)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
