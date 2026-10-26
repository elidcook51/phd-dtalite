"""Compare a PySim arm against a DTALite arm of a wholerunner suite.

v2.1: emissions-ready, path-time numeric, total_*/CO_(g/hr) detection, agent-only zero FAIL, mpr_table6 report-only, LinkTDMOE triple join (see emissions paragraph below). Core 13 contract
fields are unchanged from v1; only the emissions handling changed.

Usage (from the repo root, after both arms have run):
    python compare_arms_v2.py [--pysim results] [--dtalite results_dtalite]
                           [--tests 1,2,3,4,5,6] [--atol 1e-6] [--rtol 1e-9]
                           [--report compare_report.txt]

Expected layout (wholerunner, pysim-integration v1):
    <root>/test1/full_results/{n}{UOinfor,predict,actual}_{agent,LinkMOE,LinkTDMOE}.csv
    <root>/test1/proposed_summary.txt, engine_calls.log (pysim only)
    <root>/test2|test3|test4/{*_summary.txt, *_routes.csv, predicted_LinkMOE.csv,
                             full_results/output_*.csv, engine_calls.log}
    <root>/test5/proposed_sweep/bigloop_{0..10}/{n}...csv (9 files each)
    <root>/test5/ue_sweep/bigloop_{1..10}/{n}...csv (9 files each)
    <root>/test5/mpr_table5.csv, mpr_table6.csv, mpr_summary.txt
    <root>/test6/volume_estimation.csv, volume_estimation_summary.txt

Comparison rules:
  - CSVs: headers must match exactly (after stripping whitespace).
    Rows are joined on the first column when its values are unique
    (agent_id, link id), else on (column 0, column 1) when unique
    (link, timestamp for LinkTDMOE), else compared in file order.
    Numeric cells pass within atol + rtol * |b|. Other string cells must
    match exactly. Path/sequence columns are reported as a match rate,
    and any mismatch in them fails the file.
  - Emissions columns (TotalEnergy, CO2, NOX, CO, HC, PM, PM2.5) are
    COMPARED, not skipped: PySim v1 now has its own VSP post-pass, so
    per-column sums, mean/median relative differences, and the
    zero-pattern are reported for every agent and Table 6 file.
    Emissions NEVER fail on size of difference alone (the two models
    differ structurally, see README), but a zero-pattern mismatch
    (>1% of rows where one engine is zero and the other is not) DOES
    fail, because that is a bug signature, not a model difference.
  - Summary .txt files are compared with both roots masked out; any
    difference is a WARN, never a FAIL (summaries can carry run metadata,
    and every number they carry is derived from the CSVs, which are
    compared exactly).
  - Missing core files (the 9 test1/test5 CSVs, mpr tables, test6 CSV)
    FAIL. A missing output_*.csv in test2/3/4 full_results is a WARN:
    PySim's output repertoire may legitimately be a subset.
  - engine_calls.log is expected under pysim and absent under dtalite;
    violations are WARN, not FAIL (it is an audit artifact, not results).

Exit code: 0 when no FAIL items, 1 otherwise. Stdlib only, Python 3.9
compatible. Files are read fully into memory; fine at Sioux Falls scale.
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import Counter

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)

TAGS = ("UOinfor", "predict", "actual")
KINDS = ("agent", "LinkMOE", "LinkTDMOE")
EMISSION_STEMS = ("totalenergy", "co2", "nox", "hc", "pm")
EMISSION_EXACT = ("co", "cog")


def norm_col(name):
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


def is_emission(norm):
    if norm in EMISSION_EXACT or norm.startswith(EMISSION_STEMS):
        return True
    if norm.startswith("cog"):
        return True
    if norm.startswith("total"):
        stripped = norm[5:]
        if stripped == "co" or stripped.startswith("cog"):
            return True
        if any(stripped.startswith(s) for s in ("co2", "nox", "hc", "pm", "energy")):
            return True
    return False


def is_path(norm):
    return "path" in norm or "sequence" in norm


def parse_float(text):
    try:
        value = float(text.strip())
    except (ValueError, AttributeError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def read_csv(path):
    """Return (header, rows) with stripped headers; skips blank rows."""
    with open(path, "r", encoding="utf-8-sig", errors="replace",
              newline="") as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            return [], []
        header = [h.strip() for h in raw_header]
        rows = []
        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue
            rows.append(row)
    return header, rows


def pad(row, width):
    if len(row) >= width:
        return row
    return row + [""] * (width - len(row))


class FileResult(object):
    def __init__(self, label):
        self.label = label
        self.verdict = "PASS"
        self.notes = []

    def fail(self, note):
        self.verdict = "FAIL"
        self.notes.append(note)

    def warn(self, note):
        self.notes.append("WARN: " + note)


def compare_csv(pa, pb, label, atol, rtol):
    """Compare two CSVs. Returns a FileResult."""
    res = FileResult(label)
    if not os.path.isfile(pa):
        res.fail("missing in pysim arm: %s" % pa)
        return res
    if not os.path.isfile(pb):
        res.fail("missing in dtalite arm: %s" % pb)
        return res
    try:
        ha, rows_a = read_csv(pa)
        hb, rows_b = read_csv(pb)
    except OSError as exc:
        res.fail("unreadable: %s" % exc)
        return res
    if ha != hb:
        only_a = [h for h in ha if h not in hb]
        only_b = [h for h in hb if h not in ha]
        res.fail("header mismatch (pysim-only: %s; dtalite-only: %s)"
                 % (only_a[:5], only_b[:5]))
        return res
    width = len(ha)
    norms = [norm_col(h) for h in ha]
    rows_a = [pad(r, width) for r in rows_a]
    rows_b = [pad(r, width) for r in rows_b]
    if len(rows_a) != len(rows_b):
        res.fail("row count differs: pysim %d, dtalite %d"
                 % (len(rows_a), len(rows_b)))
    # Join strategy: first column if unique, else (col0, col1) if unique,
    # else file order.
    mode = "ordered"
    pairs = []
    if width >= 1:
        ka = [r[0] for r in rows_a]
        kb = [r[0] for r in rows_b]
        if len(set(ka)) == len(ka) and len(set(kb)) == len(kb):
            mode = "join col0"
            da = {r[0]: r for r in rows_a}
            db = {r[0]: r for r in rows_b}
            only_a = sorted(set(da) - set(db))[:5]
            only_b = sorted(set(db) - set(da))[:5]
            if only_a or only_b:
                res.fail("key mismatch (pysim-only keys: %s; "
                         "dtalite-only keys: %s)" % (only_a, only_b))
            pairs = [(da[k], db[k]) for k in sorted(set(da) & set(db))]
        elif width >= 3:
            ka = [(r[0], r[1], r[2]) for r in rows_a]
            kb = [(r[0], r[1], r[2]) for r in rows_b]
            if len(set(ka)) == len(ka) and len(set(kb)) == len(kb):
                mode = "join col0+col1+col2"
                da = {(r[0], r[1], r[2]): r for r in rows_a}
                db = {(r[0], r[1], r[2]): r for r in rows_b}
                if set(da) != set(db):
                    res.fail("key mismatch: %d pysim-only, %d dtalite-only"
                             % (len(set(da) - set(db)),
                                len(set(db) - set(da))))
                pairs = [(da[k], db[k]) for k in sorted(set(da) & set(db))]
        if mode == "ordered" and width >= 2:
            ka = [(r[0], r[1]) for r in rows_a]
            kb = [(r[0], r[1]) for r in rows_b]
            if len(set(ka)) == len(ka) and len(set(kb)) == len(kb):
                mode = "join col0+col1"
                da = {(r[0], r[1]): r for r in rows_a}
                db = {(r[0], r[1]): r for r in rows_b}
                if set(da) != set(db):
                    res.fail("key mismatch: %d pysim-only, %d dtalite-only"
                             % (len(set(da) - set(db)),
                                len(set(db) - set(da))))
                pairs = [(da[k], db[k]) for k in sorted(set(da) & set(db))]
    if mode == "ordered":
        pairs = list(zip(rows_a, rows_b))
    res.notes.append("mode=%s rows pysim=%d dtalite=%d"
                     % (mode, len(rows_a), len(rows_b)))
    n_num_bad = 0
    max_diff = 0.0
    max_where = ""
    n_str_bad = 0
    str_examples = []
    n_path = 0
    n_path_bad = 0
    em_cols = {}
    for j, norm in enumerate(norms):
        if is_emission(norm):
            em_cols[j] = {"name": ha[j], "n": 0, "sum_a": 0.0, "sum_b": 0.0,
                          "rel": [], "zero_a": 0, "zero_b": 0, "zero_mis": 0}
    for ra, rb in pairs:
        for j in range(width):
            norm = norms[j]
            va, vb = ra[j], rb[j]
            if is_emission(norm) or ("mpr_table6" in label and j != 0):
                fa, fb = parse_float(va), parse_float(vb)
                if j not in em_cols:
                    em_cols[j] = {"name": ha[j], "n": 0, "sum_a": 0.0, "sum_b": 0.0,
                                  "rel": [], "zero_a": 0, "zero_b": 0, "zero_mis": 0}
                st = em_cols[j]
                st["n"] += 1
                if fa is not None and fb is not None:
                    st["sum_a"] += fa
                    st["sum_b"] += fb
                    if fa == 0:
                        st["zero_a"] += 1
                    if fb == 0:
                        st["zero_b"] += 1
                    if (fa == 0) != (fb == 0):
                        st["zero_mis"] += 1
                    elif fb != 0:
                        st["rel"].append(abs(fa - fb) / abs(fb))
                elif va.strip() != vb.strip():
                    st["zero_mis"] += 1
                continue
            if "time" in norm and "sequence" in norm:
                la = [parse_float(x) for x in va.split(";") if x.strip() != ""]
                lb = [parse_float(x) for x in vb.split(";") if x.strip() != ""]
                if len(la) == len(lb) and all(
                        x is not None and y is not None and abs(x - y) <= atol + rtol * abs(y)
                        for x, y in zip(la, lb)):
                    if is_path(norm):
                        n_path += 1
                    continue
                n_str_bad += 1
                if len(str_examples) < 5:
                    str_examples.append("key=%s col=%s pysim=%r dtalite=%r" % (ra[0], ha[j], va, vb))
                if is_path(norm):
                    n_path_bad += 1
                    n_path += 1
                continue
            fa, fb = parse_float(va), parse_float(vb)
            if fa is not None and fb is not None:
                d = abs(fa - fb)
                if d > atol + rtol * abs(fb):
                    n_num_bad += 1
                    if d > max_diff:
                        max_diff = d
                        max_where = "key=%s col=%s pysim=%s dtalite=%s" % (
                            ra[0], ha[j], va, vb)
            else:
                if va.strip() != vb.strip():
                    n_str_bad += 1
                    if len(str_examples) < 5:
                        str_examples.append(
                            "key=%s col=%s pysim=%r dtalite=%r"
                            % (ra[0], ha[j], va, vb))
                    if is_path(norm):
                        n_path_bad += 1
            if is_path(norm):
                n_path += 1
    if n_num_bad and "mpr_table6" not in label:
        res.fail("numeric mismatches: %d (max abs diff %.6g at %s)"
                 % (n_num_bad, max_diff, max_where))
    if n_num_bad and "mpr_table6" in label:
        res.notes.append("table6 numeric differences (report-only): %d cells, max %.6g at %s"
                         % (n_num_bad, max_diff, max_where))
    if n_str_bad:
        res.fail("string mismatches: %d (e.g. %s)"
                 % (n_str_bad, "; ".join(str_examples)))
    if n_path:
        res.notes.append("path cells: %d/%d exact"
                         % (n_path - n_path_bad, n_path))
    for j, st in em_cols.items():
        if st["n"] == 0:
            continue
        if st["sum_a"] == 0 and st["sum_b"] == 0:
            res.notes.append("emissions %s: n=%d, both sums zero "
                             "(rate file empty for this pollutant)"
                             % (st["name"], st["n"]))
            continue
        pct = 100.0 * (st["sum_a"] - st["sum_b"]) / st["sum_b"] \
            if st["sum_b"] != 0 else float("inf")
        rel = sorted(st["rel"])
        med = rel[len(rel) // 2] if rel else 0.0
        mean_rel = sum(rel) / len(rel) if rel else 0.0
        w10 = 100.0 * sum(1 for r in rel if r <= 0.10) / len(rel) if rel else 0.0
        w25 = 100.0 * sum(1 for r in rel if r <= 0.25) / len(rel) if rel else 0.0
        res.notes.append(
            "emissions %s: n=%d sum pysim=%.6g dtalite=%.6g "
            "sum diff=%+.1f%% mean rel=%.1f%% median rel=%.1f%% "
            "within10%%=%.1f%% within25%%=%.1f%% zero pysim=%d "
            "dtalite=%d mismatch=%d"
            % (st["name"], st["n"], st["sum_a"], st["sum_b"], pct,
               100.0 * mean_rel, 100.0 * med, w10, w25,
               st["zero_a"], st["zero_b"], st["zero_mis"]))
        if st["zero_mis"] > 0.01 * st["n"] and "_agent.csv" in label:
            res.fail("emissions zero-pattern mismatch in %s: %d/%d rows "
                     "(pysim zero=%d, dtalite zero=%d)"
                     % (st["name"], st["zero_mis"], st["n"],
                        st["zero_a"], st["zero_b"]))
    return res


def compare_text(pa, pb, label, roots):
    """Compare two text files with roots masked. Differences are WARN only."""
    res = FileResult(label)
    if not os.path.isfile(pa):
        res.warn("missing in pysim arm: %s" % pa)
        return res
    if not os.path.isfile(pb):
        res.warn("missing in dtalite arm: %s" % pb)
        return res
    try:
        with open(pa, "r", encoding="utf-8-sig", errors="replace") as f:
            ta = f.read()
        with open(pb, "r", encoding="utf-8-sig", errors="replace") as f:
            tb = f.read()
    except OSError as exc:
        res.warn("unreadable: %s" % exc)
        return res
    masked = []
    for text in (ta, tb):
        for root in roots:
            if root:
                text = text.replace(root, "<ROOT>")
        lines = [ln.rstrip() for ln in text.splitlines()]
        while lines and lines[-1] == "":
            lines.pop()
        masked.append(lines)
    la, lb = masked
    if la == lb:
        res.notes.append("identical (%d lines, roots masked)" % len(la))
        return res
    n_diff = 0
    examples = []
    for i in range(max(len(la), len(lb))):
        a = la[i] if i < len(la) else "<missing>"
        b = lb[i] if i < len(lb) else "<missing>"
        if a != b:
            n_diff += 1
            if len(examples) < 5:
                examples.append("line %d: pysim=%r dtalite=%r" % (i + 1, a, b))
    res.warn("%d differing lines (pysim %d lines, dtalite %d lines); "
             "e.g. %s" % (n_diff, len(la), len(lb), "; ".join(examples)))
    return res


def discover_test1_bigloop(full_dir):
    """Last bigloop number with both actual and UOinfor agent files."""
    try:
        names = os.listdir(full_dir)
    except OSError:
        return None
    found = {}
    for name in names:
        m = re.match(r"^(\d+)(UOinfor|predict|actual)_", name)
        if m:
            found.setdefault(int(m.group(1)), set()).add(m.group(2))
    cands = [b for b, tags in found.items()
             if "actual" in tags and "UOinfor" in tags]
    return max(cands) if cands else None


def compare_core9(dir_a, dir_b, n, label_prefix, atol, rtol):
    """Compare the 9 {n}{tag}_{kind}.csv files. Returns [FileResult]."""
    out = []
    for tag in TAGS:
        for kind in KINDS:
            fn = "%d%s_%s.csv" % (n, tag, kind)
            out.append(compare_csv(os.path.join(dir_a, fn),
                                   os.path.join(dir_b, fn),
                                   "%s/%s" % (label_prefix, fn),
                                   atol, rtol))
    return out


def list_names(directory, predicate):
    try:
        return sorted(n for n in os.listdir(directory) if predicate(n))
    except OSError:
        return []


class Reporter(object):
    def __init__(self):
        self.results = []
        self.checks = []

    def add(self, result):
        self.results.append(result)

    def check(self, ok, note):
        self.checks.append((ok, note))

    def n_fail(self):
        return sum(1 for r in self.results if r.verdict == "FAIL")

    def n_pass(self):
        return sum(1 for r in self.results if r.verdict == "PASS")


def section_test1(root_a, root_b, rep, atol, rtol):
    full_a = os.path.join(root_a, "test1", "full_results")
    full_b = os.path.join(root_b, "test1", "full_results")
    if not os.path.isdir(full_a) and not os.path.isdir(full_b):
        rep.check(True, "test1: skipped (no full_results in either arm)")
        return
    if not os.path.isdir(full_a) or not os.path.isdir(full_b):
        r = FileResult("test1/full_results")
        r.fail("present in only one arm")
        rep.add(r)
        return
    na = discover_test1_bigloop(full_a)
    nb = discover_test1_bigloop(full_b)
    if na is None or nb is None:
        r = FileResult("test1/full_results")
        r.fail("no completed bigloop (pysim=%s dtalite=%s)" % (na, nb))
        rep.add(r)
        return
    if na != nb:
        r = FileResult("test1/full_results")
        r.fail("converged bigloop differs: pysim %d, dtalite %d "
               "(file-level comparison needs matching bigloops)"
               % (na, nb))
        rep.add(r)
        return
    for r in compare_core9(full_a, full_b, na, "test1/full_results",
                           atol, rtol):
        rep.add(r)


def section_test234(t, root_a, root_b, rep, atol, rtol, roots):
    da = os.path.join(root_a, "test%d" % t)
    db = os.path.join(root_b, "test%d" % t)
    if not os.path.isdir(da) and not os.path.isdir(db):
        rep.check(True, "test%d: skipped (no results in either arm)" % t)
        return
    if not os.path.isdir(da) or not os.path.isdir(db):
        r = FileResult("test%d" % t)
        r.fail("present in only one arm")
        rep.add(r)
        return
    rep.add(compare_csv(os.path.join(da, "predicted_LinkMOE.csv"),
                        os.path.join(db, "predicted_LinkMOE.csv"),
                        "test%d/predicted_LinkMOE.csv" % t, atol, rtol))
    routes = sorted(set(list_names(da, lambda n: n.endswith("_routes.csv")))
                    | set(list_names(db, lambda n: n.endswith("_routes.csv"))))
    for fn in routes:
        rep.add(compare_csv(os.path.join(da, fn), os.path.join(db, fn),
                            "test%d/%s" % (t, fn), atol, rtol))
    summaries = sorted(set(list_names(da, lambda n: n.endswith("_summary.txt")))
                       | set(list_names(db, lambda n: n.endswith("_summary.txt"))))
    for fn in summaries:
        rep.add(compare_text(os.path.join(da, fn), os.path.join(db, fn),
                             "test%d/%s" % (t, fn), roots))
    fa = os.path.join(da, "full_results")
    fb = os.path.join(db, "full_results")
    outs = sorted(set(list_names(fa, lambda n: n.startswith("output_")
                                  and n.endswith(".csv")))
                  | set(list_names(fb, lambda n: n.startswith("output_")
                                   and n.endswith(".csv"))))
    for fn in outs:
        pa, pb = os.path.join(fa, fn), os.path.join(fb, fn)
        if os.path.isfile(pa) and os.path.isfile(pb):
            rep.add(compare_csv(pa, pb, "test%d/full_results/%s" % (t, fn),
                                atol, rtol))
        else:
            r = FileResult("test%d/full_results/%s" % (t, fn))
            r.warn("present in only one arm (PySim output subset)")
            rep.add(r)


def section_test5(root_a, root_b, rep, atol, rtol, roots):
    for sweep, lo in (("proposed_sweep", 0), ("ue_sweep", 1)):
        sa = os.path.join(root_a, "test5", sweep)
        sb = os.path.join(root_b, "test5", sweep)
        if not os.path.isdir(sa) and not os.path.isdir(sb):
            rep.check(True, "test5/%s: skipped (no sweep in either arm)"
                      % sweep)
            continue
        if not os.path.isdir(sa) or not os.path.isdir(sb):
            r = FileResult("test5/%s" % sweep)
            r.fail("present in only one arm")
            rep.add(r)
            continue
        pat = re.compile(r"^bigloop_(\d+)$")
        na = set(int(pat.match(n).group(1)) for n in os.listdir(sa)
                 if pat.match(n) and os.path.isdir(os.path.join(sa, n)))
        nb = set(int(pat.match(n).group(1)) for n in os.listdir(sb)
                 if pat.match(n) and os.path.isdir(os.path.join(sb, n)))
        for n in sorted(na | nb):
            if n not in na or n not in nb:
                r = FileResult("test5/%s/bigloop_%d" % (sweep, n))
                r.fail("bigloop dir present in only one arm")
                rep.add(r)
                continue
            for r in compare_core9(os.path.join(sa, "bigloop_%d" % n),
                                   os.path.join(sb, "bigloop_%d" % n),
                                   n, "test5/%s/bigloop_%d" % (sweep, n),
                                   atol, rtol):
                rep.add(r)
    for fn in ("mpr_table5.csv", "mpr_table6.csv"):
        rep.add(compare_csv(os.path.join(root_a, "test5", fn),
                            os.path.join(root_b, "test5", fn),
                            "test5/%s" % fn, atol, rtol))
    rep.add(compare_text(os.path.join(root_a, "test5", "mpr_summary.txt"),
                         os.path.join(root_b, "test5", "mpr_summary.txt"),
                         "test5/mpr_summary.txt", roots))


def section_test6(root_a, root_b, rep, atol, rtol, roots):
    da = os.path.join(root_a, "test6")
    db = os.path.join(root_b, "test6")
    if not os.path.isdir(da) and not os.path.isdir(db):
        rep.check(True, "test6: skipped (no results in either arm)")
        return
    if not os.path.isdir(da) or not os.path.isdir(db):
        r = FileResult("test6")
        r.fail("present in only one arm")
        rep.add(r)
        return
    rep.add(compare_csv(os.path.join(da, "volume_estimation.csv"),
                        os.path.join(db, "volume_estimation.csv"),
                        "test6/volume_estimation.csv", atol, rtol))
    rep.add(compare_text(os.path.join(da, "volume_estimation_summary.txt"),
                         os.path.join(db, "volume_estimation_summary.txt"),
                         "test6/volume_estimation_summary.txt", roots))


def check_engine_logs(root_a, root_b, rep):
    for sub in ("test1", "test2", "test3", "test4",
                os.path.join("test5", "proposed_sweep"),
                os.path.join("test5", "ue_sweep")):
        pa = os.path.join(root_a, sub, "engine_calls.log")
        pb = os.path.join(root_b, sub, "engine_calls.log")
        if not (os.path.isdir(os.path.join(root_a, sub))
                or os.path.isdir(os.path.join(root_b, sub))):
            continue
        if not os.path.isfile(pa):
            rep.check(False, "engine_calls.log MISSING in pysim arm: %s" % sub)
        if os.path.isfile(pb):
            rep.check(False, "engine_calls.log PRESENT in dtalite arm: %s"
                      % sub)


def parse_tests(spec):
    tests = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            tests.append(int(part))
    return sorted(set(t for t in tests if 1 <= t <= 6))


def render(rep, root_a, root_b, atol, rtol):
    lines = []
    lines.append("PySim vs DTALite comparison")
    lines.append("  pysim arm : %s" % root_a)
    lines.append("  dtalite arm: %s" % root_b)
    lines.append("  tolerance: atol=%g rtol=%g" % (atol, rtol))
    lines.append("")
    for r in rep.results:
        lines.append("[%s] %s" % (r.verdict, r.label))
        for note in r.notes:
            lines.append("         %s" % note)
    if rep.checks:
        lines.append("")
        lines.append("Checks (WARN only, do not flip the verdict):")
        for ok, note in rep.checks:
            lines.append("  [%s] %s" % ("ok" if ok else "WARN", note))
    n_fail = rep.n_fail()
    n_pass = rep.n_pass()
    lines.append("")
    lines.append("Files compared: %d, PASS: %d, FAIL: %d, checks: %d"
                 % (len(rep.results), n_pass, n_fail, len(rep.checks)))
    lines.append("OVERALL: %s" % ("PASS" if n_fail == 0 else "FAIL"))
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compare PySim vs DTALite wholerunner arms.")
    ap.add_argument("--pysim", default="results",
                    help="pysim results root (default: results)")
    ap.add_argument("--dtalite", default="results_dtalite",
                    help="dtalite results root (default: results_dtalite)")
    ap.add_argument("--tests", default="1,2,3,4,5,6",
                    help="comma-separated tests to compare")
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--report", default=None,
                    help="also write the report to this file")
    args = ap.parse_args(argv)
    root_a = os.path.abspath(args.pysim)
    root_b = os.path.abspath(args.dtalite)
    if not os.path.isdir(root_a):
        sys.stderr.write("error: pysim results dir not found: %s\n" % root_a)
        return 2
    if not os.path.isdir(root_b):
        sys.stderr.write("error: dtalite results dir not found: %s\n" % root_b)
        return 2
    roots = sorted({root_a, args.pysim, root_b, args.dtalite},
                   key=len, reverse=True)
    rep = Reporter()
    tests = parse_tests(args.tests)
    if 1 in tests:
        section_test1(root_a, root_b, rep, args.atol, args.rtol)
    for t in (2, 3, 4):
        if t in tests:
            section_test234(t, root_a, root_b, rep, args.atol, args.rtol,
                            roots)
    if 5 in tests:
        section_test5(root_a, root_b, rep, args.atol, args.rtol, roots)
    if 6 in tests:
        section_test6(root_a, root_b, rep, args.atol, args.rtol, roots)
    check_engine_logs(root_a, root_b, rep)
    text = render(rep, root_a, root_b, args.atol, args.rtol)
    sys.stdout.write(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    return 0 if rep.n_fail() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
