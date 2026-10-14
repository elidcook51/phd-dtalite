#!/usr/bin/env python3
"""Test 6: Future Traffic Conditions Estimation.

Compares predicted vs actual link volumes for the Proposed system (Test 1)
and bins the per-link volume discrepancy the way Bingrong's Chapter 4
Figure 8 does.

Inputs are DTALite LinkMOE CSVs. Their header row repeats
from_node_id,to_node_id (a trailing-comma quirk from the exe output), so
this script parses with the csv module and keeps the FIRST occurrence of
each duplicated header. pandas misaligns these files, so it is not used.

Read-only on inputs: nothing under --results-dir is ever written to,
except the designated --output-dir.
"""

import argparse
import csv
import glob
import os
import sys

PREDICT_SUFFIX = "predict_LinkMOE.csv"
ACTUAL_SUFFIX = "actual_LinkMOE.csv"

REQUIRED_COLUMNS = ("from_node_id", "to_node_id", "total_link_volume")

BINS = ("<10%", "10-20%", "20-50%", "50-100%", ">100%")
UNDEFINED_BIN = "undefined"

# Systems whose guidance-run / intermediate link volumes the pipeline does
# not persist, so no predicted-vs-actual comparison is possible for them.
SKIP_REASONS = {
    "realtime_1": "guidance-run link volumes overwritten, not persisted",
    "realtime_10": "guidance-run link volumes overwritten, not persisted",
    "ue_baseline": "final MSA iteration link flows overwritten, not persisted",
}


def warn(msg):
    sys.stderr.write("WARNING: " + msg + "\n")


def info(msg):
    sys.stdout.write(msg + "\n")


def load_link_volumes(path):
    """Load {(from_node_id, to_node_id): (volume, raw_text)} from a LinkMOE CSV.

    Handles the trailing-comma quirk: duplicated headers keep their first
    occurrence, so extra trailing fields are ignored. Malformed rows are
    skipped with a warning.
    """
    volumes = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        col = {}
        for row in reader:
            if not col:
                for i, name in enumerate(row):
                    name = name.strip()
                    if name and name not in col:
                        col[name] = i
                missing = [c for c in REQUIRED_COLUMNS if c not in col]
                if missing:
                    raise ValueError(
                        "%s: missing required column(s): %s"
                        % (path, ", ".join(missing))
                    )
                continue
            if not row or not any(cell.strip() for cell in row):
                continue
            try:
                from_node = row[col["from_node_id"]].strip()
                to_node = row[col["to_node_id"]].strip()
                raw_volume = row[col["total_link_volume"]].strip()
                volume = float(raw_volume)
            except (IndexError, ValueError):
                warn("skipping malformed row in %s" % path)
                continue
            if not from_node or not to_node:
                warn("skipping row with blank node id in %s" % path)
                continue
            volumes[(from_node, to_node)] = (volume, raw_volume)
    return volumes


def find_proposed_pair(full_output_dir):
    """Find the (prefix, predict_path, actual_path) pair to analyze.

    The bigloop number is the filename prefix; the predict and actual files
    must share it. When several bigloops are present, the highest number
    wins. Returns None when no complete pair exists.
    """
    by_prefix = {}
    for path in glob.glob(os.path.join(full_output_dir, "*" + PREDICT_SUFFIX)):
        prefix = os.path.basename(path)[: -len(PREDICT_SUFFIX)]
        by_prefix.setdefault(prefix, {})["predict"] = path
    for path in glob.glob(os.path.join(full_output_dir, "*" + ACTUAL_SUFFIX)):
        prefix = os.path.basename(path)[: -len(ACTUAL_SUFFIX)]
        by_prefix.setdefault(prefix, {})["actual"] = path

    complete = [p for p, d in by_prefix.items()
                if "predict" in d and "actual" in d]
    if not complete:
        return None

    def rank(prefix):
        return (0, int(prefix)) if prefix.isdigit() else (1, prefix)

    best = max(complete, key=rank)
    return best, by_prefix[best]["predict"], by_prefix[best]["actual"]


def bin_for(discrepancy_pct):
    if discrepancy_pct < 10.0:
        return "<10%"
    if discrepancy_pct < 20.0:
        return "10-20%"
    if discrepancy_pct < 50.0:
        return "20-50%"
    if discrepancy_pct < 100.0:
        return "50-100%"
    return ">100%"


def analyze_proposed(results_dir):
    """Analyze the Proposed system. Returns a result dict, or None if skipped."""
    full_output = os.path.join(results_dir, "test1", "full_results")
    if not os.path.isdir(full_output):
        warn("proposed: directory not found: %s -- skipping" % full_output)
        return None

    found = find_proposed_pair(full_output)
    if found is None:
        has_predict = glob.glob(os.path.join(full_output, "*" + PREDICT_SUFFIX))
        has_actual = glob.glob(os.path.join(full_output, "*" + ACTUAL_SUFFIX))
        if has_predict and not has_actual:
            warn("proposed: found *%s but no matching *%s -- skipping"
                 % (PREDICT_SUFFIX, ACTUAL_SUFFIX))
        elif has_actual and not has_predict:
            warn("proposed: found *%s but no matching *%s -- skipping"
                 % (ACTUAL_SUFFIX, PREDICT_SUFFIX))
        else:
            warn("proposed: no *%s / *%s files in %s -- skipping"
                 % (PREDICT_SUFFIX, ACTUAL_SUFFIX, full_output))
        return None

    prefix, predict_path, actual_path = found
    info("proposed: using %s and %s"
         % (os.path.basename(predict_path), os.path.basename(actual_path)))

    predicted = load_link_volumes(predict_path)
    actual = load_link_volumes(actual_path)

    common = set(predicted) & set(actual)
    for key in sorted(set(predicted) - set(actual)):
        warn("proposed: link %s->%s in predicted but not in actual -- dropped" % key)
    for key in sorted(set(actual) - set(predicted)):
        warn("proposed: link %s->%s in actual but not in predicted -- dropped" % key)

    rows = []
    bin_counts = {b: 0 for b in BINS}
    undefined = 0
    for key in sorted(common):
        pred_vol, pred_raw = predicted[key]
        act_vol, act_raw = actual[key]
        if act_vol == 0.0:
            rows.append((key[0], key[1], pred_raw, act_raw, "", UNDEFINED_BIN))
            undefined += 1
            continue
        disc = abs(pred_vol - act_vol) / abs(act_vol) * 100.0
        b = bin_for(disc)
        bin_counts[b] += 1
        rows.append((key[0], key[1], pred_raw, act_raw, "%.2f" % disc, b))

    return {
        "system": "test1",
        "bigloop": prefix if prefix else None,
        "rows": rows,
        "bins": bin_counts,
        "undefined": undefined,
        "total": len(rows),
    }


def write_csv(path, analyzed):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)  # csv module default: CRLF line terminator
        writer.writerow(["system", "from_node_id", "to_node_id",
                         "predicted_volume", "actual_volume",
                         "discrepancy_pct", "bin"])
        for res in analyzed:
            for from_node, to_node, pred_raw, act_raw, disc, b in res["rows"]:
                writer.writerow([res["system"], from_node, to_node,
                                 pred_raw, act_raw, disc, b])


def write_summary(path, analyzed, skipped):
    lines = []
    lines.append("Future Traffic Conditions Estimation (Test 6)")
    lines.append("=============================================")
    lines.append("")
    for res in analyzed:
        label = res["system"]
        if res["bigloop"]:
            label += " (bigloop %s)" % res["bigloop"]
        lines.append("   System: %s" % label)
        lines.append("     Links analyzed: %d" % res["total"])
        for b in BINS:
            lines.append("     %-8s: %d links" % (b, res["bins"][b]))
        lines.append("     Undefined (zero actual volume): %d links"
                     % res["undefined"])
        lines.append("")
    if skipped:
        lines.append("   Systems skipped (predicted volumes not saved):")
        for system, reason in skipped:
            lines.append("     - %s: %s" % (system, reason))
        lines.append("")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("\r\n".join(lines))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Test 6: per-link volume discrepancy between predicted "
                    "and actual link volumes (Bingrong Ch.4 Figure 8).")
    parser.add_argument("--results-dir", default="results",
                        help="root containing test1/, test2/, ... "
                             "(default: results)")
    parser.add_argument("--output-dir", default=None,
                        help="where to write outputs "
                             "(default: <results-dir>/volume_estimation)")
    parser.add_argument("--systems",
                        default="test1,test2,test3,test4",
                        help="comma-separated systems to analyze")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        sys.stderr.write("ERROR: results dir not found: %s\n" % results_dir)
        return 1
    output_dir = args.output_dir or os.path.join(results_dir, "volume_estimation")
    os.makedirs(output_dir, exist_ok=True)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    analyzed = []
    skipped = []
    for system in systems:
        if system in ("proposed", "test1"):  # "proposed" kept as a standalone alias
            res = analyze_proposed(results_dir)
            if res is None:
                skipped.append((system, "predicted/actual LinkMOE pair not found"))
            else:
                analyzed.append(res)
        else:
            reason = SKIP_REASONS.get(system, "predicted link volumes not available")
            warn("Predicted link volumes not saved for %s -- skipping. (%s.)"
                 % (system, reason))
            skipped.append((system, reason))

    csv_path = os.path.join(output_dir, "volume_estimation.csv")
    summary_path = os.path.join(output_dir, "volume_estimation_summary.txt")
    write_csv(csv_path, analyzed)
    write_summary(summary_path, analyzed, skipped)
    info("wrote %s" % csv_path)
    info("wrote %s" % summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
