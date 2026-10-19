"""Baltimore input data auditor for DTALite 50GB memory issue.

Checks baltimore_files/ for the data-integrity problems the converter
(build_baltimore_inputs.py) does NOT validate, any of which can make
DTALite allocate excessive memory or accumulate it during simulation:

  1. Non-positive / NaN link fields (length, free_speed, free_flow_time,
     number_of_lanes, lane_capacity_in_vhc_per_hour). Zero-capacity or
     zero-speed links cause unbounded queue growth -> gradual RAM climb.
  2. Link endpoints referencing node IDs missing from input_node.csv
     (DTALite creates phantom nodes).
  3. link_type IDs in input_link.csv missing from input_link_type.csv
     (Halbert's file may not define all types the source uses).
  4. Agent departure times outside the 1020-1080 simulation window.
  5. Agent origin/destination nodes missing from the network.
  6. Zone-count consistency: input_zone.csv vs input_activity_location.csv
     vs nodes with zone_id > 0 in input_node.csv (virtual-connector blowup
     if thousands of nodes are zoned).

Usage (on Eli's Windows machine, from the repo root):
    python baltimore_data_audit.py --dir baltimore_files

Only reads; never modifies. Prints a PASS/FAIL summary per check.
"""
import argparse
import csv
import os
import sys


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def is_bad_number(s):
    """True if s is NaN, non-numeric, or <= 0."""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return True
    return not (v > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="baltimore_files",
                    help="folder with DTALite input files")
    ap.add_argument("--window", default="1020,1080",
                    help="demand window start,end in minutes")
    a = ap.parse_args()
    d = os.path.abspath(a.dir)
    w0, w1 = (int(x) for x in a.window.split(","))

    failures = []

    def check(name, ok, detail=""):
        print(("[PASS] " if ok else "[FAIL] ") + name
              + (" - " + detail if detail else ""))
        if not ok:
            failures.append(name)

    # ---- nodes ---------------------------------------------------------
    node_path = os.path.join(d, "input_node.csv")
    nodes = load_csv(node_path)
    node_ids = set()
    zoned_nodes = 0
    for r in nodes:
        try:
            nid = int(float(r["node_id"]))
        except (ValueError, KeyError, TypeError):
            continue
        node_ids.add(nid)
        try:
            if int(float(r.get("zone_id", 0))) > 0:
                zoned_nodes += 1
        except (ValueError, TypeError):
            pass
    print("nodes: %d, unique ids: %d, zoned (zone_id>0): %d"
          % (len(nodes), len(node_ids), zoned_nodes))
    check("node_ids_unique", len(nodes) == len(node_ids),
          "%d duplicate node rows" % (len(nodes) - len(node_ids)))
    check("zoned_nodes_reasonable", zoned_nodes <= 1000,
          "%d nodes have zone_id>0 (expect ~405 centroids; thousands => "
          "DTALite creates 2 virtual links EACH)" % zoned_nodes)

    # ---- links ---------------------------------------------------------
    link_path = os.path.join(d, "input_link.csv")
    links = load_csv(link_path)
    num_fields = ["length", "free_speed", "free_flow_time",
                  "number_of_lanes", "lane_capacity_in_vhc_per_hour"]
    bad_counts = {f: 0 for f in num_fields}
    missing_field = {f: 0 for f in num_fields}
    dangling = 0
    type_ids = set()
    n_links = 0
    for r in links:
        n_links += 1
        for f in num_fields:
            if f not in r:
                missing_field[f] += 1
            elif is_bad_number(r[f]):
                bad_counts[f] += 1
        try:
            u = int(float(r["from_node_id"]))
            v = int(float(r["to_node_id"]))
        except (ValueError, KeyError, TypeError):
            dangling += 1
            continue
        if u not in node_ids or v not in node_ids:
            dangling += 1
        try:
            type_ids.add(int(float(r["link_type"])))
        except (ValueError, KeyError, TypeError):
            pass
    print("links: %d" % n_links)
    for f in num_fields:
        if missing_field[f]:
            check("link_col_%s_present" % f, False,
                  "column missing in %d rows" % missing_field[f])
        else:
            check("link_%s_positive" % f, bad_counts[f] == 0,
                  "%d rows with NaN/<=0 (zero capacity/speed => unbounded "
                  "queues)" % bad_counts[f])
    check("link_endpoints_exist", dangling == 0,
          "%d links reference nodes missing from input_node.csv "
          "(phantom nodes)" % dangling)

    # ---- link types vs Halbert's file ----------------------------------
    lt_path = os.path.join(d, "input_link_type.csv")
    if os.path.isfile(lt_path):
        defined = set()
        for r in load_csv(lt_path):
            try:
                defined.add(int(float(r["link_type"])))
            except (ValueError, KeyError, TypeError):
                pass
        undef = sorted(type_ids - defined)
        check("link_types_defined", not undef,
              "link_type ids used but not defined: %s" % undef)
    else:
        check("link_type_file_present", False, lt_path + " not found")

    # ---- agents --------------------------------------------------------
    ag_path = os.path.join(d, "input_agent.csv")
    if os.path.isfile(ag_path):
        agents = load_csv(ag_path)
        cols = set(agents[0].keys()) if agents else set()
        need = {"from_origin_node_id", "to_destination_node_id"}
        check("agent_od_columns", need <= cols,
              "missing: %s" % sorted(need - cols))
        # departure time column: try common names
        dep_col = next((c for c in
                        ["departure_time_in_min", "departure_time",
                         "departure_time_min"] if c in cols), None)
        out_of_window = 0
        bad_dep = 0
        bad_od = 0
        for r in agents:
            if dep_col:
                try:
                    t = float(r[dep_col])
                    if not (w0 <= t <= w1):
                        out_of_window += 1
                except (ValueError, TypeError):
                    bad_dep += 1
            for c in ["from_origin_node_id", "to_destination_node_id"]:
                if c in r:
                    try:
                        if int(float(r[c])) not in node_ids:
                            bad_od += 1
                            break
                    except (ValueError, TypeError):
                        bad_od += 1
                        break
        print("agents: %d" % len(agents))
        if dep_col:
            check("agent_departures_in_window", out_of_window == 0,
                  "%d agents depart outside %d-%d" % (out_of_window, w0, w1))
            check("agent_departures_numeric", bad_dep == 0,
                  "%d agents with non-numeric departure" % bad_dep)
        else:
            check("agent_departure_column", False,
                  "no departure-time column found; columns: %s"
                  % sorted(cols))
        check("agent_od_nodes_exist", bad_od == 0,
              "%d agents reference nodes missing from input_node.csv"
              % bad_od)
    else:
        check("agent_file_present", False, ag_path + " not found")

    # ---- zones ---------------------------------------------------------
    z_path = os.path.join(d, "input_zone.csv")
    al_path = os.path.join(d, "input_activity_location.csv")
    n_zones = len(load_csv(z_path)) if os.path.isfile(z_path) else -1
    al_rows = load_csv(al_path) if os.path.isfile(al_path) else []
    al_nodes = set()
    for r in al_rows:
        try:
            al_nodes.add(int(float(r["node_id"])))
        except (ValueError, KeyError, TypeError):
            pass
    print("zones: input_zone.csv=%d, input_activity_location.csv=%d, "
          "zoned nodes=%d" % (n_zones, len(al_rows), zoned_nodes))
    check("zone_counts_consistent",
          n_zones == len(al_rows) == zoned_nodes,
          "mismatch => DTALite zone/virtual-link structures inconsistent")
    # activity nodes must exist and (ideally) be the zoned centroids
    missing_al = [n for n in al_nodes if n not in node_ids]
    check("activity_nodes_exist", not missing_al,
          "%d activity nodes missing from input_node.csv" % len(missing_al))

    print()
    if failures:
        print("RESULT: %d FAILING CHECK(S): %s"
              % (len(failures), ", ".join(failures)))
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
