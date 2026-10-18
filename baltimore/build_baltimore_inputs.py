# build_baltimore_inputs.py -- Phase 1 of the Baltimore migration.
#
# Assembles a self-contained baltimore_files/ folder (the Baltimore analog of
# DTALite_Files/) from Eli's Baltimore source files, in the shapes the
# DTALite + Python pipeline already understands.
#
# Run from the repo root on the remote machine:
#   python baltimore/build_baltimore_inputs.py --src <Baltimore source folder>
#
#   --src DIR            folder holding the Baltimore source files (required)
#   --out DIR            output folder (default: ./baltimore_files)
#   --dtalite-dir DIR    folder to copy DTALite.exe + support files from
#                        (default: ./DTALite_Files)
#   --n N                only take the first N internal agents (smoke tests)
#   --seed S             RNG seed for weight assignment (default 42)
#   --with-background    concatenate EE + warmup agents into input_agent.csv
#                        (frozen background traffic; needs the full pipeline to
#                        actually use them -- off by default for now)
#
# Requires: h5py  (pip install h5py)
#
# Source files needed in --src:
#   input_agent_peak_405.csv    276,757 internal agents, 5-6 PM, no paths
#   input_link.csv              297,732 links, noKB network (bridge removed)
#   input_node.csv              133,080 nodes, 405 zones
#   choice_set_noKB_k5_v3.mat   v7.3: finallist 5x23016 cells, routelocation
#   input_agent_ee_final.csv    29,200 EE agents (only with --with-background)
#   input_agent_warmup.csv      275,372 warmup agents (only with --with-background)
#
# Output (baltimore_files/):
#   input_agent.csv, input_link.csv, input_node.csv, input_zone.csv,
#   choice_sets.pkl     {(origin_zone, dest_zone): [[node ids], ...]}  (k=5)
#   agent_weights.csv   agent_id,pref  (pref = 1..33300 weight-pool index)
#   baltimore_config.json
#   DTALite.exe, DTASettings.txt, support + emission CSVs (from --dtalite-dir)

import argparse
import csv
import json
import os
import pickle
import random
import shutil
import sys

WEIGHT_POOL_SIZE = 33300
TIME_WINDOW = [1020, 1080]  # 5:00 PM - 6:00 PM, minutes since midnight
KB_LINK_IDS = [285974, 285975, 285976, 285977, 299079, 299080, 299081]

# DTALite support files copied from the Sioux Falls folder (network-agnostic).
DTALITE_SUPPORT_FILES = [
    "DTALite.exe",
    # DTASettings.txt is written directly (Halbert's Baltimore settings),
    # not copied from Sioux Falls.
    "input_link_type.csv",
    "input_node_control_type.csv",
    "input_movement.csv",  # header-only in Sioux Falls; safe to copy
    "input_scenario_settings.csv",
    "input_demand.csv",
    "input_demand_type.csv",
    "input_demand_file_list.csv",
    "optional_MOE_settings.csv",
    "optional_vehicle_emission_rate.csv",
    "optional_vehicle_type.csv",
    # Preference-profile pool (33,300 synthetic profiles, Sun Dataset 1).
    # Reused for Baltimore: no Baltimore-specific weights were made.
    "SiouxFalls_net.xlsx",
]


def log(s=""):
    print(s, flush=True)


def read_choice_set_v73(mat_path):
    """Read a v7.3 (HDF5) choice-set .mat.

    MATLAB layout:  finallist  5 x nOD cell, finallist{k,i} = ';'-joined
                    node-id string for OD i, path k
                    routelocation nOD x 4 = [origin_zone, dest_zone, 1, i]
    Returns ({(oz, dz): [[int, ...], ...]}, n_ods).
    """
    import h5py
    import numpy as np

    def decode_chars(obj):
        arr = np.array(obj).flatten()
        return "".join(chr(int(c)) for c in arr)

    choice = {}
    with h5py.File(mat_path, "r") as f:
        if "finallist" not in f or "routelocation" not in f:
            raise ValueError("expected 'finallist' and 'routelocation', got: %s"
                             % sorted(f.keys()))
        fin = f["finallist"]          # h5py shape (nOD, 5) == MATLAB (5, nOD)
        route = np.array(f["routelocation"]).T  # -> (nOD, 4)
        n_od = fin.shape[0]
        k = fin.shape[1]
        n_empty = 0
        for i in range(n_od):
            oz, dz = int(route[i, 0]), int(route[i, 1])
            paths = []
            for kk in range(k):
                try:
                    obj = f[fin[i, kk]]
                except Exception:
                    n_empty += 1
                    continue
                s = decode_chars(obj)
                nodes = [int(x) for x in s.split(";") if x.strip()]
                if nodes:
                    paths.append(nodes)
                else:
                    n_empty += 1
            if paths:
                choice[(oz, dz)] = paths
    if n_empty:
        log("  note: %d empty path cells skipped" % n_empty)
    return choice, n_od


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="folder holding the Baltimore source files")
    ap.add_argument("--out", default="baltimore_files",
                    help="output folder (default: ./baltimore_files)")
    ap.add_argument("--dtalite-dir", default="DTALite_Files",
                    help="folder to copy DTALite.exe + support files from")
    ap.add_argument("--n", type=int, default=None,
                    help="only take the first N internal agents (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-background", action="store_true",
                    help="append EE + warmup agents to input_agent.csv")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    out = os.path.abspath(a.out)
    dtalite_dir = os.path.abspath(a.dtalite_dir)
    need = ["input_agent_peak_405.csv", "input_link.csv", "input_node.csv",
            "choice_set_noKB_k5_v3.mat"]
    if a.with_background:
        need += ["input_agent_ee_final.csv", "input_agent_warmup.csv"]
    for fn in need:
        if not os.path.isfile(os.path.join(src, fn)):
            sys.exit("missing input: %s" % os.path.join(src, fn))
    try:
        import h5py  # noqa: F401
    except ImportError:
        sys.exit("h5py is required: pip install h5py")

    os.makedirs(out, exist_ok=True)
    log("=" * 70)
    log("BUILD BALTIMORE INPUTS")
    log("src:         " + src)
    log("dtalite dir: " + dtalite_dir)
    log("out:         " + out)
    log("=" * 70)

    # ---- 1. agents ----------------------------------------------------
    log("\n[1/6] agents ...")
    agent_src = os.path.join(src, "input_agent_peak_405.csv")
    agent_rows = []
    with open(agent_src, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames
        for row in reader:
            agent_rows.append(row)
            if a.n is not None and len(agent_rows) >= a.n:
                break
    log("  internal agents: %d%s" % (len(agent_rows),
                                    " (subset)" if a.n else ""))
    if a.with_background:
        for fn in ["input_agent_ee_final.csv", "input_agent_warmup.csv"]:
            with open(os.path.join(src, fn), newline="", encoding="utf-8-sig") as fh:
                r2 = list(csv.DictReader(fh))
            agent_rows.extend(r2)
            log("  + %s: %d" % (fn, len(r2)))
    with open(os.path.join(out, "input_agent.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(agent_rows)
    log("  wrote input_agent.csv (%d rows)" % len(agent_rows))

    # ---- 2. network files (copied as-is; already DTALite-native) -------
    log("\n[2/6] network files ...")
    n_links = 0
    kb_found = set()
    with open(os.path.join(src, "input_link.csv"), newline="",
              encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            n_links += 1
            try:
                lid = int(float(row["link_id"]))
            except (ValueError, KeyError):
                continue
            if lid in KB_LINK_IDS:
                kb_found.add(lid)
    for fn in ["input_link.csv", "input_node.csv"]:
        shutil.copy2(os.path.join(src, fn), os.path.join(out, fn))
    log("  links: %d" % n_links)
    log("  Key Bridge links present: %s"
        % (sorted(kb_found) if kb_found else "NONE (noKB network, as expected)"))

    # DTALite expects input_zone.csv; build it from the node file's zones
    log("  building input_zone.csv ...")
    zones = set()
    with open(os.path.join(src, "input_node.csv"), newline="",
              encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                z = int(float(row["zone_id"]))
            except (ValueError, KeyError):
                continue
            if z > 0:
                zones.add(z)
    with open(os.path.join(out, "input_zone.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["zone_id", "geometry"])
        for z in sorted(zones):
            w.writerow([z, ""])
    log("  zones: %d" % len(zones))

    # Build Baltimore-compatible input_activity_location.csv (zone -> node).
    # The Sioux Falls version references wrong node IDs; DTALite validates
    # these, so we generate from the Baltimore node file instead.
    log("  building input_activity_location.csv ...")
    # DTALite requires activity nodes to have link connectivity:
    # destination nodes need incoming links, origin nodes need outgoing.
    # Build sets from input_link.csv, then pick the first zone node
    # that has BOTH (or set external_od_flag if only one direction).
    _has_in = set()
    _has_out = set()
    with open(os.path.join(src, "input_link.csv"), newline="",
              encoding="utf-8-sig") as _fh:
        for _row in csv.DictReader(_fh):
            try:
                _u = int(float(_row["from_node_id"]))
                _v = int(float(_row["to_node_id"]))
            except (ValueError, KeyError):
                continue
            _has_out.add(_u)
            _has_in.add(_v)
    zone_node = {}  # z -> (node_id, flag)
    with open(os.path.join(src, "input_node.csv"), newline="",
              encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                z = int(float(row["zone_id"]))
                n = int(float(row["node_id"]))
            except (ValueError, KeyError):
                continue
            if z <= 0 or z in zone_node:
                continue
            _in_ok = n in _has_in
            _out_ok = n in _has_out
            if _in_ok and _out_ok:
                zone_node[z] = (n, "")
            elif _in_ok:
                zone_node[z] = (n, "1")   # destination only
            elif _out_ok:
                zone_node[z] = (n, "-1")  # origin only
            # else: skip nodes with no links at all
    # Fallback: if a zone has no connected node, use first node anyway
    # (DTALite will warn but may still run)
    with open(os.path.join(out, "input_activity_location.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["zone_id", "node_id", "external_OD_flag"])
        for z in sorted(zone_node):
            n, flag = zone_node[z]
            w.writerow([z, n, flag])
    log("  activity locations: %d" % len(zone_node))
    # ---- 6c. Clear zone_id for non-centroid nodes --------------------
    # DTALite (main_api.cpp) creates 2 virtual connector links for every
    # node with zone_id >= 1. The source input_node.csv has ~66k nodes
    # with zone_id set, causing 133k virtual links (430k total) and GBs
    # of wasted memory. Keep zone_id only on the 405 centroid nodes.
    _centroids = set(n for n, _flag in zone_node.values())
    _node_path = os.path.join(out, "input_node.csv")
    _rows = []
    with open(_node_path, newline="", encoding="utf-8-sig") as _fh:
        _reader = csv.DictReader(_fh)
        _fields = _reader.fieldnames
        for _row in _reader:
            try:
                _nid = int(float(_row["node_id"]))
            except (ValueError, KeyError):
                _rows.append(_row)
                continue
            if _nid not in _centroids:
                if "zone_id" in _row:
                    _row["zone_id"] = "0"
            _rows.append(_row)
    with open(_node_path, "w", newline="", encoding="utf-8") as _fh:
        _w = csv.DictWriter(_fh, fieldnames=_fields)
        _w.writeheader()
        _w.writerows(_rows)
    log("  cleared zone_id for non-centroid nodes (kept %d)" % len(_centroids))

    # ---- 3. choice sets (.mat v7.3 -> pickle) ---------------------------
    log("\n[3/6] choice sets ...")
    choice, n_od = read_choice_set_v73(
        os.path.join(src, "choice_set_noKB_k5_v3.mat"))
    k_vals = {len(v) for v in choice.values()}
    log("  OD pairs in .mat: %d, kept: %d, k values: %s"
        % (n_od, len(choice), sorted(k_vals)))
    with open(os.path.join(out, "choice_sets.pkl"), "wb") as fh:
        pickle.dump(choice, fh, protocol=4)
    log("  wrote choice_sets.pkl")

    # coverage check: agent ODs missing from the choice set
    od_needed = {(int(float(r["from_zone_id"])), int(float(r["to_zone_id"])))
                 for r in agent_rows
                 if r["from_zone_id"] != r["to_zone_id"]}
    missing = [od for od in od_needed if od not in choice]
    log("  agent ODs without a choice set: %d (of %d inter-zonal)"
        % (len(missing), len(od_needed)))

    # ---- 4. per-agent preference weights -------------------------------
    log("\n[4/6] preference weights ...")
    rng = random.Random(a.seed)
    with open(os.path.join(out, "agent_weights.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["agent_id", "pref"])
        for r in agent_rows:
            w.writerow([r["agent_id"], rng.randint(1, WEIGHT_POOL_SIZE)])
    log("  wrote agent_weights.csv (pool 1..%d, seed %d)"
        % (WEIGHT_POOL_SIZE, a.seed))

    # ---- 5. config ------------------------------------------------------
    log("\n[5/6] config ...")
    cfg = {
        "network": "baltimore",
        "bridge": "none (noKB)",
        "time_window_min": TIME_WINDOW,
        "n_internal_agents": len([r for r in agent_rows
                                  if int(float(r["agent_id"])) < 60000000]),
        "n_choice_set_ods": len(choice),
        "choice_set_k": sorted(k_vals),
        "weight_pool_size": WEIGHT_POOL_SIZE,
        "weight_seed": a.seed,
        "subset_n": a.n,
        "with_background": a.with_background,
        "files": {
            "agents": "input_agent.csv",
            "links": "input_link.csv",
            "nodes": "input_node.csv",
            "zones": "input_zone.csv",
            "choice_sets": "choice_sets.pkl",
            "weights": "agent_weights.csv",
        },
    }
    with open(os.path.join(out, "baltimore_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    log("  wrote baltimore_config.json")

    # ---- 6. DTALite support files --------------------------------------
    log("\n[6/6] DTALite support files ...")
    if not os.path.isdir(dtalite_dir):
        log("  WARNING: not found, skipped: " + dtalite_dir)
    else:
        n_copied = 0
        # Files that should NOT be overwritten if already present
        # (e.g., Halbert's Baltimore-specific versions).
        PRESERVE_IF_EXISTS = {
            "input_link_type.csv",
            "input_node_control_type.csv",
            "input_movement.csv",
            "input_scenario_settings.csv",
        }
        for fn in DTALITE_SUPPORT_FILES:
            dest = os.path.join(out, fn)
            if fn in PRESERVE_IF_EXISTS and os.path.isfile(dest):
                log("  preserving existing: " + fn)
                n_copied += 1
                continue
            sp = os.path.join(dtalite_dir, fn)
            if os.path.isfile(sp):
                shutil.copy2(sp, dest)
                n_copied += 1
            else:
                log("  WARNING: not found, skipped: " + fn)
        log("  copied %d/%d support files" % (n_copied,
                                             len(DTALITE_SUPPORT_FILES)))

    # ---- 6b. DTASettings.txt: Halbert's Baltimore settings -----------
    # Write the full DTASettings.txt directly (do NOT copy from Sioux Falls).
    # Based on Halbert et al. WSC 2026 Baltimore configuration:
    # - 300-min TDMOE aggregation (not 1-min; avoids 45GB memory blowup)
    # - emissions off (not needed for MSA)
    # - 32 threads
    # - 6-second simulation intervals
    # Demand window 1020-1080 for Baltimore (vs Halbert's 900-1200).
    _dta = os.path.join(out, "DTASettings.txt")
    _halbert_settings = """[shortest_path]
max_size_of_adjacent_links=20
[simulation]
planning_horizon=180
simulation_start_time_in_min=1020
simulation_end_time_in_min=1080
number_of_seconds_per_interval=6
use_point_queue_for_on_ramps=1
use_point_queue_for_off_ramps=1
default_saturation_flow_rate_in_vehphpl=1800.00
demand_capcaity_scaling_factor= 1.00
[assignment]
agent_based_assignment=1
number_of_iterations=1
aggregation_time_interval_in_min=15
convergency_relative_gap_threshold_percentage= 5.00
travel_time_difference_for_switching_in_min= 1.00
relative_travel_time_difference_in_percentage_for_switching_in_percentage=15.00
use_link_travel_time_from_external_input=0
use_link_travel_time_from_external_input_gain_factor= 0.50
[output]
TDLinkMOE_aggregation_time_interval_in_min=300
time_dependent_RoutingPolicy_file=0
time_dependent_ODMOE_file=0
time_dependent_ODMOE_aggregation_time_interval=30
[computation]
max_number_of_threads_to_be_used=32
max_number_of_threads_for_real_time_routing=1
[emission]
output_emission_data=0
target_vehicle_id_for_output_second_by_second_emission_data=0
[traveler_information]
information_updating_interval_in_min=1
[ABM_integration]
RT_Input_Agent_Frequency_in_Seconds=0
RT_Input_Routing_Policy_Frequency_in_Seconds=0
RT_Output_LinkMOE_Frequency_in_Seconds=0
RT_Output_TDODMOE_in_Seconds=0
RT_Output_Complete_Agent_Frequency_in_Seconds=0
RT_Output_Tagged_Agent_Frequency_in_Seconds=0
dependency_agent_reading_percentage=0
day_by_day_output=0
day_by_day_output_starting_iteration=100
[agent_input]
demand_loading_start_time_in_min=1020
demand_loading_end_time_in_min=1080
"""
    with open(_dta, "w", encoding="utf-8", newline="\n") as _fh:
        _fh.write(_halbert_settings)
    log("  wrote DTASettings.txt (Halbert Baltimore settings)")

    # input_demand_file_list.csv: the agent file's loading window must cover
    # Baltimore departures (1020-1080), not Sioux Falls (360-420).
    _dfl = os.path.join(out, "input_demand_file_list.csv")
    if os.path.isfile(_dfl):
        _rows = open(_dfl, encoding="utf-8").read().splitlines()
        _out = []
        for _r in _rows:
            _c = _r.split(",")
            if len(_c) > 8 and _c[2].strip() == "input_agent.csv":
                _c[6], _c[7] = "1020", "1080"
                _r = ",".join(_c)
            _out.append(_r)
        open(_dfl, "w", encoding="utf-8").write("\n".join(_out) + "\n")
        log("  patched input_demand_file_list.csv window to 1020-1080")

    # ---- 7. input_agent_initial.csv ------------------------------------
    # The pipeline reads input_agent_initial.csv as the iteration-0 demand
    # and writes per-iteration input_agent.csv from it.
    shutil.copy2(os.path.join(out, "input_agent.csv"),
                 os.path.join(out, "input_agent_initial.csv"))
    log("  wrote input_agent_initial.csv")

    log("\nDone. Run folder: " + out)
    log("(add 'baltimore_files/' to .gitignore to keep it out of git)")


if __name__ == "__main__":
    main()
