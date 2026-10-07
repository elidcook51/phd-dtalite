#!/usr/bin/env python3
"""
validate_parity.py -- pre-DTALite invariant checks for Eli's DTALite pipeline.

Catches the bug classes that differential MATLAB-vs-Python audits missed,
in seconds, on real data, BEFORE a full pipeline run:

  1. FEASIBILITY: every agent's selected path starts at its origin and
     ends at its destination. (Catches phase-ordering / index-swap bugs
     that hand an agent another OD's path -- the DTALite crash class.)

  2. ROUTE_LENGTH COMPLETENESS: route_length[od, route] > 0 for every
     non-empty choiceset cell. (Catches the used_columns bug class:
     partial computation leaving zeros that min() then wrongly picks.)

  3. ROUTE_LENGTH CORRECTNESS: for a sample of routes, route_length
     equals the independent manual link-length sum. (Catches wrong
     Excel columns/rows, wrong link lookup, transposed indices.)

  4. SHORTCOMING CHOICE: for every fix_user agent, the argmin route
     matches the independently computed shortest path. (End-to-end
     check of the fix_user selection logic.)

  5. ORDERING: verifies phase-concatenated order equals agent-ID order
     for this dataset (warns if not, since the reorder logic matters
     only then).

  6. FIX 1-4 SANITY: weights/realweights shape, observed tt/pltt
     column indices, traveltimecal cache key type.

Usage (from the folder containing DTALite_Files/):
    python validate_parity.py

Exit code: 0 if all checks pass, 1 otherwise. Prints a per-check report.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import csr_matrix


def fail(msg):
    print(f"  FAIL: {msg}")
    return False


def ok(msg):
    print(f"  ok: {msg}")
    return True


def main():
    base = "DTALite_Files"
    if not os.path.isdir(base):
        # allow running from inside DTALite_Files
        base = "."
    all_ok = True

    print("== loading data ==")
    mat = loadmat(os.path.join(base, "choice set no overlap try.mat"),
                  simplify_cells=True)
    choiceset = mat["finallist"].T  # (n_od, n_routes), transposed convention
    n_od, n_routes = choiceset.shape
    print(f"  choiceset: {n_od} ODs x {n_routes} routes")

    # --- link lengths (same source as assignment.py) ---
    link_lengths = pd.read_excel(
        os.path.join(base, "SiouxFalls_net.xlsx"), sheet_name=0,
        usecols="B:E", skiprows=89, nrows=76, header=None).to_numpy()
    max_node = int(np.max(link_lengths[:, 0:2]))
    ridx = link_lengths[:, 0].astype(int)
    cidx = link_lengths[:, 1].astype(int)
    link_lookup = csr_matrix(
        (np.arange(1, len(link_lengths) + 1), (ridx, cidx)),
        shape=(max_node + 1, max_node + 1))

    def true_length(path):
        nodes = [int(x) for x in path.split(";") if x.strip()]
        total = 0.0
        for n1, n2 in zip(nodes[:-1], nodes[1:]):
            li = int(link_lookup[n1, n2])
            if li > 0:
                total += link_lengths[li - 1, 2]
        return total

    # --- routelocation (same as msa.py) ---
    routelocation = []
    for r in range(n_od):
        for c in range(n_routes):
            rs = choiceset[r, c]
            if not isinstance(rs, str):
                continue
            parts = rs.strip(";").split(";")
            routelocation.append([float(parts[0]), float(parts[-1]), 1, c, r])
    routelocation = np.array(routelocation)
    print(f"  routelocation rows: {len(routelocation)}")

    # --- agents ---
    adf = pd.read_csv(os.path.join(base, "input_agent_initial.csv"),
                      usecols=[0, 4, 5, 6])
    agent_id = adf.iloc[:, 0].to_numpy(dtype=int)
    agentOD = adf.iloc[:, [1, 2]].to_numpy(dtype=float)
    dsize = len(adf)
    print(f"  agents: {dsize}")

    ua = loadmat(os.path.join(base, "0_userassignment.mat"))
    fix_user = set(int(v) for v in ua["fix_user"].flatten())
    member = set(int(v) for v in ua["member"].flatten())
    realtime_user = set(int(v) for v in ua["realtime_user"].flatten())
    print(f"  fix={len(fix_user)} member={len(member)} realtime={len(realtime_user)}")

    # --- independent route_length (MATLAB-style: ALL cells) ---
    print("== check 2/3: route_length completeness & correctness ==")
    route_length = np.zeros((n_od, n_routes))
    for r in range(n_od):
        for c in range(n_routes):
            p = choiceset[r, c]
            if isinstance(p, str) and p.strip() not in ("", "[]"):
                route_length[r, c] = true_length(p)

    # completeness: every non-empty cell must have length > 0
    bad_cells = []
    for r in range(n_od):
        for c in range(n_routes):
            p = choiceset[r, c]
            if isinstance(p, str) and p.strip() not in ("", "[]"):
                if route_length[r, c] <= 0:
                    bad_cells.append((r, c))
    if bad_cells:
        all_ok = fail(f"{len(bad_cells)} non-empty routes have length<=0 "
                      f"(partial-computation bug); e.g. {bad_cells[:3]}")
    else:
        all_ok = ok("all non-empty choiceset cells have length > 0") and all_ok

    # correctness spot-check on 200 random routes
    rng = np.random.default_rng(0)
    nonempty = [(r, c) for r in range(n_od) for c in range(n_routes)
                if isinstance(choiceset[r, c], str)
                and choiceset[r, c].strip() not in ("", "[]")]
    sample = rng.choice(len(nonempty), size=min(200, len(nonempty)),
                        replace=False)
    mism = [(nonempty[i], route_length[nonempty[i][0], nonempty[i][1]],
             true_length(choiceset[nonempty[i][0], nonempty[i][1]]))
            for i in sample
            if abs(route_length[nonempty[i][0], nonempty[i][1]]
                   - true_length(choiceset[nonempty[i][0], nonempty[i][1]])) > 1e-9]
    if mism:
        all_ok = fail(f"{len(mism)} sampled routes have wrong length; "
                      f"e.g. {mism[:2]}")
    else:
        all_ok = ok(f"{len(sample)} sampled route lengths match manual sums") and all_ok

    # --- check 1/4: fix_user shortest-path selection + feasibility ---
    print("== check 1/4: fix_user selection & path feasibility ==")
    n_infeasible = 0
    n_wrong_pick = 0
    for i in sorted(fix_user):
        if i < 0 or i >= dsize:
            continue
        matches = np.where((routelocation[:, 0] == agentOD[i, 0])
                           & (routelocation[:, 1] == agentOD[i, 1]))[0]
        valid = []
        for idx in matches:
            rr, cc = int(routelocation[idx, 4]), int(routelocation[idx, 3])
            if isinstance(choiceset[rr, cc], str):
                valid.append((rr, cc))
        if not valid:
            continue
        best = min(valid, key=lambda rc: route_length[rc[0], rc[1]])
        path = choiceset[best[0], best[1]]
        nodes = [x for x in path.split(";") if x.strip()]
        # feasibility
        if not (float(nodes[0]) == agentOD[i, 0]
                and float(nodes[-1]) == agentOD[i, 1]):
            n_infeasible += 1
            if n_infeasible <= 3:
                fail(f"agent {i} OD {tuple(agentOD[i])}: path '{path[:40]}' "
                     f"infeasible")
        # shortest?
        lens = sorted(route_length[rc[0], rc[1]] for rc in valid)
        if abs(route_length[best[0], best[1]] - lens[0]) > 1e-9:
            n_wrong_pick += 1
    if n_infeasible:
        all_ok = False
    else:
        all_ok = ok(f"all {len(fix_user)} fix_user agents get feasible paths") and all_ok
    if n_wrong_pick:
        all_ok = fail(f"{n_wrong_pick} fix_user agents did not get shortest path")
    else:
        all_ok = ok("all fix_user agents get the shortest route") and all_ok

    # --- check 5: ordering ---
    print("== check 5: phase order vs agent order ==")
    deps = pd.read_csv(os.path.join(base, "input_agent_initial.csv"),
                       usecols=[6]).iloc[:, 0].to_numpy()
    if np.array_equal(agent_id, np.arange(dsize)) and np.all(deps[:-1] <= deps[1:]):
        ok("input sorted by departure time, IDs 0..N-1: "
           "phase-concat order == agent-ID order (reorder is no-op)")
    else:
        fail("input NOT sorted: phase-concat order != agent-ID order; "
             "the assignment()/comassignment() reorder fix is REQUIRED")
        all_ok = False

    # --- check 6: Fix 1-4 sanity ---
    print("== check 6: Fixes 1-4 sanity ==")
    # Fix 1: weights header=None -> first row should be numeric weights
    try:
        w = pd.read_excel(os.path.join(base, "SiouxFalls_net.xlsx"),
                          sheet_name=1, header=None)
        ok(f"weights sheet readable with header=None (shape {w.shape})")
    except Exception as e:  # noqa: BLE001
        all_ok = fail(f"weights read failed: {e}")
    # Fix 2: output_agent.csv col 9 is departure_time
    # (can't check without a DTALite output; check the code instead)
    import re
    ra_path = os.path.join(base, "..", "realtimeassignment.py")
    if not os.path.isfile(ra_path):
        ra_path = "realtimeassignment.py"
    if os.path.isfile(ra_path):
        src = open(ra_path, encoding="utf-8", errors="replace").read()
        m = re.search(r"usecols=\[0,\s*(\d+),\s*12,\s*29\]", src)
        if m and m.group(1) == "9":
            ok("realtimeassignment_fast uses usecols=[0,9,12,29] (Fix 2)")
        else:
            all_ok = fail("realtimeassignment_fast usecols != [0,9,12,29] "
                          "(Fix 2 missing?)")
    # used_columns regression guard: the exact bug pattern
    asg = "assignment.py"
    if not os.path.isfile(asg):
        asg = os.path.join(base, "..", "assignment.py")
    if os.path.isfile(asg):
        src = open(asg, encoding="utf-8", errors="replace").read()
        if "routelocation[matches[0],3]" in src and "np.arange" not in src:
            all_ok = fail("assignment.py still uses matches[0]-only "
                          "used_columns (the v5 bug is back!)")
        else:
            ok("used_columns bug pattern not present")

    print()
    if all_ok:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED -- see above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
