import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix
from pathlib import Path
import subprocess
import pickle
from traveltimecal import traveltimecal, traveltimecal_fast, traveltimecal_fastv2
from realtimeassignment import comrealtimeassignment, realtimeassignment_fast
import time

#Need input_agent.csv, output_agent.csv
#Need Each iteration.csv, iteration_LinkTDMOE.csv

def assignment(itr, choiceset, user, routelocation, weights, meanstd28, meanstd2, member, fix_user, realtime_user, a0, inputagent, bigloop, dsize, num_tdlink_rows):

    onlylike0 = []
    onlylike1 = []

    ttt = 0

    if itr == 1:
        S = pd.read_csv('DTALite_Files/input_agent.csv')

        agent = np.column_stack([
            S.iloc[:, 0],
            S.iloc[:, 6]
        ])

        agentOD = np.column_stack([
            S.iloc[:, 4],
            S.iloc[:,5]
        ])

        agentpath = None

        TDlink_table = pd.read_csv('DTALite_Files/Each iteration.csv')

    else:
        T = pd.read_csv('DTALite_Files/output_agent.csv')

        agent = np.column_stack([
            T.iloc[:,0],
            T.iloc[:,9],
            T.iloc[:,12]
        ])

        agentpath = T.iloc[:,29].astype(str).tolist()

        ttt = np.sum(agent[:,2])

        agentOD = np.column_stack([
            T.iloc[:,7],
            T.iloc[:,8]
        ])

        TDlink_table = pd.read_csv('DTALite_Files/iteration_LinkTDMOE.csv')

    TDlink = TDlink_table.iloc[:, [0,1,4,5]].to_numpy()

    link_lengths = pd.read_excel('DTALite_Files/SiouxFalls_net.xlsx', sheet_name = 0, usecols = 'B:E', skiprows = 89, nrows = 76).to_numpy()

    max_node = int(np.max(link_lengths[:,0:2]))

    row_idx = link_lengths[:,0].astype(int)
    col_idx = link_lengths[:,1].astype(int)

    data = np.arange(len(link_lengths))

    link_lookup = csr_matrix(
        (data, (row_idx, col_idx)),
        shape = (max_node+1, max_node+1)
    )

    used_od_pairs = np.unique(agentOD, axis = 0)

    used_columns = []

    for od in used_od_pairs:

        matches = np.where(
            (routelocation[:,0] == od[0]) & 
            (routelocation[:,1] == od[1])
        )[0]

        if len(matches) > 0:
            used_columns.append(
                int(routelocation[matches[0],3])
            )

    used_columns = np.unique(used_columns)

    nrows = choiceset.shape[0]
    ncols = choiceset.shape[1]

    route_length = np.zeros((nrows, ncols))
    nc = np.zeros((nrows, ncols)) 

    for col in used_columns:
        for i in range(nrows):

            path = choiceset[i, col]

            if path is None or (isinstance(path, np.ndarray) and path.size == 0) or path == "" or str(path).strip() == "[]":
                continue

            nodes = [
                int(x) for x in path.split(';') if x.strip()
            ]

            if len(nodes) < 2:
                continue

            for n1, n2 in zip(nodes[:-1], nodes[1:]):

                if (
                    0 < n1 <= max_node and
                    0 < n2 <= max_node
                ):
                    link_idx = link_lookup[n1, n2]

                    if link_idx >= 0:
                        route_length[i, col] += (
                            link_lengths[int(link_idx)-1,2]
                        )

            nc[i, col] = len(nodes) - 1

    gas = 3

    tt = {}
    pltt = {}
    fuelcost = {}

    for col in used_columns:
        for i in range(nrows):

            path = choiceset[i, col]

            if isinstance(path, np.ndarray):
                if path.size == 0:
                    continue
                path = path.item()

            if path is None or path == "":
                continue

            tt[(i, col)] = np.zeros(60)
            pltt[(i, col)] = np.zeros(60)
            fuelcost[(i, col)] = np.zeros(60)

            for k in range(60):
                departure_time = k + 899

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
                fuelcost[(i, col)][k] = fuel

    rposition = np.zeros(dsize, dtype = int)
    nchoice = np.zeros(dsize, dtype = int)

    for i in range(dsize):
        matches = np.where(
            (routelocation[:,0] == agentOD[i,0]) & 
            (routelocation[:,1] == agentOD[i,1])
        )[0]

        if len(matches) == 0:
            continue

        f = int(routelocation[matches[0], 3])

        po = int(np.floor(agent[i, 1]) - 899)

        po = max(1, min(po, 60))
        po -= 1


        if i in fix_user:

            valid_routes = []

            for idx in matches:

                route_row = int(routelocation[idx, 4])
                route_col = int(routelocation[idx, 3])

                path = choiceset[route_row, route_col]

                if not isinstance(path, str):
                    continue

                valid_routes.append((route_row, route_col))

            if len(valid_routes) == 0:
                continue

            best_row, best_col = min(valid_routes, key = lambda rc: route_length[rc[0], rc[1]])

            nchoice[i] = best_row
            rposition[i] = best_col

            continue

        if i in member:

            available = []

            for idx in matches:
                route_row = int(routelocation[idx, 4])
                route_col = int(routelocation[idx, 3])

                if (route_row, route_col) in tt:
                    available.append(
                        (route_row, route_col)
                    )

            if len(available) == 0:
                continue

            best_row, best_col = available[0]

            current = np.array([
                route_length[best_row, best_col],
                tt[(best_row, best_col)][po],
                pltt[(best_row, best_col)][po],
                fuelcost[(best_row, best_col)][po],
                nc[best_row, best_col]
            ])

            for route_row, route_col in available[1:]:

                if choiceset[route_row, route_col] is None:
                    continue

                candidate = np.array([
                    route_length[route_row, route_col],
                    tt[(route_row, route_col)][po],
                    pltt[(route_row, route_col)][po],
                    fuelcost[(route_row, route_col)][po],
                    nc[route_row, route_col]
                ])

                diff = current - candidate

                uid = int(user[i,1])

                utility = (
                    np.dot(
                        diff,
                        weights[uid, :5]
                    )
                    + weights[uid,5]
                )

                choice = (
                    0 if abs(utility-1) > abs(utility + 1) else 1
                )

                if choice == 0:
                    current = candidate
                    best_row = route_row
                    best_col = route_col

            nchoice[i] = best_row
            rposition[i] = best_col

        if i in realtime_user:

            available = []

            for idx in matches:
                route_row = int(routelocation[idx, 4])
                route_col = int(routelocation[idx, 3])

                if (route_row, route_col) in tt:
                    available.append(
                        (route_row, route_col)
                    )

            if len(available) == 0:
                continue

            best_row, best_col = available[0]

            nchoice[i] = best_row
            rposition[i] = best_col

    if itr == 18:

        savemat(
            f"pathinfo_ass{bigloop}.mat",
            {
                'len': route_length,
                'tt': tt,
                'pltt': pltt,
                'fuelcost': fuelcost,
                'nc': nc
            }
        )

    phase = 6
    phlength = 60 / phase

    final_choice = []
    final_rposition = []

    for subitr in range(1, phase + 1):

        (
            ite_choice,
            updaterposition,
            tt,
            inform,
            check
        ) = realtimeassignment_fast(
            subitr,
            choiceset,
            routelocation,
            phlength,
            realtime_user,
            nchoice,
            rposition,
            bigloop,
            dsize,
            num_tdlink_rows
        )

        final_choice.extend(ite_choice)
        final_rposition.extend(updaterposition)

        for k, (choice, pos) in enumerate(
            zip(final_choice[:10], final_rposition[:10])
        ):
            print(
                k,
                type(choice),
                choice,
                type(pos),
                pos
            )

        ypath = [
            choiceset[int(choice),int(pos)] for choice, pos in zip(final_choice, final_rposition)
        ]

        for idx, path in enumerate(ypath):
            inputagent[idx][11] = path

        columns = ['agent_id', 'tour_id', 'from_zone_id', 'to_zone_id', 'from_origin_node_id', 'to_destination_node_id','departure_time_in_min', 'demand_type', 'PCE', 'information_type', 'vehicle_age', 'path_node_sequence', 'vehicle_type', 'pricing_type', 'value_of_time']

        agent_df = pd.DataFrame(inputagent[:len(final_choice)], columns = columns)

        agent_df.to_csv('DTALite_Files/input_agent.csv', index = False)

        start_time = time.time()
        print("Starting DTA Lite running!")
        subprocess.run(
            ['DTALite_Files/DTAlite.exe'], cwd = 'DTALite_Files', check = True
        )  
        print(f"Finished DTA Lite in {time.time() - start_time}")

    savemat(
        f"bef_aft_realtime{itr}.mat",
        {
            'nchoice':np.array(nchoice),
            'final_choice': np.array(final_choice),
            'final_rposition': np.array(final_rposition)
        }
    )

    return (
        np.asarray(final_choice, dtype=int).reshape(-1),
        np.asarray(final_rposition, dtype=int).reshape(-1),
        nchoice,
        rposition,
        ttt
        )


def fixedcomassignment(
    itr,
    choiceset,
    routelocation,
    nchoice,
    rposition,
    weights,
    meanstd28,
    meanstd2,
    user,
    fix_user,
    dsize,
    num_tdlink_rows
):

    onlylike0 = []
    onlylike1 = []

    ttt = 0

    # ============================================================
    # Read agent data
    # ============================================================

    if itr == 1:
        S = pd.read_csv("DTALite_Files/input_agent_initial.csv")

        agent = np.zeros((len(S), 2))
        agent[:, 0] = S.iloc[:, 0]      # agent id
        agent[:, 1] = S.iloc[:, 6]      # departure time

        agentOD = np.column_stack([
            S.iloc[:, 4],
            S.iloc[:, 5]
        ])

        agentpath = []

    else:
        T = pd.read_csv("DTALite_Files/output_agent.csv")

        agent = np.zeros((len(T), 3))
        agent[:, 0] = T.iloc[:, 0]      # agent id
        agent[:, 1] = T.iloc[:, 9]      # departure time
        agent[:, 2] = T.iloc[:, 12]     # travel time

        agentpath = T.iloc[:, 29].tolist()

        ttt = np.sum(agent[:, 2])

        agentOD = np.column_stack([
            T.iloc[:, 7],
            T.iloc[:, 8]
        ])

    # ============================================================
    # Read time-dependent link performance
    # ============================================================

    if itr == 1:
        df = pd.read_csv("DTALite_Files/Each iteration.csv")

    else:
        df = pd.read_csv("DTALite_Files/output_LinkTDMOE.csv")

    TDlink = np.column_stack([
        df.iloc[:num_tdlink_rows, 0],   # from node
        df.iloc[:num_tdlink_rows, 1],   # to node
        df.iloc[:num_tdlink_rows, 4],   # timestamp
        df.iloc[:num_tdlink_rows, 5],   # travel time
    ])

    # ============================================================
    # Route attributes (distance + intersections)
    # ============================================================

    gas = 3.0
    links = pd.read_csv('DTALite_Files/input_link.csv')
    
    length_data = links[['from_node_id', 'to_node_id', 'length']].to_numpy()

    max_node = int(np.max(length_data[:,0:2]))

    row_idx = length_data[:, 0].astype(int)
    col_idx = length_data[:,1].astype(int)
    data = np.arange(len(length_data))

    link_lookup = csr_matrix(
        (data, (row_idx, col_idx)),
        shape = (max_node+1, max_node+1)
    )

    num_routes = len(choiceset)
    num_columns = len(choiceset[0])

    len_mat = np.zeros((num_routes, num_columns))
    nc = np.zeros((num_routes, num_columns))

    for h in range(num_columns):

        for i in range(num_routes):

            path = choiceset[i, h]

            if (
                path is None
                or (isinstance(path, np.ndarray) and path.size == 0)
                or path == ""
            ):
                continue

            nodes = [int(x) for x in path.split(';') if x.strip()]

            if len(nodes) < 2:
                continue

            for n1, n2 in zip(nodes[:-1], nodes[1:]):
                if (0 < n1 <= max_node and 0 < n2 <= max_node):
                    link_idx = link_lookup[n1, n2]

                    if link_idx >= 0:
                        len_mat[i, h] += length_data[int(link_idx)-1,2]

            nc[i, h] = len(nodes) - 1

    # ============================================================
    # Travel time, planning time, fuel cost
    # ============================================================

    tt = {}
    pltt = {}
    fuelcost = {}

    for h in range(num_columns):

        for i in range(num_routes):

            path = choiceset[i, h]

            if path is None or (isinstance(path, np.ndarray) and path.size == 0) or path == "" or str(path).strip() == "[]":
                continue
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

            tt[(i, h)] = np.zeros((300, 1))
            pltt[(i, h)] = np.zeros((300, 1))
            fuelcost[(i, h)] = np.zeros((300, 1))

            for k in range(300):

                dep_time = k + 900  # MATLAB used (k+899) with k starting at 1

                if m == 1:

                    tt_val, fc_val = traveltimecal_fastv2(
                        dep_time,
                        TDlink,
                        path,
                        length_data,
                        gas,
                        itr,
                        0
                    )

                    tt[(i, h)][k] = tt_val
                    pltt[(i, h)][k] = tt_val
                    fuelcost[(i, h)][k] = fc_val

                else:

                    if itr == 1:
                        ttloc = []

                    else:

                        ttloc = np.where(
                            np.floor(agentn[:, 1]) == dep_time
                        )[0]

                    if len(ttloc) == 0:

                        tt_val, fc_val = traveltimecal_fastv2(
                            dep_time,
                            TDlink,
                            path,
                            length_data,
                            gas,
                            itr,
                            0
                        )

                        tt[(i, h)][k] = tt_val
                        pltt[(i, h)][k] = tt_val
                        fuelcost[(i, h)][k] = fc_val

                    else:

                        a = agentn[ttloc, 2]

                        tt[(i, h)][k] = np.mean(a)
                        pltt[(i, h)][k] = np.max(a)

                        _, fc_val = traveltimecal_fastv2(
                            dep_time,
                            TDlink,
                            path,
                            length_data,
                            gas,
                            itr,
                            0
                        )

                        fuelcost[(i, h)][k] = fc_val

    # ============================================================
    # Route choice assignment
    # ============================================================

    fix_user_set = set(fix_user)

    for i in range(dsize):

        if i in fix_user_set:

            ff = np.where(
                (routelocation[:, 0] == agentOD[i, 0]) &
                (routelocation[:, 1] == agentOD[i, 1])
            )[0]

            po = int(np.floor(agent[i, 1]) - 899)
            po = max(1, min(po, 300))
            po -= 1

            available = []

            for idx in ff:
                route_row = int(routelocation[idx, 4])
                route_col = int(routelocation[idx, 3])

                if (route_row, route_col) in tt:
                    available.append((route_row, route_col))

            if len(available) == 0:
                continue

            best_row, best_col = available[0]

            b1 = np.array([
                len_mat[best_row, best_col],
                float(tt[(best_row,best_col)][po]),
                float(pltt[(best_row, best_col)][po]),
                float(fuelcost[(best_row, best_col)][po]),
                nc[best_row,best_col]
            ])

            for route_row, route_col in available[1:]:
                if(
                    choiceset[route_row,route_col] is None or
                    choiceset[route_row, route_col] == ""
                ):
                    continue

                b0 = np.array([
                    len_mat[route_row, route_col],
                    float(tt[(route_row, route_col)][po]),
                    float(pltt[(route_row,route_col)][po]),
                    float(fuelcost[(route_row,route_col)][po]),
                    nc[route_row,route_col]
                ])

                b = b1 - b0

                if i in onlylike1:
                    choice = 0

                elif i in onlylike0:
                    choice = 1

                else:

                    if user[i, 1] < 16650:

                        bscale = b.copy()

                        ojvalue = np.sum(
                            bscale *
                            weights[int(user[i, 1]), :5]
                        )

                    else:

                        print(meanstd2.shape)

                        bscale = (
                            b - meanstd2[0, :]
                        ) / meanstd2[1, :]

                        ojvalue = np.sum(
                            bscale *
                            weights[int(user[i, 1]), :5]
                        )

                    if abs(ojvalue - 1) > abs(ojvalue + 1):
                        choice = 0
                    else:
                        choice = 1

                if choice == 0:
                    b1 = b0
                    best_row = route_row
                    best_col = route_col

            nchoice[i] = best_row
            rposition[i] = best_col

    return nchoice, rposition
    
def comassignment(
    itr,
    choiceset,
    user,
    routelocation,
    weights,
    meanstd28,
    meanstd2,
    member,
    fix_user,
    realtime_user,
    a0,
    inputagent,
    bigloop,
    dsize,
    num_tdlink_rows,
):
    import numpy as np
    import pandas as pd
    import scipy.io
    import os

    onlylike0 = []
    onlylike1 = []

    ttt = 0

    # ------------------------------------------------------------------
    # Read agent files
    # ------------------------------------------------------------------
    if itr == 1:
        S = pd.read_csv("DTALite_Files/input_agent.csv")

        agent = np.zeros((len(S), 2))
        agent[:, 0] = S.iloc[:, 0]      # agent id
        agent[:, 1] = S.iloc[:, 6]      # departure time

        agentOD = np.column_stack(
            [S.iloc[:, 4].to_numpy(), S.iloc[:, 5].to_numpy()]
        )

    elif itr == 100:
        T = pd.read_csv("DTALite_Files/UOinfor_agent.csv")

        agent = np.zeros((len(T), 3))
        agent[:, 0] = T.iloc[:, 0]
        agent[:, 1] = T.iloc[:, 9]
        agent[:, 2] = T.iloc[:, 12]

        agentpath = T.iloc[:, 29].tolist()

        ttt = np.sum(agent[:, 2])

        agentOD = np.column_stack(
            [T.iloc[:, 7].to_numpy(), T.iloc[:, 8].to_numpy()]
        )

    else:
        T = pd.read_csv("DTALite_Files/output_agent.csv")

        agent = np.zeros((len(T), 3))
        agent[:, 0] = T.iloc[:, 0]
        agent[:, 1] = T.iloc[:, 9]
        agent[:, 2] = T.iloc[:, 12]

        agentpath = T.iloc[:, 29].tolist()

        ttt = np.sum(agent[:, 2])

        agentOD = np.column_stack(
            [T.iloc[:, 7].to_numpy(), T.iloc[:, 8].to_numpy()]
        )

    # ------------------------------------------------------------------
    # Read TD Link Data
    # ------------------------------------------------------------------
    if itr == 1:
        tdfile = "DTALite_Files/Each iteration.csv"
    elif itr == 100:
        tdfile = "DTALite_Files/UOinfor_LinkTDMOE.csv"
    else:
        tdfile = "DTALite_Files/output_LinkTDMOE.csv"

    TD = pd.read_csv(tdfile)

    TDlink = np.column_stack(
        [
            TD.iloc[:num_tdlink_rows, 0],
            TD.iloc[:num_tdlink_rows, 1],
            TD.iloc[:num_tdlink_rows, 4],
            TD.iloc[:num_tdlink_rows, 5],
        ]
    )

    # ------------------------------------------------------------------
    # Route attributes
    # ------------------------------------------------------------------
    gas = 3.0
    links = pd.read_csv('DTALite_Files/input_link.csv')
    
    length_data = links[['from_node_id', 'to_node_id', 'length']].to_numpy()

    n_routes, n_ods = choiceset.shape

    route_len = np.zeros((n_routes, n_ods))
    nc = np.zeros((n_routes, n_ods))

    for h in range(n_ods):
        for i in range(n_routes):

            path = choiceset[i, h]

            if path is None or (isinstance(path, np.ndarray) and path.size == 0) or path == "" or str(path).strip() == "[]":
                continue

            nodes = [
                int(x)
                for x in path.split(";")
                if x.strip() != ""
            ]

            for j in range(len(nodes) - 1):
                mask = (
                    (length_data[:, 0] == nodes[j])
                    &
                    (length_data[:, 1] == nodes[j + 1])
                )

                idx = np.where(mask)[0]

                if len(idx) > 0:
                    route_len[i, h] += length_data[idx[0], 2]

            nc[i, h] = len(nodes) - 1

    # ------------------------------------------------------------------
    # Travel times
    # ------------------------------------------------------------------
    tt = {}
    pltt = {}
    fuelcost = {}

    for h in range(n_ods):
        for i in range(n_routes):

            path = choiceset[i, h]

            if path is None or (isinstance(path, np.ndarray) and path.size == 0) or path == "" or str(path).strip() == "[]":
                continue

            m = 1
            agentn = np.empty((0, 3))

            if itr == 1:
                m = 2

            else:
                matches = [
                    idx
                    for idx, p in enumerate(agentpath)
                    if p == path
                ]

                rows = []

                for idx in matches:
                    rows.append(
                        [
                            agent[idx, 0],
                            agent[idx, 1],
                            agent[idx, 2],
                        ]
                    )

                if len(rows) > 0:
                    agentn = np.array(rows)
                    m = len(rows) + 1

            tt[(i, h)] = np.zeros((300, 1))
            pltt[(i, h)] = np.zeros((300, 1))
            fuelcost[(i, h)] = np.zeros((300, 1))

            for k in range(300):

                current_time = k + 900

                if m == 1:

                    travel_time, fuel = traveltimecal_fastv2(
                        current_time,
                        TDlink,
                        path,
                        length_data,
                        gas,
                        itr,
                        0,
                    )

                    tt[(i, h)][k] = travel_time
                    pltt[(i, h)][k] = travel_time
                    fuelcost[(i, h)][k] = fuel

                else:

                    if itr == 1:
                        ttloc = []
                    else:
                        ttloc = np.where(
                            np.floor(agentn[:, 1]) == current_time
                        )[0]

                    if len(ttloc) == 0:

                        travel_time, fuel = traveltimecal_fastv2(
                            current_time,
                            TDlink,
                            path,
                            length_data,
                            gas,
                            itr,
                            0,
                        )

                        tt[(i, h)][k] = travel_time
                        pltt[(i, h)][k] = travel_time
                        fuelcost[(i, h)][k] = fuel

                    else:

                        observed_tt = agentn[ttloc, 2]

                        tt[(i, h)][k] = np.mean(observed_tt)
                        pltt[(i, h)][k] = np.max(observed_tt)

                        _, fuel = traveltimecal_fastv2(
                            current_time,
                            TDlink,
                            path,
                            length_data,
                            gas,
                            itr,
                            0,
                        )

                        fuelcost[(i, h)][k] = fuel

    # ------------------------------------------------------------------
    # Initial route assignment
    # ------------------------------------------------------------------
    rposition = np.zeros(dsize, dtype=int)
    nchoice = np.zeros(dsize, dtype=int)

    for i in range(dsize):

        ff = np.where(
            (routelocation[:, 0] == agentOD[i, 0])
            &
            (routelocation[:, 1] == agentOD[i, 1])
        )[0]

        po = int(np.floor(agent[i, 1]) - 899)
        po = max(1, min(po, 300))
        po -= 1

        available = []

        for idx in ff:
            route_row = int(routelocation[idx, 4])
            route_col = int(routelocation[idx, 3])

            if (route_row, route_col) in tt:
                available.append((route_row, route_col))

        if len(available) == 0:
            continue

        best_row, best_col = available[0]

        nchoice[i] = best_row
        rposition[i] = best_col

        if i in member:

            best_row, best_col = available[0]

            b1 = np.array([
                route_len[best_row, best_col],
                float(tt[(best_row, best_col)][po]),
                float(pltt[(best_row, best_col)][po]),
                float(fuelcost[(best_row, best_col)][po]),
                nc[best_row, best_col]
            ])

            for route_row, route_col in available:
                if (
                    choiceset[route_row, route_col] is None or
                    choiceset[route_row, route_col] == ""
                ):
                    continue

                b0 = np.array([

                    route_len[route_row, route_col],
                    float(tt[(route_row, route_col)][po]),
                    float(pltt[(route_row, route_col)][po]),
                    float(fuelcost[(route_row, route_col)][po]),
                    nc[route_row, route_col]
                ])

                b = b1 - b0

                if i in onlylike1:
                    choice = 0

                elif i in onlylike0:
                    choice = 1

                else:

                    if user[i, 1] < 16650:

                        bscale = b.copy()

                        ojvalue = np.sum(
                            bscale * weights[int(user[i, 1]), :5]
                        )

                    else:

                        bscale = (
                            b - meanstd2[0, :]
                        ) / meanstd2[1, :]

                        ojvalue = np.sum(
                            bscale * weights[int(user[i, 1]), :5]
                        )

                    if abs(ojvalue - 1) > abs(ojvalue + 1):
                        choice = 0
                    else:
                        choice = 1

                if choice != 1:
                    b1 = b0
                    best_row = route_row
                    best_col = route_col

            nchoice[i] = best_row
            rposition[i] = best_col

    # ------------------------------------------------------------------
    # Fixed-route users
    # ------------------------------------------------------------------
    nchoice, rposition = fixedcomassignment(
        1,
        choiceset,
        routelocation,
        nchoice,
        rposition,
        weights,
        meanstd28,
        meanstd2,
        user,
        fix_user,
        dsize,
        num_tdlink_rows,
    )

    # ------------------------------------------------------------------
    # Save path information
    # ------------------------------------------------------------------
    scipy.io.savemat(
        f"pathinfo_comass{bigloop}.mat",
        {
            "len": route_len,
            "tt": tt,
            "pltt": pltt,
            "fuelcost": fuelcost,
            "nc": nc,
        },
    )

    # ------------------------------------------------------------------
    # Real-time assignment process
    # ------------------------------------------------------------------
    phase = 6
    phlength = 60 / phase

    final_choice = np.zeros(dsize)
    final_rposition = np.zeros(dsize)
    final_pre_choice = np.zeros(dsize)

    for subitr in range(1, phase + 1):

        p_agent = a0[a0[:, 0] <= (360 + phlength * subitr)]
        p_agent = p_agent[
            (360 + phlength * (subitr - 1)) < p_agent[:, 0]
        ]

        (
            ite_choice,
            updaterposition,
            tt,
            inform,
            check,
            pre_choice,
            pre_position
        ) = comrealtimeassignment(
            subitr,
            choiceset,
            routelocation,
            phlength,
            realtime_user,
            nchoice,
            rposition,
            weights,
            meanstd28,
            meanstd2,
            user,
            bigloop,
            dsize,
            num_tdlink_rows,
        )

        if subitr == 1:

            final_choice = np.asarray(ite_choice)
            final_rposition = np.asarray(updaterposition)
            final_pre_choice = np.asarray(pre_choice)
            final_pre_position = np.asarray(pre_position)

            x = 0

        else:

            x = len(final_choice)

            final_choice = np.concatenate(
                [final_choice, np.asarray(ite_choice)]
            )

            final_rposition = np.concatenate(
                [final_rposition, np.asarray(updaterposition)]
            )

            final_pre_choice = np.concatenate(
                [final_pre_choice, np.asarray(pre_choice)]
            )

            final_pre_position = np.concatenate(
                [final_pre_position, np.asarray(pre_position)]
            )

        j = 4

        ypath = []

        for j in range(x + len(p_agent)):
            ypath.append(
                choiceset[
                    int(final_choice[j]),
                    int(final_rposition[j])
                ]
            )


        for j in range(x + len(p_agent)):
            print(type(inputagent))
            print(inputagent.shape)

            print(type(inputagent[0][11]))
            print(repr(inputagent[0][11]))

            print(type(ypath[0]))
            print(repr(ypath[0]))

            inputagent[j][11] = ypath[j]

        if os.path.exists("DTALite_Files/input_agent.csv"):
            os.remove("DTALite_Files/input_agent.csv")

        final_inputagent = inputagent[: (x + len(p_agent))]

        pd.DataFrame(
            final_inputagent,
            columns=[
                "agent_id",
                "tour_id",
                "from_zone_id",
                "to_zone_id",
                "from_origin_node_id",
                "to_destination_node_id",
                "departure_time_in_min",
                "demand_type",
                "PCE",
                "information_type",
                "vehicle_age",
                "path_node_sequence",
                "vehicle_type",
                "pricing_type",
                "value_of_time",
            ],
        ).to_csv("DTALite_Files/input_agent.csv", index=False)

        subprocess.run(
            [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True
        )

    return (
        final_choice,
        final_rposition,
        tt,
        ttt,
        b,
        final_pre_choice,
        final_pre_position
    )