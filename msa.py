import os

import shutil

import subprocess

from assignment import assignment, comassignment, sanitize_dtalite_outputs

import numpy as np

import pandas as pd

from scipy.io import loadmat, savemat

import time





def dividerand_matlab(bigloop):

    filename = f"DTALite_Files/{bigloop}_userassignment.mat"



    data = loadmat(filename)



    member = data['member'].flatten()

    realtime_user = data['realtime_user'].flatten()

    fix_user = data['fix_user'].flatten()



    return member, realtime_user, fix_user



def dividerand(driver_ids, p, p_realtime, p_fix):



    rng = np.random.default_rng(seed = 1)



    driver_ids = np.array(driver_ids)

    rng.shuffle(driver_ids)



    n = len(driver_ids)



    n_member = int(round(p * n))

    n_realtime = int(round(p_realtime * n))

    n_fix = int(round(p_fix * n))



    member = driver_ids[:n_member]

    realtime_user = driver_ids[n_member:n_member + n_realtime]

    fix_user = driver_ids[n_member + n_realtime:n_member + n_realtime + n_fix]



    return member, realtime_user, fix_user



def parse_path_endpoints(raw_path, row_index=None):

    """

    Convert a scalar path such as '1;2;6;5;' into its origin and

    destination nodes.



    Returns

    -------

    origin, destination, normalized_path



    Returns (None, None, None) for an empty choiceset entry.

    """



    # Unwrap NumPy arrays only if they contain exactly one scalar value.

    if isinstance(raw_path, np.ndarray):

        if raw_path.size == 0:

            return None, None, None



        if raw_path.size == 1:

            raw_path = raw_path.item()

        else:

            # An array containing multiple choiceset cells is not one path.

            all_empty = all(

                isinstance(value, np.ndarray) and value.size == 0

                for value in raw_path.ravel()

            )



            if all_empty:

                return None, None, None



            raise ValueError(

                "Expected one path string but received an entire array. "

                f"row_index={row_index}, "

                f"shape={raw_path.shape}, "

                f"value={raw_path!r}"

            )



    # Handle lists and tuples similarly.

    if isinstance(raw_path, (list, tuple)):

        if len(raw_path) == 0:

            return None, None, None



        if len(raw_path) == 1:

            raw_path = raw_path[0]

        else:

            raise ValueError(

                "Expected one path string but received a list of paths. "

                f"row_index={row_index}, value={raw_path!r}"

            )



    if raw_path is None:

        return None, None, None



    # Handle pandas/NumPy missing values.

    try:

        if pd.isna(raw_path):

            return None, None, None

    except (TypeError, ValueError):

        pass



    if not isinstance(raw_path, str):

        raise TypeError(

            "Path must be a string. "

            f"row_index={row_index}, "

            f"type={type(raw_path).__name__}, "

            f"value={raw_path!r}"

        )



    raw_path = raw_path.strip()



    if not raw_path:

        return None, None, None



    # Catch array representations accidentally converted to strings.

    if raw_path.startswith("[array(") or "dtype=" in raw_path:

        return None, None, None



    parts = [

        value.strip()

        for value in raw_path.split(";")

        if value.strip()

    ]



    if len(parts) < 2:

        return None, None, None



    try:

        nodes = [

            int(float(value))

            for value in parts

        ]

    except ValueError as exc:

        raise ValueError(

            "The path contains a nonnumeric node. "

            f"row_index={row_index}, path={raw_path!r}"

        ) from exc



    normalized_path = ";".join(

        str(node) for node in nodes

    ) + ";"



    return nodes[0], nodes[-1], normalized_path



def msa(bigloop, p, p_fix, p_realtime):

    start_time = time.time()



    temp_agent_data = pd.read_csv('DTALite_Files/input_agent_initial.csv')



    inputagent = temp_agent_data.values.tolist()



    dsize = len(inputagent)



    temp_tdlink = pd.read_csv('DTALite_Files/Each iteration.csv')

    num_tdlink_rows = len(temp_tdlink)



    driver_ids = np.arange(dsize)

    #member, realtime_user, fix_user = dividerand(driver_ids, p, p_realtime, p_fix)

    member, realtime_user, fix_user = dividerand_matlab(bigloop)



    mat = loadmat('DTALite_Files/choice set no overlap try.mat', simplify_cells = True)



    choiceset = mat['finallist'].T



    routelocation = []



    for row_idx in range(choiceset.shape[0]):

        for col_idx in range(choiceset.shape[1]):



            route_string = choiceset[row_idx, col_idx]



            if not isinstance(route_string, str):

                continue



            parts = route_string.strip(';').split(';')



            o = float(parts[0])

            d = float(parts[-1])



            routelocation.append([o,d,1,col_idx,row_idx])



    routelocation = np.array(routelocation)



    od_to_column = {}



    for row in routelocation:

        o = int(row[0])

        d = int(row[1])

        col = int(row[3])



        if (o, d) not in od_to_column:

            od_to_column[(o, d)] = col



    user = pd.read_excel(

        'DTALite_Files/SiouxFalls_net.xlsx',

        sheet_name='users',

        usecols = 'A:B',

        nrows = dsize

    ).values



    weights = pd.read_excel(

        'DTALite_Files/SiouxFalls_net.xlsx',

        sheet_name='weights',

        usecols = 'A:F',

        skiprows = 5

    ).values



    realweights = pd.read_excel(

        'DTALite_Files/SiouxFalls_net.xlsx',

        sheet_name='weights',

        usecols = 'J:O',

        skiprows = 5

    ).values



    meanstd28 = pd.read_excel(

        'DTALite_Files/SiouxFalls_net.xlsx',

        sheet_name ='weights',

        usecols = 'A:E',

        skiprows = 0,

        nrows = 3,

        header = None,

    ).values



    meanstd2 = pd.read_excel(

        'DTALite_Files/SiouxFalls_net.xlsx',

        sheet_name = 'weights',

        usecols = 'H:L',

        skiprows = 0,

        nrows = 3,

        header = None,

    ).values



    meanstd28 = meanstd28[1:3].astype(float)

    meanstd2 = meanstd2[1:3].astype(float)



    a0 = np.zeros((dsize, 3))



    for i in range(dsize):

        a0[i, 0] = inputagent[i][6]

        a0[i, 1] = inputagent[i][4]

        a0[i, 2] = inputagent[i][5]



    a = np.floor(a0)




    cc = np.zeros(25)



    linkV = []

    rprime = {}



    prev_choice = None
    prev_rposition = None



    for itr in range(1, 19):

        print(f"Iteration {itr}")

        print(f"Starting running assignment")



        (

            choice,

            final_rposition,

            nchoice,

            pre_rposition,

            ttt

        ) = assignment(

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

            num_tdlink_rows

        )



        # bprime columns: 0 = OD index, 1 = route index (both transposed
        # convention), 2 = departure time, 3 = origin, 4 = destination.
        # MATLAB's bprime(:,1) is the route index; here the route lives in
        # column 1 because the transposed choiceset swaps the two indices.
        bprime = np.column_stack([

            choice,

            final_rposition,

            a[:,0],

            a[:,1],

            a[:,2]

        ])



        # choiceset has shape [od_index, route_index] (transposed).



        n_od, n_routes = choiceset.shape





        def normalize_route_path(raw_path):

            """

            Convert one choiceset cell into a valid path string and node list.



            Returns

            -------

            normalized_path : str or None

                Example: "1;2;6;5;"

            nodes : list[int] or None

                Example: [1, 2, 6, 5]

            """



            if raw_path is None:

                return None, None



            # Handle MATLAB-loaded empty arrays and scalar arrays.

            if isinstance(raw_path, np.ndarray):

                if raw_path.size == 0:

                    return None, None



                if raw_path.size == 1:

                    raw_path = raw_path.item()

                else:

                    # An array with multiple elements is not one route string.

                    return None, None



            # Handle NaN values.

            try:

                if pd.isna(raw_path):

                    return None, None

            except (TypeError, ValueError):

                pass



            if not isinstance(raw_path, str):

                return None, None



            raw_path = raw_path.strip()



            if not raw_path:

                return None, None



            try:

                nodes = [

                    int(float(part.strip()))

                    for part in raw_path.split(";")

                    if part.strip()

                ]

            except (TypeError, ValueError):

                return None, None



            if len(nodes) < 2:

                return None, None



            normalized_path = ";".join(

                str(node) for node in nodes

            ) + ";"



            return normalized_path, nodes





        # =====================================================
        # Build rprime for every OD column
        # (MATLAB: rprime{i,h}(j,itr) -- demand on route i of OD h
        #  in period j. Keys here are 0-based (route, od, period, itr).)
        # =====================================================

        for od_idx in range(n_od):

            # Find the first valid route in this OD column.
            # Every route in the same OD column has the same
            # origin and destination.
            nodes = None

            for route_idx in range(n_routes):

                _, candidate_nodes = normalize_route_path(
                    choiceset[od_idx, route_idx]
                )

                if candidate_nodes is not None:
                    nodes = candidate_nodes
                    break

            # This OD column has no valid paths.
            if nodes is None:
                continue

            origin = float(nodes[0])
            destination = float(nodes[-1])

            for route_idx in range(n_routes):

                # Skip OD/route cells with no valid path so that
                # unchosen-but-valid routes still get a zero count.
                _, route_nodes = normalize_route_path(
                    choiceset[od_idx, route_idx]
                )
                if route_nodes is None:
                    continue

                for interval in range(60):

                    # MATLAB: j = 1..60, period = j + 359.
                    period = interval + 360

                    idx = np.where(
                        (bprime[:, 2] == period)
                        & (bprime[:, 3] == origin)
                        & (bprime[:, 4] == destination)
                        & (bprime[:, 0] == od_idx)
                        & (bprime[:, 1] == route_idx)
                    )[0]

                    rprime[
                        (
                            route_idx,
                            od_idx,
                            interval,
                            itr
                        )
                    ] = len(idx)






        # =====================================================

        # Put each agent's selected path in inputagent

        # =====================================================



        for j in range(dsize):

            # choice / final_rposition are the post-realtime OD and route
            # indices (transposed choiceset: [od_index, route_index]).

            od_index = int(choice[j])

            route_index = int(final_rposition[j])

            if not (

                0 <= od_index < n_od

                and 0 <= route_index < n_routes

            ):

                raise IndexError(

                    "Invalid choiceset indices for agent "

                    f"{j}: od_index={od_index}, "

                    f"route_index={route_index}, "

                    f"choiceset shape={choiceset.shape}"

                )

            selected_path, nodes = normalize_route_path(

                choiceset[od_index, route_index]

            )

            if selected_path is None:

                raise ValueError(

                    "The selected choiceset entry does not contain a valid "

                    f"path for agent {j}. "

                    f"od_index={od_index}, "

                    f"route_index={route_index}, "

                    f"raw value="

                    f"{choiceset[od_index, route_index]!r}"

                )



            inputagent[j][11] = selected_path



        columns = ['agent_id', 'tour_id', 'from_zone_id', 'to_zone_id', 'from_origin_node_id', 'to_destination_node_id', 'departure_time_in_min','demand_type', 'PCE', 'information_type', 'vehicle_age', 'path_node_sequence', 'vehicle_type', 'pricing_type', 'value_of_time']



        pd.DataFrame(

            inputagent,

            columns =columns

        ).to_csv(

            'DTALite_Files/input_agent.csv', index = False

        )



        subprocess.run(

            [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True

        )

        sanitize_dtalite_outputs()



        shutil.copyfile(

            'DTALite_Files/output_LinkTDMOE.csv',

            'DTALite_Files/iteration_LinkTDMOE.csv'

        )



        if itr > 1:

            # MATLAB counts agents whose *route* changed (msa.m cc block).

            # Under the transposed choiceset, `choice` holds the OD index

            # (never changes per agent); the route index is final_rposition.

            cc[itr - 1] = np.sum(final_rposition != prev_rposition)



        prev_choice = np.array(choice)
        prev_rposition = np.array(final_rposition)



        dd = pd.read_csv('DTALite_Files/output_LinkTDMOE.csv')



        linkV.append(dd.iloc[:, 7].to_numpy())



    shutil.copyfile(

        'DTALite_Files/output_agent.csv',

        f"DTALite_Files/{bigloop}UOinfor_agent.csv"

    )



    shutil.copyfile(

        'DTALite_Files/output_LinkMOE.csv',

        f"DTALite_Files/{bigloop}UOinfor_LinkMOE.csv"

    )



    shutil.copyfile(

        'DTALite_Files/output_LinkTDMOE.csv',

        f"DTALite_Files/{bigloop}UOinfor_LinkTDMOE.csv"

    )



    shutil.copyfile(

        'DTALite_Files/output_agent.csv',

        'DTALite_Files/UOinfor_agent.csv'

    )



    shutil.copyfile(

        'DTALite_Files/output_LinkMOE.csv',

        'DTALite_Files/UOinfor_LinkMOE.csv'

    )



    shutil.copyfile(

        'DTALite_Files/output_LinkTDMOE.csv',

        'DTALite_Files/UOinfor_LinkTDMOE.csv'

    )



    (

        pre_choice,

        pre_rposition,

        pre_tt,

        pre_ttt,

        pre_inform

    ) = assignment(

        2,

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

        num_tdlink_rows

    )



    pd.DataFrame(inputagent, columns = columns).to_csv('DTALite_Files/input_agent.csv')



    subprocess.run(

        [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True

    )

    sanitize_dtalite_outputs()



    shutil.copyfile(

        'DTALite_Files/output_agent.csv',

        f"DTALite_Files/{bigloop}predict_agent.csv"

    )



    shutil.copyfile(

        'DTALite_Files/output_LinkMOE.csv',

        f"DTALite_Files/{bigloop}predict_LinkMOE.csv"

    )



    shutil.copyfile(

        'DTALite_Files/output_LinkTDMOE.csv',

        f"DTALite_Files/{bigloop}predict_LinkTDMOE.csv"

    )



    print('#' * 20)

    print('\n' * 5)

    print('Starting comassignment')

    print('\n' * 5)

    print('#' * 20)

    print(f"It's been {time.time() - start_time} seconds")



    (

        com_choice,

        com_rposition,

        com_tt,

        com_ttt,

        com_inform,

        pre_choice

    ) = comassignment(

        100,

        choiceset,

        user,

        routelocation,

        realweights,

        meanstd28,

        meanstd2,

        member,

        fix_user,

        realtime_user,

        a0,

        inputagent,

        bigloop,

        dsize,

        num_tdlink_rows

    )



    for j in range(dsize):



        inputagent[j][11] = choiceset[

            int(com_choice[j]), int(com_rposition[j])

        ]



    pd.DataFrame(inputagent, columns=columns).to_csv('DTALite_Files/input_agent.csv', index = False)



    subprocess.run(

        [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True

    )

    sanitize_dtalite_outputs()



    shutil.copyfile(

        'DTALite_Files/output_agent.csv',

        f"DTALite_Files/{bigloop}actual_agent.csv"

    )



    shutil.copyfile(

        'DTALite_Files/output_LinkMOE.csv',

        f"DTALite_Files/{bigloop}actual_LinkMOE.csv"

    )



    shutil.copyfile(

        'DTALite_Files/output_LinkTDMOE.csv',

        f"DTALite_Files/{bigloop}actual_LinkTDMOE.csv"

    )



    savemat(

        f"DTALite_Files/myfile{bigloop}.mat",

        {

            'cc': cc,

            'rprime': rprime,

            'linkV': np.array(linkV, dtype = object),

            'choice': com_choice,

            'position': com_rposition,

            'pre_choice': pre_choice

        }

    )



