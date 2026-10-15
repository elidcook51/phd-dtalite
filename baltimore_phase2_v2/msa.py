import os

import shutil

import subprocess

from assignment import assignment, comassignment, sanitize_dtalite_outputs
try:
    from progress import get_pbar, dtalite_kwargs, kill_dtalite_leftovers
    _HAS_PROGRESS = True
except ImportError:
    _HAS_PROGRESS = False
    def get_pbar(*a, **k):
        class _D:
            def update(self, n=1): pass
            def set_description(self, d): pass
            def set_postfix_str(self, s): pass
            def close(self): pass
        return _D()
    def dtalite_kwargs():
        return {}

import numpy as np

import pandas as pd

from scipy.io import loadmat, savemat

import time

import netconfig





def dividerand_matlab(bigloop):

    filename = f"DTALite_Files/{bigloop}_userassignment.mat"



    data = loadmat(filename)



    member = data['member'].flatten()

    realtime_user = data['realtime_user'].flatten()

    fix_user = data['fix_user'].flatten()



    return member, realtime_user, fix_user



def dividerand(driver_ids, p, p_realtime, p_fix, seed=1):



    rng = np.random.default_rng(seed)



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



def _baltimore_fallback_paths(missing_ods):
    """Free-flow shortest node path for each missing (oz, dz).

    missing_ods: iterable of (oz, dz, origin_node, dest_node).
    Returns { (oz, dz): [nodes] }.  Dijkstra on free_flow_time from
    input_link.csv; intrazonal (origin == dest) -> single-node path.
    """
    import csv as _csv
    import heapq as _hq
    adj = {}
    with open('DTALite_Files/input_link.csv', newline='',
              encoding='utf-8-sig') as _fh:
        for _row in _csv.DictReader(_fh):
            try:
                _u = int(float(_row['from_node_id']))
                _v = int(float(_row['to_node_id']))
                _t = float(_row['free_flow_time'])
            except (ValueError, TypeError, KeyError):
                continue
            if _t > 0:
                adj.setdefault(_u, []).append((_v, _t))
    out = {}
    for _oz, _dz, _o, _d in missing_ods:
        if _o == _d:
            out[(_oz, _dz)] = [_o]
            continue
        _dist = {_o: 0.0}
        _prev = {}
        _pq = [(0.0, _o)]
        _done = False
        while _pq:
            _du, _u = _hq.heappop(_pq)
            if _u == _d:
                _done = True
                break
            if _du > _dist.get(_u, float('inf')):
                continue
            for _v, _t in adj.get(_u, ()):
                _nd = _du + _t
                if _nd < _dist.get(_v, float('inf')):
                    _dist[_v] = _nd
                    _prev[_v] = _u
                    _hq.heappush(_pq, (_nd, _v))
        if _done:
            _path = [_d]
            while _path[-1] != _o:
                _path.append(_prev[_path[-1]])
            out[(_oz, _dz)] = _path[::-1]
    return out


def msa(bigloop, p, p_fix, p_realtime, seed=1, network='siouxfalls'):

    netconfig.configure(network)

    start_time = time.time()



    temp_agent_data = pd.read_csv('DTALite_Files/input_agent_initial.csv')



    inputagent = temp_agent_data.values.tolist()



    dsize = len(inputagent)



    if netconfig.is_baltimore():
        # No per-minute seed file: iteration 1 runs on free-flow times.
        # traveltimecal falls back to length/speed whenever a
        # (timestamp, from, to) key is absent from the TDlink map, so an
        # empty table is exactly the free-flow seed (avoids a ~2GB CSV).
        temp_tdlink = pd.DataFrame()
    else:
        temp_tdlink = pd.read_csv('DTALite_Files/Each iteration.csv')

    num_tdlink_rows = len(temp_tdlink)



    driver_ids = np.arange(dsize)

    member, realtime_user, fix_user = dividerand(driver_ids, p, p_realtime, p_fix, seed=seed)

    # member, realtime_user, fix_user = dividerand_matlab(bigloop)



    if netconfig.is_baltimore():
        import pickle as _pickle
        with open('DTALite_Files/choice_sets.pkl', 'rb') as _fh:
            _choice_dict = _pickle.load(_fh)
        # Same shapes the Sioux Falls branch produces: choiceset is a
        # 2-D object array of ';'-joined node-id strings indexed
        # [od_row, route_col]; routelocation rows are [o, d, 1, col, row].
        # Baltimore ODs are zones (mirrors the MATLAB od_path_map).
        # Fallback for agent ODs with no precomputed choice set
        # (rare ODs + intrazonal): free-flow shortest path.
        _agent_ods = {}
        for _ar in temp_agent_data.itertuples():
            _oz = int(float(_ar.from_zone_id))
            _dz = int(float(_ar.to_zone_id))
            if (_oz, _dz) not in _choice_dict and (_oz, _dz) not in _agent_ods:
                _agent_ods[(_oz, _dz)] = (int(float(_ar.from_origin_node_id)),
                                          int(float(_ar.to_destination_node_id)))
        if _agent_ods:
            _fb = _baltimore_fallback_paths(
                [(_oz, _dz, _o, _d) for (_oz, _dz), (_o, _d) in _agent_ods.items()])
            for _od, _path in _fb.items():
                _choice_dict[_od] = [_path]
        _od_list = sorted(_choice_dict.keys())
        _k_max = max(len(_v) for _v in _choice_dict.values())
        choiceset = np.empty((len(_od_list), _k_max), dtype=object)
        _rl = []
        for _ri, (_oz, _dz) in enumerate(_od_list):
            _paths = _choice_dict[(_oz, _dz)]
            for _ci, _path in enumerate(_paths):
                choiceset[_ri, _ci] = ';'.join(str(_n) for _n in _path)
                _rl.append([_oz, _dz, 1, _ci, _ri])
            for _ci in range(len(_paths), _k_max):
                choiceset[_ri, _ci] = None
        routelocation = np.array(_rl)
    else:
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



    if netconfig.is_baltimore():
        # agent_weights.csv: agent_id,pref in input_agent.csv row order.
        # Match by agent_id so the mapping survives any row reordering.
        _wdf = pd.read_csv('DTALite_Files/agent_weights.csv')
        _pref_by_agent = dict(zip(_wdf['agent_id'].astype(str),
                                  _wdf['pref'].astype(int)))
        user = np.array([
            [int(_aid), _pref_by_agent.get(str(_aid), 1)]
            for _aid in temp_agent_data['agent_id']
        ], dtype=int)
    else:
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

        skiprows = 5,

        header = None

    ).values



    realweights = pd.read_excel(

        'DTALite_Files/SiouxFalls_net.xlsx',

        sheet_name='weights',

        usecols = 'J:O',

        skiprows = 5,

        header = None

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



    _pbar = get_pbar(bigloop, total=20)
    for itr in range(1, 19):

        _pbar.set_description(f"Bigloop {bigloop} itr {itr}/18")



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
                    period = interval + netconfig.TIME_START

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



        for j in list(member) + list(realtime_user):

            # choice / final_rposition are the post-realtime OD and route
            # indices (transposed choiceset: [od_index, route_index]).

            od_index = int(choice[j])

            route_index = int(final_rposition[j])

            if not (

                0 <= od_index < n_od

                and 0 <= route_index < n_routes

            ):

                # Skip out-of-bounds indices (see note above).

                continue

            selected_path, nodes = normalize_route_path(

                choiceset[od_index, route_index]

            )

            if selected_path is None:

                # Skip agents with no valid path (e.g. realtime user with
                # no valid realtime route). Their path stays at the
                # previous iteration's value. (Was ValueError; too strict
                # for production runs with mixed user types.)

                continue



            inputagent[j][11] = selected_path



        columns = ['agent_id', 'tour_id', 'from_zone_id', 'to_zone_id', 'from_origin_node_id', 'to_destination_node_id', 'departure_time_in_min','demand_type', 'PCE', 'information_type', 'vehicle_age', 'path_node_sequence', 'vehicle_type', 'pricing_type', 'value_of_time']



        pd.DataFrame(

            inputagent,

            columns =columns

        ).to_csv(

            'DTALite_Files/input_agent.csv', index = False

        )



        try:
            subprocess.run(

                [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True,

                **dtalite_kwargs()

            )
        except subprocess.TimeoutExpired:
            kill_dtalite_leftovers()
            raise RuntimeError(
                f"DTALite.exe timed out after 900s (bigloop={bigloop}, itr={itr})")
        _pbar.update(1)

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



    # FIX (parity): write the PREDICTED paths into inputagent before running
    # DTALite, mirroring MATLAB msa.m lines 236-249 and the actual phase below.
    # Previously this wrote the stale iteration-18 paths.
    for j in range(dsize):



        _p = choiceset[
            int(pre_choice[j]), int(pre_rposition[j])
        ]
        if _p is not None:
            inputagent[j][11] = _p



    pd.DataFrame(inputagent, columns=columns).to_csv('DTALite_Files/input_agent.csv', index = False)



    try:
        subprocess.run(

            [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True,

                **dtalite_kwargs()

        )
    except subprocess.TimeoutExpired:
        kill_dtalite_leftovers()
        raise RuntimeError(
            f"DTALite.exe timed out after 900s (bigloop={bigloop}, phase=predict)")

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



    _pbar.set_description(f"Bigloop {bigloop} predict+comassign")
    _pbar.update(1)
    if os.environ.get('DTALITE_VERBOSE') == '1':
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



        _p = choiceset[
            int(com_choice[j]), int(com_rposition[j])
        ]
        if _p is not None:
            inputagent[j][11] = _p



    pd.DataFrame(inputagent, columns=columns).to_csv('DTALite_Files/input_agent.csv', index = False)



    try:
        subprocess.run(

            [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True,

                **dtalite_kwargs()

        )
    except subprocess.TimeoutExpired:
        kill_dtalite_leftovers()
        raise RuntimeError(
            f"DTALite.exe timed out after 900s (bigloop={bigloop}, phase=actual)")

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
    _pbar.set_description(f"Bigloop {bigloop} done")
    _pbar.update(1)
    _pbar.close()
