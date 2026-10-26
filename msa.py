import os

import shutil

import subprocess

from assignment import assignment, comassignment, sanitize_dtalite_outputs
try:
    from progress import get_pbar, dtalite_kwargs, kill_dtalite_leftovers, dtalite_timeout, check_dtalite_outputs
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
    def dtalite_timeout():
        return 900
    def check_dtalite_outputs(label, log_path):
            """Fail loud when DTALite exits 0 but leaves empty outputs.
            DTALite's g_ProgramStop() exits 0 mid-assignment, so a zero exit
            code does NOT mean the run completed: the step-0 truncation leaves
            0-byte outputs and only the MOE writers (which never ran) would
            have filled them. Catch that here, at the call site, instead of
            letting a later pandas read die with a context-free EmptyDataError.
            """
            for _name in ("output_agent.csv", "output_LinkTDMOE.csv"):
                _path = os.path.join("DTALite_Files", _name)
                if (not os.path.exists(_path)) or os.path.getsize(_path) == 0:
                    raise RuntimeError(
                        "DTALite wrote an empty %s after %s. The exe exits 0 even "
                        "when it stops mid-assignment, so this means the run stopped "
                        "early -- see %s for the exe's own stop message."
                        % (_name, label, log_path)
                    )

import numpy as np

import pandas as pd

from scipy.io import loadmat, savemat

import time

import netconfig

import engine as _engine_mod





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



# Module-level cache for the Baltimore free-flow adjacency (v32).
# Built once per process from input_link.csv with pandas; the earlier
# concern about Dijkstra cost was really the csv-module load, and only
# a handful of ODs need fallback paths anyway.
_baltimore_adj_cache = None

def _baltimore_adjacency():
    """Adjacency {from_node: [(to_node, free_flow_time_min), ...]}."""
    global _baltimore_adj_cache
    if _baltimore_adj_cache is not None:
        return _baltimore_adj_cache
    links = pd.read_csv('DTALite_Files/input_link.csv',
                        usecols=lambda c: c in ('from_node_id', 'to_node_id',
                                                'free_flow_time', 'length',
                                                'free_speed'))
    adj = {}
    has_fft = 'free_flow_time' in links.columns
    for _f, _t, _fft, _ln, _fs in zip(links['from_node_id'],
                                      links['to_node_id'],
                                      links['free_flow_time'] if has_fft else [None] * len(links),
                                      links['length'],
                                      links['free_speed']):
        try:
            _w = float(_fft)
        except (TypeError, ValueError):
            _w = float('nan')
        if not (_w > 0):
            # free-flow time from length (miles) / free_speed (mph) * 60
            try:
                _w = float(_ln) / float(_fs) * 60.0
            except (TypeError, ValueError, ZeroDivisionError):
                continue
        if not (_w > 0):
            continue
        adj.setdefault(int(_f), []).append((int(_t), _w))
    _baltimore_adj_cache = adj
    return adj





def _baltimore_fallback_paths(missing_ods):
    """Free-flow shortest paths (Dijkstra) for ODs with no choice set.

    Restored in v32: the v25 single-node placeholders made DTALite
    silently drop those agents, desyncing input/output agent counts.
    With the pandas-based adjacency build this runs in seconds for
    the handful of missing ODs.
    """
    import heapq
    adj = _baltimore_adjacency()
    out = {}
    for _oz, _dz, _o, _d in missing_ods:
        _o, _d = int(_o), int(_d)
        if _o == _d:
            out[(_oz, _dz)] = [_o]
            continue
        dist = {_o: 0.0}
        prev = {}
        pq = [(0.0, _o)]
        found = False
        while pq:
            _du, _u = heapq.heappop(pq)
            if _u == _d:
                found = True
                break
            if _du > dist.get(_u, float('inf')):
                continue
            for _v, _w in adj.get(_u, ()):
                _nd = _du + _w
                if _nd < dist.get(_v, float('inf')):
                    dist[_v] = _nd
                    prev[_v] = _u
                    heapq.heappush(pq, (_nd, _v))
        if not found and _d not in prev:
            # No free-flow path exists; leave a single-node path so the
            # agent still appears (same last resort as before).
            out[(_oz, _dz)] = [_o]
            continue
        path = [_d]
        while path[-1] != _o:
            path.append(prev[path[-1]])
        path.reverse()
        out[(_oz, _dz)] = path
    return out
# Module-level cache for SiouxFalls_net.xlsx weights (v26).
# Four pd.read_excel() calls each re-parse the entire xlsx via openpyxl
# (~3 min each = 13 min total). Values are static; load once per process.
_xlsx_weights_cache = None

def _load_xlsx_weights():
    """Load weights/realweights/meanstd28/meanstd2 from xlsx, cached."""
    global _xlsx_weights_cache
    if _xlsx_weights_cache is not None:
        return _xlsx_weights_cache
    weights = pd.read_excel(
        'DTALite_Files/SiouxFalls_net.xlsx',
        sheet_name='weights',
        usecols='A:F',
        skiprows=5,
        header=None
    ).values
    realweights = pd.read_excel(
        'DTALite_Files/SiouxFalls_net.xlsx',
        sheet_name='weights',
        usecols='J:O',
        skiprows=5,
        header=None
    ).values
    meanstd28 = pd.read_excel(
        'DTALite_Files/SiouxFalls_net.xlsx',
        sheet_name='weights',
        usecols='A:E',
        skiprows=0,
        nrows=3,
        header=None,
    ).values
    meanstd2 = pd.read_excel(
        'DTALite_Files/SiouxFalls_net.xlsx',
        sheet_name='weights',
        usecols='H:L',
        skiprows=0,
        nrows=3,
        header=None,
    ).values
    _xlsx_weights_cache = (weights, realweights, meanstd28, meanstd2)
    return _xlsx_weights_cache


def msa(bigloop, p, p_fix, p_realtime, seed=1, network='siouxfalls',
        engine=None):

    netconfig.configure(network)
    _engine = _engine_mod.resolve(engine)

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
        # No seed row count to cap TD reads with: None makes every
        # downstream TDLinkMOE read take the full table (its real row
        # count). Capping at len(temp_tdlink) = 0 emptied them all and
        # silently priced predict/actual/realtime routes at free-flow.
        num_tdlink_rows = None
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
            # v32: real free-flow shortest paths for missing ODs
            # (Dijkstra restored; see _baltimore_fallback_paths).
            _fb = _baltimore_fallback_paths(
                [(_oz, _dz, _o, _d) for (_oz, _dz), (_o, _d) in _agent_ods.items()])
            for _od, _path in _fb.items():
                _choice_dict[_od] = [_path]
            print(f"  [baltimore] Dijkstra fallback paths for "
                  f"{len(_fb)} missing ODs", flush=True)
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



    weights, realweights, meanstd28, meanstd2 = _load_xlsx_weights()



    meanstd28 = meanstd28[1:3].astype(float)

    meanstd2 = meanstd2[1:3].astype(float)



    a0 = np.zeros((dsize, 3))



    _ia = np.asarray(inputagent)
    a0[:, 0] = _ia[:, 6]
    a0[:, 1] = _ia[:, 4]
    a0[:, 2] = _ia[:, 5]



    a = np.floor(a0)




    cc = np.zeros(25)






    prev_choice = None
    prev_rposition = None



    _pbar = get_pbar(bigloop, total=20)
    # MSA convergence: stop early if no agent changes routes for
    # 3 consecutive iterations (min 10, max 18). This is a pre-registered
    # criterion: a fixed point with cc==0 is mathematically stable, and
    # further MSA iterations with diminishing weights (1/itr) cannot move it.
    _converged_streak = 0
    _min_itr = 10
    _required_streak = 3
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

        # Put each agent's selected path in inputagent

        # =====================================================



        # Fixed users are included: their tournament-chosen path has to
        # be written too, or they simulate on the initial file's path
        # every iteration (Baltimore's initial file has no paths).
        for j in list(member) + list(realtime_user) + list(fix_user):

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
            # DTALite requires path endpoints to match OD nodes
            _nd = [x.strip() for x in str(selected_path).split(';') if x.strip()]
            if len(_nd) >= 2:
                inputagent[j][4] = _nd[0]
                inputagent[j][5] = _nd[-1]



        columns = ['agent_id', 'tour_id', 'from_zone_id', 'to_zone_id', 'from_origin_node_id', 'to_destination_node_id', 'departure_time_in_min','demand_type', 'PCE', 'information_type', 'vehicle_age', 'path_node_sequence', 'vehicle_type', 'pricing_type', 'value_of_time']



        pd.DataFrame(

            inputagent,

            columns =columns

        ).to_csv(

            'DTALite_Files/input_agent.csv', index = False

        )



        _dt_t0 = time.time()
        _call_log = "DTALite_Files/dtalite_itr%d.log" % itr
        if _engine == "pysim":
            with open(_call_log, "w") as _lf:
                _engine_mod.run_pysim(
                    "DTALite_Files", dtalite_timeout(),
                    "bigloop=%d, itr=%d" % (bigloop, itr), stdout=_lf)
        else:
            try:
                with open(_call_log, "w") as _lf:
                    subprocess.run(
                        [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True,
                        stdin=subprocess.DEVNULL, timeout=dtalite_timeout(),
                        stdout=_lf, stderr=subprocess.STDOUT,
                    )
            except subprocess.TimeoutExpired:
                kill_dtalite_leftovers()
                raise RuntimeError(
                    f"DTALite.exe timed out after {dtalite_timeout()}s (bigloop={bigloop}, itr={itr})")
        print(f"  [timing] DTALite call took {time.time() - _dt_t0:.1f}s (itr={itr})", flush=True)
        check_dtalite_outputs("bigloop %d itr %d" % (bigloop, itr), _call_log)
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

            # Convergence check: 3 consecutive iterations with zero changes
            if cc[itr - 1] == 0:
                _converged_streak += 1
            else:
                _converged_streak = 0
            if itr >= _min_itr and _converged_streak >= _required_streak:
                print("MSA converged at iteration %d "
                      "(no route changes for %d iterations)" %
                      (itr, _required_streak), flush=True)
                break



        prev_choice = np.array(choice)
        prev_rposition = np.array(final_rposition)






    shutil.copyfile(

        'DTALite_Files/output_agent.csv',

        f"DTALite_Files/{bigloop}UOinfor_agent.csv"

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
            _nodes = [x.strip() for x in str(_p).split(';') if x.strip()]
            if len(_nodes) >= 2:
                inputagent[j][4] = _nodes[0]
                inputagent[j][5] = _nodes[-1]



    pd.DataFrame(inputagent, columns=columns).to_csv('DTALite_Files/input_agent.csv', index = False)



    _dt_t0 = time.time()
    _call_log = "DTALite_Files/dtalite_predict.log"
    if _engine == "pysim":
        with open(_call_log, "w") as _lf:
            _engine_mod.run_pysim(
                "DTALite_Files", dtalite_timeout(),
                "bigloop=%d, phase=predict" % bigloop, stdout=_lf)
    else:
        try:
            with open(_call_log, "w") as _lf:
                subprocess.run(
                    [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True,
                    stdin=subprocess.DEVNULL, timeout=dtalite_timeout(),
                    stdout=_lf, stderr=subprocess.STDOUT,
                )
        except subprocess.TimeoutExpired:
            kill_dtalite_leftovers()
            raise RuntimeError(
                f"DTALite.exe timed out after {dtalite_timeout()}s (bigloop={bigloop}, phase=predict)")
    print(f"  [timing] DTALite call took {time.time() - _dt_t0:.1f}s (phase=predict)", flush=True)
    check_dtalite_outputs("bigloop %d predict" % bigloop, _call_log)

    sanitize_dtalite_outputs()






    shutil.copyfile(

        'DTALite_Files/output_LinkMOE.csv',

        f"DTALite_Files/{bigloop}predict_LinkMOE.csv"

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

        pre_choice,
        com_rows

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

        num_tdlink_rows,

        engine=_engine

    )



    # com_choice/com_rposition arrive in phase order (see
    # comassignment); map them to file rows by the returned row
    # indices. The old prefix loop assigned phase-grouped choices
    # to file rows by position, which is only correct when the
    # input file is sorted by departure time.
    for _k, _row in enumerate(com_rows):



        _p = choiceset[
            int(com_choice[_k]), int(com_rposition[_k])
        ]
        if _p is not None:
            inputagent[_row][11] = _p
            _nodes = [x.strip() for x in str(_p).split(';') if x.strip()]
            if len(_nodes) >= 2:
                inputagent[_row][4] = _nodes[0]
                inputagent[_row][5] = _nodes[-1]



    pd.DataFrame(inputagent, columns=columns).to_csv('DTALite_Files/input_agent.csv', index = False)



    _dt_t0 = time.time()
    _call_log = "DTALite_Files/dtalite_actual.log"
    if _engine == "pysim":
        with open(_call_log, "w") as _lf:
            _engine_mod.run_pysim(
                "DTALite_Files", dtalite_timeout(),
                "bigloop=%d, phase=actual" % bigloop, stdout=_lf)
    else:
        try:
            with open(_call_log, "w") as _lf:
                subprocess.run(
                    [r"DTALite_Files/DTALite.exe"], cwd = 'DTALite_Files', check = True,
                    stdin=subprocess.DEVNULL, timeout=dtalite_timeout(),
                    stdout=_lf, stderr=subprocess.STDOUT,
                )
        except subprocess.TimeoutExpired:
            kill_dtalite_leftovers()
            raise RuntimeError(
                f"DTALite.exe timed out after {dtalite_timeout()}s (bigloop={bigloop}, phase=actual)")
    print(f"  [timing] DTALite call took {time.time() - _dt_t0:.1f}s (phase=actual)", flush=True)
    check_dtalite_outputs("bigloop %d actual" % bigloop, _call_log)

    sanitize_dtalite_outputs()



    shutil.copyfile(

        'DTALite_Files/output_agent.csv',

        f"DTALite_Files/{bigloop}actual_agent.csv"

    )



    shutil.copyfile(

        'DTALite_Files/output_LinkMOE.csv',

        f"DTALite_Files/{bigloop}actual_LinkMOE.csv"

    )






    _pbar.set_description(f"Bigloop {bigloop} done")
    _pbar.update(1)
    _pbar.close()
