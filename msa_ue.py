"""
msa_ue.py
=========

UE-guidance variant of ``msa.py``.

This is a copy of ``msa.msa`` with the guidance mechanism changed: instead
of giving guided members the tournament-selected UO (user-optimal /
personalized) route from the choice set, every guided member receives its
route from a UE (user-equilibrium) reference solution loaded from
``ue_routes.csv``.

What differs from the Proposed (UO-guidance) version
-----------------------------------------------------
- ``msa_ue(bigloop, p, p_fix, p_realtime, ue_routes_path, seed=1)`` takes
  one extra required argument: the path to ``ue_routes.csv``.  The CSV has
  two columns, ``agent_id`` and ``path_node_sequence`` (format
  ``"n1;n2;...;nK;"``); ``agent_id`` values index the same agents as the
  ``j`` loop variable (0..dsize-1).  The CSV must be generated from the
  SAME demand file as ``input_agent_initial.csv`` (36k vs 54k alignment is
  by ``agent_id``), and it is loaded once, before the MSA iteration loop.
- In the MSA iteration loop (injection point A), guided members get
  ``inputagent[j][11] = ue_route[int(j)]`` instead of the choiceset lookup.
  A missing ``agent_id`` raises ``KeyError`` immediately (fail fast -- a
  silent UO fallback would contaminate the guidance).
- In the predict phase (injection point B), guided members again receive
  their UE route; non-members keep the tournament-predicted UO path.

What is NOT changed
-------------------
- The ``assignment()`` tournament still runs (its outputs are kept for
  bookkeeping), the ``dividerand`` MPR split logic is untouched, and the
  ``comassignment`` actual phase still models real user reaction to the
  guidance.
- Output CSV names and formats are identical to ``msa.py`` so downstream
  analysis code works unchanged.

Known side effect
-----------------
``cc`` and ``rprime`` saved in ``myfile{bigloop}.mat`` are still computed
from the tournament outputs (``final_rposition`` / ``prev_rposition``), not
from the UE routes actually assigned to members.  These arrays are
diagnostic-only and the mismatch is acceptable.
"""

import os

import shutil

import subprocess

from assignment import assignment, comassignment, sanitize_dtalite_outputs
try:
    from progress import get_pbar, dtalite_kwargs
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


# DTALite executable. DTALite.exe is a Windows binary; on Linux (e.g. UVA
# Rivanna) the native DTALite build is preferred, falling back to the Windows
# exe under Wine. The bare name matters: with cwd='DTALite_Files' the child
# resolves it *after* chdir, whereas Windows resolves the 'DTALite_Files/...'
# path against the parent directory. Same rule holds inside the RunX/
# sandboxes: parallel_msa_ue.py copytrees the whole DTALite_Files/ folder
# (binary included) and chdirs into RunX/.
import platform as _platform

if _platform.system() == 'Windows':
    DTALITE_CMD = ['DTALite_Files/DTALite.exe']
else:
    if os.path.isfile(os.path.join('DTALite_Files', 'DTALite')):
        DTALITE_CMD = ['./DTALite']
    else:
        DTALITE_CMD = ['wine', 'DTALite.exe']



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



def stringify_rprime_keys(rprime):

    """Convert rprime's (route, od, interval, itr) tuple keys to strings.

    scipy's savemat silently writes [[None]] for dicts with non-string keys,
    which would lose the whole per-route/OD/interval demand tensor; string
    keys round-trip fine. Keys are 0-based (route, od, interval, itr).
    """

    return {

        f"r{route:03d}_od{od:03d}_t{interval:02d}_itr{itr:02d}": count

        for (route, od, interval, itr), count in rprime.items()

    }



def msa_ue(bigloop, p, p_fix, p_realtime, ue_routes_path, seed=1):
    """
    UE-guidance variant of ``msa``: guided members receive UE routes from
    ``ue_routes.csv`` instead of the tournament-selected UO route.

    Parameters
    ----------
    bigloop, p, p_fix, p_realtime : as in ``msa.msa``.
    ue_routes_path : str or path-like (required, no default)
        Path to ``ue_routes.csv`` with columns ``agent_id`` and
        ``path_node_sequence`` (format ``"n1;n2;...;nK;"``).  ``agent_id``
        must cover the same agent index space as the demand file (0..dsize-1),
        so the CSV has to be generated from the SAME demand file as
        ``DTALite_Files/input_agent_initial.csv`` (36k vs 54k alignment is by
        ``agent_id``).  A missing or unreadable path raises immediately.
    seed : RNG seed for ``dividerand`` (user-type assignment). Default: 1.

    Known side effect
    -----------------
    ``cc`` and ``rprime`` saved in ``myfile{bigloop}.mat`` are computed from
    the tournament outputs (``final_rposition`` / ``prev_rposition``), not
    from the UE routes actually assigned to guided members.  These arrays are
    diagnostic-only; the mismatch is acceptable.
    """

    start_time = time.time()



    temp_agent_data = pd.read_csv('DTALite_Files/input_agent_initial.csv')



    inputagent = temp_agent_data.values.tolist()



    dsize = len(inputagent)



    temp_tdlink = pd.read_csv('DTALite_Files/Each iteration.csv')

    num_tdlink_rows = len(temp_tdlink)



    driver_ids = np.arange(dsize)

    member, realtime_user, fix_user = dividerand(driver_ids, p, p_realtime, p_fix, seed=seed)

    # member, realtime_user, fix_user = dividerand_matlab(bigloop)



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

    # =====================================================
    # UE-guidance routes (loaded once, before the MSA loop)
    # =====================================================
    # ue_routes.csv must be generated from the SAME demand file as
    # DTALite_Files/input_agent_initial.csv: agent_id values index the same
    # agents as the `j` loop variable (0..dsize-1).  36k vs 54k alignment is
    # by agent_id, so a CSV from the wrong demand silently misaligns --
    # hence the explicit coverage check below.
    if ue_routes_path is None:
        raise ValueError(
            "ue_routes_path is required (no default); pass the path to "
            "ue_routes.csv."
        )
    _ue_df = pd.read_csv(
        ue_routes_path, usecols=['agent_id', 'path_node_sequence']
    )
    if _ue_df['agent_id'].duplicated().any():
        _dups = _ue_df.loc[_ue_df['agent_id'].duplicated(), 'agent_id'].unique()[:5]
        raise ValueError(
            f"ue_routes.csv ({ue_routes_path!r}) has duplicate agent_id rows, "
            f"e.g. {list(_dups)}"
        )
    ue_route = dict(zip(
        _ue_df['agent_id'].astype(int),
        _ue_df['path_node_sequence'].astype(str),
    ))
    del _ue_df
    # MINOR 1 fix: catch NaN/empty paths that .astype(str) turns into 'nan'
    _bad = [k for k, v in ue_route.items() if not v or not v.strip() or v == 'nan']
    if _bad:
        raise ValueError(
            f"ue_routes.csv ({ue_routes_path!r}) has empty/invalid "
            f"path_node_sequence for {len(_bad)} agent(s), e.g. {_bad[:5]}"
        )
    member_set = set(int(x) for x in member)
    _missing_guided = member_set - set(ue_route.keys())
    if _missing_guided:
        raise KeyError(
            f"ue_routes.csv ({ue_routes_path!r}) is missing UE routes for "
            f"{len(_missing_guided)} guided agent(s), e.g. "
            f"{sorted(_missing_guided)[:5]}. The CSV must cover every guided "
            "member and must come from the SAME demand file as "
            "input_agent_initial.csv (36k vs 54k alignment by agent_id)."
        )

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



        for j in member:

            # UE-guidance variant: guided members receive their UE
            # equilibrium route from ue_routes.csv instead of the
            # tournament-selected UO route.  Fail fast on a missing
            # agent_id so a silent UO fallback can never contaminate
            # the guidance.

            jj = int(j)

            if jj not in ue_route:

                raise KeyError(

                    "Agent "

                    f"{jj} (bigloop={bigloop}) missing from UE routes "

                    f"file {ue_routes_path!r}. ue_routes.csv must be "

                    "generated from the SAME demand file as "

                    "input_agent_initial.csv (36k vs 54k alignment by "

                    "agent_id)."

                )



            inputagent[j][11] = ue_route[jj]



        columns = ['agent_id', 'tour_id', 'from_zone_id', 'to_zone_id', 'from_origin_node_id', 'to_destination_node_id', 'departure_time_in_min','demand_type', 'PCE', 'information_type', 'vehicle_age', 'path_node_sequence', 'vehicle_type', 'pricing_type', 'value_of_time']



        pd.DataFrame(

            inputagent,

            columns =columns

        ).to_csv(

            'DTALite_Files/input_agent.csv', index = False

        )



        subprocess.run(

            DTALITE_CMD, cwd = 'DTALite_Files', check = True,

            **dtalite_kwargs()

        )
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



        # MATLAB reads output_LinkMOE.csv col 8 (total_link_volume, one row
        # per link), not output_LinkTDMOE.csv col 8 (link x timestamp
        # in-volume). Select the column by name, not position.
        dd = pd.read_csv('DTALite_Files/output_LinkMOE.csv')



        linkV.append(dd['total_link_volume'].to_numpy())



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



    # UE-guidance variant: guided members receive their UE route in the
    # predict phase; non-members keep the tournament-predicted UO path.
    # (The comassignment "actual" phase below is deliberately untouched:
    # it models real user reaction to the guidance.)
    for j in range(dsize):

        jj = int(j)

        if jj in member_set:

            if jj not in ue_route:

                raise KeyError(

                    "Agent "

                    f"{jj} (bigloop={bigloop}) missing from UE routes "

                    f"file {ue_routes_path!r}."

                )

            inputagent[j][11] = ue_route[jj]

        else:

            inputagent[j][11] = choiceset[

                int(pre_choice[j]), int(pre_rposition[j])

            ]



    pd.DataFrame(inputagent, columns=columns).to_csv('DTALite_Files/input_agent.csv', index = False)



    subprocess.run(

        DTALITE_CMD, cwd = 'DTALite_Files', check = True,

            **dtalite_kwargs()

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



        inputagent[j][11] = choiceset[

            int(com_choice[j]), int(com_rposition[j])

        ]



    pd.DataFrame(inputagent, columns=columns).to_csv('DTALite_Files/input_agent.csv', index = False)



    subprocess.run(

        DTALITE_CMD, cwd = 'DTALite_Files', check = True,

            **dtalite_kwargs()

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

            'rprime': stringify_rprime_keys(rprime),

            'linkV': np.array(linkV, dtype = object),

            'choice': com_choice,

            'position': com_rposition,

            'pre_choice': pre_choice

        }

    )
    _pbar.set_description(f"Bigloop {bigloop} done")
    _pbar.update(1)
    _pbar.close()
