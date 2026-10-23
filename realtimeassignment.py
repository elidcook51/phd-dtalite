import numpy as np
import pandas as pd

import netconfig
from scipy.io import savemat
from traveltimecal import (
    traveltimecal,
    traveltimecal_fast,
    traveltimecal_fastv2,
    calculate_path_result_guarded,
)

def _normalize_nodes(path):
    """Return the node tuple for a choiceset/output path string, or None."""
    if path is None:
        return None
    if isinstance(path, np.ndarray):
        if path.size == 0:
            return None
        if path.size == 1:
            path = path.item()
        else:
            return None
    try:
        if pd.isna(path):
            return None
    except (TypeError, ValueError):
        pass
    if not isinstance(path, str):
        return None
    try:
        nodes = tuple(
            int(float(part.strip()))
            for part in path.split(";")
            if part.strip()
        )
    except (TypeError, ValueError):
        return None
    if len(nodes) < 2:
        return None
    return nodes


def _dict_to_dense(d, shape):
    """Convert a {(od, route): value} dict to a dense ndarray (NaN = missing)."""
    arr = np.full(shape, np.nan)
    for (od_idx, route_idx), value in d.items():
        arr[od_idx, route_idx] = value
    return arr


def comrealtimeassignment(
    itr,
    outer_itr,
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
):
    """Real-time route choice for the combined assignment.

    Faithful port of MATLAB ``comrealtimeassignment.m``. Returns six values
    ``(rtchoice, updaterposition, tt, b, p_agent, pre_choice)``, matching
    MATLAB's return list.

    This port keeps the transposed choiceset convention used throughout the
    Python code (first index = OD index, second index = route index), so
    ``rtchoice`` holds OD indices and ``updaterposition`` holds route
    indices -- the names are swapped relative to MATLAB but every consumer
    (``comassignment``'s ``choiceset[final_choice, final_rposition]`` lookup)
    is consistent with it.

    Realtime agents run a pairwise utility tournament over route-attribute
    differences (no intercept: "real weights don't have intercept").
    ``pre_choice`` independently tracks the minimum-travel-time route.
    Non-realtime agents keep their ``nchoice`` / ``rposition`` values.
    """

    onlylike0 = []
    onlylike1 = []

    choiceset = np.asarray(choiceset, dtype=object)
    routelocation = np.asarray(routelocation)
    n_od_rows, n_route_cols = choiceset.shape

    # --------------------------------------------------
    # Agent tables (MATLAB: input_agent_initial.csv / output_agent.csv)
    # --------------------------------------------------
    S = pd.read_csv("DTALite_Files/input_agent_initial.csv")
    agent_id_all = S.iloc[:, 0].to_numpy()
    agent_dep_all = pd.to_numeric(
        S.iloc[:, 6], errors="coerce"
    ).to_numpy(dtype=float)
    _oc = netconfig.OD_COLS
    agent_od_all = S.iloc[:, [_oc[0], _oc[1]]].to_numpy(dtype=float)

    # Observed travel times from the previous DTALite run, row-aligned with
    # the input table (DTALite preserves input row order).
    out_tt = None
    # Group observed travel times by RAW path string, matching MATLAB
    # comrealtimeassignment.m:115 exactly: find(strcmp(path,agentpath)).
    # MATLAB then filters by floor(INPUT departure)==k+359 (line 142).
    grouped_observations = {}
    if outer_itr != 1:
        T = pd.read_csv("DTALite_Files/output_agent.csv")
        n_rows = min(len(S), len(T))
        # MATLAB: agent_tt = T{:,13} (travel time), agentpath = T{:,30}.
        out_tt = pd.to_numeric(
            T.iloc[:n_rows, 12], errors="coerce"
        ).to_numpy(dtype=float)
        out_paths = T.iloc[:n_rows, 29].astype(object).to_numpy()
        in_dep = agent_dep_all[:n_rows]
        # Key: raw path string -> list of (input_departure, travel_time)
        for p_str, tt_val, dep_val in zip(out_paths, out_tt, in_dep):
            if not np.isfinite(tt_val) or not np.isfinite(dep_val):
                continue
            raw = str(p_str).strip() if isinstance(p_str, str) else str(p_str)
            if not raw:
                continue
            grouped_observations.setdefault(raw, []).append(
                (float(dep_val), float(tt_val)))

    # --------------------------------------------------
    # TD-link table
    # (MATLAB: 'Each iteration.csv' on itr==1 else 'output_LinkTDMOE.csv')
    # --------------------------------------------------
    if netconfig.is_baltimore() and outer_itr == 1:
        # Free-flow seed (see assignment.py): empty TDlink table.
        TDlink = np.empty((0, 4))
    else:
        td_file = (
            "DTALite_Files/Each iteration.csv"
            if outer_itr == 1
            else "DTALite_Files/output_LinkTDMOE.csv"
        )
        TDlink = pd.read_csv(
            td_file, usecols=[0, 1, 4, 5], nrows=num_tdlink_rows
        ).to_numpy(dtype=float)
    b = TDlink[:, 0].copy()

    # --------------------------------------------------
    # Link table (MATLAB: xlsread('SiouxFalls_net',1,'B90:E165'))
    # --------------------------------------------------
    if netconfig.is_baltimore():
        length = pd.read_csv(
            "DTALite_Files/input_link.csv",
            usecols=["from_node_id", "to_node_id", "length", "free_speed"],
        ).to_numpy(dtype=float)
    else:
        length = pd.read_excel(
            "DTALite_Files/SiouxFalls_net.xlsx", sheet_name=0,
            usecols="B:E", skiprows=89, nrows=76, header=None,
        ).to_numpy(dtype=float)

    link_len = {}
    for row in length:
        if np.all(np.isfinite(row[:4])):
            link_len[(int(row[0]), int(row[1]))] = float(row[2])

    # Dict lookups for calculate_path_result_guarded (shared with
    # realtimeassignment_fast): (minute, from, to) -> travel time and
    # (from, to) -> (length, speed limit). Same tables, same layouts.
    tdlink_map = {
        (
            int(row[2]),
            int(row[0]),
            int(row[1]),
        ): float(row[3])
        for row in TDlink
        if np.all(np.isfinite(row[:4]))
    }
    length_map = {
        (
            int(row[0]),
            int(row[1]),
        ): (
            float(row[2]),
            float(row[3]),
        )
        for row in length
        if np.all(np.isfinite(row[:4]))
    }
    # Results are valid only for this call's TDlink table.
    path_result_cache = {}

    # --------------------------------------------------
    # Route attributes: len (distance) and nc (link count)
    # --------------------------------------------------
    len_mat = np.zeros((n_od_rows, n_route_cols))
    nc = np.zeros((n_od_rows, n_route_cols))
    path_nodes = np.empty((n_od_rows, n_route_cols), dtype=object)
    path_nodes.fill(None)

    for od_idx in range(n_od_rows):
        for route_idx in range(n_route_cols):
            nodes = _normalize_nodes(choiceset[od_idx, route_idx])
            if nodes is None:
                continue
            path_nodes[od_idx, route_idx] = nodes
            total = 0.0
            for a, bb in zip(nodes[:-1], nodes[1:]):
                total += link_len.get((a, bb), 0.0)
            len_mat[od_idx, route_idx] = total
            nc[od_idx, route_idx] = len(nodes) - 1

    gas = 3.0

    # Single time step, like MATLAB's `for k=itr*phlength:itr*phlength`.
    period_key = int(itr * phlength)
    timestamp = period_key + netconfig.PERIOD_BASE

    # --------------------------------------------------
    # Route travel time / planning time / fuel cost
    # --------------------------------------------------
    tt = {}
    pltt = {}
    fuelcost = {}

    for od_idx in range(n_od_rows):
        for route_idx in range(n_route_cols):
            nodes = path_nodes[od_idx, route_idx]
            if nodes is None:
                continue
            path = choiceset[od_idx, route_idx]

            observed = None
            if outer_itr != 1:
                # MATLAB: agents on this exact (raw string) path departing at
                # floor(departure) == (k+359); tt = mean, pltt = max.
                cands = grouped_observations.get(str(path).strip())
                if cands:
                    vals = np.array([
                        tt for dep, tt in cands
                        if np.floor(dep) == timestamp and np.isfinite(tt)
                    ])
                    if vals.size:
                        observed = vals

            if observed is None:
                tt_val, fc_val = calculate_path_result_guarded(
                    nodes, timestamp, length_map, tdlink_map, gas,
                    path_result_cache,
                )
                tt[(od_idx, route_idx)] = tt_val
                pltt[(od_idx, route_idx)] = tt_val
                fuelcost[(od_idx, route_idx)] = fc_val
            else:
                tt[(od_idx, route_idx)] = float(np.mean(observed))
                pltt[(od_idx, route_idx)] = float(np.max(observed))
                _, fc_val = calculate_path_result_guarded(
                    nodes, timestamp, length_map, tdlink_map, gas,
                    path_result_cache,
                )
                fuelcost[(od_idx, route_idx)] = fc_val

    # --------------------------------------------------
    # Agents departing in this phase
    # (MATLAB: (TIME_START+phlength*(itr-1), TIME_START+phlength*itr])
    # --------------------------------------------------
    period_mask = (
        (agent_dep_all > netconfig.TIME_START + phlength * (itr - 1))
        & (agent_dep_all <= netconfig.TIME_START + phlength * itr)
    )
    p_idx = np.where(period_mask)[0]
    p_agent = np.column_stack([agent_id_all[p_idx], agent_dep_all[p_idx]])

    # -1 marks "no valid route found". Every period agent is
    # pre-filled with its existing (nchoice, rposition) first, so an
    # unprocessable realtime agent keeps its prior assignment instead
    # of the old silent (0, 0) default -- a valid choiceset index.
    # Prefill by FILE ROW (s_row), not by agent_id: nchoice,
    # rposition and user are all row-indexed, but agent IDs are
    # only row indices on Sioux Falls. On Baltimore (IDs like
    # 48560234) the old agent_id indexing failed its guard for
    # every agent, leaving -1, and comassignment then wrote
    # choiceset[-1, -1] -- the last OD's last route -- for every
    # non-realtime agent, which DTALite rejects as infeasible.
    rtchoice = np.full(len(p_idx), -1, dtype=int)
    updaterposition = np.full(len(p_idx), -1, dtype=int)
    pre_choice = np.zeros(len(p_idx), dtype=int)

    realtime_set = set(int(x) for x in np.asarray(realtime_user).ravel())
    nchoice_arr = np.asarray(nchoice).reshape(-1)
    rposition_arr = np.asarray(rposition).reshape(-1)

    for row, s_row in enumerate(p_idx):
        agent_id = int(agent_id_all[s_row])

        if not (0 <= s_row < nchoice_arr.size
                and 0 <= s_row < rposition_arr.size):
            raise ValueError(
                "agent_id %d has row %d outside nchoice/rposition "
                "size (%d, %d); refusing to keep -1"
                % (agent_id, s_row, nchoice_arr.size,
                  rposition_arr.size))
        rtchoice[row] = int(nchoice_arr[s_row])
        updaterposition[row] = int(rposition_arr[s_row])

    if (rtchoice < 0).any() or (updaterposition < 0).any():
        bad = int(np.where(rtchoice < 0)[0][0]) \
            if (rtchoice < 0).any() \
            else int(np.where(updaterposition < 0)[0][0])
        raise ValueError(
            "prefill left -1 for period agent row %d (agent_id %d); "
            "-1 must never reach comassignment's choiceset lookup"
            % (bad, int(agent_id_all[p_idx[bad]])))

    for row, s_row in enumerate(p_idx):
        agent_id = int(agent_id_all[s_row])

        # Non-realtime agents keep the pre-filled assignment above.
        if agent_id not in realtime_set:
            continue

        matches = np.where(
            (routelocation[:, 0] == agent_od_all[s_row, 0])
            & (routelocation[:, 1] == agent_od_all[s_row, 1])
        )[0]
        if len(matches) == 0:
            continue

        # OD index (first dim of the transposed choiceset).
        f = int(routelocation[matches[0], 4])

        # Route indices available for this OD, ascending (MATLAB j = 1..).
        route_ids = sorted(
            {
                int(routelocation[m, 3])
                for m in matches
                if _normalize_nodes(
                    choiceset[f, int(routelocation[m, 3])]
                )
                is not None
            }
        )
        if not route_ids:
            continue

        def attrs(r):
            return np.array(
                [
                    len_mat[f, r],
                    tt[(f, r)],
                    pltt[(f, r)],
                    fuelcost[(f, r)],
                    nc[f, r],
                ]
            )

        # The agent's own preference row: user is row-indexed,
        # so index by s_row, not agent_id (Baltimore IDs are not
        # row indices; agent_id indexing would IndexError here).
        pref = int(user[s_row, 1])
        if pref < 1:
            raise ValueError(
                f"preference id {pref} for agent {agent_id} is below 1; "
                "weights rows are 1-based, so weights[pref - 1] would "
                "silently wrap to the last row"
            )
        wrow = weights[pref - 1, :5]
        dataset2 = pref >= 16650

        b1 = attrs(route_ids[0])
        pre_b1 = b1.copy()
        croute = route_ids[0]
        pre_croute = route_ids[0]

        for r in route_ids[1:]:
            b0 = attrs(r)
            diff = b1 - b0

            if (row + 1) in onlylike1:
                choice = 0
            elif (row + 1) in onlylike0:
                choice = 1
            else:
                if dataset2:
                    bscale = (diff - meanstd2[0, :]) / meanstd2[1, :]
                else:
                    bscale = diff
                # No intercept: "real weights don't have intercept".
                ojvalue = float(np.sum(bscale * wrow))

                choice = 0 if abs(ojvalue - 1) > abs(ojvalue + 1) else 1

            if choice != 1:
                b1 = b0
                croute = r

            # Parallel minimum-travel-time tournament.
            if not pre_b1[1] < b0[1]:
                pre_croute = r
                pre_b1 = b0

        rtchoice[row] = f
        updaterposition[row] = croute
        pre_choice[row] = pre_croute

    if itr == int(np.floor(60.0 / phlength)):
        savemat(
            f"DTALite_Files/pathinfo_comrealass{bigloop}.mat",
            {
                "len": len_mat,
                "tt": _dict_to_dense(tt, (n_od_rows, n_route_cols)),
                "pltt": _dict_to_dense(pltt, (n_od_rows, n_route_cols)),
                "fuelcost": _dict_to_dense(
                    fuelcost, (n_od_rows, n_route_cols)
                ),
                "nc": nc,
            },
        )

    return rtchoice, updaterposition, tt, b, p_agent, pre_choice




def realtimeassignment_fast(
    itr,
    outer_itr,
    choiceset,
    routelocation,
    phlength,
    realtime_user,
    nchoice,
    rposition,
    bigloop,
    dsize,
    num_tdlink_rows,
):
    """
    Optimized replacement for realtimeassignment().

    Parameters
    ----------
    itr : int
        Current phase sub-iteration (1..phase); selects the departure
        window within the hour.
    outer_itr : int
        Outer MSA iteration. Gates reads of previous-iteration DTALite
        outputs (output_agent.csv, TD-link tables).

    choiceset : array-like, shape (n_od_pairs, n_routes)
        Transposed convention (same as comrealtimeassignment and
        msa.py): first index = OD index, second index = route index.
        Each nonempty element is a path string such as "1;2;3;".

    routelocation : ndarray
        Expected columns:
            0: origin node
            1: destination node
            3: choiceset route index (second dim)
            4: choiceset OD index (first dim)

    phlength : float
        Assignment period length in minutes.

    realtime_user : array-like
        Agent IDs belonging to the realtime-information group.

    nchoice : array-like
        Existing route choices for non-realtime agents.

    rposition : array-like
        Existing OD-column positions for non-realtime agents.

    bigloop : int
        Outer-loop number used in the MAT-file name.

    dsize : int
        Retained for compatibility with the original function.
        It is not needed internally.

    num_tdlink_rows : int
        Maximum number of rows to read from the TD-link file.

    Returns
    -------
    rtchoice : ndarray, shape (n_period_agents,)
        Choiceset OD index (first dim) for each agent departing in
        this period; -1 where no valid route was found.

    updaterposition : ndarray, shape (n_period_agents,)
        Choiceset route index (second dim) for each selected route;
        -1 where no valid route was found.

    tt : list[list[dict or None]]
        Mean route travel time indexed as tt[od][route][period].

    b : ndarray
        First TD-link column, preserved from the original function.

    p_agent : ndarray
        Agents departing during the current assignment period.
    """

    # Retain the parameter in the signature for drop-in compatibility.
    _ = dsize

    # ---------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------

    def valid_path(path):
        """Return True when a choiceset cell contains a usable path."""
        if path is None:
            return False

        if isinstance(path, np.ndarray):
            if path.size == 0:
                return False

            # Handle a one-element object/string ndarray.
            if path.size == 1:
                path = path.item()
            else:
                return False

        if pd.isna(path):
            return False

        return isinstance(path, str) and bool(path.strip())

    def normalize_path(path):
        """
        Normalize path formatting for reliable matching between
        choiceset paths and output_agent paths.
        """
        if not valid_path(path):
            return None

        try:
            nodes = tuple(
                int(float(value.strip()))
                for value in str(path).split(";")
                if value.strip()
            )
        except (TypeError, ValueError):
            return None

        if len(nodes) < 2:
            return None

        return nodes

    # ---------------------------------------------------------
    # Validate and normalize choiceset
    # ---------------------------------------------------------

    choiceset = np.asarray(choiceset, dtype=object)
    routelocation = np.asarray(routelocation)

    if choiceset.ndim != 2:
        raise ValueError(
            "choiceset must be a two-dimensional array with shape "
            "(number of OD pairs, number of routes)."
        )

    if routelocation.ndim != 2 or routelocation.shape[1] < 5:
        raise ValueError(
            "routelocation must be a two-dimensional array with at "
            "least five columns."
        )

    n_ods, n_routes = choiceset.shape
    period_key = int(itr * phlength)
    td_departure_time = period_key + netconfig.PERIOD_BASE
    gas = 3.0

    # ---------------------------------------------------------
    # Read initial agent information
    # ---------------------------------------------------------

    _oc2 = netconfig.OD_COLS
    agent_df = pd.read_csv(
        "DTALite_Files/input_agent_initial.csv",
        usecols=[0, _oc2[0], _oc2[1], 6],
    )

    # Preserve the same information as the original matrices.
    agent = agent_df.iloc[:, [0, 3]].to_numpy(dtype=float)
    agent_od_values = agent_df.iloc[:, [1, 2]].to_numpy(dtype=float)

    # Do not assume agent_id always equals the DataFrame row index.
    agent_id_to_row = {
        int(agent_id): row_index
        for row_index, agent_id in enumerate(agent[:, 0])
    }

    # ---------------------------------------------------------
    # Read output-agent results from the previous assignment
    # ---------------------------------------------------------

    path_observations = {}

    if outer_itr != 1:
        output_agent_df = pd.read_csv(
            "DTALite_Files/output_agent.csv",
            usecols=[0, 9, 12, 29],
        )

        output_agent_ids = pd.to_numeric(
            output_agent_df.iloc[:, 0],
            errors="coerce",
        ).to_numpy()

        output_travel_times = pd.to_numeric(
            output_agent_df.iloc[:, 2],
            errors="coerce",
        ).to_numpy()

        output_paths = output_agent_df.iloc[:, 3].to_numpy(
            dtype=object
        )

        # Group observed travel times by RAW path string, matching MATLAB
        # realtimeassignment.m:115 exactly: findagent=find(strcmp(path,agentpath)).
        # MATLAB then filters by floor(INPUT departure)==k+359 (line 142),
        # using input_agent departure times, not output departures.
        # (The old code normalized to int-tuples, which finds matches where
        # MATLAB's strcmp finds none, e.g. '10.0;17.0;' vs '10;17;'.)
        # Key: raw path string -> list of (input_departure, travel_time).
        grouped_observations = {}

        # agent[:,0]=agent_id, agent[:,1]=input departure time
        _input_dep_by_id = {}
        for _r in range(agent.shape[0]):
            _aid = int(agent[_r, 0])
            _input_dep_by_id[_aid] = float(agent[_r, 1])

        for output_id, travel_time, path in zip(
            output_agent_ids,
            output_travel_times,
            output_paths,
        ):
            if (
                not np.isfinite(output_id)
                or not np.isfinite(travel_time)
            ):
                continue
            if not isinstance(path, str):
                path = str(path)
            raw_path = path.strip()
            if not raw_path:
                continue
            aid = int(output_id)
            inp_dep = _input_dep_by_id.get(aid)
            if inp_dep is None or not np.isfinite(inp_dep):
                continue
            grouped_observations.setdefault(raw_path, []).append(
                (float(inp_dep), float(travel_time)))

        # For the current period, filter by floor(input_dep)==td_departure_time
        # and store (mean, max). Matches MATLAB's ttloc logic.
        path_observations = {}
        for raw_path, pairs in grouped_observations.items():
            vals = [tt for dep, tt in pairs
                    if int(np.floor(dep)) == td_departure_time]
            if vals:
                path_observations[raw_path] = (
                    float(np.mean(vals)),
                    float(np.max(vals)),
                )

    # ---------------------------------------------------------
    # Read current TD-link information
    # ---------------------------------------------------------

    if netconfig.is_baltimore() and outer_itr == 1:
        # Free-flow seed (see comrealtimeassignment): empty TDlink table.
        TDlink = np.empty((0, 4))
    else:
        if outer_itr == 1:
            td_file = "DTALite_Files/Each iteration.csv"
        else:
            td_file = "DTALite_Files/output_LinkTDMOE.csv"

        td_df = pd.read_csv(
            td_file,
            usecols=[0, 1, 4, 5],
            nrows=num_tdlink_rows,
        )

        TDlink = td_df.to_numpy(dtype=float)
    b = TDlink[:, 0].copy()

    # TDlink columns after selecting [0, 1, 4, 5]:
    # 0 = from node
    # 1 = to node
    # 2 = time period
    # 3 = travel time
    tdlink_map = {
        (
            int(row[2]),
            int(row[0]),
            int(row[1]),
        ): float(row[3])
        for row in TDlink
        if np.all(np.isfinite(row[:4]))
    }

    # ---------------------------------------------------------
    # Read static link information
    # ---------------------------------------------------------

    link_df = pd.read_csv(
        "DTALite_Files/input_link.csv",
        usecols=[
            "from_node_id",
            "to_node_id",
            "length",
            "speed_limit",
        ],
    )

    length_data = link_df.to_numpy(dtype=float)

    length_map = {
        (
            int(row[0]),
            int(row[1]),
        ): (
            float(row[2]),
            float(row[3]),
        )
        for row in length_data
        if np.all(np.isfinite(row[:4]))
    }

    # Results are valid only for the current TDlink table, so this
    # cache is deliberately local to the function call.
    path_result_cache = {}

    # ---------------------------------------------------------
    # Parse unique paths and calculate static route attributes
    # ---------------------------------------------------------

    len_arr = np.zeros((n_ods, n_routes), dtype=float)
    nc = np.zeros((n_ods, n_routes), dtype=float)

    path_nodes_matrix = np.empty((n_ods, n_routes), dtype=object)
    path_nodes_matrix.fill(None)

    # Cache static route attributes for duplicate path strings.
    static_path_cache = {}

    for od_index in range(n_ods):
        for route_index in range(n_routes):
            path_nodes = normalize_path(
                choiceset[od_index, route_index]
            )

            if path_nodes is None:
                continue

            path_nodes_matrix[od_index, route_index] = path_nodes

            static_result = static_path_cache.get(path_nodes)

            if static_result is None:
                total_length = 0.0

                for link_key in zip(
                    path_nodes[:-1],
                    path_nodes[1:],
                ):
                    link_info = length_map.get(link_key)

                    if link_info is not None:
                        total_length += link_info[0]

                link_count = len(path_nodes) - 1
                static_result = (total_length, link_count)
                static_path_cache[path_nodes] = static_result

            len_arr[od_index, route_index] = static_result[0]
            nc[od_index, route_index] = static_result[1]

    # ---------------------------------------------------------
    # Calculate travel time, maximum travel time, and fuel cost
    # ---------------------------------------------------------

    tt = [
        [None for _ in range(n_routes)]
        for _ in range(n_ods)
    ]

    pltt = [
        [None for _ in range(n_routes)]
        for _ in range(n_ods)
    ]

    fuelcost = [
        [None for _ in range(n_routes)]
        for _ in range(n_ods)
    ]

    # Dense arrays are faster for best-route selection, while the
    # nested list/dictionary objects preserve the original output type.
    current_tt = np.full(
        (n_ods, n_routes),
        np.inf,
        dtype=float,
    )

    current_pltt = np.full(
        (n_ods, n_routes),
        np.inf,
        dtype=float,
    )

    current_fuelcost = np.full(
        (n_ods, n_routes),
        np.nan,
        dtype=float,
    )

    # Calculate each unique path result only once.
    unique_route_results = {}

    for od_index in range(n_ods):
        for route_index in range(n_routes):
            path_nodes = path_nodes_matrix[
                od_index, route_index
            ]

            if path_nodes is None:
                continue

            _raw_path = choiceset[od_index, route_index]
            if not isinstance(_raw_path, str):
                _raw_path = str(_raw_path)
            observed_result = path_observations.get(_raw_path.strip())

            calculated_result = unique_route_results.get(path_nodes)

            if calculated_result is None:
                calculated_result = calculate_path_result_guarded(
                    path_nodes,
                    td_departure_time,
                    length_map,
                    tdlink_map,
                    gas,
                    path_result_cache,
                )
                unique_route_results[path_nodes] = calculated_result

            calculated_tt, fuel_value = calculated_result

            if observed_result is None:
                mean_tt = calculated_tt
                maximum_tt = calculated_tt
            else:
                mean_tt, maximum_tt = observed_result

            mean_tt = float(mean_tt)
            maximum_tt = float(maximum_tt)
            fuel_value = float(fuel_value)

            tt[od_index][route_index] = {
                period_key: mean_tt
            }

            pltt[od_index][route_index] = {
                period_key: maximum_tt
            }

            fuelcost[od_index][route_index] = {
                period_key: fuel_value
            }

            current_tt[od_index, route_index] = mean_tt
            current_pltt[od_index, route_index] = maximum_tt
            current_fuelcost[od_index, route_index] = fuel_value

    # ---------------------------------------------------------
    # Select agents departing in the current period
    # ---------------------------------------------------------

    period_start = netconfig.TIME_START + phlength * (itr - 1)
    period_end = netconfig.TIME_START + phlength * itr

    period_mask = (
        (agent[:, 1] > period_start)
        & (agent[:, 1] <= period_end)
    )

    p_agent = agent[period_mask]

    rtchoice = np.full(
        p_agent.shape[0],
        -1,
        dtype=int,
    )

    updaterposition = np.full(
        p_agent.shape[0],
        -1,
        dtype=int,
    )

    nchoice_array = np.asarray(nchoice).reshape(-1)
    rposition_array = np.asarray(rposition).reshape(-1)

    # Give every agent its existing assignment first.
    # Realtime agents will be updated only when a valid better route exists.
    for period_row, agent_record in enumerate(p_agent):
        agent_id = int(agent_record[0])

        if (
            0 <= agent_id < nchoice_array.size
            and 0 <= agent_id < rposition_array.size
        ):
            rtchoice[period_row] = int(
                nchoice_array[agent_id]
            )
            updaterposition[period_row] = int(
                rposition_array[agent_id]
            )

    # ---------------------------------------------------------
    # Build constant-time lookup structures
    # ---------------------------------------------------------

    realtime_set = {
        int(agent_id)
        for agent_id in np.asarray(realtime_user).ravel()
    }

    # ---------------------------------------------------------
    # Assign routes to current-period agents
    # ---------------------------------------------------------

    for period_row, agent_record in enumerate(p_agent):
        agent_id = int(agent_record[0])

        # Non-realtime agents already carry their existing assignment
        # from the pre-fill above; only realtime agents are updated.
        if agent_id not in realtime_set:
            continue

        agent_row = agent_id_to_row.get(agent_id)

        if agent_row is None:
            continue

        origin = int(agent_od_values[agent_row, 0])
        destination = int(agent_od_values[agent_row, 1])

        matches = np.where(
            (routelocation[:,0] == origin) & 
            (routelocation[:,1] == destination) 
        )[0]

        if len(matches) == 0:
            continue

        best_row = -1
        best_col = -1
        best_tt = np.inf

        for idx in matches:
            row = int(routelocation[idx, 4])
            col = int(routelocation[idx, 3])

            path = choiceset[row, col]

            if not isinstance(path, str):
                continue

            nodes = [int(x) for x in path.split(';') if x.strip()]

            if len(nodes) < 2:
                continue

            if (
                nodes[0] != origin or
                nodes[-1] != destination
            ):
                continue

            if not np.isfinite(current_tt[row, col]):
                continue

            cur_tt = current_tt[row, col]

            if cur_tt < best_tt:
                best_tt = cur_tt
                best_row = row
                best_col = col

        if best_row < 0:
            continue

        rtchoice[period_row] = best_row
        updaterposition[period_row] = best_col

    # ---------------------------------------------------------
    # Save final period route information
    # ---------------------------------------------------------

    if itr == int(np.floor(60.0 / phlength)):
        output_name = (
            f"DTALite_Files/pathinfo_realass{bigloop}.mat"
        )

        # Convert nested dictionaries to dense arrays before saving.
        # MATLAB cannot naturally represent the Python dictionary keys.
        savemat(
            output_name,
            {
                "len": len_arr,
                "tt": current_tt,
                "pltt": current_pltt,
                "fuelcost": current_fuelcost,
                "nc": nc,
                "period_key": np.array(
                    [[period_key]],
                    dtype=float,
                ),
            },
        )

    return rtchoice, updaterposition, tt, b, p_agent