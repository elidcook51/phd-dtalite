import numpy as np
import pandas as pd
from scipy.io import savemat
from traveltimecal import traveltimecal, traveltimecal_fast, traveltimecal_fastv2

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
    agent_od_all = S.iloc[:, [4, 5]].to_numpy(dtype=float)

    # Observed travel times from the previous DTALite run, row-aligned with
    # the input table (DTALite preserves input row order).
    out_tt = None
    out_path_nodes = None
    in_dep = None
    if itr != 1:
        T = pd.read_csv("DTALite_Files/output_agent.csv")
        n_rows = min(len(S), len(T))
        # MATLAB: agent_tt = T{:,13} (travel time), agentpath = T{:,30}.
        out_tt = pd.to_numeric(
            T.iloc[:n_rows, 12], errors="coerce"
        ).to_numpy(dtype=float)
        out_paths = T.iloc[:n_rows, 29].astype(object).to_numpy()
        out_path_nodes = np.array(
            [_normalize_nodes(p) for p in out_paths], dtype=object
        )
        in_dep = agent_dep_all[:n_rows]

    # --------------------------------------------------
    # TD-link table
    # (MATLAB: 'Each iteration.csv' on itr==1 else 'output_linkTDMOE.csv')
    # --------------------------------------------------
    td_file = (
        "DTALite_Files/Each iteration.csv"
        if itr == 1
        else "DTALite_Files/output_linkTDMOE.csv"
    )
    TDlink = pd.read_csv(
        td_file, usecols=[0, 1, 4, 5], nrows=num_tdlink_rows
    ).to_numpy(dtype=float)
    b = TDlink[:, 0].copy()

    # --------------------------------------------------
    # Link table (MATLAB: xlsread('SiouxFalls_net',1,'B90:E165'))
    # --------------------------------------------------
    length = pd.read_excel(
        "DTALite_Files/SiouxFalls_net.xlsx", sheet_name=0,
        usecols="B:E", skiprows=89, nrows=76, header=None,
    ).to_numpy(dtype=float)

    link_len = {}
    for row in length:
        if np.all(np.isfinite(row[:4])):
            link_len[(int(row[0]), int(row[1]))] = float(row[2])

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
    timestamp = period_key + 359

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
            if itr != 1:
                # MATLAB: agents on this exact path departing at
                # floor(departure) == (k+359); tt = mean, pltt = max.
                on_path = np.array(
                    [
                        pn is not None and pn == nodes
                        for pn in out_path_nodes
                    ]
                )
                dep_match = np.floor(in_dep[on_path]) == timestamp
                vals = out_tt[on_path][dep_match]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    observed = vals

            if observed is None:
                tt_val, fc_val = traveltimecal_fastv2(
                    timestamp, TDlink, path, length, gas, itr, 0
                )
                tt[(od_idx, route_idx)] = tt_val
                pltt[(od_idx, route_idx)] = tt_val
                fuelcost[(od_idx, route_idx)] = fc_val
            else:
                tt[(od_idx, route_idx)] = float(np.mean(observed))
                pltt[(od_idx, route_idx)] = float(np.max(observed))
                _, fc_val = traveltimecal_fastv2(
                    timestamp, TDlink, path, length, gas, itr, 0
                )
                fuelcost[(od_idx, route_idx)] = fc_val

    # --------------------------------------------------
    # Agents departing in this phase
    # (MATLAB: (360+phlength*(itr-1), 360+phlength*itr])
    # --------------------------------------------------
    period_mask = (
        (agent_dep_all > 360.0 + phlength * (itr - 1))
        & (agent_dep_all <= 360.0 + phlength * itr)
    )
    p_idx = np.where(period_mask)[0]
    p_agent = np.column_stack([agent_id_all[p_idx], agent_dep_all[p_idx]])

    rtchoice = np.zeros(len(p_idx), dtype=int)
    updaterposition = np.zeros(len(p_idx), dtype=int)
    pre_choice = np.zeros(len(p_idx), dtype=int)

    realtime_set = set(int(x) for x in np.asarray(realtime_user).ravel())
    nchoice_arr = np.asarray(nchoice).reshape(-1)
    rposition_arr = np.asarray(rposition).reshape(-1)

    for row, s_row in enumerate(p_idx):
        agent_id = int(agent_id_all[s_row])

        if agent_id not in realtime_set:
            rtchoice[row] = int(nchoice_arr[agent_id - 1])
            updaterposition[row] = int(rposition_arr[agent_id - 1])
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

        # The agent's own preference row (MATLAB 1-based IDs).
        pref = int(user[agent_id, 1])
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
        Current assignment iteration.

    choiceset : array-like, shape (n_routes, n_od_pairs)
        Each nonempty element is a path string such as "1;2;3;".

    routelocation : ndarray
        Expected columns:
            0: origin node
            1: destination node
            3: choiceset OD-column index

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
    rtchoice : ndarray, shape (n_period_agents, 1)
        Selected route index for each agent departing in this period.

    updaterposition : ndarray, shape (n_period_agents, 1)
        Choiceset OD-column index for each selected route.

    tt : list[list[dict or None]]
        Mean route travel time indexed as tt[route][od][period].

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

    def calculate_path_result(path_nodes, departure_time):
        """
        Calculate time-dependent path travel time and fuel cost.

        This uses local lookup tables that are rebuilt for the current
        function call. That is important because TDlink changes between
        DTALite iterations.
        """
        cache_key = (int(np.floor(departure_time)), path_nodes)

        cached_result = path_result_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        total_time = 0.0
        fuel_cost = 0.0

        for from_node, to_node in zip(path_nodes[:-1], path_nodes[1:]):
            link_info = length_map.get((from_node, to_node))

            current_time = int(
                np.floor(float(departure_time) + total_time)
            )

            travel_time = tdlink_map.get(
                (current_time, from_node, to_node)
            )

            if travel_time is None or not np.isfinite(travel_time):
                if link_info is None:
                    travel_time = 1.0
                else:
                    link_length, speed_limit = link_info

                    if speed_limit > 0:
                        travel_time = (
                            link_length / speed_limit * 60.0
                        )
                    else:
                        travel_time = 1.0

            travel_time = float(travel_time)

            # Protect the speed and fuel calculations from zero or
            # negative TD-link travel times.
            if travel_time <= 0:
                if link_info is not None and link_info[1] > 0:
                    travel_time = (
                        link_info[0] / link_info[1] * 60.0
                    )
                else:
                    travel_time = 1.0

            total_time += travel_time

            if link_info is not None:
                link_length = link_info[0]
                actual_speed = link_length / travel_time * 60.0

                if actual_speed > 40.0:
                    mpg = 32.0
                elif actual_speed < 25.0:
                    mpg = 22.0
                else:
                    mpg = 27.0

                fuel_cost += link_length / mpg * gas

        result = (total_time, fuel_cost)
        path_result_cache[cache_key] = result
        return result

    # ---------------------------------------------------------
    # Validate and normalize choiceset
    # ---------------------------------------------------------

    choiceset = np.asarray(choiceset, dtype=object)
    routelocation = np.asarray(routelocation)

    if choiceset.ndim != 2:
        raise ValueError(
            "choiceset must be a two-dimensional array with shape "
            "(number of routes, number of OD columns)."
        )

    if routelocation.ndim != 2 or routelocation.shape[1] < 4:
        raise ValueError(
            "routelocation must be a two-dimensional array with at "
            "least four columns."
        )

    n_routes, n_ods = choiceset.shape
    period_key = int(itr * phlength)
    td_departure_time = period_key + 359
    gas = 3.0

    # ---------------------------------------------------------
    # Read initial agent information
    # ---------------------------------------------------------

    agent_df = pd.read_csv(
        "DTALite_Files/input_agent_initial.csv",
        usecols=[0, 4, 5, 6],
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

    if itr != 1:
        output_agent_df = pd.read_csv(
            "DTALite_Files/output_agent.csv",
            usecols=[0, 9, 12, 29],
        )

        output_agent_ids = pd.to_numeric(
            output_agent_df.iloc[:, 0],
            errors="coerce",
        ).to_numpy()

        output_departures = pd.to_numeric(
            output_agent_df.iloc[:, 1],
            errors="coerce",
        ).to_numpy()

        output_travel_times = pd.to_numeric(
            output_agent_df.iloc[:, 2],
            errors="coerce",
        ).to_numpy()

        output_paths = output_agent_df.iloc[:, 3].to_numpy(
            dtype=object
        )

        # Group observed travel times by normalized path and integer
        # departure minute. This replaces a complete output_agent scan
        # for every choiceset cell.
        grouped_observations = {}

        for output_id, departure, travel_time, path in zip(
            output_agent_ids,
            output_departures,
            output_travel_times,
            output_paths,
        ):
            if (
                not np.isfinite(output_id)
                or not np.isfinite(departure)
                or not np.isfinite(travel_time)
            ):
                continue

            path_nodes = normalize_path(path)
            if path_nodes is None:
                continue

            observation_key = (
                path_nodes,
                int(np.floor(departure)),
            )

            grouped_observations.setdefault(
                observation_key, []
            ).append(float(travel_time))

        # Store mean and maximum once so they are not recalculated for
        # duplicate paths in the choiceset.
        path_observations = {
            observation_key: (
                float(np.mean(values)),
                float(np.max(values)),
            )
            for observation_key, values
            in grouped_observations.items()
        }

    # ---------------------------------------------------------
    # Read current TD-link information
    # ---------------------------------------------------------

    if itr == 1:
        td_file = "DTALite_Files/Each iteration.csv"
    else:
        td_file = "DTALite_Files/output_linkTDMOE.csv"

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

    len_arr = np.zeros((n_routes, n_ods), dtype=float)
    nc = np.zeros((n_routes, n_ods), dtype=float)

    path_nodes_matrix = np.empty((n_routes, n_ods), dtype=object)
    path_nodes_matrix.fill(None)

    # Cache static route attributes for duplicate path strings.
    static_path_cache = {}

    for route_index in range(n_routes):
        for od_index in range(n_ods):
            path_nodes = normalize_path(
                choiceset[route_index, od_index]
            )

            if path_nodes is None:
                continue

            path_nodes_matrix[route_index, od_index] = path_nodes

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

            len_arr[route_index, od_index] = static_result[0]
            nc[route_index, od_index] = static_result[1]

    # ---------------------------------------------------------
    # Calculate travel time, maximum travel time, and fuel cost
    # ---------------------------------------------------------

    tt = [
        [None for _ in range(n_ods)]
        for _ in range(n_routes)
    ]

    pltt = [
        [None for _ in range(n_ods)]
        for _ in range(n_routes)
    ]

    fuelcost = [
        [None for _ in range(n_ods)]
        for _ in range(n_routes)
    ]

    # Dense arrays are faster for best-route selection, while the
    # nested list/dictionary objects preserve the original output type.
    current_tt = np.full(
        (n_routes, n_ods),
        np.inf,
        dtype=float,
    )

    current_pltt = np.full(
        (n_routes, n_ods),
        np.inf,
        dtype=float,
    )

    current_fuelcost = np.full(
        (n_routes, n_ods),
        np.nan,
        dtype=float,
    )

    # Calculate each unique path result only once.
    unique_route_results = {}

    for route_index in range(n_routes):
        for od_index in range(n_ods):
            path_nodes = path_nodes_matrix[
                route_index, od_index
            ]

            if path_nodes is None:
                continue

            observed_result = path_observations.get(
                (path_nodes, td_departure_time)
            )

            calculated_result = unique_route_results.get(path_nodes)

            if calculated_result is None:
                calculated_result = calculate_path_result(
                    path_nodes,
                    td_departure_time,
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

            tt[route_index][od_index] = {
                period_key: mean_tt
            }

            pltt[route_index][od_index] = {
                period_key: maximum_tt
            }

            fuelcost[route_index][od_index] = {
                period_key: fuel_value
            }

            current_tt[route_index, od_index] = mean_tt
            current_pltt[route_index, od_index] = maximum_tt
            current_fuelcost[route_index, od_index] = fuel_value

    # ---------------------------------------------------------
    # Select agents departing in the current period
    # ---------------------------------------------------------

    period_start = 360.0 + phlength * (itr - 1)
    period_end = 360.0 + phlength * itr

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
            1 <= agent_id <= nchoice_array.size
            and 1 <= agent_id <= rposition_array.size
        ):
            rtchoice[period_row] = int(
                nchoice_array[agent_id - 1]
            )
            updaterposition[period_row] = int(
                rposition_array[agent_id - 1]
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

    nchoice_array = np.asarray(nchoice).reshape(-1)
    rposition_array = np.asarray(rposition).reshape(-1)

    for period_row, agent_record in enumerate(p_agent):
        agent_id = int(agent_record[0])

        # Preserve the existing route for non-realtime agents.
        if agent_id not in realtime_set:
            if (
                1 <= agent_id <= nchoice_array.size
                and 1 <= agent_id <= rposition_array.size
            ):
                rtchoice[period_row] = int(
                    nchoice_array[agent_id - 1]
                )
                updaterposition[period_row] = int(
                    rposition_array[agent_id - 1]
                )
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