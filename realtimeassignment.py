import numpy as np
import pandas as pd
from scipy.io import savemat
from traveltimecal import traveltimecal, traveltimecal_fast, traveltimecal_fastv2

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

    onlylike0 = []
    onlylike1 = []

    # --------------------------------------------------
    # Read agent information
    # --------------------------------------------------
    S = pd.read_csv("DTALite_Files/input_agent_initial.csv")

    agent = np.zeros((len(S), 2))
    agent[:, 0] = S.iloc[:, 0]
    agent[:, 1] = S.iloc[:, 6]

    agentOD = np.zeros((len(S), 2))
    agentOD[:, 0] = S.iloc[:, 4]
    agentOD[:, 1] = S.iloc[:, 5]

    if itr != 1:
        T = pd.read_csv("DTALite_Files/output_agent.csv")

        agent_tt = T.iloc[:, 12].to_numpy()
        agentpath = T.iloc[:, 29].astype(str).tolist()

    # --------------------------------------------------
    # Read TD link information
    # --------------------------------------------------
    if itr == 1:
        td_df = pd.read_csv(
            "DTALite_Files/Each iteration.csv",
            usecols=[0, 1, 4, 5]
        )
    else:
        td_df = pd.read_csv(
            "DTALite_Files/output_linkTDMOE.csv",
            usecols=[0, 1, 4, 5]
        )

    TDlink = td_df.to_numpy()

    b = TDlink[:, 0]

    # --------------------------------------------------
    # Build TD link hash map
    # --------------------------------------------------
    print("  Building TDlink hash map...")

    tdlink_map = {}

    for row in TDlink:
        key = f"{int(row[2])}_{int(row[0])}_{int(row[1])}"
        tdlink_map[key] = row[3]

    print(f"  ✓ Hash map with {len(tdlink_map)} entries")

    # --------------------------------------------------
    # Route attributes
    # --------------------------------------------------
    gas = 3.0
    links = pd.read_csv('DTALite_Files/input_link.csv')
    
    length_data = links[['from_node_id', 'to_node_id', 'length']].to_numpy()

    n_routes = len(choiceset)
    n_ods = len(choiceset[0])

    len_mat = np.zeros((n_routes, n_ods))
    nc = np.zeros((n_routes, n_ods))

    for h in range(n_ods):

        for i in range(n_routes):

            path = choiceset[i, h]

            if not path:
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
                    len_mat[i, h] += length_data[idx[0], 2]

            nc[i, h] = len(nodes) - 1

    gas = 3

    # --------------------------------------------------
    # Travel time calculation
    # --------------------------------------------------
    tt = {}
    pltt = {}
    fuelcost = {}

    for h in range(n_ods):

        for i in range(n_routes):

            path = choiceset[i, h]

            if not path:
                continue

            m = 1

            if itr == 1:

                m = 2

            else:

                findagent = [
                    idx
                    for idx, p in enumerate(agentpath)
                    if p == path
                ]

                agentn = []

                for fa in findagent:

                    agentn.append([
                        agent[fa, 0],
                        agent[fa, 1],
                        agent_tt[fa]
                    ])

                agentn = np.array(agentn)

                m = len(agentn) + 1

            for k in [itr * phlength]:

                if m == 1:

                    tt_val, fc_val = traveltimecal_fastv2(
                        k + 899,
                        TDlink,
                        path,
                        length_data,
                        gas,
                        itr,
                        0,
                    )

                    tt[(i, h, k)] = tt_val
                    fuelcost[(i, h, k)] = fc_val
                    pltt[(i, h, k)] = tt_val

                else:

                    if itr == 1:
                        ttloc = []
                    else:

                        ttloc = np.where(
                            np.floor(agentn[:, 1]) == (k + 899)
                        )[0]

                    if len(ttloc) == 0:

                        tt_val, fc_val = traveltimecal_fastv2(
                            k + 899,
                            TDlink,
                            path,
                            length_data,
                            gas,
                            itr,
                            0,
                        )

                        tt[(i, h, k)] = tt_val
                        fuelcost[(i, h, k)] = fc_val
                        pltt[(i, h, k)] = tt_val

                    else:

                        a = agentn[ttloc, 2]

                        tt[(i, h, k)] = np.mean(a)
                        pltt[(i, h, k)] = np.max(a)

                        _, fc_val = traveltimecal_fastv2(
                            k + 899,
                            TDlink,
                            path,
                            length_data,
                            gas,
                            itr,
                            0,
                        )

                        fuelcost[(i, h, k)] = fc_val

    # --------------------------------------------------
    # Agents departing in current assignment period
    # --------------------------------------------------
    p_agent = agent[
        (agent[:, 1] <= (360 + phlength * itr))
        &
        (agent[:, 1] > (360 + phlength * (itr - 1)))
    ]

    rtchoice = np.zeros(len(p_agent), dtype=int)
    updaterposition = np.zeros(len(p_agent), dtype=int)
    pre_choice = np.zeros(len(p_agent), dtype=int)
    pre_position = np.zeros(len(p_agent), dtype = int)

    # --------------------------------------------------
    # Route choice
    # --------------------------------------------------
    for i in range(len(p_agent)):

        agent_id = int(p_agent[i, 0])

        agent_id = int(p_agent[i,0])

        rtchoice[i] = int(nchoice[agent_id])
        updaterposition[i] = int(rposition[agent_id])

        if agent_id not in realtime_user:
            continue

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

        b1 = np.array([
            len_mat[best_row, best_col],
            float(tt[(best_row, best_col, po)]),
            float(pltt[(best_row, best_col, po)]),
            float(fuelcost[(best_row, best_col, po)]),
            nc[best_row, best_col]
        ])

        pre_b1 = b1.copy()
        pre_route = best_row.copy()
        pre_col = best_col.copy()

        for route_row, route_col in available:
            if (
                choiceset[route_row, route_col] is None or
                choiceset[route_row, route_col] == ""
            ):
                continue

            b0 = np.array([

                len_mat[route_row, route_col],
                float(tt[(route_row, route_col, po)]),
                float(pltt[(route_row, route_col, po)]),
                float(fuelcost[(route_row, route_col, po)]),
                nc[route_row, route_col]
            ])

            pre_b0 = b0.copy()

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

            if pre_b1[1] >= pre_b0[1]:
                pre_b1 = pre_b0
                pre_route = route_row
                pre_col = route_col

        rtchoice[i] = best_row
        updaterposition[i] = best_col
        pre_choice[i] = pre_route
        pre_position[i] = pre_col

    # --------------------------------------------------
    # Save final iteration route information
    # --------------------------------------------------
    if itr == int(np.floor(60 / phlength)):

        savemat(
            f"pathinfo_comrealass{bigloop}.mat",
            {
                "len": len_mat,
                "tt": tt,
                "pltt": pltt,
                "fuelcost": fuelcost,
                "nc": nc,
            },
        )

    return (
        rtchoice,
        updaterposition,
        tt,
        b,
        p_agent,
        pre_choice,
        pre_position
    )


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
    td_departure_time = period_key + 899
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
            usecols=[0, 6, 12, 29],
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

    nchoice_array = np.asarray(nchoice).reshape(-1)
    rposition_array = np.asarray(rposition).reshape(-1)

    for period_row, agent_record in enumerate(p_agent):
        agent_id = int(agent_record[0])

        # Preserve the existing route for non-realtime agents.
        if agent_id not in realtime_set:
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

        if len(matches == 0):
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