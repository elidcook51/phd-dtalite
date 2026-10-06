import numpy as np

def traveltimecal(timestamp, TDlink, path, length_data, gas, itr, empty):

    if not hasattr(traveltimecal, 'tdlink_map'):
        traveltimecal.tdlink_map = {}

    if not hasattr(traveltimecal, "length_map"):
        traveltimecal.length_map = {}

    sque = path.split(';')
    index = [int(x) for x in sque if x.strip() != '']

    total_time = 0
    fuelcost = 0

    if empty == 1:
        for i in range(len(index) - 1):
            from_node = index[i]
            to_node = index[i + 1]

            po = np.where(
                (length_data[:, 0] == from_node) &
                (length_data[:, 1] == to_node)
            )[0]

            if len(po) > 0:
                row = po[0]
                total_time += (
                    length_data[row, 2] /
                    length_data[row, 3] * 60
                )
                fuelcost += (
                    length_data[row, 2] / 32 * gas
                )
    else:
        if len(traveltimecal.tdlink_map) == 0 or itr == 1:
            print("  Building TDLink hash map...")

            traveltimecal.tdlink_map = {}

            for r in range(TDlink.shape[0]):
                key = (
                    f"{int(TDlink[r, 2])}_"
                    f"{int(TDlink[r, 0])}_"
                    f"{int(TDlink[r, 1])}"
                )

                traveltimecal.tdlink_map[key] = TDlink[r, 3]

            print(
                f"  Hash Map created with "
                f"{len(traveltimecal.tdlink_map)} entries"
            )

        m = 0
        tt = []
        fcost = []

        for i in range(len(index) - 1):
            from_node = index[i]

            to_node = index[i + 1]

            key = f"{int(np.floor(timestamp + m))}_{from_node}_{to_node}"

            if key in traveltimecal.tdlink_map:
                travel_time = traveltimecal.tdlink_map[key]
            else:

                po = np.where(
                    (length_data[:, 0] == from_node) & 
                    (length_data[:, 1] == to_node)
                )[0]

                if len(po) > 0:
                    row = po[0]
                    travel_time = (
                        length_data[row, 2] / 
                        length_data[row, 3] * 60
                    )

                else:
                    travel_time = 1

            tt.append(travel_time)
            m += travel_time

            po = np.where(
                (length_data[:, 0] == from_node) &
                (length_data[:, 1] == to_node)
            )[0]

            if len(po) > 0:
                row = int(po[0])

                length_val = float(length_data[row, 2])
                speed = length_val / travel_time * 60

                if speed > 40:
                    fe = 32
                elif speed < 25:
                    fe = 22
                else:
                    fe = 27

                fcost.append(length_val / fe * gas)

            else:
                fcost.append(0)

        total_time = sum(tt)
        fuelcost = sum(fcost)

    return total_time, fuelcost

import numpy as np


def traveltimecal_fast(timestamp, TDlink, path, length_data, gas, itr, empty):
    """
    Optimized travel time and fuel cost calculation.

    Parameters
    ----------
    timestamp : float
        Departure time.

    TDlink : ndarray
        Columns:
        [from_node, to_node, time_period, travel_time]

    path : str
        Example: "1;2;3;4"

    length_data : ndarray
        Columns:
        [from_node, to_node, length, freeflow_speed]

    gas : float
        Fuel price.

    itr : int
        Iteration number.

    empty : int
        1 = Free-flow conditions
        0 = Time-dependent conditions

    Returns
    -------
    total_time : float
        Total travel time (minutes)

    fuelcost : float
        Total fuel cost
    """

    # --------------------------------------------------
    # Persistent lookup tables
    # --------------------------------------------------
    if not hasattr(traveltimecal, "tdlink_map"):
        traveltimecal.tdlink_map = {}

    if not hasattr(traveltimecal, "length_map"):
        traveltimecal.length_map = {}

    # --------------------------------------------------
    # Build lookup tables on first iteration
    # --------------------------------------------------
    if itr == 1 or len(traveltimecal.tdlink_map) == 0:

        print("Building TDlink lookup table...")

        traveltimecal.tdlink_map = {}

        for r in range(TDlink.shape[0]):
            key = (
                int(TDlink[r, 2]),  # timestamp
                int(TDlink[r, 0]),  # from node
                int(TDlink[r, 1])   # to node
            )

            traveltimecal.tdlink_map[key] = float(TDlink[r, 3])

        print(
            f"✓ TDlink lookup table created with "
            f"{len(traveltimecal.tdlink_map):,} entries"
        )

        print("Building length lookup table...")

        traveltimecal.length_map = {}

        for r in range(length_data.shape[0]):

            key = (
                int(length_data[r, 0]),
                int(length_data[r, 1])
            )

            traveltimecal.length_map[key] = (
                float(length_data[r, 2]),  # length
                float(length_data[r, 3])   # freeflow speed
            )

        print(
            f"✓ Length lookup table created with "
            f"{len(traveltimecal.length_map):,} entries"
        )

    # --------------------------------------------------
    # Parse path
    # --------------------------------------------------
    nodes = [
        int(x)
        for x in path.split(";")
        if x.strip() != ""
    ]

    total_time = 0.0
    fuelcost = 0.0

    # ==================================================
    # Free-flow calculation
    # ==================================================
    if empty == 1:

        for i in range(len(nodes) - 1):

            from_node = nodes[i]
            to_node = nodes[i + 1]

            link_info = traveltimecal.length_map.get(
                (from_node, to_node)
            )

            if link_info is None:
                continue

            link_length, freeflow_speed = link_info

            travel_time = (
                link_length /
                freeflow_speed *
                60
            )

            total_time += travel_time

            fuelcost += (
                link_length / 32.0 * gas
            )

        return total_time, fuelcost

    # ==================================================
    # Time-dependent calculation
    # ==================================================
    elapsed_time = 0.0

    for i in range(len(nodes) - 1):

        from_node = nodes[i]
        to_node = nodes[i + 1]

        link_key = (from_node, to_node)

        link_info = traveltimecal.length_map.get(link_key)

        # ------------------------------------------
        # Get travel time
        # ------------------------------------------
        td_key = (
            int(np.floor(timestamp + elapsed_time)),
            from_node,
            to_node
        )

        if td_key in traveltimecal.tdlink_map:

            travel_time = traveltimecal.tdlink_map[td_key]

        else:

            # Fallback to free-flow time
            if link_info is not None:

                link_length, freeflow_speed = link_info

                travel_time = (
                    link_length /
                    freeflow_speed *
                    60
                )

            else:

                # Same as MATLAB fallback
                travel_time = 1.0

        total_time += travel_time
        elapsed_time += travel_time

        # ------------------------------------------
        # Fuel cost
        # ------------------------------------------
        if link_info is not None:

            link_length, _ = link_info

            actual_speed = (
                link_length /
                travel_time *
                60
            )

            if actual_speed > 40:
                fuel_efficiency = 32

            elif actual_speed < 25:
                fuel_efficiency = 22

            else:
                fuel_efficiency = 27

            fuelcost += (
                link_length /
                fuel_efficiency *
                gas
            )

    return total_time, fuelcost

def traveltimecal_fastv2(timestamp, TDlink, path, length_data, gas, itr, empty):

    # The TD-link table is re-read after every DTALite run, so the lookup
    # caches must be rebuilt whenever a different table object is passed
    # in. Object identity keeps repeated calls with the same table fast
    # while preventing stale travel times from leaking across iterations.
    if (
        getattr(traveltimecal_fastv2, '_tdlink_ref', None) is not TDlink
        or getattr(traveltimecal_fastv2, '_length_ref', None) is not length_data
    ):
        traveltimecal_fastv2._tdlink_ref = TDlink
        traveltimecal_fastv2._length_ref = length_data

        print('Building TDLink cache...')

        traveltimecal_fastv2.tdlink_map = {
            (
                int(row[2]),
                int(row[0]),
                int(row[1])
            ): float(row[3])
            for row in TDlink
        }

        print(f"TDLink entries = ")
        print(f"{len(traveltimecal_fastv2.tdlink_map):,}")

        print('Building length cache ...')

        traveltimecal_fastv2.length_map = {
            (
                int(row[0]),
                int(row[1])
            ): (
                float(row[2]),
                float(row[3])
            )
            for row in length_data
        }

        print('Length entries = ')
        print(f"{len(traveltimecal_fastv2.length_map):,}")

        traveltimecal_fastv2.path_result_cache = {}

    if not hasattr(traveltimecal_fastv2, 'path_cache'):
        traveltimecal_fastv2.path_cache = {}

    if path not in traveltimecal_fastv2.path_cache:

        nodes = tuple(
            int(x)
            for x in path.split(';')
            if x.strip()
        )

        links = list(
            zip(nodes[:-1], nodes[1:])
        )

        traveltimecal_fastv2.path_cache[path] = (nodes, links)

    nodes, links = (
        traveltimecal_fastv2.path_cache[path]
    )

    cache_key = (float(timestamp), path, empty)

    if cache_key in traveltimecal_fastv2.path_result_cache:
        return traveltimecal_fastv2.path_result_cache[cache_key]

    total_time = 0.0
    fuelcost = 0.0

    if empty == 1:
        for from_node, to_node in links:

            link_info = (
                traveltimecal_fastv2.length_map.get((from_node, to_node))
            )

            if link_info is None:
                continue

            length, speed = link_info

            travel_time = (length / speed * 60)

            total_time += travel_time

            fuelcost += (
                length / 32 * gas
            )

        result = (
            total_time, fuelcost
        )

        traveltimecal_fastv2.path_result_cache[cache_key] = result

        return result

    elapsed_time = 0.0

    for from_node, to_node in links:
        link_info = (
            traveltimecal_fastv2.length_map.get((from_node, to_node))
        )

        td_key = (
            int(np.floor(timestamp + elapsed_time)),
            from_node,
            to_node
        )

        travel_time = (
            traveltimecal_fastv2.tdlink_map.get(td_key)
        )

        if travel_time is None:
            if link_info is not None:
                length, speed = link_info

                travel_time = (
                    length / speed * 60
                )

            else:

                travel_time = 1

        total_time += travel_time
        elapsed_time += travel_time

        if link_info is not None:
            link_length = link_info[0]

            actual_speed = (
                link_length / travel_time * 60
            )

            if actual_speed > 40:
                mpg = 32
            elif actual_speed < 25:
                mpg = 22
            else:
                mpg = 27

            fuelcost += (
                link_length / mpg * gas
            )

    result = (total_time, fuelcost)

    traveltimecal_fastv2.path_result_cache[cache_key] = result

    return result