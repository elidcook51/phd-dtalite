# The three pipeline output files, ported from OutputVehicleTrajectoryData
# v4: writers use UTF-8, not latin-1. The pipeline reads these files as UTF-8
# (sanitize_dtalite_csv) and, on Windows, writes them back with the locale
# encoding; a latin-1 byte such as the NBSP in input_link_type.csv became
# U+FFFD on read and then crashed the cp1252 write. UTF-8 round-trips it.
# For ASCII-only Sioux Falls outputs the bytes are identical to v3.
# (StatisticsOutput.cpp:2056), OutputLinkMOEData (DTALite.cpp:3691) and
# g_OutputLinkMOESummary (StatisticsOutput.cpp:1705). printf formats are
# reproduced exactly; the LinkMOE summary uses iostream fixed precision 4.
from .cpp import f32

AGENT_HEADER = (
    "agent_id,tour_id,dependency_agent_id,duration_in_min,following_agent_id,"
    "from_zone_id, to_zone_id, from_origin_node_id, to_destination_node_id, "
    "departure_time_in_min, arrival_time_in_min, complete_flag, "
    "travel_time_in_min, demand_type, vehicle_type, PCE, information_type, "
    "value_of_time, toll_cost, distance, TotalEnergy_(KJ), CO2_(g), NOX_(g), "
    "CO_(g), HC_(g), PM_(g), PM_2.5(g),vehicle_age,number_of_nodes,"
    "path_node_sequence,org_path_node_sequence,path_time_sequence,"
    "origin_node_x,origin_node_y,destination_node_x,destination_node_y,\n")

TRIP_HEADER = (
    "agent_id,tour_id,from_zone_id,to_zone_id,from_origin_node_id,"
    "to_destination_node_id,departure_time_in_min,end_time_in_min,"
    "travel_time_in_min,demand_type,information_type,value_of_time,"
    "vehicle_type,vehicle_age,distance\n")

TDMOE_HEADER = (
    "from_node_id,to_node_id,link_id_from_to,day_no,timestamp_in_min,"
    "travel_time_in_min,delay_in_min,link_in_volume_number_of_veh,"
    "link_out_volume_number_of_veh,link_volume_in_veh_per_hour_per_lane,"
    "link_volume_in_veh_per_hour_for_all_lanes,"
    "density_in_veh_per_distance_per_lane,speed,queue_length_percentage,"
    "number_of_queued_vehicles,cumulative_arrival_count,"
    "cumulative_departure_count,total_energy,total_CO2,total_NOX,total_CO,"
    "total_HC,total_PM,total_PM2.5\n")


def _level_of_service(percentage):
    # g_GetLevelOfService (Utility.cpp:921)
    if percentage >= 90:
        return "A"
    if percentage >= 70:
        return "B"
    if percentage >= 50:
        return "C"
    if percentage >= 40:
        return "D"
    if percentage >= 33:
        return "E"
    return "F"


def write_agent_files(world, agent_path, trip_path):
    nodes = world.nodes
    links = world.links
    agent_lines = [AGENT_HEADER]
    trip_lines = [TRIP_HEADER]
    for agent_id in sorted(world.vehicles):
        veh = world.vehicles[agent_id]
        origin_number = nodes[veh.origin_node].number
        dest_number = nodes[veh.dest_node].number
        flag = "c" if veh.complete else "i"
        if veh.node_size >= 2:
            trip_time = f32(veh.arrival - veh.departure) if veh.complete \
                else f32(0)
            row = (
                f"{veh.agent_id},{veh.tour_id},{veh.dependency_agent_id},"
                f"{float(veh.duration):.0f},{veh.following_agent_id},"
                f"{veh.origin_zone},{veh.dest_zone},{origin_number},"
                f"{dest_number},{float(veh.departure):4.2f},"
                f"{float(veh.arrival):4.2f},{flag},{float(trip_time):4.2f},"
                f"{veh.demand_type},{veh.vehicle_type},{float(veh.pce):4.2f},"
                f"{veh.info_type},{float(veh.vot):4.2f},"
                f"{float(veh.toll):4.3f},{float(veh.distance):4.3f},"
                f"{0.0:f},{0.0:f},{0.0:f},{float(veh.co):f},{0.0:f},"
                f"{0.0:f},{0.0:f},{veh.age},{veh.node_size},")
            # path node sequence, derived from the current link list
            node_numbers = [nodes[links[veh.link_ary[0].link_no].from_id]
                            .number]
            for entry in veh.link_ary[:-1]:
                node_numbers.append(
                    nodes[links[entry.link_no].to_id].number)
            row += "".join("%d;" % n for n in node_numbers) + ","
            # original path (after a real-time switch); the C++ prints the
            # current path's last node first, then the stored path's nodes
            if veh.original_link_ary is not None:
                row += "%d;" % node_numbers[-1]
                for entry in veh.original_link_ary[:-1]:
                    row += "%d;" % nodes[links[entry.link_no].to_id].number
            row += ","
            row += f"{float(veh.leaving_buffer_time):4.1f};"
            for entry in veh.link_ary[:-1]:
                row += f"{float(entry.abs_arrival):4.1f};"
            row += ","
            origin_node = nodes[links[veh.link_ary[0].link_no].from_id]
            dest_node = nodes[
                links[veh.link_ary[veh.node_size - 2].link_no].to_id]
            row += f"{origin_node.x:f},{origin_node.y:f},"
            row += f"{dest_node.x:f},{dest_node.y:f},"
            row += "\n"
            agent_lines.append(row)
            trip_lines.append(
                f"{veh.agent_id},{veh.agent_id},{veh.origin_zone},"
                f"{veh.dest_zone},{origin_number},{dest_number},"
                f"{float(veh.departure):4.2f},{float(veh.arrival):4.2f},"
                f"{float(f32(veh.arrival - veh.departure)):4.2f},"
                f"{veh.demand_type},{veh.info_type},{float(veh.vot):4.2f},"
                f"{veh.vehicle_type},{veh.age},{float(veh.distance):4.2f},\n")
        else:
            agent_lines.append(
                f"{veh.agent_id},{veh.tour_id},{veh.dependency_agent_id},"
                f"{float(veh.duration):.0f},{veh.following_agent_id},"
                f"{veh.origin_zone},{veh.dest_zone},{origin_number},"
                f"{dest_number},{float(veh.departure):4.2f},"
                f"{float(veh.arrival):4.2f},{flag},{0.0:4.2f},"
                f"{veh.demand_type},{veh.vehicle_type},{veh.info_type},"
                f"{float(veh.vot):4.2f},{float(veh.toll):4.3f},"
                f"{float(veh.distance):4.3f},"
                f"{0.0:f},{0.0:f},{0.0:f},{float(veh.co):f},{0.0:f},"
                f"{veh.node_size}\n")
    with open(agent_path, "w", newline="", encoding="utf-8") as fh:
        fh.writelines(agent_lines)
    with open(trip_path, "w", newline="", encoding="utf-8") as fh:
        fh.writelines(trip_lines)


def write_link_tdmoe(world, path, day_no=1):
    # OutputLinkMOEData with the DTASettings aggregation interval (1 here)
    aggregation = world.ini.get_int(
        "output", "TDLinkMOE_aggregation_time_interval_in_min", 1)
    if aggregation < 1:
        aggregation = 1
    lines = [TDMOE_HEADER]
    for link in world.links:
        from_number = world.nodes[link.from_id].number
        to_number = world.nodes[link.to_id].number
        time = world.start
        while time <= world.horizon:
            if time < world.horizon + 1:
                moe = link.moe[time]
                in_flow = float(world.arrival_flow(link, time, aggregation))
                out_flow = float(world.departure_flow(link, time, aggregation))
                travel_time = world.travel_time_by_min(link, time, aggregation)
                queue_ratio = f32(
                    f32(moe.exit_queue_length) /
                    f32(f32(f32(link.k_jam * link.length) *
                            f32(link.inflow_lanes))))
                density = (moe.cum_arrival - moe.cum_departure) * 1.0 / \
                    float(link.length) / max(1, link.inflow_lanes)
                speed = world.link_speed(link, time)
                lines.append(
                    f"{from_number},{to_number},{from_number}->{to_number},"
                    f"{day_no},{time},{float(travel_time):6.2f},"
                    f"{float(f32(travel_time - link.fftt)):6.2f},"
                    f"{in_flow:6.2f},{out_flow:6.2f},"
                    f"{in_flow * 60.0 / max(1, link.inflow_lanes):6.2f},"
                    f"{in_flow * 60.0:6.2f},{density:6.2f},"
                    f"{float(speed):3.2f},{float(f32(queue_ratio * f32(100))):3.2f},"
                    f"{moe.exit_queue_length:d},{moe.cum_arrival:d},"
                    f"{moe.cum_departure:d}"
                    ",0.000000,0.000000,0.000000,0.000000,0.000000,"
                    "0.000000,0.000000,\n")
            time += aggregation
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.writelines(lines)


def write_link_moe(world, path):
    # g_OutputLinkMOESummary with cut_off_volume = 0
    header = (
        "from_node_id,to_node_id,type_code,link_type,link_length,"
        "start_time_in_min,end_time_in_min,total_link_volume,"
        "lane_capacity_in_vhc_per_hour,link_capacity_in_vhc_per_hour,"
        "volume_over_capacity_ratio,"
        "avg_waiting_time_on_loading_buffer_(min), speed_limit,speed,"
        "free-flow_travel_time,travel_time,percentage_of_speed_limit, "
        "level_of_service, sensor_data_flag, sensor_link_volume, "
        "simulated_link_volume, measurement_error, "
        "abs_measurement_error_percentage, simulated_AADT, "
        "TotalEnergy_(J / hr), CO2_(g / hr), NOX_(g / hr), CO_(g / hr), "
        "HC_(g / hr, PM_(g / hr), PM_2_5(g / hr),"
        "sov_volume, hov_volume, truck_volume, intermodal_volume,")
    for vtype, name in world.vehicle_types_ordered:
        header += name + ","
    header += "geometry, "
    header += "from_node_id,to_node_id,"
    for h in range(24):
        header += "vol_h%d," % (h + 1)
    for h in range(24):
        header += "speed_h%d," % (h + 1)
    for d in range(world.num_iterations + 1):
        header += "vol_d%d," % (d + 1)
    for d in range(world.num_iterations + 1):
        header += "spd_d%d," % (d + 1)
    header += "\n"
    lines = [header]
    for link in world.links:
        average_tt = world.travel_time_by_min(link, 0, world.horizon)
        speed = float(link.length) / max(0.00001, float(average_tt)) * 60
        capacity_horizon = float(link.lane_capacity) * link.outflow_lanes
        voc = link.cflow_arrival / \
            max(1, (world.end - world.start) / 60.0) / \
            max(0.1, capacity_horizon)
        percentage = int(speed / max(0.1, float(link.speed_limit)) * 100 + 0.5)
        waiting = float(f32(max(f32(0), link.buffer_waiting_time) /
                            f32(max(1, link.cflow_arrival))))
        element = world.link_type_of(link.link_type)
        row = (
            f"{world.nodes[link.from_id].number},"
            f"{world.nodes[link.to_id].number},{element.type_code},"
            f"{element.name},{float(link.length):.4f},{world.start},"
            f"{world.end},{link.cflow_arrival},"
            f"{float(link.lane_capacity):.4f},{capacity_horizon:.4f},"
            f"{voc:.4f},{waiting:.4f},{float(link.speed_limit):.4f},"
            f"{speed:.4f},{float(link.fftt):.4f},{float(average_tt):.4f},"
            f"{percentage},{_level_of_service(percentage)},0,0.0000,"
            f"{link.cflow_arrival},0.0000,0.0000,0.0000,"
            f"0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,")
        for dt in (1, 2, 3, 4):
            row += "%d," % link.counts_demand[dt]
        for vtype, name in world.vehicle_types_ordered:
            row += "%d," % link.counts_vtype[vtype]
        row += '"%s",' % link.org_geometry
        row += "%d,%d," % (world.nodes[link.from_id].number,
                            world.nodes[link.to_id].number)
        for h in range(24):
            end_time = (h + 1) * 60
            count = 0.0
            if end_time <= world.horizon + 1:
                count = float(link.moe[end_time].cum_departure -
                              link.moe[h * 60].cum_departure)
                if count < 0:
                    count = 0.0
            row += "%.4f," % count
        for h in range(24):
            # float travel_time_in_hour = TT/60.0; float speed = L/max(...)
            tt_hour = f32(float(world.travel_time_by_min(
                link, h * 60, 60)) / 60.0)
            spd = f32(float(link.length) / max(0.0001, float(tt_hour)))
            row += "%.4f," % float(spd)
        row += "\n"
        lines.append(row)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.writelines(lines)
