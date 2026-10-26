# The three pipeline output files, ported from OutputVehicleTrajectoryData
# v4: writers use UTF-8, not latin-1. The pipeline reads these files as UTF-8
# (sanitize_dtalite_csv) and, on Windows, writes them back with the locale
# encoding; a latin-1 byte such as the NBSP in input_link_type.csv became
# U+FFFD on read and then crashed the cp1252 write. UTF-8 round-trips it.
# For ASCII-only Sioux Falls outputs the bytes are identical to v3.
# (StatisticsOutput.cpp:2056), OutputLinkMOEData (DTALite.cpp:3691) and
# g_OutputLinkMOESummary (StatisticsOutput.cpp:1705). printf formats are
# reproduced exactly; the LinkMOE summary uses iostream fixed precision 4.
from .cpp import ProgramStop, f32

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


# Full-tier link values. Worker A (pysim/emissions_full.py) keys
# full_link_moe by (from_node_number, to_node_number) and
# full_link_tdmoe by (from_node_number, to_node_number, minute), each
# value a dict of per-link totals (Energy in KJ, other gases in g) or
# a 7-sequence in that order. Canonical keys are the short names;
# the total_* spellings are accepted so a dict built with C++-style
# names still wires up. Values are totals, NOT rates: see README_TIERS.
_LINK_KEYS = ("energy", "co2", "nox", "co", "hc", "pm", "pm25")
_LINK_ALIASES = {
    "energy": ("Energy", "total_energy", "TotalEnergy", "TotalEnergy_(KJ)"),  # COORDINATOR MERGE FIX: worker A keys Energy, not TotalEnergy
    "co2": ("total_co2", "total_CO2", "CO2"),
    "nox": ("total_nox", "total_NOX", "NOX"),
    "co": ("total_co", "total_CO", "CO"),
    "hc": ("total_hc", "total_HC", "HC"),
    "pm": ("total_pm", "total_PM", "PM"),
    "pm25": ("total_pm25", "total_PM2.5", "PM2_5", "PM_2.5"),
}


def _link_value(row, name):
    if isinstance(row, dict):
        if name in row:
            return float(row[name])
        for alias in _LINK_ALIASES[name]:
            if alias in row:
                return float(row[alias])
        return 0.0
    return float(row[_LINK_KEYS.index(name)])


def _full_link_row(world, table_attr, key):
    # None in off/approx mode (caller already gated on full mode):
    # a full run that reaches a link writer without its table has a
    # broken post-pass, so fail loud instead of printing silent zeros.
    # A missing link/minute key is different: an untraversed link or
    # minute genuinely has no emissions, so it reads as 0.0.
    table = getattr(world, table_attr, None)
    if table is None:
        raise ProgramStop(
            "emission mode is full but world.%s was not set by "
            "pysim/emissions_full.py" % table_attr)
    row = table.get(key)
    if row is None:
        return (0.0,) * 7
    return tuple(_link_value(row, name) for name in _LINK_KEYS)


def _link_moe_emission_text(world, from_number, to_number):
    # The 7 emission fields of a LinkMOE row, in header order. Full
    # mode only: horizon totals from the post-pass. Off/approx keep
    # the old seven 0.0000 fields exactly (approx is agent-only).
    if getattr(world, "emission_mode", None) == "full":
        vals = _full_link_row(world, "full_link_moe",
                              (from_number, to_number))
        return ",".join("%.4f" % v for v in vals) + ","
    return "0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,"


def write_agent_files(world, agent_path, trip_path):
    nodes = world.nodes
    links = world.links
    # Agent emissions, when a post-pass ran for this load (approx or
    # full). Otherwise the old behaviour is kept exactly: tally CO,
    # zeros for the other pollutants.
    use_vsp = bool(getattr(world, "emissions_computed", False))
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
                f"{float(veh.em_energy) if use_vsp else 0.0:f},"
                f"{float(veh.em_co2) if use_vsp else 0.0:f},"
                f"{float(veh.em_nox) if use_vsp else 0.0:f},"
                f"{float(veh.em_co) if use_vsp else float(veh.co):f},"
                f"{float(veh.em_hc) if use_vsp else 0.0:f},"
                f"{float(veh.em_pm) if use_vsp else 0.0:f},"
                f"{float(veh.em_pm25) if use_vsp else 0.0:f},"
                f"{veh.age},{veh.node_size},")
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
            # Padded to the full 37-field header order (see note above):
            # PCE after vehicle_type, PM/PM2.5 and vehicle_age after HC,
            # then node_size, three empty path sequences, and the OD
            # node coordinates, with the trailing comma the header has.
            origin_node = nodes[veh.origin_node]
            dest_node = nodes[veh.dest_node]
            agent_lines.append(
                f"{veh.agent_id},{veh.tour_id},{veh.dependency_agent_id},"
                f"{float(veh.duration):.0f},{veh.following_agent_id},"
                f"{veh.origin_zone},{veh.dest_zone},{origin_number},"
                f"{dest_number},{float(veh.departure):4.2f},"
                f"{float(veh.arrival):4.2f},{flag},{0.0:4.2f},"
                f"{veh.demand_type},{veh.vehicle_type},{float(veh.pce):4.2f},"
                f"{veh.info_type},"
                f"{float(veh.vot):4.2f},{float(veh.toll):4.3f},"
                f"{float(veh.distance):4.3f},"
                f"{float(veh.em_energy) if use_vsp else 0.0:f},"
                f"{float(veh.em_co2) if use_vsp else 0.0:f},"
                f"{float(veh.em_nox) if use_vsp else 0.0:f},"
                f"{float(veh.em_co) if use_vsp else float(veh.co):f},"
                f"{float(veh.em_hc) if use_vsp else 0.0:f},"
                f"{float(veh.em_pm) if use_vsp else 0.0:f},"
                f"{float(veh.em_pm25) if use_vsp else 0.0:f},"
                f"{veh.age},{veh.node_size},,,,"
                f"{origin_node.x:f},{origin_node.y:f},"
                f"{dest_node.x:f},{dest_node.y:f},\n")
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
                # Full mode only: per-minute link totals from the
                # post-pass, in header order. Off/approx keep the old
                # zeros exactly (the approx post-pass is agent-only).
                if getattr(world, "emission_mode", None) == "full":
                    td = _full_link_row(
                        world, "full_link_tdmoe",
                        (from_number, to_number, time))
                    em_text = ",".join("%f" % v for v in td) + ","
                else:
                    em_text = ("0.000000,0.000000,0.000000,0.000000,"
                               "0.000000,0.000000,0.000000,")
                lines.append(
                    f"{from_number},{to_number},{from_number}->{to_number},"
                    f"{day_no},{time},{float(travel_time):6.2f},"
                    f"{float(f32(travel_time - link.fftt)):6.2f},"
                    f"{in_flow:6.2f},{out_flow:6.2f},"
                    f"{in_flow * 60.0 / max(1, link.inflow_lanes):6.2f},"
                    f"{in_flow * 60.0:6.2f},{density:6.2f},"
                    f"{float(speed):3.2f},{float(f32(queue_ratio * f32(100))):3.2f},"
                    f"{moe.exit_queue_length:d},{moe.cum_arrival:d},"
                    f"{moe.cum_departure:d}," + em_text + "\n")
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
        from_number = world.nodes[link.from_id].number
        to_number = world.nodes[link.to_id].number
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
            f"{_link_moe_emission_text(world, from_number, to_number)}")
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
