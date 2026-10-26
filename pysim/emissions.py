# VSP operating-mode emissions for PySim, agent file only.
#
# This is a correction to DTALite, not a byte copy of it. The DTALite
# exe's post-pass (g_CalculateEmissionMOE) was found, on Sioux Falls
# output, to (a) leave ~72% of agents at exactly zero Energy/CO2/NOX/HC
# and (b) assign the processed agents almost entirely to OpMode 0
# (braking) or OpMode 1 (idling), even for free-flow cruise trips
# (e.g. a 4-mile link at 31 mph, TT 7.76 min against FFTT 7.72 min,
# whose rates match OpMode 0 to 0.3%). Neither is a plausible driving
# outcome: every OpMode has a nonzero Energy and CO2 rate, so an exact
# zero means the agent was not processed, and a free-flow cruise trip
# is not a braking trip. PySim therefore computes emissions for every
# completed vehicle, using the published MOVES Lite framework, and
# does not reproduce the sparse-zero pattern or the OpMode 0 default.
#
# Model (Zhou et al. 2015, Transp. Res. Part D 37, 123-136; Tanvir
# 2018, NC State dissertation, MOVES Lite):
#   - For each vehicle, for each link on its path, the point-queue
#     trajectory PySim itself simulates: wait at the link entrance for
#     (link travel time - free-flow travel time), then travel the link
#     at free-flow speed for the free-flow travel time. A link whose
#     travel time is at most its free-flow time is a single cruise
#     segment at its average speed (free-flow speed, up to rounding).
#   - Within a segment, speed and acceleration are constant
#     (acceleration 0), so the operating mode is constant over the
#     whole segment. Summing rate * segment duration / 3600 over
#     segments is therefore exactly the per-second sum for that
#     trajectory, without looping per second. This is what makes the
#     calculation fast enough to run inside every pipeline load.
#   - Speed changes between segments are instantaneous in the
#     mesoscopic model, so no acceleration or braking second is
#     counted for a transition. Consequence, stated openly: OpMode 0
#     (braking) never occurs in this model, because there is no
#     deceleration phase to model. That is a limitation of mesoscopic
#     emission estimation, not a claim that real vehicles never brake.
#   - VSP (kW/tonne), light-duty passenger-car coefficients, speed in
#     m/s, acceleration in m/s^2:
#         VSP = (A*v + B*v^2 + C*v^3 + m*v*a) / m
#         A = 0.1565, B = 2.002e-3, C = 4.926e-4, m = 1.479 tonne
#     These are model constants from the MOVES Lite literature, not
#     emission rates. Emission rates themselves are never hardcoded:
#     they are read at runtime from optional_vehicle_emission_rate.csv
#     by (vehicle_type, vehicle_age, OpModeID).
#   - Operating-mode bins, EPA MOVES running-exhaust, speed in mph,
#     VSP in kW/tonne. Braking (OpMode 0) and idling (OpMode 1) take
#     precedence in MOVES, but with acceleration 0 and speed >= 1 mph
#     neither applies except idling, which is speed < 1 mph here
#     (the wait segment, speed 0):
#         speed < 1:            OpMode 1
#         1 <= speed <= 25:     VSP < 0 -> 11; < 3 -> 12; < 6 -> 13;
#                               < 9 -> 14; < 12 -> 15; else 16
#         25 < speed <= 50:     VSP < 0 -> 21; < 3 -> 22; < 6 -> 23;
#                               < 9 -> 24; < 12 -> 25; < 18 -> 27;
#                               < 24 -> 28; < 30 -> 29; else 30
#         speed > 50:           VSP < 6 -> 33; < 12 -> 35; < 18 -> 37;
#                               < 24 -> 38; < 30 -> 39; else 40
#   - A missing (vehicle_type, age, OpMode) rate row is a hard error,
#     not a silent zero: silent zeros are exactly the DTALite failure
#     mode this module exists to avoid. An empty rate cell in the file
#     is read as 0.0 (that is what the file says: no rate given).
#     In the Sioux Falls file, PM and PM2.5 cells are empty in all
#     3,565 rows, so PM/PM2.5 output is 0.0 with the current file,
#     and becomes nonzero automatically if a future rate file fills
#     those cells in. The code path for PM/PM2.5 is otherwise identical
#     to the other pollutants.
#   - Truck VSP coefficients (vehicle types 2-5) are not in the rate
#     file and were not found in the literature for DTALite's model,
#     so the passenger-car coefficients are used for all types, with
#     a one-time warning per run. The emission rates themselves are
#     still the truck rates for that type, so only the bin edges are
#     approximated. Sioux Falls and Baltimore demand is type 1 only,
#     so this approximation affects no current run.
#   - Incomplete vehicles get no VSP emissions (their trip is not
#     finished, and their link arrival times are incomplete). Their
#     emission columns stay 0, as before.
import os

from .cpp import CsvParser, ProgramStop

POLLUTANTS = (
    ("energy", "meanBaseRate_TotalEnergy_(KJ/hr)"),
    ("co2", "meanBaseRate_CO2_(g/hr)"),
    ("nox", "meanBaseRate_NOX_(g/hr)"),
    ("co", "meanBaseRate_CO_(g/hr)"),
    ("hc", "meanBaseRate_HC_(g/hr)"),
    ("pm", "meanBaseRate_PM_(g/hr)"),
    ("pm25", "meanBaseRate_PM_2.5_(g/hr)"),
)

A_COEF = 0.1565
B_COEF = 2.002e-3
C_COEF = 4.926e-4
MASS_TONNE = 1.479
MPH_TO_MPS = 0.44704


def vsp_kw_per_tonne(speed_mph, accel_mps2=0.0):
    v = speed_mph * MPH_TO_MPS
    return ((A_COEF * v + B_COEF * v * v + C_COEF * v * v * v
             + MASS_TONNE * v * accel_mps2) / MASS_TONNE)


def op_mode(speed_mph, vsp):
    if speed_mph < 1.0:
        return 1
    if speed_mph <= 25.0:
        if vsp < 0:
            return 11
        if vsp < 3:
            return 12
        if vsp < 6:
            return 13
        if vsp < 9:
            return 14
        if vsp < 12:
            return 15
        return 16
    if speed_mph <= 50.0:
        if vsp < 0:
            return 21
        if vsp < 3:
            return 22
        if vsp < 6:
            return 23
        if vsp < 9:
            return 24
        if vsp < 12:
            return 25
        if vsp < 18:
            return 27
        if vsp < 24:
            return 28
        if vsp < 30:
            return 29
        return 30
    if vsp < 6:
        return 33
    if vsp < 12:
        return 35
    if vsp < 18:
        return 37
    if vsp < 24:
        return 38
    if vsp < 30:
        return 39
    return 40


def _rate_or_zero(parser, column):
    ok, value = parser.get(column, "float")
    if not ok:
        return 0.0
    return float(value)


def read_emission_rates(input_dir):
    path = os.path.join(input_dir, "optional_vehicle_emission_rate.csv")
    if not os.path.exists(path):
        raise ProgramStop(
            "output_emission_data=1 but %s not found" % path)
    parser = CsvParser(path)
    rates = {}
    while parser.read_record():
        ok_type, vtype = parser.get("vehicle_type", "int")
        ok_age, age = parser.get("age", "int")
        ok_op, op = parser.get("OpModeID", "int")
        if not (ok_type and ok_age and ok_op):
            continue
        rates[(vtype, age, op)] = tuple(
            _rate_or_zero(parser, column) for _name, column in POLLUTANTS)
    if not rates:
        raise ProgramStop(
            "optional_vehicle_emission_rate.csv has no usable rows")
    return rates


def calculate_emissions(world):
    rates = read_emission_rates(world.dir)
    warned_truck = False
    computed = 0
    for agent_id in sorted(world.vehicles):
        veh = world.vehicles[agent_id]
        totals = [0.0] * len(POLLUTANTS)
        if veh.complete and veh.link_ary is not None:
            if veh.vehicle_type != 1 and not warned_truck:
                print("warning: vehicle_type %d uses passenger-car VSP "
                      "coefficients (truck coefficients not in the rate "
                      "file); rates are still the type %d rates"
                      % (veh.vehicle_type, veh.vehicle_type))
                warned_truck = True
            # Link travel times, in path order: the first link is
            # measured from departure, each later link from the
            # previous link's arrival stamp. link_ary has one entry
            # per node, with a sentinel (MAX_LINK_NO) in the last
            # slot, so the links are link_ary[:-1].
            prev_time = float(veh.departure)
            for entry in veh.link_ary[:-1]:
                link = world.links[entry.link_no]
                link_tt = float(entry.abs_arrival) - prev_time
                prev_time = float(entry.abs_arrival)
                if link_tt <= 0:
                    continue
                fftt = float(link.fftt)
                segments = []
                if link_tt <= fftt:
                    speed = float(link.length) / (link_tt / 60.0) \
                        if link_tt > 0 else 0.0
                    segments.append((speed, link_tt))
                else:
                    segments.append((0.0, link_tt - fftt))
                    segments.append((float(link.speed_limit), fftt))
                for speed, duration_min in segments:
                    if duration_min <= 0:
                        continue
                    mode = op_mode(speed, vsp_kw_per_tonne(speed))
                    key = (veh.vehicle_type, veh.age, mode)
                    if key not in rates:
                        raise ProgramStop(
                            "no emission rate for vehicle_type=%d "
                            "age=%d OpMode=%d (agent %d)"
                            % (veh.vehicle_type, veh.age, mode, agent_id))
                    row = rates[key]
                    hours = duration_min / 60.0
                    for i in range(len(POLLUTANTS)):
                        totals[i] += row[i] * hours
            computed += 1
        veh.em_energy = totals[0]
        veh.em_co2 = totals[1]
        veh.em_nox = totals[2]
        veh.em_co = totals[3]
        veh.em_hc = totals[4]
        veh.em_pm = totals[5]
        veh.em_pm25 = totals[6]
    return computed
