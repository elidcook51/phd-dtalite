# Full-port VSP emissions for PySim: DTALite's g_CalculateEmissionMOE /
# DTALink::ComputeVSP (EmissionsOutput.cpp in the 2017 DTALite source),
# reimplemented with numpy only. No numba is assumed anywhere; the whole
# computation below runs on plain numpy, so it works wherever PySim runs.
#
# Design notes (see NOTES.md for the full list with C++ line references):
# - Records are built once per vehicle-link, then sorted by (link, lane,
#   start time, agent id) and processed in that order. That is exactly the
#   C++ order: links outer, lanes in order, vehicles sorted per lane.
#   PrevSpeed is per-vehicle state updated in that order, so processing
#   links in ascending link-number order reproduces C++'s link-major
#   PrevSpeed semantics without holding every lane list in memory.
# - The Newell car-following recurrence is evaluated in closed form with
#   a cumulative minimum (see _positions_follower). The closed form is
#   exact when the leader trajectory is nondecreasing with slope at most
#   the free-flow distance per interval, which the recurrence itself
#   guarantees; the self-test checks it against a literal sequential
#   implementation on real lanes.
# - Positions are float64 arrays (C++ uses float); speeds, accelerations
#   and PrevSpeed are rounded back to float32 at the per-second step,
#   matching C++'s float SpeedBySecond / m_PrevSpeed storage.
# - Smoothing is the standard centered moving average computed from the
#   original positions. C++'s MovingAverage is an in-place sequential
#   filter that reuses already-smoothed neighbours; the difference is
#   small (positions are nearly linear between kinks) and is quantified
#   in NOTES.md, but it is a deliberate divergence, not an oversight.
# - Vehicle totals are accumulated per second with the fallback-resolved
#   rate, which equals C++'s rate * OpMode-count / 3600 whenever the
#   rate exists at the vehicle's own age (the normal case).
import os

import numpy as np

from .cpp import CsvParser, ProgramStop, f32

INTERVALS_PER_SECOND = 10
SECONDS_PER_HOUR = 3600.0

# Pollutant order, shared with approx v1: (attribute suffix, rate column).
POLLUTANTS = (
    ("energy", "meanBaseRate_TotalEnergy_(KJ/hr)"),
    ("co2", "meanBaseRate_CO2_(g/hr)"),
    ("nox", "meanBaseRate_NOX_(g/hr)"),
    ("co", "meanBaseRate_CO_(g/hr)"),
    ("hc", "meanBaseRate_HC_(g/hr)"),
    ("pm", "meanBaseRate_PM_(g/hr)"),
    ("pm25", "meanBaseRate_PM_2.5_(g/hr)"),
)
POLLUTANT_KEYS = ("Energy", "CO2", "NOX", "CO", "HC", "PM", "PM2_5")

# DTAVehicleType constructor defaults (DTALite.h): used for any vehicle
# type whose row is missing from optional_vehicle_type.csv, and for any
# of the four coefficients missing or empty in its row.
DEFAULT_COEF = (0.156461, 0.00200193, 0.000492646, 1.4788)

# OperatingModeMap (SetupOperatingModeVector): rows are speed bins
# 0-25 / 25-50 / >50 mph, columns are VSP bins <0 / <3 / <6 / <9 / <12 /
# <18 / <24 / <30 / >=30 kW/tonne.
OP_MAP = np.array([
    [11, 12, 13, 14, 15, 16, 16, 16, 16],
    [21, 22, 23, 24, 25, 27, 28, 29, 30],
    [33, 33, 33, 35, 35, 37, 38, 39, 40],
])
VSP_THRESHOLDS = np.array([0, 3, 6, 9, 12, 18, 24, 30], dtype=np.float64)

_MPH_PER_MPS = f32(2.23693629)
_MPS_PER_MPH = f32(0.44704)
_METERS_PER_MILE = f32(1609.344)
_NO_PREV_ACCEL = f32(-99999.0)


def _to_intervals(time_in_min):
    # C++ StartTime/EndTime: DepartureTime * 60 * 10 (or AbsArrival * 60
    # * 10), each step rounded to float, then truncated to int.
    value = f32(time_in_min)
    value = f32(value * f32(60.0))
    value = f32(value * f32(10.0))
    return int(value)


def _rate_or_zero(parser, column):
    ok, value = parser.get(column, "float")
    if not ok:
        return 0.0
    return float(value)


def _read_emission_rates(path):
    # ReadInputEmissionRateFile: a missing file stops the run. A row
    # without vehicle_type or OpModeID ends the read (C++ breaks out of
    # the loop); a missing rate column stops the run (C++ Required),
    # while an empty rate cell reads as 0.0 (the Required call's return
    # is ignored, so the field keeps its initialised 0). An empty age
    # cell likewise keeps age 0. Ages >= 31 or OpModeIDs >= 41 stop the
    # run, as C++ would on storing them out of bounds.
    if not os.path.exists(path):
        raise ProgramStop(
            "emission rate file %s not found" % path)
    parser = CsvParser(path)
    for _name, column in POLLUTANTS:
        if column not in parser.field_index:
            raise ProgramStop(
                "field %s missing in %s" % (column, path))
    if "age" not in parser.field_index:
        raise ProgramStop("field age missing in %s" % path)
    rates = {}
    while parser.read_record():
        ok_type, vtype = parser.get("vehicle_type", "int")
        if not ok_type:
            break
        ok_op, opmode = parser.get("OpModeID", "int")
        if not ok_op:
            break
        ok_age, age = parser.get("age", "int")
        if not ok_age:
            age = 0
        if age < 0 or age >= 31 or opmode < 0 or opmode >= 41:
            raise ProgramStop(
                "emission rate out of range in %s: vehicle_type=%d "
                "age=%d OpMode=%d" % (path, vtype, age, opmode))
        rates[(vtype, age, opmode)] = tuple(
            _rate_or_zero(parser, column) for _name, column in POLLUTANTS)
    if not rates:
        raise ProgramStop(
            "optional_vehicle_emission_rate.csv has no usable rows")
    return rates


def _read_vehicle_coefficients(path):
    # DTALite.cpp vehicle-type read: a missing file leaves every type on
    # the constructor defaults; a row without vehicle_type ends the
    # read; a missing or empty coefficient leaves that coefficient on
    # its default. Types are keyed by vehicle_type value (C++ indexes
    # its vector by vehicle_type - 1, which agrees with this whenever
    # the file lists types 1..N in order; see NOTES.md).
    coefs = {}
    if not os.path.exists(path):
        return coefs
    parser = CsvParser(path)
    columns = ("rolling_term_a", "rotating_term_b", "drag_term_c",
               "source_mass")
    while parser.read_record():
        ok, vtype = parser.get("vehicle_type", "int")
        if not ok:
            break
        values = list(DEFAULT_COEF)
        for i, column in enumerate(columns):
            ok_value, value = parser.get(column, "float")
            if ok_value:
                values[i] = float(value)
        if values[3] == 0.0:
            raise ProgramStop(
                "source_mass is 0 for vehicle_type=%d in %s"
                % (vtype, path))
        coefs[vtype] = tuple(values)
    return coefs


def _resolve_rate(rates, cache, vtype, age, opmode, context):
    # ComputeVSP age handling: exact age first, then age rounded to the
    # nearest 5 (5 * int(age / 5 + 0.5)); if neither row is initialised,
    # fail loud. C++ instead returns from ComputeVSP for that link,
    # abandoning the rest of the link; raising aborts the run, which is
    # louder and cannot be mistaken for a completed result.
    key = (vtype, age, opmode)
    row = cache.get(key)
    if row is not None:
        return row
    row = rates.get(key)
    if row is None:
        rounded = 5 * int(age / 5.0 + 0.5)
        row = rates.get((vtype, rounded, opmode))
    if row is None:
        raise ProgramStop(
            "no emission rate for vehicle_type=%d age=%d (nearest 5: %d) "
            "OpMode=%d (%s)"
            % (vtype, age, 5 * int(age / 5.0 + 0.5), opmode, context))
    cache[key] = row
    return row


def _link_constants(link):
    # Per-link car-following constants, each rounded to float32 exactly
    # as the C++ float expressions round: free-flow distance per
    # interval, Newell time lag (rounded to the nearest interval), and
    # critical spacing. Non-positive inputs would divide by zero or
    # modulo by zero in C++; fail loud instead.
    speed_limit = float(f32(link.speed_limit))
    length = float(f32(link.length))
    k_jam = float(f32(link.k_jam))
    wave_speed = float(f32(link.wave_speed))
    if speed_limit <= 0.0 or length <= 0.0:
        raise ProgramStop(
            "link %d->%d has non-positive speed limit or length"
            % (link.from_id, link.to_id))
    if k_jam <= 0.0 or wave_speed <= 0.0:
        raise ProgramStop(
            "link %d->%d has non-positive jam density or wave speed"
            % (link.from_id, link.to_id))
    length_m = float(f32(length) * _METERS_PER_MILE)
    ff_dist = f32(f32(f32(speed_limit) * _METERS_PER_MILE) / f32(3600.0))
    ff_dist = f32(ff_dist / f32(10.0))
    denom = f32(f32(wave_speed) * f32(k_jam))
    tau = int(float(f32(36000.0) / denom) + 0.5)
    spacing = float(_METERS_PER_MILE / f32(k_jam))
    limit_mps = float(f32(speed_limit) * _MPS_PER_MPH)
    return length_m, float(ff_dist), tau, spacing, limit_mps


def _positions_first_in_lane(n, ff_dist, length_m):
    # Lane leader (v == 0): position grows by the free-flow distance
    # each interval, capped at the link length. This is exactly the
    # C++ branch max(0, min(L, prev + ff)) unrolled, since ff > 0.
    return np.minimum(length_m,
                      np.arange(n + 1, dtype=np.float64) * ff_dist)


def _positions_follower(n, ff_dist, length_m, tau, spacing, leader_pos,
                        leader_in, leader_out, start):
    # Newell car-following in closed form. Write positions as
    # pos[k] = k * ff + h[k]. The recurrence pos[k] = max(pos[k-1],
    # min(cong[k], pos[k-1] + ff, L)) becomes h[k] = max(h[k-1] - ff,
    # min(h[k-1], m[k])) with m[k] = cong[k] - k * ff in the congested
    # regime, -k * ff while the wave has not arrived (position forced
    # to 0), and L - k * ff once the leader has left. The floor at
    # h[k-1] - ff never binds: m drops by at most ff per step, because
    # the leader trajectory is nondecreasing with slope at most ff per
    # interval, cong is max(0, leader - spacing), and the regime
    # boundaries are continuous (at the first congested step cong is 0,
    # since the leader's first position is 0). So h is the cumulative
    # minimum of m, and the whole trajectory is a few numpy passes.
    # The end-time forcing (followers only) is applied after, as C++
    # does, before smoothing.
    k_a = leader_in + tau - start
    k_b = leader_out + tau - start
    m = np.empty(n + 1, dtype=np.float64)
    m[0] = 0.0
    lo_a, hi_a = 1, min(k_a - 1, n)
    if lo_a <= hi_a:
        m[lo_a:hi_a + 1] = -np.arange(lo_a, hi_a + 1, dtype=np.float64) \
            * ff_dist
    lo_b, hi_b = max(k_a, 1), min(k_b, n)
    if lo_b <= hi_b:
        kk = np.arange(lo_b, hi_b + 1, dtype=np.float64)
        j = start + np.arange(lo_b, hi_b + 1, dtype=np.int64) \
            - tau - leader_in
        cong = np.maximum(0.0, leader_pos[j] - spacing)
        m[lo_b:hi_b + 1] = cong - kk * ff_dist
    lo_c, hi_c = max(k_b + 1, 1), n
    if lo_c <= hi_c:
        kk = np.arange(lo_c, hi_c + 1, dtype=np.float64)
        m[lo_c:hi_c + 1] = length_m - kk * ff_dist
    h = np.minimum.accumulate(m)
    pos = np.arange(n + 1, dtype=np.float64) * ff_dist + h
    pos[n] = length_m
    return pos


def _moving_average(pos):
    # Centered moving average from the original positions: endpoints
    # unchanged, index 1 averaged over 3, index 2 and len-3 over 5, the
    # interior over 7. Lengths under 7 are left untouched, as in C++.
    # This differs from C++'s in-place filter, which averages over
    # already-smoothed neighbours; see NOTES.md for the measured size
    # of that difference.
    length = len(pos)
    if length < 7:
        return pos.copy()
    out = np.empty_like(pos)
    out[0] = pos[0]
    out[1] = (pos[0] + pos[1] + pos[2]) / 3.0
    out[2] = (pos[0] + pos[1] + pos[2] + pos[3] + pos[4]) / 5.0
    csum = np.concatenate(([0.0], np.cumsum(pos)))
    idx = np.arange(3, length - 3)
    out[3:length - 3] = (csum[idx + 4] - csum[idx - 3]) / 7.0
    out[length - 3] = (csum[length] - csum[length - 5]) / 5.0
    out[length - 2] = (csum[length] - csum[length - 3]) / 3.0
    out[length - 1] = pos[length - 1]
    return out


def _second_speeds(smoothed, n, limit_mps):
    # Per-second speeds from smoothed positions: speed at second k is
    # the position change over the last 10 intervals, capped at the
    # speed limit, 0 at the start second, and the speed limit at any
    # second whose next second would reach or pass the end time (so the
    # last sampled second is always the speed limit, and a trip of at
    # most one second is a single second at the speed limit).
    count = n // INTERVALS_PER_SECOND + 1
    idx = INTERVALS_PER_SECOND * np.arange(count)
    speeds = np.full(count, limit_mps, dtype=np.float64)
    speeds[0] = 0.0
    if count > 1:
        speeds[1:] = np.minimum(
            limit_mps,
            np.maximum(0.0, smoothed[idx[1:]] - smoothed[idx[1:] - 10]))
    speeds[idx + INTERVALS_PER_SECOND >= n] = limit_mps
    return speeds.astype(f32)


def _op_modes(speeds, accel, coef):
    # ComputeOperatingModeFromSpeed plus the ComputeVSP overrides.
    # speeds and accel are float32 (m/s, m/s per second); VSP is
    # computed in float64 from those values and rounded to float32 for
    # binning, matching C++'s float vsp. Speed bins use
    # floor(speed_mph + 0.005) with cutoffs at 25 and 50 mph.
    a_coef, b_coef, c_coef, mass = coef
    v = speeds.astype(np.float64)
    a = accel.astype(np.float64)
    vsp = (a_coef * v + b_coef * v * v + c_coef * v * v * v
           + v * a * mass) / mass
    vsp32 = vsp.astype(f32)
    speed_mph = speeds * _MPH_PER_MPS
    speed_int = np.floor(speed_mph.astype(np.float64) + 0.005)
    speed_bin = np.where(speed_int <= 25, 0,
                         np.where(speed_int < 50, 1, 2))
    vsp_bin = np.searchsorted(VSP_THRESHOLDS, vsp32.astype(np.float64),
                              side="right")
    mode = OP_MAP[speed_bin, vsp_bin]
    # Braking override: acceleration at or below -2 m/s^2.
    mode = np.where(accel <= f32(-2.0), 0, mode)
    # Three-consecutive-decelerations override: the previous two
    # accelerations are stored scaled by 0.44704 in C++, so the -1
    # comparison applies to scaled values (an effective -2.237 m/s^2
    # for the earlier seconds), and the -99999 sentinels mean the check
    # cannot fire on the first second but can on the second.
    count = len(accel)
    prev = np.full(count, _NO_PREV_ACCEL, dtype=f32)
    prev_prev = np.full(count, _NO_PREV_ACCEL, dtype=f32)
    if count > 1:
        prev[1:] = accel[:-1] * _MPS_PER_MPH
    if count > 2:
        prev_prev[2:] = accel[:-2] * _MPS_PER_MPH
    three = ((prev_prev < f32(-1.0)) & (prev < f32(-1.0))
             & (accel < f32(-1.0))
             & ((prev_prev != _NO_PREV_ACCEL)
                | (prev != _NO_PREV_ACCEL)))
    mode = np.where(three, 0, mode)
    # Idling override, applied last so it wins over braking: speed
    # strictly below 1 mph (the >= -1 side is vacuous, speeds are >= 0).
    speed_mph_idle = speeds / _MPS_PER_MPH
    mode = np.where((speed_mph_idle >= f32(-1.0))
                    & (speed_mph_idle < f32(1.0)), 1, mode)
    return mode


def calculate_emissions_full(world, rate_path=None, vehicle_type_path=None,
                             smooth=None):
    """Full DTALite car-following VSP emissions for one loaded world.

    Call after network loading, the way approx v1's calculate_emissions
    is called from run.py. Sets veh.em_energy/em_co2/em_nox/em_co/em_hc/
    em_pm/em_pm25 on every vehicle (0 for incomplete vehicles), sets
    world.emissions_computed = True, and sets world.full_link_moe and
    world.full_link_tdmoe. Link keys are (from_number, to_number) using node numbers
    (world.nodes[from_id].number); TDMOE
    keys add the minute. Values are dicts keyed Energy/CO2/NOX/CO/HC/
    PM/PM2_5 holding run totals in KJ (Energy) and grams (the rest):
    full_link_moe sums a link over all minutes, full_link_tdmoe splits
    it by the minute containing each vehicle-link's start (every second
    of a link traversal is bucketed into its start minute, as C++
    does). DTALite's LinkMOE columns are labelled per hour, but the
    values it prints are these same run totals; do not rescale them.

    Optional arguments override the defaults resolved from world:
    rate_path defaults to <world.dir>/optional_vehicle_emission_rate.csv
    and vehicle_type_path to <world.dir>/optional_vehicle_type.csv;
    smooth defaults to the [emission] smooth_vehicle_trajectory setting
    (default on). Returns None. Raises ProgramStop on a missing rate
    file, an unresolvable rate, or a flow model outside 1/2/3.
    """
    flow_model = getattr(world, "flow_model", None)
    if flow_model not in (1, 2, 3):
        raise ProgramStop(
            "full emissions refuse traffic_flow_model=%r (BPR and "
            "car-following flow models are out of scope)" % (flow_model,))
    data_dir = getattr(world, "dir", ".") or "."
    if rate_path is None:
        rate_path = os.path.join(data_dir,
                                 "optional_vehicle_emission_rate.csv")
    if vehicle_type_path is None:
        vehicle_type_path = os.path.join(data_dir,
                                         "optional_vehicle_type.csv")
    if smooth is None:
        smooth = True
        ini = getattr(world, "ini", None)
        get_float = getattr(ini, "get_float", None) if ini is not None \
            else None
        if callable(get_float):
            smooth = float(get_float("emission",
                                     "smooth_vehicle_trajectory", 1)) != 0.0
    rates = _read_emission_rates(rate_path)
    coefs = _read_vehicle_coefficients(vehicle_type_path)
    rate_cache = {}

    agent_ids = sorted(world.vehicles)
    veh_list = [world.vehicles[agent_id] for agent_id in agent_ids]
    n_veh = len(veh_list)
    links = world.links
    # COORDINATOR MERGE FIX: keys must be node NUMBERS, not internal
    # indices, because outputs.py looks links up by node number
    # (world.nodes[link.from_id].number). Worker A's original used
    # internal indices, which would have made every link lookup miss.
    link_keys = [(world.nodes[link.from_id].number,
                  world.nodes[link.to_id].number) for link in links]

    total_records = 0
    for veh in veh_list:
        if veh.complete and veh.link_ary is not None \
                and len(veh.link_ary) >= 2:
            total_records += len(veh.link_ary) - 1
    linknos = np.empty(total_records, dtype=np.int32)
    lanes = np.empty(total_records, dtype=np.int32)
    starts = np.empty(total_records, dtype=np.int32)
    ends = np.empty(total_records, dtype=np.int32)
    veh_idx = np.empty(total_records, dtype=np.int32)
    pos = 0
    for vi, veh in enumerate(veh_list):
        if not (veh.complete and veh.link_ary is not None
                and len(veh.link_ary) >= 2):
            continue
        prev_time = veh.departure
        for entry in veh.link_ary[:-1]:
            link_no = entry.link_no
            if link_no < 0 or link_no >= len(links):
                raise ProgramStop(
                    "agent %d path uses out-of-range link %d"
                    % (veh.agent_id, link_no))
            outflow = links[link_no].outflow_lanes
            if outflow <= 0:
                raise ProgramStop(
                    "link %d has no outflow lanes" % link_no)
            linknos[pos] = link_no
            lanes[pos] = veh.agent_id % outflow
            starts[pos] = _to_intervals(prev_time)
            ends[pos] = _to_intervals(entry.abs_arrival)
            veh_idx[pos] = vi
            pos += 1
            prev_time = entry.abs_arrival

    if total_records:
        # Link-major order: link, then lane, then start time, then
        # agent id (veh_idx order is agent-id order, since veh_list was
        # built from sorted agent ids). This is C++'s per-lane sort
        # with a deterministic tie-break where std::sort is unstable.
        order = np.lexsort((veh_idx, starts, lanes, linknos))
        linknos, lanes, starts, ends, veh_idx = (
            linknos[order], lanes[order], starts[order], ends[order],
            veh_idx[order])
        del order

    totals = np.zeros((n_veh, 7), dtype=np.float64)
    prev_speed = np.zeros(n_veh, dtype=f32)
    moe = {}
    tdmoe = {}
    cur_linkno = -1
    cur_lane = -1
    leader_pos = None
    leader_in = -1
    leader_out = -1
    length_m = ff_dist = tau = spacing = limit_mps = None
    for i in range(total_records):
        linkno = int(linknos[i])
        lane = int(lanes[i])
        start = int(starts[i])
        end = int(ends[i])
        vi = int(veh_idx[i])
        if linkno != cur_linkno or lane != cur_lane:
            if linkno != cur_linkno:
                length_m, ff_dist, tau, spacing, limit_mps = \
                    _link_constants(links[linkno])
                cur_linkno = linkno
            cur_lane = lane
            leader_pos = None
            leader_in = -1
            leader_out = -1
        n = end - start
        if n < 0:
            # No seconds at all in C++ either (both loops are empty).
            # Unlike C++, this vehicle does not become the lane leader:
            # C++ would swap in its untouched buffer, which holds stale
            # positions from an earlier vehicle, i.e. undefined
            # behaviour. Complete vehicles with monotone arrivals never
            # hit this branch.
            continue
        veh = veh_list[vi]
        if leader_pos is None:
            raw = _positions_first_in_lane(n, ff_dist, length_m)
        else:
            raw = _positions_follower(n, ff_dist, length_m, tau, spacing,
                                      leader_pos, leader_in, leader_out,
                                      start)
        smoothed = _moving_average(raw) if smooth else raw
        speeds = _second_speeds(smoothed, n, limit_mps)
        count = len(speeds)
        accel = np.empty(count, dtype=f32)
        accel[0] = speeds[0] - prev_speed[vi]
        if count > 1:
            accel[1:] = speeds[1:] - speeds[:-1]
        prev_speed[vi] = speeds[-1]
        coef = coefs.get(veh.vehicle_type, DEFAULT_COEF)
        modes = _op_modes(speeds, accel, coef)
        uniq, cnts = np.unique(modes, return_counts=True)
        link_totals = [0.0] * 7
        context = "agent %d link %d->%d" % (
            veh.agent_id, links[linkno].from_id, links[linkno].to_id)
        for mode_val, cnt in zip(uniq, cnts):
            row = _resolve_rate(rates, rate_cache, veh.vehicle_type,
                                veh.age, int(mode_val), context)
            hours = float(cnt) / SECONDS_PER_HOUR
            for j in range(7):
                link_totals[j] += row[j] * hours
        totals[vi] += link_totals
        key = link_keys[linkno]
        bucket = moe.get(key)
        if bucket is None:
            bucket = moe[key] = [0.0] * 7
        for j in range(7):
            bucket[j] += link_totals[j]
        minute = start // 600
        tdkey = (key[0], key[1], minute)
        tdbucket = tdmoe.get(tdkey)
        if tdbucket is None:
            tdbucket = tdmoe[tdkey] = [0.0] * 7
        for j in range(7):
            tdbucket[j] += link_totals[j]
        leader_pos = smoothed
        leader_in = start
        leader_out = end

    # Vehicle fields. PySim's head Vehicle has no em_* slots; approx v1
    # added them, and the integration must do the same, otherwise this
    # fails loud here instead of silently dropping the results.
    try:
        for vi, veh in enumerate(veh_list):
            row = totals[vi]
            veh.em_energy = float(row[0])
            veh.em_co2 = float(row[1])
            veh.em_nox = float(row[2])
            veh.em_co = float(row[3])
            veh.em_hc = float(row[4])
            veh.em_pm = float(row[5])
            veh.em_pm25 = float(row[6])
    except AttributeError:
        raise ProgramStop(
            "Vehicle class has no em_energy/em_co2/em_nox/em_co/em_hc/"
            "em_pm/em_pm25 slots; add them to Vehicle.__slots__ and "
            "__init__ (as approx v1 did) before calling "
            "calculate_emissions_full")
    world.full_link_moe = {
        key: dict(zip(POLLUTANT_KEYS, vals))
        for key, vals in moe.items()}
    world.full_link_tdmoe = {
        key: dict(zip(POLLUTANT_KEYS, vals))
        for key, vals in tdmoe.items()}
    world.emissions_computed = True
    return None
