"""Route-planning model for schedules, travel time, transfer handling, route validation, and optimal path search.
"""
from collections import deque
from functools import lru_cache
from heapq import heappush, heappop

from coop_navigation_sds.TransportNetwork.network import (
    LINES,
    ADJACENCY,
    TRAVEL_TIMES,
    delay_class_probability,
    line_segment_key,
    segment_fullness_percent,
    station_fullness_percent,
    station_transfer_time_min,
)


def fmt_time(minutes: int) -> str:
    """Fmt time function for this module's MVC responsibility.

    Args:
        minutes: Input value used by `fmt_time`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def normalize_allowed_modes(allowed_modes=None):
    """Return enabled public and walking modes from a route access profile."""
    if not allowed_modes:
        return frozenset(("metro", "tram", "bus", "walking"))
    return frozenset(
        str(mode).strip().lower()
        for mode in allowed_modes
        if mode and not str(mode).startswith("walking_max:")
    )


def maximum_walking_minutes(allowed_modes=None):
    """Extract the cumulative walking limit encoded in an access profile."""
    for value in allowed_modes or ():
        text = str(value).strip().lower()
        if text.startswith("walking_max:"):
            try:
                return max(0, int(float(text.split(":", 1)[1])))
            except ValueError:
                return 0
    return None


def line_mode(line_name):
    """Return the transport mode for a service line."""
    return LINES[line_name].get("mode", "bus")


def step_transport_type(step):
    """Return a stored mode, or reconstruct it when the active network permits."""
    mode = step.get("mode")
    if mode:
        return mode
    try:
        return line_mode(step["line"])
    except (KeyError, TypeError):
        return "unknown"


def line_allowed(line_name, allowed_modes=None):
    """Return whether a service is covered by the caller's access profile."""
    return line_mode(line_name) in normalize_allowed_modes(allowed_modes)


def transfer_time_at_station(station, previous_line, next_line, default_transfer_time_min):
    """Return transfer time only when the route changes lines."""
    if not previous_line or previous_line == next_line:
        return 0
    return max(int(default_transfer_time_min), station_transfer_time_min(station))


def transfer_miss_probability(station, next_line, transfer, wait, current_time_min):
    """Estimate the risk of missing a connection after a line change."""
    if transfer <= 0:
        return 0.0
    station_time = station_transfer_time_min(station)
    buffer = max(0, wait)
    fullness = station_fullness_percent(station, current_time_min)
    headway = LINES[next_line]["headway"]
    risk = 0.02 + max(0, 3 - buffer) * 0.08 + max(0, station_time - transfer) * 0.12
    risk += fullness / 500.0 + headway / 220.0
    return round(min(0.85, max(0.01, risk)), 4)


def next_departure(current_time_min: int, line_name: str) -> int:
    """Next departure function for this module's MVC responsibility.

    Args:
        current_time_min: Input value used by `next_departure`; see the function signature and caller context for the expected type.
        line_name: Input value used by `next_departure`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    headway = LINES[line_name]["headway"]
    if headway <= 0:
        return current_time_min
    remainder = current_time_min % headway
    wait = 0 if remainder == 0 else headway - remainder
    return current_time_min + wait


def line_direction_sequences(line_name: str):
    """Line direction sequences function for this module's MVC responsibility.

    Args:
        line_name: Input value used by `line_direction_sequences`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    stops = LINES[line_name]["stops"]
    if LINES[line_name].get("kind") == "Ring":
        reversed_stops = list(reversed(stops))
        return [
            stops + [stops[0]],
            reversed_stops + [reversed_stops[0]],
        ]

    return [
        stops,
        list(reversed(stops)),
    ]


def segment_travel(a: str, b: str) -> int:
    """Segment travel function for this module's MVC responsibility.

    Args:
        a: Input value used by `segment_travel`; see the function signature and caller context for the expected type.
        b: Input value used by `segment_travel`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    values = [
        minutes
        for (_line, station_a, station_b), minutes in TRAVEL_TIMES.items()
        if {station_a, station_b} == {a, b}
    ]
    if values:
        return min(values)
    raise KeyError(tuple(sorted((a, b))))


def segment_travel_on_line(line_name: str, a: str, b: str) -> int:
    """Return travel time for a specific line segment."""
    return TRAVEL_TIMES[line_segment_key(line_name, a, b)]


def line_direction_key(line_name: str, station: str, nxt: str) -> str | None:
    """Line direction key function for this module's MVC responsibility.

    Args:
        line_name: Input value used by `line_direction_key`; see the function signature and caller context for the expected type.
        station: Input value used by `line_direction_key`; see the function signature and caller context for the expected type.
        nxt: Input value used by `line_direction_key`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    for direction_index, sequence in enumerate(line_direction_sequences(line_name)):
        for a, b in zip(sequence, sequence[1:]):
            if a == station and b == nxt:
                return f"{line_name}:{direction_index}"
    return None


def line_departure_offset(line_name: str, station: str, nxt: str) -> int | None:
    """Line departure offset function for this module's MVC responsibility.

    Args:
        line_name: Input value used by `line_departure_offset`; see the function signature and caller context for the expected type.
        station: Input value used by `line_departure_offset`; see the function signature and caller context for the expected type.
        nxt: Input value used by `line_departure_offset`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    for sequence in line_direction_sequences(line_name):
        elapsed = 0
        for a, b in zip(sequence, sequence[1:]):
            if a == station and b == nxt:
                return elapsed
            elapsed += segment_travel_on_line(line_name, a, b)
    return None


def next_train_departure(current_time_min: int, line_name: str, station: str, nxt: str) -> int:
    """Next train departure function for this module's MVC responsibility.

    Args:
        current_time_min: Input value used by `next_train_departure`; see the function signature and caller context for the expected type.
        line_name: Input value used by `next_train_departure`; see the function signature and caller context for the expected type.
        station: Input value used by `next_train_departure`; see the function signature and caller context for the expected type.
        nxt: Input value used by `next_train_departure`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    offset = line_departure_offset(line_name, station, nxt)
    if offset is None:
        return next_departure(current_time_min, line_name)

    headway = LINES[line_name]["headway"]
    if headway <= 0:
        return current_time_min
    if current_time_min <= offset:
        return offset

    elapsed_since_offset = current_time_min - offset
    remainder = elapsed_since_offset % headway
    wait = 0 if remainder == 0 else headway - remainder
    return current_time_min + wait


def segment_delay_probability(line_name: str, station_a: str, station_b: str, depart_min: int) -> float:
    """Estimate operational delay risk for one traversed segment."""
    headway = LINES[line_name]["headway"]
    fullness = segment_fullness_percent(line_name, station_a, station_b, depart_min)
    travel = segment_travel_on_line(line_name, station_a, station_b)
    class_base = delay_class_probability(LINES[line_name].get("delay_probability_class", "moderate"))
    risk = class_base + (headway / 180.0) + (fullness / 600.0) + (travel / 500.0)
    return round(min(0.75, max(0.01, risk)), 4)


@lru_cache(maxsize=1024)
def optimal_time_route(start: str, goal: str, start_time_min: int, transfer_time_min: int, allowed_modes=None):
    """Optimal time route function for this module's MVC responsibility.

    Args:
        start: Input value used by `optimal_time_route`; see the function signature and caller context for the expected type.
        goal: Input value used by `optimal_time_route`; see the function signature and caller context for the expected type.
        start_time_min: Input value used by `optimal_time_route`; see the function signature and caller context for the expected type.
        transfer_time_min: Input value used by `optimal_time_route`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    walking_limit = maximum_walking_minutes(allowed_modes)
    allowed_modes = normalize_allowed_modes(allowed_modes)
    heap = []
    heappush(heap, (start_time_min, start, None, None, 0, []))
    best = {(start, None, 0): start_time_min}

    while heap:
        current_time, station, current_line, current_service, walking_minutes, path = heappop(heap)

        if station == goal:
            return current_time, path

        for nxt, line, travel in ADJACENCY[station]:
            if not line_allowed(line, allowed_modes):
                continue
            next_walking = walking_minutes + (travel if line_mode(line) == "walking" else 0)
            if walking_limit is not None and next_walking > walking_limit:
                continue
            service = line_direction_key(line, station, nxt)
            if service is None:
                continue
            transfer = transfer_time_at_station(station, current_line, line, transfer_time_min)
            ready = current_time + transfer
            continuing_same_train = current_line == line and current_service == service
            depart = current_time if continuing_same_train else next_train_departure(ready, line, station, nxt)
            arrive = depart + travel
            state = (nxt, service, next_walking)

            if arrive < best.get(state, 10**9):
                best[state] = arrive
                step = {
                    "from": station,
                    "to": nxt,
                    "line": line,
                    "mode": line_mode(line),
                    "service": service,
                    "previous_line": current_line,
                    "previous_mode": line_mode(current_line) if current_line else None,
                    "depart": depart,
                    "arrive": arrive,
                    "wait": 0 if continuing_same_train else depart - ready,
                    "travel": travel,
                    "transfer": transfer,
                }
                step["fullness"] = segment_fullness_percent(line, station, nxt, depart)
                step["delay_probability"] = segment_delay_probability(line, station, nxt, depart)
                step["transfer_station_time"] = station_transfer_time_min(station) if transfer else 0
                step["transfer_miss_probability"] = transfer_miss_probability(station, line, transfer, step["wait"], ready)
                step["cumulative_walking_min"] = next_walking
                heappush(heap, (arrive, nxt, line, service, next_walking, path + [step]))

    return None, []


@lru_cache(maxsize=512)
def candidate_time_routes(
    start: str,
    goal: str,
    start_time_min: int,
    transfer_time_min: int,
    limit: int = 4,
    max_extra_stops: int = 4,
    max_paths: int = 4000,
    allowed_modes=None,
):
    """Return distinct valid route candidates, searching shorter station paths first."""
    max_stops = len(ADJACENCY) + max_extra_stops
    queue = deque([(start, [start])])
    candidates = []
    checked_paths = 0

    while queue and checked_paths < max_paths:
        station, path = queue.popleft()
        checked_paths += 1
        if len(path) > max_stops:
            continue

        if station == goal and len(path) >= 2:
            estimate = estimate_route_time(path, start_time_min, transfer_time_min, allowed_modes=allowed_modes)
            if estimate:
                arrival, steps = estimate
                candidates.append((arrival - start_time_min, list(path), steps))
            continue

        next_stations = sorted({
            nxt
            for nxt, line, _travel in ADJACENCY[station]
            if line_allowed(line, allowed_modes)
        })
        for nxt in next_stations:
            if nxt in path:
                continue
            queue.append((nxt, path + [nxt]))

    candidates.sort(key=lambda item: (len(item[1]), item[0], item[1]))
    distinct = []
    seen = set()
    for duration, route, steps in candidates:
        key = tuple(route)
        if key in seen:
            continue
        seen.add(key)
        distinct.append((duration, route, steps))
        if len(distinct) >= limit:
            break
    return distinct


def route_station_sequence(steps):
    """Route station sequence function for this module's MVC responsibility.

    Args:
        steps: Input value used by `route_station_sequence`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    if not steps:
        return []

    seq = [steps[0]["from"]]
    for step in steps:
        seq.append(step["to"])

    return seq


def route_path_text_from_steps(steps):
    """Return a complete path with consecutive same-line edges condensed."""
    if not steps:
        return "No route found."

    clauses = [steps[0]["from"]]
    index = 0
    while index < len(steps):
        first = steps[index]
        mode = step_transport_type(first)
        if mode == "walking":
            minutes = max(1, int(first["arrive"] - first["depart"]))
            clauses.append(f"--walk {minutes} min--> {first['to']}")
            index += 1
            continue

        line = first["line"]
        destinations = [first["to"]]
        index += 1
        while index < len(steps):
            following = steps[index]
            if step_transport_type(following) == "walking" or following["line"] != line:
                break
            destinations.append(following["to"])
            index += 1
        intermediate = destinations[:-1]
        stop_text = f" ({', '.join(intermediate)})" if intermediate else ""
        clauses.append(f"--{mode} {line}{stop_text}--> {destinations[-1]}")
    return " ".join(clauses)


def route_text_from_steps(steps):
    """Return a concise spoken route grouped by boarding and change points."""
    if not steps:
        return "No route found."

    rides = route_rides(steps)
    total = steps[-1]["arrive"] - (steps[0]["depart"] - steps[0]["wait"])
    change_count = len(rides) - 1
    change_text = "no changes" if change_count == 0 else f"{change_count} changes"
    def service_name(ride):
        return f"{ride['mode']} line {ride['line']}"

    def ride_clause(ride, first=False):
        if ride["mode"] == "walking":
            minutes = max(1, int(ride["arrive"] - ride["depart"]))
            return f"walk {minutes} minutes from {ride['from']} to {ride['to']}"
        prefix = "Take " if first else ""
        service = service_name(ride) if first else f"{ride['mode']} {ride['line']}"
        return f"{prefix}{service} from {ride['from']} to {ride['to']}"

    first, *remaining = rides
    ride_text = ride_clause(first, first=True)
    ride_text += "".join(
        f"; {ride_clause(ride)}"
        for ride in remaining
    )
    spoken_route = ride_text[:1].upper() + ride_text[1:]
    return f"{spoken_route}. {total} minutes, {change_text}."


def route_step_details(steps):
    """Return stable analysis fields for every station-to-station route edge."""
    return [
        {
            "step_index": index,
            "from_station": step["from"],
            "to_station": step["to"],
            "line": None if step_transport_type(step) == "walking" else step["line"],
            "transport_type": step_transport_type(step),
        }
        for index, step in enumerate(steps, start=1)
    ]


def route_rides(steps):
    """Group adjacent steps that stay on the same line into spoken ride legs."""
    if not steps:
        return []

    rides = []
    current = {
        "line": steps[0]["line"],
        "mode": step_transport_type(steps[0]),
        "from": steps[0]["from"],
        "to": steps[0]["to"],
        "depart": steps[0]["depart"],
        "arrive": steps[0]["arrive"],
    }
    for step in steps[1:]:
        if step["line"] == current["line"]:
            current["to"] = step["to"]
            current["arrive"] = step["arrive"]
            continue
        rides.append(current)
        current = {
            "line": step["line"],
            "mode": step_transport_type(step),
            "from": step["from"],
            "to": step["to"],
            "depart": step["depart"],
            "arrive": step["arrive"],
        }
    rides.append(current)
    return rides


def route_boarding_route(steps):
    """Return start, transfer boarding points, and destination for compact speech."""
    rides = route_rides(steps)
    if not rides:
        return []
    stations = [rides[0]["from"]]
    stations.extend(ride["from"] for ride in rides[1:])
    stations.append(rides[-1]["to"])
    return stations


def route_duration_breakdown(steps):
    """Route duration breakdown function for this module's MVC responsibility.

    Args:
        steps: Input value used by `route_duration_breakdown`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    return {
        "wait": sum(step["wait"] for step in steps),
        "transfer": sum(step["transfer"] for step in steps),
        "travel": sum(step["travel"] for step in steps),
    }


def route_line_sequence(steps):
    """Route line sequence function for this module's MVC responsibility.

    Args:
        steps: Input value used by `route_line_sequence`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    if not steps:
        return []

    sequence = [steps[0]["line"]]
    for step in steps[1:]:
        line_name = step["line"]
        if line_name != sequence[-1]:
            sequence.append(line_name)
    return sequence


def route_line_change_count(steps):
    """Return the number of line changes in a scheduled route."""
    sequence = route_line_sequence(steps)
    return max(len(sequence) - 1, 0)


def route_duration_text(steps):
    """Route duration text function for this module's MVC responsibility.

    Args:
        steps: Input value used by `route_duration_text`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    if not steps:
        return "None"

    parts = route_duration_breakdown(steps)
    total = parts["wait"] + parts["transfer"] + parts["travel"]
    return (
        f"{total} minutes "
        f"(travel {parts['travel']} minutes + wait {parts['wait']} minutes + transfer {parts['transfer']} minutes)"
    )


def route_is_valid(stations, allowed_modes=None):
    """Route is valid function for this module's MVC responsibility.

    Args:
        stations: Input value used by `route_is_valid`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    if len(stations) < 2:
        return False

    allowed_modes = normalize_allowed_modes(allowed_modes)
    for a, b in zip(stations, stations[1:]):
        if not any(
            nxt == b and line_allowed(line, allowed_modes) and line_direction_key(line, a, b) is not None
            for nxt, line, _ in ADJACENCY[a]
        ):
            return False

    return True


def estimate_route_time(stations, start_time_min, transfer_time_min, allowed_modes=None):
    """Estimate route time function for this module's MVC responsibility.

    Args:
        stations: Input value used by `estimate_route_time`; see the function signature and caller context for the expected type.
        start_time_min: Input value used by `estimate_route_time`; see the function signature and caller context for the expected type.
        transfer_time_min: Input value used by `estimate_route_time`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    walking_limit = maximum_walking_minutes(allowed_modes)
    allowed_modes = normalize_allowed_modes(allowed_modes)
    if not route_is_valid(stations, allowed_modes=allowed_modes):
        return None

    current_time = start_time_min
    current_line = None
    current_service = None
    steps = []

    for a, b in zip(stations, stations[1:]):
        options = [(line, travel) for nxt, line, travel in ADJACENCY[a] if nxt == b and line_allowed(line, allowed_modes)]
        continuing_options = [
            (line, travel)
            for line, travel in options
            if current_line == line and line_direction_key(line, a, b) == current_service
        ]
        if continuing_options:
            options = continuing_options
        best_step = None

        for line, travel in options:
            service = line_direction_key(line, a, b)
            if service is None:
                continue
            transfer = transfer_time_at_station(a, current_line, line, transfer_time_min)
            ready = current_time + transfer
            continuing_same_train = current_line == line and current_service == service
            depart = current_time if continuing_same_train else next_train_departure(ready, line, a, b)
            arrive = depart + travel

            candidate = {
                "from": a,
                "to": b,
                "line": line,
                "mode": line_mode(line),
                "service": service,
                "previous_line": current_line,
                "previous_mode": line_mode(current_line) if current_line else None,
                "depart": depart,
                "arrive": arrive,
                "wait": 0 if continuing_same_train else depart - ready,
                "travel": travel,
                "transfer": transfer,
            }
            candidate["fullness"] = segment_fullness_percent(line, a, b, depart)
            candidate["delay_probability"] = segment_delay_probability(line, a, b, depart)
            candidate["transfer_station_time"] = station_transfer_time_min(a) if transfer else 0
            candidate["transfer_miss_probability"] = transfer_miss_probability(a, line, transfer, candidate["wait"], ready)

            if best_step is None or arrive < best_step["arrive"]:
                best_step = candidate

        if best_step is None:
            return None
        steps.append(best_step)
        current_time = best_step["arrive"]
        current_line = best_step["line"]
        current_service = best_step["service"]

    walking_total = sum(step["travel"] for step in steps if step.get("mode") == "walking")
    if walking_limit is not None and walking_total > walking_limit:
        return None

    return current_time, steps
