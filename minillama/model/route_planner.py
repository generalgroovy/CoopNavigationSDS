"""Route-planning model for schedules, travel time, transfer handling, route validation, and optimal path search.
"""
from collections import deque
from functools import lru_cache
from heapq import heappush, heappop

from minillama.model.metro_data import (
    LINES,
    ADJACENCY,
    TRAVEL_TIMES,
    segment_fullness_percent,
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


def next_departure(current_time_min: int, line_name: str) -> int:
    """Next departure function for this module's MVC responsibility.

    Args:
        current_time_min: Input value used by `next_departure`; see the function signature and caller context for the expected type.
        line_name: Input value used by `next_departure`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    headway = LINES[line_name]["headway"]
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
    return TRAVEL_TIMES[tuple(sorted((a, b)))]


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
            elapsed += segment_travel(a, b)
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
    travel = segment_travel(station_a, station_b)
    risk = 0.03 + (headway / 100.0) + (fullness / 300.0) + (travel / 250.0)
    return round(min(0.75, max(0.01, risk)), 4)


@lru_cache(maxsize=1024)
def optimal_time_route(start: str, goal: str, start_time_min: int, transfer_time_min: int):
    """Optimal time route function for this module's MVC responsibility.

    Args:
        start: Input value used by `optimal_time_route`; see the function signature and caller context for the expected type.
        goal: Input value used by `optimal_time_route`; see the function signature and caller context for the expected type.
        start_time_min: Input value used by `optimal_time_route`; see the function signature and caller context for the expected type.
        transfer_time_min: Input value used by `optimal_time_route`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    heap = []
    heappush(heap, (start_time_min, start, None, None, []))
    best = {(start, None): start_time_min}

    while heap:
        current_time, station, current_line, current_service, path = heappop(heap)

        if station == goal:
            return current_time, path

        for nxt, line, travel in ADJACENCY[station]:
            service = line_direction_key(line, station, nxt)
            if service is None:
                continue
            transfer = transfer_time_min if current_line and current_line != line else 0
            ready = current_time + transfer
            continuing_same_train = current_line == line and current_service == service
            depart = current_time if continuing_same_train else next_train_departure(ready, line, station, nxt)
            arrive = depart + travel
            state = (nxt, service)

            if arrive < best.get(state, 10**9):
                best[state] = arrive
                step = {
                    "from": station,
                    "to": nxt,
                    "line": line,
                    "service": service,
                    "previous_line": current_line,
                    "depart": depart,
                    "arrive": arrive,
                    "wait": 0 if continuing_same_train else depart - ready,
                    "travel": travel,
                    "transfer": transfer,
                }
                step["fullness"] = segment_fullness_percent(line, station, nxt, depart)
                step["delay_probability"] = segment_delay_probability(line, station, nxt, depart)
                heappush(heap, (arrive, nxt, line, service, path + [step]))

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
            estimate = estimate_route_time(path, start_time_min, transfer_time_min)
            if estimate:
                arrival, steps = estimate
                candidates.append((arrival - start_time_min, list(path), steps))
            continue

        for nxt, _, _ in ADJACENCY[station]:
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


def route_text_from_steps(steps):
    """Return a concise spoken route proposal grouped by boarding/change points."""
    if not steps:
        return "No route found."

    rides = route_rides(steps)
    boarding_route = " to ".join(route_boarding_route(steps))
    line_route = " to ".join(ride["line"] for ride in rides)
    start = rides[0]["from"]
    destination = rides[-1]["to"]
    total = steps[-1]["arrive"] - (steps[0]["depart"] - steps[0]["wait"])
    change_count = len(rides) - 1
    change_text = "no line changes" if change_count == 0 else f"{change_count} line changes"
    if len(rides) == 1:
        return f"Take {line_route} from {start} to {destination}. Boarding: {boarding_route}. Total {total} minutes, {change_text}."
    return f"Boarding: {boarding_route}. Via {line_route}. Total {total} minutes."


def route_rides(steps):
    """Group adjacent steps that stay on the same line into spoken ride legs."""
    if not steps:
        return []

    rides = []
    current = {
        "line": steps[0]["line"],
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


def route_is_valid(stations):
    """Route is valid function for this module's MVC responsibility.

    Args:
        stations: Input value used by `route_is_valid`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    if len(stations) < 2:
        return False

    for a, b in zip(stations, stations[1:]):
        if not any(nxt == b and line_direction_key(line, a, b) is not None for nxt, line, _ in ADJACENCY[a]):
            return False

    return True


def estimate_route_time(stations, start_time_min, transfer_time_min):
    """Estimate route time function for this module's MVC responsibility.

    Args:
        stations: Input value used by `estimate_route_time`; see the function signature and caller context for the expected type.
        start_time_min: Input value used by `estimate_route_time`; see the function signature and caller context for the expected type.
        transfer_time_min: Input value used by `estimate_route_time`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    if not route_is_valid(stations):
        return None

    current_time = start_time_min
    current_line = None
    current_service = None
    steps = []

    for a, b in zip(stations, stations[1:]):
        options = [(line, travel) for nxt, line, travel in ADJACENCY[a] if nxt == b]
        best_step = None

        for line, travel in options:
            service = line_direction_key(line, a, b)
            if service is None:
                continue
            transfer = transfer_time_min if current_line and current_line != line else 0
            ready = current_time + transfer
            continuing_same_train = current_line == line and current_service == service
            depart = current_time if continuing_same_train else next_train_departure(ready, line, a, b)
            arrive = depart + travel

            candidate = {
                "from": a,
                "to": b,
                "line": line,
                "service": service,
                "previous_line": current_line,
                "depart": depart,
                "arrive": arrive,
                "wait": 0 if continuing_same_train else depart - ready,
                "travel": travel,
                "transfer": transfer,
            }
            candidate["fullness"] = segment_fullness_percent(line, a, b, depart)
            candidate["delay_probability"] = segment_delay_probability(line, a, b, depart)

            if best_step is None or arrive < best_step["arrive"]:
                best_step = candidate

        if best_step is None:
            return None
        steps.append(best_step)
        current_time = best_step["arrive"]
        current_line = best_step["line"]
        current_service = best_step["service"]

    return current_time, steps
