"""Route-planning model for schedules, travel time, transfer handling, route validation, and optimal path search.
"""
from heapq import heappush, heappop

from minillama.metro_data import LINES, ADJACENCY, TRAVEL_TIMES


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
        return [
            stops + [stops[0]],
            list(reversed(stops)) + [stops[-1]],
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
            service = line_direction_key(line, station, nxt) or line
            transfer = transfer_time_min if current_line and current_line != line else 0
            ready = current_time + transfer
            depart = next_train_departure(ready, line, station, nxt)
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
                    "fullness": LINES[line].get("fullness", 0),
                    "depart": depart,
                    "arrive": arrive,
                    "wait": depart - ready,
                    "travel": travel,
                    "transfer": transfer,
                }
                heappush(heap, (arrive, nxt, line, service, path + [step]))

    return None, []


def candidate_time_routes(
    start: str,
    goal: str,
    start_time_min: int,
    transfer_time_min: int,
    limit: int = 4,
    max_extra_stops: int = 4,
    max_paths: int = 4000,
):
    """Return distinct valid route candidates sorted by scheduled duration."""
    max_stops = len(ADJACENCY) + max_extra_stops
    stack = [(start, [start])]
    candidates = []
    checked_paths = 0

    while stack and checked_paths < max_paths:
        station, path = stack.pop()
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
            stack.append((nxt, path + [nxt]))

    candidates.sort(key=lambda item: (item[0], len(item[1]), item[1]))
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
    """Route text from steps function for this module's MVC responsibility.
    
    Args:
        steps: Input value used by `route_text_from_steps`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    if not steps:
        return "No route found."

    return " then ".join(
        f"take {s['line']} from {s['from']} to {s['to']} at {fmt_time(s['depart'])}, arrive {fmt_time(s['arrive'])}"
        for s in steps
    )


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
        f"{total} min "
        f"(travel {parts['travel']} + wait {parts['wait']} + transfer {parts['transfer']})"
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
        if not any(nxt == b for nxt, _, _ in ADJACENCY[a]):
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
    steps = []

    for a, b in zip(stations, stations[1:]):
        options = [(line, travel) for nxt, line, travel in ADJACENCY[a] if nxt == b]
        best_step = None

        for line, travel in options:
            service = line_direction_key(line, a, b) or line
            transfer = transfer_time_min if current_line and current_line != line else 0
            ready = current_time + transfer
            depart = next_train_departure(ready, line, a, b)
            arrive = depart + travel

            candidate = {
                "from": a,
                "to": b,
                "line": line,
                "service": service,
                "previous_line": current_line,
                "fullness": LINES[line].get("fullness", 0),
                "depart": depart,
                "arrive": arrive,
                "wait": depart - ready,
                "travel": travel,
                "transfer": transfer,
            }

            if best_step is None or arrive < best_step["arrive"]:
                best_step = candidate

        steps.append(best_step)
        current_time = best_step["arrive"]
        current_line = best_step["line"]

    return current_time, steps
