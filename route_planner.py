from heapq import heappush, heappop

from metro_data import LINES, ADJACENCY


def fmt_time(minutes: int) -> str:
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def next_departure(current_time_min: int, line_name: str) -> int:
    headway = LINES[line_name]["headway"]
    remainder = current_time_min % headway
    wait = 0 if remainder == 0 else headway - remainder
    return current_time_min + wait


def optimal_time_route(start: str, goal: str, start_time_min: int, transfer_time_min: int):
    heap = []
    heappush(heap, (start_time_min, start, None, []))
    best = {(start, None): start_time_min}

    while heap:
        current_time, station, current_line, path = heappop(heap)

        if station == goal:
            return current_time, path

        for nxt, line, travel in ADJACENCY[station]:
            transfer = transfer_time_min if current_line and current_line != line else 0
            ready = current_time + transfer
            depart = next_departure(ready, line)
            arrive = depart + travel
            state = (nxt, line)

            if arrive < best.get(state, 10**9):
                best[state] = arrive
                step = {
                    "from": station,
                    "to": nxt,
                    "line": line,
                    "depart": depart,
                    "arrive": arrive,
                    "wait": depart - ready,
                    "travel": travel,
                    "transfer": transfer,
                }
                heappush(heap, (arrive, nxt, line, path + [step]))

    return None, []


def route_station_sequence(steps):
    if not steps:
        return []

    seq = [steps[0]["from"]]
    for step in steps:
        seq.append(step["to"])

    return seq


def route_text_from_steps(steps):
    if not steps:
        return "No route found."

    return " then ".join(
        f"take {s['line']} from {s['from']} to {s['to']} at {fmt_time(s['depart'])}, arrive {fmt_time(s['arrive'])}"
        for s in steps
    )


def route_duration_breakdown(steps):
    return {
        "wait": sum(step["wait"] for step in steps),
        "transfer": sum(step["transfer"] for step in steps),
        "travel": sum(step["travel"] for step in steps),
    }


def route_duration_text(steps):
    if not steps:
        return "None"

    parts = route_duration_breakdown(steps)
    total = parts["wait"] + parts["transfer"] + parts["travel"]
    return (
        f"{total} min "
        f"(travel {parts['travel']} + wait {parts['wait']} + transfer {parts['transfer']})"
    )


def route_is_valid(stations):
    if len(stations) < 2:
        return False

    for a, b in zip(stations, stations[1:]):
        if not any(nxt == b for nxt, _, _ in ADJACENCY[a]):
            return False

    return True


def estimate_route_time(stations, start_time_min, transfer_time_min):
    if not route_is_valid(stations):
        return None

    current_time = start_time_min
    current_line = None
    steps = []

    for a, b in zip(stations, stations[1:]):
        options = [(line, travel) for nxt, line, travel in ADJACENCY[a] if nxt == b]
        best_step = None

        for line, travel in options:
            transfer = transfer_time_min if current_line and current_line != line else 0
            ready = current_time + transfer
            depart = next_departure(ready, line)
            arrive = depart + travel

            candidate = {
                "from": a,
                "to": b,
                "line": line,
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
