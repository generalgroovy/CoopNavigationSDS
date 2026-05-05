import math
import random
from collections import deque

from config import (
    NUM_STATIONS,
    NUM_LINES,
    LAYOUT_COLUMNS,
    DEFAULT_HEADWAY_MIN,
    DEFAULT_TRAVEL_TIME_MIN,
    MIN_TRAVEL_TIME_MIN,
    MAX_TRAVEL_TIME_MIN,
    NETWORK_SEED,
    FORCE_CONNECTED_NETWORK,
)
from station_names import get_station_names


LINE_COLORS = [
    "#d13f31",
    "#3a8f3a",
    "#4169e1",
    "#f2c230",
    "#9b59b6",
    "#111111",
    "#e67e22",
    "#16a085",
    "#2c3e50",
    "#c0392b",
]


def grid_dimensions(num_stations: int):
    if LAYOUT_COLUMNS is not None:
        cols = max(2, int(LAYOUT_COLUMNS))
    else:
        cols = max(2, math.ceil(math.sqrt(num_stations)))

    rows = math.ceil(num_stations / cols)
    return rows, cols


def station_grid(stations):
    rows, cols = grid_dimensions(len(stations))
    grid = []

    idx = 0
    for r in range(rows):
        row = []
        for c in range(cols):
            if idx < len(stations):
                row.append(stations[idx])
            else:
                row.append(None)
            idx += 1
        grid.append(row)

    return grid, rows, cols


def generate_station_positions(stations):
    grid, rows, cols = station_grid(stations)

    positions = {}

    x_gap = 120
    y_gap = 90
    x0 = 80
    y0 = 70

    for r in range(rows):
        for c in range(cols):
            station = grid[r][c]
            if station is None:
                continue

            # Small stagger for a less rigid schematic while preserving clear directions.
            x = x0 + c * x_gap + (25 if r % 2 else 0)
            y = y0 + r * y_gap

            positions[station] = (x, y)

    return positions


def valid_sequence(seq):
    return [station for station in seq if station is not None]


def perimeter_ring(grid, rows, cols):
    ring = []

    # top row, left to right
    ring.extend(valid_sequence(grid[0]))

    # right side, top to bottom excluding corners
    for r in range(1, rows - 1):
        if grid[r][cols - 1] is not None:
            ring.append(grid[r][cols - 1])

    # bottom row, right to left
    if rows > 1:
        ring.extend(reversed(valid_sequence(grid[rows - 1])))

    # left side, bottom to top excluding corners
    for r in range(rows - 2, 0, -1):
        if grid[r][0] is not None:
            ring.append(grid[r][0])

    # Remove duplicates while preserving order.
    out = []
    for station in ring:
        if station not in out:
            out.append(station)

    return out


def row_sequence(grid, r):
    return valid_sequence(grid[r])


def column_sequence(grid, rows, c):
    return valid_sequence(grid[r][c] for r in range(rows))


def diagonal_down_right_sequences(grid, rows, cols):
    groups = {}

    for r in range(rows):
        for c in range(cols):
            station = grid[r][c]
            if station is None:
                continue
            groups.setdefault(r - c, []).append((r, c, station))

    sequences = []
    for cells in groups.values():
        cells = sorted(cells)
        seq = [station for _, _, station in cells]
        if len(seq) >= 2:
            sequences.append(seq)

    return sequences


def diagonal_down_left_sequences(grid, rows, cols):
    groups = {}

    for r in range(rows):
        for c in range(cols):
            station = grid[r][c]
            if station is None:
                continue
            groups.setdefault(r + c, []).append((r, c, station))

    sequences = []
    for cells in groups.values():
        cells = sorted(cells)
        seq = [station for _, _, station in cells]
        if len(seq) >= 2:
            sequences.append(seq)

    return sequences


def make_candidate_lines(stations):
    grid, rows, cols = station_grid(stations)

    candidates = []

    ring = perimeter_ring(grid, rows, cols)
    if len(ring) >= 4:
        candidates.append(("Ring", "Ring", ring))

    # East-west lines: one per row.
    for r in range(rows):
        seq = row_sequence(grid, r)
        if len(seq) >= 2:
            candidates.append((f"East-West-{r + 1}", "East-West", seq))

    # South-north lines: one per column.
    for c in range(cols):
        seq = column_sequence(grid, rows, c)
        if len(seq) >= 2:
            candidates.append((f"South-North-{c + 1}", "South-North", seq))

    # Diagonal lines.
    for i, seq in enumerate(diagonal_down_right_sequences(grid, rows, cols), start=1):
        candidates.append((f"Diagonal-SE-{i}", "Diagonal", seq))

    for i, seq in enumerate(diagonal_down_left_sequences(grid, rows, cols), start=1):
        candidates.append((f"Diagonal-SW-{i}", "Diagonal", seq))

    return candidates


def choose_lines(stations, seed=None):
    rng = random.Random(seed)
    candidates = make_candidate_lines(stations)

    if not candidates:
        raise ValueError("No candidate lines could be generated.")

    selected = []
    used_names = set()
    covered = set()

    def add_candidate(candidate):
        name, kind, stops = candidate
        if name in used_names:
            return False
        if len(stops) < 2:
            return False

        line_index = len(selected)
        line_name = name

        selected.append(
            (
                line_name,
                {
                    "kind": kind,
                    "color": LINE_COLORS[line_index % len(LINE_COLORS)],
                    "headway": DEFAULT_HEADWAY_MIN + (line_index % 3),
                    "stops": stops,
                },
            )
        )

        used_names.add(name)
        covered.update(stops)
        return True

    # Prefer a ring first because it makes a realistic metro-style outer connector.
    ring_candidates = [c for c in candidates if c[1] == "Ring"]
    if ring_candidates:
        add_candidate(ring_candidates[0])

    # Then add one or more directional city lines.
    preferred_kinds = ["East-West", "South-North", "Diagonal"]

    while len(selected) < NUM_LINES and len(covered) < len(stations):
        viable = [
            c for c in candidates
            if c[0] not in used_names
            and (
                not covered
                or any(station in covered for station in c[2])
                or not FORCE_CONNECTED_NETWORK
            )
        ]

        if not viable:
            break

        def score(candidate):
            name, kind, stops = candidate
            uncovered_gain = len(set(stops) - covered)
            kind_bonus = 2 if kind in preferred_kinds else 0
            length_bonus = len(stops) / 10
            return uncovered_gain * 10 + kind_bonus + length_bonus

        best_score = max(score(c) for c in viable)
        best_candidates = [c for c in viable if score(c) == best_score]
        add_candidate(rng.choice(best_candidates))

    # Fill remaining requested lines with useful alternatives.
    while len(selected) < NUM_LINES:
        viable = [c for c in candidates if c[0] not in used_names]
        if not viable:
            break

        def fill_score(candidate):
            name, kind, stops = candidate
            transfer_overlap = len(set(stops) & covered)
            length_bonus = len(stops)
            diagonal_bonus = 2 if kind == "Diagonal" else 0
            return transfer_overlap + length_bonus + diagonal_bonus

        best_score = max(fill_score(c) for c in viable)
        best_candidates = [c for c in viable if fill_score(c) == best_score]
        add_candidate(rng.choice(best_candidates))

    lines = dict(selected)

    if FORCE_CONNECTED_NETWORK:
        missing = set(stations) - covered
        if missing:
            raise RuntimeError(
                "Could not cover all stations with the current directional-line configuration. "
                f"Missing stations: {sorted(missing)}. "
                "Increase NUM_LINES or adjust LAYOUT_COLUMNS."
            )

    return lines


def generate_travel_times(lines, seed=None):
    rng = random.Random(seed)
    travel_times = {}

    for data in lines.values():
        stops = data["stops"]

        for a, b in zip(stops, stops[1:]):
            key = tuple(sorted((a, b)))

            if key not in travel_times:
                travel_times[key] = rng.randint(
                    MIN_TRAVEL_TIME_MIN,
                    MAX_TRAVEL_TIME_MIN,
                )

    return travel_times


def build_edges_and_adjacency(stations, lines, travel_times):
    edges = {}
    adjacency = {station: [] for station in stations}

    for line_name, data in lines.items():
        stops = data["stops"]

        for a, b in zip(stops, stops[1:]):
            key = tuple(sorted((a, b)))
            travel = travel_times.get(key, DEFAULT_TRAVEL_TIME_MIN)

            edges[(a, b, line_name)] = travel
            edges[(b, a, line_name)] = travel

            adjacency[a].append((b, line_name, travel))
            adjacency[b].append((a, line_name, travel))

    return edges, adjacency


def reachable_stations(start, adjacency):
    if start not in adjacency:
        return set()

    seen = {start}
    queue = deque([start])

    while queue:
        current = queue.popleft()

        for nxt, _, _ in adjacency[current]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    return seen


def connected_components(stations, adjacency):
    unseen = set(stations)
    components = []

    while unseen:
        start = next(iter(unseen))
        component = reachable_stations(start, adjacency)
        components.append(sorted(component))
        unseen -= component

    return components


def is_reachable(start, destination):
    return destination in reachable_stations(start, ADJACENCY)


def is_fully_connected():
    if not STATIONS:
        return True

    return len(reachable_stations(STATIONS[0], ADJACENCY)) == len(STATIONS)


def assert_fully_connected():
    if FORCE_CONNECTED_NETWORK and not is_fully_connected():
        raise RuntimeError(
            "Network is not fully connected although FORCE_CONNECTED_NETWORK=True. "
            f"Components: {connected_components(STATIONS, ADJACENCY)}"
        )


STATIONS = get_station_names(NUM_STATIONS)
STATION_POS = generate_station_positions(STATIONS)
LINES = choose_lines(STATIONS, NETWORK_SEED)
TRAVEL_TIMES = generate_travel_times(LINES, NETWORK_SEED)
EDGES, ADJACENCY = build_edges_and_adjacency(STATIONS, LINES, TRAVEL_TIMES)

assert_fully_connected()


def line_segment_text(line_name):
    stops = LINES[line_name]["stops"]

    return "; ".join(
        f"{a}-{b}: {TRAVEL_TIMES.get(tuple(sorted((a, b))), DEFAULT_TRAVEL_TIME_MIN)} min"
        for a, b in zip(stops, stops[1:])
    )


def compact_network_text():
    return " ".join(
        f"{line}({data['headway']}m,{data.get('kind', 'Line')}):{'-'.join(data['stops'])}."
        for line, data in LINES.items()
    )


def compact_travel_time_text():
    return " ".join(
        f"{a}-{b}:{minutes}m."
        for (a, b), minutes in sorted(TRAVEL_TIMES.items())
    )