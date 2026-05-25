"""Transit network model. It generates stations, line candidates, travel times, connectivity, and compact prompt descriptions.
"""
import math
import random
from collections import deque
from functools import lru_cache

from minillama.model.config import (
    NUM_STATIONS,
    NUM_LINES,
    LAYOUT_COLUMNS,
    START_TIME_MIN,
    DEFAULT_HEADWAY_MIN,
    DEFAULT_TRAVEL_TIME_MIN,
    MIN_TRAVEL_TIME_MIN,
    MAX_TRAVEL_TIME_MIN,
    NETWORK_SEED,
    LINE_FULLNESS_SEED,
    MIN_LINE_FULLNESS_PERCENT,
    MAX_LINE_FULLNESS_PERCENT,
    STATION_DEMAND_SEED,
    MIN_STATION_FULLNESS_PERCENT,
    MAX_STATION_FULLNESS_PERCENT,
    MORNING_PEAK_CENTER_MIN,
    EVENING_PEAK_CENTER_MIN,
    MIDDAY_PEAK_CENTER_MIN,
    LATE_PEAK_CENTER_MIN,
    PEAK_SPREAD_MIN,
    FORCE_CONNECTED_NETWORK,
    LINE_COLORS,
    STATION_X_GAP,
    STATION_Y_GAP,
    STATION_X_OFFSET,
    STATION_Y_OFFSET,
    STATION_ROW_STAGGER,
    MIN_RING_STATIONS,
    MIN_LINE_STATIONS,
    HEADWAY_VARIATION_MOD,
    LINE_COVERAGE_SCORE_WEIGHT,
    LINE_KIND_BONUS,
    LINE_LENGTH_SCORE_DIVISOR,
    LINE_STOP_OVERRIDES,
)
from minillama.model.station_names import get_station_names


def grid_dimensions(num_stations: int):
    """Grid dimensions function for this module's MVC responsibility.
    
    Args:
        num_stations: Input value used by `grid_dimensions`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    if LAYOUT_COLUMNS is not None:
        cols = max(2, int(LAYOUT_COLUMNS))
    else:
        cols = max(2, math.ceil(math.sqrt(num_stations)))

    rows = math.ceil(num_stations / cols)
    return rows, cols


def station_grid(stations):
    """Station grid function for this module's MVC responsibility.
    
    Args:
        stations: Input value used by `station_grid`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
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
    """Generate station positions function for this module's MVC responsibility.
    
    Args:
        stations: Input value used by `generate_station_positions`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    grid, rows, cols = station_grid(stations)

    positions = {}

    for r in range(rows):
        for c in range(cols):
            station = grid[r][c]
            if station is None:
                continue

            # Small stagger for a less rigid schematic while preserving clear directions.
            x = STATION_X_OFFSET + c * STATION_X_GAP + (STATION_ROW_STAGGER if r % 2 else 0)
            y = STATION_Y_OFFSET + r * STATION_Y_GAP

            positions[station] = (x, y)

    return positions


def valid_sequence(seq):
    """Valid sequence function for this module's MVC responsibility.
    
    Args:
        seq: Input value used by `valid_sequence`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return [station for station in seq if station is not None]


def perimeter_ring(grid, rows, cols):
    """Perimeter ring function for this module's MVC responsibility.
    
    Args:
        grid: Input value used by `perimeter_ring`; see the function signature and caller context for the expected type.
        rows: Input value used by `perimeter_ring`; see the function signature and caller context for the expected type.
        cols: Input value used by `perimeter_ring`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
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
    """Row sequence function for this module's MVC responsibility.
    
    Args:
        grid: Input value used by `row_sequence`; see the function signature and caller context for the expected type.
        r: Input value used by `row_sequence`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return valid_sequence(grid[r])


def column_sequence(grid, rows, c):
    """Column sequence function for this module's MVC responsibility.
    
    Args:
        grid: Input value used by `column_sequence`; see the function signature and caller context for the expected type.
        rows: Input value used by `column_sequence`; see the function signature and caller context for the expected type.
        c: Input value used by `column_sequence`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return valid_sequence(grid[r][c] for r in range(rows))


def diagonal_down_right_sequences(grid, rows, cols):
    """Diagonal down right sequences function for this module's MVC responsibility.
    
    Args:
        grid: Input value used by `diagonal_down_right_sequences`; see the function signature and caller context for the expected type.
        rows: Input value used by `diagonal_down_right_sequences`; see the function signature and caller context for the expected type.
        cols: Input value used by `diagonal_down_right_sequences`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
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
    """Diagonal down left sequences function for this module's MVC responsibility.
    
    Args:
        grid: Input value used by `diagonal_down_left_sequences`; see the function signature and caller context for the expected type.
        rows: Input value used by `diagonal_down_left_sequences`; see the function signature and caller context for the expected type.
        cols: Input value used by `diagonal_down_left_sequences`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
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
    """Make candidate lines function for this module's MVC responsibility.
    
    Args:
        stations: Input value used by `make_candidate_lines`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    grid, rows, cols = station_grid(stations)

    candidates = []

    ring = perimeter_ring(grid, rows, cols)
    if len(ring) >= MIN_RING_STATIONS:
        candidates.append(("Ring", "Ring", ring))

    # East-west lines: one per row.
    for r in range(rows):
        seq = row_sequence(grid, r)
        if len(seq) >= MIN_LINE_STATIONS:
            candidates.append((f"East-West-{r + 1}", "East-West", seq))

    # South-north lines: one per column.
    for c in range(cols):
        seq = column_sequence(grid, rows, c)
        if len(seq) >= MIN_LINE_STATIONS:
            candidates.append((f"South-North-{c + 1}", "South-North", seq))

    # Diagonal lines.
    for i, seq in enumerate(diagonal_down_right_sequences(grid, rows, cols), start=1):
        candidates.append((f"Diagonal-SE-{i}", "Diagonal", seq))

    for i, seq in enumerate(diagonal_down_left_sequences(grid, rows, cols), start=1):
        candidates.append((f"Diagonal-SW-{i}", "Diagonal", seq))

    return candidates


def choose_lines(stations, seed=None):
    """Choose lines function for this module's MVC responsibility.
    
    Args:
        stations: Input value used by `choose_lines`; see the function signature and caller context for the expected type.
        seed: Input value used by `choose_lines`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    rng = random.Random(seed)
    candidates = make_candidate_lines(stations)

    if not candidates:
        raise ValueError("No candidate lines could be generated.")

    selected = []
    used_names = set()
    covered = set()

    def add_candidate(candidate):
        """Add candidate function for this module's MVC responsibility.
        
        Args:
            candidate: Input value used by `add_candidate`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        name, kind, stops = candidate
        if name in used_names:
            return False
        if len(stops) < MIN_LINE_STATIONS:
            return False

        line_index = len(selected)
        line_name = name

        selected.append(
            (
                line_name,
                {
                    "kind": kind,
                    "color": LINE_COLORS[line_index % len(LINE_COLORS)],
                    "headway": DEFAULT_HEADWAY_MIN + (line_index % HEADWAY_VARIATION_MOD),
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
            """Score function for this module's MVC responsibility.
            
            Args:
                candidate: Input value used by `score`; see the function signature and caller context for the expected type.
            
            Returns:
                The computed value or side effect documented by the implementation.
            """
            name, kind, stops = candidate
            uncovered_gain = len(set(stops) - covered)
            kind_bonus = LINE_KIND_BONUS if kind in preferred_kinds else 0
            length_bonus = len(stops) / LINE_LENGTH_SCORE_DIVISOR
            return uncovered_gain * LINE_COVERAGE_SCORE_WEIGHT + kind_bonus + length_bonus

        best_score = max(score(c) for c in viable)
        best_candidates = [c for c in viable if score(c) == best_score]
        add_candidate(rng.choice(best_candidates))

    # Fill remaining requested lines with useful alternatives.
    while len(selected) < NUM_LINES:
        viable = [c for c in candidates if c[0] not in used_names]
        if not viable:
            break

        def fill_score(candidate):
            """Fill score function for this module's MVC responsibility.
            
            Args:
                candidate: Input value used by `fill_score`; see the function signature and caller context for the expected type.
            
            Returns:
                The computed value or side effect documented by the implementation.
            """
            name, kind, stops = candidate
            transfer_overlap = len(set(stops) & covered)
            length_bonus = len(stops)
            diagonal_bonus = LINE_KIND_BONUS if kind == "Diagonal" else 0
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

    apply_line_overrides(lines, stations)
    apply_line_fullness(lines, LINE_FULLNESS_SEED)
    return lines


def apply_line_fullness(lines, seed=None):
    """Attach deterministic current fullness percentages to each line."""
    rng = random.Random(seed)
    for line_name in sorted(lines):
        lines[line_name]["fullness"] = rng.randint(
            MIN_LINE_FULLNESS_PERCENT,
            MAX_LINE_FULLNESS_PERCENT,
        )


def build_station_profiles(stations, lines, seed=None):
    """Create deterministic station demand profiles used for time-varying crowding."""
    rng = random.Random(seed)
    grid, rows, cols = station_grid(stations)
    lookup = {}
    for r in range(rows):
        for c in range(cols):
            station = grid[r][c]
            if station is not None:
                lookup[station] = (r, c)

    profiles = {}
    mid_col = max(1, cols - 1) / 2

    for index, station in enumerate(stations):
        row, col = lookup[station]
        interchange_degree = sum(1 for data in lines.values() if station in data["stops"])
        radial_distance = abs(col - mid_col) / max(mid_col, 1)
        edge_bias = row / max(rows - 1, 1)
        local_rng = random.Random((seed or 0) + index * 97)

        if interchange_degree >= 4:
            district = "hub"
            baseline = 56 + local_rng.randint(0, 8)
            morning_peak = 18 + local_rng.randint(2, 10)
            evening_peak = 16 + local_rng.randint(2, 8)
            midday_peak = 10 + local_rng.randint(0, 6)
            late_peak = 5 + local_rng.randint(0, 4)
        elif row <= 1 and radial_distance < 0.45:
            district = "business"
            baseline = 42 + local_rng.randint(0, 8)
            morning_peak = 20 + local_rng.randint(3, 10)
            evening_peak = 10 + local_rng.randint(0, 6)
            midday_peak = 8 + local_rng.randint(0, 5)
            late_peak = 3 + local_rng.randint(0, 3)
        elif edge_bias >= 0.6:
            district = "residential"
            baseline = 24 + local_rng.randint(0, 8)
            morning_peak = 16 + local_rng.randint(3, 10)
            evening_peak = 20 + local_rng.randint(4, 10)
            midday_peak = 4 + local_rng.randint(0, 4)
            late_peak = 4 + local_rng.randint(0, 4)
        elif radial_distance < 0.35:
            district = "mixed_core"
            baseline = 36 + local_rng.randint(0, 8)
            morning_peak = 14 + local_rng.randint(2, 8)
            evening_peak = 14 + local_rng.randint(2, 8)
            midday_peak = 9 + local_rng.randint(1, 6)
            late_peak = 5 + local_rng.randint(0, 4)
        else:
            district = "leisure"
            baseline = 28 + local_rng.randint(0, 8)
            morning_peak = 8 + local_rng.randint(0, 6)
            evening_peak = 13 + local_rng.randint(1, 7)
            midday_peak = 8 + local_rng.randint(0, 5)
            late_peak = 10 + local_rng.randint(2, 8)

        profiles[station] = {
            "district": district,
            "interchange_degree": interchange_degree,
            "baseline": baseline,
            "morning_peak": morning_peak,
            "evening_peak": evening_peak,
            "midday_peak": midday_peak,
            "late_peak": late_peak,
            "wave_amplitude": 3 + local_rng.randint(0, 4),
            "wave_phase": rng.randint(0, 180),
        }

    return profiles


def gaussian_peak(minute, center_min, spread_min):
    """Return a smooth bell-shaped multiplier around a target minute."""
    delta = minute - center_min
    return math.exp(-((delta / spread_min) ** 2))


@lru_cache(maxsize=4096)
def station_fullness_percent(station, current_time_min):
    """Return the current station crowding percentage."""
    profile = STATION_PROFILES[station]
    wave = profile["wave_amplitude"] * math.sin(
        ((current_time_min + profile["wave_phase"]) / 180) * math.pi
    )
    raw = (
        profile["baseline"]
        + profile["morning_peak"] * gaussian_peak(current_time_min, MORNING_PEAK_CENTER_MIN, PEAK_SPREAD_MIN)
        + profile["evening_peak"] * gaussian_peak(current_time_min, EVENING_PEAK_CENTER_MIN, PEAK_SPREAD_MIN + 20)
        + profile["midday_peak"] * gaussian_peak(current_time_min, MIDDAY_PEAK_CENTER_MIN, PEAK_SPREAD_MIN - 10)
        + profile["late_peak"] * gaussian_peak(current_time_min, LATE_PEAK_CENTER_MIN, PEAK_SPREAD_MIN)
        + wave
    )
    return max(
        MIN_STATION_FULLNESS_PERCENT,
        min(MAX_STATION_FULLNESS_PERCENT, round(raw)),
    )


@lru_cache(maxsize=2048)
def line_fullness_percent(line_name, current_time_min):
    """Return average line crowding based on the current station demand profile."""
    stops = LINES[line_name]["stops"]
    if not stops:
        return MIN_LINE_FULLNESS_PERCENT
    values = [station_fullness_percent(station, current_time_min) for station in stops]
    return max(
        MIN_LINE_FULLNESS_PERCENT,
        min(MAX_LINE_FULLNESS_PERCENT, round(sum(values) / len(values))),
    )


def segment_fullness_percent(line_name, station_a, station_b, current_time_min):
    """Return crowding for a traversed segment using local station and line load."""
    values = [
        station_fullness_percent(station_a, current_time_min),
        station_fullness_percent(station_b, current_time_min),
        line_fullness_percent(line_name, current_time_min),
    ]
    return round(sum(values) / len(values))


def apply_line_overrides(lines, stations):
    """Apply line overrides function for this module's MVC responsibility.
    
    Args:
        lines: Input value used by `apply_line_overrides`; see the function signature and caller context for the expected type.
        stations: Input value used by `apply_line_overrides`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    for line_name, stops in LINE_STOP_OVERRIDES.items():
        if line_name in lines and all(station in stations for station in stops):
            lines[line_name]["stops"] = stops


def line_stop_pairs(line_name, data):
    """Line stop pairs function for this module's MVC responsibility.
    
    Args:
        line_name: Input value used by `line_stop_pairs`; see the function signature and caller context for the expected type.
        data: Input value used by `line_stop_pairs`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    stops = data["stops"]
    pairs = list(zip(stops, stops[1:]))
    if data.get("kind") == "Ring" and len(stops) > 2:
        pairs.append((stops[-1], stops[0]))
    return pairs


def generate_travel_times(lines, seed=None):
    """Generate travel times function for this module's MVC responsibility.
    
    Args:
        lines: Input value used by `generate_travel_times`; see the function signature and caller context for the expected type.
        seed: Input value used by `generate_travel_times`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    rng = random.Random(seed)
    travel_times = {}

    for line_name, data in lines.items():
        for a, b in line_stop_pairs(line_name, data):
            key = tuple(sorted((a, b)))

            if key not in travel_times:
                travel_times[key] = rng.randint(
                    MIN_TRAVEL_TIME_MIN,
                    MAX_TRAVEL_TIME_MIN,
                )

    return travel_times


def build_edges_and_adjacency(stations, lines, travel_times):
    """Build edges and adjacency function for this module's MVC responsibility.
    
    Args:
        stations: Input value used by `build_edges_and_adjacency`; see the function signature and caller context for the expected type.
        lines: Input value used by `build_edges_and_adjacency`; see the function signature and caller context for the expected type.
        travel_times: Input value used by `build_edges_and_adjacency`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    edges = {}
    adjacency = {station: [] for station in stations}

    for line_name, data in lines.items():
        for a, b in line_stop_pairs(line_name, data):
            key = tuple(sorted((a, b)))
            travel = travel_times.get(key, DEFAULT_TRAVEL_TIME_MIN)

            edges[(a, b, line_name)] = travel
            edges[(b, a, line_name)] = travel

            adjacency[a].append((b, line_name, travel))
            adjacency[b].append((a, line_name, travel))

    return edges, adjacency


def reachable_stations(start, adjacency):
    """Reachable stations function for this module's MVC responsibility.
    
    Args:
        start: Input value used by `reachable_stations`; see the function signature and caller context for the expected type.
        adjacency: Input value used by `reachable_stations`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
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
    """Connected components function for this module's MVC responsibility.
    
    Args:
        stations: Input value used by `connected_components`; see the function signature and caller context for the expected type.
        adjacency: Input value used by `connected_components`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    unseen = set(stations)
    components = []

    while unseen:
        start = next(iter(unseen))
        component = reachable_stations(start, adjacency)
        components.append(sorted(component))
        unseen -= component

    return components


def is_reachable(start, destination):
    """Is reachable function for this module's MVC responsibility.
    
    Args:
        start: Input value used by `is_reachable`; see the function signature and caller context for the expected type.
        destination: Input value used by `is_reachable`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return destination in reachable_stations(start, ADJACENCY)


def is_fully_connected():
    """Is fully connected function for this module's MVC responsibility.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    if not STATIONS:
        return True

    return len(reachable_stations(STATIONS[0], ADJACENCY)) == len(STATIONS)


def assert_fully_connected():
    """Assert fully connected function for this module's MVC responsibility.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
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
STATION_PROFILES = build_station_profiles(STATIONS, LINES, STATION_DEMAND_SEED)

for line_name in LINES:
    LINES[line_name]["fullness"] = line_fullness_percent(line_name, START_TIME_MIN)

assert_fully_connected()


def line_segment_text(line_name):
    """Line segment text function for this module's MVC responsibility.
    
    Args:
        line_name: Input value used by `line_segment_text`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return "; ".join(
        f"{a} to {b}: {TRAVEL_TIMES.get(tuple(sorted((a, b))), DEFAULT_TRAVEL_TIME_MIN)} minutes"
        for a, b in line_stop_pairs(line_name, LINES[line_name])
    )


@lru_cache(maxsize=1)
def compact_network_text():
    """Compact network text function for this module's MVC responsibility.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return " ".join(
        f"{line} ({data['headway']} minutes, {data.get('kind', 'Line')}): {line_stop_sequence_text(line, data)}."
        for line, data in LINES.items()
    )


@lru_cache(maxsize=256)
def compact_line_fullness_text(current_time_min=START_TIME_MIN):
    """Compact current line crowding text for prompts."""
    return " ".join(
        f"{line}: {line_fullness_percent(line, current_time_min)} percent full."
        for line in LINES
    )


@lru_cache(maxsize=256)
def compact_station_crowding_text(current_time_min=START_TIME_MIN, limit=8):
    """Return the currently busiest stations as compact prompt context."""
    busiest = sorted(
        (
            (station, station_fullness_percent(station, current_time_min))
            for station in STATIONS
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    return " ".join(f"{station}: {fullness} percent busy." for station, fullness in busiest)


def line_stop_sequence_text(line_name, data):
    """Line stop sequence text function for this module's MVC responsibility.
    
    Args:
        line_name: Input value used by `line_stop_sequence_text`; see the function signature and caller context for the expected type.
        data: Input value used by `line_stop_sequence_text`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    stops = data["stops"]
    if data.get("kind") == "Ring" and len(stops) > 2:
        return " to ".join(stops + [stops[0]])
    return " to ".join(stops)


@lru_cache(maxsize=1)
def compact_travel_time_text():
    """Compact travel time text function for this module's MVC responsibility.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return " ".join(
        f"{a} to {b}: {minutes} minutes."
        for (a, b), minutes in sorted(TRAVEL_TIMES.items())
    )
