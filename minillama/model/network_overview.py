"""Structured transit-network data for views and tests."""
from dataclasses import dataclass

from minillama.model.metro_data import (
    ADJACENCY,
    LINES,
    STATION_POS,
    line_fullness_percent,
    line_stop_pairs,
    station_fullness_percent,
)
from minillama.model.route_planner import segment_travel


@dataclass(frozen=True)
class NetworkLineRow:
    name: str
    kind: str
    headway_min: int
    fullness_percent: int
    stop_count: int
    route: str
    segments: str


@dataclass(frozen=True)
class NetworkStationRow:
    name: str
    fullness_percent: int
    lines: str
    neighbors: str
    coordinates: str


@dataclass(frozen=True)
class NetworkOverview:
    line_count: int
    station_count: int
    segment_count: int
    lines: list[NetworkLineRow]
    stations: list[NetworkStationRow]


def build_network_overview(current_time_min) -> NetworkOverview:
    """Build complete network rows without view-layer calculations."""
    line_rows = [
        NetworkLineRow(
            name=line_name,
            kind=data.get("kind", "Line"),
            headway_min=data["headway"],
            fullness_percent=line_fullness_percent(line_name, current_time_min),
            stop_count=len(data["stops"]),
            route=" -> ".join(data["stops"]),
            segments="; ".join(
                f"{a}-{b}:{segment_travel(a, b)}m"
                for a, b in line_stop_pairs(line_name, data)
            ),
        )
        for line_name, data in sorted(LINES.items())
    ]

    station_rows = []
    for station in sorted(STATION_POS):
        lines = sorted(line for line, data in LINES.items() if station in data["stops"])
        neighbors = sorted({next_station for next_station, _, _ in ADJACENCY[station]})
        x, y = STATION_POS[station]
        station_rows.append(
            NetworkStationRow(
                name=station,
                fullness_percent=station_fullness_percent(station, current_time_min),
                lines=", ".join(lines),
                neighbors=", ".join(neighbors),
                coordinates=f"{x},{y}",
            )
        )

    return NetworkOverview(
        line_count=len(line_rows),
        station_count=len(station_rows),
        segment_count=sum(len(data["stops"]) - 1 for data in LINES.values()),
        lines=line_rows,
        stations=station_rows,
    )
