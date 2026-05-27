"""Structured transit-network data for views and tests."""
from dataclasses import dataclass

from minillama.model.metro_data import (
    ADJACENCY,
    LINES,
    STATION_POS,
    STATION_CLASSES,
    line_fullness_percent,
    line_stop_pairs,
    fullness_class,
    line_delay_probability_class,
    station_fullness_percent,
    capacity_status,
    station_transfer_time_min,
    station_access_modes,
)
from minillama.model.route_planner import segment_travel_on_line


@dataclass(frozen=True)
class NetworkLineRow:
    name: str
    kind: str
    mode: str
    headway_min: int
    fullness_percent: int
    fullness_class: str
    delay_probability_class: str
    capacity_status: str
    stop_count: int
    route: str
    segments: str


@dataclass(frozen=True)
class NetworkStationRow:
    name: str
    station_class: int
    access_modes: str
    fullness_percent: int
    capacity_status: str
    transfer_time_min: int
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
    line_rows = []
    for line_name, data in sorted(LINES.items()):
        fullness_percent = line_fullness_percent(line_name, current_time_min)
        line_rows.append(
            NetworkLineRow(
                name=line_name,
                kind=data.get("kind", "Line"),
                mode=data.get("mode", "bus"),
                headway_min=data["headway"],
                fullness_percent=fullness_percent,
                fullness_class=fullness_class(fullness_percent),
                delay_probability_class=line_delay_probability_class(line_name),
                capacity_status=capacity_status(fullness_percent),
                stop_count=len(data["stops"]),
                route=" to ".join(data["stops"]),
                segments="; ".join(
                    f"{a} to {b}: {segment_travel_on_line(line_name, a, b)} minutes"
                    for a, b in line_stop_pairs(line_name, data)
                ),
            )
        )

    station_rows = []
    for station in sorted(STATION_POS):
        lines = sorted(line for line, data in LINES.items() if station in data["stops"])
        neighbors = sorted({next_station for next_station, _, _ in ADJACENCY[station]})
        x, y = STATION_POS[station]
        fullness_percent = station_fullness_percent(station, current_time_min)
        station_rows.append(
            NetworkStationRow(
                name=station,
                station_class=STATION_CLASSES[station],
                access_modes=", ".join(station_access_modes(station)),
                fullness_percent=fullness_percent,
                capacity_status=capacity_status(fullness_percent),
                transfer_time_min=station_transfer_time_min(station),
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
