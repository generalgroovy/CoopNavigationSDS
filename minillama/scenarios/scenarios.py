"""Scenario model definitions for start/destination/time/transfer settings used by dialogs and experiments.
"""
from minillama.network.config import (
    DEFAULT_ALLOWED_MODES,
    ACCEPTABLE_DURATION_RATIO,
    MAX_ROUTE_DELAY_PROBABILITY,
    MAX_TRANSFER_MISS_PROBABILITY,
    MIN_STAGE_SUBOPTIMAL_OPTIONS,
    REQUIRE_STAGE_SUBOPTIMAL_OPTIONS,
    START_TIME_MIN,
    TRANSFER_TIME_MIN,
)
from minillama.network.metro_data import STATIONS, is_reachable
from minillama.scenarios.config import DEFAULT_SCENARIO, SCENARIO_SPECS


def make_scenario(
    name,
    start_station,
    destination_station,
    destination_stations=None,
    start_time_min=START_TIME_MIN,
    transfer_time_min=TRANSFER_TIME_MIN,
    allowed_modes=DEFAULT_ALLOWED_MODES,
    max_transfer_miss_probability=MAX_TRANSFER_MISS_PROBABILITY,
    max_delay_probability=MAX_ROUTE_DELAY_PROBABILITY,
    max_walking_min=None,
    acceptable_duration_ratio=ACCEPTABLE_DURATION_RATIO,
    min_stage_suboptimal_options=MIN_STAGE_SUBOPTIMAL_OPTIONS,
    require_stage_suboptimal_options=REQUIRE_STAGE_SUBOPTIMAL_OPTIONS,
    allow_unreachable=False,
):
    """Make scenario function for this module's MVC responsibility.
    
    Args:
        name: Input value used by `make_scenario`; see the function signature and caller context for the expected type.
        start_station: Input value used by `make_scenario`; see the function signature and caller context for the expected type.
        destination_station: Input value used by `make_scenario`; see the function signature and caller context for the expected type.
        start_time_min: Input value used by `make_scenario`; see the function signature and caller context for the expected type.
        transfer_time_min: Input value used by `make_scenario`; see the function signature and caller context for the expected type.
        allow_unreachable: Input value used by `make_scenario`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    destinations = list(destination_stations or [destination_station])
    if destination_station not in destinations:
        destinations.insert(0, destination_station)

    if start_station == destination_station:
        raise ValueError("Scenario start and destination must differ.")

    if not allow_unreachable and not is_reachable(start_station, destination_station):
        raise ValueError(
            f"Scenario is unreachable: {start_station} -> {destination_station}. "
            "Set allow_unreachable=True only when intentionally testing failure cases."
        )

    return {
        "name": name,
        "start_station": start_station,
        "destination_station": destination_station,
        "destination_stations": destinations,
        "start_time_min": start_time_min,
        "transfer_time_min": transfer_time_min,
        "allowed_modes": tuple(allowed_modes or DEFAULT_ALLOWED_MODES),
        "max_transfer_miss_probability": max_transfer_miss_probability,
        "max_delay_probability": max_delay_probability,
        "max_walking_min": max_walking_min,
        "acceptable_duration_ratio": float(acceptable_duration_ratio),
        "min_stage_suboptimal_options": int(min_stage_suboptimal_options),
        "require_stage_suboptimal_options": bool(require_stage_suboptimal_options),
        "goal": "fastest_route",
        "allow_unreachable": allow_unreachable,
    }


def _station_at(index_spec):
    """Resolve an index specification into a station name."""
    if index_spec == "middle":
        return STATIONS[len(STATIONS) // 2]
    return STATIONS[index_spec]


def _stations_at(index_specs):
    return [_station_at(index_spec) for index_spec in index_specs]


SCENARIOS = {
    key: make_scenario(
        spec["name"],
        _station_at(spec["start_station_index"]),
        _station_at(spec["destination_station_index"]),
        destination_stations=_stations_at(spec.get("destination_station_indices", [spec["destination_station_index"]])),
        start_time_min=spec["start_time_min"],
        allowed_modes=spec.get("allowed_modes", DEFAULT_ALLOWED_MODES),
        max_transfer_miss_probability=spec.get("max_transfer_miss_probability", MAX_TRANSFER_MISS_PROBABILITY),
        max_delay_probability=spec.get("max_delay_probability", MAX_ROUTE_DELAY_PROBABILITY),
        max_walking_min=spec.get("max_walking_min"),
        acceptable_duration_ratio=spec.get("acceptable_duration_ratio", ACCEPTABLE_DURATION_RATIO),
        min_stage_suboptimal_options=spec.get("min_stage_suboptimal_options", MIN_STAGE_SUBOPTIMAL_OPTIONS),
        require_stage_suboptimal_options=spec.get("require_stage_suboptimal_options", REQUIRE_STAGE_SUBOPTIMAL_OPTIONS),
    )
    for key, spec in SCENARIO_SPECS.items()
}


def get_scenario(scenario_key: str):
    """Get scenario function for this module's MVC responsibility.
    
    Args:
        scenario_key: Input value used by `get_scenario`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return SCENARIOS.get(scenario_key, SCENARIOS[DEFAULT_SCENARIO])
