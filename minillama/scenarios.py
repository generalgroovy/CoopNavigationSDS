"""Scenario model definitions for start/destination/time/transfer settings used by dialogs and experiments.
"""
from minillama.config import START_TIME_MIN, TRANSFER_TIME_MIN
from minillama.metro_data import STATIONS, is_reachable


def make_scenario(
    name,
    start_station,
    destination_station,
    start_time_min=START_TIME_MIN,
    transfer_time_min=TRANSFER_TIME_MIN,
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
        "start_time_min": start_time_min,
        "transfer_time_min": transfer_time_min,
        "goal": "fastest_route",
        "allow_unreachable": allow_unreachable,
    }


SCENARIOS = {
    "default": make_scenario(
        "Default metro scenario",
        STATIONS[1],
        STATIONS[-2],
    ),
    "short_cross": make_scenario(
        "Short cross-network trip",
        STATIONS[0],
        STATIONS[min(8, len(STATIONS) - 1)],
    ),
    "long_cross": make_scenario(
        "Long cross-network trip",
        STATIONS[2],
        STATIONS[-1],
    ),
}


DEFAULT_SCENARIO = "default"


def get_scenario(scenario_key: str):
    """Get scenario function for this module's MVC responsibility.
    
    Args:
        scenario_key: Input value used by `get_scenario`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return SCENARIOS.get(scenario_key, SCENARIOS[DEFAULT_SCENARIO])
