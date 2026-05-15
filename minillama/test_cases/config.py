"""Test-case and scenario configuration."""

from minillama.agent_a.config import DEFAULT_PERSONA

DEFAULT_SCENARIO = "morning_peak_cross_city"
DEFAULT_TEST_CASE = "morning_peak_cross_city"

SCENARIO_SPECS = {
    "morning_peak_cross_city": {
        "name": "Morning peak cross-city evaluation",
        "start_station_index": 1,
        "destination_station_index": -3,
        "start_time_min": 8 * 60 + 7,
    },
    "midday_transfer": {
        "name": "Midday transfer-heavy evaluation",
        "start_station_index": 4,
        "destination_station_index": 25,
        "start_time_min": 12 * 60 + 18,
    },
    "evening_outbound": {
        "name": "Evening outbound evaluation",
        "start_station_index": "middle",
        "destination_station_index": 2,
        "start_time_min": 17 * 60 + 42,
    },
    "late_event": {
        "name": "Late-event crowding evaluation",
        "start_station_index": 10,
        "destination_station_index": -1,
        "start_time_min": 21 * 60 + 5,
    },
}

TEST_CASE_SPECS = {
    "morning_peak_cross_city": {
        "name": "Morning peak cross-city automatic-evaluation test",
        "persona_key": DEFAULT_PERSONA,
        "scenario_key": DEFAULT_SCENARIO,
    },
    "midday_transfer": {
        "name": "Midday transfer-heavy automatic-evaluation test",
        "persona_key": DEFAULT_PERSONA,
        "scenario_key": "midday_transfer",
    },
    "evening_outbound": {
        "name": "Evening outbound automatic-evaluation test",
        "persona_key": DEFAULT_PERSONA,
        "scenario_key": "evening_outbound",
    },
    "late_event": {
        "name": "Late-event crowding automatic-evaluation test",
        "persona_key": DEFAULT_PERSONA,
        "scenario_key": "late_event",
    },
}
