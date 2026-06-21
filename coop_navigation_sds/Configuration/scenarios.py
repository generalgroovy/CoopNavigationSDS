"""Test-case and scenario configuration."""

from coop_navigation_sds.NaturalLanguageGeneration.caller.config import DEFAULT_PERSONA

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
        "max_transfer_miss_probability": 0.28,
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
    "airport_connection": {
        "name": "Airport-style low-delay connection evaluation",
        "start_station_index": 0,
        "destination_station_index": -2,
        "start_time_min": 6 * 60 + 35,
        "max_delay_probability": 0.32,
        "max_transfer_miss_probability": 0.20,
    },
    "hospital_appointment": {
        "name": "Appointment reliability and low-transfer evaluation",
        "start_station_index": 7,
        "destination_station_index": 28,
        "start_time_min": 9 * 60 + 12,
        "max_transfer_miss_probability": 0.18,
    },
    "crowded_event_exit": {
        "name": "Crowded event exit comfort evaluation",
        "start_station_index": -1,
        "destination_station_index": 5,
        "start_time_min": 22 * 60 + 18,
    },
    "multi_destination_errands": {
        "name": "Multi-destination errand route evaluation",
        "start_station_index": 3,
        "destination_station_index": 16,
        "destination_station_indices": [16, 24, -4],
        "start_time_min": 14 * 60 + 5,
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
    "airport_connection": {
        "name": "Airport-style low-delay automatic-evaluation test",
        "persona_key": "delay_sensitive_traveler",
        "scenario_key": "airport_connection",
    },
    "hospital_appointment": {
        "name": "Appointment low-transfer automatic-evaluation test",
        "persona_key": "accessibility_rider",
        "scenario_key": "hospital_appointment",
    },
    "crowded_event_exit": {
        "name": "Crowded event exit automatic-evaluation test",
        "persona_key": "crowd_averse_rider",
        "scenario_key": "crowded_event_exit",
    },
    "multi_destination_errands": {
        "name": "Multi-destination automatic-evaluation test",
        "persona_key": "multi_stop_errand_runner",
        "scenario_key": "multi_destination_errands",
    },
}
