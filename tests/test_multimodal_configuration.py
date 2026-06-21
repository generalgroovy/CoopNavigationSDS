import json
from pathlib import Path
import tempfile
import unittest

from coop_navigation_sds.Configuration.jobs import (
    job_parameter_grid,
    load_experiment_job,
    numeric_range_values,
)
from coop_navigation_sds.NaturalLanguageGeneration.caller.config import PERSONAS
from coop_navigation_sds.TransportNetwork.constraints import (
    RouteConstraintProfile,
    route_allowed_modes,
    route_walking_minutes,
)
from coop_navigation_sds.TransportNetwork.network import LINES
from coop_navigation_sds.TransportNetwork.routes import estimate_route_time
from coop_navigation_sds.experiments import build_condition_grid


class MultimodalConfigurationTests(unittest.TestCase):
    def test_every_persona_has_two_tickets_and_a_reasonable_walking_limit(self):
        for key, persona in PERSONAS.items():
            preferences = persona["preferences"]
            self.assertEqual(len(preferences["ticket_modes"]), 2, key)
            self.assertIn(preferences["max_walking_min"], {5, 10}, key)

    def test_scenario_access_settings_override_persona_defaults(self):
        profile = RouteConstraintProfile.from_persona(
            PERSONAS["focused_commuter"],
            {"ticket_modes": "tram,bus", "max_walking_min": 5},
        )
        self.assertEqual(profile.ticket_modes, ("tram", "bus"))
        self.assertEqual(profile.max_walking_min, 5)
        self.assertEqual(route_allowed_modes({"ticket_modes": ["tram", "bus"], "max_walking_min": 5}), ("tram", "bus", "walking", "walking_max:5"))

    def test_walking_limit_is_enforced_cumulatively(self):
        stops = LINES["Walking"]["stops"]
        route = stops[:4]
        unrestricted = estimate_route_time(route, 480, 2, allowed_modes=("walking", "walking_max:30"))
        self.assertIsNotNone(unrestricted)
        _arrival, steps = unrestricted
        walking = route_walking_minutes(steps)
        self.assertGreater(walking, 0)
        self.assertIsNone(
            estimate_route_time(route, 480, 2, allowed_modes=("walking", f"walking_max:{walking - 1}"))
        )

    def test_numeric_job_ranges_are_inclusive_and_crossed_with_value_sets(self):
        self.assertEqual(numeric_range_values({"start": 5, "stop": 10, "step": 2.5}), [5, 7.5, 10])
        job = {
            "parameter_values": {"ticket_modes": [["metro", "tram"], ["tram", "bus"]]},
            "parameter_ranges": {"max_walking_min": {"start": 5, "stop": 10, "step": 5}},
        }
        grid = job_parameter_grid(job)
        conditions = list(build_condition_grid(
            test_case_keys=["case"],
            persona_keys=["persona"],
            speech_pattern_keys=["clean"],
            model_param_keys=["greedy"],
            iterations=1,
            parameter_grid=grid,
        ))
        self.assertEqual(len(conditions), 4)
        self.assertEqual(
            {tuple(condition.parameter_values["ticket_modes"]) for condition in conditions},
            {("metro", "tram"), ("tram", "bus")},
        )
        self.assertEqual({condition.parameter_values["max_walking_min"] for condition in conditions}, {5, 10})

    def test_job_loader_rejects_non_object_parameter_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.job"
            path.write_text(json.dumps({"parameter_ranges": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parameter_ranges"):
                load_experiment_job(path)


if __name__ == "__main__":
    unittest.main()
