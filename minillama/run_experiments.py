"""Batch experiment entry point for deterministic route-dialog conditions."""
import argparse
import re

from minillama.config import NUM_TURNS
from minillama.runner import ExperimentRunner, build_condition_grid, write_metrics_csv
from minillama.route_planner import optimal_time_route, route_duration_breakdown, route_station_sequence


class DeterministicOracleRouteAdapter:
    """Deterministic model adapter used as a batch baseline that returns optimal route text."""

    name = "deterministic-oracle-route-baseline"
    device = "none"

    def with_model_params(self, model_param_key: str):
        """Return a parameterized adapter; deterministic baseline ignores params."""
        return self

    def generate(self, prompt: str) -> str:
        """Generate an oracle route response from parsed scenario details."""
        scenario = self._extract_scenario(prompt)
        if scenario:
            start, destination, start_time_min, transfer_time_min = scenario
            _, steps = optimal_time_route(
                start,
                destination,
                start_time_min,
                transfer_time_min,
            )
            stations = route_station_sequence(steps)
            if stations:
                parts = route_duration_breakdown(steps)
                total = parts["travel"] + parts["wait"] + parts["transfer"]
                return (
                    f"The quickest connected route is {' to '.join(stations)}. "
                    f"It takes {total} minutes in total: {parts['travel']} riding, "
                    f"{parts['wait']} waiting, and {parts['transfer']} transfer minutes. "
                    f"I recommend that route from {start} to {destination}."
                )

        return (
            "I need the start station, destination station, current time, and transfer cost to compute the route. "
            "The duration is the riding time plus any waiting and transfer time."
        )

    def _extract_scenario(self, prompt: str):
        """Extract start, destination, time, and transfer settings from prompt."""
        match = re.search(
            r"Current time is (\d+) minutes after midnight\. "
            r"The traveler starts at ([A-Za-z]+) and wants to reach ([A-Za-z]+)\. "
            r"Changing lines costs (\d+) minutes\.",
            prompt,
        )
        if not match:
            return None

        start_time_min, start, destination, transfer_time_min = match.groups()
        return start, destination, int(start_time_min), int(transfer_time_min)


def parse_csv_arg(value):
    """Parse a comma-separated CLI argument into a list."""
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    """Run the configured experiment grid and write metrics output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cases", default="default")
    parser.add_argument("--personas", default="focused_commuter")
    parser.add_argument("--speech-patterns", default="clean,hesitant,compressed,noisy_station")
    parser.add_argument("--model-params", default="greedy")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--num-turns", type=int, default=NUM_TURNS)
    parser.add_argument("--output", default="experiment_metrics.csv")
    args = parser.parse_args()

    conditions = build_condition_grid(
        test_case_keys=parse_csv_arg(args.test_cases),
        persona_keys=parse_csv_arg(args.personas),
        speech_pattern_keys=parse_csv_arg(args.speech_patterns),
        model_param_keys=parse_csv_arg(args.model_params),
        iterations=args.iterations,
    )

    runner = ExperimentRunner(DeterministicOracleRouteAdapter(), args.num_turns)
    _, metrics = runner.run_grid(conditions)
    write_metrics_csv(metrics, args.output)

    print(f"wrote {len(metrics)} metric rows to {args.output}")


if __name__ == "__main__":
    main()
