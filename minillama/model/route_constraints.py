"""Constraint-aware route baselines for comparing agent proposals."""
from dataclasses import dataclass

from minillama.model.metro_data import is_near_capacity
from minillama.model.config import DEFAULT_ALLOWED_MODES, MAX_ROUTE_DELAY_PROBABILITY, MAX_TRANSFER_MISS_PROBABILITY
from minillama.model.route_planner import (
    candidate_time_routes,
    line_mode,
    normalize_allowed_modes,
    route_line_change_count,
    route_line_sequence,
)


@dataclass(frozen=True)
class RouteConstraintProfile:
    """Normalized persona route preferences used for scoring candidate routes."""
    prefer_fast: bool = True
    prefer_fewer_changes: bool = False
    prefer_less_full: bool = False
    prefer_low_delay: bool = False
    allowed_modes: tuple[str, ...] | None = None
    max_transfer_miss_probability: float = MAX_TRANSFER_MISS_PROBABILITY
    max_delay_probability: float = MAX_ROUTE_DELAY_PROBABILITY

    @classmethod
    def from_persona(cls, persona, scenario=None):
        preferences = persona.get("preferences", {})
        priority = preferences.get("priority", "").lower()
        switching = preferences.get("switching", "").lower()
        fullness = preferences.get("fullness", "").lower()
        reliability = preferences.get("reliability", "").lower()
        prefer_fast = any(term in priority for term in ("fast", "quick", "time"))
        prefer_fewer_changes = any(
            term in switching
            for term in ("fewer", "avoid", "avoiding", "unnecessary", "only for meaningful")
        )
        prefer_less_full = any(
            term in fullness
            for term in ("less crowded", "dislikes", "packed", "very full")
        ) and "does not mind" not in fullness
        prefer_low_delay = any(
            term in f"{priority} {reliability}"
            for term in ("delay", "reliable", "reliability", "on time", "low risk")
        )
        allowed_modes = route_allowed_modes(scenario or {}, persona)
        return cls(
            prefer_fast=prefer_fast or not (prefer_fewer_changes or prefer_less_full or prefer_low_delay),
            prefer_fewer_changes=prefer_fewer_changes,
            prefer_less_full=prefer_less_full,
            prefer_low_delay=prefer_low_delay,
            allowed_modes=allowed_modes,
            max_transfer_miss_probability=float(
                preferences.get(
                    "max_transfer_miss_probability",
                    (scenario or {}).get("max_transfer_miss_probability", MAX_TRANSFER_MISS_PROBABILITY),
                )
            ),
            max_delay_probability=float(
                preferences.get(
                    "max_delay_probability",
                    (scenario or {}).get("max_delay_probability", MAX_ROUTE_DELAY_PROBABILITY),
                )
            ),
        )

    @property
    def label(self):
        parts = []
        if self.prefer_fast:
            parts.append("fastest")
        if self.prefer_fewer_changes:
            parts.append("fewer changes")
        if self.prefer_less_full:
            parts.append("avoid near capacity")
        if self.prefer_low_delay:
            parts.append("lower delay risk")
        if self.allowed_modes:
            parts.append("ticket " + "/".join(self.allowed_modes))
        return ", ".join(parts) if parts else "valid route"


@dataclass(frozen=True)
class ConstraintRoute:
    """Best route under a persona's stated constraints."""
    route: list[str]
    steps: list[dict]
    duration_min: int
    line_sequence: list[str]
    line_change_count: int
    average_fullness: float
    near_capacity_count: int
    has_near_capacity: bool
    delay_probability: float
    transfer_miss_probability: float
    mode_sequence: list[str]
    score: tuple
    label: str


def route_allowed_modes(scenario, persona=None):
    """Return the scenario/persona ticket-mode intersection."""
    persona_preferences = (persona or {}).get("preferences", {})
    scenario_modes = normalize_allowed_modes((scenario or {}).get("allowed_modes"))
    persona_modes = normalize_allowed_modes(persona_preferences.get("allowed_modes"))
    if scenario_modes and persona_modes:
        return tuple(mode for mode in scenario_modes if mode in set(persona_modes)) or ("__none__",)
    return scenario_modes or persona_modes or tuple(DEFAULT_ALLOWED_MODES)


def route_average_fullness(steps):
    values = [step.get("fullness", 0) for step in steps]
    return round(sum(values) / len(values), 2) if values else 0.0


def route_near_capacity_count(steps):
    """Return the number of route segments that are near capacity."""
    return sum(1 for step in steps if is_near_capacity(step.get("fullness", 0)))


def route_has_near_capacity(steps):
    """Return whether any route segment is near capacity."""
    return route_near_capacity_count(steps) > 0


def route_delay_probability(steps):
    """Return route-level delay risk as the maximum segment risk."""
    values = [step.get("delay_probability", 0.0) for step in steps]
    return round(max(values), 4) if values else 0.0


def route_transfer_miss_probability(steps):
    """Return route-level transfer-miss risk as the maximum transfer risk."""
    values = [step.get("transfer_miss_probability", 0.0) for step in steps]
    return round(max(values), 4) if values else 0.0


def route_mode_sequence(steps):
    """Return the transport modes used by a route without consecutive duplicates."""
    sequence = []
    for step in steps:
        mode = step.get("mode") or line_mode(step["line"])
        if not sequence or mode != sequence[-1]:
            sequence.append(mode)
    return sequence


def constraint_sort_key(duration_min, steps, profile):
    line_changes = route_line_change_count(steps)
    average_fullness = route_average_fullness(steps)
    near_capacity_count = route_near_capacity_count(steps)
    delay_probability = route_delay_probability(steps)
    transfer_miss_probability = route_transfer_miss_probability(steps)
    key = []
    key.append(duration_min if profile.prefer_fast else round(duration_min * 0.25))
    if profile.prefer_fewer_changes:
        key.append(line_changes)
    if profile.prefer_less_full:
        key.append(near_capacity_count)
    if profile.prefer_low_delay:
        key.append(delay_probability)
    key.extend([
        duration_min,
        int(transfer_miss_probability > profile.max_transfer_miss_probability),
        int(delay_probability > profile.max_delay_probability),
        line_changes,
        near_capacity_count,
        average_fullness,
        transfer_miss_probability,
        delay_probability,
    ])
    return tuple(key)


def optimal_constraint_route(scenario, persona, limit=50):
    """Compute the best startup baseline route under persona constraints."""
    profile = RouteConstraintProfile.from_persona(persona, scenario)
    candidates = candidate_time_routes(
        scenario["start_station"],
        scenario["destination_station"],
        scenario["start_time_min"],
        scenario["transfer_time_min"],
        limit=limit,
        max_extra_stops=8,
        max_paths=20000,
        allowed_modes=profile.allowed_modes,
    )
    if not candidates:
        return None

    duration_min, route, steps = min(
        candidates,
        key=lambda item: constraint_sort_key(item[0], item[2], profile),
    )
    return ConstraintRoute(
        route=route,
        steps=steps,
        duration_min=duration_min,
        line_sequence=route_line_sequence(steps),
        line_change_count=route_line_change_count(steps),
        average_fullness=route_average_fullness(steps),
        near_capacity_count=route_near_capacity_count(steps),
        has_near_capacity=route_has_near_capacity(steps),
        delay_probability=route_delay_probability(steps),
        transfer_miss_probability=route_transfer_miss_probability(steps),
        mode_sequence=route_mode_sequence(steps),
        score=constraint_sort_key(duration_min, steps, profile),
        label=profile.label,
    )


def ranked_constraint_routes(scenario, persona, limit=6):
    """Return valid routes sorted by the persona's scientific comparison profile."""
    profile = RouteConstraintProfile.from_persona(persona, scenario)
    candidates = candidate_time_routes(
        scenario["start_station"],
        scenario["destination_station"],
        scenario["start_time_min"],
        scenario["transfer_time_min"],
        limit=max(limit * 6, 20),
        max_extra_stops=8,
        max_paths=20000,
        allowed_modes=profile.allowed_modes,
    )
    ranked = sorted(candidates, key=lambda item: constraint_sort_key(item[0], item[2], profile))
    out = []
    for duration_min, route, steps in ranked[:limit]:
        out.append(
            ConstraintRoute(
                route=route,
                steps=steps,
                duration_min=duration_min,
                line_sequence=route_line_sequence(steps),
                line_change_count=route_line_change_count(steps),
                average_fullness=route_average_fullness(steps),
                near_capacity_count=route_near_capacity_count(steps),
                has_near_capacity=route_has_near_capacity(steps),
                delay_probability=route_delay_probability(steps),
                transfer_miss_probability=route_transfer_miss_probability(steps),
                mode_sequence=route_mode_sequence(steps),
                score=constraint_sort_key(duration_min, steps, profile),
                label=profile.label,
            )
        )
    return out


def route_constraint_gap(steps, duration_min, constraint_route):
    """Return scalar proposal gaps from the constraint-aware startup baseline."""
    if not constraint_route or duration_min is None:
        return {}
    near_capacity_count = route_near_capacity_count(steps)
    near_capacity_gap = near_capacity_count - constraint_route.near_capacity_count
    return {
        "duration_gap_min": duration_min - constraint_route.duration_min,
        "line_change_gap": route_line_change_count(steps) - constraint_route.line_change_count,
        "fullness_gap": near_capacity_gap,
        "near_capacity_gap": near_capacity_gap,
        "near_capacity_count": near_capacity_count,
        "has_near_capacity": near_capacity_count > 0,
        "delay_probability_gap": round(route_delay_probability(steps) - constraint_route.delay_probability, 4),
        "transfer_miss_probability_gap": round(
            route_transfer_miss_probability(steps) - constraint_route.transfer_miss_probability,
            4,
        ),
        "transfer_miss_probability": route_transfer_miss_probability(steps),
        "mode_sequence": route_mode_sequence(steps),
    }
