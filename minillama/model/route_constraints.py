"""Constraint-aware route baselines for comparing agent proposals."""
from dataclasses import dataclass

from minillama.model.route_planner import (
    candidate_time_routes,
    route_line_change_count,
    route_line_sequence,
)


@dataclass(frozen=True)
class RouteConstraintProfile:
    """Normalized persona route preferences used for scoring candidate routes."""
    prefer_fast: bool = True
    prefer_fewer_changes: bool = False
    prefer_less_full: bool = False

    @classmethod
    def from_persona(cls, persona):
        preferences = persona.get("preferences", {})
        priority = preferences.get("priority", "").lower()
        switching = preferences.get("switching", "").lower()
        fullness = preferences.get("fullness", "").lower()
        prefer_fast = any(term in priority for term in ("fast", "quick", "time"))
        prefer_fewer_changes = any(
            term in switching
            for term in ("fewer", "avoid", "avoiding", "unnecessary", "only for meaningful")
        )
        prefer_less_full = any(
            term in fullness
            for term in ("less crowded", "dislikes", "packed", "very full")
        ) and "does not mind" not in fullness
        return cls(
            prefer_fast=prefer_fast or not (prefer_fewer_changes or prefer_less_full),
            prefer_fewer_changes=prefer_fewer_changes,
            prefer_less_full=prefer_less_full,
        )

    @property
    def label(self):
        parts = []
        if self.prefer_fast:
            parts.append("fastest")
        if self.prefer_fewer_changes:
            parts.append("fewer changes")
        if self.prefer_less_full:
            parts.append("less full")
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
    score: tuple
    label: str


def route_average_fullness(steps):
    values = [step.get("fullness", 0) for step in steps]
    return round(sum(values) / len(values), 2) if values else 0.0


def constraint_sort_key(duration_min, steps, profile):
    line_changes = route_line_change_count(steps)
    average_fullness = route_average_fullness(steps)
    key = []
    key.append(duration_min if profile.prefer_fast else round(duration_min * 0.25))
    if profile.prefer_fewer_changes:
        key.append(line_changes)
    if profile.prefer_less_full:
        key.append(average_fullness)
    key.extend([duration_min, line_changes, average_fullness])
    return tuple(key)


def optimal_constraint_route(scenario, persona, limit=50):
    """Compute the best startup baseline route under persona constraints."""
    profile = RouteConstraintProfile.from_persona(persona)
    candidates = candidate_time_routes(
        scenario["start_station"],
        scenario["destination_station"],
        scenario["start_time_min"],
        scenario["transfer_time_min"],
        limit=limit,
        max_extra_stops=8,
        max_paths=20000,
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
        score=constraint_sort_key(duration_min, steps, profile),
        label=profile.label,
    )


def route_constraint_gap(steps, duration_min, constraint_route):
    """Return scalar proposal gaps from the constraint-aware startup baseline."""
    if not constraint_route or duration_min is None:
        return {}
    return {
        "duration_gap_min": duration_min - constraint_route.duration_min,
        "line_change_gap": route_line_change_count(steps) - constraint_route.line_change_count,
        "fullness_gap": round(route_average_fullness(steps) - constraint_route.average_fullness, 2),
    }
