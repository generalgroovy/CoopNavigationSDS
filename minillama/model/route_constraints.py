"""Constraint-aware route baselines for comparing agent proposals."""
from dataclasses import dataclass
import math

from minillama.model.metro_data import is_near_capacity
from minillama.model.metro_data import ADJACENCY, STATION_POS
from minillama.model.config import DEFAULT_ALLOWED_MODES, MAX_ROUTE_DELAY_PROBABILITY, MAX_TRANSFER_MISS_PROBABILITY
from minillama.model.route_planner import (
    candidate_time_routes,
    line_mode,
    normalize_allowed_modes,
    route_line_change_count,
    route_line_sequence,
)


OBJECTIVE_VALID_ROUTE = "only_valid_route"
OBJECTIVE_SHORTEST_ROUTE = "shortest_valid_route"
OBJECTIVE_SHORTEST_WITH_CONSTRAINTS = "shortest_valid_route_with_constraints"
OBJECTIVE_MODES = (
    OBJECTIVE_VALID_ROUTE,
    OBJECTIVE_SHORTEST_ROUTE,
    OBJECTIVE_SHORTEST_WITH_CONSTRAINTS,
)
OBJECTIVE_MODE_LABELS = {
    OBJECTIVE_VALID_ROUTE: "Only valid route",
    OBJECTIVE_SHORTEST_ROUTE: "Shortest valid route",
    OBJECTIVE_SHORTEST_WITH_CONSTRAINTS: "Shortest valid route with 1-3 constraints",
}


def normalize_objective_mode(value):
    """Return a supported Agent A objective mode."""
    normalized = str(value or OBJECTIVE_SHORTEST_WITH_CONSTRAINTS).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "valid": OBJECTIVE_VALID_ROUTE,
        "only_valid": OBJECTIVE_VALID_ROUTE,
        "only_valid_route": OBJECTIVE_VALID_ROUTE,
        "shortest": OBJECTIVE_SHORTEST_ROUTE,
        "shortest_route": OBJECTIVE_SHORTEST_ROUTE,
        "shortest_valid": OBJECTIVE_SHORTEST_ROUTE,
        "shortest_valid_route": OBJECTIVE_SHORTEST_ROUTE,
        "constraints": OBJECTIVE_SHORTEST_WITH_CONSTRAINTS,
        "shortest_with_constraints": OBJECTIVE_SHORTEST_WITH_CONSTRAINTS,
        "shortest_valid_route_with_constraints": OBJECTIVE_SHORTEST_WITH_CONSTRAINTS,
    }
    return aliases.get(normalized, OBJECTIVE_SHORTEST_WITH_CONSTRAINTS)


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
    max_walking_min: int | None = None

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
            max_walking_min=preferences.get(
                "max_walking_min",
                (scenario or {}).get("max_walking_min"),
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
        if self.max_walking_min is not None:
            parts.append(f"walk up to {self.max_walking_min} minutes")
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
    max_delay_probability: float = MAX_ROUTE_DELAY_PROBABILITY
    max_transfer_miss_probability: float = MAX_TRANSFER_MISS_PROBABILITY
    max_walking_min: int | None = None


def probability_class(value):
    """Map an internal probability to a spoken risk class."""
    if value is None:
        return "unknown"
    if value < 0.25:
        return "low"
    if value < 0.45:
        return "medium"
    return "high"


def probability_class_allowed(value, threshold):
    """Return whether a probability falls within the configured risk class."""
    order = {"unknown": 3, "low": 0, "medium": 1, "high": 2}
    return order[probability_class(value)] <= order[probability_class(threshold)]


def walking_minutes_between(station_a, station_b):
    """Estimate pedestrian time between stations from network coordinates."""
    if station_a not in STATION_POS or station_b not in STATION_POS:
        return None
    ax, ay = STATION_POS[station_a]
    bx, by = STATION_POS[station_b]
    distance = math.hypot(ax - bx, ay - by)
    return max(1, int(math.ceil(distance / 35.0)))


def direct_public_transport_minutes_between(station_a, station_b):
    """Return direct public-transport minutes if the stations share a segment."""
    values = [travel for nxt, _line, travel in ADJACENCY.get(station_a, []) if nxt == station_b]
    return min(values) if values else None


def nearby_walking_links(max_minutes=12, limit=12):
    """Return close station pairs with walking minutes relative to direct transit time."""
    links = []
    stations = sorted(STATION_POS)
    for index, station_a in enumerate(stations):
        for station_b in stations[index + 1:]:
            walk = walking_minutes_between(station_a, station_b)
            if walk is None or walk > max_minutes:
                continue
            transit = direct_public_transport_minutes_between(station_a, station_b)
            links.append((walk, station_a, station_b, transit))
    links.sort(key=lambda item: (item[0], item[1], item[2]))
    return links[:limit]


def route_risk_viability(steps, profile):
    """Return class-based route viability for delay and transfer risk."""
    delay = route_delay_probability(steps)
    transfer = route_transfer_miss_probability(steps)
    delay_allowed = probability_class_allowed(delay, profile.max_delay_probability)
    transfer_allowed = probability_class_allowed(transfer, profile.max_transfer_miss_probability)
    return {
        "delay_risk_class": probability_class(delay),
        "transfer_miss_risk_class": probability_class(transfer),
        "max_delay_risk_class": probability_class(profile.max_delay_probability),
        "max_transfer_miss_risk_class": probability_class(profile.max_transfer_miss_probability),
        "delay_risk_unviable": not delay_allowed,
        "transfer_miss_risk_unviable": not transfer_allowed,
        "walking_unviable": not route_within_walking_limit(steps, profile),
        "risk_unviable": not (delay_allowed and transfer_allowed and route_within_walking_limit(steps, profile)),
    }


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


def route_walking_minutes(steps):
    """Return walking minutes used by a route."""
    return sum(step.get("travel", 0) for step in steps if step.get("mode") == "walking")


def route_within_walking_limit(steps, profile):
    """Return whether route walking stays inside persona limit."""
    return profile.max_walking_min is None or route_walking_minutes(steps) <= profile.max_walking_min


def constraint_sort_key(duration_min, steps, profile, objective_mode=OBJECTIVE_SHORTEST_WITH_CONSTRAINTS):
    line_changes = route_line_change_count(steps)
    average_fullness = route_average_fullness(steps)
    near_capacity_count = route_near_capacity_count(steps)
    delay_probability = route_delay_probability(steps)
    transfer_miss_probability = route_transfer_miss_probability(steps)
    objective_mode = normalize_objective_mode(objective_mode)
    if objective_mode == OBJECTIVE_VALID_ROUTE:
        return (
            int(not route_within_walking_limit(steps, profile)),
            len(steps),
            line_changes,
            duration_min,
            average_fullness,
        )
    if objective_mode == OBJECTIVE_SHORTEST_ROUTE:
        return (
            int(not route_within_walking_limit(steps, profile)),
            duration_min,
            line_changes,
            len(steps),
            average_fullness,
        )
    key = []
    key.append(duration_min if profile.prefer_fast else round(duration_min * 0.25))
    if profile.prefer_fewer_changes:
        key.append(line_changes)
    if profile.prefer_less_full:
        key.append(near_capacity_count)
    if profile.prefer_low_delay:
        key.append(delay_probability)
    key.extend([
        int(not route_within_walking_limit(steps, profile)),
        int(not probability_class_allowed(delay_probability, profile.max_delay_probability)),
        int(not probability_class_allowed(transfer_miss_probability, profile.max_transfer_miss_probability)),
        duration_min,
        line_changes,
        near_capacity_count,
        average_fullness,
        transfer_miss_probability,
        delay_probability,
    ])
    return tuple(key)


def optimal_constraint_route(scenario, persona, limit=50, objective_mode=None):
    """Compute the best startup baseline route under persona constraints."""
    profile = RouteConstraintProfile.from_persona(persona, scenario)
    objective_mode = normalize_objective_mode(objective_mode or scenario.get("agent_a_objective_mode"))
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
        key=lambda item: constraint_sort_key(item[0], item[2], profile, objective_mode),
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
        score=constraint_sort_key(duration_min, steps, profile, objective_mode),
        label=profile.label,
        max_delay_probability=profile.max_delay_probability,
        max_transfer_miss_probability=profile.max_transfer_miss_probability,
        max_walking_min=profile.max_walking_min,
    )


def ranked_constraint_routes(scenario, persona, limit=6, objective_mode=None):
    """Return valid routes sorted by the persona's scientific comparison profile."""
    profile = RouteConstraintProfile.from_persona(persona, scenario)
    objective_mode = normalize_objective_mode(objective_mode or scenario.get("agent_a_objective_mode"))
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
    ranked = sorted(candidates, key=lambda item: constraint_sort_key(item[0], item[2], profile, objective_mode))
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
                score=constraint_sort_key(duration_min, steps, profile, objective_mode),
                label=profile.label,
                max_delay_probability=profile.max_delay_probability,
                max_transfer_miss_probability=profile.max_transfer_miss_probability,
                max_walking_min=profile.max_walking_min,
            )
        )
    return out


def route_constraint_gap(steps, duration_min, constraint_route):
    """Return scalar proposal gaps from the constraint-aware startup baseline."""
    if not constraint_route or duration_min is None:
        return {}
    near_capacity_count = route_near_capacity_count(steps)
    near_capacity_gap = near_capacity_count - constraint_route.near_capacity_count
    profile = RouteConstraintProfile(
        max_delay_probability=constraint_route.max_delay_probability,
        max_transfer_miss_probability=constraint_route.max_transfer_miss_probability,
    )
    viability = route_risk_viability(steps, profile)
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
        **viability,
    }
