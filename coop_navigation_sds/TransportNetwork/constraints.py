"""Constraint-aware route baselines for comparing agent proposals."""
from dataclasses import dataclass
import math

from coop_navigation_sds.TransportNetwork.network import is_near_capacity
from coop_navigation_sds.Configuration.travel import (
    ACCEPTABLE_DURATION_RATIO,
    MAX_ROUTE_DELAY_PROBABILITY,
    MAX_TRANSFER_MISS_PROBABILITY,
    MIN_STAGE_SUBOPTIMAL_OPTIONS,
    REQUIRE_STAGE_SUBOPTIMAL_OPTIONS,
    DEFAULT_MAX_WALKING_MIN,
)
from coop_navigation_sds.TransportNetwork.routes import (
    candidate_time_routes,
    optimal_time_route,
    route_line_change_count,
    route_line_sequence,
    route_path_text_from_steps,
)


OBJECTIVE_VALID_ROUTE = "only_valid_route"
OBJECTIVE_SHORTEST_ROUTE = "shortest_valid_route"
OBJECTIVE_SHORTEST_WITH_CONSTRAINTS = "shortest_valid_route_with_constraints"
# Historical modes remain readable in stored results, but new runs expose one
# controlled task objective so objective choice cannot confound comparisons.
OBJECTIVE_MODES = (OBJECTIVE_SHORTEST_WITH_CONSTRAINTS,)
OBJECTIVE_MODE_LABELS = {
    OBJECTIVE_VALID_ROUTE: "Only valid route",
    OBJECTIVE_SHORTEST_ROUTE: "Shortest valid route",
    OBJECTIVE_SHORTEST_WITH_CONSTRAINTS: "Shortest valid route with 1-3 constraints",
}

CONSTRAINT_LABELS = {
    "transfers": "few line changes",
    "fullness": "not near capacity",
    "delay": "low delay risk",
    "transfer_miss": "safe transfer timing",
    "tickets": "available transport tickets",
    "walking": "acceptable walking time",
}

CONSTRAINT_KEYWORDS = {
    "transfers": ("few line changes", "fewer line changes", "as few line changes", "transfer count", "avoid changes", "avoid line changes", "switching"),
    "fullness": ("near-capacity", "capacity", "crowded", "packed", "full"),
    "delay": ("delay", "reliable", "risk"),
    "transfer_miss": ("transfer miss", "miss risk", "safe transfer", "safer transfer"),
    "tickets": ("ticket", "cannot take", "no bus", "no tram", "no metro"),
    "walking": ("walking", "walk", "on foot"),
}


def normalize_objective_mode(value):
    """Return the single controlled objective used by every new run."""
    return OBJECTIVE_SHORTEST_WITH_CONSTRAINTS


@dataclass(frozen=True)
class RouteConstraintProfile:
    """Normalized persona route preferences used for scoring candidate routes."""
    prefer_fast: bool = True
    prefer_fewer_changes: bool = False
    prefer_less_full: bool = False
    prefer_low_delay: bool = False
    max_transfer_miss_probability: float = MAX_TRANSFER_MISS_PROBABILITY
    max_delay_probability: float = MAX_ROUTE_DELAY_PROBABILITY
    ticket_modes: tuple[str, ...] = ("metro", "tram", "bus")
    max_walking_min: int = DEFAULT_MAX_WALKING_MIN

    @classmethod
    def from_persona(cls, persona, scenario=None):
        scenario = scenario or {}
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
        raw_ticket_modes = scenario.get("ticket_modes", preferences.get("ticket_modes", ("metro", "tram", "bus")))
        if isinstance(raw_ticket_modes, str):
            raw_ticket_modes = raw_ticket_modes.split(",")
        ticket_modes = tuple(
            mode
            for mode in (str(value).strip().lower() for value in raw_ticket_modes)
            if mode in {"metro", "tram", "bus"}
        )
        return cls(
            prefer_fast=prefer_fast or not (prefer_fewer_changes or prefer_less_full or prefer_low_delay),
            prefer_fewer_changes=prefer_fewer_changes,
            prefer_less_full=prefer_less_full,
            prefer_low_delay=prefer_low_delay,
            max_transfer_miss_probability=float(
                scenario.get(
                    "max_transfer_miss_probability",
                    preferences.get("max_transfer_miss_probability", MAX_TRANSFER_MISS_PROBABILITY),
                )
            ),
            max_delay_probability=float(
                scenario.get(
                    "max_delay_probability",
                    preferences.get("max_delay_probability", MAX_ROUTE_DELAY_PROBABILITY),
                )
            ),
            ticket_modes=ticket_modes or ("metro", "tram", "bus"),
            max_walking_min=max(
                0,
                int(scenario.get("max_walking_min", preferences.get("max_walking_min", DEFAULT_MAX_WALKING_MIN))),
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
        parts.append(f"tickets for {' and '.join(self.ticket_modes)}")
        parts.append(f"walk at most {self.max_walking_min} minutes")
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
    score: tuple
    label: str
    max_delay_probability: float = MAX_ROUTE_DELAY_PROBABILITY
    max_transfer_miss_probability: float = MAX_TRANSFER_MISS_PROBABILITY


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
        "risk_unviable": not (delay_allowed and transfer_allowed),
    }


def route_allowed_modes(scenario, persona=None):
    """Return a hashable route-access profile for tickets and walking."""
    profile = RouteConstraintProfile.from_persona(persona or {}, scenario or {})
    return (*profile.ticket_modes, "walking", f"walking_max:{profile.max_walking_min}")


def dialog_route_allowed_modes(scenario, persona=None, stated_keys=()):
    """Return access rules Agent A has revealed at the current dialog stage."""
    stated = set(stated_keys or ())
    profile = RouteConstraintProfile.from_persona(persona or {}, scenario or {})
    public_modes = profile.ticket_modes if "tickets" in stated else ("metro", "tram", "bus")
    modes = [*public_modes, "walking"]
    if "walking" in stated:
        modes.append(f"walking_max:{profile.max_walking_min}")
    return tuple(modes)


def route_walking_minutes(steps):
    """Return cumulative walking travel time."""
    return sum(step.get("travel", 0) for step in steps if step.get("mode") == "walking")


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


def available_agent_a_constraints(persona, scenario):
    """Return progressive Agent A constraints in the order they should be revealed."""
    preferences = (persona or {}).get("preferences", {})
    priority = str(preferences.get("priority", "")).lower()
    switching = str(preferences.get("switching", "")).lower()
    fullness = str(preferences.get("fullness", "")).lower()
    reliability = str(preferences.get("reliability", "")).lower()
    constraints = []

    def add(key):
        if key not in constraints:
            constraints.append(key)

    if any(term in switching for term in ("fewer", "avoid", "unnecessary", "only for meaningful")):
        add("transfers")
    if any(term in fullness for term in ("less crowded", "dislikes", "crowded", "full", "packed")) and "does not mind" not in fullness:
        add("fullness")
    if any(term in f"{priority} {reliability}" for term in ("delay", "reliable", "on time", "low risk")):
        add("delay")
    if preferences.get("max_transfer_miss_probability") is not None or (scenario or {}).get("max_transfer_miss_probability") is not None:
        add("transfer_miss")
    add("tickets")
    add("walking")
    if "transfers" not in constraints:
        constraints.append("transfers")
    if len(constraints) < 2 and "fullness" not in constraints:
        constraints.append("fullness")
    if len(constraints) < 2 and "delay" not in constraints:
        constraints.append("delay")
    return constraints


def stated_constraint_keys(conversation):
    """Infer which constraints Agent A has explicitly stated so far."""
    keys = []
    for speaker, text in conversation:
        if speaker != "Agent A":
            continue
        lower = text.lower()
        for key, keywords in CONSTRAINT_KEYWORDS.items():
            if key not in keys and any(keyword in lower for keyword in keywords):
                keys.append(key)
    return keys


def route_constraint_status(steps, persona, scenario, stated_keys, transfer_tolerance=1, constraint_route=None):
    """Evaluate a route against only the constraints Agent A has stated."""
    profile = RouteConstraintProfile.from_persona(persona or {}, scenario or {})
    baseline_changes = constraint_route.line_change_count if constraint_route else 0
    max_changes = baseline_changes + int(transfer_tolerance)
    statuses = {}
    for key in stated_keys:
        if key == "transfers":
            changes = route_line_change_count(steps)
            statuses[key] = {"satisfied": changes <= max_changes, "actual": changes, "limit": max_changes}
        elif key == "fullness":
            count = route_near_capacity_count(steps)
            statuses[key] = {"satisfied": count == 0, "actual": count, "limit": 0}
        elif key == "delay":
            risk = route_delay_probability(steps)
            statuses[key] = {
                "satisfied": probability_class_allowed(risk, profile.max_delay_probability),
                "actual": probability_class(risk),
                "limit": probability_class(profile.max_delay_probability),
            }
        elif key == "transfer_miss":
            risk = route_transfer_miss_probability(steps)
            statuses[key] = {
                "satisfied": probability_class_allowed(risk, profile.max_transfer_miss_probability),
                "actual": probability_class(risk),
                "limit": probability_class(profile.max_transfer_miss_probability),
            }
        elif key == "tickets":
            used = sorted({step.get("mode") for step in steps if step.get("mode") != "walking"})
            invalid = [mode for mode in used if mode not in profile.ticket_modes]
            statuses[key] = {
                "satisfied": not invalid,
                "actual": used,
                "limit": list(profile.ticket_modes),
            }
        elif key == "walking":
            minutes = route_walking_minutes(steps)
            statuses[key] = {
                "satisfied": minutes <= profile.max_walking_min,
                "actual": minutes,
                "limit": profile.max_walking_min,
            }
    return statuses


def unsatisfied_constraint_keys(statuses):
    """Return stated constraint keys whose status is not satisfied."""
    return [key for key, status in statuses.items() if not status.get("satisfied", False)]


def optimal_route_duration_min(scenario, persona=None):
    """Return the fastest route before private constraints are revealed."""
    scenario = scenario or {}
    arrival, _steps = optimal_time_route(
        scenario["start_station"],
        scenario["destination_station"],
        scenario["start_time_min"],
        scenario["transfer_time_min"],
        allowed_modes=dialog_route_allowed_modes(scenario, persona, ()),
    )
    return arrival - scenario["start_time_min"] if arrival is not None else None


def acceptable_duration_ratio(scenario):
    """Return the configured strict duration ratio for stage-one acceptance."""
    value = (scenario or {}).get("acceptable_duration_ratio", ACCEPTABLE_DURATION_RATIO)
    return max(1.0, float(value or ACCEPTABLE_DURATION_RATIO))


def acceptable_duration_limit(scenario, persona=None, constraint_route=None, slack_min=None):
    """Return the maximum whole-minute duration under the configured optimal-route ratio."""
    scenario = scenario or {}
    configured = scenario.get("acceptable_duration_min", scenario.get("acceptable_time_frame_min"))
    if configured is not None:
        return int(configured)

    base_duration = optimal_route_duration_min(scenario, persona)
    if base_duration is None:
        return None

    configured_slack = scenario.get("acceptable_duration_slack_min", slack_min)
    if configured_slack is not None:
        return int(math.floor(base_duration + int(configured_slack)))
    strict_limit = float(base_duration) * acceptable_duration_ratio(scenario)
    return int(math.ceil(strict_limit) - 1 if strict_limit.is_integer() else math.floor(strict_limit))


def stage_suboptimal_options_required(scenario):
    """Return whether each conversation stage must have suboptimal viable alternatives."""
    if "require_stage_suboptimal_options" in (scenario or {}):
        return bool(scenario.get("require_stage_suboptimal_options"))
    return REQUIRE_STAGE_SUBOPTIMAL_OPTIONS


def minimum_stage_suboptimal_options(scenario):
    """Return the configured number of non-best viable alternatives required per stage."""
    value = (scenario or {}).get("min_stage_suboptimal_options", MIN_STAGE_SUBOPTIMAL_OPTIONS)
    return max(0, int(value or 0))


def stage_route_options(scenario, persona, stated_keys=(), transfer_tolerance=1, limit=80):
    """Return viable route options for one staged goal state."""
    duration_limit = acceptable_duration_limit(scenario, persona)
    constraint_route = optimal_constraint_route(scenario, persona)
    profile = RouteConstraintProfile.from_persona(persona or {}, scenario or {})
    candidates = candidate_time_routes(
        scenario["start_station"],
        scenario["destination_station"],
        scenario["start_time_min"],
        scenario["transfer_time_min"],
        limit=max(limit, 20),
        max_extra_stops=8,
        max_paths=20000,
        allowed_modes=dialog_route_allowed_modes(scenario, persona, stated_keys),
    )
    viable = []
    for duration_min, route, steps in candidates:
        if duration_limit is not None and duration_min > duration_limit:
            continue
        statuses = route_constraint_status(
            steps,
            persona,
            scenario,
            stated_keys,
            transfer_tolerance=transfer_tolerance,
            constraint_route=constraint_route,
        )
        if unsatisfied_constraint_keys(statuses):
            continue
        viable.append(
            {
                "duration_min": duration_min,
                "route": route,
                "steps": steps,
                "line_sequence": route_line_sequence(steps),
                "line_change_count": route_line_change_count(steps),
                "near_capacity_count": route_near_capacity_count(steps),
                "delay_risk_class": probability_class(route_delay_probability(steps)),
                "transfer_miss_risk_class": probability_class(route_transfer_miss_probability(steps)),
                "constraint_status": statuses,
                "score": stage_constraint_sort_key(duration_min, steps, stated_keys),
            }
        )
    viable.sort(key=lambda option: option["score"])
    return viable


def stage_constraint_sort_key(duration_min, steps, stated_keys=()):
    """Rank revealed constraints before time, newest constraint first."""
    key = []
    for constraint in reversed(tuple(stated_keys or ())):
        if constraint == "transfers":
            key.append(route_line_change_count(steps))
        elif constraint == "fullness":
            key.append(route_near_capacity_count(steps))
        elif constraint == "delay":
            key.append(route_delay_probability(steps))
        elif constraint == "transfer_miss":
            key.append(route_transfer_miss_probability(steps))
        elif constraint == "walking":
            key.append(route_walking_minutes(steps))
    key.extend((duration_min, route_line_change_count(steps), len(steps)))
    return tuple(key)


def layered_optimal_routes(scenario, persona, transfer_tolerance=1, max_constraints=3):
    """Calculate validity, time, and distinct progressive constraint optima."""
    allowed_modes = dialog_route_allowed_modes(scenario, persona, ())
    candidates = candidate_time_routes(
        scenario["start_station"],
        scenario["destination_station"],
        scenario["start_time_min"],
        scenario["transfer_time_min"],
        limit=80,
        max_extra_stops=8,
        max_paths=20000,
        allowed_modes=allowed_modes,
    )
    layers = []
    previous_signature = None
    constraint_signatures = set()

    def path_signature(steps):
        return tuple(
            (step["from"], step["to"], step.get("line"), step.get("mode"))
            for step in steps
        )

    def append_layer(key, label, stated_keys, option):
        nonlocal previous_signature
        duration_min, route, steps = option if option else (None, [], [])
        signature = path_signature(steps)
        if signature:
            previous_signature = signature
        layers.append({
            "layer": key,
            "label": label,
            "stated_constraints": list(stated_keys),
            "available": bool(route),
            "route": list(route),
            "path_text": route_path_text_from_steps(steps) if steps else None,
            "duration_min": duration_min,
            "line_sequence": route_line_sequence(steps),
            "line_change_count": route_line_change_count(steps) if steps else None,
            "steps": steps,
        })

    validity = min(candidates, key=lambda item: (len(item[1]), item[0], item[1])) if candidates else None
    fastest = min(candidates, key=lambda item: (item[0], len(item[1]), item[1])) if candidates else None
    append_layer("validity", "Valid connected route", (), validity)
    append_layer("time", "Fastest valid route", (), fastest)
    baseline_signature = previous_signature

    constraint_order = available_agent_a_constraints(persona or {}, scenario or {})[:max_constraints]
    for index, constraint in enumerate(constraint_order, start=1):
        stated_keys = tuple(constraint_order[:index])
        options = stage_route_options(
            scenario,
            persona,
            stated_keys=stated_keys,
            transfer_tolerance=transfer_tolerance,
        )
        selected = None
        eligible = [
            option for option in options
            if path_signature(option["steps"]) != previous_signature
            and path_signature(option["steps"]) not in constraint_signatures
        ]
        best = next(
            (option for option in eligible if path_signature(option["steps"]) != baseline_signature),
            eligible[0] if eligible else None,
        )
        if best is not None:
            selected = (best["duration_min"], best["route"], best["steps"])
        append_layer(
            f"constraint_{index}",
            f"Constraint {index}: {CONSTRAINT_LABELS.get(constraint, constraint)}",
            stated_keys,
            selected,
        )
        if best is not None:
            constraint_signatures.add(path_signature(best["steps"]))
    return layers


def stage_viability_report(scenario, persona, transfer_tolerance=1, max_constraints=2):
    """Return whether each staged conversation goal has required alternative routes."""
    required_count = minimum_stage_suboptimal_options(scenario)
    require_options = stage_suboptimal_options_required(scenario)
    constraint_order = available_agent_a_constraints(persona or {}, scenario or {})[:max_constraints]
    layered = layered_optimal_routes(
        scenario,
        persona,
        transfer_tolerance=transfer_tolerance,
        max_constraints=max_constraints,
    )
    optimum_by_stage = [layered[1], *layered[2:]]
    require_route_changes = bool((scenario or {}).get("require_constraint_route_changes", True))
    stages = []
    for stage_index in range(max_constraints + 1):
        stated_keys = constraint_order[:stage_index]
        options = stage_route_options(
            scenario,
            persona,
            stated_keys=stated_keys,
            transfer_tolerance=transfer_tolerance,
        )
        best_route = options[0]["route"] if options else []
        suboptimal = [
            option for option in options
            if tuple(option["route"]) != tuple(best_route)
        ]
        optimum = optimum_by_stage[stage_index] if stage_index < len(optimum_by_stage) else None
        previous_optimum = optimum_by_stage[stage_index - 1] if stage_index > 0 else None
        route_changed = (
            True if stage_index == 0 else bool(
                optimum
                and previous_optimum
                and optimum.get("available")
                and previous_optimum.get("available")
                and optimum.get("path_text") != previous_optimum.get("path_text")
            )
        )
        alternatives_satisfied = (not require_options) or len(suboptimal) >= required_count
        route_change_satisfied = (not require_route_changes) or route_changed
        stages.append({
            "stage": stage_index + 1,
            "stated_constraints": stated_keys,
            "viable_option_count": len(options),
            "suboptimal_option_count": len(suboptimal),
            "required_suboptimal_option_count": required_count,
            "require_suboptimal_options": require_options,
            "requirement_satisfied": alternatives_satisfied and route_change_satisfied,
            "constraint_changes_optimal_route": route_changed,
            "prior_optimal_path": previous_optimum.get("path_text") if previous_optimum else None,
            "optimal_path": optimum.get("path_text") if optimum else None,
            "best_route": best_route,
            "best_duration_min": options[0]["duration_min"] if options else None,
            "suboptimal_options": [
                {
                    "route": option["route"],
                    "duration_min": option["duration_min"],
                    "line_sequence": option["line_sequence"],
                    "line_change_count": option["line_change_count"],
                    "near_capacity_count": option["near_capacity_count"],
                    "delay_risk_class": option["delay_risk_class"],
                    "transfer_miss_risk_class": option["transfer_miss_risk_class"],
                }
                for option in suboptimal[:required_count + 3]
            ],
        })
    return {
        "acceptable_duration_ratio": acceptable_duration_ratio(scenario),
        "acceptable_duration_limit_min": acceptable_duration_limit(scenario, persona),
        "optimal_duration_min": optimal_route_duration_min(scenario, persona),
        "constraint_order": constraint_order,
        "require_suboptimal_options": require_options,
        "require_constraint_route_changes": require_route_changes,
        "required_suboptimal_option_count": required_count,
        "all_stage_requirements_satisfied": all(stage["requirement_satisfied"] for stage in stages),
        "stages": stages,
    }


def constraint_request_text(key, persona=None, scenario=None):
    """Return one natural Agent A utterance fragment for a constraint key."""
    if key == "transfers":
        return "with as few line changes as reasonable"
    if key == "fullness":
        return "not near capacity"
    if key == "delay":
        return "with lower delay risk"
    if key == "transfer_miss":
        return "with safer transfer timing"
    if key == "tickets":
        profile = RouteConstraintProfile.from_persona(persona or {}, scenario or {})
        missing = sorted({"metro", "tram", "bus"} - set(profile.ticket_modes))
        return f"using only my {' and '.join(profile.ticket_modes)} tickets; I cannot take {missing[0]}" if missing else "using only my available tickets"
    if key == "walking":
        profile = RouteConstraintProfile.from_persona(persona or {}, scenario or {})
        return f"with no more than {profile.max_walking_min} minutes of walking"
    return CONSTRAINT_LABELS.get(key, key)


def constraint_sort_key(duration_min, steps, profile, objective_mode=OBJECTIVE_SHORTEST_WITH_CONSTRAINTS):
    line_changes = route_line_change_count(steps)
    average_fullness = route_average_fullness(steps)
    near_capacity_count = route_near_capacity_count(steps)
    delay_probability = route_delay_probability(steps)
    transfer_miss_probability = route_transfer_miss_probability(steps)
    objective_mode = normalize_objective_mode(objective_mode)
    if objective_mode == OBJECTIVE_VALID_ROUTE:
        return (
            len(steps),
            line_changes,
            duration_min,
            average_fullness,
        )
    if objective_mode == OBJECTIVE_SHORTEST_ROUTE:
        return (
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
        allowed_modes=route_allowed_modes(scenario, persona),
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
        score=constraint_sort_key(duration_min, steps, profile, objective_mode),
        label=profile.label,
        max_delay_probability=profile.max_delay_probability,
        max_transfer_miss_probability=profile.max_transfer_miss_probability,
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
        allowed_modes=route_allowed_modes(scenario, persona),
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
                score=constraint_sort_key(duration_min, steps, profile, objective_mode),
                label=profile.label,
                max_delay_probability=profile.max_delay_probability,
                max_transfer_miss_probability=profile.max_transfer_miss_probability,
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
        **viability,
    }
