"""Agent A configuration and prompt constants."""
import os

HISTORY_MESSAGES = 10
LLM_AGENT_A = os.environ.get(
    "COOP_NAVIGATION_SDS_LLM_AGENT_A",
    os.environ.get("MINILLAMA_LLM_AGENT_A", "0"),
).lower() in {"1", "true", "on", "yes"}
DEFAULT_PERSONA = "focused_commuter"

PERSONAS = {
    "focused_commuter": {
        "name": "Focused commuter",
        "description": "Direct, time-conscious, practical.",
        "preferences": {
            "switching": "accepts line changes for a faster route",
            "fullness": "does not mind fuller trains",
            "priority": "fastest route first",
        },
    },
    "distracted_multitasker": {
        "name": "Distracted multitasker",
        "description": "Sometimes loses track and asks for repetition.",
        "preferences": {
            "switching": "prefers fewer line changes",
            "fullness": "prefers less crowded trains",
            "priority": "simple route first",
        },
    },
    "verbose_planner": {
        "name": "Verbose planner",
        "description": "Compares alternatives before deciding.",
        "preferences": {
            "switching": "accepts transfers if justified",
            "fullness": "considers fullness as a secondary factor",
            "priority": "balanced comparison of time, transfers, and fullness",
        },
    },
    "hesitant_speaker": {
        "name": "Hesitant speaker",
        "description": "Uncertain and often asks for confirmation.",
        "preferences": {
            "switching": "prefers avoiding line changes",
            "fullness": "prefers less crowded trains",
            "priority": "confidence and simplicity over small time savings",
        },
    },
    "adversarial_tester": {
        "name": "Adversarial tester",
        "description": "Challenges route suggestions.",
        "preferences": {
            "switching": "questions unnecessary line changes",
            "fullness": "challenges crowded-train choices",
            "priority": "prove the best tradeoff, not just the first route",
        },
    },
    "non_native_speaker": {
        "name": "Non-native speaker",
        "description": "Uses simple English and wants station names repeated.",
        "preferences": {
            "switching": "prefers fewer changes",
            "fullness": "does not mind fullness if the route is clear",
            "priority": "clear and simple route",
        },
    },
    "frustrated_user": {
        "name": "Frustrated user",
        "description": "Impatient and wants concise directions quickly.",
        "preferences": {
            "switching": "accepts line changes only for meaningful savings",
            "fullness": "dislikes very full trains",
            "priority": "fast answer with obvious tradeoffs",
        },
    },
    "crowd_averse_rider": {
        "name": "Crowd-averse rider",
        "description": "Calm, direct, and willing to spend a few extra minutes to avoid packed trains.",
        "preferences": {
            "switching": "accepts one extra change if it avoids a packed train",
            "fullness": "strongly prefers less crowded trains",
            "priority": "balance travel time and lower fullness",
            "reliability": "avoid very full trains that increase delay risk",
        },
    },
    "delay_sensitive_traveler": {
        "name": "Delay-sensitive traveler",
        "description": "Needs a reliable route and asks for risk tradeoffs.",
        "preferences": {
            "switching": "accepts transfers when they reduce delay risk",
            "fullness": "fullness matters if it increases delay probability",
            "priority": "low delay risk and on-time arrival",
            "reliability": "prefer lower delay probability",
            "max_transfer_miss_probability": 0.20,
            "max_delay_probability": 0.32,
        },
    },
    "accessibility_rider": {
        "name": "Accessibility-focused rider",
        "description": "Prefers a route that is easy to follow and avoids unnecessary changes.",
        "preferences": {
            "switching": "avoid unnecessary line changes",
            "fullness": "prefers enough space to board comfortably",
            "priority": "simple route with low transfer burden",
            "reliability": "prefer predictable services",
            "max_transfer_miss_probability": 0.18,
        },
    },
    "multi_stop_errand_runner": {
        "name": "Multi-stop errand runner",
        "description": "May need to visit more than one destination and asks for an extensible route.",
        "preferences": {
            "switching": "accepts changes if the route extends cleanly to later stops",
            "fullness": "prefers moderate fullness",
            "priority": "clear route that can expand to multiple destinations",
            "reliability": "avoid high delay risk when connecting between stops",
        },
    },
    "budget_simplifier": {
        "name": "Budget-conscious traveler",
        "description": "Accepts practical routes but wants unnecessary changes avoided.",
        "preferences": {
            "switching": "accepts changes only when they clearly improve the route",
            "fullness": "fullness is secondary to route simplicity",
            "priority": "valid route with few unnecessary changes",
        },
    },
    "low_effort_traveler": {
        "name": "Low-effort traveler",
        "description": "Wants a simple route and avoids unnecessary changes.",
        "preferences": {
            "switching": "prefers fewer changes",
            "fullness": "prefers space but can tolerate busy services",
            "priority": "simple route while keeping travel time reasonable",
        },
    },
    "transfer_confident_traveler": {
        "name": "Transfer-confident traveler",
        "description": "Comfortable changing services when it produces a materially faster route.",
        "preferences": {
            "switching": "readily accepts transfers for meaningful time savings",
            "fullness": "does not mind near-capacity services",
            "priority": "fastest route even with additional changes",
        },
    },
    "risk_averse_novice": {
        "name": "Risk-averse novice",
        "description": "Needs simple, reliable instructions and rejects tight or delay-prone connections.",
        "preferences": {
            "switching": "avoids changes unless the connection is generous",
            "fullness": "prefers services that are not near capacity",
            "priority": "low-risk route that is easy to follow",
            "reliability": "reject high delay and missed-transfer risk",
            "max_transfer_miss_probability": 0.16,
            "max_delay_probability": 0.24,
        },
    },
}

# Route-access profiles are explicit experiment factors for every caller. The
# startup configuration can override either value without mutating the catalog.
_TICKET_PROFILES = (("metro", "tram"), ("tram", "bus"), ("metro", "bus"))
for _index, (_persona_key, _persona) in enumerate(PERSONAS.items()):
    _preferences = _persona["preferences"]
    _preferences.setdefault("ticket_modes", _TICKET_PROFILES[_index % len(_TICKET_PROFILES)])
    _preferences.setdefault(
        "max_walking_min",
        5 if _persona_key in {"accessibility_rider", "low_effort_traveler"} else 10,
    )

AGENT_RULES = (
    "Speak like a live phone call in 1-3 natural sentences. "
    "React to the latest message while retaining the established conversation state. "
    "Repair only the unclear part, then continue from the last understood route. "
    "Enunciate station, line, and time words clearly unless the configured audio condition says otherwise. "
    "No hidden reasoning, code, JSON, tables, bullets, or empty replies."
)

ROUTE_TASK = (
    "Primary goal: a valid connected route from start to destination using metro, tram, bus, or walking segments, waits, and transfer times. "
    "Secondary goal: satisfy preferences such as avoiding near-capacity trains and fewer changes. "
    "Delay risk is a secondary reliability constraint and is spoken only as low, medium, or high. "
    "Best time means riding + waiting + transfer time, with transfer time only when the line changes. "
    "Short transfer buffers can add low, medium, or high missed-connection risk. "
    "Respect the caller's two available transport tickets and cumulative walking limit after those constraints are stated. "
    "Compare alternatives only when they are valid and meaningfully affect time or constraints. "
    "All listed segments work both ways. "
    "Say stations in travel order and keep line changes explicit."
)
