"""Persona model definitions for Agent A behavior and standardized experiment conditions.
"""
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
}


DEFAULT_PERSONA = "focused_commuter"


def get_persona(persona_key: str):
    """Get persona function for this module's MVC responsibility.
    
    Args:
        persona_key: Input value used by `get_persona`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return PERSONAS.get(persona_key, PERSONAS[DEFAULT_PERSONA])


def preference_text(persona: dict):
    """Format a persona preference profile for prompts."""
    preferences = persona.get("preferences", {})
    if not preferences:
        return "Preferences: fastest connected route."
    return (
        "Preferences: "
        f"{preferences.get('priority', 'fastest route')}; "
        f"{preferences.get('switching', 'neutral on line changes')}; "
        f"{preferences.get('fullness', 'neutral on train fullness')}."
    )
