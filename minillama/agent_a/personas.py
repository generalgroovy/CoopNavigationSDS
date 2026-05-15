"""Persona model definitions for Agent A behavior and standardized experiment conditions."""

from minillama.agent_a.config import DEFAULT_PERSONA, PERSONAS


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
        return "Prefs: fastest connected route."
    return (
        "Prefs: "
        f"{preferences.get('priority', 'fastest route')}; "
        f"{preferences.get('switching', 'neutral on line changes')}; "
        f"{preferences.get('fullness', 'neutral on train fullness')}."
    )
