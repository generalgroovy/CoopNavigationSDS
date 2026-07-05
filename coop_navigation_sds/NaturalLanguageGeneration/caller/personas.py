"""Persona model definitions for Agent A behavior and standardized experiment conditions."""

from coop_navigation_sds.NaturalLanguageGeneration.caller.config import DEFAULT_PERSONA, PERSONAS


def get_persona(persona_key: str):
    """Get persona function for this module's MVC responsibility.

    Args:
        persona_key: Input value used by `get_persona`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    key = str(persona_key or DEFAULT_PERSONA)
    if key not in PERSONAS:
        raise ValueError(
            f"Unknown persona '{key}'. Available: {', '.join(sorted(PERSONAS))}."
        )
    return PERSONAS[key]


def preference_text(persona: dict):
    """Format a persona preference profile for prompts."""
    preferences = persona.get("preferences", {})
    if not preferences:
        return "Prefs: fastest connected route."
    text = (
        "Prefs: "
        f"{preferences.get('priority', 'fastest route')}; "
        f"{preferences.get('switching', 'neutral on line changes')}; "
        f"{preferences.get('fullness', 'neutral on train fullness')}"
    )
    if preferences.get("reliability"):
        text += f"; {preferences['reliability']}"
    tickets = preferences.get("ticket_modes", ("metro", "tram"))
    text += f"; tickets for {' and '.join(tickets)}"
    text += f"; walk at most {preferences.get('max_walking_min', 10)} minutes"
    return text + "."
