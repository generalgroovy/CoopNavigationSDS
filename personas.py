PERSONAS = {
    "focused_commuter": {
        "name": "Focused commuter",
        "description": "Direct, time-conscious, practical.",
    },
    "distracted_multitasker": {
        "name": "Distracted multitasker",
        "description": "Sometimes loses track and asks for repetition.",
    },
    "verbose_planner": {
        "name": "Verbose planner",
        "description": "Compares alternatives before deciding.",
    },
    "hesitant_speaker": {
        "name": "Hesitant speaker",
        "description": "Uncertain and often asks for confirmation.",
    },
    "adversarial_tester": {
        "name": "Adversarial tester",
        "description": "Challenges route suggestions.",
    },
    "non_native_speaker": {
        "name": "Non-native speaker",
        "description": "Uses simple English and wants station names repeated.",
    },
    "frustrated_user": {
        "name": "Frustrated user",
        "description": "Impatient and wants concise directions quickly.",
    },
}


DEFAULT_PERSONA = "focused_commuter"


def get_persona(persona_key: str):
    return PERSONAS.get(persona_key, PERSONAS[DEFAULT_PERSONA])