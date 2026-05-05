import logging
import re
import torch

from config import (
    DEVICE,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    HISTORY_MESSAGES,
    LLM_AGENT_A,
)
from metro_data import STATION_POS
from route_planner import fmt_time
from prompt_data import AGENT_RULES, compact_prompt_context


STATION_NAMES = list(STATION_POS)
STATION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(station) for station in STATION_NAMES) + r")\b",
    re.IGNORECASE,
)
STATION_LOOKUP = {station.lower(): station for station in STATION_NAMES}


def build_agent_a_system(persona, scenario):
    return (
        f"You are Agent A, a traveler calling a transit hotline. "
        f"Persona: {persona['name']}. {persona['description']} "
        f"{AGENT_RULES} "
        "Reason with Agent B to build a connected route step by step. "
        "Only judge the route by whether it goes from the start station to the destination and by its total duration. "
        "Total duration is travel time plus waiting time plus transfer time. "
        f"{compact_prompt_context(scenario)}"
    )


def build_agent_b_system(scenario):
    return (
        "You are Agent B, a transit hotline assistant. "
        "You do not know the route in advance. "
        "Answer in full natural sentences, as if speaking to a passenger on the phone. "
        "Reason with Agent A to build a connected route step by step. "
        "Judge the route only by correctness from start to destination and by total duration. "
        "Total duration is the sum of riding time, waiting time, and transfer time. "
        "Mention station names in travel order when proposing route segments. "
        "Keep the dialogue verbal and natural; do not use route labels, code, JSON, tables, or machine-readable formatting. "
        "Explain how waiting, transfer, and riding time affect the current route duration. "
        f"{AGENT_RULES} "
        f"{compact_prompt_context(scenario)}"
    )


def initial_conversation(scenario):
    return [
        (
            "Agent A",
            (
                f"I am at {scenario['start_station']} at {fmt_time(scenario['start_time_min'])} "
                f"and need to get to {scenario['destination_station']}. "
                "Please reason with me step by step so we build a connected route and calculate its duration."
            ),
        )
    ]


def build_prompt(active_agent_name, active_agent_system, history):
    parts = [f"<|system|>\n{active_agent_system}\n</s>\n"]

    for speaker, text in history[-HISTORY_MESSAGES:]:
        role = "assistant" if speaker == active_agent_name else "user"
        parts.append(f"<|{role}|>\n{speaker}: {text}\n</s>\n")

    parts.append("<|assistant|>\n")
    return "".join(parts)


def clean_reply(text):
    text = text.strip()

    stop_markers = [
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "</s>",
        "Agent A:",
        "Agent B:",
        "```",
        "{",
        "[",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    text = " ".join(line.strip() for line in text.splitlines() if line.strip())

    banned_fragments = [
        "def ",
        "import ",
        "print(",
        "return ",
        "json",
        "{",
        "}",
        "[",
        "]",
        "|",
    ]

    if any(fragment in text.lower() for fragment in banned_fragments):
        return ""

    return text


def fallback_reply(active_agent_name, scenario):
    start = scenario["start_station"]
    destination = scenario["destination_station"]
    transfer = scenario["transfer_time_min"]

    if active_agent_name == "Agent B":
        return (
            f"I would start by checking a route that stays on one line from {start} toward {destination}. "
            f"I would also compare one transfer option, but that change adds about {transfer} minutes plus any waiting time. "
            "For either route, the duration is riding time plus waiting time plus transfer time."
        )

    return (
        "Please build the route one part at a time. "
        "First compare the simple option with one transfer option. "
        "Then add up riding, waiting, and transfer time."
    )


def generate_llm_reply(active_agent_name, active_agent_system, history, tokenizer, model, scenario):
    prompt = build_prompt(active_agent_name, active_agent_system, history)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    if input_ids.shape[1] > MAX_INPUT_TOKENS:
        logging.warning(
            "%s prompt has %s tokens, above configured MAX_INPUT_TOKENS=%s. "
            "Keeping the full prompt to preserve context.",
            active_agent_name,
            input_ids.shape[1],
            MAX_INPUT_TOKENS,
        )

    logging.info(f"{active_agent_name} input shape: {tuple(input_ids.shape)}")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    raw_reply = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    reply = clean_reply(raw_reply)
    return reply if reply else fallback_reply(active_agent_name, scenario)


def generate_agent_a_template(turn, persona, scenario):
    destination = scenario["destination_station"]
    persona_key = persona.get("key", "focused_commuter")

    templates = {
        "focused_commuter": [
            (
                f"Please start by comparing the main ways to reach {destination}. "
                "Let us check that each route is connected before we compare duration."
            ),
            (
                "Can we add up the riding time, waiting time, and any transfer time for the route so far? "
                "I want the duration calculation to be explicit."
            ),
            (
                "Now please confirm a connected route from my start station to the destination. "
                "Give the station order and the total duration."
            ),
        ],
        "distracted_multitasker": [
            (
                f"Sorry, can you start with two connected ways to {destination}? "
                "Keep it simple because I may miss details. "
                "Then we can add up the duration."
            ),
            (
                "Can you repeat which route is connected so far? "
                "Also tell me if there is a transfer. "
                "I want the riding, waiting, and transfer minutes kept separate."
            ),
            (
                "Please say the final route slowly, station by station. "
                "Mention where I change if I need to change. "
                "Keep the duration calculation clear."
            ),
        ],
        "verbose_planner": [
            (
                f"Let us compare a direct-looking route and a transfer route to {destination}. "
                "I want to verify connectivity and duration before deciding. "
                "Please build the reasoning step by step."
            ),
            (
                "Now add waiting time and transfer time to the riding time. "
                "Please show how those parts form the duration. "
                "Which route is fully connected so far?"
            ),
            (
                "Please summarize the connected route we have built. "
                "Give the station order clearly. "
                "Include whether there is a line change and the duration."
            ),
        ],
        "hesitant_speaker": [
            (
                f"I am not sure which way to {destination} is connected. "
                "Could we look at the options one at a time? "
                "I want to avoid choosing too early."
            ),
            (
                "Can you confirm whether the transfer is actually worth it? "
                "I do not want to change lines unless it clearly saves time. "
                "Please compare it carefully."
            ),
            (
                "I think I understand, but please confirm the final route. "
                "Say each station in order. "
                "Also say the total duration."
            ),
        ],
        "adversarial_tester": [
            (
                f"Do not assume the first route to {destination} is connected. "
                "Compare it against another viable option. "
                "I want to see the duration calculation for any transfer."
            ),
            (
                "That option sounds plausible, but justify it. "
                "Waiting time and transfer time can change the result. "
                "Which station links and duration parts are you using?"
            ),
            (
                "Now give me the final route only after checking each segment connects. "
                "Then give the station order and duration."
            ),
        ],
        "non_native_speaker": [
            (
                f"Please compare two ways to {destination} in simple words. "
                "Go step by step. "
                "Say station names clearly."
            ),
            (
                "How long is the route with waiting and changing trains? "
                "Please use simple sentences. "
                "Tell me if I must change."
            ),
            (
                "Please repeat the connected route with station names. "
                "Say where I change trains. "
                "Also say the total duration."
            ),
        ],
        "frustrated_user": [
            (
                f"I need a connected route to {destination}, but compare the main options first. "
                "Do it step by step, not too long. "
                "Then tell me the duration."
            ),
            (
                "Just check whether the transfer option is connected. "
                "Include waiting time and transfer time. "
                "Then narrow it down."
            ),
            (
                "Now give me the final route clearly. "
                "I need it to connect from start to destination. "
                "Mention any transfer and the total duration."
            ),
        ],
    }

    options = templates.get(persona_key, templates["focused_commuter"])
    return options[turn % len(options)]


def generate_reply(
    active_agent_name,
    active_agent_system,
    history,
    tokenizer,
    model,
    scenario,
    persona=None,
    turn=0,
):
    if active_agent_name == "Agent A" and not LLM_AGENT_A:
        return generate_agent_a_template(turn, persona or {}, scenario)

    return generate_llm_reply(
        active_agent_name,
        active_agent_system,
        history,
        tokenizer,
        model,
        scenario,
    )

