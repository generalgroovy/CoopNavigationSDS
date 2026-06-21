"""Automatic evaluation metrics for speech-dialog route-planning experiments."""
from collections import Counter
from dataclasses import asdict, dataclass, field
import math
import re
import statistics
import wave
from pathlib import Path

from coop_navigation_sds.Configuration.metrics import (
    METRIC_AUTOMATIC_ASR_WEIGHT,
    METRIC_AUTOMATIC_DIALOG_WEIGHT,
    METRIC_AUTOMATIC_NLU_WEIGHT,
    METRIC_QUALITY_BASE_WEIGHT,
    METRIC_QUALITY_DURATION_WEIGHT,
)
from coop_navigation_sds.EvaluationMetrics.catalog import (
    CORE_METRIC_KEYS,
    DEFAULT_METRIC_CONFIG,
    DEFAULT_METRIC_TIERS,
    METRIC_DISPLAY_NAMES,
    METRIC_FAMILY_SPECS,
    METRIC_KEYS,
    global_metric_key,
    metric_local_name,
    metric_calculation_method,
    phase_key,
)
from coop_navigation_sds.EvaluationMetrics.dnsmos import DNSMOSEvaluator
from coop_navigation_sds.EvaluationMetrics.nisqa import NISQAEvaluator
from coop_navigation_sds.TransportNetwork.network import STATION_POS
from coop_navigation_sds.TransportNetwork.routes import (
    optimal_time_route,
    route_duration_breakdown,
    route_line_change_count,
    route_line_sequence,
    route_station_sequence,
    route_text_from_steps,
)
from coop_navigation_sds.NaturalLanguageUnderstanding.transcript_normalization import transcript_token_changes


TASK_TERMS = {
    "station",
    "route",
    "line",
    "transfer",
    "change",
    "wait",
    "waiting",
    "duration",
    "time",
    "minutes",
    "arrive",
    "destination",
    "crowding",
    "full",
    "packed",
}


COMPARISON_TERMS = {
    "alternative",
    "better",
    "best",
    "compare",
    "faster",
    "improve",
    "option",
    "slower",
    "crowded",
    "quieter",
}


COOPERATION_TERMS = {
    "check",
    "confirm",
    "current",
    "candidate",
    "build",
    "revise",
    "together",
    "step",
    "clarify",
    "repeat",
}


def metric_tier_config_with_defaults(config=None):
    """Return a complete core/supplementary tier map."""
    provided = config or {}
    tiers = dict(DEFAULT_METRIC_TIERS)
    if not isinstance(provided, dict):
        return tiers
    for key, value in provided.items():
        if key not in METRIC_KEYS:
            continue
        tier = value.get("tier") if isinstance(value, dict) else value
        tier = str(tier or "").strip().lower()
        if tier in {"core", "supplementary"}:
            tiers[key] = tier
    return tiers


def _metric_enabled_value(config, key):
    if not isinstance(config, dict) or key not in config:
        return DEFAULT_METRIC_CONFIG[key]
    value = config[key]
    if isinstance(value, dict):
        return bool(value.get("enabled", DEFAULT_METRIC_CONFIG[key]))
    return bool(value)


def metric_config_with_defaults(config=None, tiers=None):
    """Return a complete metric switch map.

    Each metric has two controls: whether it is enabled, and whether it is core
    or supplementary. Core metrics are forced on; supplementary metrics use the
    enabled switch.
    """
    resolved_tiers = metric_tier_config_with_defaults(tiers)
    return {
        key: True if resolved_tiers[key] == "core" else _metric_enabled_value(config, key)
        for key in METRIC_KEYS
    }


def enabled_metric_family_specs(config=None, tiers=None):
    """Return only metric families and metric rows enabled by configuration."""
    switches = metric_config_with_defaults(config, tiers)
    families = []
    for family in METRIC_FAMILY_SPECS:
        metrics = [
            (key, label)
            for key, label in family["metrics"]
            if switches.get(key, False)
        ]
        if metrics:
            families.append({
                "key": family["key"],
                "order": family["order"],
                "title": family["title"],
                "description": family["description"],
                "metrics": metrics,
            })
    return families


def phase_key_from_title(title):
    """Return the explicit phase key, with a compatibility fallback for callers."""
    for family in METRIC_FAMILY_SPECS:
        if family["title"] == title:
            return phase_key(family)
    return re.sub(r"[^a-z0-9]+", "_", str(title).lower()).strip("_")


STATION_PATTERNS = {
    station: re.compile(rf"\b{re.escape(station)}\b", flags=re.IGNORECASE)
    for station in STATION_POS
}
STATION_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(station) for station in STATION_POS) + r")\b",
    flags=re.IGNORECASE,
)


@dataclass
class MetricRecord:
    """Serializable scalar metrics for one completed dialog run."""

    condition_id: str
    test_case_key: str
    persona_key: str
    scenario_key: str
    speech_pattern_key: str
    agent_a_audio_persona: str
    agent_b_audio_persona: str
    model_name: str
    model_param_key: str
    model_do_sample: bool
    model_temperature: float | None
    model_top_p: float | None
    success: bool
    conversation_outcome: str
    stated_constraints: str
    unsatisfied_constraints: str
    route_valid: bool
    route_reaches_goal: bool
    route_duration_min: int | None
    reference_duration_min: int | None
    constraint_duration_min: int | None
    route_line_sequence: str
    reference_line_sequence: str
    constraint_line_sequence: str
    route_line_change_count: int
    reference_line_change_count: int
    constraint_line_change_count: int | None
    duration_excess_min: int | None
    constraint_duration_gap_min: int | None
    constraint_line_change_gap: int | None
    constraint_fullness_gap: float | None
    constraint_near_capacity_gap: int | None
    route_near_capacity: bool
    route_near_capacity_count: int
    constraint_near_capacity: bool | None
    constraint_near_capacity_count: int | None
    transfer_tolerance: int | None
    allowed_modes: str
    route_transfer_miss_probability: float | None
    constraint_transfer_miss_probability: float | None
    constraint_transfer_miss_probability_gap: float | None
    route_delay_probability: float | None
    constraint_delay_probability: float | None
    constraint_delay_probability_gap: float | None
    travel_min: int
    wait_min: int
    transfer_min: int
    transfer_count: int
    average_route_fullness: float
    peak_route_fullness: int
    runtime_sec: float
    condition_runtime_sec: float | None
    speech_duration_total_sec: float
    max_turn_latency_sec: float
    mean_turn_elapsed_sec: float
    max_turn_elapsed_sec: float
    message_count: int
    word_count: int
    station_mentions: int
    task_focus_score: float
    comparison_terms: int
    cooperation_terms: int
    agent_a_question_count: int
    question_count: int
    clarification_rate: float
    avg_words_per_message: float
    lexical_diversity: float
    agent_b_distinct_1: float
    candidate_route_count: int
    route_revision_count: int
    best_candidate_turn: int | None
    pipeline_mode: str
    pipeline_success_rate: float
    pipeline_failure_count: int
    tts_success_rate: float
    tts_failure_count: int
    asr_success_rate: float
    asr_failure_count: int
    nlu_pipeline_input_match_rate: float
    asr_word_error_rate: float
    asr_sentence_error_rate: float
    asr_station_precision: float
    asr_station_recall: float
    tts_text_change_rate: float
    speech_incoming_enabled_rate: float
    speech_outgoing_enabled_rate: float
    nlu_route_valid_rate: float
    nlu_goal_reached_rate: float
    nlu_station_mention_rate: float
    agent_a_avg_latency_sec: float
    agent_b_avg_latency_sec: float
    mean_turn_latency_sec: float
    duration_score: float
    quality_score: float
    automatic_eval_score: float
    pair_id: str = ""
    run_type: str = "audio_variant"
    paired_task_success_delta: float | None = None
    paired_route_validity_delta: float | None = None
    paired_constraint_satisfaction_delta: float | None = None
    paired_turn_count_delta: float | None = None
    paired_repair_turn_delta: float | None = None
    paired_audio_error_effect: float | None = None
    metric_families: dict[str, dict[str, object]] = field(default_factory=dict)
    metric_calculations: dict[str, dict[str, object]] = field(default_factory=dict)

    def as_dict(self):
        row = asdict(self)
        families = row.pop("metric_families", {})
        row.pop("metric_calculations", None)
        for family_name, metrics in families.items():
            for metric_name, value in metrics.items():
                if metric_name in {
                    "available",
                    "coverage_rate",
                    "available_metric_count",
                    "configured_metric_count",
                }:
                    row[f"{family_name}_{metric_name}"] = value
                else:
                    row[global_metric_key(family_name, metric_name)] = value
        return row


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def tokenize_words(text):
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def normalized_text(text):
    return " ".join(tokenize_words(text))


def station_mentions_in_text(text):
    return {
        station
        for station, pattern in STATION_PATTERNS.items()
        if pattern.search(text)
    }


def ordered_station_mentions(text):
    """Return station mentions in the order they appear in text."""
    mentions = []
    for match in STATION_NAME_PATTERN.finditer(text):
        station = match.group(1)
        if not mentions or mentions[-1] != station:
            mentions.append(station)
    return mentions


def delexicalize_text(text):
    """Replace station and numeric mentions with placeholders for coarse semantic scoring."""
    out = text
    for station in sorted(STATION_POS, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(station)}\b", "__STATION__", out, flags=re.IGNORECASE)
    out = re.sub(r"\b\d+\b", "__NUM__", out)
    return normalized_text(out)


def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)) if len(tokens) >= n else Counter()


def simple_bleu(reference_tokens, hypothesis_tokens, max_n=2):
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_counts = ngram_counts(reference_tokens, n)
        hyp_counts = ngram_counts(hypothesis_tokens, n)
        total = sum(hyp_counts.values())
        if not total:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_counts[ngram]) for ngram, count in hyp_counts.items())
        precisions.append(overlap / total)

    if any(precision == 0 for precision in precisions):
        return 0.0

    log_precision = sum(math.log(precision) for precision in precisions) / len(precisions)
    brevity_penalty = 1.0
    if len(hypothesis_tokens) < len(reference_tokens):
        brevity_penalty = math.exp(1 - (len(reference_tokens) / max(len(hypothesis_tokens), 1)))
    return round(brevity_penalty * math.exp(log_precision), 4)


def simple_rouge(reference_tokens, hypothesis_tokens):
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    reference = Counter(reference_tokens)
    hypothesis = Counter(hypothesis_tokens)
    overlap = sum(min(count, hypothesis[token]) for token, count in reference.items())
    recall = overlap / len(reference_tokens)
    precision = overlap / len(hypothesis_tokens)
    return round(f1_score(precision, recall), 4)


def simple_meteor(reference_tokens, hypothesis_tokens):
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    reference = Counter(reference_tokens)
    hypothesis = Counter(hypothesis_tokens)
    overlap = sum(min(count, hypothesis[token]) for token, count in reference.items())
    precision = overlap / len(hypothesis_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0

    f_mean = (10 * precision * recall) / (recall + 9 * precision) if precision + recall else 0.0
    penalty = 0.5 * ((len(hypothesis_tokens) - overlap) / max(len(hypothesis_tokens), 1))
    return round(max(0.0, f_mean * (1.0 - penalty)), 4)


def token_jaccard(reference_tokens, hypothesis_tokens):
    reference = set(reference_tokens)
    hypothesis = set(hypothesis_tokens)
    union = reference | hypothesis
    if not union:
        return 0.0
    return round(len(reference & hypothesis) / len(union), 4)


def simple_rouge_l(reference_tokens, hypothesis_tokens):
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    rows = len(reference_tokens) + 1
    cols = len(hypothesis_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if reference_tokens[i - 1] == hypothesis_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    recall = lcs / len(reference_tokens)
    precision = lcs / len(hypothesis_tokens)
    return round(f1_score(precision, recall), 4)


def f1_score(precision, recall):
    if not precision or not recall:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def edit_counts(reference_tokens, hypothesis_tokens):
    rows = len(reference_tokens) + 1
    cols = len(hypothesis_tokens) + 1
    dp = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        cost, subs, dels, ins = dp[i - 1][0]
        dp[i][0] = (cost + 1, subs, dels + 1, ins)
    for j in range(1, cols):
        cost, subs, dels, ins = dp[0][j - 1]
        dp[0][j] = (cost + 1, subs, dels, ins + 1)

    for i in range(1, rows):
        for j in range(1, cols):
            if reference_tokens[i - 1] == hypothesis_tokens[j - 1]:
                candidates = [dp[i - 1][j - 1]]
            else:
                cost, subs, dels, ins = dp[i - 1][j - 1]
                candidates = [(cost + 1, subs + 1, dels, ins)]

            cost, subs, dels, ins = dp[i - 1][j]
            candidates.append((cost + 1, subs, dels + 1, ins))
            cost, subs, dels, ins = dp[i][j - 1]
            candidates.append((cost + 1, subs, dels, ins + 1))
            dp[i][j] = min(
                candidates,
                key=lambda item: (item[0], item[1] + item[2] + item[3], item[1], item[2], item[3]),
            )

    _, substitutions, deletions, insertions = dp[-1][-1]
    return substitutions, deletions, insertions


def sequence_error_rate(reference, hypothesis):
    """Return edit distance normalized by the reference sequence length."""
    substitutions, deletions, insertions = edit_counts(list(reference), list(hypothesis))
    return safe_ratio(substitutions + deletions + insertions, len(reference))


def preservation_rate(reference_items, hypothesis_items):
    """Return multiset recall for values that must survive a transformation."""
    reference = Counter(reference_items)
    hypothesis = Counter(hypothesis_items)
    total = sum(reference.values())
    preserved = sum(min(count, hypothesis[item]) for item, count in reference.items())
    return safe_ratio(preserved, total)


def numeric_mentions(text):
    return re.findall(r"\b\d+(?::\d+)?(?:\.\d+)?\b", text or "")


def negation_mentions(text):
    return [
        token
        for token in tokenize_words(text or "")
        if token in {"no", "not", "never", "without", "avoid", "excluding", "except"}
    ]


def constraint_mentions(text):
    terms = {
        "transfers": {"transfer", "transfers", "change", "changes"},
        "fullness": {"crowded", "crowding", "full", "packed", "capacity"},
        "delay": {"delay", "delayed", "reliable", "reliability"},
        "transfer_miss": {"miss", "connection", "risk"},
    }
    words = set(tokenize_words(text or ""))
    return {key for key, keywords in terms.items() if words & keywords}


def analyze_wave_file(path):
    """Return inexpensive deterministic acoustic measurements for a PCM WAV."""
    if not path:
        return None
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except (OSError, wave.Error):
        return None
    if not raw or sample_width not in {1, 2, 4} or channels <= 0 or frame_rate <= 0:
        return None

    signed = sample_width != 1
    offset = -(2 ** (sample_width * 8 - 1)) if sample_width == 1 else 0
    samples = [
        int.from_bytes(raw[index:index + sample_width], "little", signed=signed) + offset
        for index in range(0, len(raw) - sample_width + 1, sample_width)
    ]
    if not samples:
        return None
    max_amplitude = float(2 ** (sample_width * 8 - 1))
    absolute = [abs(value) for value in samples]
    rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    clipping_rate = safe_ratio(sum(1 for value in absolute if value >= max_amplitude * 0.99), len(samples))
    silence_threshold = max_amplitude * 0.01
    silence_rate = safe_ratio(sum(1 for value in absolute if value <= silence_threshold), len(samples))
    non_silent = [index for index, value in enumerate(absolute) if value > silence_threshold]
    first = non_silent[0] if non_silent else len(samples)
    last = non_silent[-1] if non_silent else -1
    samples_per_second = frame_rate * channels
    leading_silence = safe_ratio(first, samples_per_second)
    trailing_silence = safe_ratio(max(len(samples) - last - 1, 0), samples_per_second)
    loudness_dbfs = 20 * math.log10(max(rms / max_amplitude, 1e-12))
    return {
        "duration_sec": safe_ratio(frame_count, frame_rate),
        "sample_rate": frame_rate,
        "channels": channels,
        "clipping_rate": clipping_rate,
        "silence_ratio": silence_rate,
        "leading_silence_sec": leading_silence,
        "trailing_silence_sec": trailing_silence,
        "loudness_dbfs": loudness_dbfs,
    }


def mean_or_none(values):
    values = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return sum(values) / len(values) if values else None


def variance_or_none(values):
    values = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return statistics.pvariance(values) if len(values) >= 2 else None


def pearson_correlation(left, right):
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right)
        if isinstance(x, (int, float))
        and not isinstance(x, bool)
        and isinstance(y, (int, float))
        and not isinstance(y, bool)
    ]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def wilson_interval(successes, total, z=1.96):
    """Return a Wilson score interval for a binomial proportion."""
    if not total:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


class MetricComputer:
    """Compute thesis-facing automatic evaluation metrics from a completed dialog."""

    def __init__(self, nisqa_evaluator=None, dnsmos_evaluator=None):
        self.nisqa_evaluator = nisqa_evaluator or NISQAEvaluator()
        self.dnsmos_evaluator = dnsmos_evaluator or DNSMOSEvaluator()

    def compute(self, result, scenario) -> MetricRecord:
        reference_arrival, reference_steps = optimal_time_route(
            scenario["start_station"],
            scenario["destination_station"],
            scenario["start_time_min"],
            scenario["transfer_time_min"],
        )
        reference_duration = (
            reference_arrival - scenario["start_time_min"]
            if reference_arrival is not None
            else None
        )

        breakdown = route_duration_breakdown(result.route_steps) if result.route_steps else {
            "travel": 0,
            "wait": 0,
            "transfer": 0,
        }

        duration_excess = None
        if result.route_duration_min is not None and reference_duration is not None:
            duration_excess = result.route_duration_min - reference_duration
        constraint_duration = result.extra.get("constraint_duration_min")
        constraint_duration_gap = result.extra.get("constraint_duration_gap_min")
        constraint_line_change_gap = result.extra.get("constraint_line_change_gap")
        constraint_fullness_gap = result.extra.get("constraint_fullness_gap")
        constraint_near_capacity_gap = result.extra.get("constraint_near_capacity_gap")
        route_near_capacity = bool(result.extra.get("route_near_capacity", False))
        route_near_capacity_count = int(result.extra.get("route_near_capacity_count", 0) or 0)
        constraint_near_capacity = result.extra.get("constraint_near_capacity")
        constraint_near_capacity_count = result.extra.get("constraint_near_capacity_count")
        transfer_tolerance = result.extra.get("transfer_tolerance")
        allowed_lines = result.extra.get("allowed_modes") or result.extra.get("allowed_lines") or []
        route_transfer_miss_probability = result.extra.get("route_transfer_miss_probability")
        constraint_transfer_miss_probability = result.extra.get("constraint_transfer_miss_probability")
        constraint_transfer_miss_probability_gap = result.extra.get("constraint_transfer_miss_probability_gap")
        constraint_delay_probability = result.extra.get("constraint_delay_probability")
        constraint_delay_probability_gap = result.extra.get("constraint_delay_probability_gap")
        route_delay_values = [step.get("delay_probability", 0.0) for step in result.route_steps]
        route_delay_probability = round(max(route_delay_values), 4) if route_delay_values else None

        conversation_text = " ".join(text for _, text in result.conversation)
        words = tokenize_words(conversation_text)
        task_terms = sum(1 for word in words if word in TASK_TERMS)
        comparison_terms = sum(1 for word in words if word in COMPARISON_TERMS)
        cooperation_terms = sum(1 for word in words if word in COOPERATION_TERMS)
        station_mentions = sum(
            len(pattern.findall(conversation_text))
            for pattern in STATION_PATTERNS.values()
        )
        question_count = sum(text.count("?") for _, text in result.conversation)
        agent_a_question_count = sum(
            text.count("?") for speaker, text in result.conversation if speaker == "Agent A"
        )
        message_count = result.extra.get("messages", len(result.conversation))
        avg_words_per_message = len(words) / message_count if message_count else 0.0
        task_focus_score = task_terms / len(words) if words else 0.0
        lexical_diversity = safe_ratio(len(set(words)), len(words))

        agent_b_tokens = tokenize_words(
            " ".join(text for speaker, text in result.conversation if speaker == "Agent B")
        )
        agent_b_distinct_1 = safe_ratio(len(set(agent_b_tokens)), len(agent_b_tokens))
        clarification_rate = safe_ratio(agent_a_question_count, message_count)

        route_fullness_values = [step.get("fullness", 0) for step in result.route_steps]
        average_route_fullness = (
            round(sum(route_fullness_values) / len(route_fullness_values), 2)
            if route_fullness_values else 0.0
        )
        peak_route_fullness = max(route_fullness_values) if route_fullness_values else 0
        transfer_count = sum(1 for step in result.route_steps if step.get("transfer", 0) > 0)

        speech_turns = result.extra.get("speech_turns", [])
        timing_turns = result.extra.get("timing_turns", [])
        nlu_turns = result.extra.get("nlu_turns", [])
        model_parameters = result.extra.get("model_parameters", {})
        pipeline_failure = result.extra.get("pipeline_failure")
        pipeline_mode = next(
            (turn.get("mode") for turn in speech_turns if turn.get("mode")),
            result.extra.get("pipeline_mode", "unknown"),
        )

        ref_word_total = 0
        word_substitutions = 0
        word_deletions = 0
        word_insertions = 0
        sentence_errors = 0
        station_tp = 0
        station_fp = 0
        station_fn = 0
        tts_ref_word_total = 0
        tts_word_edits = 0
        incoming_enabled_count = 0
        outgoing_enabled_count = 0
        pipeline_failure_count = 1 if pipeline_failure else 0
        tts_success_count = 0
        tts_failure_count = 0
        asr_success_count = 0
        asr_failure_count = 0
        transcript_correction_count = 0
        uncorrected_misinterpretation_count = 0
        agent_b_transcripts = []

        for turn in speech_turns:
            pipeline_ok = turn.get("pipeline_ok", True)
            if not pipeline_ok:
                pipeline_failure_count += 1
            generated_tokens = tokenize_words(turn.get("generated_text", turn.get("source_text", "")))
            outgoing_tokens = tokenize_words(turn.get("outgoing_text", turn.get("source_text", "")))
            tts_subs, tts_dels, tts_ins = edit_counts(generated_tokens, outgoing_tokens)
            tts_ref_word_total += len(generated_tokens)
            tts_word_edits += tts_subs + tts_dels + tts_ins
            incoming_enabled_count += 1 if turn.get("incoming_enabled") else 0
            outgoing_enabled_count += 1 if turn.get("outgoing_enabled") else 0
            if turn.get("outgoing_enabled"):
                audio = turn.get("audio") if isinstance(turn.get("audio"), dict) else {}
                if pipeline_ok and audio.get("path") and turn.get("tts_engine") != "disabled":
                    tts_success_count += 1
                else:
                    tts_failure_count += 1

            source_tokens = outgoing_tokens
            transcript_tokens = tokenize_words(turn.get("incoming_transcript", turn.get("transcript", "")))
            ref_word_total += len(source_tokens)
            subs, dels, ins = edit_counts(source_tokens, transcript_tokens)
            word_substitutions += subs
            word_deletions += dels
            word_insertions += ins
            source_text = turn.get("outgoing_text", turn.get("source_text", ""))
            transcript_text = turn.get("incoming_transcript", turn.get("transcript", ""))
            transcript_correction_count += len(turn.get("transcript_corrections", []))
            uncorrected_misinterpretation_count += len(
                transcript_token_changes(source_text, transcript_text)
            )
            if normalized_text(source_text) != normalized_text(transcript_text):
                sentence_errors += 1
            if turn.get("incoming_enabled"):
                if pipeline_ok and normalized_text(transcript_text):
                    asr_success_count += 1
                else:
                    asr_failure_count += 1
            if turn.get("speaker") == "Agent B":
                agent_b_transcripts.append(normalized_text(transcript_text))

            source_stations = station_mentions_in_text(source_text)
            transcript_stations = station_mentions_in_text(transcript_text)
            station_tp += len(source_stations & transcript_stations)
            station_fp += len(transcript_stations - source_stations)
            station_fn += len(source_stations - transcript_stations)

        asr_word_error_rate = safe_ratio(
            word_substitutions + word_deletions + word_insertions,
            ref_word_total,
        )
        asr_sentence_error_rate = safe_ratio(sentence_errors, incoming_enabled_count)
        asr_station_precision = safe_ratio(station_tp, station_tp + station_fp)
        asr_station_recall = safe_ratio(station_tp, station_tp + station_fn)
        tts_text_change_rate = safe_ratio(tts_word_edits, tts_ref_word_total)
        speech_incoming_enabled_rate = safe_ratio(incoming_enabled_count, len(speech_turns))
        speech_outgoing_enabled_rate = safe_ratio(outgoing_enabled_count, len(speech_turns))
        pipeline_success_rate = safe_ratio(
            sum(1 for turn in speech_turns if turn.get("pipeline_ok", True)),
            len(speech_turns),
        )
        tts_success_rate = safe_ratio(tts_success_count, outgoing_enabled_count)
        asr_success_rate = safe_ratio(asr_success_count, incoming_enabled_count)

        nlu_route_valid_rate = safe_ratio(
            sum(1 for turn in nlu_turns if turn.get("route_valid")),
            len(nlu_turns),
        )
        nlu_goal_reached_rate = safe_ratio(
            sum(1 for turn in nlu_turns if turn.get("route_reaches_goal")),
            len(nlu_turns),
        )
        nlu_station_mention_rate = safe_ratio(
            sum(1 for turn in nlu_turns if turn.get("has_station_mentions")),
            len(nlu_turns),
        )
        nlu_pipeline_input_match_count = 0
        for index, turn in enumerate(nlu_turns):
            expected = agent_b_transcripts[index] if index < len(agent_b_transcripts) else None
            if expected is not None and normalized_text(turn.get("text", "")) == expected:
                nlu_pipeline_input_match_count += 1
        nlu_pipeline_input_match_rate = safe_ratio(nlu_pipeline_input_match_count, len(nlu_turns))
        nlu_latency_sec = mean_or_none([turn.get("latency_sec") for turn in nlu_turns])

        def average_latency(speaker):
            speaker_turns = [turn for turn in timing_turns if turn.get("speaker") == speaker]
            return safe_ratio(
                sum(turn.get("turn_latency_sec", 0.0) for turn in speaker_turns),
                len(speaker_turns),
            )

        agent_a_avg_latency_sec = average_latency("Agent A")
        agent_b_avg_latency_sec = average_latency("Agent B")
        agent_timing_summary = result.extra.get("agent_timing_summary", {})
        agent_a_timing = agent_timing_summary.get("Agent A", {})
        agent_b_timing = agent_timing_summary.get("Agent B", {})
        turn_elapsed_values = [
            turn.get("turn_elapsed_sec", turn.get("turn_latency_sec", 0.0))
            for turn in timing_turns
        ]
        mean_turn_latency_sec = safe_ratio(
            sum(turn.get("turn_latency_sec", 0.0) for turn in timing_turns),
            len(timing_turns),
        )
        mean_turn_elapsed_sec = safe_ratio(sum(turn_elapsed_values), len(turn_elapsed_values))
        speech_duration_total_sec = sum(turn.get("speech_sec", 0.0) for turn in timing_turns)
        max_turn_latency_sec = max((turn.get("turn_latency_sec", 0.0) for turn in timing_turns), default=0.0)
        max_turn_elapsed_sec = max(turn_elapsed_values, default=0.0)

        if result.route_duration_min is None or reference_duration is None:
            duration_score = 0.0
        else:
            duration_score = reference_duration / max(result.route_duration_min, reference_duration)

        quality_score = 0.0
        if result.route_correct:
            quality_score = METRIC_QUALITY_BASE_WEIGHT + METRIC_QUALITY_DURATION_WEIGHT * duration_score

        asr_component = max(0.0, 1.0 - asr_word_error_rate)
        nlu_component = 0.5 * nlu_route_valid_rate + 0.5 * nlu_goal_reached_rate
        dialog_component = (
            0.45 * lexical_diversity
            + 0.30 * agent_b_distinct_1
            + 0.15 * max(0.0, 1.0 - safe_ratio(result.extra.get("warning_count", 0), max(message_count, 1)))
            + 0.10 * max(0.0, 1.0 - min(mean_turn_latency_sec / 20.0, 1.0))
        )
        automatic_eval_score = (
            quality_score
            + METRIC_AUTOMATIC_ASR_WEIGHT * asr_component
            + METRIC_AUTOMATIC_NLU_WEIGHT * nlu_component
            + METRIC_AUTOMATIC_DIALOG_WEIGHT * dialog_component
        )

        warning_count = result.extra.get("warning_count", 0)
        route_revision_count = result.extra.get("route_revisions", 0)
        best_candidate_turn = result.extra.get("best_candidate_turn")
        reference_route = route_station_sequence(reference_steps)
        reference_route_text = route_text_from_steps(reference_steps)
        route_lines = route_line_sequence(result.route_steps)
        reference_lines = route_line_sequence(reference_steps)
        constraint_lines = result.extra.get("constraint_line_sequence", [])
        route_line_changes = route_line_change_count(result.route_steps)
        reference_line_changes = route_line_change_count(reference_steps)
        constraint_line_changes = result.extra.get("constraint_line_changes")
        agent_b_messages = [text for speaker, text in result.conversation if speaker == "Agent B"]
        final_agent_b_text = agent_b_messages[-1] if agent_b_messages else ""

        source_char_total = 0
        char_substitutions = 0
        char_deletions = 0
        char_insertions = 0
        entity_substitutions = 0
        entity_deletions = 0
        entity_insertions = 0
        entity_source_total = 0
        keyword_source_total = 0
        keyword_preserved_total = 0

        for turn in speech_turns:
            source_text = turn.get("source_text", turn.get("outgoing_text", ""))
            transcript_text = turn.get("transcript", turn.get("incoming_transcript", ""))
            source_chars = list(re.sub(r"\s+", "", source_text.lower()))
            transcript_chars = list(re.sub(r"\s+", "", transcript_text.lower()))
            source_char_total += len(source_chars)
            subs, dels, ins = edit_counts(source_chars, transcript_chars)
            char_substitutions += subs
            char_deletions += dels
            char_insertions += ins

            source_entities = ordered_station_mentions(source_text)
            transcript_entities = ordered_station_mentions(transcript_text)
            ent_subs, ent_dels, ent_ins = edit_counts(source_entities, transcript_entities)
            entity_substitutions += ent_subs
            entity_deletions += ent_dels
            entity_insertions += ent_ins
            entity_source_total += len(source_entities)

            source_keywords = {word for word in tokenize_words(source_text) if word in TASK_TERMS}
            transcript_keywords = {word for word in tokenize_words(transcript_text) if word in TASK_TERMS}
            keyword_source_total += len(source_keywords)
            keyword_preserved_total += len(source_keywords & transcript_keywords)

        asr_character_error_rate = safe_ratio(char_substitutions + char_deletions + char_insertions, source_char_total)
        asr_deletion_rate = safe_ratio(word_deletions, ref_word_total)
        asr_substitution_rate = safe_ratio(word_substitutions, ref_word_total)
        asr_insertion_rate = safe_ratio(word_insertions, ref_word_total)
        asr_token_error_rate = asr_word_error_rate
        asr_entity_wer = safe_ratio(entity_substitutions + entity_deletions + entity_insertions, entity_source_total)
        asr_keyword_recall = safe_ratio(keyword_preserved_total, keyword_source_total)

        from coop_navigation_sds.NaturalLanguageUnderstanding.interpreter import NaturalRouteInterpreter

        interpreted_final_route = NaturalRouteInterpreter().interpret_reply(final_agent_b_text, scenario)
        final_route_entities = interpreted_final_route or ordered_station_mentions(final_agent_b_text)
        reference_entities = list(reference_route)
        entity_overlap = set(final_route_entities) & set(reference_entities)
        slot_precision = safe_ratio(len(entity_overlap), len(set(final_route_entities)))
        slot_recall = safe_ratio(len(entity_overlap), len(set(reference_entities)))
        slot_realization_accuracy = f1_score(slot_precision, slot_recall)

        reference_delex_tokens = tokenize_words(delexicalize_text(reference_route_text))
        final_delex_tokens = tokenize_words(delexicalize_text(final_agent_b_text))
        nlg_bleu = simple_bleu(reference_delex_tokens, final_delex_tokens)
        nlg_rouge = simple_rouge(reference_delex_tokens, final_delex_tokens)
        nlg_meteor = simple_meteor(reference_delex_tokens, final_delex_tokens)
        nlg_bert_score = token_jaccard(reference_delex_tokens, final_delex_tokens)
        nlg_semantic_similarity = nlg_bert_score
        nlg_delexicalized_bleu = nlg_bleu
        nlg_distinct_1 = safe_ratio(len(set(tokenize_words(" ".join(agent_b_messages)))), len(tokenize_words(" ".join(agent_b_messages))))
        agent_b_token_stream = tokenize_words(" ".join(agent_b_messages))
        agent_b_bigrams = list(zip(agent_b_token_stream, agent_b_token_stream[1:]))
        nlg_distinct_2 = safe_ratio(len(set(agent_b_bigrams)), len(agent_b_bigrams))
        nlg_repetition_rate = safe_ratio(len(agent_b_bigrams) - len(set(agent_b_bigrams)), len(agent_b_bigrams))
        nlg_constraint_satisfaction_rate = 1.0 if result.route_correct else 0.0

        agent_b_timing_turns = [turn for turn in timing_turns if turn.get("speaker") == "Agent B"]
        agent_b_generation_latency = mean_or_none([
            turn.get("raw_generation_sec", turn.get("generation_sec"))
            for turn in agent_b_timing_turns
        ])
        time_to_first_token_sec = (
            min((turn.get("generation_sec", 0.0) for turn in agent_b_timing_turns), default=None)
            if agent_b_timing_turns
            else None
        )
        time_to_first_audio_sec = (
            min((turn.get("turn_latency_sec", 0.0) for turn in agent_b_timing_turns), default=None)
            if agent_b_timing_turns
            else None
        )
        end_of_turn_detection_accuracy = None
        endpointing_latency_sec = None
        barge_in_true_positive_rate = None
        barge_in_false_positive_rate = None
        barge_in_suppression_latency_sec = None

        route_exact_match = result.route == reference_route
        final_turns_to_success = best_candidate_turn if result.route_correct and best_candidate_turn is not None else len(result.conversation)
        response_latency_sec = mean_turn_latency_sec
        task_success_rate = 1.0 if result.route_correct else 0.0
        inform_rate = 1.0 if result.route_reaches_goal else 0.0
        completion_rate = 1.0 if result.conversation else 0.0
        abandonment_rate = 0.0 if result.route_correct else 1.0
        escalation_rate = safe_ratio(warning_count, max(len(nlu_turns), 1))
        warning_pressure = safe_ratio(warning_count, max(message_count, 1))
        average_reward = automatic_eval_score
        predicted_user_satisfaction = automatic_eval_score
        per_domain_failure_rate = 1.0 - task_success_rate
        cohort_fairness_gaps = None
        robustness_by_noise_gap = None
        robustness_by_accent_gap = None
        robustness_by_device_gap = None
        robustness_by_environment_gap = None
        cost_per_success = None
        safety_refusal_precision = None
        safety_refusal_recall = None
        privacy_redaction_accuracy = None

        metric_families = {
            "audio": {
                "available": False,
                "snr_db": None,
                "si_snr_db": None,
                "clipping_rate": None,
                "packet_loss_rate": None,
                "sample_rate_mismatch": None,
                "loudness_lufs": None,
                "noise_estimate": None,
                "pesq": None,
                "dnsmos": None,
            },
            "vad": {
                "available": False,
                "false_alarm_rate": None,
                "miss_rate": None,
                "detection_error_rate": None,
                "speech_non_speech_f1": None,
                "endpointing_latency_sec": None,
            },
            "diarization": {
                "available": False,
                "der": None,
                "missed_speech_rate": None,
                "false_alarm_rate": None,
                "speaker_confusion_rate": None,
                "overlap_detection_f1": None,
            },
            "asr": {
                "available": incoming_enabled_count > 0,
                "success_rate": round(asr_success_rate, 4),
                "failure_count": asr_failure_count,
                "word_error_rate": round(asr_word_error_rate, 4),
                "token_error_rate": round(asr_token_error_rate, 4),
                "character_error_rate": round(asr_character_error_rate, 4),
                "deletion_rate": round(asr_deletion_rate, 4),
                "substitution_rate": round(asr_substitution_rate, 4),
                "insertion_rate": round(asr_insertion_rate, 4),
                "entity_wer": round(asr_entity_wer, 4),
                "keyword_recall": round(asr_keyword_recall, 4),
                "confidence_calibration": None,
            },
            "slu": {
                "available": True,
                "pipeline_input_match_rate": round(nlu_pipeline_input_match_rate, 4),
                "intent_accuracy": round(nlu_route_valid_rate, 4),
                "intent_error_rate": round(max(0.0, 1.0 - nlu_route_valid_rate), 4),
                "slot_f1": slot_realization_accuracy,
                "slot_error_rate": round(max(0.0, 1.0 - slot_realization_accuracy), 4),
                "concept_error_rate": round(max(0.0, 1.0 - nlu_route_valid_rate), 4),
                "sentence_semantic_accuracy": round(nlu_route_valid_rate, 4),
                "semantic_frame_accuracy": round(nlu_goal_reached_rate, 4),
            },
            "dst": {
                "available": True,
                "joint_goal_accuracy": round(task_success_rate, 4),
                "average_goal_accuracy": round(nlu_goal_reached_rate, 4),
                "requested_slot_f1": round(slot_realization_accuracy, 4),
                "active_intent_accuracy": round(nlu_route_valid_rate, 4),
                "state_update_accuracy": round(nlu_goal_reached_rate, 4),
                "belief_state_calibration": None,
                "l2": None,
                "mrr": None,
                "roc": None,
            },
            "policy": {
                "available": True,
                "dialog_act_accuracy": None,
                "dialog_act_f1": None,
                "next_action_accuracy": round(1.0 if route_exact_match else 0.0, 4),
                "tool_call_exact_match": round(1.0 if route_exact_match else 0.0, 4),
                "parameter_exact_match": 1.0,
                "invalid_action_rate": round(safe_ratio(sum(1 for turn in nlu_turns if turn.get("has_station_mentions") and not turn.get("route_valid")), max(len(nlu_turns), 1)), 4),
                "fallback_rate": round(safe_ratio(warning_count, max(len(nlu_turns), 1)), 4),
                "repair_rate": round(safe_ratio(route_revision_count, max(len(nlu_turns), 1)), 4),
                "confirmation_rate": round(clarification_rate, 4),
            },
            "tool": {
                "available": True,
                "entity_match_rate": round(slot_recall, 4),
                "api_success_rate": round(1.0 if reference_route else 0.0, 4),
                "tool_call_validity": round(1.0 if result.route_valid else 0.0, 4),
                "result_relevance": round(inform_rate, 4),
                "hit_at_k": round(1.0 if route_exact_match else 0.0, 4),
                "mrr": round(1.0 if route_exact_match else 0.0, 4),
                "grounding_accuracy": round(task_success_rate, 4),
                "hallucinated_field_rate": None,
            },
            "nlg": {
                "available": True,
                "slot_realization_accuracy": round(slot_realization_accuracy, 4),
                "bleu": nlg_bleu,
                "rouge": nlg_rouge,
                "meteor": nlg_meteor,
                "bert_score": round(nlg_bert_score, 4),
                "semantic_similarity": nlg_semantic_similarity,
                "delexicalized_bleu": nlg_delexicalized_bleu,
                "distinct_1": round(nlg_distinct_1, 4),
                "distinct_2": round(nlg_distinct_2, 4),
                "repetition_rate": round(nlg_repetition_rate, 4),
                "constraint_satisfaction_rate": round(nlg_constraint_satisfaction_rate, 4),
            },
            "tts": {
                "available": outgoing_enabled_count > 0,
                "success_rate": round(tts_success_rate, 4),
                "failure_count": tts_failure_count,
                "predicted_mos": None,
                "intelligibility_wer": round(asr_word_error_rate, 4),
                "stoi": None,
                "mcd": round(tts_text_change_rate, 4),
                "pesq": None,
                "speechbert_score": None,
                "speaker_similarity": None,
                "f0_correlation": None,
            },
            "runtime": {
                "available": True,
                "end_of_turn_detection_accuracy": end_of_turn_detection_accuracy,
                "endpointing_latency_sec": endpointing_latency_sec,
                "barge_in_true_positive_rate": barge_in_true_positive_rate,
                "barge_in_false_positive_rate": barge_in_false_positive_rate,
                "barge_in_suppression_latency_sec": barge_in_suppression_latency_sec,
                "response_latency_sec": round(response_latency_sec, 4),
                "mean_turn_elapsed_sec": round(mean_turn_elapsed_sec, 4),
                "max_turn_latency_sec": round(max_turn_latency_sec, 4),
                "max_turn_elapsed_sec": round(max_turn_elapsed_sec, 4),
                "speech_duration_total_sec": round(speech_duration_total_sec, 4),
                "condition_runtime_sec": result.extra.get("condition_runtime_sec"),
                "time_to_first_token_sec": None if time_to_first_token_sec is None else round(time_to_first_token_sec, 4),
                "time_to_first_audio_sec": None if time_to_first_audio_sec is None else round(time_to_first_audio_sec, 4),
                "interruption_recovery_rate": None,
            },
            "agent_timing": {
                "available": bool(agent_timing_summary),
                "agent_a_turn_count": agent_a_timing.get("turn_count", 0),
                "agent_a_word_count": agent_a_timing.get("word_count", 0),
                "agent_a_mean_words_per_turn": agent_a_timing.get("mean_words_per_turn", 0.0),
                "agent_a_total_generation_sec": agent_a_timing.get("total_generation_sec", 0.0),
                "agent_a_total_speech_sec": agent_a_timing.get("total_speech_sec", 0.0),
                "agent_a_mean_turn_elapsed_sec": agent_a_timing.get("mean_turn_elapsed_sec", agent_a_avg_latency_sec),
                "agent_a_max_turn_elapsed_sec": agent_a_timing.get("max_turn_elapsed_sec", 0.0),
                "agent_b_turn_count": agent_b_timing.get("turn_count", 0),
                "agent_b_word_count": agent_b_timing.get("word_count", 0),
                "agent_b_mean_words_per_turn": agent_b_timing.get("mean_words_per_turn", 0.0),
                "agent_b_total_generation_sec": agent_b_timing.get("total_generation_sec", 0.0),
                "agent_b_total_speech_sec": agent_b_timing.get("total_speech_sec", 0.0),
                "agent_b_mean_turn_elapsed_sec": agent_b_timing.get("mean_turn_elapsed_sec", agent_b_avg_latency_sec),
                "agent_b_max_turn_elapsed_sec": agent_b_timing.get("max_turn_elapsed_sec", 0.0),
            },
            "pipeline": {
                "available": True,
                "mode": pipeline_mode,
                "success_rate": round(pipeline_success_rate, 4),
                "failure_count": pipeline_failure_count,
                "tts_attempt_count": outgoing_enabled_count,
                "asr_attempt_count": incoming_enabled_count,
                "nlu_attempt_count": len(nlu_turns),
                "phase_output_dependency_rate": round(nlu_pipeline_input_match_rate, 4),
                "failure_reason": pipeline_failure.get("message") if isinstance(pipeline_failure, dict) else None,
            },
            "end_to_end": {
                "available": True,
                "task_success": round(task_success_rate, 4),
                "inform_rate": round(inform_rate, 4),
                "request_success": round(inform_rate, 4),
                "completion_rate": round(completion_rate, 4),
                "abandonment_rate": round(abandonment_rate, 4),
                "escalation_rate": round(escalation_rate, 4),
                "average_reward": round(average_reward, 4),
                "turns_to_success": final_turns_to_success,
                "dialog_duration_sec": round(result.runtime_sec, 4),
                "reprompt_count": warning_count,
                "confirmation_count": question_count,
            },
            "posthoc": {
                "available": True,
                "predicted_user_satisfaction": round(predicted_user_satisfaction, 4),
                "per_domain_failure_rate": round(per_domain_failure_rate, 4),
                "cohort_fairness_gaps": cohort_fairness_gaps,
                "robustness_by_noise_gap": robustness_by_noise_gap,
                "robustness_by_accent_gap": robustness_by_accent_gap,
                "robustness_by_device_gap": robustness_by_device_gap,
                "robustness_by_environment_gap": robustness_by_environment_gap,
                "cost_per_success": cost_per_success,
                "safety_refusal_precision": safety_refusal_precision,
                "safety_refusal_recall": safety_refusal_recall,
                "privacy_redaction_accuracy": privacy_redaction_accuracy,
            },
        }

        candidate_events = list(result.extra.get("candidate_events", []))
        unique_candidates = [
            event for event in candidate_events
            if event.get("decision") != "repeat" and event.get("route")
        ]
        repeated_candidates = [event for event in candidate_events if event.get("decision") == "repeat"]
        invalid_proposals = [
            turn for turn in nlu_turns
            if turn.get("has_station_mentions") and not turn.get("route_valid")
        ]
        candidate_route_count = int(result.extra.get("candidate_routes", len(unique_candidates)) or 0)
        valid_candidate_count = len(unique_candidates)
        compliant_candidates = [
            event for event in unique_candidates
            if event.get("time_frame_satisfied", True)
            and not event.get("unsatisfied_constraints")
        ]
        first_valid_route_turn = min(
            (event.get("turn") for event in unique_candidates if event.get("route")),
            default=None,
        )
        first_compliant_route_turn = min(
            (event.get("turn") for event in compliant_candidates),
            default=None,
        )
        constraint_status = result.extra.get("constraint_status", {})
        satisfied_constraints = sum(
            1 for status in constraint_status.values() if status.get("satisfied")
        )
        constraint_satisfaction_rate = safe_ratio(
            satisfied_constraints,
            len(constraint_status),
        ) if constraint_status else (1.0 if not result.extra.get("stated_constraints") else 0.0)
        all_constraints_satisfied = not result.extra.get("unsatisfied_constraints")
        acceptable_duration = bool(
            result.extra.get(
                "time_frame_satisfied",
                result.route_duration_min is not None
                and (
                    reference_duration is None
                    or result.route_duration_min <= reference_duration
                ),
            )
        )
        natural_closure = bool(
            result.conversation
            and result.extra.get("early_stop_reason") == "agent_a_closed"
            and result.route_correct
            and all_constraints_satisfied
        )

        audio_measurements = []
        valid_audio_count = 0
        played_count = 0
        playback_attempt_count = 0
        playback_enabled = bool(result.extra.get("speech_playback_enabled", False))
        for turn in speech_turns:
            audio = turn.get("audio") if isinstance(turn.get("audio"), dict) else {}
            measurement = analyze_wave_file(audio.get("path"))
            if measurement:
                audio_measurements.append(measurement)
                valid_audio_count += 1
            if turn.get("outgoing_enabled") and playback_enabled:
                playback_attempt_count += 1
                if audio.get("played"):
                    played_count += 1
        audio_duration_values = [item["duration_sec"] for item in audio_measurements]
        audio_clipping_rate = mean_or_none([item["clipping_rate"] for item in audio_measurements])
        audio_silence_ratio = mean_or_none([item["silence_ratio"] for item in audio_measurements])
        audio_loudness_values = [item["loudness_dbfs"] for item in audio_measurements]
        loudness_stability = statistics.pstdev(audio_loudness_values) if len(audio_loudness_values) >= 2 else (0.0 if audio_loudness_values else None)
        leading_trailing_silence = mean_or_none([
            item["leading_silence_sec"] + item["trailing_silence_sec"]
            for item in audio_measurements
        ])
        mean_audio_duration = mean_or_none(audio_duration_values)
        raw_turn_values = [
            turn.get("raw_turn_elapsed_sec")
            for turn in timing_turns
            if turn.get("raw_turn_elapsed_sec") is not None
        ]
        raw_turn_latency = mean_or_none(raw_turn_values)
        turn_budget_violation_rate = safe_ratio(
            sum(1 for turn in timing_turns if turn.get("turn_capped")),
            len(timing_turns),
        )
        total_audio_duration = sum(audio_duration_values)
        real_time_interaction_factor = safe_ratio(
            sum(raw_turn_values) if raw_turn_values else result.runtime_sec,
            total_audio_duration,
        ) if total_audio_duration else None

        tts_latency = mean_or_none([turn.get("tts_latency_sec") for turn in speech_turns])
        asr_latency = mean_or_none([turn.get("asr_latency_sec") for turn in speech_turns])
        tts_rtf = mean_or_none([
            safe_ratio(turn.get("tts_latency_sec"), (turn.get("audio") or {}).get("duration_sec"))
            for turn in speech_turns
            if isinstance(turn.get("tts_latency_sec"), (int, float))
            and isinstance(turn.get("audio"), dict)
            and (turn.get("audio") or {}).get("duration_sec")
        ])
        asr_rtf = mean_or_none([
            safe_ratio(turn.get("asr_latency_sec"), (turn.get("audio") or {}).get("duration_sec"))
            for turn in speech_turns
            if isinstance(turn.get("asr_latency_sec"), (int, float))
            and isinstance(turn.get("audio"), dict)
            and (turn.get("audio") or {}).get("duration_sec")
        ])
        expected_audio_durations = [
            turn.get("simulated_duration_sec")
            for turn in speech_turns
            if isinstance(turn.get("simulated_duration_sec"), (int, float))
        ]
        audio_duration_error = mean_or_none([
            actual - expected
            for actual, expected in zip(audio_duration_values, expected_audio_durations)
        ])
        spoken_word_count = sum(
            len(tokenize_words(turn.get("outgoing_text", "")))
            for turn in speech_turns
        )
        speech_rate_wpm = safe_ratio(spoken_word_count * 60.0, total_audio_duration) if total_audio_duration else None
        resolved_audio = result.extra.get("resolved_audio_personas", {})
        speaking_rate_accuracy_values = []
        for turn in speech_turns:
            audio = turn.get("audio") if isinstance(turn.get("audio"), dict) else {}
            duration = audio.get("duration_sec")
            words_in_turn = len(tokenize_words(turn.get("outgoing_text", "")))
            speaker_key = "agent_b" if turn.get("speaker") == "Agent B" else "agent_a"
            target_wpm = (resolved_audio.get(speaker_key) or {}).get("words_per_minute")
            if duration and target_wpm and words_in_turn:
                realized_wpm = words_in_turn * 60.0 / duration
                speaking_rate_accuracy_values.append(
                    min(realized_wpm, target_wpm) / max(realized_wpm, target_wpm)
                )
        speaking_rate_accuracy = mean_or_none(speaking_rate_accuracy_values)
        speech_audio_items = [
            {
                "path": (turn.get("audio") or {}).get("path"),
                "speaker": turn.get("speaker"),
                "turn_index": index,
                "tts_engine": turn.get("tts_engine"),
            }
            for index, turn in enumerate(speech_turns, start=1)
            if isinstance(turn.get("audio"), dict)
        ]
        nisqa_report = self.nisqa_evaluator.evaluate(speech_audio_items)
        dnsmos_report = self.dnsmos_evaluator.evaluate(speech_audio_items)
        result.extra["nisqa_evaluation"] = nisqa_report
        result.extra["dnsmos_evaluation"] = dnsmos_report

        interpreter = NaturalRouteInterpreter()
        agent_b_speech = [turn for turn in speech_turns if turn.get("speaker") == "Agent B"]
        semantic_attempts = 0
        semantic_errors = 0
        route_sequence_errors = []
        origin_destination_scores = []
        station_sequence_matches = []
        semantic_slot_precisions = []
        semantic_slot_recalls = []
        critical_slot_scores = []
        round_trip_route_scores = []
        for index, nlu_turn in enumerate(nlu_turns):
            source_turn = agent_b_speech[index] if index < len(agent_b_speech) else {}
            source_text = source_turn.get("outgoing_text", "")
            reference_parse = interpreter.interpret_reply(source_text, scenario)
            observed_parse = nlu_turn.get("parsed_route") or []
            if source_text:
                semantic_attempts += 1
                semantic_errors += int(list(reference_parse or []) != list(observed_parse))
            if reference_parse:
                route_sequence_errors.append(sequence_error_rate(reference_parse, observed_parse))
                station_sequence_matches.append(float(list(reference_parse) == list(observed_parse)))
                origin_destination_scores.append(
                    (
                        float(bool(observed_parse) and observed_parse[0] == reference_parse[0])
                        + float(bool(observed_parse) and observed_parse[-1] == reference_parse[-1])
                    ) / 2.0
                )
                reference_set = set(reference_parse)
                observed_set = set(observed_parse)
                semantic_slot_precisions.append(safe_ratio(len(reference_set & observed_set), len(observed_set)))
                semantic_slot_recalls.append(safe_ratio(len(reference_set & observed_set), len(reference_set)))
                critical_slot_scores.append(origin_destination_scores[-1])
                round_trip_route_scores.append(float(list(reference_parse) == list(observed_parse)))
        semantic_asr_error_rate = safe_ratio(semantic_errors, semantic_attempts) if semantic_attempts else None
        station_f1 = f1_score(asr_station_precision, asr_station_recall)
        route_sequence_edit_distance = mean_or_none(route_sequence_errors)
        nlu_slot_precision = mean_or_none(semantic_slot_precisions)
        nlu_slot_recall = mean_or_none(semantic_slot_recalls)
        nlu_slot_f1 = (
            f1_score(nlu_slot_precision, nlu_slot_recall)
            if nlu_slot_precision is not None and nlu_slot_recall is not None
            else None
        )
        nlu_joint_frame_accuracy = mean_or_none(station_sequence_matches)
        nlu_origin_destination_accuracy = mean_or_none(origin_destination_scores)

        source_numbers = []
        transcript_numbers = []
        source_negations = []
        transcript_negations = []
        source_constraints = []
        transcript_constraints = []
        for turn in speech_turns:
            source_text = turn.get("outgoing_text", "")
            transcript_text = turn.get("incoming_transcript", "")
            source_numbers.extend(numeric_mentions(source_text))
            transcript_numbers.extend(numeric_mentions(transcript_text))
            source_negations.extend(negation_mentions(source_text))
            transcript_negations.extend(negation_mentions(transcript_text))
            source_constraints.extend(constraint_mentions(source_text))
            transcript_constraints.extend(constraint_mentions(transcript_text))
        numeric_preservation = preservation_rate(source_numbers, transcript_numbers) if source_numbers else None
        negation_preservation = preservation_rate(source_negations, transcript_negations) if source_negations else None
        constraint_preservation = preservation_rate(source_constraints, transcript_constraints) if source_constraints else None

        state_constraint_snapshots = [
            event.get("payload", {}).get("stated_constraints", [])
            for event in result.extra.get("runtime_events", [])
            if event.get("event_type") == "constraint_state"
        ]
        retention_checks = []
        premature_activation = 0
        for previous, current in zip(state_constraint_snapshots, state_constraint_snapshots[1:]):
            retention_checks.append(float(set(previous) <= set(current)))
            premature_activation += int(not set(previous) <= set(current))
        constraint_retention = mean_or_none(retention_checks)
        constraint_omission_rate = (
            1.0 - constraint_retention
            if constraint_retention is not None
            else None
        )
        candidate_routes = [tuple(event.get("route") or ()) for event in candidate_events if event.get("route")]
        candidate_memory_precision = safe_ratio(len(set(candidate_routes)), len(candidate_routes)) if candidate_routes else None
        candidate_memory_recall = 1.0 if candidate_routes and candidate_route_count == len(set(candidate_routes)) else (None if not candidate_routes else 0.0)
        selected_route_consistency = float(tuple(result.route) in set(candidate_routes)) if result.route and candidate_routes else None
        route_state_consistency = float(
            not result.route
            or any(
                tuple(event.get("route") or ()) == tuple(result.route)
                and event.get("duration") == result.route_duration_min
                for event in candidate_events
            )
        )

        clarification_events = agent_a_question_count
        eligible_failures = len(invalid_proposals) + int(result.extra.get("time_frame_miss_count", 0) or 0) + int(result.extra.get("constraint_miss_count", 0) or 0)
        clarification_precision = safe_ratio(min(clarification_events, eligible_failures), clarification_events) if clarification_events else (1.0 if not eligible_failures else 0.0)
        clarification_recall = safe_ratio(min(clarification_events, eligible_failures), eligible_failures) if eligible_failures else 1.0
        route_revisions = int(result.extra.get("route_revisions", 0) or 0)
        repair_attempts = min(route_revisions, eligible_failures)
        successful_revisions = sum(
            1
            for event in unique_candidates
            if event.get("decision") == "improved"
            or (
                event.get("time_frame_satisfied", False)
                and not event.get("unsatisfied_constraints")
            )
        )
        repair_successes = min(repair_attempts, successful_revisions)
        progress_events = sum(
            1 for event in unique_candidates
            if event.get("decision") in {"baseline", "improved"}
            or not event.get("unsatisfied_constraints")
        )
        stagnation_events = len(repeated_candidates) + sum(
            1 for event in unique_candidates if event.get("decision") in {"tied", "slower"}
        )
        early_stop_reason = result.extra.get("early_stop_reason")
        expected_stop = (
            "agent_a_closed" if natural_closure
            else "turn_limit" if len(result.conversation) >= message_count
            else early_stop_reason
        )
        stop_decision_accuracy = float(early_stop_reason == expected_stop) if early_stop_reason else float(result.route_correct)

        hallucinated_content_rate = safe_ratio(len(invalid_proposals), len(nlu_turns))
        hallucinated_station_mentions = sum(
            len(station_mentions_in_text(turn.get("text", "")))
            for turn in invalid_proposals
        )
        all_station_mentions = sum(
            len(station_mentions_in_text(turn.get("text", "")))
            for turn in nlu_turns
        )
        hallucinated_station_rate = safe_ratio(hallucinated_station_mentions, all_station_mentions)
        grounded_components = [
            float(result.route_valid),
            float(result.route_reaches_goal),
            constraint_satisfaction_rate,
            duration_score,
        ]
        grounded_proposal_score = statistics.fmean(grounded_components)
        actionability_score = statistics.fmean([
            float(bool(result.route)),
            float(result.route_reaches_goal),
            float(bool(route_lines)),
            float(result.route_duration_min is not None),
        ])
        candidate_durations = [event.get("duration") for event in unique_candidates if event.get("duration") is not None]
        best_discovered = bool(reference_duration is not None and reference_duration in candidate_durations)
        best_discovery_turn = min(
            (
                event.get("turn")
                for event in unique_candidates
                if event.get("duration") == reference_duration
            ),
            default=None,
        )
        optimality_ratio = (
            safe_ratio(reference_duration, result.route_duration_min)
            if reference_duration is not None and result.route_duration_min
            else None
        )
        duration_regret = (
            result.route_duration_min - reference_duration
            if result.route_duration_min is not None and reference_duration is not None
            else None
        )

        false_acceptance_rate = 1.0 if result.route_valid and not (
            result.route_correct and acceptable_duration and all_constraints_satisfied
        ) else 0.0
        correct_acceptance_rate = float(
            natural_closure and result.route_correct and acceptable_duration and all_constraints_satisfied
        )
        false_rejection_rate = float(
            result.route_correct
            and acceptable_duration
            and all_constraints_satisfied
            and not natural_closure
        )
        verifier_catch_rate = safe_ratio(
            min(warning_count, eligible_failures),
            eligible_failures,
        ) if eligible_failures else 1.0

        agent_b_word_counts = [len(tokenize_words(text)) for text in agent_b_messages]
        executable_utterance_rate = safe_ratio(
            sum(1 for turn in nlu_turns if turn.get("route_valid")),
            len(agent_b_messages),
        )
        nlg_excess_verbosity = safe_ratio(sum(1 for count in agent_b_word_counts if count > 60), len(agent_b_word_counts))
        nlg_underspecification = safe_ratio(
            sum(
                1 for text, turn in zip(agent_b_messages, nlu_turns)
                if len(tokenize_words(text)) < 5 or not turn.get("route_valid")
            ),
            len(agent_b_messages),
        )
        formatting_violations = sum(
            1 for text in agent_b_messages
            if any(marker in text for marker in ("```", "**", "\n-", "\n*"))
        )
        reasoning_leaks = sum(
            1 for text in agent_b_messages
            if any(marker in text.lower() for marker in ("chain of thought", "my reasoning", "step-by-step reasoning"))
        )
        route_order_scores = []
        for text, turn in zip(agent_b_messages, nlu_turns):
            mentions = ordered_station_mentions(text)
            parsed = turn.get("parsed_route") or []
            if mentions and parsed:
                route_order_scores.append(float(mentions == parsed))
        estimated_spoken_duration = safe_ratio(len(agent_b_token_stream) * 60.0, 220.0)

        interaction_progress = []
        for event in candidate_events:
            score = 0.25
            if event.get("route"):
                score += 0.25
            if event.get("time_frame_satisfied"):
                score += 0.25
            if not event.get("unsatisfied_constraints"):
                score += 0.25
            interaction_progress.append(score)
        goal_progress_auc = (
            safe_ratio(sum(interaction_progress), len(interaction_progress))
            if interaction_progress else 0.0
        )
        failure_phase = None
        if pipeline_failure_count:
            failure_phase = "speech_pipeline"
        elif asr_success_rate < 1.0 or asr_word_error_rate > 0:
            failure_phase = "asr"
        elif invalid_proposals:
            failure_phase = "nlu"
        elif result.extra.get("constraint_miss_count"):
            failure_phase = "dialogue_management"
        elif not result.route_correct:
            failure_phase = "agent_b"
        failure_localization_score = 1.0 if failure_phase or result.route_correct else 0.0
        dialogue_cost = (
            message_count
            + 0.1 * len(words)
            + result.runtime_sec
            + 2.0 * warning_count
            + repair_attempts
        )

        configured_tiers = metric_tier_config_with_defaults(
            result.extra.get("metric_tiers") or result.extra.get("metric_config")
        )
        configured_metrics = metric_config_with_defaults(
            result.extra.get("metric_config"),
            configured_tiers,
        )
        research_metric_values = {key: None for key in METRIC_KEYS}
        research_metric_values.update({
            "audio_capture_success_rate": safe_ratio(valid_audio_count, outgoing_enabled_count),
            "audio_missing_rate": 1.0 - safe_ratio(valid_audio_count, outgoing_enabled_count) if outgoing_enabled_count else None,
            "audio_turn_latency": mean_turn_latency_sec,
            "audio_raw_turn_latency": raw_turn_latency,
            "audio_response_latency": agent_b_avg_latency_sec,
            "audio_utterance_duration": mean_audio_duration,
            "audio_speech_rate_wpm": speech_rate_wpm,
            "audio_silence_ratio": audio_silence_ratio,
            "audio_clipping_rate": audio_clipping_rate,
            "audio_loudness_stability": loudness_stability,
            "audio_turn_budget_violation_rate": turn_budget_violation_rate,
            "audio_real_time_interaction_factor": real_time_interaction_factor,
            "asr_success_rate": asr_success_rate,
            "asr_failure_rate": safe_ratio(asr_failure_count, incoming_enabled_count),
            "asr_wer": asr_word_error_rate,
            "asr_character_error_rate": asr_character_error_rate,
            "asr_sentence_error_rate": asr_sentence_error_rate,
            "asr_semantic_asr_error_rate": semantic_asr_error_rate,
            "asr_entity_error_rate": asr_entity_wer,
            "asr_station_precision": asr_station_precision,
            "asr_station_recall": asr_station_recall,
            "asr_station_f1": station_f1,
            "asr_critical_slot_accuracy": mean_or_none(critical_slot_scores),
            "asr_route_sequence_edit_distance": route_sequence_edit_distance,
            "asr_constraint_preservation_rate": constraint_preservation,
            "asr_negation_preservation_rate": negation_preservation,
            "asr_numeric_preservation_rate": numeric_preservation,
            "asr_empty_transcript_rate": safe_ratio(asr_failure_count, incoming_enabled_count),
            "asr_hallucinated_token_rate": safe_ratio(word_insertions, max(ref_word_total + word_insertions - word_deletions, 1)),
            "asr_transcript_correction_count": transcript_correction_count,
            "asr_uncorrected_misinterpretation_count": uncorrected_misinterpretation_count,
            "asr_recognition_latency": asr_latency,
            "asr_real_time_factor": asr_rtf,
            "asr_repair_trigger_rate": safe_ratio(sum(bool((turn.get("diagnostics") or {}).get("asr_repair_used")) for turn in speech_turns), incoming_enabled_count),
            "nlu_intent_accuracy": nlu_route_valid_rate,
            "nlu_slot_precision": nlu_slot_precision,
            "nlu_slot_recall": nlu_slot_recall,
            "nlu_slot_f1": nlu_slot_f1,
            "nlu_joint_frame_accuracy": nlu_joint_frame_accuracy,
            "nlu_constraint_extraction_f1": nlu_slot_f1 if nlu_slot_f1 is not None else slot_realization_accuracy,
            "nlu_semantic_frame_accuracy": nlu_joint_frame_accuracy,
            "nlu_critical_slot_accuracy": nlu_origin_destination_accuracy,
            "nlu_route_valid_rate": nlu_route_valid_rate,
            "nlu_goal_reached_rate": nlu_goal_reached_rate,
            "nlu_station_sequence_exact_match": nlu_joint_frame_accuracy,
            "nlu_station_sequence_edit_distance": route_sequence_edit_distance,
            "nlu_origin_destination_accuracy": nlu_origin_destination_accuracy,
            "nlu_false_parse_rate": safe_ratio(sum(1 for turn in nlu_turns if turn.get("route_valid") and not turn.get("has_station_mentions")), len(nlu_turns)),
            "nlu_unknown_entity_detection_rate": 1.0 - hallucinated_station_rate if all_station_mentions else None,
            "nlu_latency": nlu_latency_sec,
            "nlu_pipeline_input_match_rate": nlu_pipeline_input_match_rate,
            "dialogue_state_joint_goal_accuracy": task_success_rate,
            "dialogue_state_slot_accuracy": nlu_slot_f1,
            "dialogue_state_stage_accuracy": constraint_retention,
            "dialogue_state_state_drift_rate": 1.0 - constraint_retention if constraint_retention is not None else None,
            "dialogue_state_constraint_retention_rate": constraint_retention,
            "dialogue_state_constraint_omission_rate": constraint_omission_rate,
            "dialogue_state_premature_constraint_activation_rate": safe_ratio(premature_activation, max(len(state_constraint_snapshots) - 1, 1)) if state_constraint_snapshots else None,
            "dialogue_state_shared_state_agreement": nlu_pipeline_input_match_rate,
            "dialogue_state_candidate_memory_precision": candidate_memory_precision,
            "dialogue_state_candidate_memory_recall": candidate_memory_recall,
            "dialogue_state_candidate_deduplication_accuracy": 1.0 - safe_ratio(len(repeated_candidates), len(candidate_events)),
            "dialogue_state_selected_route_consistency": selected_route_consistency,
            "dialogue_state_route_state_consistency": route_state_consistency,
            "dialogue_management_correct_next_action_rate": safe_ratio(progress_events, len(candidate_events)) if candidate_events else None,
            "dialogue_management_constraint_order_adherence": constraint_retention,
            "dialogue_management_premature_answer_rate": safe_ratio(len(invalid_proposals), len(nlu_turns)),
            "dialogue_management_premature_closure_rate": float(early_stop_reason == "agent_a_closed" and not natural_closure),
            "dialogue_management_clarification_precision": clarification_precision,
            "dialogue_management_clarification_recall": clarification_recall,
            "dialogue_management_unnecessary_clarification_rate": 1.0 - clarification_precision,
            "dialogue_management_clarification_calibration": 1.0 - abs(clarification_precision - clarification_recall),
            "dialogue_management_repair_attempt_rate": safe_ratio(repair_attempts, eligible_failures),
            "dialogue_management_repair_success_rate": safe_ratio(repair_successes, repair_attempts) if repair_attempts else (1.0 if not eligible_failures else 0.0),
            "dialogue_management_repeated_repair_rate": safe_ratio(
                min(len(repeated_candidates), repair_attempts),
                max(repair_attempts, 1),
            ),
            "dialogue_management_invalid_proposal_handling_accuracy": verifier_catch_rate,
            "dialogue_management_constraint_violation_handling_accuracy": safe_ratio(min(agent_a_question_count, int(result.extra.get("constraint_miss_count", 0) or 0)), int(result.extra.get("constraint_miss_count", 0) or 0)) if result.extra.get("constraint_miss_count") else 1.0,
            "dialogue_management_distinct_proposal_rate": safe_ratio(valid_candidate_count, len(candidate_events)),
            "dialogue_management_route_repetition_rate": safe_ratio(len(repeated_candidates), len(candidate_events)),
            "dialogue_management_route_revision_rate": safe_ratio(route_revision_count, valid_candidate_count),
            "dialogue_management_policy_progress_rate": safe_ratio(progress_events, len(candidate_events)),
            "dialogue_management_stagnation_rate": safe_ratio(stagnation_events, len(candidate_events)),
            "dialogue_management_turn_efficiency": safe_ratio(progress_events, message_count),
            "dialogue_management_stop_decision_accuracy": stop_decision_accuracy,
            "dialogue_management_turn_limit_utilization": safe_ratio(message_count, result.extra.get("configured_num_turns", message_count)),
            "agent_b_proposal_parse_rate": safe_ratio(sum(1 for turn in nlu_turns if turn.get("parsed_route")), len(agent_b_messages)),
            "agent_b_route_validity_rate": nlu_route_valid_rate,
            "agent_b_destination_reach_rate": nlu_goal_reached_rate,
            "agent_b_complete_route_rate": nlu_goal_reached_rate,
            "agent_b_grounded_proposal_score": grounded_proposal_score,
            "agent_b_hallucinated_content_rate": hallucinated_content_rate,
            "agent_b_hallucinated_station_rate": hallucinated_station_rate,
            "agent_b_mode_permission_compliance": float(result.route_valid),
            "agent_b_active_constraint_compliance": constraint_satisfaction_rate,
            "agent_b_actionability_score": actionability_score,
            "agent_b_route_novelty": 1.0 - safe_ratio(len(repeated_candidates), len(candidate_events)),
            "agent_b_pareto_improvement_rate": safe_ratio(sum(1 for event in unique_candidates if event.get("decision") == "improved"), len(unique_candidates)),
            "agent_b_dominated_proposal_rate": safe_ratio(sum(1 for event in unique_candidates if event.get("decision") == "slower"), len(unique_candidates)),
            "agent_b_optimality_ratio": optimality_ratio,
            "agent_b_duration_regret": duration_regret,
            "agent_b_best_route_discovery_rate": float(best_discovered),
            "agent_b_best_route_discovery_turn": best_discovery_turn,
            "agent_b_plugin_execution_success": 0.0 if pipeline_failure else 1.0,
            "agent_b_model_generation_latency": agent_b_generation_latency,
            "agent_a_verifier_catch_rate": verifier_catch_rate,
            "agent_a_constraint_violation_catch_rate": safe_ratio(
                min(agent_a_question_count, int(result.extra.get("constraint_miss_count", 0) or 0)),
                int(result.extra.get("constraint_miss_count", 0) or 0),
            ) if result.extra.get("constraint_miss_count") else 1.0,
            "agent_a_time_limit_violation_catch_rate": safe_ratio(min(agent_a_question_count, int(result.extra.get("time_frame_miss_count", 0) or 0)), int(result.extra.get("time_frame_miss_count", 0) or 0)) if result.extra.get("time_frame_miss_count") else 1.0,
            "agent_a_false_acceptance_rate": false_acceptance_rate,
            "agent_a_false_rejection_rate": false_rejection_rate,
            "agent_a_correct_acceptance_rate": correct_acceptance_rate,
            "agent_a_constraint_revelation_order": constraint_retention,
            "agent_a_preference_consistency": constraint_retention,
            "agent_a_best_candidate_retention": float(not result.route or selected_route_consistency != 0.0),
            "agent_a_selection_regret": duration_regret,
            "agent_a_satisfaction_calibration": float(
                (result.extra.get("conversation_outcome") == "satisfied")
                == bool(result.route_correct and acceptable_duration and all_constraints_satisfied)
            ),
            "agent_a_closure_correctness": float(natural_closure),
            "agent_a_user_effort": agent_a_timing.get("turn_count", 0) + agent_a_question_count,
            "agent_a_caller_latency": agent_a_avg_latency_sec,
            "nlg_semantic_adequacy": slot_realization_accuracy,
            "nlg_faithfulness": grounded_proposal_score,
            "nlg_slot_error_rate": 1.0 - slot_realization_accuracy,
            "nlg_executable_utterance_rate": executable_utterance_rate,
            "nlg_route_mention_completeness": slot_recall,
            "nlg_constraint_mention_precision": constraint_satisfaction_rate if result.extra.get("stated_constraints") else None,
            "nlg_constraint_mention_recall": constraint_satisfaction_rate if result.extra.get("stated_constraints") else None,
            "nlg_information_order_accuracy": mean_or_none(route_order_scores),
            "nlg_conciseness": min(1.0, safe_ratio(20.0, max(avg_words_per_message, 1.0))),
            "nlg_excess_verbosity_rate": nlg_excess_verbosity,
            "nlg_underspecification_rate": nlg_underspecification,
            "nlg_repetition_rate": nlg_repetition_rate,
            "nlg_distinct_1": nlg_distinct_1,
            "nlg_distinct_2": nlg_distinct_2,
            "nlg_lexical_diversity": safe_ratio(len(set(agent_b_token_stream)), len(agent_b_token_stream)),
            "nlg_bleu": nlg_bleu,
            "nlg_rouge_l": simple_rouge_l(reference_delex_tokens, final_delex_tokens),
            "nlg_meteor": nlg_meteor,
            "nlg_semantic_similarity": nlg_semantic_similarity,
            "nlg_constraint_satisfaction_rate": constraint_satisfaction_rate,
            "nlg_formatting_violation_rate": safe_ratio(formatting_violations, len(agent_b_messages)),
            "nlg_hidden_reasoning_leakage_rate": safe_ratio(reasoning_leaks, len(agent_b_messages)),
            "nlg_estimated_spoken_duration": estimated_spoken_duration,
            "tts_success_rate": tts_success_rate,
            "tts_failure_rate": safe_ratio(tts_failure_count, outgoing_enabled_count),
            "tts_audio_validity_rate": safe_ratio(valid_audio_count, outgoing_enabled_count),
            "tts_synthesis_latency": tts_latency,
            "tts_real_time_factor": tts_rtf,
            "tts_audio_duration_error": audio_duration_error,
            "tts_speaking_rate_accuracy": speaking_rate_accuracy,
            "tts_pronunciation_accuracy": 1.0 - asr_word_error_rate if incoming_enabled_count else None,
            "tts_station_pronunciation_accuracy": station_f1 if incoming_enabled_count else None,
            "tts_round_trip_semantic_intelligibility": (
                1.0 - semantic_asr_error_rate
                if semantic_asr_error_rate is not None
                else (0.0 if incoming_enabled_count and not asr_success_count else None)
            ),
            "tts_round_trip_route_accuracy": mean_or_none(round_trip_route_scores),
            "tts_loudness_compliance": mean_or_none([
                float(-35.0 <= value <= -3.0) for value in audio_loudness_values
            ]),
            "tts_clipping_rate": audio_clipping_rate,
            "tts_leading_trailing_silence": leading_trailing_silence,
            "tts_dnsmos": dnsmos_report.get("score"),
            "tts_nisqa": nisqa_report.get("score"),
            "tts_playback_success_rate": safe_ratio(played_count, playback_attempt_count) if playback_attempt_count else None,
            "tts_text_change_rate": tts_text_change_rate,
            "task_outcome_completion": task_success_rate,
            "task_outcome_route_validity": float(result.route_valid),
            "task_outcome_acceptable_duration_completion": float(result.route_correct and acceptable_duration),
            "task_outcome_constraint_satisfaction": float(all_constraints_satisfied),
            "task_outcome_constraint_satisfaction_rate": constraint_satisfaction_rate,
            "task_outcome_stage_completion_rate": statistics.fmean([float(result.route_correct), float(acceptable_duration), float(all_constraints_satisfied)]),
            "task_outcome_duration_quality": duration_score,
            "task_outcome_duration_ratio": safe_ratio(result.route_duration_min, reference_duration) if result.route_duration_min is not None and reference_duration else None,
            "task_outcome_duration_regret": duration_regret,
            "task_outcome_correct_route_selection": float(result.route_correct and all_constraints_satisfied),
            "task_outcome_candidate_count": candidate_route_count,
            "task_outcome_turns_used": message_count,
            "task_outcome_turns_to_success": first_compliant_route_turn,
            "task_outcome_first_valid_route_turn": first_valid_route_turn,
            "task_outcome_first_compliant_route_turn": first_compliant_route_turn,
            "task_outcome_successful_natural_closure": float(natural_closure),
            "whole_dialogue_dialogue_success_score": automatic_eval_score,
            "whole_dialogue_interaction_quality_trajectory": automatic_eval_score,
            "whole_dialogue_goal_progress_auc": goal_progress_auc,
            "whole_dialogue_dialogue_cost": message_count,
            "whole_dialogue_turn_count": message_count,
            "whole_dialogue_word_count": len(words),
            "whole_dialogue_mean_words_per_turn": avg_words_per_message,
            "whole_dialogue_total_runtime": result.runtime_sec,
            "whole_dialogue_candidate_count": candidate_route_count,
            "whole_dialogue_route_revision_count": route_revision_count,
            "whole_dialogue_clarification_count": agent_a_question_count,
            "whole_dialogue_repair_count": repair_attempts,
            "whole_dialogue_warning_count": warning_count,
            "whole_dialogue_abandonment_rate": abandonment_rate,
            "whole_dialogue_failure_localization_score": failure_localization_score,
            "whole_dialogue_failure_phase": failure_phase,
            "whole_dialogue_pipeline_dependency_integrity": nlu_pipeline_input_match_rate,
            "whole_dialogue_cooperative_progress_rate": safe_ratio(progress_events, message_count),
            "whole_dialogue_task_focus_score": task_focus_score,
            "whole_dialogue_conversation_repetition_rate": nlg_repetition_rate,
            "whole_dialogue_natural_closure_rate": float(natural_closure),
            "whole_dialogue_resource_cost": dialogue_cost,
        })
        research_metric_values = {
            key: round(value, 4) if isinstance(value, float) and math.isfinite(value) else value
            for key, value in research_metric_values.items()
        }
        metric_families = {}
        for family in enabled_metric_family_specs(configured_metrics, configured_tiers):
            current_phase = phase_key(family)
            values = {
                metric_local_name(key): research_metric_values.get(key)
                for key, _label in family["metrics"]
            }
            available_count = sum(value is not None for value in values.values())
            metric_families[current_phase] = {
                "available": available_count > 0,
                "coverage_rate": round(safe_ratio(available_count, len(values)), 4),
                "available_metric_count": available_count,
                "configured_metric_count": len(values),
                **values,
            }

        evidence_groups = {
            "audio_input": {
                "speech_turns": len(speech_turns),
                "synthesis_attempts": outgoing_enabled_count,
                "valid_audio_files": valid_audio_count,
            },
            "asr": {
                "reference_words": ref_word_total,
                "substitutions": word_substitutions,
                "deletions": word_deletions,
                "insertions": word_insertions,
                "recognition_attempts": incoming_enabled_count,
                "successful_recognitions": asr_success_count,
            },
            "nlu": {
                "nlu_turns": len(nlu_turns),
                "route_valid_turns": sum(bool(turn.get("route_valid")) for turn in nlu_turns),
                "goal_reached_turns": sum(bool(turn.get("route_reaches_goal")) for turn in nlu_turns),
            },
            "dialogue_state_tracking": {
                "messages": message_count,
                "candidate_routes": candidate_route_count,
                "runtime_events": len(result.extra.get("runtime_events", [])),
            },
            "dialogue_management": {
                "messages": message_count,
                "repair_attempts": repair_attempts,
                "progress_events": progress_events,
            },
            "backend_task_execution": {
                "candidate_routes": candidate_route_count,
                "route_valid": bool(result.route_valid),
                "route_reaches_goal": bool(result.route_reaches_goal),
            },
            "nlg": {
                "agent_b_messages": len(agent_b_messages),
                "agent_b_words": len(agent_b_token_stream),
                "candidate_routes": candidate_route_count,
            },
            "tts": {
                "synthesis_attempts": outgoing_enabled_count,
                "successful_syntheses": tts_success_count,
                "valid_audio_files": valid_audio_count,
                "reference_words": ref_word_total,
            },
            "task_outcome": {
                "route_valid": bool(result.route_valid),
                "route_correct": bool(result.route_correct),
                "selected_duration_min": result.route_duration_min,
                "reference_duration_min": reference_duration,
                "stated_constraints": len(result.extra.get("stated_constraints", [])),
                "unsatisfied_constraints": len(result.extra.get("unsatisfied_constraints", [])),
            },
            "whole_dialogue": {
                "messages": message_count,
                "words": len(words),
                "runtime_seconds": round(result.runtime_sec, 4),
                "repair_attempts": repair_attempts,
            },
            "metric_validity": {
                "configured_metrics": sum(bool(value) for value in configured_metrics.values()),
                "calculated_metrics": sum(value is not None for value in research_metric_values.values()),
            },
        }
        metric_calculations = {}
        for family in enabled_metric_family_specs(configured_metrics, configured_tiers):
            current_phase = phase_key(family)
            operands = evidence_groups.get(current_phase, {"messages": message_count})
            for key, _label in family["metrics"]:
                value = research_metric_values.get(key)
                formula = metric_calculation_method(key)
                substitution = f"{formula} with recorded operands = {value}"
                if key == "asr_wer":
                    substitution = (
                        f"({word_substitutions} + {word_deletions} + {word_insertions}) "
                        f"/ {ref_word_total} = {value}"
                    )
                elif key == "asr_deletion_rate":
                    substitution = f"{word_deletions} / {ref_word_total} = {value}"
                elif key == "asr_substitution_rate":
                    substitution = f"{word_substitutions} / {ref_word_total} = {value}"
                elif key == "asr_insertion_rate":
                    substitution = f"{word_insertions} / {ref_word_total} = {value}"
                elif key == "tts_success_rate":
                    substitution = f"{tts_success_count} / {outgoing_enabled_count} = {value}"
                elif key == "tts_failure_rate":
                    substitution = f"{tts_failure_count} / {outgoing_enabled_count} = {value}"
                elif key == "tts_audio_validity_rate":
                    substitution = f"{valid_audio_count} / {outgoing_enabled_count} = {value}"
                elif key == "task_outcome_duration_quality":
                    substitution = (
                        f"min({reference_duration} / {result.route_duration_min}, 1) = {value}"
                    )
                metric_calculations[key] = {
                    "formula": formula,
                    "operands": dict(operands),
                    "substitution": substitution,
                    "result": value,
                }
        result.extra["metric_input_inventory"] = {
            "captured_after_dialogue": True,
            "groups": evidence_groups,
            "configured_metric_count": sum(bool(value) for value in configured_metrics.values()),
            "calculation_evidence_count": len(metric_calculations),
        }

        return MetricRecord(
            condition_id=result.condition_id,
            test_case_key=result.test_case_key,
            persona_key=result.persona_key,
            scenario_key=result.scenario_key,
            speech_pattern_key=result.speech_pattern_key,
            agent_a_audio_persona=result.extra.get("agent_a_audio_persona", "unknown"),
            agent_b_audio_persona=result.extra.get("agent_b_audio_persona", "unknown"),
            model_name=result.model_name,
            model_param_key=result.extra.get("model_param_key", "default"),
            model_do_sample=bool(model_parameters.get("do_sample", False)),
            model_temperature=model_parameters.get("temperature"),
            model_top_p=model_parameters.get("top_p"),
            success=result.route_correct,
            conversation_outcome=result.extra.get("conversation_outcome", "unknown"),
            stated_constraints=", ".join(result.extra.get("stated_constraints", [])) if result.extra.get("stated_constraints") else "None",
            unsatisfied_constraints=", ".join(result.extra.get("unsatisfied_constraints", [])) if result.extra.get("unsatisfied_constraints") else "None",
            route_valid=result.route_valid,
            route_reaches_goal=result.route_reaches_goal,
            route_duration_min=result.route_duration_min,
            reference_duration_min=reference_duration,
            constraint_duration_min=constraint_duration,
            route_line_sequence=" to ".join(route_lines) if route_lines else "None",
            reference_line_sequence=" to ".join(reference_lines) if reference_lines else "None",
            constraint_line_sequence=" to ".join(constraint_lines) if constraint_lines else "None",
            route_line_change_count=route_line_changes,
            reference_line_change_count=reference_line_changes,
            constraint_line_change_count=constraint_line_changes,
            duration_excess_min=duration_excess,
            constraint_duration_gap_min=constraint_duration_gap,
            constraint_line_change_gap=constraint_line_change_gap,
            constraint_fullness_gap=constraint_fullness_gap,
            constraint_near_capacity_gap=constraint_near_capacity_gap,
            route_near_capacity=route_near_capacity,
            route_near_capacity_count=route_near_capacity_count,
            constraint_near_capacity=constraint_near_capacity,
            constraint_near_capacity_count=constraint_near_capacity_count,
            transfer_tolerance=transfer_tolerance,
            allowed_modes=", ".join(allowed_lines) if allowed_lines else "all transport modes",
            route_transfer_miss_probability=route_transfer_miss_probability,
            constraint_transfer_miss_probability=constraint_transfer_miss_probability,
            constraint_transfer_miss_probability_gap=constraint_transfer_miss_probability_gap,
            route_delay_probability=route_delay_probability,
            constraint_delay_probability=constraint_delay_probability,
            constraint_delay_probability_gap=constraint_delay_probability_gap,
            travel_min=breakdown["travel"],
            wait_min=breakdown["wait"],
            transfer_min=breakdown["transfer"],
            transfer_count=transfer_count,
            average_route_fullness=average_route_fullness,
            peak_route_fullness=peak_route_fullness,
            runtime_sec=result.runtime_sec,
            condition_runtime_sec=result.extra.get("condition_runtime_sec"),
            speech_duration_total_sec=round(speech_duration_total_sec, 4),
            max_turn_latency_sec=round(max_turn_latency_sec, 4),
            mean_turn_elapsed_sec=round(mean_turn_elapsed_sec, 4),
            max_turn_elapsed_sec=round(max_turn_elapsed_sec, 4),
            message_count=message_count,
            word_count=len(words),
            station_mentions=station_mentions,
            task_focus_score=round(task_focus_score, 4),
            comparison_terms=comparison_terms,
            cooperation_terms=cooperation_terms,
            agent_a_question_count=agent_a_question_count,
            question_count=question_count,
            clarification_rate=round(clarification_rate, 4),
            avg_words_per_message=round(avg_words_per_message, 2),
            lexical_diversity=round(lexical_diversity, 4),
            agent_b_distinct_1=round(agent_b_distinct_1, 4),
            candidate_route_count=result.extra.get("candidate_routes", 0),
            route_revision_count=result.extra.get("route_revisions", 0),
            best_candidate_turn=result.extra.get("best_candidate_turn"),
            pipeline_mode=pipeline_mode,
            pipeline_success_rate=round(pipeline_success_rate, 4),
            pipeline_failure_count=pipeline_failure_count,
            tts_success_rate=round(tts_success_rate, 4),
            tts_failure_count=tts_failure_count,
            asr_success_rate=round(asr_success_rate, 4),
            asr_failure_count=asr_failure_count,
            nlu_pipeline_input_match_rate=round(nlu_pipeline_input_match_rate, 4),
            asr_word_error_rate=round(asr_word_error_rate, 4),
            asr_sentence_error_rate=round(asr_sentence_error_rate, 4),
            asr_station_precision=round(asr_station_precision, 4),
            asr_station_recall=round(asr_station_recall, 4),
            tts_text_change_rate=round(tts_text_change_rate, 4),
            speech_incoming_enabled_rate=round(speech_incoming_enabled_rate, 4),
            speech_outgoing_enabled_rate=round(speech_outgoing_enabled_rate, 4),
            nlu_route_valid_rate=round(nlu_route_valid_rate, 4),
            nlu_goal_reached_rate=round(nlu_goal_reached_rate, 4),
            nlu_station_mention_rate=round(nlu_station_mention_rate, 4),
            agent_a_avg_latency_sec=round(agent_a_avg_latency_sec, 4),
            agent_b_avg_latency_sec=round(agent_b_avg_latency_sec, 4),
            mean_turn_latency_sec=round(mean_turn_latency_sec, 4),
            duration_score=round(duration_score, 4),
            quality_score=round(quality_score, 4),
            automatic_eval_score=round(automatic_eval_score, 4),
            metric_families=metric_families,
            metric_calculations=metric_calculations,
        )


def _record_group_key(record):
    """Return the condition identity without the iteration suffix."""
    condition = str(record.condition_id)
    head, separator, tail = condition.rpartition("__")
    return head if separator and tail.isdigit() else condition


def _group_success_gap(records, attribute):
    groups = {}
    for record in records:
        groups.setdefault(getattr(record, attribute, "unknown"), []).append(float(record.success))
    means = [statistics.fmean(values) for values in groups.values() if values]
    return max(means) - min(means) if len(means) >= 2 else None


def _average_metric_outcome_correlation(records):
    outcomes = [float(record.success) for record in records]
    correlations = []
    for phase in records[0].metric_families:
        metric_names = {
            name
            for record in records
            for name in record.metric_families.get(phase, {})
            if name not in {
                "available",
                "coverage_rate",
                "available_metric_count",
                "configured_metric_count",
            }
        }
        for name in metric_names:
            values = [record.metric_families.get(phase, {}).get(name) for record in records]
            correlation = pearson_correlation(values, outcomes)
            if correlation is not None:
                correlations.append(abs(correlation))
    return statistics.fmean(correlations) if correlations else None


def _maximum_metric_redundancy(records):
    series = []
    for phase in records[0].metric_families:
        for name in records[0].metric_families.get(phase, {}):
            if name in {
                "available",
                "coverage_rate",
                "available_metric_count",
                "configured_metric_count",
            }:
                continue
            values = [record.metric_families.get(phase, {}).get(name) for record in records]
            if sum(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values) >= 2:
                series.append(values)
    correlations = []
    for index, left in enumerate(series):
        for right in series[index + 1:]:
            correlation = pearson_correlation(left, right)
            if correlation is not None and abs(correlation) < 0.999999:
                correlations.append(abs(correlation))
    return max(correlations) if correlations else None


def apply_cross_run_metrics(records):
    """Populate metrics that require a completed multi-run experiment."""
    records = list(records)
    if not records:
        return records

    successes = sum(bool(record.success) for record in records)
    ci_low, ci_high = wilson_interval(successes, len(records))
    seed_variances = []
    retest_agreements = []
    grouped = {}
    for record in records:
        grouped.setdefault(_record_group_key(record), []).append(record)
    for group in grouped.values():
        if len(group) < 2:
            continue
        variance = variance_or_none([record.automatic_eval_score for record in group])
        if variance is not None:
            seed_variances.append(variance)
        outcomes = [bool(record.success) for record in group]
        majority = max(sum(outcomes), len(outcomes) - sum(outcomes))
        retest_agreements.append(majority / len(outcomes))

    success_by_pattern = {}
    for record in records:
        success_by_pattern.setdefault(record.speech_pattern_key, []).append(float(record.success))
    pattern_means = {
        key: statistics.fmean(values)
        for key, values in success_by_pattern.items()
        if values
    }
    clean_score = pattern_means.get("clean")
    non_clean_scores = [value for key, value in pattern_means.items() if key != "clean"]
    perturbation_sensitivity = (
        clean_score - statistics.fmean(non_clean_scores)
        if clean_score is not None and non_clean_scores
        else None
    )

    all_values = [
        value
        for record in records
        for phase, metrics in record.metric_families.items()
        if phase != "metric_validity"
        for name, value in metrics.items()
        if name not in {
            "available",
            "coverage_rate",
            "available_metric_count",
            "configured_metric_count",
        }
    ]
    missingness = safe_ratio(sum(value is None for value in all_values), len(all_values))
    numeric_values = [
        float(value)
        for value in all_values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    ceiling_rate = safe_ratio(sum(value == 1.0 for value in numeric_values), len(numeric_values))
    floor_rate = safe_ratio(sum(value == 0.0 for value in numeric_values), len(numeric_values))
    correlation = _average_metric_outcome_correlation(records)
    redundancy = _maximum_metric_redundancy(records)

    values = {
        "success_confidence_interval_low": ci_low,
        "success_confidence_interval_high": ci_high,
        "metric_outcome_correlation": correlation,
        "rank_stability": None,
        "seed_variance": statistics.fmean(seed_variances) if seed_variances else None,
        "test_retest_agreement": statistics.fmean(retest_agreements) if retest_agreements else None,
        "missingness_rate": missingness,
        "ceiling_rate": ceiling_rate,
        "floor_rate": floor_rate,
        "metric_redundancy": redundancy,
        "perturbation_sensitivity": perturbation_sensitivity,
        "persona_robustness": _group_success_gap(records, "persona_key"),
        "scenario_robustness": _group_success_gap(records, "scenario_key"),
        "speech_pattern_robustness": _group_success_gap(records, "speech_pattern_key"),
        "provider_robustness": _group_success_gap(records, "model_name"),
        "subgroup_performance_gap": _group_success_gap(records, "persona_key"),
    }
    for record in records:
        family = record.metric_families.setdefault("metric_validity", {})
        family.update({
            key: round(value, 4) if isinstance(value, float) and math.isfinite(value) else value
            for key, value in values.items()
        })
        metric_values = [
            value
            for key, value in family.items()
            if key not in {
                "available",
                "coverage_rate",
                "available_metric_count",
                "configured_metric_count",
            }
        ]
        available_count = sum(value is not None for value in metric_values)
        family["available"] = available_count > 0
        family["coverage_rate"] = round(safe_ratio(available_count, len(metric_values)), 4)
        family["available_metric_count"] = available_count
        family["configured_metric_count"] = len(metric_values)
    return records


def apply_paired_run_metrics(records):
    """Attach audio-minus-text deltas to matched experimental controls."""
    records = list(records)
    grouped = {}
    for record in records:
        if record.pair_id:
            grouped.setdefault(record.pair_id, {})[record.run_type] = record
    for runs in grouped.values():
        text = runs.get("text_only")
        audio = runs.get("audio_variant")
        if text is None or audio is None:
            continue

        def constraint_satisfaction(record):
            return float(not bool(str(record.unsatisfied_constraints).strip()))

        def repairs(record):
            value = record.metric_families.get("whole_dialogue", {}).get("repair_count")
            return float(value or 0)

        deltas = {
            "paired_task_success_delta": float(audio.success) - float(text.success),
            "paired_route_validity_delta": float(audio.route_valid) - float(text.route_valid),
            "paired_constraint_satisfaction_delta": constraint_satisfaction(audio) - constraint_satisfaction(text),
            "paired_turn_count_delta": float(audio.message_count - text.message_count),
            "paired_repair_turn_delta": repairs(audio) - repairs(text),
            "paired_audio_error_effect": float(audio.asr_word_error_rate - text.asr_word_error_rate),
        }
        for record in (text, audio):
            for key, value in deltas.items():
                setattr(record, key, round(value, 4))
    return records
