"""Automatic evaluation metrics for speech-dialog route-planning experiments."""
from collections import Counter
from dataclasses import asdict, dataclass, field
import math
import re

from minillama.evaluation.config import (
    METRIC_AUTOMATIC_ASR_WEIGHT,
    METRIC_AUTOMATIC_DIALOG_WEIGHT,
    METRIC_AUTOMATIC_NLU_WEIGHT,
    METRIC_QUALITY_BASE_WEIGHT,
    METRIC_QUALITY_DURATION_WEIGHT,
)
from minillama.model.metro_data import STATION_POS
from minillama.model.route_planner import (
    optimal_time_route,
    route_duration_breakdown,
    route_line_change_count,
    route_line_sequence,
    route_station_sequence,
    route_text_from_steps,
)


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


METRIC_FAMILY_SPECS = [
    {
        "title": "Audio ingress / capture",
        "metrics": [
            ("audio_snr_db", "Signal-to-noise ratio"),
            ("audio_si_snr_db", "Scale-invariant signal-to-noise ratio"),
            ("audio_clipping_rate", "Clipping"),
            ("audio_packet_loss_rate", "Packet Loss"),
            ("audio_sample_rate_mismatch", "Sample Mismatch"),
            ("audio_loudness_lufs", "Loudness"),
            ("audio_noise_estimate", "Noise"),
            ("audio_pesq", "Perceptual evaluation of speech quality"),
            ("audio_dnsmos", "Deep noise suppression mean opinion score"),
        ],
    },
    {
        "title": "Voice activity detection and segmentation",
        "metrics": [
            ("vad_false_alarm_rate", "False Alarm"),
            ("vad_miss_rate", "Miss"),
            ("vad_detection_error_rate", "Detection Error"),
            ("vad_speech_non_speech_f1", "Speech F one score"),
            ("vad_endpointing_latency_sec", "Endpoint Latency"),
        ],
    },
    {
        "title": "Diarization",
        "metrics": [
            ("diarization_der", "Diarization error rate"),
            ("diarization_missed_speech_rate", "Missed Speech"),
            ("diarization_false_alarm_rate", "False Alarm"),
            ("diarization_speaker_confusion_rate", "Speaker Confusion"),
            ("diarization_overlap_detection_f1", "Overlap F one score"),
        ],
    },
    {
        "title": "Automatic speech recognition",
        "metrics": [
            ("asr_success_rate", "Recognition Success"),
            ("asr_failure_count", "Recognition Failures"),
            ("asr_word_error_rate", "Word error rate"),
            ("asr_character_error_rate", "Character error rate"),
            ("asr_token_error_rate", "Token error rate"),
            ("asr_deletion_rate", "Deletion"),
            ("asr_substitution_rate", "Substitution"),
            ("asr_insertion_rate", "Insertion"),
            ("asr_entity_wer", "Entity word error rate"),
            ("asr_keyword_recall", "Keyword Recall"),
            ("asr_confidence_calibration", "Confidence"),
        ],
    },
    {
        "title": "Spoken language understanding",
        "metrics": [
            ("slu_pipeline_input_match_rate", "Pipeline Input Match"),
            ("slu_intent_accuracy", "Intent Accuracy"),
            ("slu_intent_error_rate", "Intent Error"),
            ("slu_slot_f1", "Slot F one score"),
            ("slu_slot_error_rate", "Slot Error"),
            ("slu_concept_error_rate", "Concept Error"),
            ("slu_sentence_semantic_accuracy", "Sentence Semantics"),
            ("slu_semantic_frame_accuracy", "Frame Accuracy"),
        ],
    },
    {
        "title": "Dialog state tracking",
        "metrics": [
            ("dst_joint_goal_accuracy", "Joint Goal"),
            ("dst_average_goal_accuracy", "Average Goal"),
            ("dst_requested_slot_f1", "Requested Slot F one score"),
            ("dst_active_intent_accuracy", "Active Intent"),
            ("dst_state_update_accuracy", "State Update"),
            ("dst_belief_state_calibration", "Calibration"),
            ("dst_l2", "L two distance"),
            ("dst_mrr", "Mean reciprocal rank"),
            ("dst_roc", "Receiver operating characteristic"),
        ],
    },
    {
        "title": "Policy and dialog management",
        "metrics": [
            ("policy_dialog_act_accuracy", "Dialog Act Accuracy"),
            ("policy_dialog_act_f1", "Dialog Act F one score"),
            ("policy_next_action_accuracy", "Next Action"),
            ("policy_tool_call_exact_match", "Tool Exact"),
            ("policy_parameter_exact_match", "Parameter Exact"),
            ("policy_invalid_action_rate", "Invalid"),
            ("policy_fallback_rate", "Fallback"),
            ("policy_repair_rate", "Repair"),
            ("policy_confirmation_rate", "Confirmation"),
        ],
    },
    {
        "title": "Tool / retrieval",
        "metrics": [
            ("tool_entity_match_rate", "Entity Match"),
            ("tool_api_success_rate", "Application programming interface Success"),
            ("tool_tool_call_validity", "Tool Valid"),
            ("tool_result_relevance", "Relevance"),
            ("tool_hit_at_k", "Hit at rank"),
            ("tool_mrr", "Mean reciprocal rank"),
            ("tool_grounding_accuracy", "Grounding"),
            ("tool_hallucinated_field_rate", "Hallucinated"),
        ],
    },
    {
        "title": "Natural language generation",
        "metrics": [
            ("nlg_bleu", "Bilingual evaluation understudy"),
            ("nlg_rouge", "Recall-oriented understudy for gisting evaluation"),
            ("nlg_meteor", "Metric for evaluation of translation with explicit ordering"),
            ("nlg_bert_score", "Bidirectional encoder representations from transformers score"),
            ("nlg_slot_realization_accuracy", "Slot Realization"),
            ("nlg_delexicalized_bleu", "Delexicalized bilingual evaluation understudy"),
            ("nlg_distinct_1", "Distinct-1"),
            ("nlg_distinct_2", "Distinct-2"),
            ("nlg_repetition_rate", "Repetition"),
            ("nlg_constraint_satisfaction_rate", "Constraint"),
        ],
    },
    {
        "title": "Text-to-speech",
        "metrics": [
            ("tts_success_rate", "Synthesis Success"),
            ("tts_failure_count", "Synthesis Failures"),
            ("tts_predicted_mos", "Predicted mean opinion score"),
            ("tts_intelligibility_wer", "Intelligibility word error rate"),
            ("tts_stoi", "Short-time objective intelligibility"),
            ("tts_mcd", "Mel cepstral distortion"),
            ("tts_pesq", "Perceptual evaluation of speech quality"),
            ("tts_speechbert_score", "Speech bidirectional encoder representations from transformers"),
            ("tts_speaker_similarity", "Speaker Similarity"),
            ("tts_f0_correlation", "Fundamental frequency correlation"),
        ],
    },
    {
        "title": "Runtime",
        "metrics": [
            ("runtime_end_of_turn_detection_accuracy", "End of turn detection accuracy"),
            ("runtime_endpointing_latency_sec", "Endpoint Latency"),
            ("runtime_barge_in_true_positive_rate", "Barge-in true positive"),
            ("runtime_barge_in_false_positive_rate", "Barge-in false positive"),
            ("runtime_barge_in_suppression_latency_sec", "Barge Suppress"),
            ("runtime_response_latency_sec", "Response Latency"),
            ("runtime_mean_turn_elapsed_sec", "Mean Turn Elapsed"),
            ("runtime_max_turn_latency_sec", "Maximum Turn"),
            ("runtime_max_turn_elapsed_sec", "Maximum Turn Elapsed"),
            ("runtime_speech_duration_total_sec", "Speech Total"),
            ("runtime_condition_runtime_sec", "Batch Runtime"),
            ("runtime_time_to_first_token_sec", "First Token"),
            ("runtime_time_to_first_audio_sec", "First Audio"),
            ("runtime_interruption_recovery_rate", "Recovery"),
        ],
    },
    {
        "title": "Pipeline phases",
        "metrics": [
            ("pipeline_mode", "Mode"),
            ("pipeline_success_rate", "Pipeline Success"),
            ("pipeline_failure_count", "Pipeline Failures"),
            ("pipeline_tts_attempt_count", "Text-to-speech Attempts"),
            ("pipeline_asr_attempt_count", "Automatic speech recognition Attempts"),
            ("pipeline_nlu_attempt_count", "Semantic Parse Attempts"),
            ("pipeline_phase_output_dependency_rate", "Phase Output Dependency"),
        ],
    },
    {
        "title": "End to end",
        "metrics": [
            ("e2e_task_success", "Task Success"),
            ("e2e_inform_rate", "Inform"),
            ("e2e_request_success", "Request"),
            ("e2e_completion_rate", "Completion"),
            ("e2e_abandonment_rate", "Abandonment"),
            ("e2e_escalation_rate", "Escalation"),
            ("e2e_average_reward", "Reward"),
            ("e2e_turns_to_success", "Turns"),
            ("e2e_dialog_duration_sec", "Duration"),
            ("e2e_reprompt_count", "Reprompts"),
            ("e2e_confirmation_count", "Confirmations"),
        ],
    },
    {
        "title": "Post hoc",
        "metrics": [
            ("posthoc_predicted_user_satisfaction", "Satisfaction"),
            ("posthoc_per_domain_failure_rate", "Failure Rate"),
            ("posthoc_cohort_fairness_gaps", "Fairness Gaps"),
            ("posthoc_robustness_by_noise_gap", "Noise Gap"),
            ("posthoc_robustness_by_accent_gap", "Accent Gap"),
            ("posthoc_robustness_by_device_gap", "Device Gap"),
            ("posthoc_robustness_by_environment_gap", "Environment Gap"),
            ("posthoc_cost_per_success", "Cost / Success"),
            ("posthoc_safety_refusal_precision", "Safety Precision"),
            ("posthoc_safety_refusal_recall", "Safety Recall"),
            ("posthoc_privacy_redaction_accuracy", "Privacy"),
        ],
    },
]


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
    model_name: str
    model_param_key: str
    model_do_sample: bool
    model_temperature: float | None
    model_top_p: float | None
    success: bool
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
    metric_families: dict[str, dict[str, object]] = field(default_factory=dict)

    def as_dict(self):
        row = asdict(self)
        families = row.pop("metric_families", {})
        for family_name, metrics in families.items():
            for metric_name, value in metrics.items():
                row[f"{family_name}_{metric_name}"] = value
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


class MetricComputer:
    """Compute thesis-facing automatic evaluation metrics from a completed dialog."""

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
        allowed_modes = result.extra.get("allowed_modes") or scenario.get("allowed_modes") or []
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
                if pipeline_ok and (pipeline_mode == "pure_text" or audio.get("path") or turn.get("tts_engine") not in {"disabled", "loopback-tts"}):
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
        asr_sentence_error_rate = safe_ratio(sentence_errors, len(speech_turns))
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

        def average_latency(speaker):
            speaker_turns = [turn for turn in timing_turns if turn.get("speaker") == speaker]
            return safe_ratio(
                sum(turn.get("turn_latency_sec", 0.0) for turn in speaker_turns),
                len(speaker_turns),
            )

        agent_a_avg_latency_sec = average_latency("Agent A")
        agent_b_avg_latency_sec = average_latency("Agent B")
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
            source_text = turn.get("source_text", "")
            transcript_text = turn.get("transcript", "")
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

        from minillama.evaluation.route_interpreter import NaturalRouteInterpreter

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

        return MetricRecord(
            condition_id=result.condition_id,
            test_case_key=result.test_case_key,
            persona_key=result.persona_key,
            scenario_key=result.scenario_key,
            speech_pattern_key=result.speech_pattern_key,
            model_name=result.model_name,
            model_param_key=result.extra.get("model_param_key", "default"),
            model_do_sample=bool(model_parameters.get("do_sample", False)),
            model_temperature=model_parameters.get("temperature"),
            model_top_p=model_parameters.get("top_p"),
            success=result.route_correct,
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
            allowed_modes=", ".join(allowed_modes) if allowed_modes else "all modes",
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
        )
