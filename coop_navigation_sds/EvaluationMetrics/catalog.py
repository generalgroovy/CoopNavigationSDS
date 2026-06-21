"""Canonical selectable metric catalog in dialogue-system execution order."""


METRIC_FAMILY_SPECS = [
    {
        "key": "user_simulation",
        "order": 0,
        "title": "0. User Simulation and Input",
        "description": "Caller behavior, request formulation, critique, acceptance, and closure.",
        "metrics": [
            ("agent_a_verifier_catch_rate", "Invalid-Route Catch Rate"),
            ("agent_a_constraint_violation_catch_rate", "Constraint-Violation Catch Rate"),
            ("agent_a_time_limit_violation_catch_rate", "Time-Limit Violation Catch Rate"),
            ("agent_a_false_acceptance_rate", "False Acceptance Rate"),
            ("agent_a_false_rejection_rate", "False Rejection Rate"),
            ("agent_a_correct_acceptance_rate", "Correct Acceptance Rate"),
            ("agent_a_critique_specificity", "Critique Specificity"),
            ("agent_a_critique_factuality", "Critique Factuality"),
            ("agent_a_constraint_revelation_accuracy", "Constraint Revelation Accuracy"),
            ("agent_a_constraint_revelation_order", "Constraint Revelation Order"),
            ("agent_a_preference_consistency", "Preference Consistency"),
            ("agent_a_best_candidate_retention", "Best-Candidate Retention"),
            ("agent_a_selection_regret", "Selection Regret"),
            ("agent_a_satisfaction_calibration", "Satisfaction Calibration"),
            ("agent_a_closure_correctness", "Closure Correctness"),
            ("agent_a_user_effort", "User Effort"),
            ("agent_a_caller_latency", "Caller Latency"),
        ],
    },
    {
        "key": "audio_input",
        "order": 1,
        "title": "1. Audio Input and Turn-Taking",
        "description": "Audio capture, endpointing, pacing, overlap, and interaction timing.",
        "metrics": [
            ("audio_capture_success_rate", "Audio Capture Success Rate"),
            ("audio_missing_rate", "Missing Audio Rate"),
            ("audio_turn_latency", "Turn Latency"),
            ("audio_raw_turn_latency", "Raw Turn Latency"),
            ("audio_response_latency", "Response Latency"),
            ("audio_endpointing_latency", "Endpointing Latency"),
            ("audio_early_endpoint_rate", "Early Endpoint Rate"),
            ("audio_late_endpoint_rate", "Late Endpoint Rate"),
            ("audio_end_of_utterance_error", "End-of-Utterance Error"),
            ("audio_overlap_rate", "Overlap Rate"),
            ("audio_interruption_rate", "Interruption Rate"),
            ("audio_inter_turn_gap", "Inter-Turn Gap"),
            ("audio_excessive_gap_rate", "Excessive Gap Rate"),
            ("audio_barge_in_success_rate", "Barge-In Success Rate"),
            ("audio_utterance_duration", "Utterance Duration"),
            ("audio_speech_rate_wpm", "Speech Rate"),
            ("audio_silence_ratio", "Silence Ratio"),
            ("audio_clipping_rate", "Clipping Rate"),
            ("audio_loudness_stability", "Loudness Stability"),
            ("audio_signal_to_noise_estimate", "Signal-to-Noise Estimate"),
            ("audio_turn_budget_violation_rate", "Turn Budget Violation Rate"),
            ("audio_real_time_interaction_factor", "Real-Time Interaction Factor"),
        ],
    },
    {
        "key": "asr",
        "order": 2,
        "title": "2. Automatic Speech Recognition",
        "description": "Conversion of incoming audio into the transcript consumed by NLU.",
        "metrics": [
            ("asr_success_rate", "Recognition Success Rate"),
            ("asr_failure_rate", "Recognition Failure Rate"),
            ("asr_wer", "WER"),
            ("asr_character_error_rate", "Character Error Rate"),
            ("asr_sentence_error_rate", "Sentence Error Rate"),
            ("asr_semantic_asr_error_rate", "Semantic ASR Error Rate"),
            ("asr_entity_error_rate", "Entity Error Rate"),
            ("asr_station_precision", "Station Precision"),
            ("asr_station_recall", "Station Recall"),
            ("asr_station_f1", "Station F1"),
            ("asr_critical_slot_accuracy", "Critical-Slot Accuracy"),
            ("asr_route_sequence_edit_distance", "Route Sequence Edit Distance"),
            ("asr_constraint_preservation_rate", "Constraint Preservation Rate"),
            ("asr_negation_preservation_rate", "Negation Preservation Rate"),
            ("asr_numeric_preservation_rate", "Numeric Preservation Rate"),
            ("asr_empty_transcript_rate", "Empty Transcript Rate"),
            ("asr_hallucinated_token_rate", "Hallucinated Token Rate"),
            ("asr_transcript_correction_count", "Transcript Correction Count"),
            ("asr_uncorrected_misinterpretation_count", "Uncorrected Misinterpretation Count"),
            ("asr_recognition_latency", "Recognition Latency"),
            ("asr_real_time_factor", "ASR Real-Time Factor"),
            ("asr_confidence_calibration_error", "Confidence Calibration Error"),
            ("asr_repair_trigger_rate", "ASR Repair-Trigger Rate"),
            ("asr_repair_success_rate", "ASR Repair Success"),
        ],
    },
    {
        "key": "nlu",
        "order": 3,
        "title": "3. Spoken-Language Understanding",
        "description": "Intent, entity, constraint, temporal, and route-frame interpretation.",
        "metrics": [
            ("nlu_intent_accuracy", "Intent Accuracy"),
            ("nlu_intent_macro_f1", "Intent Macro F1"),
            ("nlu_slot_precision", "Slot Precision"),
            ("nlu_slot_recall", "Slot Recall"),
            ("nlu_slot_f1", "Slot F1"),
            ("nlu_joint_frame_accuracy", "Joint Frame Accuracy"),
            ("nlu_constraint_extraction_f1", "Constraint Extraction F1"),
            ("nlu_constraint_value_accuracy", "Constraint Value Accuracy"),
            ("nlu_semantic_frame_accuracy", "Semantic Frame Accuracy"),
            ("nlu_critical_slot_accuracy", "Critical Slot Accuracy"),
            ("nlu_route_valid_rate", "Route Parse Success Rate"),
            ("nlu_goal_reached_rate", "Goal-Reaching Parse Rate"),
            ("nlu_station_sequence_exact_match", "Station Sequence Exact Match"),
            ("nlu_station_sequence_edit_distance", "Station Sequence Edit Distance"),
            ("nlu_origin_destination_accuracy", "Origin/Destination Accuracy"),
            ("nlu_temporal_expression_accuracy", "Temporal Expression Accuracy"),
            ("nlu_negation_scope_accuracy", "Negation-Scope Accuracy"),
            ("nlu_ambiguity_detection_rate", "Ambiguity Detection Rate"),
            ("nlu_false_parse_rate", "False Parse Rate"),
            ("nlu_unknown_entity_detection_rate", "Unknown-Entity Detection Rate"),
            ("nlu_latency", "NLU Latency"),
            ("nlu_confidence_calibration_error", "NLU Confidence Calibration Error"),
            ("nlu_pipeline_input_match_rate", "Pipeline Input Agreement"),
        ],
    },
    {
        "key": "dialogue_state_tracking",
        "order": 4,
        "title": "4. Dialogue State Tracking",
        "description": "Turn-to-turn goals, constraints, stages, and candidate-memory state.",
        "metrics": [
            ("dialogue_state_joint_goal_accuracy", "Joint Goal Accuracy"),
            ("dialogue_state_slot_accuracy", "Slot Accuracy"),
            ("dialogue_state_stage_accuracy", "Stage Accuracy"),
            ("dialogue_state_state_drift_rate", "State Drift Rate"),
            ("dialogue_state_constraint_retention_rate", "Constraint Retention Rate"),
            ("dialogue_state_constraint_corruption_rate", "Constraint Corruption Rate"),
            ("dialogue_state_constraint_omission_rate", "Constraint Omission Rate"),
            ("dialogue_state_premature_constraint_activation_rate", "Premature Constraint Activation"),
            ("dialogue_state_shared_state_agreement", "Shared State Agreement"),
            ("dialogue_state_candidate_memory_precision", "Candidate-Memory Precision"),
            ("dialogue_state_candidate_memory_recall", "Candidate-Memory Recall"),
            ("dialogue_state_candidate_deduplication_accuracy", "Candidate Deduplication Accuracy"),
            ("dialogue_state_selected_route_consistency", "Selected-Route Consistency"),
            ("dialogue_state_route_state_consistency", "Route-State Consistency"),
            ("dialogue_state_state_contradiction_rate", "State Contradiction Rate"),
            ("dialogue_state_update_latency", "State Update Latency"),
            ("dialogue_state_recovery_rate", "State Recovery Rate"),
        ],
    },
    {
        "key": "dialogue_management",
        "order": 5,
        "title": "5. Dialogue Management and Policy",
        "description": "Next-action selection, clarification, repair, progression, and stopping.",
        "metrics": [
            ("dialogue_management_correct_next_action_rate", "Correct Next-Action Rate"),
            ("dialogue_management_stage_transition_precision", "Stage-Transition Precision"),
            ("dialogue_management_stage_transition_recall", "Stage-Transition Recall"),
            ("dialogue_management_stage_skip_rate", "Stage Skip Rate"),
            ("dialogue_management_constraint_order_adherence", "Constraint-Order Adherence"),
            ("dialogue_management_premature_answer_rate", "Premature Answer Rate"),
            ("dialogue_management_premature_closure_rate", "Premature Closure Rate"),
            ("dialogue_management_clarification_precision", "Clarification Precision"),
            ("dialogue_management_clarification_recall", "Clarification Recall"),
            ("dialogue_management_unnecessary_clarification_rate", "Unnecessary Clarification Rate"),
            ("dialogue_management_clarification_calibration", "Clarification Calibration"),
            ("dialogue_management_repair_attempt_rate", "Repair Attempt Rate"),
            ("dialogue_management_repair_success_rate", "Repair Success Rate"),
            ("dialogue_management_repeated_repair_rate", "Repeated Repair Rate"),
            ("dialogue_management_invalid_proposal_handling_accuracy", "Invalid-Proposal Handling Accuracy"),
            ("dialogue_management_constraint_violation_handling_accuracy", "Constraint-Violation Handling Accuracy"),
            ("dialogue_management_distinct_proposal_rate", "Distinct Proposal Rate"),
            ("dialogue_management_route_repetition_rate", "Route Repetition Rate"),
            ("dialogue_management_route_revision_rate", "Route Revision Rate"),
            ("dialogue_management_policy_progress_rate", "Policy Progress Rate"),
            ("dialogue_management_stagnation_rate", "Stagnation Rate"),
            ("dialogue_management_turn_efficiency", "Turn Efficiency"),
            ("dialogue_management_stop_decision_accuracy", "Stop-Decision Accuracy"),
            ("dialogue_management_turn_limit_utilization", "Turn-Limit Utilization"),
            ("dialogue_management_policy_latency", "Policy Latency"),
        ],
    },
    {
        "key": "backend_task_execution",
        "order": 6,
        "title": "6. Backend Task Execution and Grounding",
        "description": "Route planning, network/tool grounding, factuality, and candidate quality.",
        "metrics": [
            ("agent_b_proposal_parse_rate", "Proposal Parse Rate"),
            ("agent_b_route_validity_rate", "Route Validity Rate"),
            ("agent_b_destination_reach_rate", "Destination Reach Rate"),
            ("agent_b_complete_route_rate", "Complete Route Rate"),
            ("agent_b_grounded_proposal_score", "Grounded Proposal Score"),
            ("agent_b_hallucinated_content_rate", "Hallucinated Content Rate"),
            ("agent_b_hallucinated_station_rate", "Hallucinated Station Rate"),
            ("agent_b_hallucinated_connection_rate", "Hallucinated Connection Rate"),
            ("agent_b_unsupported_attribute_rate", "Unsupported Attribute Rate"),
            ("agent_b_mode_permission_compliance", "Line-Use Compliance"),
            ("agent_b_duration_factuality", "Duration Factuality"),
            ("agent_b_transfer_count_factuality", "Transfer-Count Factuality"),
            ("agent_b_active_constraint_compliance", "Active-Constraint Compliance"),
            ("agent_b_actionability_score", "Actionability Score"),
            ("agent_b_route_novelty", "Route Novelty"),
            ("agent_b_pareto_improvement_rate", "Pareto Improvement Rate"),
            ("agent_b_dominated_proposal_rate", "Dominated Proposal Rate"),
            ("agent_b_optimality_ratio", "Optimality Ratio"),
            ("agent_b_duration_regret", "Duration Regret"),
            ("agent_b_viable_alternative_coverage", "Viable Alternative Coverage"),
            ("agent_b_best_route_discovery_rate", "Best-Route Discovery Rate"),
            ("agent_b_best_route_discovery_turn", "Best-Route Discovery Turn"),
            ("agent_b_plugin_execution_success", "Plugin Execution Success"),
            ("agent_b_model_generation_latency", "Model Generation Latency"),
        ],
    },
    {
        "key": "nlg",
        "order": 7,
        "title": "7. Natural-Language Generation",
        "description": "Realization of grounded system actions as concise spoken-language text.",
        "metrics": [
            ("nlg_semantic_adequacy", "Semantic Adequacy"),
            ("nlg_faithfulness", "Faithfulness"),
            ("nlg_slot_error_rate", "Slot Error Rate"),
            ("nlg_executable_utterance_rate", "Executable Utterance Rate"),
            ("nlg_route_mention_completeness", "Route Mention Completeness"),
            ("nlg_constraint_mention_precision", "Constraint Mention Precision"),
            ("nlg_constraint_mention_recall", "Constraint Mention Recall"),
            ("nlg_information_order_accuracy", "Information-Order Accuracy"),
            ("nlg_conciseness", "Conciseness"),
            ("nlg_excess_verbosity_rate", "Excess Verbosity Rate"),
            ("nlg_underspecification_rate", "Underspecification Rate"),
            ("nlg_repetition_rate", "Repetition Rate"),
            ("nlg_distinct_1", "Distinct-1"),
            ("nlg_distinct_2", "Distinct-2"),
            ("nlg_lexical_diversity", "Lexical Diversity"),
            ("nlg_bleu", "BLEU"),
            ("nlg_rouge_l", "ROUGE-L"),
            ("nlg_meteor", "METEOR"),
            ("nlg_semantic_similarity", "Semantic Similarity"),
            ("nlg_constraint_satisfaction_rate", "Constraint Satisfaction Rate"),
            ("nlg_formatting_violation_rate", "Formatting Violation Rate"),
            ("nlg_hidden_reasoning_leakage_rate", "Hidden-Reasoning Leakage Rate"),
            ("nlg_grammatical_acceptability", "Grammatical Acceptability"),
            ("nlg_text_naturalness", "Text Naturalness"),
            ("nlg_estimated_spoken_duration", "Estimated Spoken Duration"),
        ],
    },
    {
        "key": "tts",
        "order": 8,
        "title": "8. Text-to-Speech and Audio Output",
        "description": "Synthesis, playback, acoustic validity, and round-trip intelligibility.",
        "metrics": [
            ("tts_success_rate", "Synthesis Success Rate"),
            ("tts_failure_rate", "Synthesis Failure Rate"),
            ("tts_audio_validity_rate", "Audio Validity Rate"),
            ("tts_synthesis_latency", "Synthesis Latency"),
            ("tts_real_time_factor", "TTS Real-Time Factor"),
            ("tts_audio_duration_error", "Audio Duration Error"),
            ("tts_speaking_rate_accuracy", "Speaking-Rate Accuracy"),
            ("tts_pause_pattern_adherence", "Pause-Pattern Adherence"),
            ("tts_pronunciation_accuracy", "Pronunciation Accuracy"),
            ("tts_station_pronunciation_accuracy", "Station Pronunciation Accuracy"),
            ("tts_round_trip_semantic_intelligibility", "Round-Trip Semantic Intelligibility"),
            ("tts_round_trip_route_accuracy", "Round-Trip Route Accuracy"),
            ("tts_loudness_compliance", "Loudness Compliance"),
            ("tts_clipping_rate", "Clipping Rate"),
            ("tts_leading_trailing_silence", "Leading/Trailing Silence"),
            ("tts_speaker_consistency", "Speaker Consistency"),
            ("tts_nisqa", "NISQA"),
            ("tts_dnsmos", "DNSMOS"),
            ("tts_playback_success_rate", "Playback Success Rate"),
            ("tts_text_change_rate", "Text Change Rate"),
        ],
    },
    {
        "key": "task_outcome",
        "order": 9,
        "title": "9. End-to-End Task Outcome",
        "description": "Final route success, constraints, optimality, completion speed, and closure.",
        "metrics": [
            ("task_outcome_completion", "Task Completion"),
            ("task_outcome_route_validity", "Valid-Route Completion"),
            ("task_outcome_acceptable_duration_completion", "Acceptable-Duration Completion"),
            ("task_outcome_constraint_satisfaction", "All-Constraint Satisfaction"),
            ("task_outcome_constraint_satisfaction_rate", "Constraint Satisfaction Rate"),
            ("task_outcome_stage_completion_rate", "Stage Completion Rate"),
            ("task_outcome_duration_quality", "Duration Quality"),
            ("task_outcome_duration_ratio", "Duration Ratio"),
            ("task_outcome_duration_regret", "Duration Regret"),
            ("task_outcome_correct_route_selection", "Correct Route Selection"),
            ("task_outcome_candidate_count", "Candidate Route Count"),
            ("task_outcome_turns_used", "Turns Used"),
            ("task_outcome_turns_to_success", "Turns to Success"),
            ("task_outcome_first_valid_route_turn", "First Valid Route Turn"),
            ("task_outcome_first_compliant_route_turn", "First Compliant Route Turn"),
            ("task_outcome_successful_natural_closure", "Successful Natural Closure"),
        ],
    },
    {
        "key": "whole_dialogue",
        "order": 10,
        "title": "10. Whole-Dialogue Interaction",
        "description": "Dialogue-level progress, efficiency, cost, repetition, and failure location.",
        "metrics": [
            ("whole_dialogue_dialogue_success_score", "Dialogue Success Score"),
            ("whole_dialogue_interaction_quality_trajectory", "Interaction Quality Trajectory"),
            ("whole_dialogue_goal_progress_auc", "Goal-Progress Area Under Curve"),
            ("whole_dialogue_dialogue_cost", "Dialogue Cost"),
            ("whole_dialogue_turn_count", "Turn Count"),
            ("whole_dialogue_word_count", "Word Count"),
            ("whole_dialogue_mean_words_per_turn", "Mean Words per Turn"),
            ("whole_dialogue_total_runtime", "Total Runtime"),
            ("whole_dialogue_candidate_count", "Candidate Count"),
            ("whole_dialogue_route_revision_count", "Route Revision Count"),
            ("whole_dialogue_clarification_count", "Clarification Count"),
            ("whole_dialogue_repair_count", "Repair Count"),
            ("whole_dialogue_warning_count", "Warning Count"),
            ("whole_dialogue_abandonment_rate", "Abandonment Rate"),
            ("whole_dialogue_failure_localization_score", "Failure Localization Score"),
            ("whole_dialogue_failure_phase", "Earliest Failing Phase"),
            ("whole_dialogue_pipeline_dependency_integrity", "Pipeline Dependency Integrity"),
            ("whole_dialogue_cooperative_progress_rate", "Cooperative Progress Rate"),
            ("whole_dialogue_task_focus_score", "Task-Focus Score"),
            ("whole_dialogue_conversation_repetition_rate", "Conversation Repetition Rate"),
            ("whole_dialogue_natural_closure_rate", "Natural Closure Rate"),
            ("whole_dialogue_resource_cost", "Resource Cost"),
            ("whole_dialogue_estimated_monetary_cost", "Estimated Monetary Cost"),
        ],
    },
    {
        "key": "metric_validity",
        "order": 11,
        "title": "11. Cross-Run Validity and Robustness",
        "description": "Uncertainty, repeatability, sensitivity, redundancy, and subgroup robustness.",
        "metrics": [
            ("metric_validity_success_confidence_interval_low", "Success CI Low"),
            ("metric_validity_success_confidence_interval_high", "Success CI High"),
            ("metric_validity_metric_outcome_correlation", "Metric-Outcome Correlation"),
            ("metric_validity_rank_stability", "Rank Stability"),
            ("metric_validity_seed_variance", "Seed Variance"),
            ("metric_validity_test_retest_agreement", "Test-Retest Agreement"),
            ("metric_validity_missingness_rate", "Missingness Rate"),
            ("metric_validity_ceiling_rate", "Ceiling Rate"),
            ("metric_validity_floor_rate", "Floor Rate"),
            ("metric_validity_metric_redundancy", "Metric Redundancy"),
            ("metric_validity_perturbation_sensitivity", "Perturbation Sensitivity"),
            ("metric_validity_persona_robustness", "Persona Robustness"),
            ("metric_validity_scenario_robustness", "Scenario Robustness"),
            ("metric_validity_speech_pattern_robustness", "Speech-Pattern Robustness"),
            ("metric_validity_provider_robustness", "Provider Robustness"),
            ("metric_validity_subgroup_performance_gap", "Subgroup Performance Gap"),
        ],
    },
]


METRIC_DISPLAY_NAMES = {
    key: label
    for family in METRIC_FAMILY_SPECS
    for key, label in family["metrics"]
}
PHASE_DISPLAY_NAMES = {
    family["key"]: family["title"]
    for family in METRIC_FAMILY_SPECS
}
METRIC_KEYS = tuple(METRIC_DISPLAY_NAMES)
CORE_METRICS_BY_PHASE = {
    "user_simulation": (
        "agent_a_verifier_catch_rate",
        "agent_a_constraint_violation_catch_rate",
        "agent_a_false_acceptance_rate",
        "agent_a_critique_specificity",
        "agent_a_constraint_revelation_order",
        "agent_a_best_candidate_retention",
        "agent_a_closure_correctness",
    ),
    "audio_input": (
        "audio_capture_success_rate",
        "audio_turn_latency",
        "audio_response_latency",
        "audio_endpointing_latency",
        "audio_end_of_utterance_error",
        "audio_overlap_rate",
        "audio_utterance_duration",
    ),
    "asr": (
        "asr_success_rate",
        "asr_wer",
        "asr_semantic_asr_error_rate",
        "asr_entity_error_rate",
        "asr_station_f1",
        "asr_critical_slot_accuracy",
        "asr_recognition_latency",
    ),
    "nlu": (
        "nlu_intent_accuracy",
        "nlu_slot_f1",
        "nlu_joint_frame_accuracy",
        "nlu_constraint_extraction_f1",
        "nlu_semantic_frame_accuracy",
        "nlu_critical_slot_accuracy",
        "nlu_route_valid_rate",
    ),
    "dialogue_state_tracking": (
        "dialogue_state_joint_goal_accuracy",
        "dialogue_state_stage_accuracy",
        "dialogue_state_state_drift_rate",
        "dialogue_state_constraint_retention_rate",
        "dialogue_state_shared_state_agreement",
        "dialogue_state_candidate_deduplication_accuracy",
        "dialogue_state_selected_route_consistency",
    ),
    "dialogue_management": (
        "dialogue_management_correct_next_action_rate",
        "dialogue_management_stage_transition_precision",
        "dialogue_management_constraint_order_adherence",
        "dialogue_management_premature_answer_rate",
        "dialogue_management_clarification_calibration",
        "dialogue_management_repair_success_rate",
        "dialogue_management_stop_decision_accuracy",
    ),
    "backend_task_execution": (
        "agent_b_route_validity_rate",
        "agent_b_destination_reach_rate",
        "agent_b_grounded_proposal_score",
        "agent_b_hallucinated_content_rate",
        "agent_b_active_constraint_compliance",
        "agent_b_actionability_score",
        "agent_b_route_novelty",
    ),
    "nlg": (
        "nlg_semantic_adequacy",
        "nlg_faithfulness",
        "nlg_executable_utterance_rate",
        "nlg_route_mention_completeness",
        "nlg_information_order_accuracy",
        "nlg_conciseness",
        "nlg_repetition_rate",
    ),
    "tts": (
        "tts_success_rate",
        "tts_audio_validity_rate",
        "tts_synthesis_latency",
        "tts_pronunciation_accuracy",
        "tts_round_trip_semantic_intelligibility",
        "tts_round_trip_route_accuracy",
        "tts_nisqa",
        "tts_dnsmos",
    ),
    "task_outcome": (
        "task_outcome_completion",
        "task_outcome_route_validity",
        "task_outcome_acceptable_duration_completion",
        "task_outcome_constraint_satisfaction",
        "task_outcome_stage_completion_rate",
        "task_outcome_duration_quality",
        "task_outcome_correct_route_selection",
    ),
    "whole_dialogue": (
        "whole_dialogue_dialogue_success_score",
        "whole_dialogue_interaction_quality_trajectory",
        "whole_dialogue_goal_progress_auc",
        "whole_dialogue_dialogue_cost",
        "whole_dialogue_failure_localization_score",
        "whole_dialogue_failure_phase",
        "whole_dialogue_pipeline_dependency_integrity",
    ),
    "metric_validity": (
        "metric_validity_success_confidence_interval_low",
        "metric_validity_success_confidence_interval_high",
        "metric_validity_metric_outcome_correlation",
        "metric_validity_rank_stability",
        "metric_validity_seed_variance",
        "metric_validity_test_retest_agreement",
        "metric_validity_missingness_rate",
    ),
}
CORE_METRIC_KEYS = {
    key
    for keys in CORE_METRICS_BY_PHASE.values()
    for key in keys
}
SUPPLEMENTARY_METRIC_KEYS = set(METRIC_KEYS) - CORE_METRIC_KEYS
DEFAULT_METRIC_CONFIG = {key: key in CORE_METRIC_KEYS for key in METRIC_KEYS}
DEFAULT_METRIC_TIERS = {
    key: "core" if key in CORE_METRIC_KEYS else "supplementary"
    for key in METRIC_KEYS
}


LEARNED_METRIC_KEYS = {
    "audio_signal_to_noise_estimate",
    "nlg_grammatical_acceptability",
    "nlg_text_naturalness",
    "tts_speaker_consistency",
    "tts_nisqa",
    "tts_dnsmos",
}

REFERENCE_METRIC_PREFIXES = (
    "asr_",
    "nlu_",
    "dialogue_state_",
)

PHASE_TRACE_REQUIREMENTS = {
    "user_simulation": ["caller turns", "constraint status", "candidate history"],
    "audio_input": ["speech_turns", "timing_turns", "audio WAV artifacts"],
    "asr": ["outgoing_text", "incoming_transcript", "ASR timing"],
    "nlu": ["ASR transcript", "parsed_route", "source route frame", "NLU timing"],
    "dialogue_state_tracking": ["runtime_events", "constraint snapshots", "candidate_events"],
    "dialogue_management": ["candidate_events", "warnings", "turn outcomes"],
    "backend_task_execution": ["NLU turns", "candidate_events", "network route authority"],
    "nlg": ["Agent B text", "reference route text", "parsed route"],
    "tts": ["outgoing_text", "audio WAV artifacts", "TTS timing", "ASR round trip"],
    "task_outcome": ["final route", "constraint status", "reference route"],
    "whole_dialogue": ["conversation", "timing", "candidate events", "outcome"],
    "metric_validity": ["multiple completed MetricRecord rows"],
}


METRIC_PREFIXES = (
    "dialogue_management_",
    "dialogue_state_",
    "task_outcome_",
    "whole_dialogue_",
    "metric_validity_",
    "agent_a_",
    "agent_b_",
    "audio_",
    "asr_",
    "nlu_",
    "nlg_",
    "tts_",
)


def phase_key(family):
    """Return the explicit stable phase identifier for a catalog family."""
    return family["key"]


def metric_local_name(metric_key):
    """Return a readable phase-local metric name while retaining its global key."""
    for prefix in METRIC_PREFIXES:
        if metric_key.startswith(prefix):
            return metric_key[len(prefix):]
    return metric_key


def global_metric_key(phase, local_name):
    """Resolve a phase-local exported name back to the stable global metric key."""
    for family in METRIC_FAMILY_SPECS:
        if family["key"] != phase:
            continue
        for key, _label in family["metrics"]:
            if metric_local_name(key) == local_name:
                return key
    return f"{phase}_{local_name}"


def metric_metadata(key, phase, metric_tiers=None):
    """Return manifest metadata for one catalog metric."""
    tiers = dict(DEFAULT_METRIC_TIERS)
    if isinstance(metric_tiers, dict):
        tiers.update({
            metric_key: str(tier).strip().lower()
            for metric_key, tier in metric_tiers.items()
            if metric_key in METRIC_KEYS and str(tier).strip().lower() in {"core", "supplementary"}
        })
    if key in LEARNED_METRIC_KEYS:
        evidence_class = "L"
    elif phase == "metric_validity":
        evidence_class = "D"
    elif any(
        marker in key
        for marker in (
            "success_rate",
            "failure_rate",
            "latency",
            "real_time_factor",
            "empty_transcript",
            "route_valid_rate",
            "goal_reached_rate",
            "pipeline_input_match",
            "candidate_memory_precision",
            "candidate_memory_recall",
            "selected_route_consistency",
            "route_state_consistency",
            "distinct_proposal",
            "route_repetition",
            "route_revision",
            "policy_progress",
            "stagnation",
            "turn_efficiency",
            "turn_limit_utilization",
        )
    ):
        evidence_class = "D"
    elif key.startswith(REFERENCE_METRIC_PREFIXES):
        evidence_class = "R"
    else:
        evidence_class = "D"

    if any(term in key for term in ("latency", "_runtime", "duration_error", "silence")):
        unit = "seconds"
    elif key in {"tts_nisqa", "tts_dnsmos"}:
        unit = "mean_opinion_score_1_to_5"
    elif key.endswith(("_count", "_turn", "_turns_used", "_word_count")):
        unit = "count"
    elif key.endswith(("_regret", "_gap")):
        unit = "minutes_or_domain_units"
    elif key.endswith("_phase"):
        unit = "category"
    elif key.endswith("_wpm"):
        unit = "words_per_minute"
    else:
        unit = "ratio_or_score"

    metadata = {
        "key": key,
        "class": evidence_class,
        "tier": tiers.get(key, DEFAULT_METRIC_TIERS.get(key, "supplementary")),
        "scope": "cross_run" if phase == "metric_validity" else "single_run",
        "unit": unit,
        "calculation": metric_calculation_method(key),
        "required_trace_fields": PHASE_TRACE_REQUIREMENTS.get(phase, []),
        "missing_data_policy": "null",
    }
    if key == "tts_nisqa":
        metadata["estimator"] = {
            "name": "NISQA v2.0",
            "implementation": "TorchMetrics functional audio NISQA",
            "output": "overall mean opinion score",
            "range": [1.0, 5.0],
        }
    elif key == "tts_dnsmos":
        metadata["estimator"] = {
            "name": "DNSMOS",
            "implementation": "TorchMetrics functional audio DNSMOS",
            "output": "overall mean opinion score",
            "dimensions": [
                "P.808 MOS",
                "signal MOS",
                "background MOS",
                "overall MOS",
            ],
            "range": [1.0, 5.0],
            "personalized": False,
        }
    return metadata


def metric_calculation_method(key):
    """Return the concise calculation shown in console and research manifests."""
    explicit = {
        "asr_wer": "(substitutions + deletions + insertions) / reference words",
        "tts_nisqa": "NISQA model inference over synthesized waveform",
        "tts_dnsmos": "DNSMOS model inference over synthesized waveform",
        "task_outcome_duration_quality": "min(reference duration / selected duration, 1)",
        "task_outcome_stage_completion_rate": "mean(valid route, acceptable duration, constraints satisfied)",
        "whole_dialogue_dialogue_cost": "number of dialogue messages",
        "whole_dialogue_failure_phase": "first pipeline phase with recorded failure evidence",
        "metric_validity_missingness_rate": "missing configured values / configured values",
        "metric_validity_seed_variance": "variance across repeated random-seed conditions",
    }
    if key in explicit:
        return explicit[key]
    if key.endswith(("_count", "_turn", "_turns_used", "_word_count")):
        return "count of matching trace events"
    if any(term in key for term in ("latency", "duration", "runtime", "gap")):
        return "arithmetic aggregation of captured timing values"
    if key.endswith(("_rate", "_accuracy", "_precision", "_recall", "_f1", "_score")):
        return "eligible successful observations / eligible observations"
    if key.endswith(("_error", "_error_rate")):
        return "eligible error observations / eligible observations"
    return "deterministic derivation from the recorded pipeline trace"
