"""Evaluation-layer configuration."""

ROUTE_INTERPRETER_SCORING = {
    "marker_bonus": 30,
    "fragment_index_weight": 10,
    "route_length_weight": 1,
    "starts_correctly_bonus": 100,
    "reaches_goal_bonus": 200,
    "arrival_penalty_divisor": 10000,
}

METRIC_QUALITY_BASE_WEIGHT = 0.55
METRIC_QUALITY_DURATION_WEIGHT = 0.20
METRIC_AUTOMATIC_ASR_WEIGHT = 0.10
METRIC_AUTOMATIC_NLU_WEIGHT = 0.10
METRIC_AUTOMATIC_DIALOG_WEIGHT = 0.05
