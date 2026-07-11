"""Stable schemas, paths, and reproducibility metadata shared by all controllers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import re
import sys
from collections.abc import Mapping
from types import MappingProxyType


CONFIG_SCHEMA_VERSION = 5
JOB_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 3
RESULT_SCHEMA_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = str(PROJECT_ROOT / "results")

# Stable result names are part of the research-data contract. Writers and
# readers reference this map instead of repeating filename literals.
RESULT_FILES = MappingProxyType({
    "summary": "run_summary.json",
    "artifact_inventory": "artifact_inventory.json",
    "conditions": "conditions.jsonl",
    "metric_inputs": "metric_inputs.json",
    "metrics_long": "metrics_long.csv",
    "metrics_wide": "metrics_wide.csv",
    "metrics_workbook": "automatic_eval_metrics.xlsx",
    "protocols": "conversation_protocols.jsonl",
    "transcripts": "conversation_transcripts.txt",
    "protocol_index": "index.jsonl",
    "runtime_events": "runtime_events.jsonl",
    "runtime_sessions": "runtime_sessions.jsonl",
    "runtime_log": "runtime.log",
    "network_data": "network_overview.json",
    "network_graph": "network_graph.svg",
    "analysis_overview": "analysis_overview.html",
    "condition_analysis": "condition_analysis.csv",
    "run_analysis": "run_analysis.csv",
    "phase_scorecard": "run_phase_scorecard.csv",
    "performance_band_summary": "performance_band_summary.csv",
})

# Roles distinguish irreplaceable observations from views that can be rebuilt
# from those observations. This distinction is written into every run.
CANONICAL_RESULT_FILES = frozenset({
    RESULT_FILES["conditions"],
    RESULT_FILES["metric_inputs"],
    RESULT_FILES["metrics_long"],
    RESULT_FILES["protocols"],
    RESULT_FILES["transcripts"],
    RESULT_FILES["runtime_events"],
    RESULT_FILES["network_data"],
})


def safe_artifact_name(value, maximum_length=96):
    """Return one portable, bounded filename component."""
    text = str(value or "run").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text).strip("._") or "run"
    limit = max(16, int(maximum_length))
    if len(text) > limit:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        text = f"{text[: limit - 9].rstrip('._')}-{digest}"
    return text


def resolve_results_root(value=None):
    """Return one stable results root independent of the process working directory."""
    path = Path(value or DEFAULT_RESULTS_ROOT).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def normalized_result_group_parts(group):
    """Return canonical result-group parts beneath the configured results root.

    Model run folders are intentionally shallow: legacy groups such as
    ``agent_b/userlm_transformers/01-small-tinyllama-1.1b/userlm`` resolve to
    ``01-small-tinyllama-1.1b/userlm``. Derived cross-run artifacts use explicit
    top-level groups such as ``comparison`` and ``general``.
    """
    raw = str(group or "").strip().replace("\\", "/")
    if not raw:
        return []
    raw_parts = raw.split("/")
    if (
        raw.startswith("/")
        or any(part in {"", ".", ".."} for part in raw_parts)
        or any(":" in part for part in raw_parts)
    ):
        raise ValueError("Result group must be a relative path without traversal.")
    if raw_parts[0] == "agent_b" and len(raw_parts) >= 4:
        raw_parts = raw_parts[-2:]
    return [safe_artifact_name(part, maximum_length=64) for part in raw_parts]


def resolve_result_group(results_root, group=None):
    """Resolve a portable relative result group beneath one results root."""
    root = Path(resolve_results_root(results_root))
    safe_parts = normalized_result_group_parts(group)
    if not safe_parts:
        return str(root)
    selected = root.joinpath(*safe_parts).resolve()
    if root != selected and root not in selected.parents:
        raise ValueError("Result group resolves outside the configured results root.")
    return str(selected)

SECRET_CONFIG_FIELDS = frozenset({"model_api_key", "token", "api_key"})
RAW_TRACE_COLLECTIONS = (
    "conversation",
    "agent_memories",
    "speech_pipeline",
    "timing",
    "phase_timing",
    "semantic_parsing",
    "runtime_events",
    "candidate_routes",
    "prompt_audits",
)


@dataclass(frozen=True)
class RunArtifactPaths:
    """Canonical flat artifact paths for one completed experiment run."""

    run_dir: Path
    prefix: str

    def as_dict(self):
        base = self.run_dir
        prefix = self.prefix
        return {
            "protocol": base / f"{prefix}_protocol.json",
            "transcript_txt": base / f"{prefix}_conversation_transcript.txt",
            "conversation_wav": base / f"{prefix}_conversation.wav",
            "metric_inputs": base / "metric_inputs.json",
        }


def sanitized_config(config):
    """Return a serializable configuration without persisted credentials."""
    def serializable(value):
        if isinstance(value, Mapping):
            return {str(key): serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [serializable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    return {
        key: (
            "<redacted>"
            if key in SECRET_CONFIG_FIELDS and value
            else serializable(value)
        )
        for key, value in dict(config or {}).items()
        if key not in {"execution_run_dir"}
    }


def runtime_environment_metadata():
    """Capture lightweight reproducibility metadata without optional dependencies."""
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "process_id": os.getpid(),
    }
