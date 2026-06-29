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


CONFIG_SCHEMA_VERSION = 5
JOB_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 3
RESULT_SCHEMA_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = str(PROJECT_ROOT / "results")


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
