"""Immutable resolved experiment specifications and provenance."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from coop_navigation_sds.Configuration.schema import CONFIG_SCHEMA_VERSION, SECRET_CONFIG_FIELDS


NON_EXPERIMENT_IDENTITY_FIELDS = frozenset({
    "execution_run_dir",
    "speech_audio_dir",
    "results_root",
    "model_api_key",
})


def freeze_value(value):
    """Recursively convert configuration data into immutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def thaw_value(value):
    """Return a JSON-compatible copy of recursively frozen data."""
    if isinstance(value, Mapping):
        return {str(key): thaw_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [thaw_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def configuration_fingerprint(values):
    """Return a stable SHA-256 identity for resolved non-secret values."""
    identity = {
        key: ("<redacted>" if key in SECRET_CONFIG_FIELDS else value)
        for key, value in thaw_value(values).items()
        if key not in NON_EXPERIMENT_IDENTITY_FIELDS
    }
    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentSpecification(Mapping):
    """Read-only configuration used by every phase of one experiment run."""

    data: Mapping
    fingerprint: str
    source: str
    resolved_at_utc: str
    schema_version: int = CONFIG_SCHEMA_VERSION

    @classmethod
    def resolve(cls, values, *, source="runtime"):
        plain = thaw_value(values)
        frozen = freeze_value(plain)
        return cls(
            data=frozen,
            fingerprint=configuration_fingerprint(plain),
            source=str(source),
            resolved_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def to_dict(self):
        return thaw_value(self.data)

    def provenance(self):
        return {
            "immutable": True,
            "configuration_schema_version": self.schema_version,
            "fingerprint_sha256": self.fingerprint,
            "source": self.source,
            "resolved_at_utc": self.resolved_at_utc,
        }


def ensure_experiment_specification(values, *, source="runtime"):
    """Return an existing specification or resolve one immutable snapshot."""
    if isinstance(values, ExperimentSpecification):
        return values
    return ExperimentSpecification.resolve(values, source=source)
