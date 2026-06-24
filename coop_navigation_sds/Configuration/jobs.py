"""Load scriptable batch experiment job files."""

import json
from decimal import Decimal
from pathlib import Path

from coop_navigation_sds.Configuration.schema import JOB_SCHEMA_VERSION


def load_experiment_job(path, _seen=None):
    """Load and minimally validate a JSON ``.job`` experiment definition."""
    if not path:
        return {"schema_version": JOB_SCHEMA_VERSION, "config": {}, "grid": {}}
    job_path = Path(path)
    job_path = job_path.resolve()
    seen = set(_seen or ())
    if job_path in seen:
        raise ValueError(f"Cyclic experiment job inheritance at {job_path}")
    seen.add(job_path)
    document = json.loads(job_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Experiment job must contain a JSON object.")
    version = int(document.get("schema_version", JOB_SCHEMA_VERSION))
    if version != JOB_SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment job schema version: {version}")
    inherited = {}
    if document.get("extends"):
        parent_path = Path(str(document["extends"]))
        if not parent_path.is_absolute():
            parent_path = job_path.parent / parent_path
        inherited = load_experiment_job(parent_path, seen)
    raw_config = document.get("config", {})
    raw_grid = document.get("grid", {})
    raw_parameter_values = document.get("parameter_values", {})
    raw_parameter_ranges = document.get("parameter_ranges", {})
    if not all(
        isinstance(value, dict)
        for value in (
            raw_config,
            raw_grid,
            raw_parameter_values,
            raw_parameter_ranges,
        )
    ):
        raise ValueError(
            "Experiment job 'config', 'grid', 'parameter_values', and "
            "'parameter_ranges' values must be JSON objects."
        )
    config = {**inherited.get("config", {}), **raw_config}
    grid = {**inherited.get("grid", {}), **raw_grid}
    parameter_values = {
        **inherited.get("parameter_values", {}),
        **raw_parameter_values,
    }
    parameter_ranges = {
        **inherited.get("parameter_ranges", {}),
        **raw_parameter_ranges,
    }
    parameter_profiles = document.get(
        "parameter_profiles",
        inherited.get("parameter_profiles", []),
    )
    if not isinstance(parameter_profiles, list) or not all(
        isinstance(profile, dict) for profile in parameter_profiles
    ):
        raise ValueError("Experiment job 'parameter_profiles' must be a list of JSON objects.")
    profile_keys = [
        str(profile.get("profile_key", "")).strip()
        for profile in parameter_profiles
    ]
    if any(not key for key in profile_keys):
        raise ValueError("Every parameter profile requires a non-empty 'profile_key'.")
    if len(profile_keys) != len(set(profile_keys)):
        raise ValueError("Parameter profile keys must be unique.")
    return {
        "schema_version": version,
        "name": str(document.get("name", inherited.get("name", job_path.stem))),
        "description": str(document.get("description", inherited.get("description", ""))),
        "config": dict(config),
        "grid": dict(grid),
        "parameter_values": dict(parameter_values),
        "parameter_ranges": dict(parameter_ranges),
        "parameter_profiles": [dict(profile) for profile in parameter_profiles],
        "iterations": max(1, int(document.get("iterations", inherited.get("iterations", 1)))),
        "source": str(job_path),
    }


def job_grid_value(job, key, fallback):
    """Return a comma-separated command-line default from a job grid."""
    value = job.get("grid", {}).get(key, fallback)
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return value


def numeric_range_values(spec):
    """Expand an inclusive numeric range with stable decimal arithmetic."""
    if not isinstance(spec, dict):
        raise ValueError("Parameter range must contain start, stop, and step values.")
    start = Decimal(str(spec["start"]))
    stop = Decimal(str(spec["stop"]))
    step = Decimal(str(spec["step"]))
    if step == 0 or (stop - start) * step < 0:
        raise ValueError("Parameter range step must move from start toward stop.")
    values = []
    current = start
    comparison = (lambda value: value <= stop) if step > 0 else (lambda value: value >= stop)
    while comparison(current):
        values.append(int(current) if current == current.to_integral_value() else float(current))
        current += step
        if len(values) > 10000:
            raise ValueError("Parameter range expands to more than 10000 values.")
    return values


def job_parameter_grid(job):
    """Return arbitrary condition parameters from value lists and ranges."""
    parameters = {}
    for key, values in job.get("parameter_values", {}).items():
        parameters[str(key)] = list(values) if isinstance(values, (list, tuple)) else [values]
    for key, spec in job.get("parameter_ranges", {}).items():
        if key in parameters:
            raise ValueError(f"Batch parameter '{key}' has both values and a range.")
        parameters[str(key)] = numeric_range_values(spec)
    return parameters


def job_parameter_profiles(job):
    """Return linked named treatments without forming a parameter cross product."""
    profiles = list(job.get("parameter_profiles") or [])
    return profiles or [{"profile_key": "default"}]
