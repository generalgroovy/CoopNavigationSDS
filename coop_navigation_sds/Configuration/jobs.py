"""Load scriptable batch experiment job files."""

import json
from decimal import Decimal
from pathlib import Path


JOB_SCHEMA_VERSION = 1


def load_experiment_job(path):
    """Load and minimally validate a JSON ``.job`` experiment definition."""
    if not path:
        return {"schema_version": JOB_SCHEMA_VERSION, "config": {}, "grid": {}}
    job_path = Path(path)
    document = json.loads(job_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Experiment job must contain a JSON object.")
    version = int(document.get("schema_version", JOB_SCHEMA_VERSION))
    if version != JOB_SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment job schema version: {version}")
    config = document.get("config", {})
    grid = document.get("grid", {})
    parameter_values = document.get("parameter_values", {})
    parameter_ranges = document.get("parameter_ranges", {})
    if not all(isinstance(value, dict) for value in (config, grid, parameter_values, parameter_ranges)):
        raise ValueError(
            "Experiment job 'config', 'grid', 'parameter_values', and "
            "'parameter_ranges' values must be JSON objects."
        )
    return {
        "schema_version": version,
        "name": str(document.get("name", job_path.stem)),
        "description": str(document.get("description", "")),
        "config": dict(config),
        "grid": dict(grid),
        "parameter_values": dict(parameter_values),
        "parameter_ranges": dict(parameter_ranges),
        "iterations": max(1, int(document.get("iterations", 1))),
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
