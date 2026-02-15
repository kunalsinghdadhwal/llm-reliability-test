"""Config loader for LLM Reliability Gate."""

from pathlib import Path

import yaml

_REQUIRED_FIELDS = [
    ("use_case",),
    ("risk_tolerance", "deploy_threshold"),
    ("risk_tolerance", "warn_threshold"),
    ("doc_sources",),
    ("model", "provider"),
    ("model", "model_id"),
    ("elasticsearch", "host"),
    ("elasticsearch", "index"),
]


def _check_required(cfg: dict) -> None:
    for parts in _REQUIRED_FIELDS:
        obj = cfg
        for part in parts:
            if not isinstance(obj, dict) or part not in obj:
                dotted = ".".join(parts)
                raise ValueError(f"Missing required config field: {dotted}")
            obj = obj[part]


def load_config(path: str | None = None) -> dict:
    """Load and validate an LLM Reliability Gate config file.

    Args:
        path: Path to a YAML config file.  Defaults to
              ".llm-reliability.yaml" in the current working directory.

    Returns:
        Normalized configuration dict with flattened thresholds.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If a required field is missing.
    """
    if path is None:
        path = ".llm-reliability.yaml"

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(file_path) as fh:
        cfg = yaml.safe_load(fh)

    _check_required(cfg)

    # Build normalized output, preserving extra fields.
    result = dict(cfg)

    # Flatten risk_tolerance -> thresholds
    rt = result.pop("risk_tolerance")
    result["thresholds"] = {
        "deploy": rt["deploy_threshold"],
        "warn": rt["warn_threshold"],
        "reject": rt.get("reject_threshold", 0.3),
    }

    return result
