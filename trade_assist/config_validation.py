from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

from .policy.registry import get_policy_adapter

_SCHEMA_FILE = {
    "backtest": "backtest.schema.json",
    "recommend": "recommend.schema.json",
    "tune": "tune.schema.json",
    "policy": "policy.schema.json",
}


class TradeAssistUserError(ValueError):
    """User-facing error that should be rendered cleanly by the CLI."""


class ConfigValidationError(TradeAssistUserError):
    """Raised when a command config does not satisfy its schema."""


class PolicyValidationError(TradeAssistUserError):
    """Raised when a policy payload does not satisfy its schema."""


def _load_schema_by_filename(filename: str) -> dict[str, Any]:
    schema_path = Path(__file__).resolve().parent / "schemas" / filename
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_schema(command: str) -> dict[str, Any]:
    if command not in _SCHEMA_FILE:
        raise ValueError(f"Unknown command for schema validation: {command}")
    return _load_schema_by_filename(_SCHEMA_FILE[command])


def _path_label(exc: ValidationError) -> str:
    path = ".".join(str(part) for part in exc.path)
    return path or "top level"


def _missing_required_property(exc: ValidationError) -> str | None:
    if exc.validator != "required" or not isinstance(exc.instance, dict):
        return None
    for key in exc.validator_value:
        if key not in exc.instance:
            return str(key)
    return None


def _unexpected_property(exc: ValidationError) -> str | None:
    if exc.validator != "additionalProperties" or not isinstance(exc.instance, dict):
        return None
    properties = exc.schema.get("properties", {})
    for key in exc.instance:
        if key not in properties:
            return str(key)
    return None


def _format_validation_message(exc: ValidationError, label: str) -> str:
    missing = _missing_required_property(exc)
    if missing is not None:
        return (
            f"Invalid {label}: missing required field '{missing}' "
            f"at {_path_label(exc)}."
        )

    extra = _unexpected_property(exc)
    if extra is not None:
        return (
            f"Invalid {label}: unknown field '{extra}' "
            f"at {_path_label(exc)}."
        )

    return f"Invalid {label} at {_path_label(exc)}: {exc.message}"


def validate_config(payload: dict[str, Any], command: str) -> None:
    schema = _load_schema(command)
    try:
        validate(instance=payload, schema=schema)
    except ValidationError as exc:
        raise ConfigValidationError(
            _format_validation_message(exc, f"{command} config")
        ) from exc


def validate_policy(payload: dict[str, Any], policy_type: str = "v1") -> None:
    adapter = get_policy_adapter(policy_type)
    schema = adapter.policy_schema()
    try:
        validate(instance=payload, schema=schema)
    except ValidationError as exc:
        raise PolicyValidationError(_format_validation_message(exc, "policy")) from exc
