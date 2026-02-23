from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

_SCHEMA_FILE = {
    "backtest": "backtest.schema.json",
    "recommend": "recommend.schema.json",
    "policy": "policy.schema.json",
}


def _load_schema(command: str) -> dict[str, Any]:
    if command not in _SCHEMA_FILE:
        raise ValueError(f"Unknown command for schema validation: {command}")
    schema_path = Path(__file__).resolve().parent / "schemas" / _SCHEMA_FILE[command]
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_config(payload: dict[str, Any], command: str) -> None:
    schema = _load_schema(command)
    try:
        validate(instance=payload, schema=schema)
    except ValidationError as exc:
        path = ".".join(str(p) for p in exc.path)
        where = f" at '{path}'" if path else ""
        raise ValueError(f"Config validation failed{where}: {exc.message}") from exc


def validate_policy(payload: dict[str, Any]) -> None:
    schema = _load_schema("policy")
    try:
        validate(instance=payload, schema=schema)
    except ValidationError as exc:
        path = ".".join(str(p) for p in exc.path)
        where = f" at '{path}'" if path else ""
        raise ValueError(f"Policy validation failed{where}: {exc.message}") from exc
