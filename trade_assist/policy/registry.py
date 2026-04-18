from __future__ import annotations

import importlib
import pkgutil
from importlib.metadata import entry_points
from typing import Any, Dict

from .adapter import PolicyAdapter

_POLICY_ADAPTERS: Dict[str, PolicyAdapter] = {}
_DISCOVERED = False

ENTRYPOINT_GROUP = "trade_assist.policy_adapters"
ADAPTER_ATTR_NAMES = ("ADAPTER", "adapter", "POLICY_ADAPTER")


def _is_policy_adapter(candidate: Any) -> bool:
    return all(
        hasattr(candidate, attr)
        for attr in (
            "policy_type",
            "policy_schema",
            "default_policy_payload",
            "build_policy_config",
            "run_backtest",
            "recommend",
        )
    )


def _coerce_adapter(candidate: Any) -> PolicyAdapter:
    if _is_policy_adapter(candidate):
        return candidate
    if isinstance(candidate, type):
        instance = candidate()
        if _is_policy_adapter(instance):
            return instance
    if callable(candidate):
        maybe_instance = candidate()
        if _is_policy_adapter(maybe_instance):
            return maybe_instance
    raise ValueError("Candidate is not a valid policy adapter")


def _register_module_adapters(module_name: str) -> None:
    module = importlib.import_module(module_name)
    for attr_name in ADAPTER_ATTR_NAMES:
        if not hasattr(module, attr_name):
            continue
        raw_candidate = getattr(module, attr_name)
        adapter = _coerce_adapter(raw_candidate)
        register_policy_adapter(adapter)


def _discover_builtin_adapters() -> None:
    package_name = "trade_assist.policy.adapters"
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        _register_module_adapters(module_info.name)


def _discover_entrypoint_adapters() -> None:
    try:
        selected = entry_points(group=ENTRYPOINT_GROUP)
    except TypeError:  # pragma: no cover
        selected = entry_points().select(group=ENTRYPOINT_GROUP)

    for ep in selected:
        try:
            loaded = ep.load()
            adapter = _coerce_adapter(loaded)
            register_policy_adapter(adapter)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to load policy adapter entry point '{ep.name}': {exc}"
            ) from exc


def _ensure_discovered() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _discover_builtin_adapters()
    _discover_entrypoint_adapters()
    _DISCOVERED = True


def refresh_policy_adapters() -> None:
    global _DISCOVERED
    _POLICY_ADAPTERS.clear()
    _DISCOVERED = False
    _ensure_discovered()


def register_policy_adapter(adapter: PolicyAdapter) -> None:
    policy_type = adapter.policy_type.strip().lower()
    if not policy_type:
        raise ValueError("Policy adapter must define a non-empty policy_type")
    _POLICY_ADAPTERS[policy_type] = adapter


def get_policy_adapter(policy_type: str) -> PolicyAdapter:
    _ensure_discovered()
    key = policy_type.strip().lower()
    if key not in _POLICY_ADAPTERS:
        known = ", ".join(sorted(_POLICY_ADAPTERS.keys()))
        raise ValueError(
            f"Unknown policy_type '{policy_type}'. "
            f"Known policy types: {known or 'none'}"
        )
    return _POLICY_ADAPTERS[key]


def list_policy_types() -> list[str]:
    _ensure_discovered()
    return sorted(_POLICY_ADAPTERS.keys())
