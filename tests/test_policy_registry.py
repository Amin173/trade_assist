from __future__ import annotations

import pytest

from trade_assist.policy import (
    get_policy_adapter,
    list_policy_types,
    refresh_policy_adapters,
    register_policy_adapter,
)


def test_policy_registry_includes_v1():
    refresh_policy_adapters()
    types = list_policy_types()
    assert "v1" in types

    adapter = get_policy_adapter("v1")
    assert adapter.policy_type == "v1"
    schema = adapter.policy_schema()
    assert schema["title"] == "trade_assist policy config"


def test_get_policy_adapter_rejects_unknown_policy_type():
    refresh_policy_adapters()
    with pytest.raises(ValueError):
        get_policy_adapter("does_not_exist")


def test_register_policy_adapter_supports_runtime_extension():
    class DummyAdapter:
        policy_type = "dummy"

        def policy_schema(self):
            return {"type": "object"}

        def default_policy_payload(self):
            return {}

        def build_policy_config(self, payload=None):
            return payload or {}

        def run_backtest(self, **kwargs):
            raise NotImplementedError

        def recommend(self, **kwargs):
            raise NotImplementedError

    refresh_policy_adapters()
    register_policy_adapter(DummyAdapter())
    assert get_policy_adapter("dummy").policy_type == "dummy"
