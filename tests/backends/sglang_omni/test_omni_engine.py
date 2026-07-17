# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.backends.sglang_omni.omni_engine import SGLangOmniEngine


def _args(**overrides) -> SimpleNamespace:
    values = {
        "rollout_external": True,
        "rollout_external_engine_addrs": ["omni.example:30000"],
        "fully_async": False,
        "offload_rollout": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_omni_engine_init_uses_health_without_router_registration(monkeypatch) -> None:
    calls: list[tuple[str, float]] = []

    class Response:
        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "relax.backends.sglang_omni.omni_engine.requests.get",
        lambda url, timeout: calls.append((url, timeout)) or Response(),
    )
    engine = SGLangOmniEngine(_args(), rank=0)

    engine.init(
        dist_init_addr="omni.example:30000",
        port=30000,
        nccl_port=None,
        host="omni.example",
        router_ip="omni.example",
        router_port=30000,
    )

    assert engine.get_url() == "http://omni.example:30000"
    assert calls == [
        ("http://omni.example:30000/health", 10),
        ("http://omni.example:30000/model_info", 10),
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rollout_external": False}, "--rollout-external"),
        ({"fully_async": True}, "--fully-async"),
        ({"offload_rollout": True}, "offload"),
        ({"rollout_external_engine_addrs": []}, "exactly one"),
        (
            {"rollout_external_engine_addrs": ["one:1", "two:2"]},
            "exactly one",
        ),
    ],
)
def test_omni_engine_rejects_unsupported_modes(overrides, message) -> None:
    engine = SGLangOmniEngine(_args(**overrides), rank=0)

    with pytest.raises(ValueError, match=message):
        engine._validate_mode()


def test_omni_engine_flush_and_tensor_load_use_admin_post(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"success": True}

    monkeypatch.setattr(
        "relax.backends.sglang.sglang_engine.requests.post",
        lambda url, json: calls.append((url, json)) or Response(),
    )
    engine = SGLangOmniEngine(_args(), rank=0)
    engine.node_rank = 0
    engine.server_host = "omni.example"
    engine.server_port = 30000
    payload = {
        "lora_name": "policy",
        "serialized_tensors": "encoded",
        "config_dict": {"peft_type": "LORA"},
        "load_format": "flattened_bucket",
    }

    assert engine.flush_cache() == {"success": True}
    assert (
        engine.load_lora_adapter_from_tensors(
            lora_name=payload["lora_name"],
            serialized_tensors=payload["serialized_tensors"],
            config_dict=payload["config_dict"],
            load_format=payload["load_format"],
        )
        == {"success": True}
    )
    assert calls == [
        ("http://omni.example:30000/flush_cache", {}),
        (
            "http://omni.example:30000/load_lora_adapter_from_tensors",
            payload | {"pinned": False},
        ),
    ]
