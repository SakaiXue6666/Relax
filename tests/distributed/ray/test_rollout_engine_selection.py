# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.backends.sglang.sglang_engine import SGLangEngine
from relax.backends.sglang_omni.omni_engine import SGLangOmniEngine
from relax.distributed.ray.rollout import _resolve_rollout_engine_class


def test_standard_rollout_keeps_standard_sglang_engine() -> None:
    args = SimpleNamespace(
        rollout_function_path="relax.engine.rollout.sglang_rollout.generate_rollout"
    )

    assert _resolve_rollout_engine_class(args) is SGLangEngine


def test_omni_rollout_selects_independent_omni_engine() -> None:
    args = SimpleNamespace(
        rollout_function_path=(
            "relax.engine.rollout.sglang_omni_rollout.generate_rollout"
        )
    )

    assert _resolve_rollout_engine_class(args) is SGLangOmniEngine


def test_invalid_engine_marker_fails_clearly(monkeypatch) -> None:
    module = SimpleNamespace(ROLLOUT_ENGINE_CLASS="not-a-class")
    monkeypatch.setattr(
        "relax.distributed.ray.rollout.importlib.import_module",
        lambda _: module,
    )
    args = SimpleNamespace(rollout_function_path="custom.rollout.generate")

    with pytest.raises(ValueError, match="fully qualified class path"):
        _resolve_rollout_engine_class(args)
