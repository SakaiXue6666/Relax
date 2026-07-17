# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Standard Relax rollout orchestration backed by an external SGLang-Omni router."""

from relax.engine.rollout.sglang_rollout import generate_rollout


ROLLOUT_ENGINE_CLASS = "relax.backends.sglang_omni.omni_engine.SGLangOmniEngine"

__all__ = ["generate_rollout"]
