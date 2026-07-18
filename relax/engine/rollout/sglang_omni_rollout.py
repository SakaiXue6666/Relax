# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Standard Relax rollout orchestration backed by an external SGLang-Omni router."""

from typing import Any

from relax.engine.rollout.sglang_rollout import generate_rollout
from relax.utils.http_utils import post


ROLLOUT_ENGINE_CLASS = "relax.backends.sglang_omni.omni_engine.SGLangOmniEngine"
ROLLOUT_ABORT_FUNCTION = (
    "relax.engine.rollout.sglang_omni_rollout.abort_rollout"
)


async def abort_rollout(args: Any) -> None:
    """Abort outstanding Omni generations without a standard SGLang router."""
    base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    pause_result = await post(
        f"{base_url}/pause_generation",
        {"mode": "abort"},
    )
    if not pause_result.get("success"):
        raise RuntimeError(f"SGLang-Omni abort failed: {pause_result}")
    continue_result = await post(
        f"{base_url}/continue_generation",
        {},
    )
    if not continue_result.get("success"):
        raise RuntimeError(f"SGLang-Omni resume failed: {continue_result}")


__all__ = ["abort_rollout", "generate_rollout"]
