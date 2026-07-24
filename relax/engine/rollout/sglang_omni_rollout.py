# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# 🚨 ===== [YULIN-MOD] START: (omni) 声明 Omni engine 和 abort hook，同时复用标准 rollout 编排 =====

# 这个文件可以理解成 Omni rollout 的“身份证”。
#
# Relax 会根据 --rollout-function-path 找到当前 rollout 模块，
# 然后读取模块声明的两个可选配置：
#
# ROLLOUT_ENGINE_CLASS
# → 告诉 Relax 这次 rollout 应使用 SGLangOmniEngine，
#   而不是默认的标准 SGLangEngine。
#
# ROLLOUT_ABORT_FUNCTION
# → 告诉 Relax 终止本轮请求时应调用 Omni 自己的 abort hook，
#   而不是标准 SGLang router 的 /workers + /abort_request。
#
# 这里只声明 Omni 与标准 SGLang 不同的部分。
#
# 真正的批量调度、并发限制、Sample 管理、reward 计算和数据传输，
# 仍然复用：
#
# relax.engine.rollout.sglang_rollout.generate_rollout
#
# 因此这里没有复制一整套 rollout 编排，也没有替换标准 SGLang 路径。
# 普通 rollout 模块不声明这两个配置时，仍然使用原来的 SGLangEngine
# 和原来的 /abort_request 终止逻辑。
#
# Omni router 没有标准 SGLang 的：
#
# Router /workers
# → 找到所有 worker
# → 对每个 worker 调 /abort_request
#
# 这是因为一个 Omni 请求可能同时跨过 Thinker、Talker、Code2Wav
# 等多个 stage，不能只终止某个普通 SGLang worker。
#
# 所以 Omni 的 abort hook 会：
#
# pause_generation(mode="abort")
# → 统一终止并清理尚未完成的 Omni 请求
# → continue_generation
# → 恢复服务，让下一轮 rollout 继续复用同一个外部 Omni router

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

# 🚨 ===== [YULIN-MOD] END =====