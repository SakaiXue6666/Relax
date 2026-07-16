# 🚨 ===== [YULIN-MOD] START =====
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""定长同传环境：把整段音频切成固定时长 chunk，每轮吐一个。

约定（与 rollout.py 配合）：
- ``build_env(sample, args)`` 时把整段音频从 ``sample.multimodal_inputs["audio"]``
  取出、按 ``simul_chunk_ms`` 切块，**只把第 0 块留在初始 prompt 里**（首轮就带音频），
  其余块由 ``step()`` 逐轮吐出。
- 环境是脚本化的：``step()`` 忽略模型输出，直接推进到下一块；chunk 用完 ``done=True``。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from examples.deepeyes.base_env import BaseInteractionEnv


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_MS = 960


def split_audio_into_chunks(
    waveform: np.ndarray, chunk_ms: int = DEFAULT_CHUNK_MS, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> list[np.ndarray]:
    """把 1-D 波形按固定时长切块；最后一块不足时长也保留（不丢尾音）。"""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    chunk_len = int(round(sample_rate * chunk_ms / 1000.0))
    if chunk_len <= 0 or waveform.shape[0] <= chunk_len:
        return [waveform]
    chunks = [waveform[i : i + chunk_len] for i in range(0, waveform.shape[0], chunk_len)]
    return [c for c in chunks if c.shape[0] > 0]


class AudioChunkEnv(BaseInteractionEnv):
    """脚本化的定长切块环境。"""

    def __init__(self, sample: Any, args: Any) -> None:
        self.args = args
        self.sample = sample

        chunk_ms = int(getattr(args, "simul_chunk_ms", DEFAULT_CHUNK_MS))
        sample_rate = int(getattr(args, "audio_sample_rate", None) or DEFAULT_SAMPLE_RATE)
        self.sample_rate = sample_rate

        mm = dict(sample.multimodal_inputs or {})
        audios = list(mm.get("audio") or [])
        assert len(audios) == 1, (
            f"AudioChunkEnv 期望初始恰好 1 段完整音频，实际拿到 {len(audios)} 段。"
            "请确认数据里每条样本只有一个 <audio> 占位符。"
        )

        self._chunks = split_audio_into_chunks(audios[0], chunk_ms=chunk_ms, sample_rate=sample_rate)

        # 首轮只带第 0 块；其余留给 step()。用新 dict/list，避免污染同组其他样本共享的对象。
        mm["audio"] = [self._chunks[0]]
        sample.multimodal_inputs = mm

        self._next_idx = 1

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._next_idx = 1

    def step(self, response_text: str):
        """脚本化推进：忽略模型输出，吐下一个 chunk；没有更多 chunk 则结束。

        返回 ``(observation, done, info)``：
        - 还有 chunk：observation 携带下一块音频，done=False；
        - 没有 chunk：done=True（上一轮已翻译完最后一块）。
        """
        info = {"chunk_index": self._next_idx, "num_chunks": len(self._chunks)}
        if self._next_idx >= len(self._chunks):
            return {}, True, info

        chunk = self._chunks[self._next_idx]
        self._next_idx += 1
        observation = {"role": "user", "audio": chunk, "obs_str": ""}
        return observation, False, info

    def format_observation(self, observation: dict) -> dict:
        """把观测拼成一个 user turn（音频 + 可选文本）。覆盖基类的「只处理图像」版本。"""
        observation = observation or {}
        content: list[dict] = []
        chunk = observation.get("audio")
        if chunk is not None:
            content.append({"type": "audio", "audio": chunk})
        obs_str = observation.get("obs_str", "")
        if obs_str:
            content.append({"type": "text", "text": obs_str})
        role = observation.get("role") or "user"
        return {"role": role, "content": content}


def build_env(sample: Any, args: Any) -> AudioChunkEnv:
    """rollout.py 通过 ``--rollout-interaction-env-path`` 找到这个工厂函数。"""
    return AudioChunkEnv(sample=sample, args=args)
# 🚨 ===== [YULIN-MOD] END =====