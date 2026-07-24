# 🚨 ===== [YULIN-MOD] START: 用纯 Python 自测音频切块和多轮环境，不依赖 GPU =====
"""本地自测：只验证纯 Python 的切块 env 逻辑（不依赖 GPU/relax 重栈）。

跑法（在 Relax 根目录）：
    python -m examples.simul_s2tt._selftest_env
"""

from __future__ import annotations

import math
import types

import numpy as np

from examples.simul_s2tt.audio_chunk_env import (
    DEFAULT_CHUNK_MS,
    DEFAULT_SAMPLE_RATE,
    AudioChunkEnv,
    build_env,
    split_audio_into_chunks,
)


def _mk_sample(waveform):
    # 模拟 Sample：只需 multimodal_inputs（三键，和 dataset 一致）。
    return types.SimpleNamespace(
        multimodal_inputs={"images": [], "videos": [], "audio": [waveform]},
    )


def _mk_args(chunk_ms=DEFAULT_CHUNK_MS, sr=DEFAULT_SAMPLE_RATE):
    return types.SimpleNamespace(simul_chunk_ms=chunk_ms, audio_sample_rate=sr)


def test_split_counts():
    sr, chunk_ms = 16000, 960
    chunk_len = int(round(sr * chunk_ms / 1000.0))  # 15360
    assert chunk_len == 15360

    # 4s -> 64000 样本 -> ceil(64000/15360)=5 块（末块 2560）
    wav = np.zeros(64000, dtype=np.float32)
    chunks = split_audio_into_chunks(wav, chunk_ms, sr)
    assert len(chunks) == math.ceil(64000 / chunk_len) == 5
    assert sum(c.shape[0] for c in chunks) == 64000  # 不丢样本
    assert chunks[-1].shape[0] == 64000 - 4 * chunk_len == 2560
    print(f"[ok] split 4s -> {len(chunks)} chunks, last={chunks[-1].shape[0]}")

    # 短于一个 chunk -> 单块（整段）
    short = np.zeros(1000, dtype=np.float32)
    assert len(split_audio_into_chunks(short, chunk_ms, sr)) == 1
    print("[ok] short audio -> 1 chunk")


def test_env_loop():
    sr, chunk_ms = 16000, 960
    wav = np.arange(64000, dtype=np.float32)  # 用递增值便于校验切块内容
    sample = _mk_sample(wav)
    args = _mk_args(chunk_ms, sr)

    env = build_env(sample, args)
    assert isinstance(env, AudioChunkEnv)
    assert env.num_chunks == 5

    # 首轮：sample.multimodal_inputs 只保留第 0 块，其余键不动
    assert len(sample.multimodal_inputs["audio"]) == 1
    first_chunk = sample.multimodal_inputs["audio"][0]
    assert first_chunk.shape[0] == 15360
    assert first_chunk[0] == 0.0 and first_chunk[-1] == 15359.0
    assert sample.multimodal_inputs["images"] == [] and sample.multimodal_inputs["videos"] == []
    print("[ok] initial turn carries chunk0 only, images/videos preserved")

    # 模拟 rollout 循环：首轮已用 chunk0，之后 step() 逐块吐，直到 done。
    env.reset()
    served = [first_chunk]  # 首轮
    fmt_ok = True
    for _ in range(100):  # 安全上限
        obs, done, info = env.step("模型输出(被忽略)")
        if done:
            assert info["chunk_index"] == info["num_chunks"] == 5
            break
        m = env.format_observation(obs)
        # observation 必须是一个含 audio 的 user turn
        assert m["role"] == "user"
        audio_items = [c for c in m["content"] if c.get("type") == "audio"]
        assert len(audio_items) == 1
        served.append(audio_items[0]["audio"])
    else:
        fmt_ok = False
    assert fmt_ok, "step 循环没有正常结束（可能死循环）"

    # 一共服务 5 块 = num_chunks；拼回去应等于原始波形（顺序/内容都对）。
    assert len(served) == env.num_chunks == 5
    recon = np.concatenate(served)
    assert recon.shape[0] == 64000
    assert np.array_equal(recon, wav)
    print(f"[ok] served {len(served)} chunks, reconstruct == original waveform")


def test_last_chunk_not_dropped():
    # 恰好整除时不应产生空尾块
    sr, chunk_ms = 16000, 1000  # chunk_len=16000
    wav = np.zeros(48000, dtype=np.float32)  # 正好 3 块
    chunks = split_audio_into_chunks(wav, chunk_ms, sr)
    assert len(chunks) == 3
    assert all(c.shape[0] == 16000 for c in chunks)
    print("[ok] exact-divisible -> no empty tail chunk")


if __name__ == "__main__":
    test_split_counts()
    test_env_loop()
    test_last_chunk_not_dropped()
    print("\nALL ENV SELFTESTS PASSED")
# 🚨 ===== [YULIN-MOD] END =====