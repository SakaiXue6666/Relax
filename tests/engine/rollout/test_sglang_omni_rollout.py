# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from examples.simul_s2tt import omni_rollout
from examples.simul_s2tt.rollout import _run_inference_step as standard_inference_step
from relax.utils.types import Sample


def test_omni_audio_data_url_adapts_legacy_relax_encoding() -> None:
    payload = "UklGRg=="

    assert (
        omni_rollout._as_omni_audio_data_url(f"data:,{payload}")
        == f"data:audio/wav;base64,{payload}"
    )
    assert (
        omni_rollout._as_omni_audio_data_url(f"data:audio/wav;base64,{payload}")
        == f"data:audio/wav;base64,{payload}"
    )


def test_omni_inference_payload_uses_messages_audio_metadata_and_thinker_lora(
    monkeypatch,
) -> None:
    captured: list[dict] = []

    async def fake_post(url, payload):
        captured.append({"url": url, "payload": payload})
        return {
            "text": "hello",
            "meta_info": {
                "output_token_logprobs": [[-0.1, 7]],
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr(omni_rollout, "post", fake_post)
    messages = [{"role": "user", "content": [{"type": "audio"}]}]
    result = asyncio.run(
        omni_rollout._run_inference_step(
            "http://omni/generate",
            messages,
            {"max_new_tokens": 8},
            ["audio0"],
            SimpleNamespace(lora_enable=True, lora_name="policy"),
        )
    )

    assert result == ("hello", [7], [-0.1], "stop")
    assert captured == [
        {
            "url": "http://omni/generate",
            "payload": {
                "messages": messages,
                "sampling_params": {"max_new_tokens": 8},
                "metadata": {"audios": ["audio0"]},
                "return_logprob": True,
                "output_modalities": ["text"],
                "stage_params": {"thinker": {"lora_name": "policy"}},
            },
        }
    ]


def test_standard_sglang_inference_payload_is_unchanged(monkeypatch) -> None:
    captured: list[dict] = []

    async def fake_post(url, payload):
        captured.append(payload)
        return {
            "text": "hello",
            "meta_info": {
                "output_token_logprobs": [[-0.1, 7]],
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr("examples.simul_s2tt.rollout.post", fake_post)
    asyncio.run(
        standard_inference_step(
            "http://sglang/generate",
            [1, 2],
            {"max_new_tokens": 8},
            ["audio0"],
            SimpleNamespace(lora_enable=True, lora_name="policy"),
        )
    )

    assert captured == [
        {
            "input_ids": [1, 2],
            "sampling_params": {"max_new_tokens": 8},
            "return_logprob": True,
            "lora_path": "policy",
            "audio_data": ["audio0"],
        }
    ]


def test_omni_inference_requires_output_token_logprobs(monkeypatch) -> None:
    async def fake_post(url, payload):
        return {
            "text": "hello",
            "meta_info": {"finish_reason": {"type": "stop"}},
        }

    monkeypatch.setattr(omni_rollout, "post", fake_post)

    with pytest.raises(RuntimeError, match="missing output_token_logprobs"):
        asyncio.run(
            omni_rollout._run_inference_step(
                "http://omni/generate",
                [{"role": "user", "content": [{"type": "audio"}]}],
                {"max_new_tokens": 8},
                ["audio0"],
                SimpleNamespace(lora_enable=False),
            )
        )


def test_omni_multiturn_accumulates_messages_audios_and_training_alignment(
    monkeypatch,
) -> None:
    class FakeEnv:
        num_chunks = 3

        def __init__(self):
            self.index = 1

        def reset(self):
            self.index = 1

        def step(self, _):
            if self.index >= self.num_chunks:
                return {}, True, {}
            chunk = np.asarray([self.index], dtype=np.float32)
            self.index += 1
            return {"role": "user", "audio": chunk, "obs_str": ""}, False, {}

        def format_observation(self, observation):
            return {
                "role": "user",
                "content": [{"type": "audio", "audio": observation["audio"]}],
            }

    request_snapshots: list[dict] = []

    async def fake_inference(_, messages, __, encoded_audios, ___):
        turn = len(request_snapshots)
        request_snapshots.append(
            {
                "messages": [dict(message) for message in messages],
                "audios": list(encoded_audios),
            }
        )
        return f"text{turn}", [20 + turn], [-0.1 - turn], "stop"

    monkeypatch.setattr(
        omni_rollout,
        "GenerateState",
        lambda _: SimpleNamespace(tokenizer=object(), processor=object()),
    )
    monkeypatch.setattr(omni_rollout, "build_env", lambda sample, args: FakeEnv())
    monkeypatch.setattr(
        omni_rollout,
        "_encode_initial_inputs",
        lambda *args: ([1], [10, 11], ["audio0"], None),
    )
    monkeypatch.setattr(
        omni_rollout,
        "_encode_audio_observation",
        lambda chunk, *args: ([30, 31], [3], f"audio{int(chunk[0])}", None),
    )
    monkeypatch.setattr(omni_rollout, "_run_inference_step", fake_inference)
    sample = Sample(
        prompt=[{"role": "user", "content": [{"type": "audio"}]}],
        multimodal_inputs={"audio": [np.asarray([0.0], dtype=np.float32)]},
    )
    args = SimpleNamespace(
        max_turns=3,
        partial_rollout=False,
        optimize_routing_replay=False,
        rollout_max_response_len=8,
        sglang_router_ip="omni",
        sglang_router_port=30000,
    )

    result = asyncio.run(omni_rollout.generate(args, sample, {"max_new_tokens": 8}))

    assert [len(request["audios"]) for request in request_snapshots] == [1, 2, 3]
    assert [
        [message["role"] for message in request["messages"]]
        for request in request_snapshots
    ] == [
        ["user"],
        ["user", "assistant", "user"],
        ["user", "assistant", "user", "assistant", "user"],
    ]
    assert result.tokens == [10, 11, 20, 30, 31, 21, 30, 31, 22]
    assert result.loss_mask == [1, 0, 0, 1, 0, 0, 1]
    assert result.rollout_log_probs == [-0.1, 0.0, 0.0, -1.1, 0.0, 0.0, -2.1]
    assert result.response_length == 7
    assert result.status == Sample.Status.COMPLETED
