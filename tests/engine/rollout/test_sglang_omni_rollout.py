# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from examples.simul_s2tt import omni_rollout
from examples.simul_s2tt.rollout import _run_inference_step as standard_inference_step
from relax.engine.rollout import sglang_rollout
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


def test_omni_initial_encoding_uses_processor_owned_chat_template(monkeypatch) -> None:
    class FakeTokenizer:
        chat_template = None

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [
                {"role": "user", "content": [{"type": "audio"}]}
            ]
            assert self.chat_template == "official-omni-template"
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            return "rendered prompt"

        def encode(self, text, *, add_special_tokens):
            assert text == "rendered prompt"
            assert add_special_tokens is False
            return [1, 2]

    tokenizer = FakeTokenizer()
    processor = SimpleNamespace(chat_template="official-omni-template")
    monkeypatch.setattr(
        omni_rollout,
        "_run_processor",
        lambda *args: ([3, 4, 5], {"input_features": "features"}),
    )
    monkeypatch.setattr(
        omni_rollout,
        "encode_audio_for_rollout_engine",
        lambda audio, sample_rate: f"audio:{audio[0]}:{sample_rate}",
    )
    sample = Sample(
        prompt=[],
        multimodal_inputs={"audio": [np.asarray([0.25], dtype=np.float32)]},
    )

    result = omni_rollout._encode_initial_inputs(
        sample,
        [{"role": "user", "content": [{"type": "audio"}]}],
        processor,
        tokenizer,
        SimpleNamespace(
            apply_chat_template_kwargs=None,
            audio_sample_rate=16000,
        ),
    )

    assert tokenizer.chat_template == "official-omni-template"
    assert result == (
        [1, 2],
        [3, 4, 5],
        ["audio:0.25:16000"],
        {"input_features": "features"},
    )


@pytest.mark.parametrize("marker", ["<image>", "<video>"])
def test_omni_messages_reject_unsupported_media_markers(marker) -> None:
    with pytest.raises(NotImplementedError, match="supports audio and text only"):
        omni_rollout._sanitize_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": marker},
                        {"type": "audio"},
                    ],
                }
            ]
        )


def test_omni_abort_uses_pause_then_continue(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    async def fake_post(url, payload):
        calls.append((url, payload))
        return {"success": True}

    monkeypatch.setattr(
        "relax.engine.rollout.sglang_omni_rollout.post",
        fake_post,
    )
    from relax.engine.rollout.sglang_omni_rollout import abort_rollout

    asyncio.run(
        abort_rollout(
            SimpleNamespace(
                sglang_router_ip="omni",
                sglang_router_port=30000,
            )
        )
    )

    assert calls == [
        ("http://omni:30000/pause_generation", {"mode": "abort"}),
        ("http://omni:30000/continue_generation", {}),
    ]


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


def test_omni_inference_translates_real_relax_sampling_params(monkeypatch) -> None:
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

    monkeypatch.setattr(omni_rollout, "post", fake_post)
    asyncio.run(
        omni_rollout._run_inference_step(
            "http://omni/generate",
            [{"role": "user", "content": [{"type": "audio"}]}],
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": 8,
                "stop": None,
                "stop_token_ids": None,
                "skip_special_tokens": False,
                "no_stop_trim": True,
                "spaces_between_special_tokens": False,
                "sampling_seed": 42,
            },
            ["audio0"],
            SimpleNamespace(lora_enable=False),
        )
    )

    assert captured[0]["sampling_params"] == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": 8,
        "stop": None,
        "stop_token_ids": None,
        "seed": 42,
    }


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


def test_custom_omni_generate_skips_standard_group_multimodal_preencoding(
    monkeypatch,
) -> None:
    encode_calls: list[dict] = []

    async def fake_encode(multimodal_inputs):
        encode_calls.append(multimodal_inputs)
        return {"audio_data": ["unused"]}, 0.1

    async def fake_generate_and_rm(args, sample, sampling_params, evaluation=False):
        return sample

    monkeypatch.setattr(
        sglang_rollout,
        "GenerateState",
        lambda _: SimpleNamespace(aborted=False),
    )
    monkeypatch.setattr(
        sglang_rollout,
        "_encode_multimodal_inputs",
        fake_encode,
    )
    monkeypatch.setattr(
        sglang_rollout,
        "generate_and_rm",
        fake_generate_and_rm,
    )
    shared_multimodal_inputs = {"audio": [np.asarray([0.0], dtype=np.float32)]}
    group = [
        Sample(prompt=[], multimodal_inputs=shared_multimodal_inputs),
        Sample(prompt=[], multimodal_inputs=shared_multimodal_inputs),
    ]
    args = SimpleNamespace(
        custom_generate_function_path="examples.simul_s2tt.omni_rollout.generate",
        sglang_enable_deterministic_inference=False,
        group_rm=False,
    )

    result = asyncio.run(
        sglang_rollout.generate_and_rm_group(
            args,
            group,
            {"max_new_tokens": 8},
        )
    )

    assert result == group
    assert encode_calls == []
    assert not any(hasattr(sample, "_pre_encoded_mm") for sample in group)


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
