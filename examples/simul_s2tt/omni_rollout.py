# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang-Omni multi-turn S2TT rollout using structured messages."""

from __future__ import annotations

import asyncio
from typing import Any

from examples.simul_s2tt.audio_chunk_env import build_env
from examples.simul_s2tt.rollout import (
    _append_generated,
    _append_observation,
    _clean_gen_text,
    _encode_audio_observation,
    _merge_mm_train,
    _run_processor,
)
from relax.engine.rollout.sglang_rollout import GenerateState
from relax.utils.data.processing_utils import (
    _ENCODE_EXECUTOR,
    encode_audio_for_rollout_engine,
)
from relax.utils.http_utils import post
from relax.utils.types import Sample


_OMNI_SAMPLING_PARAM_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "stop",
    "stop_token_ids",
    "seed",
    "max_new_tokens",
    "max_tokens",
}


def _sanitize_sampling_params(
    sampling_params: dict[str, Any],
) -> dict[str, Any]:
    """Translate Relax sampling params to SGLang-Omni's strict protocol."""
    sanitized = {
        key: value
        for key, value in sampling_params.items()
        if key in _OMNI_SAMPLING_PARAM_KEYS
    }
    if "sampling_seed" in sampling_params:
        sanitized["seed"] = sampling_params["sampling_seed"]
    return sanitized


def _reject_unsupported_media_markers(text: str) -> None:
    markers = [marker for marker in ("<image>", "<video>") if marker in text]
    if markers:
        raise NotImplementedError(
            "The initial SGLang-Omni S2TT rollout supports audio and text only; "
            f"found unsupported marker(s): {', '.join(markers)}"
        )


def _sanitize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise ValueError(
            "SGLang-Omni rollout requires a structured prompt; do not use --apply-chat-template"
        )
    sanitized: list[dict[str, Any]] = []
    audio_markers = 0
    for original_message in messages:
        if not isinstance(original_message, dict):
            raise ValueError("Each SGLang-Omni prompt message must be a dictionary")
        message = dict(original_message)
        content = message.get("content", "")
        if isinstance(content, str):
            _reject_unsupported_media_markers(content)
            sanitized.append(message)
            continue
        if not isinstance(content, list):
            raise ValueError("SGLang-Omni message content must be a string or a list")
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("SGLang-Omni structured content parts must be dictionaries")
            part_type = part.get("type")
            if part_type == "audio":
                audio_markers += 1
                parts.append({"type": "audio"})
            elif part_type == "text":
                text = str(part.get("text") or "")
                _reject_unsupported_media_markers(text)
                parts.append({"type": "text", "text": text})
            elif part_type in {"image", "video"}:
                raise NotImplementedError(
                    "The initial SGLang-Omni S2TT rollout supports audio and text only"
                )
            else:
                parts.append(dict(part))
        message["content"] = parts
        sanitized.append(message)
    if audio_markers != 1:
        raise ValueError(
            f"SGLang-Omni S2TT expects exactly one initial audio marker, got {audio_markers}"
        )
    return sanitized


def _as_omni_audio_data_url(encoded_audio: str) -> str:
    """Adapt Relax's legacy SGLang audio encoding to a standards-compliant data URL."""
    if encoded_audio.startswith("data:,"):
        return "data:audio/wav;base64," + encoded_audio.removeprefix("data:,")
    return encoded_audio


def _ensure_omni_chat_template(
    tokenizer: Any,
    processor: Any,
    apply_kwargs: dict[str, Any],
) -> None:
    """Expose the processor-owned Qwen3-Omni template to the tokenizer."""
    if getattr(tokenizer, "chat_template", None) or apply_kwargs.get("chat_template"):
        return
    processor_template = getattr(processor, "chat_template", None)
    if not processor_template:
        raise ValueError(
            "Qwen3-Omni chat template is missing from both tokenizer and processor; "
            "provide it through --apply-chat-template-kwargs"
        )
    tokenizer.chat_template = processor_template


def _encode_initial_inputs(
    sample: Sample,
    messages: list[dict[str, Any]],
    processor: Any,
    tokenizer: Any,
    args: Any,
) -> tuple[list[int], list[int], list[str], dict | None]:
    mm_inputs = sample.multimodal_inputs or {}
    audios = list(mm_inputs.get("audio") or [])
    if len(audios) != 1:
        raise ValueError(
            f"SGLang-Omni S2TT expects one initial audio chunk, got {len(audios)}"
        )
    apply_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    _ensure_omni_chat_template(tokenizer, processor, apply_kwargs)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **apply_kwargs,
    )
    unexpanded_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if processor is None:
        expanded_ids = unexpanded_ids
        mm_train = None
    else:
        expanded_ids, mm_train = _run_processor(
            processor,
            tokenizer,
            prompt_text,
            mm_inputs,
            args,
        )
    sample_rate = int(getattr(args, "audio_sample_rate", None) or 16000)
    encoded_audios = [
        encode_audio_for_rollout_engine(audio, sample_rate) for audio in audios
    ]
    return unexpanded_ids, expanded_ids, encoded_audios, mm_train


async def _run_inference_step(
    url: str,
    messages: list[dict[str, Any]],
    sampling_params: dict[str, Any],
    encoded_audios: list[str],
    args: Any,
) -> tuple[str, list[int], list[float], str]:
    payload: dict[str, Any] = {
        "messages": messages,
        "sampling_params": _sanitize_sampling_params(sampling_params),
        "metadata": {
            "audios": [_as_omni_audio_data_url(audio) for audio in encoded_audios]
        },
        "return_logprob": True,
        "output_modalities": ["text"],
    }
    if getattr(args, "lora_enable", False):
        payload["stage_params"] = {
            "thinker": {"lora_name": getattr(args, "lora_name", None) or "policy"}
        }
    output = await post(url, payload)
    meta = output.get("meta_info") or {}
    token_logprobs = meta.get("output_token_logprobs")
    if not isinstance(token_logprobs, list):
        raise RuntimeError("SGLang-Omni response is missing output_token_logprobs")
    try:
        new_logprobs = [float(item[0]) for item in token_logprobs]
        new_tokens = [int(item[1]) for item in token_logprobs]
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("SGLang-Omni returned malformed output_token_logprobs") from exc
    if len(new_tokens) != len(new_logprobs):
        raise RuntimeError("SGLang-Omni token and logprob counts do not match")
    finish_reason = meta.get("finish_reason")
    if not isinstance(finish_reason, dict) or not finish_reason.get("type"):
        raise RuntimeError("SGLang-Omni response is missing finish_reason.type")
    return str(output.get("text") or ""), new_tokens, new_logprobs, str(finish_reason["type"])


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate one chunked S2TT sample against an external SGLang-Omni router."""
    if getattr(args, "partial_rollout", False) or sample.response_length:
        raise NotImplementedError("SGLang-Omni multi-turn partial rollout is not supported")
    if getattr(args, "optimize_routing_replay", False):
        raise NotImplementedError("SGLang-Omni multi-turn routing replay is not supported")

    max_turns = getattr(args, "max_turns", None)
    if max_turns is None or max_turns <= 0:
        raise ValueError("SGLang-Omni simul rollout requires max_turns > 0")

    state = GenerateState(args)
    tokenizer, processor = state.tokenizer, state.processor
    messages = _sanitize_messages(sample.prompt)
    env = build_env(sample, args)
    env.reset()
    loop = asyncio.get_running_loop()
    unexpanded_ids, expanded_ids, encoded_audios, initial_mm_train = await loop.run_in_executor(
        _ENCODE_EXECUTOR,
        _encode_initial_inputs,
        sample,
        messages,
        processor,
        tokenizer,
        args,
    )

    sample.tokens = list(expanded_ids)
    sample.rollout_tokens = list(unexpanded_ids)
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.response_length = 0
    sample.metadata = sample.metadata or {}
    response_tokens: list[int] = []
    response_text_parts: list[str] = []
    mm_train_buffer = [initial_mm_train] if initial_mm_train else []
    max_response_len = getattr(args, "rollout_max_response_len", None)
    generated_count = 0
    stop_reason = "completed"
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    for turn_idx in range(max_turns):
        current_sampling_params = sampling_params.copy()
        if max_response_len is not None:
            remaining = max_response_len - generated_count
            if remaining <= 0:
                sample.status = Sample.Status.TRUNCATED
                stop_reason = "budget_exhausted"
                break
            current_sampling_params["max_new_tokens"] = min(
                current_sampling_params.get("max_new_tokens", remaining),
                remaining,
            )

        text, new_tokens, new_logprobs, finish_type = await _run_inference_step(
            url,
            messages,
            current_sampling_params,
            encoded_audios,
            args,
        )
        _append_generated(sample, response_tokens, new_tokens, new_logprobs)
        clean_text = _clean_gen_text(text)
        response_text_parts.append(clean_text)
        generated_count += len(new_tokens)

        if finish_type == "abort":
            sample.status = Sample.Status.ABORTED
            stop_reason = "abort"
            break
        if finish_type == "length":
            sample.status = Sample.Status.TRUNCATED
            stop_reason = "length"
            break

        observation, done, _ = env.step(text)
        if done:
            sample.status = Sample.Status.COMPLETED
            stop_reason = "chunks_exhausted"
            break

        observation_message = env.format_observation(observation)
        audio_parts = [
            part
            for part in observation_message["content"]
            if part.get("type") == "audio"
        ]
        if len(audio_parts) != 1:
            raise RuntimeError("SGLang-Omni environment must return one audio chunk")
        chunk = audio_parts[0]["audio"]
        expanded_observation, unexpanded_observation, encoded_audio, mm_train = (
            await loop.run_in_executor(
                _ENCODE_EXECUTOR,
                _encode_audio_observation,
                chunk,
                processor,
                tokenizer,
                args,
            )
        )
        if encoded_audio is None:
            raise RuntimeError("Failed to encode SGLang-Omni audio observation")
        _append_observation(
            sample,
            response_tokens,
            expanded_observation,
            unexpanded_observation,
        )
        encoded_audios.append(encoded_audio)
        if mm_train:
            mm_train_buffer.append(mm_train)
        messages.append({"role": "assistant", "content": clean_text})
        messages.append({"role": "user", "content": [{"type": "audio"}]})

        if turn_idx + 1 >= max_turns:
            sample.status = Sample.Status.TRUNCATED
            stop_reason = "max_turns"
            break

    sample.multimodal_train_inputs = _merge_mm_train(mm_train_buffer)
    sample.response = "".join(response_text_parts)
    sample.response_length = len(response_tokens)
    if sample.status in (None, Sample.Status.PENDING):
        sample.status = Sample.Status.COMPLETED
    sample.metadata["simul_stop_reason"] = stop_reason
    sample.metadata["simul_num_chunks"] = env.num_chunks
    if len(sample.loss_mask) != sample.response_length:
        raise RuntimeError("SGLang-Omni loss mask and response lengths do not match")
    if len(sample.rollout_log_probs) != sample.response_length:
        raise RuntimeError(
            "SGLang-Omni rollout logprob and response lengths do not match"
        )
    if len(sample.tokens) != len(expanded_ids) + sample.response_length:
        raise RuntimeError("SGLang-Omni training token alignment is inconsistent")
    return sample
