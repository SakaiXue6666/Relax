# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""同传多轮自定义 generate（音频逐块）。

挂载方式（训练脚本）：
    --custom-generate-function-path examples.simul_s2tt.rollout.generate
    --custom-config-path examples/simul_s2tt/config.yaml

其中 config.yaml（键会被 setattr 到 args，任意键均可）至少含：
    max_turns: 64        # 安全上限；实际轮数由音频切块数决定
    simul_chunk_ms: 960  # 每块时长（可选，默认 960）
（env 在本文件内直接 import build_env，无需 --rollout-interaction-env-path。）

它只替换 sglang_rollout 里「生成一条样本」这一层：外层批量编排、并发信号量、reward、
abort 全部仍由核心处理。每轮都打同一个 sglang engine（同一 /generate 端点、同一 LoRA
adapter），只是把「调一次」变成「循环调 N 次并逐块注入音频」。

结构参考 examples/deepeyes/rollout.py，差异：
- 观测是音频 chunk 而非图像；
- 环境脚本化（step 忽略模型输出，直接吐下一块）；
- 音频必须走「双 token 流」：deepeyes 对图像可直接把 processor 展开后的 sample.tokens 发
  sglang，但音频在本套 sglang 上不接受「预展开」占位符（会与 audio_data 对不上导致引擎崩，
  已实测 503/worker died）。故对齐已验证的 s2tt 单轮核心路径：
    · sample.rollout_tokens（tokenizer 未展开，1 marker/段）+ audio_data → 发 sglang，引擎内部展开；
    · sample.tokens（processor 展开，含 N 个占位 token）+ multimodal_train_inputs → 训练用。

★ 首次真机仍需确认：LoRA rollout 是否带上 lora_path（见 _run_inference_step 的 NOTE(2)）。
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import torch

from examples.simul_s2tt.audio_chunk_env import build_env
from relax.engine.rollout.sglang_rollout import GenerateState
from relax.utils.data.processing_utils import (
    _ENCODE_EXECUTOR,
    encode_audio_for_rollout_engine,
)
from relax.utils.http_utils import post
from relax.utils.types import Sample


# 用于计算 observation 轮 chat 模板的「系统前言」长度，之后裁掉，只保留本轮 token。
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _run_processor(processor, tokenizer, text, multimodal_inputs: dict, args):
    """跑 Qwen3-Omni processor，返回 (input_ids, mm_train_inputs)。

    与核心 sglang_rollout._run_image_processor 完全一致：splat 整个
    multimodal_inputs（恒含 images/videos/audio 三键），避免与已验证 s2tt 路径有差异。
    """
    processor_output = processor(
        text=text,
        return_mm_token_type_ids=False,
        use_audio_in_video=getattr(args, "use_audio_in_video", False),
        **multimodal_inputs,
    )
    input_ids = processor_output["input_ids"][0]
    mm_train = {
        k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
        for k, v in processor_output.items()
        if k not in ("input_ids", "attention_mask")
    } or None
    return input_ids, mm_train


def _encode_initial_inputs(sample: Sample, processor, tokenizer, args):
    """首轮编码，返回 (unexp_ids, exp_ids, audio_data, mm_train)。

    ★ 关键（对齐已验证 s2tt 核心路径 sglang_rollout.generate）：
    - unexp_ids：tokenizer.encode(prompt)，音频占位符「未展开」（每段 1 个 marker）。
      → 发给 sglang 的 input_ids 用它 + audio_data，由 sglang 内部展开（音频这条链必须如此，
        否则 sglang 拿到「预展开」占位符会与 audio_data 对不上而崩）。
    - exp_ids：processor 展开后的 input_ids（含 N 个音频占位 token）+ mm_train 特征。
      → 训练侧 sample.tokens 用它（与 multimodal_train_inputs 对齐）。
    env 已把 sample.multimodal_inputs["audio"] 换成 [chunk0]（images/videos 键保留）。
    """
    mm = sample.multimodal_inputs or {}
    audios = list(mm.get("audio") or [])
    unexp_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    if processor is None:
        return unexp_ids, unexp_ids, [], None

    exp_ids, mm_train = _run_processor(processor, tokenizer, sample.prompt, mm, args)
    sample_rate = int(getattr(args, "audio_sample_rate", None) or 16000)
    audio_data = [encode_audio_for_rollout_engine(a, sample_rate) for a in audios]
    return unexp_ids, exp_ids, audio_data, mm_train


def _strip_bos(tokenizer, ids: list[int]) -> list[int]:
    bos_id = tokenizer.bos_token_id
    if bos_id is not None and ids and ids[0] == bos_id:
        return ids[1:]
    return ids


def _encode_audio_observation(chunk, processor, tokenizer, args):
    """后续轮：把「一个音频 chunk 的 user turn」编码，返回 (exp_ids, unexp_ids, audio_str, mm_train)。

    仿 deepeyes._encode_observation_for_generation：套 chat 模板 + 裁掉系统前言，只保留本轮
    新增 token。同样维护两条流：
    - unexp_ids：tokenizer.encode(本轮 chat 串)，音频「未展开」→ 追加到 rollout_tokens 发 sglang。
    - exp_ids：processor 展开（含 N 个占位 token）→ 追加到 sample.tokens 供训练。
    dummy 前言无音频，故 trim_length（tokenizer 计）对两条流都适用。
    """
    message = {"role": "user", "content": [{"type": "audio", "audio": chunk}]}
    sample_rate = int(getattr(args, "audio_sample_rate", None) or 16000)
    audio_str = encode_audio_for_rollout_engine(chunk, sample_rate)

    apply_kwargs = args.apply_chat_template_kwargs or {}
    trim_length = 0
    if args.apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES, tokenize=False, add_generation_prompt=False, **apply_kwargs
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message], tokenize=False, add_generation_prompt=True, **apply_kwargs
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = tokenizer.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True, **apply_kwargs
        )

    unexp_full = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    unexp_ids = unexp_full[trim_length:] if trim_length else unexp_full
    unexp_ids = _strip_bos(tokenizer, unexp_ids)

    if processor is None:
        return unexp_ids, unexp_ids, audio_str, None

    # images/videos 必须是 None（不是 []）：processor 对空 list 仍会进图像处理链，在
    # group_images_by_shape([]) 处越界崩溃；核心 process_multimodal_info 无图像时也返回 None。
    mm_obs = {"images": None, "videos": None, "audio": [chunk]}
    exp_ids, mm_train = _run_processor(processor, tokenizer, formatted_prompt, mm_obs, args)
    if trim_length:
        exp_ids = exp_ids[trim_length:]
    exp_ids = _strip_bos(tokenizer, exp_ids)

    return exp_ids, unexp_ids, audio_str, mm_train


# 音频特征键：各块帧数（最后一维）不同，需先 pad 到 max 再 cat dim0，复现「单次 processor
# 多音频调用」的规范布局（input_features [N,128,maxT] / feature_attention_mask [N,maxT]）。
# 与 megatron/data.py 的 PAD_RULES 语义一致（那里跨样本 pad，这里样本内 pad）。
_AUDIO_PAD_LAST_DIM = {"input_features": 0.0, "feature_attention_mask": 0}


def _merge_mm_train(chunks: list[dict | None]) -> dict | None:
    """把各块 mm_train_inputs 按 key 合并（只合并 torch.Tensor 值）。

    - 音频特征键（帧数变长）：pad 最后一维到 max 后 cat dim0；
    - 其余键（如定长网格）：直接 cat dim0。
    """
    values_by_key: dict[str, list] = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)

    merged: dict[str, Any] = {}
    for key, values in values_by_key.items():
        if not all(isinstance(v, torch.Tensor) for v in values):
            continue
        if len(values) == 1:
            merged[key] = values[0]
        elif key in _AUDIO_PAD_LAST_DIM:
            pad_val = _AUDIO_PAD_LAST_DIM[key]
            max_last = max(v.shape[-1] for v in values)
            padded = [
                torch.nn.functional.pad(v, [0, max_last - v.shape[-1]], value=pad_val)
                if v.shape[-1] < max_last
                else v
                for v in values
            ]
            merged[key] = torch.cat(padded, dim=0)
        else:
            merged[key] = torch.cat(values, dim=0)
    return merged or None


def _append_generated(sample: Sample, response_tokens: list[int], tokens: list[int], logprobs: list[float]):
    """模型生成 token：两条流里一致（纯文本 token，无音频占位），loss_mask=1 参与训练。"""
    sample.tokens.extend(tokens)
    sample.rollout_tokens.extend(tokens)
    response_tokens.extend(tokens)
    sample.loss_mask.extend([1] * len(tokens))
    sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def _append_observation(sample: Sample, response_tokens: list[int], exp_ids: list[int], unexp_ids: list[int]):
    """注入的音频观测：训练流(sample.tokens)用展开 ids，sglang 流(rollout_tokens)用未展开 ids；
    loss_mask=0 不参与训练。response_tokens/loss_mask/logprobs 都跟展开流对齐。"""
    sample.tokens.extend(exp_ids)
    response_tokens.extend(exp_ids)
    sample.loss_mask.extend([0] * len(exp_ids))
    sample.rollout_log_probs.extend([0.0] * len(exp_ids))
    sample.rollout_tokens.extend(unexp_ids)
    sample.response_length = len(response_tokens)


async def _run_inference_step(url, tokens, sampling_params, audio_data, args):
    payload = {
        "input_ids": tokens,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    # NOTE(2): LoRA 训练时 rollout 必须引用刚热推上去的同名 adapter，否则用的是 base。
    if getattr(args, "lora_enable", False):
        payload["lora_path"] = getattr(args, "lora_name", None) or "policy"
    if audio_data:
        payload["audio_data"] = audio_data

    output = await post(url, payload)
    meta = output["meta_info"]
    if "output_token_logprobs" in meta:
        new_tokens = [item[1] for item in meta["output_token_logprobs"]]
        new_logprobs = [item[0] for item in meta["output_token_logprobs"]]
    else:
        new_tokens, new_logprobs = [], []
    finish_type = meta["finish_reason"]["type"]
    return output["text"], new_tokens, new_logprobs, finish_type


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """同传多轮 rollout：整段音频切块，每轮喂一块并生成一段增量译文。"""
    state = GenerateState(args)
    tokenizer, processor = state.tokenizer, state.processor
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        raise ValueError("simul_s2tt 需要 --max-turns（作为安全上限；实际轮数由切块数决定）。")

    # build_env 会把整段音频切块，并把「第 0 块」写回 sample.multimodal_inputs（首轮带音频）。
    env = build_env(sample, args)
    env.reset()

    loop = asyncio.get_running_loop()

    # ---- 首轮输入编码（双流：unexp 发 sglang / exp 供训练）----
    unexp_prompt_ids, exp_prompt_ids, audio_data, init_mm_train = await loop.run_in_executor(
        _ENCODE_EXECUTOR, _encode_initial_inputs, sample, processor, tokenizer, args
    )
    # 两条 token 流（对齐已验证 s2tt 核心路径）：
    # - sample.rollout_tokens（未展开）→ 每轮发 sglang 的 input_ids，配 audio_data 由 sglang 展开。
    # - sample.tokens（processor 展开，含音频占位 token）→ 训练用，与 multimodal_train_inputs 对齐。
    sample.tokens = list(exp_prompt_ids)
    sample.rollout_tokens = list(unexp_prompt_ids)
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    response_tokens: list[int] = []  # 展开流的 gen+obs，保持与 sample.tokens/loss_mask/response_length 对齐
    response_text_parts: list[str] = []  # 只收模型各轮生成文本（干净译文，喂 BLEU）
    sample.response_length = 0

    mm_train_buffer: list[dict | None] = []
    if init_mm_train:
        mm_train_buffer.append(init_mm_train)

    sample.metadata = sample.metadata or {}

    # 总生成预算（只算模型 token，不含观测音频）：防多轮累积把序列撑爆。simul 脚本没设
    # --rollout-max-context-len，故这里用 rollout_max_response_len 收口，逐轮扣减。
    max_resp = getattr(args, "rollout_max_response_len", None)
    generated_count = 0

    stop_reason = "completed"
    for turn_idx in range(max_turns):
        cur_sp = sampling_params.copy()
        if max_resp is not None:
            remaining = max_resp - generated_count
            if remaining <= 0:
                sample.status = Sample.Status.TRUNCATED
                stop_reason = "budget_exhausted"
                break
            cur_sp["max_new_tokens"] = min(cur_sp.get("max_new_tokens", remaining), remaining)

        # 发给 sglang 的是「未展开」的 rollout_tokens + audio_data（sglang 内部展开音频占位）。
        _text, new_tokens, new_logprobs, finish_type = await _run_inference_step(
            url, sample.rollout_tokens, cur_sp, audio_data, args
        )
        # 模型生成的 token 参与训练（loss_mask=1），两条流一致。
        _append_generated(sample, response_tokens, new_tokens, new_logprobs)
        response_text_parts.append(_text)
        generated_count += len(new_tokens)

        if finish_type == "abort":
            sample.status = Sample.Status.ABORTED
            stop_reason = "abort"
            break

        # 某轮把预算打满（length 截断）：本轮译文已被切断，继续下一块无意义，收口。
        if finish_type == "length":
            sample.status = Sample.Status.TRUNCATED
            stop_reason = "length"
            break

        # 脚本化推进：拿下一块音频（step 忽略模型输出）。
        observation, done, _info = env.step(_text)
        if done:
            sample.status = Sample.Status.COMPLETED
            stop_reason = "chunks_exhausted"
            break

        obs_message = env.format_observation(observation)
        chunk = None
        for c in obs_message["content"]:
            if c.get("type") == "audio":
                chunk = c["audio"]
                break

        obs_exp_ids, obs_unexp_ids, obs_audio_str, obs_mm_train = await loop.run_in_executor(
            _ENCODE_EXECUTOR, _encode_audio_observation, chunk, processor, tokenizer, args
        )
        # 注入的观测 token 不参与训练（loss_mask=0）：exp 进训练流，unexp 进 sglang 流。
        _append_observation(sample, response_tokens, obs_exp_ids, obs_unexp_ids)
        if obs_audio_str is not None:
            audio_data = (audio_data or []) + [obs_audio_str]
        if obs_mm_train:
            mm_train_buffer.append(obs_mm_train)

        if turn_idx + 1 >= max_turns:
            sample.status = Sample.Status.TRUNCATED
            stop_reason = "max_turns"
            break

    sample.multimodal_train_inputs = _merge_mm_train(mm_train_buffer)
    # response 用各轮「生成文本」拼接（干净译文）而非 decode(response_tokens)：后者含观测音频
    # 占位符 token，会污染 BLEU。response_length 仍用 token 数（含 obs），供训练对齐 loss_mask。
    sample.response = "".join(response_text_parts)
    sample.response_length = len(response_tokens)
    # status 默认是 PENDING（非 None）；正常结束的分支都已显式置位，这里兜底避免残留 PENDING。
    if sample.status in (None, Sample.Status.PENDING):
        sample.status = Sample.Status.COMPLETED
    sample.metadata["simul_stop_reason"] = stop_reason
    sample.metadata["simul_num_chunks"] = env.num_chunks
    return sample
