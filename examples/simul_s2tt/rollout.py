# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# 🚨 ===== [YULIN-MOD] START: 将整段音频切成多块，执行多轮同传 rollout =====

# 整段音频
#   ↓ build_env 切块
# chunk 0 + 初始 prompt
#   ↓
# 模型生成第 1 段译文
#   ↓ 注入 chunk 1
# 模型生成第 2 段译文
#   ↓ 注入 chunk 2
# ……
#   ↓
# 拼接所有增量译文
#   ↓
# 得到训练 token、loss mask、行为 logprob、多模态特征和 BLEU 文本

# =======================================================

# 数据	| 用途	| 音频占位符
# sample.rollout_tokens	| 发给 SGLang 推理	| 未展开，每段音频一个 marker
# sample.tokens	| 交给 Megatron 训练 | processor 已展开，与音频特征对齐

# =======================================================

# 完整逻辑轨迹：audio1 → text1 → audio2 → text2
#
# 同一条逻辑轨迹，在 SGLang 推理和 Megatron 训练中使用不同的音频表示。
#
# （为简化说明，以上省略了 system、user、assistant 等 chat-template 边界 token。）
#
# 一、SGLang 推理流
#
# sample.rollout_tokens = [
#     AUDIO_MARKER_1,       # 表示这里需要插入第1段音频
#     text1_tokens,
#     AUDIO_MARKER_2,       # 表示这里需要插入第2段音频
#     text2_tokens,
# ]
#
# audio_data = [
#     AUDIO_DATA_1, 
#     AUDIO_DATA_2,
# ]
#
#
# 二、Megatron 训练流
#
# sample.tokens = [
#     A1, A2, A3, A4,               # audio1 展开的占位 token
#     text1_tokens,
#     B1, B2, B3, B4, B5, B6,       # audio2 展开的占位 token
#     text2_tokens,
# ]
#
# _merge_mm_train() 将两份特征 pad 并沿 dim=0 合并：
#
# audio1 → text1 → audio2 → text2 → audio3
#   │                 │                 │
#   ▼                 ▼                 ▼
# feature1          feature2          feature3
# [1,128,100]       [1,128,130]       [1,128,80]
#   │                 │                 │
#   └────── pad 最后一维到 130 ─────────┘
#                 │
#                 ▼
#       input_features
#          [3,128,130]
#
#
# 训练轨迹：
# audio1, text1, audio2, text2, audio3, text3, ...
#
# loss mask：
#    0,     1,      0,     1,      0,     1, ...
#
# BLEU 文本：
# text1 + text2 + text3 + ... + textN


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

==================================

Qwen3-Omni 同传 S2TT 的自定义多轮 generate。

这个模块把普通的一次 /generate 请求扩展为多次请求：

1. build_env 将整段音频切成若干 chunk；
2. 首轮输入包含 chunk 0；
3. 模型每生成一段增量译文后，环境注入下一个音频 chunk；
4. 重复生成，直到音频块耗尽、达到生成预算、达到 max_turns 或请求被中止；
5. 将所有轮次的干净译文拼接为 sample.response；
6. 同时构造供 Megatron 训练使用的 tokens、loss_mask、logprob 和音频特征。

这里只负责单个 Sample 的生成过程。
批量调度、并发限制、reward 计算和 abort 管理由 Relax 外层负责。

音频必须维护两条 token 流：

- sample.rollout_tokens：
  tokenizer 编码的未展开音频 marker，连同 audio_data 发给 SGLang；
  音频占位符由 SGLang 内部展开。

- sample.tokens：
  Qwen3-Omni processor 展开后的音频占位 token；
  与 multimodal_train_inputs 对齐，供 Megatron 训练。

不能直接把 processor 预展开后的音频 token 发给当前 SGLang，
否则预展开 token 数可能与 audio_data 无法对应，导致请求 503 或 worker 崩溃。
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import numpy as np
import torch


# 匹配 <|im_end|>、<|endoftext|> 等 Qwen 风格特殊 token。
# sglang 每轮返回的 text 会带结束符（如 <|im_end|>）等特殊 token。拼进 sample.response 会污染
# BLEU：既凭空多出参考里没有的 token 拉低 precision，更会把「跨块」的 2/3/4-gram 打断（每两块
# 之间插一个 marker），让高阶 n-gram 几乎全废。实测每样本平均 ~9.5 个 marker，BLEU 被压到真实
# 值的 ~40%（7.2 vs 17.9）。故拼接 response 前先去掉所有 <|...|> 特殊 token。
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


def _clean_gen_text(text: str) -> str:
    """去掉模型生成文本里的 <|...|> 特殊 token（喂 BLEU 的干净译文用）。"""
    # 这个函数只影响 sample.response，也就是最终送去计算 BLEU 的文本；
    # 不修改训练使用的 token IDs。

    # 如果将每轮末尾的 <|im_end|> 直接拼入译文：

    # 1. BLEU 会把它视为参考译文中不存在的额外 token；
    # 2. 它会插在相邻音频块的译文之间；
    # 3. 跨块的 2/3/4-gram 会被特殊 token 截断；
    # 4. BLEU 会显著低估真实翻译质量。
    return _SPECIAL_TOKEN_RE.sub("", text or "")

from examples.simul_s2tt.audio_chunk_env import build_env
from relax.engine.rollout.sglang_rollout import GenerateState
from relax.engine.rollout.request_permit import GenerationAborted
from relax.utils.data.processing_utils import (
    encode_audio_for_rollout_engine,
    get_encode_executor,
)
from relax.utils.types import Sample


# 用于计算 observation 轮 chat 模板的「系统前言」长度，之后裁掉，只保留本轮 token。
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _run_processor(processor, tokenizer, text, multimodal_inputs: dict, args):
    """运行 Qwen3-Omni processor。

    返回：
        input_ids:
            processor 展开后的 token IDs，音频 marker 会展开成与音频特征对应的
            多个占位 token。

        mm_train:
            除 input_ids 和 attention_mask 以外的多模态训练输入，
            例如 input_features、feature_attention_mask 等。

    该函数生成的是 Megatron 训练侧需要的数据，
    不是直接发给 SGLang 的未展开 token 流。
    """
    processor_output = processor(
        # 已经套好 chat template 的文本。
        text=text,
        # 当前训练链路不需要 processor 返回 mm_token_type_ids。
        return_mm_token_type_ids=False,
        # 是否把视频中的音频一起作为输入。
        use_audio_in_video=getattr(args, "use_audio_in_video", False),
        # 传入 images、videos、audio 等完整多模态字段。
        **multimodal_inputs,
    )
    # processor 输出通常带 batch 维；
    # 当前函数一次只处理一条样本，因此取第 0 条。
    input_ids = processor_output["input_ids"][0]
    # 取出训练所需的多模态特征。
    #
    # input_ids 已经单独返回；
    # attention_mask 会由训练数据管线根据 token 序列重新处理，因此不放入这里。
    mm_train = {
        k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
        for k, v in processor_output.items()
        if k not in ("input_ids", "attention_mask")
    } or None
    return input_ids, mm_train


# 首轮执行后形成：
# sample.rollout_tokens = 未展开 prompt + 1 个 chunk0 marker
# sample.tokens         = 展开 prompt + chunk0 对应的 N 个音频 token
# audio_data            = [chunk0 的编码音频]
# mm_train              = chunk0 的训练特征
def _encode_initial_inputs(sample: Sample, processor, tokenizer, args):
    """首轮编码，返回 (unexp_ids, exp_ids, audio_data, mm_train)。

    ★ 关键（对齐已验证 s2tt 核心路径 sglang_rollout.generate）：
    - unexp_ids：tokenizer.encode(prompt)，音频占位符「未展开」（每段 1 个 marker）。
      → 发给 sglang 的 input_ids 用它 + audio_data，由 sglang 内部展开（音频这条链必须如此，
        否则 sglang 拿到「预展开」占位符会与 audio_data 对不上而崩）。
    - exp_ids：processor 展开后的 input_ids（含 N 个音频占位 token）+ mm_train 特征。
      → 训练侧 sample.tokens 用它（与 multimodal_train_inputs 对齐）。
    env 已把 sample.multimodal_inputs["audio"] 换成 [chunk0]（images/videos 键保留）。
    
    =======================================================

    编码首轮输入，建立训练流和推理流。

    返回：
        unexp_ids:
            tokenizer 编码的未展开 token。
            每段音频仍表现为一个 marker，用于 SGLang input_ids。

        exp_ids:
            processor 展开后的 token。
            音频 marker 已展开为多个占位 token，用于 Megatron 训练。

        audio_data:
            SGLang 接口接收的序列化音频数据列表。

        mm_train:
            Megatron 训练侧需要的音频特征。

    调用该函数前，build_env 已经：
        1. 将整段音频切块；
        2. 将 sample.multimodal_inputs["audio"] 替换为 [chunk0]。
    """
    # 取得首轮多模态输入。
    mm = sample.multimodal_inputs or {}
    # 取出当前首轮包含的音频列表。
    # 正常情况下这里是 [chunk0]。
    audios = list(mm.get("audio") or [])
    # 推理流：
    # tokenizer 只处理 prompt 文本中的音频 marker，不展开成音频帧级占位 token。
    unexp_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    # 如果没有 processor，只能退化为纯文本路径。
    # 此时训练流与推理流相同，也没有音频特征。
    if processor is None:
        return unexp_ids, unexp_ids, [], None

    # 训练流：
    # processor 根据 chunk0 的真实音频长度展开音频 token，
    # 并生成对应的音频特征。
    exp_ids, mm_train = _run_processor(processor, tokenizer, sample.prompt, mm, args)
    # 默认采样率为 16 kHz。
    sample_rate = int(getattr(args, "audio_sample_rate", None) or 16000)
    # 将每段音频编码为 SGLang /generate 接口接受的格式。
    audio_data = [encode_audio_for_rollout_engine(a, sample_rate) for a in audios]
    return unexp_ids, exp_ids, audio_data, mm_train


def _strip_bos(tokenizer, ids: list[int]) -> list[int]:
    """删除增量 token 序列开头可能重复出现的 BOS。

    每个后续音频 chunk 都会被独立编码。
    某些 tokenizer 会在每次独立编码时自动加入 BOS。

    但累计上下文开头已经有 BOS，
    如果每轮都继续追加 BOS，会在人为拼接的对话中插入多个序列起始符。
    """
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

    =======================================================

    把一个后续音频 chunk 编码为新的 user turn。

    返回：
        exp_ids:
            processor 展开的增量 token，追加到 sample.tokens。

        unexp_ids:
            tokenizer 未展开的增量 token，追加到 sample.rollout_tokens。

        audio_str:
            SGLang 接口使用的音频编码。

        mm_train:
            当前 chunk 对应的训练音频特征。

    只返回“本轮新增部分”，不返回重复的 system prompt 或历史前言。
    """

    # 最终每个后续 chunk 产生四份数据：
    # chunk
    # ├─ exp_ids      → sample.tokens
    # ├─ unexp_ids    → sample.rollout_tokens
    # ├─ audio_str    → 累计 audio_data
    # └─ mm_train     → 累计 multimodal_train_inputs

    # 将当前音频块包装成 Qwen chat template 所需的多模态 user message。
    message = {"role": "user", "content": [{"type": "audio", "audio": chunk}]}
    sample_rate = int(getattr(args, "audio_sample_rate", None) or 16000)
    # 当前 chunk 单独编码为 SGLang audio_data 项。
    audio_str = encode_audio_for_rollout_engine(chunk, sample_rate)

    # 允许调用方传额外的 chat template 参数。
    apply_kwargs = args.apply_chat_template_kwargs or {}
    # trim_length 表示固定对话前言占用了多少 tokenizer token。
    trim_length = 0
    if args.apply_chat_template:
        # 只编码固定 dummy 前言，不加入 assistant generation prompt。
        #
        # 其 token 长度用于确定 formatted_prompt 中应该裁掉多少前缀。
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES, tokenize=False, add_generation_prompt=False, **apply_kwargs
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message], tokenize=False, add_generation_prompt=True, **apply_kwargs
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        # 即使配置上不走外层 apply_chat_template，
        # 这里仍调用 tokenizer.apply_chat_template 来把音频消息变成模型格式。
        #
        # 区别是此分支不额外加入 DUMMY_MESSAGES，也不做前缀裁剪。
        formatted_prompt = tokenizer.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True, **apply_kwargs
        )

    # 推理流：用 tokenizer 编码，音频 marker 保持未展开。
    unexp_full = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    # 如果前面使用了 dummy 对话，就裁掉固定前言，
    # 只保留当前音频 turn 新增的 token。
    unexp_ids = unexp_full[trim_length:] if trim_length else unexp_full
    # 防止增量部分开头重复携带 BOS。
    unexp_ids = _strip_bos(tokenizer, unexp_ids)

    # 没有 processor 时退化为纯 tokenizer 路径。
    if processor is None:
        return unexp_ids, unexp_ids, audio_str, None

    # 当前 observation 只有音频，没有图像和视频。
    #
    # images/videos 必须传 None，而不能传空列表 []。
    # 一些 processor 对 [] 仍会进入图像分组逻辑，
    # 随后在 group_images_by_shape([]) 中发生越界。
    mm_obs = {"images": None, "videos": None, "audio": [chunk]}
    
    # 训练流：processor 展开音频占位 token，并生成当前块的训练特征。
    exp_ids, mm_train = _run_processor(processor, tokenizer, formatted_prompt, mm_obs, args)
    # processor 的 token 序列也包含相同的固定文本前言。
    # dummy 前言本身不含音频，因此其 tokenizer 长度可用于裁剪两条流。
    if trim_length:
        exp_ids = exp_ids[trim_length:]
    # 同样去掉增量部分可能重复出现的 BOS。
    exp_ids = _strip_bos(tokenizer, exp_ids)

    return exp_ids, unexp_ids, audio_str, mm_train


# 音频特征键：各块帧数（最后一维）不同，需先 pad 到 max 再 cat dim0，复现「单次 processor
# 多音频调用」的规范布局（input_features [N,128,maxT] / feature_attention_mask [N,maxT]）。
# 与 megatron/data.py 的 PAD_RULES 语义一致（那里跨样本 pad，这里样本内 pad）。

# 不同 chunk 的音频时长可能不同，因此 processor 输出的时间维 T 不同。
#
# 合并多个 chunk 前需要将最后一维 pad 到统一长度。
#
# 声学特征缺失位置补 0.0。
# attention mask 缺失位置补 0，表示 padding 帧不可见。
#
# 典型形状：
# chunk 0 input_features: [1, 128, T0]
# chunk 1 input_features: [1, 128, T1]
# chunk 2 input_features: [1, 128, T2]

# pad 到 max(T0,T1,T2) 后：

# merged input_features: [3, 128, maxT]
_AUDIO_PAD_LAST_DIM = {"input_features": 0.0, "feature_attention_mask": 0}


def _merge_mm_train(chunks: list[dict | None]) -> dict | None:
    """把各块 mm_train_inputs 按 key 合并（只合并 torch.Tensor 值）。

    - 音频特征键（帧数变长）：pad 最后一维到 max 后 cat dim0；
    - 其余键（如定长网格）：直接 cat dim0。

    =======================================================

    将每个音频块的 processor 输出合并成一条样本的多模态训练输入。

    合并规则：

    1. 忽略 None 和空字典；
    2. 按特征名称收集所有 chunk 的值；
    3. 只合并 torch.Tensor；
    4. input_features 和 feature_attention_mask：
       先在最后一维补齐，再沿第 0 维拼接；
    5. 其他 Tensor：
       直接沿第 0 维拼接。
    """

    # 将：
    # [
    #   {"input_features": f0, "mask": m0},
    #   {"input_features": f1, "mask": m1},
    # ]
    #
    # 变成：
    # {
    #   "input_features": [f0, f1],
    #   "mask": [m0, m1],
    # }
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
        # 非 Tensor 元数据不在这里合并。
        # 该函数只负责训练 tensor。
        if not all(isinstance(v, torch.Tensor) for v in values):
            continue
        if len(values) == 1:
            # 只有一块时无需 padding 或复制。
            merged[key] = values[0]
        elif key in _AUDIO_PAD_LAST_DIM:
            # 音频时间维长度不一致，需要补齐最后一维。
            pad_val = _AUDIO_PAD_LAST_DIM[key]
            # 找到所有 chunk 的最大时间长度。
            max_last = max(v.shape[-1] for v in values)
            padded = [
                # torch.nn.functional.pad 的 [0, n]
                # 表示只在最后一维右侧补 n 个值。
                torch.nn.functional.pad(v, [0, max_last - v.shape[-1]], value=pad_val)
                if v.shape[-1] < max_last
                else v
                for v in values
            ]
            # 沿第 0 维合并多个音频段。
            merged[key] = torch.cat(padded, dim=0)
        else:
            # 其他形状兼容的多模态 Tensor 直接拼接。
            merged[key] = torch.cat(values, dim=0)
    return merged or None


def _append_generated(sample: Sample, response_tokens: list[int], tokens: list[int], logprobs: list[float]):
    """模型生成 token：两条流里一致（纯文本 token，无音频占位），loss_mask=1 参与训练。
    
    =======================================================

    将一轮模型生成的文本 token 加入完整训练轨迹。

    模型生成的是纯文本，不包含需要展开的音频 marker，
    因此训练流与 SGLang 推理流中的 token 完全相同。

    模型生成 token 属于策略动作：
        loss_mask = 1
        rollout_log_probs = SGLang 返回的行为策略 logprob
    """
    # 加入 Megatron 训练流。
    sample.tokens.extend(tokens)
    # 加入下一轮继续发送给 SGLang 的累计上下文。
    sample.rollout_tokens.extend(tokens)
    # response_tokens 记录 prompt 之后的展开流：
    # 包括模型生成 token 和后来注入的 observation token。
    #
    # 它不包括初始 prompt。
    response_tokens.extend(tokens)
    # 模型生成 token 参与 GRPO/PPO 策略损失。
    sample.loss_mask.extend([1] * len(tokens))
    # 保存 rollout 时旧策略对这些动作 token 给出的 logprob。
    sample.rollout_log_probs.extend(logprobs)
    # response_length 按 prompt 之后的展开流长度计算。
    sample.response_length = len(response_tokens)


def _append_observation(sample: Sample, response_tokens: list[int], exp_ids: list[int], unexp_ids: list[int]):
    """注入的音频观测：训练流(sample.tokens)用展开 ids，sglang 流(rollout_tokens)用未展开 ids；
    loss_mask=0 不参与训练。response_tokens/loss_mask/logprobs 都跟展开流对齐。
    
    =======================================================

    音频 observation 是环境输入，不是模型动作：

        sample.tokens          追加 exp_ids
        sample.rollout_tokens  追加 unexp_ids
        loss_mask              填 0
        rollout_log_probs      填 0.0

    loss_mask/logprob 使用 exp_ids 的长度，
    因为训练对齐以 sample.tokens 这条展开流为准。
    """
    # 最关键的对齐关系是：
    # len(sample.loss_mask)
    # = len(sample.rollout_log_probs)
    # = response_length
    # 其中这些长度只覆盖“初始 prompt 之后”的部分，不包含初始 prompt token。
    #
    # 而：
    # len(sample.tokens)
    # = prompt_length + response_length
    
    # 训练流追加 processor 展开的音频 token。
    sample.tokens.extend(exp_ids)
    # response_tokens 也按展开流记录，
    # 用于计算训练侧 response_length。
    response_tokens.extend(exp_ids)
    # observation 不属于策略输出，不计算策略损失。
    sample.loss_mask.extend([0] * len(exp_ids))
    # observation 没有旧策略生成概率。
    # 为保持数组长度与展开训练流对齐，填充 0.0。
    sample.rollout_log_probs.extend([0.0] * len(exp_ids))
    # 推理流追加未展开音频 marker。
    sample.rollout_tokens.extend(unexp_ids)
    sample.response_length = len(response_tokens)


async def _run_inference_step(state, url, tokens, sampling_params, audio_data, args):
    """使用累计 token 和累计音频执行一轮增量生成。"""
    payload = {
        # 未展开的累计 token 流。
        "input_ids": tokens,
        # 当前轮采样参数；主循环可能会修改 max_new_tokens。
        "sampling_params": sampling_params,
        # 训练需要旧策略生成 token 的 logprob。
        "return_logprob": True,
    }
    # NOTE(2): LoRA 训练时 rollout 必须引用刚热推上去的同名 adapter，否则用的是 base。
    if getattr(args, "lora_enable", False):
        # 明确选择训练侧刚热推到 SGLang 的 adapter。
        #
        # 如果没有该字段，即使 adapter 已加载，
        # 当前请求仍可能使用纯 base model。
        payload["lora_path"] = getattr(args, "lora_name", None) or "policy"
    if audio_data:
        # audio_data 是到当前轮为止的所有音频块。
        #
        # rollout_tokens 中也累计了对应数量的未展开音频 marker。
        # SGLang 根据 marker 和 audio_data 在内部展开音频输入。
        payload["audio_data"] = audio_data

    # 逐轮领一个 per-request permit：permit 在 post_generate 返回时立刻释放，
    # 生成之间的切块/编码不占槽位。必须与文件末尾的
    # `generate.manages_inference_permit = True` 成对出现，否则 v2 会抛
    # RuntimeError 提示会死锁（session 锁不可重入）。
    output = await state.post_generate(url, payload)
    meta = output["meta_info"]
    if "output_token_logprobs" in meta:
        # SGLang 返回的每项通常包含：
        # item[0] = token logprob
        # item[1] = token ID
        new_tokens = [item[1] for item in meta["output_token_logprobs"]]
        new_logprobs = [item[0] for item in meta["output_token_logprobs"]]
    else:
        # 没有 token logprob 时返回空生成轨迹。
        new_tokens, new_logprobs = [], []
    # 常见值包括 stop、length、abort。
    finish_type = meta["finish_reason"]["type"]
    return output["text"], new_tokens, new_logprobs, finish_type


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """同传多轮 rollout：整段音频切块，每轮喂一块并生成一段增量译文。"""
    try:
        return await _generate_impl(args, sample, sampling_params)
    except GenerationAborted:
        # v2 的 per-request permit 可能在任意一轮抛出。核心会把它映射成 ABORTED 样本，
        # 但那条路径拿不到我们已经拼了一半的 response/loss_mask，所以在这里落状态：
        # 已完成的轮次仍然自洽（三个长度对齐），只是标记为 ABORTED 不参与训练。
        sample.status = Sample.Status.ABORTED
        sample.metadata["simul_stop_reason"] = "aborted"
        return sample


async def _generate_impl(args: Any, sample: Sample, sampling_params) -> Sample:

    # 这里有三种“response”概念：
    # 1. response_tokens：训练对齐使用，包含生成 token 和后续观测 token。
    # 2. sample.response_length：等于 len(response_tokens)。
    # 3. sample.response：BLEU 使用，只包含模型生成的干净译文字符串。

    # GenerateState 负责取得 tokenizer 和多模态 processor。
    state = GenerateState(args)
    tokenizer, processor = state.tokenizer, state.processor
    # 所有轮次都请求同一个 SGLang router。
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # 检查安全轮数上限
    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        raise ValueError("simul_s2tt 需要 --max-turns（作为安全上限；实际轮数由切块数决定）。")

    # build_env 读取 sample 中的完整音频并按 simul_chunk_ms 切块。
    #
    # 它还会把 sample.multimodal_inputs["audio"]
    # 替换成只包含 chunk0 的列表，使首轮只看到第一块音频。
    env = build_env(sample, args)
    # 将环境游标复位到开头。
    env.reset()

    loop = asyncio.get_running_loop()

    # ---- 首轮输入编码（双流：unexp 发 sglang / exp 供训练）----
    unexp_prompt_ids, exp_prompt_ids, audio_data, init_mm_train = await loop.run_in_executor(
        get_encode_executor(), _encode_initial_inputs, sample, processor, tokenizer, args
    )

    # 两条 token 流（对齐已验证 s2tt 核心路径）：
    # - sample.rollout_tokens（未展开）→ 每轮发 sglang 的 input_ids，配 audio_data 由 sglang 展开。
    # - sample.tokens（processor 展开，含音频占位 token）→ 训练用，与 multimodal_train_inputs 对齐。
    
    # Megatron 训练流：processor 已展开音频 token。
    sample.tokens = list(exp_prompt_ids)
    # SGLang 推理流：音频 marker 未展开。
    sample.rollout_tokens = list(unexp_prompt_ids)
    # 保留已有列表；若为空或 None，则初始化为空列表。
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    # 记录初始 prompt 之后的展开 token。
    #
    # 后续会同时加入：
    # - 模型生成 token；
    # - 环境注入的展开 observation token。
    #
    # 不包含初始 prompt。
    response_tokens: list[int] = []  # 展开流的 gen+obs，保持与 sample.tokens/loss_mask/response_length 对齐
    # 只记录模型每轮生成的干净文本。
    # 不加入音频 observation，也不加入特殊 token。
    response_text_parts: list[str] = []  # 只收模型各轮生成文本（干净译文，喂 BLEU）
    sample.response_length = 0

    # 按块暂存 processor 输出，最后统一 pad 和合并。
    mm_train_buffer: list[dict | None] = []
    if init_mm_train:
        mm_train_buffer.append(init_mm_train)

    # 确保 metadata 可以写入终止原因等字段。
    sample.metadata = sample.metadata or {}

    # 总生成预算（只算模型 token，不含观测音频）：防多轮累积把序列撑爆。simul 脚本没设
    # --rollout-max-context-len，故这里用 rollout_max_response_len 收口，逐轮扣减。
    max_resp = getattr(args, "rollout_max_response_len", None)
    generated_count = 0

    # 开始多轮生成循环：
    stop_reason = "completed"
    for turn_idx in range(max_turns):
        cur_sp = sampling_params.copy()
        if max_resp is not None:
            # 计算所有轮次共享预算中尚未使用的部分。
            remaining = max_resp - generated_count
            if remaining <= 0:
                sample.status = Sample.Status.TRUNCATED
                stop_reason = "budget_exhausted"
                break
            # 本轮不能生成超过总剩余预算的 token。
            #
            # 如果原 sampling_params 的 max_new_tokens 更小，
            # 则仍遵循原本更严格的限制。
            cur_sp["max_new_tokens"] = min(cur_sp.get("max_new_tokens", remaining), remaining)

        # 请求 SGLang
        # 发给 sglang 的是「未展开」的 rollout_tokens + audio_data（sglang 内部展开音频占位）。
        _text, new_tokens, new_logprobs, finish_type = await _run_inference_step(
            state, url, sample.rollout_tokens, cur_sp, audio_data, args
        )
        # 模型生成的 token 参与训练（loss_mask=1），两条流一致。
        _append_generated(sample, response_tokens, new_tokens, new_logprobs)
        # 去掉结束符等特殊 token 再收集：拼进 sample.response 的必须是干净译文，否则 BLEU 被
        # marker 打断跨块 n-gram、并凭空多出参考没有的 token（实测拉低 ~60%）。
        response_text_parts.append(_clean_gen_text(_text))
        # 总生成预算只计算模型输出 token。
        generated_count += len(new_tokens)

        # 处理 SGLang 终止原因
        if finish_type == "abort":
            sample.status = Sample.Status.ABORTED
            stop_reason = "abort"
            break

        # 某轮把预算打满（length 截断）：本轮译文已被切断，继续下一块无意义，收口。
        if finish_type == "length":
            sample.status = Sample.Status.TRUNCATED
            stop_reason = "length"
            break

        # 从环境取得下一块音频
        # 脚本化推进：拿下一块音频（step 忽略模型输出）。
        observation, done, _info = env.step(_text)
        if done:
            # 当前轮生成后已经没有下一块音频。
            sample.status = Sample.Status.COMPLETED
            stop_reason = "chunks_exhausted"
            break

        # 从 observation 中提取音频
        # 将环境 observation 格式化为多模态 chat message。
        obs_message = env.format_observation(observation)
        chunk = None
        # 在 message content 中找到音频项。
        for c in obs_message["content"]:
            if c.get("type") == "audio":
                chunk = c["audio"]
                break

        # 编码并注入下一块音频

        # 每次追加后，两条流仍保持各自内部一致：
        # rollout_tokens 中的音频 marker 数
        # ↔ audio_data 中的音频段数

        # sample.tokens 中的展开音频 token
        # ↔ mm_train_buffer 中的音频特征

        obs_exp_ids, obs_unexp_ids, obs_audio_str, obs_mm_train = await loop.run_in_executor(
            get_encode_executor(), _encode_audio_observation, chunk, processor, tokenizer, args
        )
        # 注入的观测 token 不参与训练（loss_mask=0）：exp 进训练流，unexp 进 sglang 流。
        _append_observation(sample, response_tokens, obs_exp_ids, obs_unexp_ids)
        if obs_audio_str is not None:
            # SGLang 每轮收到的是累计音频列表。
            audio_data = (audio_data or []) + [obs_audio_str]
        if obs_mm_train:
            # Megatron 训练特征最后统一合并。
            mm_train_buffer.append(obs_mm_train)

        # 达到 max_turns
        if turn_idx + 1 >= max_turns:
            sample.status = Sample.Status.TRUNCATED
            stop_reason = "max_turns"
            break

    # 生成结束后的样本整理
    # 合并多块音频特征
    sample.multimodal_train_inputs = _merge_mm_train(mm_train_buffer)
    # response 用各轮「生成文本」拼接（干净译文）而非 decode(response_tokens)：后者含观测音频
    # 占位符 token，会污染 BLEU。response_length 仍用 token 数（含 obs），供训练对齐 loss_mask。
    sample.response = "".join(response_text_parts)
    # response_length 服务于 token/mask/logprob 对齐，
    # 不是 sample.response 字符串长度。
    #
    # 它包含：
    # - 模型生成 token；
    # - 后续注入的展开 observation token。
    #
    # 它不包含初始 prompt token。
    sample.response_length = len(response_tokens)
    # 状态兜底
    # status 默认是 PENDING（非 None）；正常结束的分支都已显式置位，这里兜底避免残留 PENDING。
    if sample.status in (None, Sample.Status.PENDING):
        sample.status = Sample.Status.COMPLETED
    sample.metadata["simul_stop_reason"] = stop_reason
    sample.metadata["simul_num_chunks"] = env.num_chunks
    return sample


# Opt-in：本 rollout 自己按轮管理 per-request permit（见 _run_inference_step 里的
# state.post_generate）。不声明的话 v2 会让整个多轮序列全程占住 session 级信号量的一个
# 槽位——不死锁、不串行，但一个样本从头到尾占着，切块与编码的时间也算在里面，吞吐会掉。
# 上游注释：“Custom multi-turn rollouts should use inference_permit()/post_generate()”。
generate.manages_inference_permit = True

# 🚨 ===== [YULIN-MOD] END =====