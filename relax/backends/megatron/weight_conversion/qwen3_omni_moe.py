# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re

import torch


def convert_qwen3omni_to_hf(args, name, param):
    if ".audio_model." in name:
        name = name.replace("module.module.thinker.audio_model.", "thinker.audio_tower.")
        return [(name, param)]
    if ".vision_model." in name:
        name = name.replace("module.module.thinker.vision_model.", "thinker.visual.")
        return [(name, param)]
    if ".video_model." in name:
        name = name.replace("module.module.thinker.video_model.", "thinker.video_tower.")
        return [(name, param)]
    if name == "module.module.thinker.language_model.embedding.word_embeddings.weight":
        return [("thinker.model.embed_tokens.weight", param)]
    if name == "module.module.thinker.language_model.output_layer.weight":
        return [("thinker.lm_head.weight", param)]
    if name == "module.module.thinker.language_model.decoder.final_layernorm.weight":
        return [("thinker.model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.thinker\.language_model\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # 🚨 ===== [YULIN-MOD] START: 识别 Qwen3-Omni adapter 参数 =====

        # Megatron 这边说：
        # “我这里叫 self_attention.linear_qkv.adapter.linear_out.weight，
        # 而且 Q/K/V 是按 query group 混着放的。”
        # 
        # SGLang 那边说：
        # “我只认 qkv_proj.lora_B.weight，
        # 而且你得按 [所有 Q; 所有 K; 所有 V] 的顺序给我。”
        
        # LoRA adapter 权重(Megatron-Bridge PEFT: ParallelLinearAdapter)
        # 命名形如 self_attention.linear_qkv.adapter.linear_in/out.weight
        # 转成 sglang 期望的 base_model.model.thinker.model.layers.N.self_attn.* 命名。
        
        # base：普通 base 权重继续执行原有转换；
        # adapter：只有包含 .adapter. 的 Megatron-Bridge PEFT 参数进入 LoRA 分支。
        if ".adapter." in rest:
            return _convert_qwen3omni_lora_adapter(
                args=args,
                layer_idx=layer_idx,
                rest=rest,
                param=param,
                head_dim=head_dim,
                value_num_per_group=value_num_per_group,
            )

        # 🚨 ===== [YULIN-MOD] END =====

        # experts
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                outputs = [
                    (f"thinker.model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"thinker.model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
                return outputs
            elif rest == "linear_fc2":
                outputs = [
                    (f"thinker.model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param),
                ]
                return outputs
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"thinker.model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight", gate_weight),
                    (f"thinker.model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"thinker.model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight", param)]
            elif rest == "gate_weight":
                return [(f"thinker.model.layers.{layer_idx}.mlp.shared_expert_gate.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"thinker.model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"thinker.model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"thinker.model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"thinker.model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"thinker.model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"thinker.model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"thinker.model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"thinker.model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"thinker.model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"thinker.model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"thinker.model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"thinker.model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"thinker.model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"thinker.model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"thinker.model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"thinker.model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"thinker.model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")


# 🚨 ===== [YULIN-MOD] START: 将 Megatron LoRA 转成 SGLang/PEFT 命名与布局 =====

# 1. 改名字
# 对应关系大概是：
# linear_qkv  -> qkv_proj
# linear_proj -> o_proj
# linear_in   -> lora_A
# linear_out  -> lora_B
#
# 2. 重新排列 lora_B 的内容
# Megatron 的存法像这样，按 query group 混着放：
# group 0: Q Q K V
# group 1: Q Q K V
# 连起来就是：
# Q Q K V  Q Q K V
#
# 但 SGLang 想要的是：
# 所有 Q 放一起，所有 K 放一起，所有 V 放一起
# 也就是：
# Q Q Q Q  K K  V V

# sglang 从 PEFT 风格的 adapter 名读取 LoRA,统一带 base_model.model. 前缀。
_LORA_PREFIX = "base_model.model.thinker.model.layers"


def _reorder_qkv_lora_b(param, num_query_groups, value_num_per_group, head_dim):
    """linear_qkv 的 lora_B(linear_out)输出维按 Megatron 的「分组交错」排布,
    需重排成 sglang qkv_proj 期望的 [q; k; v] 拼接顺序(与 base 权重转换一致)。

    入参 param 形状 [qkv_out, r],其中
    qkv_out = num_query_groups * (value_num_per_group + 2) * head_dim。
    """
    # lora_B 的形状为 [qkv_out, rank]。
    rank = param.shape[1]
    # Megatron 的 qkv_out 按 query group 组织：
    # [num_query_groups, 每组Q数量+K+V, head_dim, rank]
    param = param.view(num_query_groups, value_num_per_group + 2, head_dim, rank)
    # 把每个 group 里的 Q、K、V 拆出来。
    q_param, k_param, v_param = torch.split(
        param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
    )
    # 把所有 group 的 Q 汇总到一起，所有 K 汇总到一起，所有 V 汇总到一起。
    q_param = q_param.reshape(-1, rank)
    k_param = k_param.reshape(-1, rank)
    v_param = v_param.reshape(-1, rank)
    # 按 SGLang 要的顺序重新拼起来 [所有 Q; 所有 K; 所有 V] 。
    return torch.cat([q_param, k_param, v_param], dim=0)


def _convert_qwen3omni_lora_adapter(*, args, layer_idx, rest, param, head_dim, value_num_per_group):
    """把 Megatron-Bridge 的 ParallelLinearAdapter 权重转成 sglang LoRA 命名。

    映射约定(只支持 thinker 注意力的 qkv / o_proj):
      - linear_qkv.adapter.linear_in  -> qkv_proj.lora_A  (形状 [r, hidden],sglang 内部自动 repeat 3 份)
      - linear_qkv.adapter.linear_out -> qkv_proj.lora_B  (形状 [qkv_out, r],需重排成 [q;k;v])
      - linear_proj.adapter.linear_in -> o_proj.lora_A
      - linear_proj.adapter.linear_out-> o_proj.lora_B
    """
    prefix = f"{_LORA_PREFIX}.{layer_idx}.self_attn"

    # linear_in 是 LoRA A：
    # hidden_size → rank。
    if rest == "self_attention.linear_qkv.adapter.linear_in.weight":
        return [(f"{prefix}.qkv_proj.lora_A.weight", param)]
    # linear_out 是 LoRA B：
    # rank → qkv_out。
    if rest == "self_attention.linear_qkv.adapter.linear_out.weight":
        param = _reorder_qkv_lora_b(param, args.num_query_groups, value_num_per_group, head_dim)
        return [(f"{prefix}.qkv_proj.lora_B.weight", param)]
    # 注意力输出投影的 LoRA A。
    if rest == "self_attention.linear_proj.adapter.linear_in.weight":
        return [(f"{prefix}.o_proj.lora_A.weight", param)]
    # 注意力输出投影的 LoRA B。
    if rest == "self_attention.linear_proj.adapter.linear_out.weight":
        return [(f"{prefix}.o_proj.lora_B.weight", param)]

    # 当前同步契约只支持 Thinker 注意力的 QKV 和 O 投影。
    # 如果 target_modules 扩展到 MLP，这里也必须同步扩展转换逻辑。
    raise ValueError(
        f"不支持的 LoRA adapter 参数: layer={layer_idx} rest={rest}. "
        "当前 Block 3 仅支持 thinker 注意力的 linear_qkv / linear_proj。"
    )

# 🚨 ===== [YULIN-MOD] END =====