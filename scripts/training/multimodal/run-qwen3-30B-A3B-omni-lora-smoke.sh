#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# 🚨 ===== [YULIN-MOD] START: 验证 LoRA 挂载、训练、热推和 rollout 的完整链路 =====

# Qwen3-Omni-30B-A3B thinker-LoRA「启动冒烟」脚本(colocate,4 卡)。
#
# 目的:只验证 Block 3/4/5 这条 LoRA 链路在真实 Relax 运行时能不能跑通,
#       不看训练效果。跑 2 个 rollout step 就停。
#
#   - Block 4:--lora-enable -> Megatron-Bridge PEFT 给 thinker 挂 LoRA,
#              并由 _assert_lora_attached 在建优化器前自检 adapter 挂没挂对。
#   - Block 3:每次 update_weights 用 UpdateLoRAFromTensor 只把 adapter
#              热推给 sglang(看 sglang 日志 "load Lora adapter from tensors")。
#   - Block 5:rollout 的 generate 请求带 lora_path=policy(看首轮 rollout 不报 503)。
#
# 确前提:4×A100-80GB 或 4×H100。30B-A3B bf16≈60GB,colocate TP4 后每卡基座≈15GB。
#
# 用法(在 Relax 仓库根目录):
#   MODEL_DIR=/path/to/models DATA=/path/to/toy.jsonl \
#   bash scripts/training/multimodal/run-qwen3-30B-A3B-omni-lora-smoke.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-omni-30B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/omni-lora-smoke}"
EXP_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../../exps}"

# 冒烟:只跑 2 个 rollout step 就停。
NUM_ROLLOUT="${NUM_ROLLOUT:=2}"

# HF_CKPT 可由外部(如 Modal)覆盖,直接指向已缓存的权重目录。
HF_CKPT="${HF_CKPT:-${EXP_DIR}/Qwen3-Omni-30B-A3B-Instruct}"
CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${HF_CKPT}
   --megatron-to-hf-mode bridge
)

# 存档/续跑（正式实验用，抗 Modal 抢占）：SAVE_DIR 设了就开。
# --save 与 --load 指向同一目录：首跑空目录→从 hf-checkpoint 全新；
# 被抢重试→该目录已有 Megatron ckpt→自动从上次 rollout_id 续跑（含 optimizer）。
if [ -n "${SAVE_DIR:-}" ]; then
   CKPT_ARGS+=(
      --save "${SAVE_DIR}"
      --load "${SAVE_DIR}"
      --save-interval "${SAVE_INTERVAL:-5}"
      --max-actor-ckpt-to-keep "${MAX_CKPT_KEEP:-1}"
      # 续训时若改了总步数（如 40→100），checkpoint 里 LR 调度器的
      # total-iters 与新配置不符会触发 Megatron 断言。此 flag 让调度器
      # 用命令行新值覆盖 checkpoint 值；lr-decay-style=constant 下对 LR 无影响。
      --override-opt_param-scheduler
   )
fi

SYSTEM_PROMPT="You are a helpful assistant."

# 冒烟用一份很小的 toy 数据即可(几十条就够触发 rollout/train/sync)。
PROMPT_SET=${DATA:-${EXP_DIR}/AVQA/AVQA_R1/train/omni_rl_format_train_convert.jsonl}

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type ${RM_TYPE:-multiple_choice}
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH:-8}
   --n-samples-per-prompt ${N_SAMPLES:-4}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-256}
   ${ROLLOUT_MAX_PROMPT_LEN:+--rollout-max-prompt-len ${ROLLOUT_MAX_PROMPT_LEN}}
   --rollout-temperature ${ROLLOUT_TEMPERATURE:-0.8}
   --global-batch-size ${GLOBAL_BATCH:-32}
   --balance-data
   --use-fault-tolerance
   --system-prompt "${SYSTEM_PROMPT}"
)

# Qwen3-Omni 的 chat_template 在 processor(chat_template.json)里，裸 AutoTokenizer
# 在 transformers 5.x 下加载不到 -> tokenizer.chat_template 为空。冒烟时由外部
# (Modal) 把模板 JSON 注入 CHAT_TEMPLATE_KWARGS，显式喂给 apply_chat_template。
if [ -n "${CHAT_TEMPLATE_KWARGS:-}" ]; then
   ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}")
fi

# multimodal-keys 可由外部覆盖;若为空串则走纯文本(Modal 冒烟最快路径)。
# ${VAR-default}(无冒号):仅当 VAR 未设置时填默认;显式空串会被保留。
MULTIMODAL_KEYS="${MULTIMODAL_KEYS-'{"image":"image","audio":"audio"}'}"
if [ -n "${MULTIMODAL_KEYS}" ]; then
   ROLLOUT_ARGS+=(--multimodal-keys "${MULTIMODAL_KEYS}")
fi

# 端到端验证(坑23 cpu_backup):设 DUMP_DETAILS 把每个 rollout step 的样本
# (含 tokens + 解码文本)dump 成 .pt,跑后解码判断 base 在 offload/resume 后是否存活。
if [ -n "${DUMP_DETAILS:-}" ]; then
   ROLLOUT_ARGS+=(--dump-details "${DUMP_DETAILS}")
fi

# ---- LoRA(Block 4:训练侧挂 adapter;Block 5:rollout 引用同名 adapter)----
# lora-target-modules 用默认值(*language_model*linear_qkv / linear_proj),
# 已由代码三方证据确认能命中 thinker 且不误挂 audio/vision,故此处不显式写。
LORA_ARGS=(
   --lora-enable
   --lora-rank 16
   --lora-alpha 32
   --lora-dropout 0.0
   --lora-name policy
)

# ---- GRPO:冒烟关掉 KL,省掉独立 reference,4 卡更宽裕 ----
GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR:-1e-5}
   --lr-decay-style constant
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.95
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
)

# ---- sglang:开 LoRA(这些 flag 由 ServerArgs.add_cli_args 自动透传)----
# sglang 侧 target 用 Block 3 转换器产出的融合命名:qkv_proj / o_proj。
# max-lora-rank 必须 >= 训练侧 lora-rank。
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.7
   --sglang-enable-lora
   --sglang-max-lora-rank 16
   --sglang-max-loras-per-batch 1
   --sglang-lora-target-modules qkv_proj o_proj
   # 本地注入的 sglang 较新，要求 flashinfer>=0.6.11，但 slime 镜像只有 0.6.3。
   # 冒烟只验 LoRA 链路，不依赖 flashinfer attention 性能 -> 切 triton backend 绕开
   # 版本断言（配合 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 跳过 sgl-kernel 断言）。
   --sglang-attention-backend triton
   # 本地 sglang 的 custom all-reduce v2 内核（tvm_ffi/jit_kernel）与镜像 sgl-kernel
   # 二进制不匹配，TP=4 在 cuda graph 捕获时崩 "CUDA error: invalid argument"。
   # 冒烟禁掉 cuda graph + custom all-reduce（退回 NCCL），不影响 LoRA 链路验证。
   --sglang-disable-cuda-graph
   --sglang-disable-custom-all-reduce
)

# ---- 并行度:colocate 4 卡,TP4 / EMP4(128%4=0) / PP1 ----
# PEFT 安全默认:关 sequence-parallel 和 recompute(曾出现 recompute+SP+LoRA
# 导致 lora_B backward NaN 的兼容性问题,NVIDIA Megatron-Bridge VLM PEFT
# recipe 也显式关闭这两项)。
PERF_ARGS=(
   --train-backend megatron
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1
   --micro-batch-size 1
)

WANDB_ARGS=(
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen3-omni-lora-smoke-${now}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address=${RAY_ADDRESS:-"http://127.0.0.1:8265"} \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   ${RUNTIME_ENV_JSON:+--runtime-env-json="${RUNTIME_ENV_JSON}"} \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 4], "rollout": [1, 4]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-omni-lora-smoke-${now}.log

# 🚨 ===== [YULIN-MOD] END =====
