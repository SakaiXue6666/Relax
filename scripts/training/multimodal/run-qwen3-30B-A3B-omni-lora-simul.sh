#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-Omni-30B-A3B thinker-LoRA 同传（simultaneous S2TT）多轮 rollout 脚本。
#
# 与 run-...-omni-lora-smoke.sh 的唯一区别：挂上自定义多轮 generate + 其配置。
#   - --custom-generate-function-path examples.simul_s2tt.rollout.generate
#   - --custom-config-path examples/simul_s2tt/config.yaml（max_turns / simul_chunk_ms）
# 其余（LoRA / GRPO / colocate TP4 / sglang / bleu reward）与 smoke 完全一致：
# 数据仍是整段音频（切块在 env 里做），reward 仍是整段拼接译文的 BLEU。

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-omni-30B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/omni-lora-simul}"
EXP_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../../exps}"

NUM_ROLLOUT="${NUM_ROLLOUT:=2}"

HF_CKPT="${HF_CKPT:-${EXP_DIR}/Qwen3-Omni-30B-A3B-Instruct}"
CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${HF_CKPT}
   --megatron-to-hf-mode bridge
)

if [ -n "${SAVE_DIR:-}" ]; then
   CKPT_ARGS+=(
      --save "${SAVE_DIR}"
      --load "${SAVE_DIR}"
      --save-interval "${SAVE_INTERVAL:-5}"
      --max-actor-ckpt-to-keep "${MAX_CKPT_KEEP:-1}"
      --override-opt_param-scheduler
   )
fi

SYSTEM_PROMPT="You are a helpful assistant."

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

if [ -n "${CHAT_TEMPLATE_KWARGS:-}" ]; then
   ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}")
fi

MULTIMODAL_KEYS="${MULTIMODAL_KEYS-'{"image":"image","audio":"audio"}'}"
if [ -n "${MULTIMODAL_KEYS}" ]; then
   ROLLOUT_ARGS+=(--multimodal-keys "${MULTIMODAL_KEYS}")
fi

# ---- 同传专属：自定义多轮 generate + 其配置（唯一相对 smoke 新增的部分）----
SIMUL_ARGS=(
   --custom-generate-function-path examples.simul_s2tt.rollout.generate
   --custom-config-path "${SCRIPT_DIR}/../../../examples/simul_s2tt/config.yaml"
)

LORA_ARGS=(
   --lora-enable
   --lora-rank 16
   --lora-alpha 32
   --lora-dropout 0.0
   --lora-name policy
)

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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   # 2a(不 offload)：Megatron base 常驻，需给它腾显存，故调低 sglang 静态占比。
   # 可由 SGLANG_MEM_FRACTION 覆盖以便调参；默认 0.55（与 noffload 脚本一致）。
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION:-0.55}
   --sglang-enable-lora
   --sglang-max-lora-rank 16
   --sglang-max-loras-per-batch 1
   --sglang-lora-target-modules qkv_proj o_proj
   --sglang-attention-backend triton
   --sglang-disable-cuda-graph
   --sglang-disable-custom-all-reduce
)

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
   --tb-experiment-name qwen3-omni-lora-simul-${now}
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
   --no-offload-train \
   --no-offload-rollout \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${SIMUL_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-omni-lora-simul-${now}.log
