#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -ex
set -o pipefail

: "${OMNI_ROUTER_HOST:?Set OMNI_ROUTER_HOST to the external SGLang-Omni router host}"
: "${OMNI_ROUTER_PORT:?Set OMNI_ROUTER_PORT to the external SGLang-Omni router port}"

now=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-omni-30B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/omni-lora-omni-simul}"
EXP_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../../exps}"
NUM_ROLLOUT="${NUM_ROLLOUT:=2}"
HF_CKPT="${HF_CKPT:-${EXP_DIR}/Qwen3-Omni-30B-A3B-Instruct}"
PROMPT_SET=${DATA:-${EXP_DIR}/AVQA/AVQA_R1/train/omni_rl_format_train_convert.jsonl}
OMNI_ROUTER_ADDR="${OMNI_ROUTER_HOST}:${OMNI_ROUTER_PORT}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${HF_CKPT}"
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

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --rollout-shuffle
   --rm-type "${RM_TYPE:-multiple_choice}"
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH:-8}"
   --n-samples-per-prompt "${N_SAMPLES:-4}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-256}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.8}"
   --global-batch-size "${GLOBAL_BATCH:-32}"
   --balance-data
   --system-prompt "You are a helpful assistant."
   --multimodal-keys '{"audio":"audio"}'
   --rollout-function-path relax.engine.rollout.sglang_omni_rollout.generate_rollout
   --custom-generate-function-path examples.simul_s2tt.omni_rollout.generate
   --custom-config-path "${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/../../../examples/simul_s2tt/config.yaml}"
)

if [ -n "${ROLLOUT_MAX_PROMPT_LEN:-}" ]; then
   ROLLOUT_ARGS+=(--rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}")
fi

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
   --lr "${LR:-1e-5}"
   --lr-decay-style constant
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.95
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
)

OMNI_ARGS=(
   --rollout-external
   --rollout-external-engine-addrs "${OMNI_ROUTER_ADDR}"
   --sglang-router-ip "${OMNI_ROUTER_HOST}"
   --sglang-router-port "${OMNI_ROUTER_PORT}"
   --rollout-num-gpus-per-engine 4
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
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "qwen3-omni-lora-omni-simul-${now}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

HEALTH_ARGS=(
   --use-health-check
)
if [ -n "${MAX_GLOBAL_RESTART:-}" ]; then
   HEALTH_ARGS+=(--max-global-restart "${MAX_GLOBAL_RESTART}")
fi

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   ${RUNTIME_ENV_JSON:+--runtime-env-json="${RUNTIME_ENV_JSON}"} \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 4], "rollout": [1, 4]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --no-offload-train \
   --no-offload-rollout \
   "${HEALTH_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${OMNI_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen3-omni-lora-omni-simul-${now}.log"
