#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-Omni-30B-A3B thinker-LoRA銆屽惎鍔ㄥ啋鐑熴€嶈剼鏈紙colocate锛? 鍗★級銆?
#
# 鐩殑锛氬彧楠岃瘉 Block 3/4/5 杩欐潯 LoRA 閾捐矾鍦ㄧ湡瀹?Relax 杩愯鏃惰兘涓嶈兘璺戦€氾紝
#       涓嶇湅璁粌鏁堟灉銆傝窇 2 涓?rollout step 灏卞仠銆?
#
#   - Block 4锛?-lora-enable -> Megatron-Bridge PEFT 缁?thinker 鎸?LoRA锛?
#              骞剁敱 _assert_lora_attached 鍦ㄥ缓浼樺寲鍣ㄥ墠鑷 adapter 鎸傚娌°€?
#   - Block 3锛氭瘡娆?update_weights 鐢?UpdateLoRAFromTensor 鍙妸 adapter
#              鐑帹缁?sglang锛堢湅 sglang 鏃ュ織 "load Lora adapter from tensors"锛夈€?
#   - Block 5锛歳ollout 鐨?generate 璇锋眰甯?lora_path=policy锛堢湅棣栬疆 rollout 涓嶆姤 503锛夈€?
#
# 纭墠鎻愶細4脳A100-80GB 鎴?4脳H100銆?0B-A3B bf16鈮?0GB锛宑olocate TP4 鍚庢瘡鍗″熀搴р増15GB銆?
#
# 鐢ㄦ硶锛堝湪 Relax 浠撳簱鏍圭洰褰曪級锛?
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

# 鍐掔儫锛氬彧璺?2 涓?rollout step 灏卞仠銆?
NUM_ROLLOUT="${NUM_ROLLOUT:=2}"

# HF_CKPT 鍙敱澶栭儴锛堝 Modal锛夎鐩栵紝鐩存帴鎸囧悜宸茬紦瀛樼殑鏉冮噸鐩綍銆?
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

# 鍐掔儫鐢ㄤ竴浠藉緢灏忕殑 toy 鏁版嵁鍗冲彲锛堝嚑鍗佹潯灏卞瑙﹀彂 rollout/train/sync锛夈€?
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

# multimodal-keys 鍙敱澶栭儴瑕嗙洊锛涜涓虹┖涓插垯璧扮函鏂囨湰锛圡odal 鍐掔儫鏈€蹇矾寰勶級銆?
# ${VAR-default}锛堟棤鍐掑彿锛夛細浠呭綋 VAR 鏈缃椂濉粯璁わ紱鏄惧紡绌轰覆浼氳淇濈暀銆?
MULTIMODAL_KEYS="${MULTIMODAL_KEYS-'{"image":"image","audio":"audio"}'}"
if [ -n "${MULTIMODAL_KEYS}" ]; then
   ROLLOUT_ARGS+=(--multimodal-keys "${MULTIMODAL_KEYS}")
fi

# 端到端验证(坑23 cpu_backup):设 DUMP_DETAILS 把每个 rollout step 的样本
# (含 tokens + 解码文本)dump 成 .pt,跑后解码判断 base 在 offload/resume 后是否存活。
if [ -n "${DUMP_DETAILS:-}" ]; then
   ROLLOUT_ARGS+=(--dump-details "${DUMP_DETAILS}")
fi

# ---- LoRA锛圔lock 4锛氳缁冧晶鎸?adapter锛汢lock 5锛歳ollout 寮曠敤鍚屽悕 adapter锛?---
# lora-target-modules 鐢ㄩ粯璁ゅ€硷紙*language_model*linear_qkv / linear_proj锛夛紝
# 宸茬敱浠ｇ爜涓夋柟璇佹嵁纭鑳藉懡涓?thinker 涓斾笉璇寕 audio/vision锛屾晠姝ゅ涓嶆樉寮忓啓銆?
LORA_ARGS=(
   --lora-enable
   --lora-rank 16
   --lora-alpha 32
   --lora-dropout 0.0
   --lora-name policy
)

# ---- GRPO锛氬啋鐑熷叧鎺?KL锛岀渷鎺夌嫭绔?reference锛? 鍗℃洿瀹借 ----
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

# ---- sglang锛氬紑 LoRA锛堣繖浜?flag 鐢?ServerArgs.add_cli_args 鑷姩閫忎紶锛?---
# sglang 渚?target 鐢?Block 3 杞崲鍣ㄤ骇鍑虹殑铻嶅悎鍛藉悕锛歲kv_proj / o_proj銆?
# max-lora-rank 蹇呴』 >= 璁粌渚?lora-rank銆?
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   # 2a(不 offload)：Megatron base 常驻，需给它腾显存，故调低 sglang 静态占比。
   # 可由 SGLANG_MEM_FRACTION 覆盖以便调参；默认 0.55。
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION:-0.55}
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

# ---- 骞惰搴︼細colocate 4 鍗★紝TP4 / EMP4(128%4=0) / PP1 ----
# PEFT 瀹夊叏榛樿锛氬叧 sequence-parallel 涓?recompute锛堟浘鍑虹幇 recompute+SP+LoRA
# 瀵艰嚧 lora_B backward NaN 鐨勫吋瀹规€ч棶棰橈紝NVIDIA Megatron-Bridge VLM PEFT
# recipe 涔熸樉寮忓叧闂繖涓ら」锛夈€?
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
   --no-offload-train \
   --no-offload-rollout \
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
