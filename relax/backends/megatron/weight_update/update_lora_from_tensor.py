# 🚨 ===== [YULIN-MOD] START: 从 Megatron 收集最新 adapter，并通过 IPC 热推给 SGLang =====

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Block 3: 把 Megatron 训练侧的 LoRA adapter 权重热推到 colocate 的 sglang rollout 引擎。

与全量权重同步(``UpdateWeightFromTensor``)的区别:
  * 只同步 adapter 权重(基座冻结、无需重传):``HfWeightIteratorDirect`` 用
    ``name_filter`` 把范围缩到 ``.adapter.`` 参数,只 gather/转换 adapter;
  * 转换走 ``convert_to_hf``→``convert_qwen3omni_to_hf`` 的 raw 路径,把 Megatron
    adapter 命名(``...linear_qkv.adapter.linear_in/out``)转成 HF/PEFT 命名
    (``...self_attn.{q,k,v,o}_proj.lora_{A,B}``),且 **不 merge 进 base**
    —— 满足「只 lora、无 base、无 merge」;sglang 侧 ``normalize_qkv_proj`` 会自动
    把 q/k/v_proj stack 成 ``qkv_proj``;
  * 发送时打一个 ``FlattenedTensorBucket``,经 IPC 调
    ``load_lora_adapter_from_tensors``(而非 ``update_weights_from_tensor``)。

历史(坑 16/17):曾尝试改走 Megatron-Bridge ``export_adapter_weights()``,但镜像 pin
的 redai-fork(f13bec09,带 Qwen3-Omni)早于该 API,``AutoBridge`` 无此方法;升级
bridge 又会丢 Qwen3-Omni。故回退 direct 路径,并修复坑 16:adapter 是分布式优化器
连续 buffer 的非连续视图,NCCL all_gather 前需 ``.contiguous()``(见 common.py)。

当前仅支持 colocate(IPC)路径;分布式(NCCL)LoRA 推送尚未实现。
"""

# 总体数据流：
#
# Megatron adapter
# → PP/EP broadcast
# → TP all-gather
# → Megatron 名称转换
# → HF/PEFT LoRA 名称
# → FlattenedTensorBucket
# → pickle/base64
# → Ray IPC
# → SGLang unload old adapter
# → SGLang load new adapter

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import pickle

import pybase64
import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle

from relax.utils.distributed_utils import get_gloo_group

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .hf_weight_iterator_direct import HfWeightIteratorDirect


def _is_lora_adapter_name(name: str) -> bool:
    """判断是否为 Megatron PEFT adapter 参数(命名含 ``.adapter.``)。"""
    # Megatron-Bridge PEFT 的参数名包含 .adapter.。
    return ".adapter." in name


class UpdateLoRAFromTensor:
    """把 LoRA adapter 权重从训练侧热推到 colocate 的 rollout 引擎。

    load(adapter→GPU) → broadcast PP/EP → gather TP → 转 sglang 命名 → 打 bucket
    → IPC ``load_lora_adapter_from_tensors``。
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict | None = None,
    ) -> None:
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.weight_version = 0

        # LoRA 不做量化:adapter 权重小且需保持精度,强制 None。
        # 用 direct iterator + name_filter,把范围缩到 .adapter. 参数:只 gather/转换 adapter。
        self._hf_weight_iterator = HfWeightIteratorDirect(
            args=args,
            model=model,
            model_name=model_name,
            # LoRA 不量化。
            quantization_config=None,
            # 在参数元数据构建阶段就过滤，只处理 adapter。
            name_filter=_is_lora_adapter_name,
        )

        # rollout 引擎侧用这个名字注册/更新 adapter(Block 5 的 rollout 请求据此引用)。
        self.lora_name = getattr(args, "lora_name", None) or "policy"

        # adapter 的 PEFT 配置(sglang LoRAConfig.from_dict 需要 r / lora_alpha / target_modules)。
        self._config_dict = {
            "peft_type": "LORA",
            "r": int(args.lora_rank),
            "lora_alpha": float(args.lora_alpha),
            "lora_dropout": float(getattr(args, "lora_dropout", 0.0)),
            # SGLang 会把 q/k/v 归一到融合的 qkv_proj。
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
        }

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._loaded_once = False

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """建立 colocate IPC gather group,并把训练 rank 映射到各 colocate 引擎。

        逻辑与 ``UpdateWeightFromTensor`` 的 colocate 部分一致;若检测到分布式
        (非 colocate)引擎则直接报错,因为 LoRA 的 NCCL 推送路径尚未实现。
        """
        # 计算每个 rollout engine 占用的训练 GPU 范围。
        # 每组 GPU 创建一个 Gloo group，用于决定哪个训练 rank 是发送源。

        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        # 如果有 rollout engine 位于 actor GPU 范围之外，
        # 说明不是完整 colocate 部署。
        if colocate_engine_nums < len(rollout_engines):
            raise NotImplementedError(
                "LoRA 热更新目前只支持 colocate(IPC)引擎;检测到分布式 rollout 引擎。"
                "请使用 colocate 部署,或扩展 UpdateLoRAFromTensor 的 NCCL 路径。"
            )
        
        # 每个训练 rank 记录自己对应的 rollout engine；
        # 每个 engine 只由其 group 的首 rank 真正发送 adapter。

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = colocate_gpu_offsets[i]

        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self) -> None:
        """version++ → 暂停/flush → gather adapter → 打 bucket → IPC 推送 → 恢复。

        注意:迭代器内部含 TP/PP 集合通信,所有 rank 都必须完整跑完循环;
        但只有每个 colocate 引擎对应的 gather 源 rank 真正打包并发送。
        """
        # 记录 LoRA updater 自己的更新次数。
        self.weight_version += 1
        rank = dist.get_rank()

        # 只有对应 engine 的 gather 源 rank 才负责最终序列化和发送。
        is_sender = self._ipc_engine is not None and rank == self._ipc_gather_src

        if rank == 0:
            # 暂停生成，防止加载 adapter 时仍有请求使用旧权重。
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            # 清理前一轮 generation cache。
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        # 确保所有训练 rank 同步进入集合通信阶段。
        dist.barrier(group=get_gloo_group())

        # 根据 actor.py 提供的 getter 获取最新 adapter。
        # offload 模式下可能来自 CPU backup；
        # no-offload 模式下来自 GPU live tensor。
        megatron_local_weights = self.weights_getter()

        # 坑 16 诊断:确认取到的 adapter 张量来源/状态(device/contiguous),便于定位 all_gather 崩因。
        if rank == 0 and megatron_local_weights:
            _names = list(megatron_local_weights)
            _samp = megatron_local_weights[_names[0]]
            print(
                f"[LoRA-sync][rank0] 取到 adapter 张量 {len(_names)} 个; "
                f"样例 name={_names[0]} device={_samp.device} contiguous={_samp.is_contiguous()} "
                f"shape={tuple(_samp.shape)} dtype={_samp.dtype}",
                flush=True,
            )

        # 把所有 chunk 的完整 adapter 张量累积到一个 dict(仅发送 rank 需要保留)。
        full_named_tensors: list[tuple[str, torch.Tensor]] = []
        # 所有 rank 必须完整执行 iterator，
        # 因为 iterator 内部包含 PP/EP/TP 集合通信。
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            if is_sender:
                # 发送 rank 只保存最终转换出的 lora_A/lora_B。
                full_named_tensors.extend(
                    (n, t) for n, t in hf_named_tensors if ".lora_A." in n or ".lora_B." in n
                )

        if is_sender:
            self._push_lora_to_engine(full_named_tensors)
        del full_named_tensors

        # 等待所有 engine 完成 adapter 更新。
        dist.barrier(group=get_gloo_group())
        if rank == 0:
            # 恢复 rollout 生成。
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])

    def _push_lora_to_engine(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        # 如果没有任何 LoRA 参数，说明挂载或 target_modules 配置有问题。
        assert len(named_tensors) > 0, "未收集到任何 LoRA adapter 权重,检查 target_modules / 是否已挂 LoRA。"

        # FlattenedTensorBucket 只保存一份统一 dtype 的扁平张量。
        dtypes = {t.dtype for _, t in named_tensors}
        assert len(dtypes) == 1, f"LoRA adapter 权重存在混合 dtype {dtypes},无法打进单个 bucket。"

        # 将所有 adapter 张量展平为一个 bucket，并保留恢复名字/形状所需的 metadata。
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor().cpu(),
            "metadata": flattened_tensor_bucket.get_metadata(),
        }
        # 不使用 MultiprocessingSerializer：
        # 它可能通过 fd sharing 传 CPU Tensor，
        # 跨 Ray actor 进程树时可能因 authkey 不同触发 AuthenticationError。
        #
        # 普通 pickle 会把数据直接内联到 bytes，再用 base64 放入 RPC payload。
        serialized = pybase64.b64encode(pickle.dumps(flattened_tensor_data)).decode("utf-8")

        # SGLang 不允许重复加载同名 adapter。
        # RL 第二次及之后更新时，先卸载旧 policy。
        if self._loaded_once:
            ray.get(self._ipc_engine.unload_lora_adapter.remote(self.lora_name))

        # 从内存张量加载新的 adapter，不经过磁盘。
        ray.get(
            self._ipc_engine.load_lora_adapter_from_tensors.remote(
                lora_name=self.lora_name,
                serialized_tensors=serialized,
                config_dict=self._config_dict,
                load_format="flattened_bucket",
            )
        )
        self._loaded_once = True

# 🚨 ===== [YULIN-MOD] END =====