import dataclasses
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from tqdm import tqdm

from relax.utils import device as device_utils
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.types import ParamInfo

from ..sglang import monkey_patch_torch_reductions
from ..weight_conversion import convert_to_hf
from .common import all_gather_params_async, named_params_and_buffers
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorDirect(HfWeightIteratorBase):
    def __init__(self, *args, name_filter=None, **kwargs):
        # ``name_filter`` (可选): 形如 ``Callable[[str], bool]``,只保留返回 True 的
        # 全局参数名。LoRA 同步用它把范围缩到 adapter 权重,避免 gather/转换整座基座。
        super().__init__(*args, **kwargs)

        # 🚨 ===== [YULIN-MOD] START: 让 direct iterator 只建立 LoRA 参数桶 =====
        
        # 保存过滤函数，便于记录当前 iterator 的同步范围。
        self.name_filter = name_filter
        # 构造参数桶时立刻过滤。
        # LoRA 场景只会为 .adapter. 参数创建 ParamInfo，
        # 基座权重不会进入后续 gather/转换阶段。
        self.megatron_local_param_info_buckets = _get_megatron_local_param_info_buckets(
            self.args, self.model, name_filter=name_filter
        )

        # 🚨 ===== [YULIN-MOD] END =====

    def get_hf_weight_chunks(self, megatron_local_weights):
        rank = dist.get_rank()

        for megatron_local_param_infos in tqdm(
            self.megatron_local_param_info_buckets, disable=rank != 0, desc="Update weights"
        ):
            megatron_full_params = _get_megatron_full_params(megatron_local_param_infos, megatron_local_weights)
            hf_named_tensors = self._convert_to_hf_named_tensors(megatron_full_params, megatron_local_param_infos)
            yield hf_named_tensors
            del megatron_full_params

    def _convert_to_hf_named_tensors(self, megatron_full_params: Sequence[torch.Tensor], param_infos: list[ParamInfo]):
        hf_named_tensors = []
        for info, param in zip(param_infos, megatron_full_params, strict=False):
            hf_named_tensors.extend(
                convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config)
            )
        return hf_named_tensors


def _get_megatron_full_params(
    megatron_local_param_infos: Sequence[ParamInfo],
    megatron_local_weights,
) -> Sequence[torch.Tensor]:
    monkey_patch_torch_reductions()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()
    rank = dist.get_rank()
    # init params:
    params = []
    for info in megatron_local_param_infos:
        if dist.get_rank() == info.src_rank:
            params.append(
                torch.nn.Parameter(
                    megatron_local_weights[info.name]
                    .to(device=device_utils.make_current_torch_device(), non_blocking=True)
                    
                    # 🚨 ===== [YULIN-MOD] START: 在 PP/EP/TP 通信前连续化 LoRA 张量 =====
                    
                    # 坑 16:LoRA adapter 是分布式优化器连续 buffer 的非连续视图,
                    # broadcast/all_gather 前强制连续(全量 base 已连续→no-op)。

                    # adapter 可能是优化器 buffer 的非连续视图。
                    # 在 broadcast/all-gather 前强制转换成连续布局。
                    .contiguous(),

                    # 🚨 ===== [YULIN-MOD] END =====
                    requires_grad=False,
                )
            )
        else:
            params.append(torch.empty(info.shape, dtype=info.dtype, device=device_utils.make_current_torch_device()))
    device_utils.synchronize()

    # broadcast params across pp ranks
    if pp_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
                handles.append(
                    torch.distributed.broadcast(
                        param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group(), async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    # broadcast params across ep ranks
    if ep_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if ".experts." in info.name:
                src_rank = (
                    info.src_rank
                    if info.src_rank in dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
                    else rank
                )
                handles.append(
                    torch.distributed.broadcast(
                        param, src=src_rank, group=mpu.get_expert_model_parallel_group(), async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    # Set tp attrs for all params
    for info, param in zip(megatron_local_param_infos, params, strict=False):
        for key, value in info.attrs.items():
            setattr(param, key, value)

    # Batch async all_gather for all parameters
    gathered_params = all_gather_params_async(list(zip(megatron_local_param_infos, params, strict=False)))

    return gathered_params


def _get_megatron_local_param_info_buckets(
    args: Namespace, model: Sequence[torch.nn.Module], name_filter=None
) -> list[list[ParamInfo]]:
    """Partition params into buckets ≤ update_weight_buffer_size (with TP
    replication)."""

    # 🚨 ===== [YULIN-MOD] START: 参数分桶沿用相同的 name_filter =====

    param_infos = _get_megatron_local_param_infos(args, model, name_filter=name_filter)
    
    # 🚨 ===== [YULIN-MOD] END =====

    param_info_buckets = [[]]  # Start with one empty bucket
    buffer_size = 0  # Track current bucket size in bytes

    for info in param_infos:
        # Expert params use expert-TP size, others use regular-TP size
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()

        # Full param size = shard size × TP replicas (all-gather will reconstruct full param)
        param_size = info.size * tp_size

        # If adding this param exceeds limit AND current bucket has params: start new bucket
        if buffer_size + param_size > args.update_weight_buffer_size and len(param_info_buckets[-1]) > 0:
            param_info_buckets.append([])
            buffer_size = 0

        # Add param to current bucket and update size
        param_info_buckets[-1].append(info)
        buffer_size += param_size

    return param_info_buckets


def _get_megatron_local_param_infos(
    args: Namespace, model: Sequence[torch.nn.Module], name_filter=None
) -> list[ParamInfo]:
    """Build global param metadata: collect → exchange PP/EP → resolve
    duplicates (MTP virtual PP) by min src_rank → validate.

    Returns sorted ParamInfo identical across all ranks.

    ``name_filter`` (可选): 只保留满足谓词的全局参数名(基于名字的确定性过滤,
    各 rank 一致),用于把同步范围缩到 LoRA adapter 权重。
    """
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    param_infos = {}
    rank = dist.get_rank()
    for name, param in named_params_and_buffers(args, model):

        # 🚨 ===== [YULIN-MOD] START: 跳过所有非目标参数 =====

        if name_filter is not None and not name_filter(name):
            continue

        # 🚨 ===== [YULIN-MOD] END =====

        param_infos[name] = ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={
                "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                "partition_dim": getattr(param, "partition_dim", -1),
                "partition_stride": getattr(param, "partition_stride", 1),
                "parallel_mode": getattr(param, "parallel_mode", None),
            },
            size=param.numel() * param.element_size(),
            src_rank=rank,
        )

    if pp_size > 1:
        param_infos_list = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_pipeline_model_parallel_group()
        )
        for src_rank, infos in param_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                if name in param_infos:
                    assert args.mtp_num_layers is not None
                    old_info = param_infos[name]
                    if old_info.src_rank > src_rank:
                        param_infos[name] = info
                else:
                    param_infos[name] = info

    if ep_size > 1:
        param_infos_list = [None] * ep_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_expert_model_parallel_group()
        )
        for src_rank, infos in param_infos_list:
            for name, info in infos.items():
                if name not in param_infos:
                    # here we need to set the src_rank to the rank within the expert model parallel group
                    info = dataclasses.replace(info, src_rank=src_rank)
                    param_infos[name] = info

    param_infos = list(param_infos.values())
    param_infos = sorted(param_infos, key=lambda info: info.name)

    # Check all ranks has the same parameter info
    all_param_info_list = [None] * dist.get_world_size()
    dist.all_gather_object(
        obj=param_infos,
        object_list=all_param_info_list,
        group=get_gloo_group(),
    )
    for i, param_info in enumerate(param_infos):
        for infos in all_param_info_list:
            assert infos[i].name == param_info.name, f"Parameter name mismatch: {infos[i].name} != {param_info.name}"
            assert infos[i].shape == param_info.shape, (
                f"Parameter shape mismatch: {infos[i].shape} != {param_info.shape}"
            )
            assert infos[i].dtype == param_info.dtype, (
                f"Parameter dtype mismatch: {infos[i].dtype} != {param_info.dtype}"
            )

            # 🚨 ===== [YULIN-MOD] START: 提前发现跨 rank 集合通信条件不一致 =====

            # 坑 16:attrs(tensor_model_parallel/partition_dim)若各 rank 不一致,会让
            # all_gather_params_async 里部分 rank 进 gather 分支、部分跳过 → 集合通信错配 →
            # CUDA illegal memory access(且 NCCL watchdog 异步报,极难定位)。这里提前断言。
            
            for _k in ("tensor_model_parallel", "partition_dim", "parallel_mode"):
                assert infos[i].attrs.get(_k) == param_info.attrs.get(_k), (
                    f"Parameter attr '{_k}' mismatch for {param_info.name}: "
                    f"{infos[i].attrs.get(_k)} != {param_info.attrs.get(_k)}"
                )

            # 🚨 ===== [YULIN-MOD] END =====

    return param_infos
