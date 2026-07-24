# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import dataclasses

from relax.utils import megatron_bridge_utils
from relax.utils.misc import chunk_named_params_by_size

from ..misc_utils import strip_param_name_prefix
from ..weight_conversion import postprocess_hf_param
from ..weight_conversion.processors import quantize_params
from .hf_weight_iterator_base import HfWeightIteratorBase


# 🚨 ===== [YULIN-MOD] START: 判断参数是否属于 PEFT LoRA =====

def is_lora_weight_name(name: str) -> bool:
    """判断是否为 LoRA adapter 权重(HF/PEFT 命名:含 .lora_A. / .lora_B.)。"""
    return ".lora_A." in name or ".lora_B." in name

# 🚨 ===== [YULIN-MOD] END =====

class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

    def get_hf_weight_chunks(self, megatron_local_weights, weight_type: str = "base"):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):

            # 🚨 ===== [YULIN-MOD] START: 为 Bridge iterator 增加 base/lora 两种导出模式 =====
            
            if weight_type == "lora":
                # 让 Bridge 直接从模型读取 adapter，
                # 并在内部完成 TP/PP/EP gather 和融合 QKV 拆分。
                #
                # 不依赖 megatron_local_weights，也不 merge 到 base。
                named_weights = self._bridge.export_adapter_weights(
                    self.model,
                    cpu=False,
                    show_progress=False,
                )
            elif weight_type == "base":
                # 原有基座导出流程。
                conversion_tasks = self._bridge.get_conversion_tasks(self.model)
                conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
                # 即使模型已挂 adapter，也明确禁止将 adapter merge 进基座。
                named_weights = self._bridge.export_hf_weights(
                    self.model, cpu=False, conversion_tasks=conversion_tasks, merge_adapter_weights=False
                )
            else:
                raise ValueError(f"未知 weight_type={weight_type!r}（应为 'base' 或 'lora'）")
            
            # 🚨 ===== [YULIN-MOD] END =====

            def iter_quantized_named_weights():
                for hf_param_name, weight, megatron_param_name in named_weights:
                    
                    # 🚨 ===== [YULIN-MOD] START: 移除 PEFT 包装引入的 base_layer 路径，保持 SGLang 权重命名 =====
                    
                    # PEFT 包装原层后，参数路径中可能多出 .base_layer.。
                    # SGLang 仍按原模块路径匹配，因此需要删除这一层。
                    hf_param_name = hf_param_name.replace(".base_layer.", ".")
                    
                    # 🚨 ===== [YULIN-MOD] END =====
                    
                    processed_weight = postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    )

                    converted_named_params = [(hf_param_name, processed_weight)]

                    # 🚨 ===== [YULIN-MOD] START: adapter 保持原始精度，只有 base 执行量化 =====

                    # LoRA adapter 不量化(权重小且需保精度);base 维持原量化逻辑
                    if weight_type == "lora":
                        quantized_batch = converted_named_params
                    else:
                        quantized_batch = quantize_params(
                            args=self.args,
                            megatron_name=megatron_param_name,
                            converted_named_params=converted_named_params,
                            quantization_config=self.quantization_config,
                        )

                    # 🚨 ===== [YULIN-MOD] END =====

                    yield from quantized_batch

            # 🚨 ===== [YULIN-MOD] START: 防止 base 和 adapter 在导出结果中混合 =====
            
            # 按 weight_type 过滤:base 路径剔除 lora 权重,lora 路径只留 lora_A/lora_B
            def iter_filtered():
                for name, tensor in iter_quantized_named_weights():
                    # LoRA 模式只保留 lora_A/lora_B。
                    if weight_type == "lora" and not is_lora_weight_name(name):
                        continue
                    # base 模式排除所有 adapter。
                    if weight_type == "base" and is_lora_weight_name(name):
                        continue
                    yield name, tensor

            # 🚨 ===== [YULIN-MOD] END =====

            yield from chunk_named_params_by_size(
                iter_filtered(),
                chunk_size=self.args.update_weight_buffer_size,
            )


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert weight_dict_key in new_weight_dict, (
            f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"
        )

        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
