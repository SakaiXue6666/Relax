# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import logging

from relax.utils import device as device_utils


try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        device_utils.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")


def patch_rotary_embedding(cls):
    _original_forward = cls.forward

    # 🚨 ===== [YULIN-MOD] START: 忽略不兼容的 rotary embedding 关键字参数 =====

    def _patched_forward(self, *args, **kwargs):
        # 保留位置参数，但不把 kwargs 转发给旧版原始 forward。
        #
        # 用于兼容调用方和当前 Megatron rotary embedding 实现之间的签名差异：
        # 调用方可能传入新版才支持的关键字参数，而旧版 forward 不认识。
        return _original_forward(self, *args)

    # 🚨 ===== [YULIN-MOD] END =====

    cls.forward = _patched_forward


try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
        Qwen3VLMoETextRotaryEmbedding,
        Qwen3VLTextRotaryEmbedding,
    )

    patch_rotary_embedding(Qwen3VLTextRotaryEmbedding)
    patch_rotary_embedding(Qwen3VLMoETextRotaryEmbedding)
except ImportError:
    pass

try:
    from megatron.bridge.models.qwen_omni.modelling_qwen3_omni.text_model import Qwen3OmniMoeThinkerTextRotaryEmbedding

    patch_rotary_embedding(Qwen3OmniMoeThinkerTextRotaryEmbedding)
except ImportError:
    pass


# 🚨 ===== [YULIN-MOD] START: 修复 Qwen3-Omni 多段音频位置索引的设备不一致 =====

def patch_qwen3_omni_rope_index_device():
    """修 redai bridge get_rope_index 的「多段音频 device mismatch」bug。

    该函数把 M-RoPE 位置 id 全程在 CPU 上用 torch.arange(...) 构建、最后再 .to(device)。
    作者在 video 分支已显式 `second_per_grids[i].cpu()` 保持 CPU，却漏了 audio_seqlens
    （model.py 里 = feature_attention_mask.sum(1)，是 CUDA 张量）。当一条序列含 >=2 段音频时，
    第 1 段处理后计数器 st 被 audio_len(CUDA) 污染，第 2 段那轮 `st_idx += text_len(CUDA)`
    触发 'cuda and cpu' 崩溃。单轮/单段音频只跑一轮不暴露（单轮 s2tt 因此一直没事）。

    修法：包一层，调用前把 audio_seqlens / *_grid_thw / second_per_grids 统一 .cpu()，
    与作者对 video 的处理一致；结果 position_ids 仍在 input_ids.device，数值完全不变。
    幂等：重复调用不会二次包裹。
    """
    import torch

    from megatron.bridge.models.qwen_omni.modelling_qwen3_omni import model as _omni_model

    # 幂等检查：如果已经打过补丁，直接返回。
    # 防止模块被重复 import 时套上多层 wrapper。
    if getattr(_omni_model.get_rope_index, "_relax_device_patched", False):
        return

    # 保存 Bridge 原始实现。
    _orig_get_rope_index = _omni_model.get_rope_index

    def _patched_get_rope_index(*args, **kwargs):
        # get_rope_index 内部使用 CPU torch.arange 等操作构造位置索引。
        # 这些长度/网格参数如果留在 CUDA，会在多段累计过程中污染 CPU 计数器。
        for _k in ("audio_seqlens", "image_grid_thw", "video_grid_thw", "second_per_grids"):
            _v = kwargs.get(_k)
            if isinstance(_v, torch.Tensor):
                # 统一放到 CPU，仅改变计算设备，不改变长度或网格数值。
                kwargs[_k] = _v.cpu()
        # 调用 Bridge 原始实现。
        # 原实现最后仍会按 input_ids.device 返回 position_ids。
        return _orig_get_rope_index(*args, **kwargs)

    # 给 wrapper 打标记，供上面的幂等检查识别。
    _patched_get_rope_index._relax_device_patched = True
    # 用 wrapper 替换模块级函数。
    _omni_model.get_rope_index = _patched_get_rope_index


try:
    # import Relax Megatron backend 时自动应用补丁。
    patch_qwen3_omni_rope_index_device()
except ImportError:
    # 如果当前 Bridge 版本没有 Qwen3-Omni 模块，
    # 不影响其他模型继续使用 Relax。
    pass

# 🚨 ===== [YULIN-MOD] END =====

logging.getLogger("megatron").setLevel(logging.WARNING)
