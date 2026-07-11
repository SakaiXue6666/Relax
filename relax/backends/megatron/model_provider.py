# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Adapt from https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/pretrain_gpt.py
import argparse
import functools
import inspect
import os
import pickle
import re
from contextlib import nullcontext
from typing import Literal

import torch
import torch.distributed as dist
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args

from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function


logger = get_logger(__name__)


# Adapt from https://github.com/volcengine/verl/blob/c3b20575d2bc815fcccd84bddb4c0401fc4b632b/verl/models/llama/megatron/layers/parallel_linear.py#L82
class LinearForLastLayer(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True
            if bias:
                self.bias.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


def get_model_provider_func(
    args: argparse.Namespace,
    role: Literal["actor", "critic"] = "actor",
):
    # Support custom model provider path (similar to --custom-rm-path for reward models)
    if getattr(args, "custom_model_provider_path", None):

        def wrapped_model_provider(
            pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None
        ) -> GPTModel:
            custom_model_provider = load_function(args.custom_model_provider_path)
            # Check if the custom provider supports vp_stage parameter
            has_vp_stage = "vp_stage" in inspect.signature(custom_model_provider).parameters
            if has_vp_stage:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            else:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process)
            # Apply critic output layer if needed
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(
                    input_size=model.config.hidden_size, output_size=1, config=model.config
                )
            return model

        return wrapped_model_provider

    if args.megatron_to_hf_mode == "bridge":
        from megatron.bridge import AutoBridge

        
        # 🚨 ===== [YULIN-MOD] START: 为旧版 Bridge 注册 Qwen3-Omni 输出层的并行类型 =====
        
        # Qwen3-Omni 为了节省显存，可能使用 LinearCrossEntropyModule 作为输出层。
        # 部分旧版 Megatron-Bridge 的 AutoMapping 注册表不认识该模块，
        # 加载权重时就无法判断它采用哪种张量并行方式。
        # （新版 bridge 已默认注册为 "column"），这里幂等补上，兼容老版本。
        try:
            from megatron.bridge.models.conversion.param_mapping import AutoMapping

            # LinearCrossEntropyModule 本质上属于列并行输出层。
            # 将它注册为 column，使 Bridge 能正确切分和加载它的权重。
            # 对已经内置该映射的新版 Bridge，这个操作应当是幂等或可安全失败的。
            AutoMapping.register_module_type("LinearCrossEntropyModule", "column")
        except Exception as exc:  # noqa: BLE001
            # 不把注册失败直接升级成致命错误：
            # 一种常见情况是新版 Bridge 已经注册过该类型。
            logger.warning("注册 LinearCrossEntropyModule 并行类型失败（可能新版已内置）：%s", exc)
        
        # 🚨 ===== [YULIN-MOD] END =====

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        # Override provider attributes with matching args values
        bridge_keys = [
            "attention_backend",
            "tensor_model_parallel_size",
            "sequence_parallel",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "variable_seq_lengths",
            "dsa_indexer_loss_coeff",
            "dsa_indexer_use_sparse_loss",
            "attention_softmax_in_fp32",
            "bias_dropout_fusion",
            "apply_rope_fusion",
            "recompute_granularity",
            "recompute_method",
            "recompute_num_layers",
            "distribute_saved_activations",
            "moe_router_load_balancing_type",
            "moe_router_dtype",
            "moe_aux_loss_coeff",
            "moe_token_dispatcher_type",
            "moe_enable_deepep",
            "moe_flex_dispatcher_backend",
            "use_audio_in_video",
            "freeze_language_model",
            "freeze_vision_model",
            "freeze_vision_projection",
            # https://github.com/redai-infra/Megatron-Bridge/commit/960bb5f18800d3e1fb9815e95daa185ab06c09ea
            "vision_dp_when_tp",
        ]

        args_dict = vars(args)
        for attr in vars(provider):
            if attr in args_dict and attr in bridge_keys:
                old_val = getattr(provider, attr)
                new_val = args_dict[attr]
                if old_val != new_val:
                    logger.info(f"Override provider.{attr}: {old_val!r} -> {new_val!r}")
                setattr(provider, attr, new_val)

        # Handle name-mismatched attributes that require explicit mapping
        if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
            provider.num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
        if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
            provider.num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers

        if args.fp16:
            provider.fp16 = True
            provider.bf16 = False
            provider.params_dtype = torch.float16
        elif args.bf16:
            provider.fp16 = False
            provider.bf16 = True
            provider.params_dtype = torch.bfloat16

        provider.finalize()

        # Pickle provider for offline inspection / reproducibility (only on rank 0)
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_path = getattr(args, "save", None) or "/tmp/relax"
            os.makedirs(save_path, exist_ok=True)
            pkl_path = os.path.join(save_path, "transformer_config.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(provider, f)
            logger.info(f"Provider config saved to {pkl_path}")

        return provider.provide

    def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None) -> GPTModel:
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        use_te = args.transformer_impl == "transformer_engine"

        # Experimental loading arguments from yaml
        config: TransformerConfig = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
            # Allow the spec to be a function so that user can use customized Megatron easier.
            if callable(transformer_layer_spec):
                transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
        else:
            if args.num_experts:
                # Define the decoder block spec
                kwargs = {
                    "use_transformer_engine": use_te,
                }
                if vp_stage is not None:
                    kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except Exception as e:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                ) from e

        kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
        }

        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage

        if args.mtp_num_layers:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

            mtp_kwargs = {
                "use_transformer_engine": use_te,
            }
            if vp_stage is not None:
                mtp_kwargs["vp_stage"] = vp_stage

            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
            kwargs["mtp_block_spec"] = mtp_block_spec

        with build_model_context(**build_model_context_args):
            model = GPTModel(**kwargs)

        if post_process and role == "critic":
            model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)

        return model

    return model_provider


def wrap_model_provider_with_freeze(original_provider, args):
    def wrapped_provider(pre_process=True, post_process=True, vp_stage=None):
        sig = inspect.signature(original_provider)
        if "vp_stage" in sig.parameters:
            model = original_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        else:
            model = original_provider(pre_process=pre_process, post_process=post_process)

        freeze_model_params(model, args)

        return model

    return wrapped_provider


def freeze_model_params(model: GPTModel, args: argparse.Namespace):
    if args.only_train_params_name_list:
        for name, param in model.named_parameters():
            param.requires_grad = False
            for pattern in args.only_train_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = True
                    break

    if args.freeze_params_name_list:
        for name, param in model.named_parameters():
            for pattern in args.freeze_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = False
                    break


# 🚨 ===== [YULIN-MOD] START: 给模型挂 LoRA，并保证只有正确的 adapter 可训练 =====

def apply_lora_to_model(model: GPTModel, args: argparse.Namespace) -> None:
    """用 Megatron-Bridge 的 LoRA 给单个模型 chunk 挂 adapter。

    要点（已由 verify_lora_attach.py 在真实 Megatron 环境实测）：
      - 用普通 ``LoRA``（不是 ``VLMLoRA``）：它先冻结全部基座、再让 adapter 可训，
        不依赖任何 LLaVA 风格属性名，多模态塔自动冻结。
      - target 必须用通配符限定到 thinker 的 ``language_model``，否则会误挂到
        audio/vision 塔（产出 sglang 无法加载的张量）。
      - adapter 权重命名为 ``...linear_qkv.adapter.linear_in/out``，
        与 sglang 侧 ``qkv_proj.lora_A/B`` 契约一致。
    """
    # 不同版本的 Megatron-Bridge 导出 LoRA 类的位置不同。
    # redai fork 可能只在 megatron.bridge.peft.lora 子模块中提供 LoRA，
    # 而某些新版本会从 megatron.bridge.peft 顶层重新导出。
    try:
        from megatron.bridge.peft.lora import LoRA
    except ImportError:
        from megatron.bridge.peft import LoRA

    # 根据命令行参数构造 Megatron-Bridge 的 LoRA 配置。
    peft = LoRA(
        # 指定在哪些 Megatron 模块上挂 adapter。
        # 默认值只匹配 Thinker 文本模型的 linear_qkv 和 linear_proj，
        # 避免误挂到视觉塔或音频塔。
        target_modules=list(args.lora_target_modules),
        # LoRA 的低秩维度 r。
        dim=args.lora_rank,
        # LoRA 缩放系数 alpha；实际增量通常按 alpha/r 缩放。
        alpha=args.lora_alpha,
        # adapter 分支上的 dropout。
        dropout=args.lora_dropout,
    )
    # 就地修改模型：
    # 1. 冻结基座参数；
    # 2. 在匹配模块上插入 adapter；
    # 3. 让 adapter 参数保持 requires_grad=True；
    # 4. 切换到训练状态。
    peft(model, training=True)

    # 在创建优化器之前验证 LoRA 是否真正挂载成功。
    _assert_lora_attached(model, args)


def _assert_lora_attached(model: GPTModel, args: argparse.Namespace) -> None:
    """挂 LoRA 后的防御性自检——在构建优化器之前把配置错误暴露出来。

    Megatron 的优化器(`_get_param_groups`)和 DDP grad buffer 都以
    ``requires_grad`` 为准，只收 adapter，因此“只优化 adapter”由 Megatron 原生保证。
    真正隐蔽的坑是 ``--lora-target-modules`` 通配符没匹配上任何模块：PEFT 会把
    基座全冻结却一个 adapter 都没挂，导致优化器参数组为空、训练静默 noop。
    另一个坑是误挂到 vision/audio 塔，产出 sglang 无法加载的张量。
    """
    trainable: list[str] = []
    n_trainable_elems = 0
    n_frozen = 0
    # 统计模型中所有可训练和冻结参数。
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
            n_trainable_elems += param.numel()
        else:
            n_frozen += 1

    # 第 1 重检查：必须至少存在一个可训练参数。
    #
    # 如果 target_modules 通配符没有匹配任何模块，
    # PEFT 可能已经冻结全部基座，但没有成功插入任何 adapter。
    # 这种情况下优化器参数组为空，训练看似运行，实际上完全没有学习。
    assert trainable, (
        "LoRA 已应用但没有任何可训参数：--lora-target-modules "
        f"{list(args.lora_target_modules)} 很可能没匹配到任何模块。"
    )

    # 识别 Megatron-Bridge 或 PEFT 风格的 adapter 参数名。
    def _is_adapter(n: str) -> bool:
        low = n.lower()
        return (".adapter." in n) or ("lora_a" in low) or ("lora_b" in low)

    # 第 2 重检查：所有可训练参数都必须属于 adapter。
    #
    # 如果出现普通基座参数，说明 PEFT 没有把基座完全冻结，
    # 此时就不再是“只训练 LoRA”。
    non_adapter = [n for n in trainable if not _is_adapter(n)]
    assert not non_adapter, (
        "LoRA 模式下出现非 adapter 的可训参数（基座没冻干净）：" f"{non_adapter[:5]}"
    )

    # 第 3 重检查：检测 adapter 是否误挂到视觉塔或音频塔。
    #
    # 当前同步和 SGLang 加载链路只支持 Thinker 文本注意力层；
    # 如果 adapter 挂到 audio/vision 模块，之后会生成 SGLang 无法加载的张量。
    suspicious = [n for n in trainable if re.search(r"visual|vision|audio", n)]
    if suspicious:
        logger.warning(
            "[LoRA] 检测到 adapter 挂在疑似多模态塔上（应只挂 thinker 文本侧）：%s",
            suspicious[:5],
        )

    # 多进程训练时只让 rank 0 输出统计，避免日志重复。
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    logger.info(
        "[LoRA] adapter 已挂载：可训参数 %d 个 / %.3fM 元素，冻结 %d 个；样例=%s",
        len(trainable),
        n_trainable_elems / 1e6,
        n_frozen,
        trainable[:3],
    )


# relax/backends/megatron/model.py
def wrap_model_provider_with_lora(original_provider, args):
    """在模型 provider 外再包一层：构建出 GPTModel 后立刻挂 LoRA。

    必须在 DDP 包裹之前应用（即在 provider 内部），这样 adapter 会随后被
    一并 DDP 包裹、移动到设备、转 dtype。

    签名透明性（关键，见 LORA_RL_INTEGRATION.md 坑 9）：
      Megatron-LM 的 ``build_model`` 会按 provider 的**真实签名**决定传哪些
      kwarg（如 ``config``/``pg_collection``）。bridge 模式下 ``original_provider``
      是 ``ModelProvider.provide``，其签名只有 ``(pre_process, post_process,
      vp_stage)``，并不收 ``config``。若 wrapper 暴露成 ``(*args, **kwargs)``，
      Megatron 的签名探测会误以为"什么都收"而把 ``config`` 传进来，转发给
      ``provide`` 就撞 ``unexpected keyword argument 'config'``。

    故这里：
      1) 用 ``functools.wraps`` 把 ``original_provider`` 的签名透传出去，让
         Megatron 的探测看到与裸 provider 完全一致的参数；
      2) 运行期再按原签名过滤一次 kwarg（原 provider 没有 ``**kwargs`` 时丢掉
         它不认识的键），对"无条件传 config"的 Megatron 版本也安全。
    """
    # 读取原 provider 的真实函数签名。
    # Megatron 会根据 provider 的签名决定向它传哪些关键字参数。
    try:
        sig = inspect.signature(original_provider)
    except (TypeError, ValueError):
        sig = None

    # 判断原 provider 是否接受任意 **kwargs。
    accepts_var_kw = sig is not None and any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    # 得到原 provider 明确声明可以接受的参数名。
    allowed_keys = set(sig.parameters) if sig is not None else None

    # functools.wraps 不只是复制名字和文档；
    # 它还通过 __wrapped__ 保留原 provider 的签名信息，
    # 避免 Megatron 误判 wrapper 可以接受任意参数。
    @functools.wraps(original_provider)
    def wrapped_provider(*provider_args, **provider_kwargs):
        # 某些 Megatron 版本会无条件传 config、pg_collection 等参数。
        # 如果原 provider 没有 **kwargs，就只保留它真正声明过的参数。
        if allowed_keys is not None and not accepts_var_kw:
            provider_kwargs = {k: v for k, v in provider_kwargs.items() if k in allowed_keys}
        # 先按照原逻辑创建基础模型。
        model = original_provider(*provider_args, **provider_kwargs)
        # 必须在 DDP 包装、设备迁移和 dtype 转换之前挂 LoRA。
        # 这样新插入的 adapter 才会跟随模型一起进入后续 Megatron 流程。
        apply_lora_to_model(model, args)
        return model

    return wrapped_provider

# 🚨 ===== [YULIN-MOD] END =====