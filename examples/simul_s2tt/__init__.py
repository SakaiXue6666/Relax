# 🚨 ===== [YULIN-MOD] START: 定义定长音频分块的多轮同传 rollout 示例 =====
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""同传（simultaneous speech-to-text translation）多轮 rollout 示例。

把已实现的单轮 s2tt（英语音频→中文，核心走 relax.engine.rollout.sglang_rollout
的内置 generate）扩展成「定长 chunk 多轮」：整段音频按固定时长（默认 960ms）切块，
每轮喂一个 chunk、模型输出一段增量译文，直到 chunk 用完。

- 多轮骨架参考 examples/deepeyes/rollout.py（同一套 sglang_rollout 编排 + 同一个
  sglang engine，仅替换「生成一条样本」这层）。
- 与 deepeyes 的差异：环境是「脚本化」的（每轮直接吐下一个 chunk，不依赖模型动作），
  且观测是音频而非图像。
"""
# 🚨 ===== [YULIN-MOD] END =====