# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# 🚨 ===== [YULIN-MOD] START: 为翻译任务提供平滑的句级 BLEU 奖励 =====

"""自包含的句级 BLEU 奖励（翻译任务）。

- 返回 [0, 1] 的连续分数：天然有方差，避免 0/1 二值奖励的「全对/全错」零方差。
- 带平滑（对缺失的高阶 n-gram 用 1/(2*total) 兜底），使部分匹配也得正分，梯度更平滑。
- 面向英文 reference（按空白+标点切词、小写）。中文 reference 走字符级兜底。
- response/label 均优先从 <answer></answer> 取，取不到则用整串 strip()。
"""

import math
import re
from collections import Counter


try:
    import sacrebleu  # 与 my_omni quality_reward 一致：中文用 zh tokenizer
    _HAS_SACREBLEU = True
except ImportError:
    _HAS_SACREBLEU = False


ANS_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def _extract(text: str) -> str:
    m = ANS_TAG.search(text or "")
    return (m.group(1).strip() if m else (text or "").strip())


def _ref_text(label) -> str:
    """从 label 取参考译文：支持 str / {'ground_truth': ...} / list。"""
    if isinstance(label, dict):
        ref = label.get("ground_truth") or label.get("label") or ""
    elif isinstance(label, (list, tuple)):
        ref = label[0] if label else ""
    else:
        ref = label
    return _extract(str(ref or ""))


def _sacre_tokenize(tgt_lang, ref: str) -> str:
    lang = (tgt_lang or "").lower()
    if lang in ("zh", "zh-cn") or _CJK.search(ref):
        return "zh"
    if lang == "ja":
        return "ja-mecab"
    if lang == "ko":
        return "ko-mecab"
    return "13a"


def _tokenize(s: str) -> list[str]:
    s = s.strip()
    if _CJK.search(s):
        # 中文无空格：按字符切（去空白）。
        return [ch for ch in s if not ch.isspace()]
    s = s.lower()
    s = re.sub(r"([.,!?;:\"()\[\]])", r" \1 ", s)
    return s.split()


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(hyp: list[str], ref: list[str], max_n: int = 4) -> float:
    if not hyp or not ref:
        return 0.0
    log_p = []
    for n in range(1, max_n + 1):
        hyp_ng = _ngram_counts(hyp, n)
        ref_ng = _ngram_counts(ref, n)
        total = sum(hyp_ng.values())
        if total == 0:
            # hyp 比 n 还短：用平滑兜底，避免直接 0。
            log_p.append(math.log(1.0 / (2.0 * max_n)))
            continue
        overlap = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
        p = overlap / total
        if p == 0.0:
            p = 1.0 / (2.0 * total)  # 平滑：避免 log(0)
        log_p.append(math.log(p))
    geo_mean = math.exp(sum(log_p) / max_n)
    hl, rl = len(hyp), len(ref)
    bp = 1.0 if hl > rl else math.exp(1.0 - rl / max(1, hl))
    return bp * geo_mean


# 奖励入口
def get_bleu_reward(response, label, metadata=None) -> float:
    """句级 BLEU 奖励 ∈ [0,1]。

    - label 支持裸串 / {'ground_truth': ...} / list（S2TT 用 dict 形态）。
    - 装了 sacrebleu 时用它（中文走 zh tokenizer），对齐 my_omni quality_reward；
      否则回退自带的字符级/空白级平滑 BLEU。
    - metadata['tgt_lang'] 用于选 tokenizer；缺失则按 ref 是否含 CJK 自动判定。
    """
    hyp_str = _extract(response)
    ref_str = _ref_text(label)
    if not hyp_str or not ref_str:
        return 0.0

    if _HAS_SACREBLEU:
        tgt_lang = (metadata or {}).get("tgt_lang") if isinstance(metadata, dict) else None
        tok = _sacre_tokenize(tgt_lang, ref_str)
        try:
            return max(0.0, min(1.0, sacrebleu.sentence_bleu(hyp_str, [ref_str], tokenize=tok).score / 100.0))
        except Exception:
            try:
                return max(0.0, min(1.0, sacrebleu.sentence_bleu(hyp_str, [ref_str]).score / 100.0))
            except Exception:
                pass

    return float(sentence_bleu(_tokenize(hyp_str), _tokenize(ref_str)))

# 🚨 ===== [YULIN-MOD] END =====