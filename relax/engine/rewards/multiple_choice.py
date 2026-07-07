# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re


ANS_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


def extract_answer(text: str) -> str:
    # 有 <answer> 标签就取标签内容；没有则退回裸串 strip()
    # （label 常是裸字母如 "C"，response 才带标签——和 openr1mm 的契约一致）
    m = ANS_TAG.search(text)
    return m.group(1).strip() if m else text.strip()


def get_multiple_choice_reward(response, label):
    response = extract_answer(response)
    label = extract_answer(label)
    reward = 1.0 if response == label else 0.0
    return reward
