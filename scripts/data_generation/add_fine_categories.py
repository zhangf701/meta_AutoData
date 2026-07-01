"""add_fine_categories.py: Add sub_category field to dataset questions.

Phase 4 Task 4.1: Classifies each question into fine-grained technical
sub-categories using DeepSeek, replacing the single broad category per level.

Input: generated_eval_set_v3_l3_expanded.json
Output: generated_eval_set_v3_subcat.json
"""

import json
import os
import re
import sys
import time
from collections import Counter

from openai import OpenAI

sys.path.insert(0, r"D:\coding\meta_AutoData\scripts")
from data_generation.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STRONG_MODEL, DATA_DIR,
)

# ── Sub-category taxonomy ───────────────────────────────────────────────

L1_SUBCATEGORIES = [
    "稳定分类与定义",
    "数值阈值与限值",
    "设备参数与容量",
    "安全距离与净距",
    "电压等级与范围",
    "保护配置要求",
    "无功补偿参数",
    "电网结构参数",
]

L2_SUBCATEGORIES = [
    "稳定校核",
    "短路电流",
    "过电压保护",
    "接地设计",
    "设备选型",
    "无功补偿",
    "电网结构",
    "保护配置",
]

L3_SUBCATEGORIES = [
    "方案比选-接入系统",
    "方案比选-变电站设计",
    "风险研判-N-2",
    "多标准协调-无功配置",
    "综合决策-短路电流控制",
    "综合决策-继电保护",
    "综合决策-新能源并网",
    "方案比选-调峰与备用",
]

CATEGORY_PROMPT = """你是电力系统设计审查专家。请根据题目的query和answer，将其归类到最匹配的技术子类别中。

## 题目信息
Query: {query}
Answer: {answer}
当前分类: {category}
难度等级: {level}

## 可选的子类别
{options}

## 要求
1. 只从上述选项中选择最匹配的一个
2. 如果题目涉及多个主题，选择最主要的
3. 输出格式：只输出子类别名称，不要任何解释

子类别:"""


def classify_question(question, client):
    """Classify a question into a fine-grained sub-category."""
    level = question.get("level", "L2")

    if level == "L1":
        options = L1_SUBCATEGORIES
    elif level == "L3":
        options = L3_SUBCATEGORIES
    else:
        options = L2_SUBCATEGORIES

    options_text = "\n".join(f"- {opt}" for opt in options)

    prompt = CATEGORY_PROMPT.format(
        query=question.get("query", "")[:400],
        answer=question.get("expected_answer", "")[:400],
        category=question.get("category", ""),
        level=level,
        options=options_text,
    )

    try:
        response = client.chat.completions.create(
            model=STRONG_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        result = response.choices[0].message.content.strip()

        # Validate against options
        for opt in options:
            if opt in result:
                return opt

        # Fuzzy match
        result_clean = result.split("\n")[0].strip()
        if len(result_clean) <= 20 and any(c in result_clean for c in "稳定短路电压接地设备无功电网保护方案风险综合".split()):
            return result_clean

        return "其他"
    except Exception as e:
        return "其他"


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 4.1: Add fine-grained sub-categories")
    print("=" * 60)

    # Load latest dataset
    input_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_l3_expanded.json")
    if not os.path.exists(input_path):
        input_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_loopjudge.json")
    if not os.path.exists(input_path):
        input_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_l1_fixed.json")

    print(f"\nLoading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  {len(questions)} questions")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    stats = {"L1": Counter(), "L2": Counter(), "L3": Counter()}
    t0 = time.time()

    for i, q in enumerate(questions):
        level = q.get("level", "L2")
        sub_cat = classify_question(q, client)
        q["sub_category"] = sub_cat
        stats[level][sub_cat] += 1

        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/300] rate={(i+1)/elapsed*60:.0f}/min")
            for lvl in ["L1", "L2", "L3"]:
                top = stats[lvl].most_common(3)
                print(f"    {lvl} top: {top}")

        time.sleep(0.15)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    print(f"\n  Sub-category distribution:")
    for lvl in ["L1", "L2", "L3"]:
        n_cats = len(stats[lvl])
        print(f"  {lvl} ({n_cats} categories):")
        for cat, count in stats[lvl].most_common():
            print(f"    {cat}: {count}")

    output_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_subcat.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
