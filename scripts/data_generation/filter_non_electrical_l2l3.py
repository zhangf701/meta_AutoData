"""Filter non-electrical L2/L3 questions from v4 dataset using LLM classification.

Identifies questions whose core expertise is NOT electrical engineering
(civil, geological, environmental, fire protection, HVAC, etc.) even when
the query mentions electrical terms like "变电站" or "配电装置".

Usage:
  python filter_non_electrical_l2l3.py          # Full classification
  python filter_non_electrical_l2l3.py --dry-run  # Preview only, no API calls
"""

import json
import os
import re
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)

from data_generation.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STRONG_MODEL, OUTPUT_DIR,
)
from openai import OpenAI

V4_DATASET = os.path.join(OUTPUT_DIR, "generated_eval_set_v4.json")
FILTERED_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "l2l3_filter_results.json",
)

BATCH_CLASSIFY_PROMPT = """判断以下电力系统试题涉及的核心专业知识是否属于电气工程领域。

电气工程领域（保留）：
电力系统稳定分析、继电保护、高电压与绝缘、电力电子、电机与变压器、
输配电工程、变电站电气设计（电气主接线、配电装置选型、接地网设计、
站用电系统、无功补偿）、直流输电、新能源并网、短路电流计算、
电网规划与运行、安全自动装置。

非电气领域（剔除）：
土建结构（基础、构架、房屋建筑）、岩土地质勘察、防洪防涝、
给排水工程、暖通空调、消防工程（防火隔墙、灭火系统）、
环境保护（水土保持、噪声治理、碳排放评估）、施工组织设计、
工程造价与经济评价、通信协议（非电力部分）、城乡规划。

判断标准：看答案所解决的核心问题需要哪个领域的专业知识。
如果答案主要依赖土建/地质/环保/消防等非电气专业的判断，即使题干提到变电站，也属于非电气。

试题列表：
{questions_text}

输出一个JSON数组，每道题对应一个对象：
[{{"id":"题目ID","is_electrical":true/false,"reason":"简短理由≤30字"}}]"""


def classify_batch(client, questions_batch):
    """Classify a batch of questions in one API call."""
    questions_text = ""
    for q in questions_batch:
        qid = q.get("question_id", "?")
        query = q.get("query", "")[:400]
        answer = q.get("expected_answer", "")[:400]
        questions_text += f"\n--- 题目ID: {qid} ---\n题干: {query}\n答案: {answer}\n"

    prompt = BATCH_CLASSIFY_PROMPT.format(questions_text=questions_text)

    try:
        resp = client.chat.completions.create(
            model=STRONG_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        response_text = resp.choices[0].message.content.strip()

        # Parse JSON array
        results = []
        try:
            results = json.loads(response_text)
        except json.JSONDecodeError:
            # Try extract array from markdown or raw text
            match = re.search(r'\[[\s\S]*\]', response_text)
            if match:
                try:
                    results = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        if results and isinstance(results, list) and len(results) > 0:
            output = []
            for r in results:
                output.append({
                    "question_id": r.get("id", "?"),
                    "is_electrical": r.get("is_electrical", True),
                    "reason": r.get("reason", ""),
                })
            return output
    except Exception as e:
        print(f"  [ERR] Batch classification failed: {e}")

    # Fallback: keyword-based per question
    return [classify_question_fallback(q) for q in questions_batch]


def classify_question_fallback(q):
    """Keyword-based fallback classification."""
    query = q.get("query", "")
    answer = q.get("expected_answer", "")
    combined = query + answer
    is_elec = not bool(re.search(
        r'防洪|防涝|土石方|桩基|混凝土|给排水|暖通|消防|防火|'
        r'环保|水土保持|绿化|噪声|噪音|碳排放|自然保护区|人文遗址|'
        r'飞机场|生活污水|采暖|通风空调|排烟|地基|抗倾覆',
        combined
    ))
    return {
        "question_id": q.get("question_id", "?"),
        "is_electrical": is_elec,
        "reason": "keyword fallback",
        "query_preview": query[:100],
    }


def run_filter(dry_run=False):
    """Run the full L2/L3 non-electrical classification."""
    if not os.path.exists(V4_DATASET):
        print(f"[FATAL] v4 dataset not found: {V4_DATASET}")
        return

    with open(V4_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)

    l2 = [q for q in data if q.get("level") == "L2"]
    l3 = [q for q in data if q.get("level") == "L3"]
    all_questions = l2 + l3

    print(f"Loaded: {len(l2)} L2 + {len(l3)} L3 = {len(all_questions)} questions")

    if dry_run:
        # Quick keyword-based preview
        non_elec_kw = re.compile(
            r'防洪|防涝|土石方|桩基|混凝土|给排水|暖通|消防|防火|'
            r'环保|水土保持|绿化|噪声|噪音|碳排放|自然保护区|人文遗址|'
            r'飞机场|生活污水|采暖|通风空调|排烟|地基|抗倾覆'
        )
        for q in all_questions:
            if non_elec_kw.search(q["query"] + q["expected_answer"]):
                print(f"  [KEYWORD HIT] {q['question_id']}: "
                      f"{non_elec_kw.search(q['query'] + q['expected_answer']).group(0)}")
        return

    if not DEEPSEEK_API_KEY:
        print("[FATAL] DEEPSEEK_API_KEY not set")
        return

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    results = []
    non_elec_ids = []
    total = len(all_questions)

    print(f"Classifying {total} questions via DeepSeek LLM (batch size=5)...")
    batch_size = 5
    results = []
    non_elec_ids = []

    for batch_start in range(0, total, batch_size):
        batch = all_questions[batch_start:batch_start + batch_size]
        batch_results = classify_batch(client, batch)
        results.extend(batch_results)

        for r in batch_results:
            if not r["is_electrical"]:
                non_elec_ids.append(r["question_id"])
                print(f"  [{min(batch_start+batch_size, total)}/{total}] "
                      f"{r['question_id']}: NON-ELECTRICAL — {r['reason']}")
            else:
                pass  # silently accept

        if (batch_start + batch_size) % 50 == 0:
            print(f"  [{min(batch_start+batch_size, total)}/{total}] "
                  f"Processed... ({len(non_elec_ids)} non-electrical so far)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Classification Complete")
    print(f"{'='*60}")
    print(f"Total classified: {total}")
    print(f"Electrical (keep): {total - len(non_elec_ids)}")
    print(f"Non-electrical (remove): {len(non_elec_ids)}")
    print(f"\nNon-electrical IDs: {non_elec_ids}")

    # Save results
    output = {
        "total": total,
        "electrical_count": total - len(non_elec_ids),
        "non_electrical_count": len(non_elec_ids),
        "non_electrical_ids": non_elec_ids,
        "results": results,
    }
    with open(FILTERED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {FILTERED_OUTPUT}")

    # Print ready-to-use ID list for the regeneration script
    print(f"\nCopy-paste for removal:")
    print(f"NON_ELECTRICAL_IDS = {non_elec_ids}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter non-electrical L2/L3 questions"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick keyword-based preview without API calls")
    args = parser.parse_args()
    run_filter(dry_run=args.dry_run)
