"""fix_l3_short_queries.py: Expand L3 queries below 300-char minimum.

Phase 3 Task 3.2: Targets 35 L3 questions with query < 300 chars.
Reuses PARAMETER_SANDBOX from expand_scenarios.py for constraint-injected
prompt expansion via DeepSeek.

Input: generated_eval_set_v3_loopjudge.json (or l1_fixed.json)
Output: generated_eval_set_v3_l3_expanded.json
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
from data_generation.expand_scenarios import PARAMETER_SANDBOX, STANDARD_SCOPE


# ── L3 query expansion prompt ───────────────────────────────────────────

L3_EXPAND_PROMPT = """你是电力系统规划设计专家。下面的L3评测题目场景描述不足300字，需要扩展。

## 原始题目
Query: {query}
Answer概要: {answer_summary}
涉及标准: {source_standard}

## 参数沙箱约束
电压等级常用参数范围:
- 220kV: 线路容量100-600MW, 短路电流10-50kA, 变压器90-360MVA
- 330kV: 线路容量300-1000MW, 短路电流15-50kA, 变压器150-750MVA
- 500kV: 线路容量600-3000MW, 短路电流20-63kA, 变压器500-1500MVA
- 750kV: 线路容量1500-5000MW, 短路电流25-63kA, 变压器1000-2100MVA

## 任务
1. 保持原始query中方案A和方案B的核心技术逻辑不变
2. 补充具体的工程参数（电压等级、容量、距离、短路电流等）使场景描述≥300字
3. 参数必须在上述沙箱约束范围内
4. 保持原始answer不变

## 输出格式
纯JSON对象:
{{"query":"扩展后的完整query（≥300字）","expected_answer":"保持原始answer不变（直接复制）"}}

只输出JSON。"""


def expand_query(question, client):
    """Expand a short L3 query with realistic engineering parameters."""
    query = question.get("query", "")
    answer = question.get("expected_answer", "")
    source_std = question.get("source_standard", "")

    # Build constraint hint from sandbox
    prompt = L3_EXPAND_PROMPT.format(
        query=query,
        answer_summary=answer[:300],
        source_standard=source_std,
    )

    try:
        response = client.chat.completions.create(
            model=STRONG_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2048,
        )
        text = response.choices[0].message.content

        # Parse JSON
        result = None
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            code = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if code:
                try:
                    result = json.loads(code.group(1))
                except json.JSONDecodeError:
                    pass
            if not result:
                obj = re.search(r"\{[\s\S]*\}", text)
                if obj:
                    try:
                        result = json.loads(obj.group(0))
                    except json.JSONDecodeError:
                        pass

        if result:
            new_query = result.get("query", query)
            if len(new_query) >= 280:  # Allow 20 char tolerance
                question["query"] = new_query
                return True
            else:
                question["query"] = new_query
                return len(new_query) > len(query)  # Improved even if below 300
        return False
    except Exception as e:
        print(f"    [Expand Error] {e}")
        return False


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 3.2: Expand L3 short queries (< 300 chars)")
    print("=" * 60)

    # Determine input file
    loopjudge_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_loopjudge.json")
    l1_fixed_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_l1_fixed.json")

    for path in [loopjudge_path, l1_fixed_path]:
        if os.path.exists(path):
            input_path = path
            break

    print(f"\nLoading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    l3 = [q for q in questions if q.get("level") == "L3"]
    short_l3 = [q for q in l3 if len(q.get("query", "")) < 300]

    print(f"  Total L3: {len(l3)}")
    print(f"  Short (<300 chars): {len(short_l3)}")

    if not short_l3:
        print("  No short L3 queries found. Done.")
        return

    # Show before stats
    lengths_before = [len(q.get("query", "")) for q in short_l3]
    print(f"  Before: min={min(lengths_before)}, max={max(lengths_before)}, "
          f"mean={sum(lengths_before)/len(lengths_before):.0f}")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    expanded = 0
    improved = 0
    failed = 0
    t0 = time.time()

    for i, q in enumerate(short_l3):
        qid = q.get("question_id", "?")
        before_len = len(q.get("query", ""))

        success = expand_query(q, client)
        after_len = len(q.get("query", ""))

        if after_len >= 300:
            expanded += 1
        elif after_len > before_len:
            improved += 1
        else:
            failed += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(short_l3)}] >=300={expanded} improved={improved} "
                  f"failed={failed} rate={(i+1)/elapsed*60:.0f}/min")

        time.sleep(0.3)

    elapsed = time.time() - t0
    lengths_after = [len(q.get("query", "")) for q in short_l3]
    meets_target = sum(1 for l in lengths_after if l >= 300)

    print(f"\n  Done in {elapsed:.0f}s")
    print(f"  >=300 chars: {expanded} -> {meets_target}/{len(short_l3)}")
    print(f"  Improved (<300): {improved}")
    print(f"  Failed: {failed}")

    # Overall L3 stats
    all_l3_lengths = [len(q.get("query", "")) for q in l3]
    meets = sum(1 for l in all_l3_lengths if l >= 300)
    print(f"\n  Overall L3 >=300 chars: {meets}/{len(l3)} ({meets/len(l3)*100:.0f}%)")
    print(f"  L3 query length: min={min(all_l3_lengths)}, max={max(all_l3_lengths)}, "
          f"mean={sum(all_l3_lengths)/len(all_l3_lengths):.0f}")

    # Show sample
    if short_l3:
        q = short_l3[0]
        print(f"\n  Sample [{q.get('question_id')}]:")
        print(f"    Query length: {len(q.get('query'))} chars")
        print(f"    First 200: {q['query'][:200]}...")

    output_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_l3_expanded.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
