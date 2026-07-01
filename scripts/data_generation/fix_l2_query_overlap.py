"""fix_l2_query_overlap.py: Rewrite L2 queries with high scenario/question overlap.

Phase 层二: Detects L2 questions where the scenario and question sections
have excessive textual overlap (bigram Jaccard >= 0.25), and rewrites the
question part to use concise reference-style phrasing.

Input: generated_eval_set_v4.json
Output: generated_eval_set_v4.json (in-place update of 27 questions)
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


# ── Overlap detection ────────────────────────────────────────────────────

def bigrams(text):
    return set(text[i:i+2] for i in range(len(text)-1))


def overlap_ratio(query):
    """Compute bigram Jaccard between scenario and question parts."""
    sc_m = re.search(r'【场景】(.*?)(?=【问题】|$)', query, re.DOTALL)
    qu_m = re.search(r'【问题】(.*?)$', query, re.DOTALL)
    if not sc_m or not qu_m:
        return 0
    sc = sc_m.group(1).strip()
    qu = qu_m.group(1).strip()
    if len(sc) < 20 or len(qu) < 10:
        return 0
    sc_bg = bigrams(sc)
    qu_bg = bigrams(qu)
    union = sc_bg | qu_bg
    return len(sc_bg & qu_bg) / len(union) if union else 0


# ── Query rewrite ────────────────────────────────────────────────────────

REWRITE_PROMPT = """你是一名电力系统数据集质量审核专家。以下L2题目的【场景】和【问题】存在内容重复，需要重写问题部分。

## 当前query
{query}

## 要求
1. 【场景】部分保持不变
2. 只重写【问题】部分，改为简洁的引用式提问
   - 禁止重复场景中已出现的电压等级、设备参数、标准编号
   - 使用"在上述场景下，请判断/分析/计算..."的引用式开头
   - 问题长度不超过80字
3. 问题仍应要求推理判断（非简单查表）
4. 保持与场景的技术关联性

## 输出格式
纯JSON对象:
{{"query":"【场景】...（保持原场景不变）\n\n【问题】...（新问题）"}}

只输出JSON。"""


def rewrite_query(question, client):
    """Rewrite the question part of an overlapping L2 query."""
    query = question.get("query", "")

    prompt = REWRITE_PROMPT.format(query=query)

    try:
        response = client.chat.completions.create(
            model=STRONG_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        text = response.choices[0].message.content

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

        if result and "query" in result:
            new_query = result["query"]
            new_overlap = overlap_ratio(new_query)
            if new_overlap < 0.25 and len(new_query) >= 150:
                question["query"] = new_query
                return True, new_overlap
        return False, overlap_ratio(query)
    except Exception as e:
        return False, overlap_ratio(query)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("层二: Fix L2 query scenario/question overlap")
    print("=" * 60)

    input_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v4.json")
    print(f"\nLoading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    l2 = [q for q in questions if q.get("level") == "L2"]
    print(f"  L2 questions: {len(l2)}")

    # Detect overlapping questions
    overlap_data = []
    for q in l2:
        ratio = overlap_ratio(q.get("query", ""))
        if ratio >= 0.25:
            overlap_data.append((q, ratio))

    print(f"  High/medium overlap (>=0.25): {len(overlap_data)}")
    if not overlap_data:
        print("  No overlapping questions found. Done.")
        return

    # Show distribution
    for q, r in sorted(overlap_data, key=lambda x: -x[1])[:5]:
        print(f"    {q.get('question_id')}: {r:.3f}")

    # Pre-fix stats
    all_ratios = [overlap_ratio(q.get("query", "")) for q in l2]
    print(f"\n  Pre-fix mean overlap: {sum(all_ratios)/len(all_ratios):.3f}")
    print(f"  Pre-fix median overlap: {sorted(all_ratios)[len(all_ratios)//2]:.3f}")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    fixed = 0
    failed = 0
    t0 = time.time()

    for i, (q, before_ratio) in enumerate(overlap_data):
        qid = q.get("question_id", "?")
        success, after_ratio = rewrite_query(q, client)

        if success:
            fixed += 1
        else:
            failed += 1

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(overlap_data)}] fixed={fixed} failed={failed} "
                  f"rate={(i+1)/elapsed*60:.0f}/min")

        time.sleep(0.3)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")
    print(f"  Fixed: {fixed}, Failed: {failed}")

    # Post-fix stats
    post_ratios = [overlap_ratio(q.get("query", "")) for q in l2]
    print(f"\n  Post-fix mean overlap: {sum(post_ratios)/len(post_ratios):.3f}")
    print(f"  Post-fix median overlap: {sorted(post_ratios)[len(post_ratios)//2]:.3f}")
    high_remaining = sum(1 for r in post_ratios if r >= 0.25)
    print(f"  Remaining high overlap (>=0.25): {high_remaining}/{len(l2)}")

    # Check query length
    short = sum(1 for q in l2 if len(q.get("query", "")) < 150)
    print(f"  Short queries (<150 chars): {short}/{len(l2)}")

    # Save
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"\n  Updated: {input_path}")


if __name__ == "__main__":
    main()
