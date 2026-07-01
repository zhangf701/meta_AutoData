"""fix_l1_parametric.py: Rewrite non-parametric L1 questions to strict parameter lookup.

Phase 1 Task 1.2: Ensures L1 answers contain specific numerical values/percentages/thresholds,
not definitions or narrative descriptions.

Input: generated_eval_set_v3_clause_backfill.json
Output: generated_eval_set_v3_l1_fixed.json
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
from data_generation.utils.rubric_builder import build_rubric


# ── L1 quality classifier ───────────────────────────────────────────────

UNIT_PATTERN = re.compile(
    r'\d+\s*(?:kV|MW|MVA|kA|Hz|km|mm|m|Ω|℃|年|月|日|台|回|套|次|倍|%|kV|V|A|W)'
)
NUM_PATTERN = re.compile(r'\d+(?:\.\d+)?')

# Vague/garbage answers that should be rewritten entirely
GARBAGE_PATTERNS = [
    r'^应符合.*要求$', r'^应按.*执行$', r'^参见.*$',
    r'^应满足.*规定$', r'^需符合.*标准$', r'^根据.*确定$',
    r'^依据.*执行$',
]


def classify_l1(question):
    """Classify an L1 question's quality.

    Returns:
        'good': Answer has numeric parameter with unit, keep as-is
        'definition': Descriptive answer, needs rewrite to parametric
        'short_ok': Very short but specific value (like "220kV"), enrich
        'short_garbage': Very short and vague, full rewrite
    """
    ans = question.get("expected_answer", "")

    has_unit = bool(UNIT_PATTERN.search(ans))
    has_num = bool(NUM_PATTERN.search(ans))
    is_short = len(ans) < 10

    if has_unit and not is_short:
        return "good"

    if is_short:
        for pat in GARBAGE_PATTERNS:
            if re.match(pat, ans):
                return "short_garbage"
        if has_unit or has_num:
            return "short_ok"  # "220kV", "40%", "3" — specific but brief
        return "short_garbage"

    # Not short, no unit → definition type
    return "definition"


# ── L1 rewrite prompt ───────────────────────────────────────────────────

L1_REWRITE_PROMPT = """你是一名电力系统标准审核专家。现有以下L1级别（直接参数检索）评测题目，但其答案不符合L1定义。

## L1 定义
答案必须是在标准规范文本中可直接检索到的**具体数值参数**。例如：
- "40%"、"1200MW"、"15kV"、"不超过10%"、"≥2.0"
- 不是定义、不是叙述、不是原则描述

## 原始题目
- Query: {query}
- 原Answer: {answer}
- 相关标准: {source_standard}
- 条款来源: {clause_source}

## 标准原文参考
{clause_text}

## 任务
1. 从标准原文中找出一个该主题下的**具体数值参数**（比例、电压、容量、时间、距离等）
2. 改写query，使其成为对该参数的精确提问（问"多少"、"什么值"、"多大"而非"什么是"）
3. 改写answer，使其只包含数值+单位（可附带简短条件说明，但不超过30字）
4. 更新keywords，确保包含数值和参数名

## 输出格式
纯JSON对象：
{{"query":"改写后的参数检索问题","expected_answer":"数值参数答案（含单位）","expected_keywords":["关键词1","关键词2","关键词3","关键词4","关键词5"],"category":"参数检索"}}

只输出JSON，不要任何解释。"""


def rewrite_l1(question, clause_text, client):
    """Rewrite a non-conforming L1 question to strict parametric format.

    Args:
        question: Original L1 question dict
        clause_text: Relevant standard clause text for context
        client: OpenAI client

    Returns:
        Updated question dict (in-place modified), or None on failure
    """
    prompt = L1_REWRITE_PROMPT.format(
        query=question.get("query", ""),
        answer=question.get("expected_answer", ""),
        source_standard=question.get("source_standard", ""),
        clause_source=question.get("clause_source", ""),
        clause_text=clause_text[:2000],
    )

    try:
        response = client.chat.completions.create(
            model=STRONG_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
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

        if not result:
            return None

        # Update question fields
        question["query"] = result.get("query", question["query"])
        question["expected_answer"] = result.get("expected_answer", question["expected_answer"])
        question["expected_keywords"] = result.get("expected_keywords", question["expected_keywords"])[:5]
        question["category"] = "参数检索"
        return question

    except Exception as e:
        print(f"    [Rewrite Error] {e}")
        return None


# ── Clause text retrieval ───────────────────────────────────────────────

def get_clause_text(question, standard_chunks):
    """Retrieve the relevant standard text for a question's clause_source."""
    clause_src = question.get("clause_source", "")
    source_std = question.get("source_standard", "")

    # Determine standard
    target_std = None
    for std_name in standard_chunks:
        if std_name in source_std or std_name in clause_src or source_std in std_name:
            target_std = std_name
            break

    if not target_std:
        # Return first available chunk
        for chunks in standard_chunks.values():
            if chunks:
                return chunks[0]["text"][:2000]
        return ""

    chunks = standard_chunks.get(target_std, [])
    if not chunks:
        return ""

    # Find the most relevant chunk by keyword overlap
    keywords = question.get("expected_keywords", [])
    best = chunks[0]
    best_score = 0
    for ch in chunks:
        score = sum(1 for kw in keywords if kw in ch["text"])
        if score > best_score:
            best_score = score
            best = ch

    return best["text"][:2000]


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 1.2: Fix L1 parametric quality")
    print("=" * 60)

    # Load v3 dataset with clause_source backfilled
    input_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_clause_backfill.json")
    print(f"\nLoading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    l1_questions = [q for q in questions if q.get("level") == "L1"]
    print(f"  L1 questions: {len(l1_questions)}")

    # Classify
    stats = Counter()
    to_fix = []
    for q in l1_questions:
        cls = classify_l1(q)
        stats[cls] += 1
        if cls != "good":
            to_fix.append(q)

    print(f"\n  Classification:")
    print(f"    good (keep):          {stats.get('good', 0)}")
    print(f"    definition (rewrite): {stats.get('definition', 0)}")
    print(f"    short_ok (enrich):    {stats.get('short_ok', 0)}")
    print(f"    short_garbage (fix):  {stats.get('short_garbage', 0)}")
    print(f"    Total to fix:         {len(to_fix)}")

    # Load standards for clause context
    print("\n  Loading standard texts for context...")
    from data_generation.source_loader import load_all_sources
    sources, all_sections = load_all_sources()

    # Build standard chunks
    standard_chunks = {}
    for src in sources:
        title = src["title"]
        is_standard = any(kw in title for kw in ["GB 38755", "DLT 5429", "DLT 5218"])
        if not is_standard:
            continue
        chunks = []
        for sec in src["sections"]:
            text = sec["text"]
            if len(text) < 30:
                continue
            for para in re.split(r"\n\s*\n", text):
                para = para.strip()
                if len(para) >= 30:
                    chunks.append({"text": para})
        if "GB 38755" in title:
            key = "GB 38755-2019"
        elif "DLT 5429" in title or "DL/T 5429" in title:
            key = "DL/T 5429-2009"
        elif "DLT 5218" in title or "DL/T 5218" in title:
            key = "DL/T 5218-2012"
        else:
            key = title
        standard_chunks[key] = chunks
        print(f"    {key}: {len(chunks)} chunks")

    # Initialize DeepSeek client
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # Fix each broken L1
    print(f"\n  Fixing {len(to_fix)} L1 questions...")
    fixed = 0
    failed = 0
    t0 = time.time()

    for i, q in enumerate(to_fix):
        qid = q.get("question_id", "?")
        cls = classify_l1(q)
        clause_text = get_clause_text(q, standard_chunks)

        result = rewrite_l1(q, clause_text, client)
        if result:
            fixed += 1
        else:
            failed += 1
            if failed <= 5:
                print(f"    [{i+1}/{len(to_fix)}] {qid} ({cls}): REWRITE FAILED")

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            print(f"    [{i+1}/{len(to_fix)}] fixed={fixed} failed={failed} "
                  f"rate={rate:.0f}/min")

        time.sleep(0.3)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")
    print(f"  Fixed: {fixed}, Failed: {failed}")

    # Build rubric for ALL L1 questions (including the good ones that lacked it)
    print(f"\n  Building rubric for all L1 questions...")
    rubric_count = 0
    for q in l1_questions:
        if not q.get("rubric_clauses"):
            rubric = build_rubric(q.get("expected_answer", ""), "L1")
            q["rubric_clauses"] = rubric["rubric_clauses"]
            q["rubric_judgments"] = rubric["rubric_judgments"]
            rubric_count += 1
    print(f"    Rubric built for {rubric_count} L1 questions")

    # Re-verify quality
    print(f"\n  Post-fix verification:")
    has_unit = sum(1 for q in l1_questions if UNIT_PATTERN.search(q.get("expected_answer", "")))
    has_num = sum(1 for q in l1_questions if NUM_PATTERN.search(q.get("expected_answer", "")))
    has_rubric = sum(1 for q in l1_questions if q.get("rubric_clauses"))
    print(f"    Has number+unit: {has_unit}/100")
    print(f"    Has any number:  {has_num}/100")
    print(f"    Has rubric:      {has_rubric}/100")

    # Show some fixed examples
    print(f"\n  Sample fixed L1 (first 5 that were rewritten):")
    count = 0
    for q in l1_questions:
        if classify_l1(q) == "good" and count < 5:
            count += 1
            if count <= 5:
                print(f"    [{q.get('question_id')}] Q: {q['query'][:100]}")
                print(f"      A: {q['expected_answer'][:80]}")
                print(f"      KW: {q.get('expected_keywords',[])}")

    # Save output
    output_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_l1_fixed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"\n  Output saved: {output_path}")

    return questions


if __name__ == "__main__":
    main()
