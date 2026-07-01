"""backfill_clause_source.py: Retroactively fill clause_source field for v3 dataset.

Phase 1 Task 1.1: Uses keyword matching + DeepSeek verification to trace each
question back to specific standard clauses.

Input: generated_eval_set_v3.json (300 questions, clause_source = 0%)
Output: generated_eval_set_v3_clause_backfill.json (clause_source fill rate >= 90%)
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
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STRONG_MODEL,
    BASE_DIR, DATA_DIR, STANDARDS_DIR,
)
from data_generation.source_loader import load_all_sources, detect_and_read


# ── Load standard texts ─────────────────────────────────────────────────

def load_standard_texts():
    """Load all three standard documents as flat text with section markers.

    Uses source_loader for encoding detection, then splits into
    paragraph-level chunks for keyword matching.
    """
    sources, all_sections = load_all_sources()

    # Build standard name → chunks mapping
    standard_chunks = {}
    for src in sources:
        title = src["title"]
        # Only process the 3 standards
        is_standard = any(kw in title for kw in ["GB 38755", "DLT 5429", "DLT 5218"])
        if not is_standard:
            continue

        chunks = []
        for sec in src["sections"]:
            text = sec["text"]
            if len(text) < 30:
                continue
            # Split long sections into paragraphs
            paragraphs = re.split(r"\n\s*\n", text)
            for para in paragraphs:
                para = para.strip()
                if len(para) >= 30:
                    chunks.append({
                        "text": para,
                        "section_title": sec.get("section_title", ""),
                        "source_file": title,
                    })

        # Build a short name key
        if "GB 38755" in title:
            key = "GB 38755-2019"
        elif "DLT 5429" in title or "DL/T 5429" in title:
            key = "DL/T 5429-2009"
        elif "DLT 5218" in title or "DL/T 5218" in title:
            key = "DL/T 5218-2012"
        else:
            key = title

        standard_chunks[key] = chunks
        print(f"  {key}: {len(chunks)} chunks from {len(src['sections'])} sections")

    return standard_chunks


# ── Keyword-based candidate retrieval ───────────────────────────────────

def find_candidate_chunks(question, standard_chunks, top_k=5):
    """Find the most relevant standard chunks for a question.

    Uses expected_keywords + source_standard as anchors.
    """
    keywords = question.get("expected_keywords", [])
    source_std = question.get("source_standard", "")
    query = question.get("query", "")

    # Determine which standard to search
    target_std = None
    for std_name in standard_chunks:
        if std_name in source_std or source_std in std_name:
            target_std = std_name
            break

    # If no match, search all standards
    if target_std:
        candidate_pool = [(target_std, c) for c in standard_chunks.get(target_std, [])]
    else:
        candidate_pool = []
        for std_name, chunks in standard_chunks.items():
            for c in chunks:
                candidate_pool.append((std_name, c))

    if not candidate_pool:
        return []

    # Score each chunk by keyword hits
    scored = []
    query_chars = set(query + " ".join(keywords))
    for std_name, chunk in candidate_pool:
        text = chunk["text"]
        score = 0
        for kw in keywords:
            if kw in text:
                score += 3
        # Bonus for character overlap with query
        text_chars = set(text)
        overlap = len(query_chars & text_chars)
        score += overlap / max(len(query_chars), 1) * 0.5
        if score > 0:
            scored.append((score, std_name, chunk))

    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


# ── DeepSeek clause pinpointing ─────────────────────────────────────────

CLAUSE_PROMPT = """你是一名电力系统标准审核专家。给定一道评测题目和标准原文段落，请找出该题目所引用的精确标准条款编号。

## 题目
- Query: {query}
- Answer: {answer}
- 标注的标准: {source_standard}

## 标准原文段落
{chunks_text}

## 要求
1. 找出题目答案中引用的精确条款编号（如"GB 38755-2019 第3.1.8条"或"DL/T 5429-2009 第6.2.3条"）
2. 如果答案引用了多个条款，用逗号分隔
3. 如果从原文段落中可以明确确定条款编号，即使答案中只写了"根据标准"也要标出
4. 如果完全无法确定，返回空字符串

## 输出格式
只输出条款编号字符串，不要任何解释。例如：
GB 38755-2019 第3.1.8条, DL/T 5429-2009 第6.2.3条"""


def pinpoint_clause(question, candidates, client):
    """Use DeepSeek to pinpoint the exact clause reference."""
    if not candidates:
        return ""

    # Build context from top candidates
    chunks_text = ""
    for score, std_name, chunk in candidates:
        chunks_text += f"\n[{std_name}] {chunk['section_title']}\n{chunk['text'][:500]}\n"

    prompt = CLAUSE_PROMPT.format(
        query=question.get("query", "")[:400],
        answer=question.get("expected_answer", "")[:600],
        source_standard=question.get("source_standard", ""),
        chunks_text=chunks_text[:3000],
    )

    try:
        response = client.chat.completions.create(
            model=STRONG_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        result = response.choices[0].message.content.strip()
        # Clean up
        if result in ("", "无", "N/A", "无法确定"):
            return ""
        return result
    except Exception as e:
        print(f"    [DeepSeek Error] {e}")
        return ""


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 1.1: Backfill clause_source for v3 dataset")
    print("=" * 60)

    # Load v3 dataset
    v3_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3.json")
    print(f"\nLoading dataset: {v3_path}")
    with open(v3_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  {len(questions)} questions loaded")

    # Check current state
    empty_count = sum(1 for q in questions if not q.get("clause_source"))
    print(f"  clause_source empty: {empty_count}/{len(questions)}")

    # Load standard texts
    print("\nLoading standard texts...")
    standard_chunks = load_standard_texts()
    print(f"  Standards loaded: {list(standard_chunks.keys())}")

    # Initialize DeepSeek client
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # Process each question
    print("\nProcessing questions...")
    filled = 0
    skipped = 0
    errors = 0
    stats_by_level = Counter()

    t0 = time.time()
    for i, q in enumerate(questions):
        # Skip if already has clause_source
        if q.get("clause_source") and q["clause_source"].strip():
            skipped += 1
            continue

        level = q.get("level", "L1")

        # Find candidate chunks
        candidates = find_candidate_chunks(q, standard_chunks, top_k=5)

        if not candidates:
            # No matching chunks found — try with broader search
            errors += 1
            q["clause_source"] = ""
            if errors <= 5:
                print(f"  [{i+1}/{len(questions)}] {q.get('question_id','?')} ({level}): no candidates found")
            continue

        # Pinpoint clause via DeepSeek
        clause_ref = pinpoint_clause(q, candidates, client)

        if clause_ref:
            q["clause_source"] = clause_ref
            filled += 1
            stats_by_level[level] += 1
        else:
            q["clause_source"] = ""
            errors += 1

        # Progress
        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            print(f"  [{i+1}/{len(questions)}] filled={filled}, errors={errors}, "
                  f"rate={rate:.0f}/min, elapsed={elapsed:.0f}s")

        # Rate limiting
        time.sleep(0.3)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Filled: {filled}, Skipped (already had): {skipped}, Errors: {errors}")
    print(f"  Fill rate: {filled}/{len(questions)} ({filled/len(questions)*100:.1f}%)")
    print(f"  By level: {dict(stats_by_level)}")

    # Save output
    output_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_clause_backfill.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"\n  Output saved: {output_path}")

    # Quick verification
    still_empty = sum(1 for q in questions if not q.get("clause_source"))
    print(f"\n  Verification: {still_empty}/{len(questions)} still empty ({still_empty/len(questions)*100:.1f}%)")

    # Show some samples
    print("\n  Sample results:")
    for q in questions[:5]:
        print(f"    [{q.get('question_id','?')}] {q.get('level','')}: "
              f"clause_source='{q.get('clause_source','')[:100]}'")

    for q in questions[100:105]:
        print(f"    [{q.get('question_id','?')}] {q.get('level','')}: "
              f"clause_source='{q.get('clause_source','')[:100]}'")

    for q in questions[200:205]:
        print(f"    [{q.get('question_id','?')}] {q.get('level','')}: "
              f"clause_source='{q.get('clause_source','')[:100]}'")

    return questions


if __name__ == "__main__":
    main()
