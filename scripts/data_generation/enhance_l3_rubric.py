"""enhance_l3_rubric.py: Enhance L3 rubric clause extraction.

Phase 4 Task 4.2: Process L3 answers' five-section structure to extract
clause references from each section, boosting average rubric_clauses from
~2.0 to >=3.0.

Input: generated_eval_set_v3_subcat.json
Output: generated_eval_set_v3_rubric_enhanced.json
"""

import json
import os
import re
import sys
import time

from openai import OpenAI

sys.path.insert(0, r"D:\coding\meta_AutoData\scripts")
from data_generation.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STRONG_MODEL, DATA_DIR,
)
from data_generation.utils.rubric_builder import (
    extract_clause_refs, extract_judgments, STD_PATTERN,
)

# ── Five-section structure extractor ────────────────────────────────────

SECTION_PATTERNS = [
    (r"现状诊断[：:]\s*(.+?)(?=多维冲突分析|$)", "现状诊断"),
    (r"多维冲突分析[：:]\s*(.+?)(?=多方案比选|$)", "多维冲突分析"),
    (r"多方案比选[：:]\s*(.+?)(?=折中综合方案|$)", "多方案比选"),
    (r"折中综合方案[：:]\s*(.+?)(?=控制优先级链条|$)", "折中综合方案"),
    (r"控制优先级链条[：:]\s*(.+?)$", "控制优先级链条"),
]

L3_PER_SECTION_PROMPT = """从以下电力工程L3综合评估答案的"{section_name}"部分中，提取所有引用的标准条款编号。

答案片段：
{section_text}

输出格式：纯JSON字符串数组，如 ["GB 38755-2019 第3.1.8条", "DL/T 5429-2009 第6.2.3条"]
如果没有条款引用，返回 []。
只输出JSON数组。"""


def extract_clauses_per_section(answer_text, client):
    """Extract clauses from each of the five L3 answer sections.

    For sections that don't have explicit clause references,
    use DeepSeek to infer which clauses are implicitly referenced.
    """
    all_clauses = []

    # First, try regex extraction on the full answer
    regex_clauses = extract_clause_refs(answer_text)

    # Parse sections
    sections_found = []
    for pattern, name in SECTION_PATTERNS:
        m = re.search(pattern, answer_text, re.DOTALL)
        if m:
            section_text = m.group(1).strip()
            if len(section_text) > 20:
                sections_found.append((name, section_text))

    if len(sections_found) < 3:
        # Not a proper five-section answer, fall back to regex
        all_clauses = regex_clauses
    else:
        all_clauses = list(regex_clauses)  # Start with regex results

        # For each section beyond the first 2 (which usually have regex hits),
        # ask DeepSeek for implicit clause references
        for name, text in sections_found[2:]:  # 多方案比选 onwards
            section_clauses = extract_clause_refs(text)
            if len(section_clauses) >= 1:
                all_clauses.extend(section_clauses)
            else:
                # DeepSeek inference for implicit references
                prompt = L3_PER_SECTION_PROMPT.format(
                    section_name=name,
                    section_text=text[:800],
                )
                try:
                    response = client.chat.completions.create(
                        model=STRONG_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=256,
                    )
                    resp_text = response.choices[0].message.content
                    arr = re.search(r"\[.*\]", resp_text, re.DOTALL)
                    if arr:
                        ds_clauses = json.loads(arr.group(0))
                        all_clauses.extend(ds_clauses)
                except Exception:
                    pass

    # Also do a full-answer DeepSeek extraction if we have < 3 clauses
    if len(set(all_clauses)) < 3:
        try:
            prompt = f"""从以下电力工程L3综合评估答案中，提取所有明确或隐含引用的标准条款编号。

答案文本：
{answer_text[:2500]}

要求：
1. 提取所有明确引用的条款（如"GB 38755-2019 第3.1.8条"）
2. 推断隐含引用的条款（如提到"短路比要求"则推断GB 38755-2019相关条款）
3. 合并重复引用

输出格式：纯JSON字符串数组，如 ["GB 38755-2019 第3.1.8条", "DL/T 5429-2009 第6.2.3条", ...]
只输出JSON数组。"""

            response = client.chat.completions.create(
                model=STRONG_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            resp_text = response.choices[0].message.content
            arr = re.search(r"\[.*\]", resp_text, re.DOTALL)
            if arr:
                ds_clauses = json.loads(arr.group(0))
                all_clauses.extend(ds_clauses)
        except Exception:
            pass

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in all_clauses:
        if c not in seen and len(c) > 5:
            seen.add(c)
            unique.append(c)

    return unique


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 4.2: Enhance L3 rubric clause coverage")
    print("=" * 60)

    input_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_subcat.json")
    print(f"\nLoading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    l3 = [q for q in questions if q.get("level") == "L3"]
    print(f"  L3 questions: {len(l3)}")

    # Current rubric stats
    current_clauses = [len(q.get("rubric_clauses", [])) for q in l3]
    print(f"  Current avg rubric_clauses: {sum(current_clauses)/len(current_clauses):.1f}")
    print(f"  Current >=3 clauses: {sum(1 for c in current_clauses if c >= 3)}/{len(l3)}")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    enhanced = 0
    t0 = time.time()

    for i, q in enumerate(l3):
        answer = q.get("expected_answer", "")
        clauses = extract_clauses_per_section(answer, client)

        if len(clauses) > len(q.get("rubric_clauses", [])):
            enhanced += 1

        q["rubric_clauses"] = clauses
        # Refresh judgments too
        q["rubric_judgments"] = extract_judgments(answer)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            avg = sum(len(q2.get("rubric_clauses", [])) for q2 in l3[:i+1]) / (i+1)
            print(f"  [{i+1}/{len(l3)}] enhanced={enhanced} avg_clauses={avg:.1f} "
                  f"rate={(i+1)/elapsed*60:.0f}/min")

        time.sleep(0.3)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")

    # Post stats
    new_clauses = [len(q.get("rubric_clauses", [])) for q in l3]
    print(f"  New avg rubric_clauses: {sum(new_clauses)/len(new_clauses):.1f}")
    print(f"  New >=3 clauses: {sum(1 for c in new_clauses if c >= 3)}/{len(l3)}")
    print(f"  Enhanced: {enhanced}/{len(l3)} questions")

    output_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_rubric_enhanced.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
