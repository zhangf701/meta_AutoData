"""L1 QC Verification Tool.

Validates the v5 dataset's L1 questions against quality criteria.
Run independently to verify generation quality.

Usage:
  python test_l1_qc.py                           # Check v5 dataset
  python test_l1_qc.py --dataset path/to/file.json  # Check custom file
"""

import json
import os
import re
import sys
import argparse
from collections import Counter


# ── QC Check Patterns ────────────────────────────────────────────────────

VAGUE_PATTERN = re.compile(
    r'无具体|未明确|未给出|需参考其他|需查阅|标准中未|详见|参见'
)

CIRCULAR_PATTERN = re.compile(
    r'不超过规定(的)?(事故过负荷能力|过负荷能力)'
)

NON_ELECTRICAL_PATTERN = re.compile(
    r'防洪|防涝|土石方|飞机场|自然保护区|人文遗址'
    r'|消防|防火|环保|水土保持|绿化|噪声|噪音'
    r'|基础抗拔|抗倾覆|地基|桩基|混凝土|给排水|暖通'
)

NUMERICAL_PATTERN = re.compile(
    r'\d+(?:\.\d+)?\s*[%~kMGVWVAHhzΩ℃倍年月日台回套米秒]'
)


def check_answer_vague(answer):
    """Check if answer is vague placeholder text."""
    if VAGUE_PATTERN.search(answer):
        return True, VAGUE_PATTERN.search(answer).group(0)
    return False, ""


def check_answer_circular(answer):
    """Check if answer is circular (asking 'how much' and answering 'as specified')."""
    if CIRCULAR_PATTERN.search(answer):
        return True
    return False


def check_non_electrical(query):
    """Check if query contains non-electrical keywords."""
    match = NON_ELECTRICAL_PATTERN.search(query)
    if match:
        return True, match.group(0)
    return False, ""


def check_has_content(answer):
    """Check if answer has meaningful content (numerical or prescriptive judgment)."""
    has_numerical = bool(NUMERICAL_PATTERN.search(answer))
    prescriptive_kw = [
        '应', '必须', '宜', '可', '不应', '不得', '禁止', '不宜',
        '直接接地', '全部接地', '经', '采用', '配置', '设置',
        '接线', '母线', '保护', '接地', '变压器', '断路器',
        '为', '是', '指', '包括', '含',
    ]
    has_prescriptive = any(kw in answer for kw in prescriptive_kw)
    has_min_length = len(answer) >= 4
    ok = has_numerical or (has_prescriptive and has_min_length)
    return ok, has_numerical, has_prescriptive, len(answer)


def check_duplicates(questions):
    """Detect duplicate/similar questions by query prefix overlap."""
    seen = {}
    dups = []
    for q in questions:
        qid = q.get("question_id", "?")
        q_start = q.get("query", "")[:60]
        if q_start in seen:
            dups.append((seen[q_start], qid))
        else:
            seen[q_start] = qid
    return dups


def check_structure(question):
    """Check required fields exist and are properly typed."""
    issues = []
    required_fields = [
        "query", "expected_answer", "expected_keywords", "question_id",
        "question_class", "level", "source_standard", "grading_method",
        "knowledge_base", "rubric_clauses", "rubric_judgments",
    ]
    for field in required_fields:
        if field not in question:
            issues.append(f"缺少字段: {field}")
        elif question[field] is None:
            issues.append(f"字段为空(None): {field}")

    # Keywords check
    kws = question.get("expected_keywords", [])
    if not isinstance(kws, list):
        issues.append(f"expected_keywords不是list")
    elif len(kws) < 3:
        issues.append(f"关键词不足: {len(kws)}个(需≥3)")

    return issues


def run_qc(dataset_path):
    """Run comprehensive QC on L1 questions in dataset."""
    print("=" * 70)
    print(f"L1 QC Verification: {dataset_path}")
    print("=" * 70)

    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"[FATAL] Dataset not found: {dataset_path}")
        return False

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check encoding first
    try:
        test_str = json.dumps(data[:1] if data else [], ensure_ascii=False)
        if '�' in test_str:
            print("[FAIL] Encoding issue: Unicode replacement characters found!")
    except Exception:
        pass

    l1_questions = [q for q in data if q.get("level") == "L1"]
    print(f"\nTotal L1 questions: {len(l1_questions)}")

    # ── Run checks ──────────────────────────────────────────────────
    all_pass = True
    issues_found = []

    # 1. Vague answers
    print("\n--- Check 1: Vague Answers ---")
    vague_count = 0
    for q in l1_questions:
        is_vague, match = check_answer_vague(q.get("expected_answer", ""))
        if is_vague:
            vague_count += 1
            issue = f"  {q['question_id']}: vague answer (matched '{match}')"
            issues_found.append(issue)
            print(issue)
    if vague_count == 0:
        print("  [PASS] No vague answers found")
    else:
        print(f"  [FAIL] {vague_count} vague answers")
        all_pass = False

    # 2. Circular answers
    print("\n--- Check 2: Circular Answers ---")
    circular_count = 0
    for q in l1_questions:
        if check_answer_circular(q.get("expected_answer", "")):
            circular_count += 1
            issue = f"  {q['question_id']}: circular answer"
            issues_found.append(issue)
            print(issue)
    if circular_count == 0:
        print("  [PASS] No circular answers found")
    else:
        print(f"  [FAIL] {circular_count} circular answers")
        all_pass = False

    # 3. Non-electrical content
    print("\n--- Check 3: Electrical Relevance ---")
    nonelec_count = 0
    for q in l1_questions:
        is_nonelec, kw = check_non_electrical(q.get("query", ""))
        if is_nonelec:
            nonelec_count += 1
            issue = f"  {q['question_id']}: non-electrical keyword '{kw}'"
            issues_found.append(issue)
            print(issue)
    if nonelec_count == 0:
        print("  [PASS] All questions are electrical engineering related")
    else:
        print(f"  [FAIL] {nonelec_count} questions with non-electrical content")
        all_pass = False

    # 3.5 Query language quality
    print("\n--- Check 3.5: Query Language Quality ---")
    lang_fail_count = 0
    list_marker_start = re.compile(r'^[a-z]\)|^\d+\.\s|^[（(]\d+[）)]|^[①②③④⑤⑥⑦⑧⑨⑩]')
    for q in l1_questions:
        query = q.get("query", "")
        issues_local = []
        if list_marker_start.search(query.strip()):
            issues_local.append("starts with list marker")
        if len(query) < 10:
            issues_local.append(f"too short ({len(query)} chars)")
        if len(query) > 300:
            issues_local.append(f"too long ({len(query)} chars)")
        if issues_local:
            lang_fail_count += 1
            issue = f"  {q['question_id']}: {', '.join(issues_local)}"
            issues_found.append(issue)
            print(issue)
    if lang_fail_count == 0:
        print("  [PASS] All queries are well-formed natural language sentences")
    else:
        print(f"  [FAIL] {lang_fail_count} queries with language quality issues")
        all_pass = False

    # 4. Answer content quality
    print("\n--- Check 4: Answer Content Quality ---")
    content_fail_count = 0
    for q in l1_questions:
        ok, has_num, has_pres, length = check_has_content(q.get("expected_answer", ""))
        if not ok:
            content_fail_count += 1
            issue = (f"  {q['question_id']}: insufficient answer content "
                     f"(numerical={has_num}, prescriptive={has_pres}, len={length})")
            issues_found.append(issue)
            print(issue)
    if content_fail_count == 0:
        print("  [PASS] All answers have meaningful content")
    else:
        print(f"  [FAIL] {content_fail_count} answers with insufficient content")
        all_pass = False

    # 5. Duplicate detection
    print("\n--- Check 5: Duplicate Detection ---")
    dups = check_duplicates(l1_questions)
    if len(dups) == 0:
        print("  [PASS] No duplicate questions found")
    else:
        print(f"  [FAIL] {len(dups)} duplicate pairs found:")
        for d1, d2 in dups:
            print(f"    {d1} <-> {d2}")
        all_pass = False

    # 6. Structure validation
    print("\n--- Check 6: Structure Validation ---")
    struct_issues = 0
    for q in l1_questions:
        issues = check_structure(q)
        if issues:
            struct_issues += 1
            for iss in issues:
                print(f"  {q['question_id']}: {iss}")
    if struct_issues == 0:
        print("  [PASS] All questions have complete structure")
    else:
        print(f"  [FAIL] {struct_issues} questions with structural issues")
        all_pass = False

    # 7. Numerical ratio
    print("\n--- Check 7: Numerical vs Prescriptive Ratio ---")
    numerical_count = sum(
        1 for q in l1_questions
        if NUMERICAL_PATTERN.search(q.get("expected_answer", ""))
    )
    prescriptive_count = len(l1_questions) - numerical_count
    num_pct = 100 * numerical_count // max(len(l1_questions), 1)
    print(f"  Numerical: {numerical_count}/{len(l1_questions)} ({num_pct}%)")
    print(f"  Prescriptive: {prescriptive_count}/{len(l1_questions)} "
          f"({100 - num_pct}%)")
    if num_pct >= 40:
        print(f"  [PASS] Numerical ratio >= 40%")
    else:
        print(f"  [FAIL] Numerical ratio {num_pct}% < 40% target")
        all_pass = False

    # 8. Encoding check
    print("\n--- Check 8: UTF-8 Encoding ---")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            content = f.read()
        if '�' in content:
            print("  [FAIL] Unicode replacement characters found!")
            all_pass = False
        else:
            # Check for GBK corruption indicators
            gbk_indicators = ['�', '锛', '銆', '鈥']
            for indicator in gbk_indicators:
                if indicator in content:
                    print(f"  [FAIL] GBK corruption indicator '{indicator}' found!")
                    all_pass = False
                    break
            else:
                print("  [PASS] UTF-8 encoding is clean")
    except UnicodeDecodeError as e:
        print(f"  [FAIL] Not valid UTF-8: {e}")
        all_pass = False

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("QC SUMMARY")
    print("=" * 70)
    categories = Counter(q.get("topic", "?") for q in l1_questions)
    print(f"Topics: {dict(categories)}")
    print(f"Issues found: {len(issues_found)}")
    if all_pass:
        print("\n  *** ALL CHECKS PASSED ***")
    else:
        print("\n  *** SOME CHECKS FAILED - see details above ***")

    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L1 QC Verification Tool")
    parser.add_argument(
        "--dataset", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "questions", "generated_eval_set_v5.json"
        ),
        help="Path to dataset to verify"
    )
    args = parser.parse_args()

    success = run_qc(args.dataset)
    sys.exit(0 if success else 1)
