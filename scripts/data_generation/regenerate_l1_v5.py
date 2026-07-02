"""Regenerate L1 questions for v5 dataset.

Improvements over v4:
1. Expanded clause extraction: numerical + pure-text prescriptive clauses
2. Non-electrical content filtering
3. QC verification on every generated L1 question
4. UTF-8 encoding throughout

Output:
  clauses_v5.json  — filtered electrical-compliance L1 clause pool
  generated_eval_set_v5.json — new L1 + v4's L2/L3

Usage:
  python regenerate_l1_v5.py                # Full pipeline
  python regenerate_l1_v5.py --clauses-only  # Only build clause pool
  python regenerate_l1_v5.py --generate-only  # Only generate L1 (requires clauses_v5.json)
"""

import json
import os
import re
import sys
import argparse

# Ensure project path — need scripts/ dir for 'data_generation' package import
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)  # scripts/ contains data_generation/ package

from data_generation.config import (
    SCRIPTS_DIR, QUESTIONS_OUTPUT, OUTPUT_DIR,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STRONG_MODEL,
)
from data_generation.clause_extractor import classify_source
from data_generation.generators.l1_generator import L1Generator
from data_generation.utils.id_manager import IDManager
from data_generation.utils.format_validator import validate_question
from data_generation.utils.rubric_builder import build_rubric

# ── Paths ───────────────────────────────────────────────────────────────
V4_DATASET = os.path.join(OUTPUT_DIR, "generated_eval_set_v4.json")
V5_DATASET = os.path.join(OUTPUT_DIR, "generated_eval_set_v5.json")
CLAUSES_V1 = os.path.join(SCRIPTS_DIR, "clauses_v1.json")
CLAUSES_V5 = os.path.join(SCRIPTS_DIR, "clauses_v5.json")

# ── Non-electrical filter patterns ──────────────────────────────────────
NON_ELECTRICAL_RE = re.compile(
    r'防洪|防涝|土石方|飞机场|自然保护区|人文遗址'
    r'|生产和生活用水|消防|环保|水土保持|绿化|噪声|噪音'
    r'|基础抗拔|抗倾覆|地基|桩基|混凝土|给排水|暖通'
)


def is_electrical_clause(clause):
    """Return True if clause is electrical engineering related."""
    text = clause.get("clause_text", "")
    section = clause.get("section", "")
    topic = clause.get("topic", "")
    combined = f"{text} {section} {topic}"
    return not NON_ELECTRICAL_RE.search(combined)


def deduplicate_clauses(clauses):
    """Remove near-duplicate clauses by character overlap ratio."""
    unique, seen_texts = [], []
    for c in clauses:
        txt = c.get("clause_text", "")
        ratio = max(
            (len(set(txt) & set(s)) / max(len(set(txt)), 1))
            for s in seen_texts
        ) if seen_texts else 0
        if ratio < 0.7:
            unique.append(c)
            seen_texts.append(txt)
    return unique


# ── Step 1: Build clause pool ───────────────────────────────────────────

def load_standard_texts():
    """Load RAW text from standard files (bypass desensitization).

    Uses source_loader.detect_and_read() for encoding detection,
    but skips clean_text()/desensitize() which corrupts technical terms.
    """
    from data_generation.source_loader import detect_and_read
    from data_generation.config import SOURCE_FILES

    texts = {}
    for filepath in SOURCE_FILES:
        if not os.path.exists(filepath):
            continue
        title = os.path.splitext(os.path.basename(filepath))[0].replace("+", " ")
        src_type = classify_source(title)
        if src_type != "standard":
            continue
        raw_text = detect_and_read(filepath)
        # Only basic cleanup: normalize whitespace, no desensitization
        raw_text = re.sub(r'\r\n', '\n', raw_text)
        raw_text = re.sub(r'\n{4,}', '\n\n\n', raw_text)
        raw_text = re.sub(r' {3,}', '  ', raw_text)
        texts[title] = raw_text
    return texts


def extract_l1_clauses_deepseek(standard_texts, target_numerical=80, target_prescriptive=40):
    """Call DeepSeek API to extract L1 compliance clauses with improved prompt.

    Runs two passes:
    1. Numerical clauses (target 80+)
    2. Prescriptive clauses (target 40+)
    This ensures the 60:40 numerical:prescriptive ratio.
    """
    if not DEEPSEEK_API_KEY:
        print("  [WARN] DEEPSEEK_API_KEY not set, skipping DeepSeek extraction")
        return []

    combined = "\n\n".join(
        f"=== {title} ===\n{text}"
        for title, text in standard_texts.items()
    )

    all_clauses = []

    # ── Pass 1: Numerical clauses ──
    prompt_num = f"""从以下电力系统标准中提取所有包含具体数值参数的规范性条款。

包括：比例、电压、电流、容量、时间、距离、温度、电阻、频率等具体数值要求。

排除以下非电气内容：
- 土建结构：防洪、防涝、土石方、基础抗拔、抗倾覆、地基、桩基、混凝土
- 给排水、消防（防火隔墙、灭火装置）、暖通空调
- 环保：水土保持、噪声、绿化
- 站址与机场/自然保护区的距离

提取格式（纯JSON数组）：
[{{"standard":"标准编号","section":"条款号","clause_text":"完整条款原文（含数值参数）","topic":"主题分类","clause_type":"numerical"}}]

要求：尽可能多地提取数值型条款，目标≥{target_numerical}条。每条条款的clause_text必须包含具体数值。

标准文本：
{combined[:12000]}

直接返回JSON数组："""

    numerical_clauses = _call_deepseek_and_parse(prompt_num, "numerical")
    if numerical_clauses:
        all_clauses.extend(numerical_clauses)
        print(f"    Pass 1 (numerical): {len(numerical_clauses)} clauses")

    # ── Pass 2: Prescriptive clauses ──
    prompt_pres = f"""从以下电力系统标准中提取纯文字规定性条款（不含数值但属于电气专业的强制性/规定性要求）。

例如：
- 设备配置要求（如"应装设两台站用变压器"）
- 接线方式规定（如"宜采用双母线分段接线"）
- 中性点接地方式（如"变压器中性点宜全部接地"）
- 保护配置原则（如"应配置双重化保护"）
- 电压等级适用范围（如"适用于220kV及以上电力系统"）

排除：土建、消防、环保、防洪、给排水、暖通等非电气内容。

提取格式（纯JSON数组）：
[{{"standard":"标准编号","section":"条款号","clause_text":"完整条款原文","topic":"主题分类","clause_type":"prescriptive"}}]

要求：提取最重要的规定性条款，目标≥{target_prescriptive}条。

标准文本：
{combined[:10000]}

直接返回JSON数组："""

    prescriptive_clauses = _call_deepseek_and_parse(prompt_pres, "prescriptive")
    if prescriptive_clauses:
        all_clauses.extend(prescriptive_clauses)
        print(f"    Pass 2 (prescriptive): {len(prescriptive_clauses)} clauses")

    return all_clauses


def _call_deepseek_and_parse(prompt, clause_type_label):
    """Call DeepSeek and parse the JSON array response."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=STRONG_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8192,
        )
        response_text = resp.choices[0].message.content

        # Clean markdown wrapping
        cleaned = response_text.strip()
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
        cleaned = re.sub(r'\n?\s*```\s*$', '', cleaned)
        cleaned = cleaned.strip()

        # Strategy 1: Direct JSON parse
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and len(parsed) > 0:
                return _build_deepseek_clauses(parsed)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Fix common JSON issues (trailing commas, truncation)
        fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
        if not fixed.rstrip().endswith(']'):
            fixed = re.sub(r',\s*$', '', fixed.rstrip())
            if not fixed.rstrip().endswith(']'):
                fixed = fixed.rstrip() + '\n]'
        try:
            parsed = json.loads(fixed)
            if isinstance(parsed, list) and len(parsed) > 0:
                return _build_deepseek_clauses(parsed)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Regex-based object extraction
        clauses = []
        obj_pattern = re.compile(
            r'\{\s*"standard":\s*"([^"]*)",\s*"section":\s*"([^"]*)",'
            r'\s*"clause_text":\s*"((?:[^"\\]|\\.)*)",\s*"topic":\s*"([^"]*)",'
            r'\s*"clause_type":\s*"([^"]*)"\s*\}'
        )
        for m in obj_pattern.finditer(cleaned):
            clauses.append({
                "standard": m.group(1), "section": m.group(2),
                "clause_text": m.group(3), "topic": m.group(4),
                "clause_type": m.group(5),
            })
        if clauses:
            return _build_deepseek_clauses(clauses)

        print(f"    [WARN] Could not parse {clause_type_label} response. "
              f"First 200 chars: {repr(response_text[:200])}")
        return []
    except Exception as e:
        print(f"    [ERR] DeepSeek {clause_type_label} extraction failed: {e}")
        return []


def _build_deepseek_clauses(raw_facts):
    """Convert raw DeepSeek facts to standard clause dicts with quality filtering."""
    clauses = []
    for f in raw_facts:
        clause_text = f.get("clause_text", "")
        # Quality filter: meaningful length
        if len(clause_text) < 12 or len(clause_text) > 500:
            continue
        # Quality filter: must have content beyond just the section number
        if re.match(r'^[\d.\s]+$', clause_text):
            continue
        clause = {
            "standard": f.get("standard", "未知"),
            "section": f.get("section", ""),
            "clause_text": clause_text,
            "difficulty": "L1",
            "topic": f.get("topic", "通用要求"),
            "has_numerical_threshold": f.get("clause_type") == "numerical",
            "has_conditional_logic": False,
            "references_other_standards": False,
            "key_terms": [f.get("parameter", f.get("topic", ""))],
            "source_file": f.get("standard", ""),
            "source_type": "standard",
            "extraction_method": "deepseek",
            "clause_type": f.get("clause_type", "prescriptive"),
        }
        clauses.append(clause)
    return clauses


def load_clauses_v1():
    """Load existing clauses_v1.json as fallback."""
    if not os.path.exists(CLAUSES_V1):
        print(f"  [WARN] {CLAUSES_V1} not found")
        return []
    with open(CLAUSES_V1, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("clauses", [])
    return data


def build_clause_pool():
    """Build filtered electrical-only L1 clause pool.

    Strategy:
    1. Try DeepSeek extraction with improved prompt (if API key available)
    2. Merge with existing clauses_v1.json
    3. Filter non-electrical + dedup
    4. Save as clauses_v5.json
    """
    print("=" * 60)
    print("Step 1: Building electrical compliance L1 clause pool")
    print("=" * 60)

    all_clauses = []

    # Source 1: Load existing clauses_v1.json
    v1_clauses = load_clauses_v1()
    if v1_clauses:
        l1_v1 = [c for c in v1_clauses if c.get("difficulty") == "L1"]
        print(f"  Loaded {len(l1_v1)} L1 clauses from clauses_v1.json")
        all_clauses.extend(l1_v1)

    # Source 2: DeepSeek extraction with improved prompt
    standard_texts = load_standard_texts()
    if standard_texts:
        print(f"  Loaded {len(standard_texts)} standard texts for DeepSeek extraction")
        ds_clauses = extract_l1_clauses_deepseek(standard_texts)
        if ds_clauses:
            print(f"  DeepSeek extracted {len(ds_clauses)} clauses")
            all_clauses.extend(ds_clauses)

    if not all_clauses:
        print("[FATAL] No clauses available. Cannot proceed.")
        return None

    # Filter: electrical only
    before_filter = len(all_clauses)
    all_clauses = [c for c in all_clauses if is_electrical_clause(c)]
    filtered_out = before_filter - len(all_clauses)
    print(f"  Non-electrical filtered: {filtered_out} removed, {len(all_clauses)} remain")

    # Dedup
    all_clauses = deduplicate_clauses(all_clauses)
    print(f"  After dedup: {len(all_clauses)} unique L1 clauses")

    # Assign IDs
    for i, c in enumerate(all_clauses):
        c["clause_id"] = f"CL5-{i+1:04d}"
        if "difficulty" not in c:
            c["difficulty"] = "L1"
        if "clause_type" not in c:
            # Auto-detect clause type
            has_num = bool(re.search(
                r'\d+(?:\.\d+)?\s*[%~kMGVWVAHhzΩ℃倍年月日台回套米秒]',
                c.get("clause_text", "")
            ))
            c["clause_type"] = "numerical" if has_num else "prescriptive"

    # Save clauses_v5.json
    numerical_count = sum(1 for c in all_clauses if c.get("clause_type") == "numerical")
    prescriptive_count = len(all_clauses) - numerical_count

    output = {
        "total": len(all_clauses),
        "numerical": numerical_count,
        "prescriptive": prescriptive_count,
        "description": "Electrical compliance L1 clauses for v5 dataset",
        "clauses": all_clauses,
    }
    with open(CLAUSES_V5, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(all_clauses)} clauses ({numerical_count} numerical, "
          f"{prescriptive_count} prescriptive) to clauses_v5.json")

    return all_clauses


# ── Step 2: Generate L1 questions ────────────────────────────────────────

def generate_l1_questions(clauses, target_count=100):
    """Generate L1 questions with QC verification."""
    print(f"\n{'=' * 60}")
    print(f"Step 2: Generating L1 questions (target: {target_count})")
    print(f"{'=' * 60}")

    if not clauses:
        print("[FATAL] No clauses provided for L1 generation")
        return []

    id_manager = IDManager(l1_start=1)
    generator = L1Generator(clauses, id_manager, min_numerical_ratio=0.6)
    questions = generator.generate(target_count=target_count)

    return questions


# ── Step 3: Assemble v5 dataset ──────────────────────────────────────────

def assemble_v5(l1_questions):
    """Combine new L1 with v4's L2/L3, renumber IDs, build rubrics, filter non-electrical."""
    print(f"\n{'=' * 60}")
    print("Step 3: Building rubrics, filtering, and assembling v5 dataset")
    print(f"{'=' * 60}")

    # ── L1: renumber sequentially ──
    for i, q in enumerate(l1_questions):
        q["question_id"] = f"L1-{i+1:03d}"

    # ── L1: build rubrics from clause metadata (not answer text parsing) ──
    for q in l1_questions:
        answer = q.get("expected_answer", "")
        standard = q.get("source_standard", "")
        clause_id = q.get("clause_source", "")

        # rubric_clauses: extract ONLY standard number + section number
        # Strip descriptive text after section numbers
        # Input:  "GB 38755-2019 4.2.3 第二级安全稳定标准"
        # Output: "GB 38755-2019 §4.2.3"
        std_ref = standard.strip() if standard else ""
        # Match standard prefix + number + section numbers, discard trailing text
        std_match = re.match(
            r'((?:GB|DL/T|Q/GDW|IEEE|IEC|SDJ|NB)\s*[0-9]{2,6}(?:[-–—][0-9]{2,4})?)'
            r'\s+([0-9]+(?:\.[0-9]+){0,3})',
            std_ref
        )
        if std_match:
            std_prefix = std_match.group(1)
            section = std_match.group(2)
            q["rubric_clauses"] = [f"{std_prefix} §{section}"]
        elif std_ref:
            q["rubric_clauses"] = [std_ref]
        else:
            q["rubric_clauses"] = []

        # rubric_judgments: format like v4: "答案应为: δ < 90°"
        q["rubric_judgments"] = [f"答案应为: {answer}"]
        q["rubric_refusal_expected"] = False
        q["scoring_version"] = "v5-rubric"

        if "topic" not in q:
            q["topic"] = "通用要求"
        if "sub_category" not in q:
            ans = q.get("expected_answer", "")
            if re.search(r'\d+', ans):
                q["sub_category"] = "数值阈值与限值"
            else:
                q["sub_category"] = "规定性要求"

    # ── Load v4 dataset ──
    if not os.path.exists(V4_DATASET):
        print(f"[FATAL] v4 dataset not found: {V4_DATASET}")
        return

    with open(V4_DATASET, "r", encoding="utf-8") as f:
        v4_data = json.load(f)

    v4_l2 = [q for q in v4_data if q.get("level") == "L2"]
    v4_l3 = [q for q in v4_data if q.get("level") == "L3"]

    # ── L2/L3: filter non-electrical ──
    filter_file = os.path.join(SCRIPTS_DIR, "l2l3_filter_results.json")
    non_elec_ids = set()
    if os.path.exists(filter_file):
        with open(filter_file, "r", encoding="utf-8") as f:
            filter_data = json.load(f)
        non_elec_ids = set(filter_data.get("non_electrical_ids", []))
        print(f"  Loaded {len(non_elec_ids)} non-electrical IDs from filter results")
    else:
        print(f"  [WARN] Filter results not found, using keyword-based fallback")
        non_elec_re = re.compile(
            r'防洪|防涝|土石方|桩基|混凝土|给排水|暖通|消防|防火|'
            r'环保|水土保持|绿化|噪声|噪音|碳排放|自然保护区|人文遗址'
        )
        for q in v4_l2 + v4_l3:
            if non_elec_re.search(q.get("query", "") + q.get("expected_answer", "")):
                non_elec_ids.add(q.get("question_id", ""))

    v4_l2_filtered = [q for q in v4_l2 if q.get("question_id") not in non_elec_ids]
    v4_l3_filtered = [q for q in v4_l3 if q.get("question_id") not in non_elec_ids]

    removed_l2 = len(v4_l2) - len(v4_l2_filtered)
    removed_l3 = len(v4_l3) - len(v4_l3_filtered)
    print(f"  L2: {removed_l2} removed ({len(v4_l2_filtered)} remain)")
    print(f"  L3: {removed_l3} removed ({len(v4_l3_filtered)} remain)")

    # ── Supplement if below 100 ──
    target_per_level = 100
    for level_name, filtered, original in [
        ("L2", v4_l2_filtered, v4_l2),
        ("L3", v4_l3_filtered, v4_l3),
    ]:
        if len(filtered) < target_per_level:
            shortfall = target_per_level - len(filtered)
            # Get remaining questions sorted by strong_score descending
            existing_ids = {q.get("question_id") for q in filtered}
            candidates = [
                q for q in original
                if q.get("question_id") not in existing_ids
                and q.get("question_id") not in non_elec_ids
            ]
            candidates.sort(
                key=lambda q: q.get("strong_score", q.get("loopjudge_verdict", 0)),
                reverse=True,
            )
            supplement = candidates[:shortfall]
            filtered.extend(supplement)
            print(f"  {level_name}: supplemented {len(supplement)} questions to reach {len(filtered)}")

    # Renumber L2/L3 sequentially
    for i, q in enumerate(v4_l2_filtered):
        q["question_id"] = f"L2-{i+1:03d}"
    for i, q in enumerate(v4_l3_filtered):
        q["question_id"] = f"L3-{i+1:03d}"

    print(f"  Final: L1:{len(l1_questions)} + L2:{len(v4_l2_filtered)} + L3:{len(v4_l3_filtered)}")

    # ── Build v5 ──
    v5_data = l1_questions + v4_l2_filtered + v4_l3_filtered

    for q in v5_data:
        q["question_class"] = q.get("level", "L1")

    with open(V5_DATASET, "w", encoding="utf-8") as f:
        json.dump(v5_data, f, ensure_ascii=False, indent=2)

    print(f"  v5 dataset saved: {V5_DATASET}")
    print(f"  Total: {len(v5_data)} questions")

    # Validation
    levels = {}
    for q in v5_data:
        lvl = q.get("level", "?")
        levels[lvl] = levels.get(lvl, 0) + 1
    print(f"  Level distribution: {levels}")

    numerical_l1 = sum(
        1 for q in l1_questions
        if re.search(r'\d+(?:\.\d+)?\s*[%~kMGVWVAHhzΩ℃倍年月日台回套米秒]',
                     q.get("expected_answer", ""))
    )
    print(f"  L1 numerical: {numerical_l1}/{len(l1_questions)} "
          f"({100 * numerical_l1 // max(len(l1_questions), 1)}%)")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regenerate L1 for v5 dataset")
    parser.add_argument("--clauses-only", action="store_true",
                        help="Only build clause pool, skip question generation")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate L1 questions from existing clauses_v5.json")
    parser.add_argument("--target", type=int, default=100,
                        help="Target L1 question count (default: 100)")
    args = parser.parse_args()

    if args.generate_only:
        # Load existing clauses_v5.json
        if not os.path.exists(CLAUSES_V5):
            print(f"[FATAL] {CLAUSES_V5} not found. Run without --generate-only first.")
            sys.exit(1)
        with open(CLAUSES_V5, "r", encoding="utf-8") as f:
            data = json.load(f)
        clauses = data.get("clauses", [])
        print(f"Loaded {len(clauses)} clauses from clauses_v5.json")
    else:
        # Build clause pool
        clauses = build_clause_pool()
        if not clauses:
            sys.exit(1)

    if args.clauses_only:
        print("\nClause pool built. Skipping question generation.")
        return

    # Generate L1 questions
    l1_questions = generate_l1_questions(clauses, target_count=args.target)

    if not l1_questions:
        print("[FATAL] No L1 questions generated.")
        sys.exit(1)

    # Assemble v5
    assemble_v5(l1_questions)

    print(f"\n{'=' * 60}")
    print("v5 regeneration complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
