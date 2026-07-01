"""Phase 2 v2: Extract clauses and scenario materials with source-type awareness.

Strategy:
- Standards (GB/DLT): Regex-extract normative clauses → L1/L2/L3 seeds
- Design docs & plans: Extract scenario parameters (NOT clauses) → inject into L2/L3 generation
- Feedback forms: Extract real review comments → high-quality L2 seed scenarios

Output:
- clauses_v1.json: normative clauses from standards only
- scenario_materials_v1.json: engineering parameters & conflict situations from design docs
"""

import json
import re
import time
from data_generation.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STRONG_MODEL, CLAUSES_OUTPUT,
)
from data_generation.source_loader import load_all_sources


# ── Source type classification ───────────────────────────────────────

def classify_source(title):
    """Classify source file by type."""
    if any(kw in title for kw in ["GB 38755", "DLT 5429", "DLT 5218"]):
        return "standard"
    if "反馈单" in title or "反馈" in title or title.endswith(".md"):
        return "feedback"
    if "方案" in title or "可研" in title:
        return "design"
    if "规划" in title:
        return "plan"
    return "other"


# ── Clause extraction from STANDARDS only ────────────────────────────

# Pre-compiled patterns for normative clauses
L1_PATTERNS = [
    # Generic numerical clauses
    re.compile(r"([^。；\n]{5,80}(?:应|必须|不应|不得|严禁)[^。；\n]{5,120}"
               r"(?:\d+(?:\.\d+)?\s*[%~kMGVWVAHhzΩ℃倍年月日台回套])[^。；\n]{0,60})"
               r"[。；\n]"),
    re.compile(r"([^。；\n]{5,100}(?:不低于|不大于|不超过|不小于|达到|满足|[0-9]+[%~]~[0-9]+[%~])"
               r"[^。；\n]{0,80})[。；\n]"),
    # Substation design — electrical safety distances (NOT civil/structural)
    re.compile(r"([^。；\n]{5,80}(?:安全净距|安全距离|最小距离|带电距离|相间距离)"
               r"(?:[^。；\n](?!围墙|道路|排水|消防|防火|大门)){5,120})[。；\n]"),
    # Substation design — electrical numerical requirements ONLY
    re.compile(r"([^。；\n]{5,100}(?:站用电容量|配电装置|电缆沟|照明.*?电压|"
               r"构架均压|接触电压|跨步电压|接地电阻|二次系统)"
               r"[^。；\n]{5,120})[。；\n]"),
    # Electrical equipment parameters
    re.compile(r"([^。；\n]{5,60}(?:变压器|电抗器|电容器|断路器|隔离开关|互感器|避雷器|"
               r"GIS|HGIS|AIS|母线|开关柜|消弧线圈|接地变|站用变|SVG|SVC)"
               r"[^。；\n]{5,120}(?:\d+(?:\.\d+)?\s*(?:kV|MW|MVA|kA|mm|m|℃))"
               r"[^。；\n]{0,60})[。；\n]"),
]

L2_PATTERNS = [
    re.compile(r"([^。；\n]{5,60}(?:当|若|如|发生)[^。；\n]{10,120}"
               r"(?:时|情况下|工况下|条件下|方式下)"
               r"[^。；\n]{5,200})[。；\n]"),
    re.compile(r"([^。；\n]{5,60}(?:N-?[12]|单一故障|任一元件|某一回|一回线路|一台机组)"
               r"[^。；\n]{5,200})[。；\n]"),
    re.compile(r"([^。；\n]{5,40}(?:根据|依据|按照|参照|执行)[^。；\n]{5,200})[。；\n]"),
    # Substation design — conditional requirements
    re.compile(r"([^。；\n]{5,60}(?:采用|选用|宜采用|不宜采用|不应采用|推荐采用)"
               r"[^。；\n]{10,120}(?:时|情况下|条件下|地区|区域|环境)"
               r"[^。；\n]{5,120})[。；\n]"),
    # Substation design — comparative selection
    re.compile(r"([^。；\n]{5,80}(?:接线(?:形式|方式)|布置(?:形式|方式)|主接线|电气主接线)"
               r"[^。；\n]{5,150})[。；\n]"),
]

L3_PATTERNS = [
    re.compile(r"([^。；\n]{5,60}(?:符合|遵守|满足|执行)"
               r"\s*(?:GB|DL/?T|Q/GDW|IEEE|IEC)\s*[0-9]{2,6}[^。；\n]{5,120})"
               r"[。；\n]"),
]

TOPIC_KEYWORDS = {
    "无功补偿": ["无功", "补偿", "电容器", "电抗器", "SVC", "STATCOM"],
    "电压控制": ["电压", "过电压", "工频", "操作过电压"],
    "稳定标准": ["稳定", "暂态", "动态", "N-1", "N-2", "功角", "振荡"],
    "接地设计": ["接地", "零序"],
    "导线选择": ["导线", "截面", "电晕", "经济电流"],
    "短路电流": ["短路", "断路器", "遮断"],
    "调峰方案": ["调峰", "备用", "负荷"],
    "黑启动": ["黑启动", "恢复", "自励磁"],
    "电网结构": ["网架", "分层", "分区", "环网", "联络"],
    "绝缘配合": ["绝缘", "间隙", "爬电"],
    "继电保护": ["保护", "重合闸", "潜供"],
    "新能源": ["新能源", "风电", "光伏", "变流器", "惯量"],
    "直流输电": ["直流", "换流", "HVDC"],
    # Substation design topics — ELECTRICAL ONLY (no civil/fire/structural)
    "电气主接线": ["接线", "母线", "分段", "一个半断路器", "双母线", "单母线", "桥形", "线变组"],
    "配电装置": ["配电装置", "间隔", "构架", "GIS", "HGIS", "AIS", "开关柜"],
    "站用电系统": ["站用电", "所用变", "备用电源", "BZT", "备自投", "交流电源", "直流电源"],
    "安全净距": ["安全净距", "安全距离", "带电距离", "相间距离", "对地距离"],
    "过电压与绝缘": ["过电压", "绝缘配合", "避雷器", "避雷针", "接地", "接触电压", "跨步电压"],
    "二次系统": ["继电保护", "监控", "自动化", "通信", "远动", "电能计量", "同步相量"],
    "电缆敷设": ["电缆", "电缆沟", "电缆层", "防火封堵", "阻燃"],
}


def detect_topic(text, section_title):
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text or kw in section_title for kw in keywords):
            return topic
    return "通用要求"


def extract_standard_name(text, source_title):
    # Match known standards by number pattern
    std_patterns = [
        (r"GB\s*38755", "GB 38755-2019"),
        (r"DL/?T\s*5429", "DL/T 5429-2009"),
        (r"DL/?T\s*5218", "DL/T 5218-2012"),
        (r"GB/T\s*([0-9]{4,6})", None),  # Generic GB/T
        (r"DL/?T\s*([0-9]{4})", None),     # Generic DL/T
        (r"SD\s*([0-9]{2,4})", None),       # SD standards
        (r"Q/GDW\s*([0-9]{2,6})", None),    # Q/GDW standards
    ]
    for pat, known_name in std_patterns:
        m = re.search(pat, source_title + " " + text[:200])
        if m:
            if known_name:
                return known_name
            num = m.group(1)
            prefix = "GB/T" if "GB/T" in m.group(0) else "DL/T" if "DL" in m.group(0) else m.group(0)[:5].strip()
            return f"{prefix} {num}"
    return "未知标准"


def extract_key_terms(text):
    terms = []
    for t in ["无功补偿", "静态稳定", "暂态稳定", "短路电流",
              "N-1", "N-2", "工频过电压", "潜供电流",
              "并联电抗器", "串联补偿", "PSS", "SVC",
              "自励磁", "黑启动", "电磁环网", "分层分区",
              "短路比", "电压崩溃", "低频振荡", "调峰",
              "备用容量", "导线截面", "电晕", "绝缘配合",
              "接地电阻", "变压器", "断路器", "GIS",
              "特高压", "直流输电", "紧凑型线路"]:
        if t in text:
            terms.append(t)
    return terms[:5]


def build_clause(clause_text, section, source_title, difficulty):
    standard = extract_standard_name(clause_text, source_title)
    topic = detect_topic(clause_text, section.get("section_title", ""))
    has_num = bool(re.search(r"\d+(?:\.\d+)?\s*(?:%|[kMGV]?[WVAHhzΩ℃]|倍)", clause_text))
    has_cond = bool(re.search(r"(?:当|若|按照|根据|在.{2,10}(?:时|下|中))", clause_text))
    has_ref = bool(re.search(r"(?:GB|DL/?T|Q/GDW|IEEE|IEC)\s*[0-9]{2,6}", clause_text))

    return {
        "standard": standard,
        "section": section.get("section_number", "")[:80],
        "clause_text": clause_text,
        "difficulty": difficulty,
        "topic": topic,
        "has_numerical_threshold": has_num,
        "has_conditional_logic": has_cond,
        "references_other_standards": has_ref,
        "key_terms": extract_key_terms(clause_text),
        "source_file": source_title,
        "source_type": "standard",
        "extraction_method": "regex",
    }


def extract_clauses_from_standards(sections, source_title):
    """Extract normative clauses ONLY from standard documents."""
    clauses = []

    for sec in sections:
        text = sec["text"]
        if len(text) < 50:
            continue

        for pattern in L1_PATTERNS:
            for match in pattern.finditer(text):
                ct = match.group(1).strip()
                if 20 < len(ct) < 600:
                    clauses.append(build_clause(ct, sec, source_title, "L1"))

        for pattern in L2_PATTERNS:
            for match in pattern.finditer(text):
                ct = match.group(1).strip()
                if 20 < len(ct) < 600:
                    if not any(c["clause_text"] == ct for c in clauses):
                        clauses.append(build_clause(ct, sec, source_title, "L2"))

        for pattern in L3_PATTERNS:
            for match in pattern.finditer(text):
                ct = match.group(1).strip()
                if 20 < len(ct) < 600:
                    if not any(c["clause_text"] == ct for c in clauses):
                        clauses.append(build_clause(ct, sec, source_title, "L3"))

    return clauses


# ── Scenario material extraction from design docs & plans ─────────────

def extract_scenario_materials(sections, source_title, source_type):
    """Extract concrete engineering parameters, constraints, and conflicts
    from design documents. These are NOT clauses — they are raw material
    for the Challenger to build realistic scenarios around standard clauses.
    """
    materials = []

    for sec in sections:
        text = sec["text"]
        if len(text) < 100:
            continue

        material = {"section_title": sec.get("section_title", ""), "source_type": source_type}

        # Extract engineering parameters: voltage + number + unit
        params = re.findall(
            r"(\d+)\s*(kV|MW|MVA|kA|Hz|km|mm|Ω|℃|%|台|回|套|万元|亿元)",
            text
        )
        if params:
            material["has_params"] = True
            material["param_count"] = len(params)
            material["sample_params"] = [f"{v}{u}" for v, u in params[:10]]

        # Extract standard references
        std_refs = re.findall(
            r"(?:GB|DL/?T|Q/GDW)\s*[0-9]{4,6}(?:[-–—][0-9]{2,4})?",
            text
        )
        if std_refs:
            material["has_std_refs"] = True
            material["std_refs"] = list(set(std_refs))[:10]

        # Detect conflict/decision patterns (valuable for L3)
        conflicts = []
        if "方案" in text and ("比选" in text or "比较" in text or "推荐" in text):
            conflicts.append("方案比选")
        if any(kw in text for kw in ["冲突", "矛盾", "制约", "博弈"]):
            conflicts.append("多约束冲突")
        if re.search(r"(?:不符合|不满足|低于|超过|超标|违规)", text):
            conflicts.append("合规性问题")
        if conflicts:
            material["has_conflicts"] = True
            material["conflict_types"] = conflicts

        if material.get("has_params") or material.get("has_std_refs") or material.get("has_conflicts"):
            material["preview"] = text[:300]
            materials.append(material)

    return materials


# ── DeepSeek extraction for complex narrative documents ───────────────

def build_narrative_prompt(sections_batch, source_title):
    sections_text = "\n\n---\n\n".join(
        f"[{s.get('section_number','')[:60]}] {s['text'][:800]}"
        for s in sections_batch
    )
    return f"""从以下电力工程文档中提取有价值的信息：

## 提取重点
1. 具体的工程设计参数（电压等级、容量、截面等）
2. 方案比选中的技术决策依据
3. 与国家标准相关的合规性要求引用
4. 工程约束和冲突点（如短路电流超标、电压越限、走廊受限等）

## 输出格式
纯JSON数组，每项：
{{"type":"scenario_material","section":"...","content":"...","keywords":[...],"references_standards":[...],"conflict_type":"compliance/design_tradeoff/constraint/none"}}

文档内容：
{sections_text}"""


def call_deepseek(prompt, max_tokens=4096):
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=STRONG_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def parse_response(text):
    for pattern in [text, *re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text),
                     re.findall(r"\[\s*\{[\s\S]*\}\s*\]", text)]:
        try:
            result = json.loads(pattern) if isinstance(pattern, str) else pattern
            if isinstance(result, list): return result
        except: continue
    return []


def extract_narrative_materials(sections, source_title, batch_size=10):
    """Use DeepSeek to extract scenario materials from large narrative docs."""
    eligible = [s for s in sections if 200 <= s.get("char_count", 0) <= 3000]
    all_materials = []
    for i in range(0, len(eligible), batch_size):
        batch = eligible[i:i+batch_size]
        print(f"    Batch {i//batch_size+1}/{(len(eligible)-1)//batch_size+1}")
        try:
            resp = call_deepseek(build_narrative_prompt(batch, source_title))
            materials = parse_response(resp)
            for m in materials:
                m["source_file"] = source_title
            all_materials.extend(materials)
            print(f"      -> {len(materials)} materials")
        except Exception as e:
            print(f"      -> [ERR] {e}")
            time.sleep(3)
        time.sleep(1)
    return all_materials


# ── Feedback extraction ───────────────────────────────────────────────

def extract_feedback_items(sections, source_title):
    """Extract structured review templates from feedback/meeting-minutes docs.

    Captures the triple structure:
      (问题描述, 审查意见, 答复)
    which maps to L2 format:
      (scenario, implicit-review-context, expected_answer)

    For .md review-meeting minutes, extracts each question-answer pair
    as a review_template with full context.
    """
    items = []
    for sec in sections:
        text = sec["text"]
        if len(text) < 50:
            continue

        # ── Pattern A: Review meeting minutes (.md files) ──
        # Format: 问题N: [description] ... 答复: [response]
        qa_pattern = re.compile(
            r'(?:问题|问额)\s*(\d+)\s*[：:.]\s*(.+?)'
            r'(?=(?:问题|间题|问额)\s*\d+\s*[：:.]|答复\s*\d*[：:.]|###|\Z)',
            re.DOTALL
        )
        ans_pattern = re.compile(
            r'答复\s*(\d*)\s*[：:.]\s*(.+?)(?=(?:问题|间题|问额)\s*\d+\s*[：:.]|答复\s*\d*[：:.]|###|\Z)',
            re.DOTALL
        )

        questions = [(m.group(1), m.group(2).strip()) for m in qa_pattern.finditer(text)]
        answers = {m.group(1): m.group(2).strip() for m in ans_pattern.finditer(text)}

        for num, q_text in questions:
            if len(q_text) < 20:
                continue
            ans_text = answers.get(num, "")
            items.append({
                "type": "review_template",
                "question_number": int(num),
                "question_text": q_text[:800],
                "answer_text": ans_text[:800] if ans_text else "",
                "source_file": source_title,
                "template_format": "问题→答复",
            })

        # ── Pattern B: Design review feedback forms ──
        if not questions:
            fb_pattern = re.compile(
                r"(?:问题|间题|问额)\s*\d+\s*[：:]\s*(.+?)(?=(?:问题|间题|问额)\s*\d+\s*[：:]|\Z)",
                re.DOTALL
            )
            for match in fb_pattern.finditer(text):
                content = match.group(1).strip()
                if len(content) > 20:
                    items.append({
                        "type": "review_feedback",
                        "content": content[:500],
                        "source_file": source_title,
                    })

    return items


# ── Dedup ─────────────────────────────────────────────────────────────

def deduplicate(clauses):
    unique, seen = [], []
    for c in clauses:
        txt = c.get("clause_text", "")
        ratio = max(len(set(txt) & set(s)) / max(len(set(txt)), 1) for s in seen) if seen else 0
        if ratio < 0.7:
            unique.append(c)
            seen.append(txt)
    return unique


# ── L1 fact extraction from standards via DeepSeek ─────────────────────

def extract_l1_facts_from_standards(standard_texts, target_count=80):
    """Use DeepSeek to extract all numerical parameter facts from standards.

    This supplements regex extraction with better coverage.
    """
    combined = "\n\n".join(
        f"=== {title} ===\n{text[:5000]}"
        for title, text in standard_texts.items()
    )

    prompt = f"""从以下电力系统标准中提取所有包含具体数值参数的规范条款。

提取格式（纯JSON数组）：
{{"standard":"标准编号","section":"条款号","parameter":"参数名称","value":"数值要求","clause_text":"完整条款原文","topic":"主题分类"}}

要求：
- 每个数值参数一条记录（比例、电压、电流、容量、时间、温度等）
- 参数必须来自标准原文，不编造
- 提取所有可以找到的数值型条款

标准文本：
{combined}

输出纯JSON数组："""

    try:
        response = call_deepseek(prompt, max_tokens=8192)
        facts = parse_response(response)
        return [{
            "standard": f.get("standard", "未知"),
            "section": f.get("section", ""),
            "clause_text": f.get("clause_text", f.get("parameter", "") + ": " + f.get("value", "")),
            "difficulty": "L1",
            "topic": f.get("topic", "通用要求"),
            "has_numerical_threshold": True,
            "has_conditional_logic": False,
            "references_other_standards": False,
            "key_terms": [f.get("parameter", "")],
            "source_file": f.get("standard", ""),
            "source_type": "standard",
            "extraction_method": "deepseek",
        } for f in facts]
    except Exception as e:
        print(f"    [L1 DeepSeek extraction error] {e}")
        return []


# ── Main ──────────────────────────────────────────────────────────────

def run_extraction():
    print("Phase 2 v2: Extracting clauses (standards) + scenario materials (docs)\n")
    sources, all_sections = load_all_sources()

    all_clauses = []
    all_scenario_materials = []

    for src in sources:
        title = src["title"]
        src_type = classify_source(title)
        n_sections = len(src["sections"])
        print(f"\n  {title} [{src_type}] ({n_sections} sections)")

        if src_type == "standard":
            clauses = extract_clauses_from_standards(src["sections"], title)
            all_clauses.extend(clauses)
            print(f"    -> {len(clauses)} normative clauses")

        elif src_type in ("design", "plan"):
            # Extract scenario materials (params + conflicts)
            materials = extract_scenario_materials(src["sections"], title, src_type)
            for m in materials:
                m["source_file"] = title

            # For large docs, also use DeepSeek for richer extraction
            if src["char_count"] > 50000 and src_type == "design":
                print(f"    DeepSeek extraction for large doc...")
                ds_materials = extract_narrative_materials(src["sections"], title)
                materials.extend(ds_materials)

            all_scenario_materials.extend(materials)
            print(f"    -> {len(materials)} scenario materials")

        elif src_type == "feedback":
            items = extract_feedback_items(src["sections"], title)
            for item in items:
                item["source_type"] = "feedback"
            all_scenario_materials.extend(items)
            print(f"    -> {len(items)} review feedback items")

    # Extract L1 facts via DeepSeek for better coverage
    print(f"\n  DeepSeek L1 fact extraction from standards...")
    standard_texts = {}
    for src in sources:
        if classify_source(src["title"]) == "standard":
            standard_texts[src["title"]] = src["raw_text"]
    if standard_texts:
        l1_facts = extract_l1_facts_from_standards(standard_texts, target_count=80)
        all_clauses.extend(l1_facts)
        print(f"    -> {len(l1_facts)} additional L1 facts from DeepSeek")

    # Deduplicate clauses
    before = len(all_clauses)
    all_clauses = deduplicate(all_clauses)
    print(f"\n  Clauses dedup: {before} -> {len(all_clauses)}")

    # Assign IDs
    for i, c in enumerate(all_clauses):
        c["clause_id"] = f"CL-{i+1:04d}"

    # Stats
    by_level = {"L1": 0, "L2": 0, "L3": 0}
    by_topic = {}
    for c in all_clauses:
        by_level[c.get("difficulty", "L1")] = by_level.get(c.get("difficulty", "L1"), 0) + 1
        t = c.get("topic", "未分类")
        by_topic[t] = by_topic.get(t, 0) + 1

    print(f"\n{'='*60}")
    print(f"Extraction Summary v2")
    print(f"{'='*60}")
    print(f"  Normative clauses (standards only): {len(all_clauses)}")
    for lvl in ["L1", "L2", "L3"]:
        print(f"    {lvl}: {by_level.get(lvl, 0)}")
    print(f"  Scenario materials (design/plan/feedback): {len(all_scenario_materials)}")
    print(f"  Topics: {dict(sorted(by_topic.items(), key=lambda x: -x[1])[:8])}")

    # Save clauses
    output = {"total": len(all_clauses), "by_level": by_level, "by_topic": by_topic,
              "clauses": all_clauses}
    with open(CLAUSES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Clauses saved: {CLAUSES_OUTPUT}")

    # Save scenario materials
    scenario_path = CLAUSES_OUTPUT.replace("clauses_v1.json", "scenario_materials_v1.json")
    with open(scenario_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(all_scenario_materials), "materials": all_scenario_materials},
                  f, ensure_ascii=False, indent=2)
    print(f"  Scenarios saved: {scenario_path}")

    return all_clauses, all_scenario_materials


if __name__ == "__main__":
    run_extraction()
