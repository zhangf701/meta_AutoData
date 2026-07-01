"""Rubric Builder: Extract rubric_clauses and rubric_judgments from generated answers.

Dual-channel approach:
1. Regex extraction for standard clause references
2. DeepSeek API fallback if regex returns < 2 clauses
"""

import re
import json
from openai import OpenAI
from data_generation.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STRONG_MODEL


# Known standard patterns
STD_PATTERN = re.compile(
    r"(?:GB|DL/?T|Q/GDW|IEEE|IEC|SDJ|NB)\s*[0-9]{2,6}"
    r"(?:[-–—][0-9]{2,4})?"
    r"(?:\s*[§§条款]+\s*[0-9]+(?:\.[0-9]+)*)?",
)


def extract_clause_refs(text):
    """Extract standard clause references from answer text."""
    if not isinstance(text, str):
        return []
    refs = []
    seen = set()

    for match in STD_PATTERN.finditer(text):
        ref = match.group(0).strip()
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)

    # Also match bare § references
    bare = re.findall(r"§\s*[0-9]+(?:\.[0-9]+)*(?:\.[0-9]+)*", text)
    for b in bare:
        if b not in seen:
            seen.add(b)
            refs.append(b)

    # If answer uses numeric reference tags like "参照§3.1.8" or "依据§6.2.3"
    context_refs = re.findall(r"(?:参照|依据|详见|见|参见|按照)\s*(?:标准条款)?\s*[§]?\s*[0-9]+(?:\.[0-9]+)*", text)
    for r in context_refs:
        ref = r.strip()
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)

    return refs


def extract_judgments(text):
    """Decompose answer into individual scoreable judgments."""
    if not isinstance(text, str):
        return []
    sentences = re.split(r"[。；;\.]\s*", text)
    judgments = []
    for s in sentences:
        s = s.strip()
        # Keep sentences that look like substantive claims
        if 15 < len(s) < 300 and any(kw in s for kw in ["应", "需", "必须", "要求", "规定", "标准", "规范", "不得", "禁止", "符合", "满足", "低于", "高于", "超过"]):
            judgments.append(s)
    return judgments[:15]  # Max 15 judgments


def deepseek_extract_clauses(answer_text, max_retries=2):
    """Use DeepSeek to extract clause references when regex fails."""
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    prompt = f"""从以下电力工程标准答案中，提取所有引用的标准条款编号。

答案文本：
{answer_text[:3000]}

输出格式：纯JSON字符串数组，如 ["GB 38755-2019 §3.1.8", "DL/T 5429-2009 §6.2.3"]

注意：
- 包含完整的标准编号和条款号
- 如果答案中提到了具体的条款号（如"§4.2.3"、"3.1.4条"），必须提取
- 如果是描述性的引述（如"强制性要求的直流短路比达标"），尝试推断条款位置
- 如果没有明确的条款引用，返回空数组 []"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=STRONG_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            text = response.choices[0].message.content
            # Parse JSON array
            arr = re.search(r"\[.*\]", text, re.DOTALL)
            if arr:
                return json.loads(arr.group(0))
            return []
        except Exception:
            if attempt == max_retries - 1:
                return []
    return []


def build_rubric(answer_text, level):
    """Build complete rubric for a generated answer.

    Args:
        answer_text: The expected_answer text
        level: "L1", "L2", or "L3"

    Returns:
        dict with rubric_clauses, rubric_judgments, rubric_refusal_expected, scoring_version
    """
    clauses = extract_clause_refs(answer_text)

    # For L3, if regex finds < 2 clauses, use DeepSeek fallback
    if level == "L3" and len(clauses) < 2:
        ds_clauses = deepseek_extract_clauses(answer_text)
        clauses = list(set(clauses + ds_clauses))

    judgments = extract_judgments(answer_text)

    return {
        "rubric_clauses": clauses,
        "rubric_judgments": judgments,
        "rubric_refusal_expected": False,
        "scoring_version": "v3-rubric",
    }
