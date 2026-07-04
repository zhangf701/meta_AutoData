"""Challenger Agent: Generate Q&A pairs from normative clauses.

Implements level-specific generation strategies:
- L1: LLM-based parameter/prescriptive extraction
- L2: LLM scenario construction with 3-shot examples
- L3: LLM cross-standard synthesis with scenario inlining
"""

import json
import re
import os
from openai import OpenAI
from data_generation.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STRONG_MODEL, L2_FEWSHOT_FILE


# ── L1 LLM-based Generation ────────────────────────────────────────────

L1_SYSTEM_PROMPT = """你是一名电力系统标准专家，有10年电力标准评审经验。
你的任务是将标准条款转化为L1级（参数检索/规定性要求）评测题目。

## L1 题目定义
L1题目考察对标准中具体参数或规定性要求的直接检索能力。
答案应简短精确（一个数值、一个范围、或一句规定性判断）。
题干应为一句完整的汉语问句，能独立理解。

## 题目要求（严格）
1. 题干：一句通顺独立的汉语问句，30-100字，以"根据{标准编号}"开头
2. 题干中不得出现条款原文的长句复制，必须提炼为简洁提问
3. 题干中不得出现"a)""b)""1.""2.""（1）"等编号或列表标记
4. 题干不得以"应符合什么规定/要求"等空泛句式结尾——必须指向具体参数
5. 答案：从条款原文中精准提取，不得编造、不得推断
6. 关键词：3-5个核心技术关键词，排除"a)""b)"等无意义词汇

## 数值型条款的正确示例
条款："短路比的最小值不应小于2.0"
→ {"query":"根据GB 38755-2019，短路比的最小允许值是多少？","expected_answer":"不小于2.0","expected_keywords":["短路比","最小允许值","2.0"]}

条款："系统总备用容量可按系统最大发电负荷的15%~20%考虑"
→ {"query":"根据DL/T 5429-2009，系统总备用容量占系统最大发电负荷的比例范围是多少？","expected_answer":"15%~20%","expected_keywords":["系统总备用容量","最大发电负荷","比例","15%","20%"]}

## 纯文字规定性条款的正确示例
条款："500kV及以上变压器中性点宜全部接地"
→ {"query":"根据DL/T 5429-2009，500kV及以上变压器中性点应采用什么接地方式？","expected_answer":"宜全部接地","expected_keywords":["变压器","中性点","接地方式","500kV"]}

## 错误示例（严禁出现）
❌ 题干复制了条款原文长句：
  "根据DL/T 5218-2012，220kV变电站中的220kV配电装置，当在系统中居重要地位、出线回路数为4回及以上时..."
❌ 题干出现编号或列表标记：
  "根据DL/T 5429-2009，2. 事故备用为8%~10%中规定的..."
❌ 答案出现占位文本：
  expected_answer: "无具体数值"
❌ 题干出现非电气内容（土建、消防、防洪、环保）

## 输出格式
纯JSON对象，不要markdown代码块包裹：
{"query":"一句通顺的汉语问句","expected_answer":"精准的数值或规定性判断","expected_keywords":["kw1","kw2","kw3"],"source_standard":"标准编号 条款号"}"""


def _extract_numerical_hints(text):
    """Pre-extract numerical parameters as hints for the LLM."""
    hints = []
    # Percentage ranges
    for m in re.finditer(r"(\d+(?:\.\d+)?\s*[%~](?:\s*[~～-]\s*\d+(?:\.\d+)?\s*[%~])?)", text):
        hints.append(f"比例范围: {m.group(1)}")
    # Threshold values
    for m in re.finditer(r"(不低于|不大于|不小于|不超过)\s*(\d+(?:\.\d+)?\s*[%kMGVWVAHhzΩ℃倍年月日台回套米秒]*)", text):
        hints.append(f"阈值: {m.group(0)}")
    # Number+unit pairs
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(kV|MW|MVA|kA|Hz|Ω|mm|km|年|月|日|台|回|套)", text):
        hints.append(f"参数: {m.group(0)}")
    return hints[:5]  # top 5 hints


def generate_l1_from_clause(clause):
    """Generate an L1 question from a single clause using LLM.

    Sends clause text + pre-extracted numerical hints to DeepSeek,
    which formulates a concise, natural-language question and extracts
    the exact answer.

    Returns dict with all required fields, or None on failure.
    """
    text = clause.get("clause_text", "")
    standard = clause.get("standard", "")
    section = clause.get("section", "")
    topic = clause.get("topic", "通用要求")

    # Skip unworkable clauses
    if len(text) < 12:
        return None
    if re.match(r'^[\d]+\.?\s*$', text.strip()):
        return None
    if re.match(r'^[a-z]\)\s*$', text.strip()):
        return None

    # Format standard reference cleanly: "GB 38755-2019" with space
    std_ref = standard.strip() if standard else "相关标准"
    std_ref = re.sub(r'(\d{4})(\d)', r'\1 \2', std_ref)  # Fix "38755-20194" → "38755-2019 4"

    # Format section number cleanly
    section_str = section.strip() if section else ""

    # Pre-extract numerical hints
    hints = _extract_numerical_hints(text)
    hints_str = "\n".join(hints) if hints else "（未检测到明确数值参数）"

    # Truncate clause text to avoid LLM copying long text verbatim
    clause_text_short = text[:250] if len(text) > 250 else text

    user_prompt = f"""## 标准信息
标准编号：{std_ref}
条款位置：{section_str}
主题分类：{topic}

## 条款原文
{clause_text_short}

## 预提取参数（辅助参考）
{hints_str}

请将以上条款转化为一道L1评测题目。题干必须是简洁的通顺问句，不得复制条款原文的长句。直接输出JSON："""

    try:
        response = call_llm(L1_SYSTEM_PROMPT, user_prompt, max_tokens=1024)
        result = parse_json_response(response)
        if result and "query" in result and "expected_answer" in result:
            query = result.get("query", "")
            answer = result.get("expected_answer", "")

            # ── Reject bad outputs ──
            # List marker start
            if re.match(r'^[a-z]\)|^\d+\.\s|^[（(]\d+[）)]|^[①②③]', query.strip()):
                return None
            # Too short/long
            if len(query) < 15 or len(query) > 200:
                return None
            # Raw clause text embedded: contains garbled desensitization artifacts
            if re.search(r'某专家|某区域|变电站[A-Z][^侧站路]|线路[A-Z][^路]', query):
                return None
            # Vague answer
            if re.search(r'无具体|未明确|未给出|需参考其他|需查阅|标准中未', answer):
                return None
            # Answer is just the clause text rephrased (too long)
            if len(answer) > 120:
                return None
            # Keywords contain garbled terms
            keywords = result.get("expected_keywords", [])
            clean_keywords = [k for k in keywords
                            if not re.search(r'[a-z]\)|^\d+\.|某', k)
                            and len(k) >= 2][:5]

            return {
                "question_class": "L1", "level": "L1",
                "category": result.get("category", "参数检索"),
                "query": query,
                "expected_answer": answer,
                "expected_keywords": clean_keywords,
                "source_standard": result.get("source_standard",
                                              f"{std_ref} {section_str}" if std_ref else ""),
                "grading_method": "auto_keyword_match",
                "knowledge_base": "General-KB",
                "clause_source": clause.get("clause_id", ""),
            }
    except Exception as e:
        print(f"    [L1 LLM Error] {e}")

    return None


# ── L2/L3 LLM-based Generation ───────────────────────────────────────

def _load_fewshot_examples():
    """Load L2 examples from the manually-written question bank."""
    if not os.path.exists(L2_FEWSHOT_FILE):
        return []

    with open(L2_FEWSHOT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    examples = []
    # Parse [问题 N] sections
    problem_pattern = re.compile(r"##\s*【问题\s*(\d+)】(.*?)(?=##\s*【问题|\Z)", re.DOTALL)
    for match in problem_pattern.finditer(content):
        block = match.group(2)

        scenario = re.search(r"\*\*场景\*\*[：:]\s*(.+?)(?=\*\*问题\*\*)", block, re.DOTALL)
        question = re.search(r"\*\*问题\*\*[：:]\s*(.+?)(?=\*\*标准答案\*\*)", block, re.DOTALL)
        answer = re.search(r"\*\*标准答案\*\*[：:]\s*(.+?)(?=\*\*关键词\*\*)", block, re.DOTALL)
        keywords = re.search(r"\*\*关键词\*\*[：:]\s*(.+?)(?=\*\*相关标准条款)", block, re.DOTALL)

        if scenario and question and answer:
            examples.append({
                "scenario": scenario.group(1).strip(),
                "question": question.group(1).strip(),
                "answer": answer.group(1).strip(),
                "keywords": keywords.group(1).strip() if keywords else "",
                "full_query": f"【场景】{scenario.group(1).strip()}\n\n【问题】{question.group(1).strip()}",
            })

    return examples[:6]  # Use up to 6 examples


L2_SYSTEM_PROMPT = """你是一名资深电力系统设计审查专家，有15年输变电工程评审经验。
你的任务是基于标准条款创作L2级（推理型）评测题目。

## 题目要求
1. **场景**：基于条款发明一个具体的工程设计场景，包含具体参数（电压等级、设备容量、故障类型等）
2. **问题**：要求进行条件判断或跨参数推理，而非简单查表
3. **答案**：必须引用标准条款编号，给出推理过程和结论
4. **关键词**：提取5个核心技术关键词
5. **场景与问题严格分离**：
   - 场景只描述工程背景和参数，不提问
   - 问题只提问（如"在上述场景下，请判断..."），不得重复场景中已给出的电压等级、设备参数、标准编号
   - 场景中已出现的技术参数（电压、容量、距离、电流等级等），问题中禁止再次出现

## L2 定义
答案需要条件判断或跨参数推理（非简单查表）。例如：
- "该方案在N-1故障后是否满足静态稳定储备要求？"
- "当短路电流超标时，哪种限流措施最经济合理？"

## 参考格式（场景≠问题）
【场景】某500kV枢纽变电站规划安装2台1000MVA主变压器，500kV侧短路电流水平约为45kA，220kV侧短路电流水平约为35kA。该站500kV侧采用3/2断路器接线，220kV侧采用双母线分段接线。站址位于沿海高盐雾地区，对设备外绝缘有较高要求。

【问题】在上述场景下，该站220kV侧断路器的额定遮断容量应如何选取？是否需要采取限流措施？请引用相关标准条款说明。

## 输出格式
纯JSON对象：
{"query":"【场景】...（150-300字具体场景，含工程参数）\n\n【问题】...（简短引用式提问，≤80字）","expected_answer":"基于标准条款的详细推理答案","expected_keywords":["kw1","kw2","kw3","kw4","kw5"],"source_standard":"标准编号 条款号","category":"审查判断/条款解释/条款比对/条件推理"}"""


L3_SYSTEM_PROMPT = """你是一名电力系统规划设计总工程师，负责大型输变电工程方案的最终技术决策。
你的任务是基于多条来自不同标准的条款，创作L3级（综合型）评测题目。

## 题目要求【严格】
1. **场景**：必须包含工程背景（≥300字）和至少两个完整的技术方案（方案A、方案B），每个方案含具体参数
2. **问题**：必须要求跨标准综合论证（同时引用不同标准的多条条款），涉及方案比选和技术决策
3. **答案**：必须包含五段式结构——
   - 现状诊断（问题识别）
   - 多维冲突分析（不同标准条款间的约束冲突）
   - 多方案比选（方案A vs 方案B的技术经济对比）
   - 折中综合方案（给出唯一推荐方案及理由）
   - 控制优先级链条（按优先级列出技术措施序列）

## 严禁
- 在query中只写"方案A"和"方案B"而不描述方案具体内容
- 答案中使用"根据相关标准"而不注明具体标准编号和条款号
- 使用LaTeX数学公式中的转义字符

## 输出格式
纯JSON对象：
{"query":"【场景】...（≥300字，含方案A和方案B的完整描述）\n\n【问题】...","expected_answer":"现状诊断：...\n多维冲突分析：...\n多方案比选：...\n折中综合方案：...\n控制优先级链条：...","expected_keywords":["kw1","kw2","kw3","kw4","kw5"],"source_standard":"标准1, 标准2","category":"综合评估/风险研判/方案对比/多标准协调"}"""


def call_llm(system_prompt, user_prompt, max_tokens=4096):
    """Call DeepSeek API for question generation."""
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=STRONG_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def parse_json_response(text):
    """Parse LLM response into JSON object."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code blocks
    code = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code:
        try:
            return json.loads(code.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding JSON object
    obj = re.search(r"\{[\s\S]*\}", text)
    if obj:
        try:
            return json.loads(obj.group(0))
        except json.JSONDecodeError:
            pass
    return None


class Challenger:
    """Generate Q&A pairs from clauses using level-appropriate strategies."""

    def __init__(self):
        self.l2_examples = _load_fewshot_examples()
        print(f"  [Challenger] Loaded {len(self.l2_examples)} L2 few-shot examples")

    def generate(self, clause_or_clauses, level, narrow_feedback=None, scenario_material=None):
        """Route to level-appropriate generator.

        Args:
            clause_or_clauses: Single clause dict for L1/L2, or list of clauses for L3
            level: "L1", "L2", or "L3"
            narrow_feedback: Optional feedback from LoopJudge for L3 narrowing
            scenario_material: Optional scenario context from design docs

        Returns:
            dict with all question fields, or None on failure
        """
        if level == "L1":
            result = generate_l1_from_clause(clause_or_clauses)
        elif level == "L2":
            result = self._generate_l2(clause_or_clauses, narrow_feedback, scenario_material)
        elif level == "L3":
            result = self._generate_l3(clause_or_clauses, narrow_feedback, scenario_material)
        else:
            return None

        if result:
            result["level"] = level
        return result

    def _generate_l2(self, clause, narrow_feedback=None, scenario_material=None):
        """Generate L2 question with scenario construction."""
        examples_text = ""
        if self.l2_examples:
            for i, ex in enumerate(self.l2_examples[:3]):
                examples_text += f"\n## 示例{i+1}\n场景：{ex['scenario'][:200]}\n问题：{ex['question'][:200]}\n答案概要：{ex['answer'][:200]}\n"

        clause_text = clause.get("clause_text", "")
        standard = clause.get("standard", "")
        topic = clause.get("topic", "")

        # Inject scenario material if available
        scenario_hint = ""
        if scenario_material:
            preview = scenario_material.get("preview", scenario_material.get("content", ""))
            conflicts = scenario_material.get("conflict_types", [])
            params = scenario_material.get("sample_params", [])
            scenario_hint = f"\n## 可用的工程场景素材（来自设计文档）\n背景文本：{preview[:800]}\n"
            if conflicts:
                scenario_hint += f"冲突类型：{', '.join(conflicts)}\n"
            if params:
                scenario_hint += f"相关参数：{', '.join(params[:20])}\n"
            scenario_hint += "\n请基于上述场景素材发明一个具体的工程场景，将标准条款应用于该场景中。\n"

        feedback_block = ""
        if narrow_feedback:
            feedback_block = "## 改进建议\n" + narrow_feedback + "\n\n"

        user_prompt = f"""## 参考标准条款
标准：{standard}
主题：{topic}
条款内容：{clause_text}

## 参考示例（格式参考，请勿照抄内容）
{examples_text}

{feedback_block}请基于上述条款创作一道L2级别的推理型评测题目。"""

        try:
            response = call_llm(L2_SYSTEM_PROMPT, user_prompt)
            result = parse_json_response(response)
            if result:
                result["clause_source"] = clause.get("clause_id", "")
                result["grading_method"] = "manual_review"
                result["knowledge_base"] = "General-KB"
                return result
        except Exception as e:
            print(f"    [Challenger L2 Error] {e}")

        return None

    def _generate_l3(self, clauses, narrow_feedback=None, scenario_material=None):
        """Generate L3 question with cross-standard synthesis.

        Requires 2+ clauses from different standards.
        """
        if isinstance(clauses, dict):
            clauses = [clauses]

        # Build clause context
        clauses_text = ""
        standards_used = set()
        for c in clauses:
            std = c.get("standard", "")
            standards_used.add(std)
            clauses_text += f"\n【{std}】§{c.get('section','')}\n{c.get('clause_text','')}\n"
            clauses_text += f"主题：{c.get('topic','')}\n"

        standards_list = ", ".join(standards_used)

        # Inject scenario material with conflicts
        scenario_hint = ""
        if scenario_material:
            preview = scenario_material.get("preview", scenario_material.get("content", ""))
            conflicts = scenario_material.get("conflict_types", [])
            params = scenario_material.get("sample_params", [])
            scenario_hint = (
                f"\n## 可用的工程冲突场景素材（来自设计文档）\n"
                f"背景：{preview[:400]}\n"
            )
            if conflicts:
                scenario_hint += f"工程冲突类型：{', '.join(conflicts)}\n"
            if params:
                scenario_hint += f"具体参数：{', '.join(params[:10])}\n"
            scenario_hint += "\n请基于上述工程冲突，构造方案A和方案B，将多条标准条款的约束冲突嵌入这两个方案中。\n"

        feedback_block = ""
        if narrow_feedback:
            feedback_block = "## 改进建议（来自上一轮评审）\n" + narrow_feedback + "\n\n"

        user_prompt = f"""## 待综合的标准条款
{clauses_text}

涉及的规范标准：{standards_list}

{feedback_block}## 注意
- query中的场景描述必须≥300字，且必须完整描述方案A和方案B
- 答案必须包含现状诊断、多维冲突分析、多方案比选、折中综合方案、控制优先级链条五个部分
- 每个判断必须引用具体标准条款编号"""

        try:
            response = call_llm(L3_SYSTEM_PROMPT, user_prompt, max_tokens=8192)
            result = parse_json_response(response)
            if result:
                result["clause_source"] = ",".join(c.get("clause_id", "") for c in clauses)
                result["grading_method"] = "manual_review"
                result["knowledge_base"] = "General-KB"
                return result
        except Exception as e:
            print(f"    [Challenger L3 Error] {e}")

        return None
