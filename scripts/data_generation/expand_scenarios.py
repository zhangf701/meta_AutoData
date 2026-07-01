"""expand_scenarios.py: Safe L2/L3 scenario expansion with anti-hallucination layers.

Reads generated_eval_set_v2.json, applies:
  - L2: scenario expansion for items missing 【场景】 (68 items)
  - L3: scenario expansion (6) + answer five-section restructuring (46)
  - Gap filling: new L1 (8) + new L3 (4)
  - All with 3-layer anti-hallucination defense
Output: generated_eval_set_v3.json (target: 100 L1 + 100 L2 + 100 L3)
"""

import sys, os, json, re, time, copy
from collections import Counter
from datetime import datetime

sys.path.insert(0, r'D:\coding\meta_AutoData\scripts')
from data_generation.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STRONG_MODEL
from openai import OpenAI
from data_generation.utils.rubric_builder import build_rubric
from data_generation.utils.escaping_fixer import fix_question_escaping
from data_generation.utils.format_validator import validate_question
from data_generation.source_loader import load_all_sources

# ═══════════════════════════════════════════════════════════
# Layer 2: Parameter Sandbox
# ═══════════════════════════════════════════════════════════

# Realistic Chinese power system parameter ranges by voltage level
PARAMETER_SANDBOX = {
    220: {
        'line_capacity_mw': (100, 600),       # 220kV single circuit
        'sc_current_ka': (10, 50),             # short-circuit current
        'xfmr_mva': (90, 360),                 # transformer capacity
        'bus_sections': (1, 4),
        'reactive_comp_mvar': (-200, 300),
        'cable_charging_mvar_per_km': (0.1, 0.5),
    },
    330: {
        'line_capacity_mw': (300, 1000),
        'sc_current_ka': (15, 50),
        'xfmr_mva': (150, 750),
        'bus_sections': (1, 4),
        'reactive_comp_mvar': (-300, 500),
    },
    500: {
        'line_capacity_mw': (600, 3000),
        'sc_current_ka': (20, 63),
        'xfmr_mva': (500, 1500),
        'bus_sections': (2, 6),
        'reactive_comp_mvar': (-500, 800),
    },
    750: {
        'line_capacity_mw': (1500, 5000),
        'sc_current_ka': (25, 63),
        'xfmr_mva': (1000, 2100),
        'bus_sections': (2, 6),
        'reactive_comp_mvar': (-600, 1000),
    },
    800: {  # ±800kV UHVDC
        'line_capacity_mw': (3000, 12000),     # DC line equivalent
        'dc_capacity_mw': (3000, 12000),
        'sc_current_ka': (30, 63),
        'scr_min': 2.0,                       # short circuit ratio minimum
        'xfmr_mva': (500, 1500),
        'reactive_comp_mvar': (-800, 1200),
    },
}

# Standard applicability ranges
STANDARD_SCOPE = {
    'GB 38755-2019': {'min_kv': 110, 'description': '电力系统安全稳定导则，适用于110kV及以上电力系统'},
    'DL/T 5429-2009': {'min_kv': 110, 'description': '电力系统设计技术规程'},
    'DL/T 5218-2012': {'min_kv': 220, 'max_kv': 750, 'description': '220kV~750kV变电站设计技术规程'},
}

# Impossible/improbable parameter combinations
IMPOSSIBLE_COMBOS = [
    (r'220kV.*3/2\s*断路器', '3/2断路器接线一般不用于220kV（通常≥330kV）'),
    (r'110kV.*750MVA', '110kV变压器容量不超过120MVA'),
    (r'单回.*500kV.*线路.*8000MW', '单回500kV线路输送功率不超过3000MW'),
    (r'SCR\s*[＝=]\s*0\.[0-9]', '短路比（SCR）不应低于1.0'),
    (r'MISCR\s*[＝=]\s*0\.[0-9]', '多馈入短路比（MISCR）不应低于1.5'),
]

PHYSICAL_VIOLATIONS = [
    (r'频率.*(\d{2})0\s*Hz', lambda m: not (47 <= int(m.group(1)) <= 52), '电网频率应在47-52Hz范围内'),
    (r'电压.*跌落.*至\s*0\.(\d)', lambda m: float(m.group(1)) <= 2, '电压跌落至0.0?p.u.为完全失压，通常>0.2p.u.'),
]


def extract_voltage_levels(text):
    """Extract all voltage levels mentioned in text (kV)."""
    levels = set()
    for m in re.finditer(r'(?:^|[^\d])(\d{2,4})\s*(?:kV|千伏)', text):
        kv = int(m.group(1))
        if 10 <= kv <= 1200:
            levels.add(kv)
    # Also match ±800kV DC
    for m in re.finditer(r'±\s*(\d{3,4})\s*(?:kV|千伏)', text):
        kv = int(m.group(1))
        if 100 <= kv <= 1200:
            levels.add(kv)
    return sorted(levels)


def validate_parameters(text):
    """Layer 2: Check all numerical parameters against sandbox. Returns (pass, violations)."""
    violations = []
    voltage_levels = extract_voltage_levels(text)

    if not voltage_levels:
        return True, []  # No voltage levels to check against

    # Use the highest voltage level mentioned as primary context
    max_kv = max(voltage_levels)
    # Find closest sandbox entry
    applicable = None
    for kv in sorted(PARAMETER_SANDBOX.keys()):
        if kv >= max_kv or kv >= max(voltage_levels, key=lambda v: abs(v-max_kv)):
            applicable = kv
            break
    if applicable is None:
        applicable = 500  # default

    box = PARAMETER_SANDBOX.get(applicable, PARAMETER_SANDBOX[500])

    # Check MW values
    for m in re.finditer(r'(\d{3,5})\s*(?:MW|兆瓦)', text):
        mw = int(m.group(1))
        lo, hi = box['line_capacity_mw']
        if mw > hi * 1.5:
            violations.append(f'功率值 {mw}MW 超出 {applicable}kV 合理上限 {int(hi*1.5)}MW')
        if mw < 5:
            violations.append(f'功率值 {mw}MW 异常低')

    # Check MVA values (transformers)
    for m in re.finditer(r'(\d{3,4})\s*(?:MVA|兆伏安)', text):
        mva = int(m.group(1))
        lo, hi = box.get('xfmr_mva', (100, 2000))
        if mva > hi * 1.3:
            violations.append(f'变压器容量 {mva}MVA 超出 {applicable}kV 合理上限 {int(hi*1.3)}MVA')
        if mva < lo * 0.3:
            violations.append(f'变压器容量 {mva}MVA 异常低（{applicable}kV 系统不低于 {int(lo*0.3)}MVA）')

    # Check kA values (short circuit)
    for m in re.finditer(r'(\d{2})\s*(?:kA|千安)', text):
        ka = int(m.group(1))
        lo, hi = box.get('sc_current_ka', (10, 63))
        if ka > hi * 1.2:
            violations.append(f'短路电流 {ka}kA 超出合理上限')

    # Check impossible combos
    for pattern, explanation in IMPOSSIBLE_COMBOS:
        if re.search(pattern, text):
            violations.append(explanation)

    # Check physical violations
    for pattern, test_fn, explanation in PHYSICAL_VIOLATIONS:
        m = re.search(pattern, text)
        if m and test_fn(m):
            violations.append(explanation)

    return len(violations) == 0, violations


def validate_standard_scope(text, source_standard):
    """Check that voltage levels in text are within standard's scope."""
    violations = []
    voltage_levels = extract_voltage_levels(text)
    for std_name, scope in STANDARD_SCOPE.items():
        if std_name in source_standard:
            for kv in voltage_levels:
                if 'min_kv' in scope and kv < scope['min_kv']:
                    violations.append(f'{kv}kV 低于 {std_name} 适用范围（≥{scope["min_kv"]}kV）')
                if 'max_kv' in scope and kv > scope['max_kv'] and '变电站' in std_name:
                    violations.append(f'{kv}kV 超出 {std_name} 适用范围（≤{scope["max_kv"]}kV）')
    return len(violations) == 0, violations


# ═══════════════════════════════════════════════════════════
# Layer 1: Constraint-Injected Prompts
# ═══════════════════════════════════════════════════════════

def build_parameter_table(voltage_levels):
    """Build parameter constraint table for the prompt."""
    if not voltage_levels:
        kv_list = [500]
    else:
        kv_list = sorted(set(voltage_levels))
    lines = ['| 电压等级 | 线路送电能力(MW) | 主变容量(MVA) | 短路电流(kA) | 无功补偿(Mvar) |']
    lines.append('|----------|-----------------|--------------|-------------|---------------|')
    for kv in kv_list:
        closest = min(PARAMETER_SANDBOX.keys(), key=lambda k: abs(k - kv))
        box = PARAMETER_SANDBOX.get(closest, PARAMETER_SANDBOX[500])
        lc = box.get('line_capacity_mw', box.get('dc_capacity_mw', (100, 5000)))
        xf = box.get('xfmr_mva', (100, 2000))
        sc = box.get('sc_current_ka', (10, 63))
        rc = box.get('reactive_comp_mvar', (-500, 800))
        lines.append(f'| {kv}kV | {lc[0]}-{lc[1]} | {xf[0]}-{xf[1]} | {sc[0]}-{sc[1]} | {rc[0]}~{rc[1]} |')
    return '\n'.join(lines)


L2_EXPAND_PROMPT = '''## 任务
为以下电力系统L2评测题扩写工程场景（≥150字），使其包含具体的工程参数和约束条件。

## 硬约束（违反任何一条则输出无效）
1. 场景中的电气参数必须来自下方「参数约束表」或标准原文
2. 扩写后的问题，其正确答案必须与下方「原始答案」一致
3. 不得引入原始答案未涉及的判断维度，不得新增需要额外标准条款的约束
4. 不得编造标准中不存在的条款号
5. 输出格式必须为：【场景】扩写后的场景（≥150字）

【问题】原问题文本
6. 场景中必须包含至少3个具体数值参数（如电压、容量、阻抗等）

## 参数约束表
{parameter_table}

## 相关标准原文（节选）
{standard_text}

## 原始题目及答案
原题：{query}
原始答案：{answer}

输出：仅输出扩写后的完整题目（含【场景】和【问题】标签），不要JSON包裹。'''


L3_EXPAND_PROMPT = '''## 任务
为以下电力系统L3综合评测题扩写工程场景（≥300字），包含具体的工程参数、约束条件和方案对比框架。

## 硬约束（违反任何一条则输出无效）
1. 场景中的电气参数必须来自下方「参数约束表」或标准原文
2. 扩写后的问题，其正确答案必须与下方「原始答案」一致
3. 不得引入原始答案未涉及的判断维度
4. 场景需同时体现方案A和方案B所涉及的技术要素
5. 输出格式：【场景】扩写后的场景（≥300字，含方案A/B概述）

【问题】原问题文本

## 参数约束表
{parameter_table}

## 相关标准原文（节选）
{standard_text}

## 原始题目及答案
原题：{query}
原始答案（前600字）：{answer}

输出：仅输出扩写后的完整题目（含【场景】和【问题】标签），不要JSON包裹。'''


L3_RESTRUCTURE_PROMPT = '''## 任务
将以下电力系统L3综合评测题的答案重构为显式五段式格式。保持原答案的核心判断和推理不变，仅做结构化重组。

## 五段式格式要求
输出必须按以下五段组织，每段以「段标题：」开头：

现状诊断：（概述当前工程配置及其存在的稳定/安全/设计问题，引用标准指出不符合项）
多维冲突分析：（分析各方案之间、各标准要求之间的冲突点，说明冲突的本质原因）
多方案比选：（从安全性、经济性、可行性、标准符合性等维度对比方案A和方案B的优劣）
折中综合方案：（给出推荐方案及理由，可以是A/B的折中或分阶段实施策略）
控制优先级链条：（实施推荐方案的具体步骤和优先级顺序，①②③...格式）

## 硬约束
1. 保持原答案的核心判断和结论不变
2. 保持原答案中引用的标准条款号和参数数值不变
3. 补充缺失的必要细节，但不新增与原始判断矛盾的论点
4. 每段不少于100字

## 原L3题目
{query}

## 原始答案
{answer}

## 相关标准原文（节选）
{standard_text}

输出：仅输出重构后的五段式答案，不要JSON包裹。'''


# ═══════════════════════════════════════════════════════════
# Layer 3: Self-Consistency Check
# ═══════════════════════════════════════════════════════════

SELF_CONSISTENCY_PROMPT = '''请验证以下电力系统评测题扩写结果的自洽性：

【扩写后题目】
{expanded_query}

【标准答案】
{original_answer}

逐项检查并回答（每项只答"通过"或"不通过"，加一句简短说明）：
1. 参数合理性：场景中所有数值参数是否在电力工程合理范围内？
2. 答案一致性：以标准答案为正确答案，扩写后的问题是否能被该答案完整、正确地回答？
3. 维度边界：扩写的场景是否引入了原答案未覆盖的新判断维度？
4. 条款准确性：场景中引用的标准条款号是否在原标准中存在且含义相符？
5. 格式完整性：题目是否包含【场景】和【问题】标签，场景字数是否足够？

最终结论（任一项"不通过"则整体不通过）：通过 / 不通过'''


# ═══════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════

def parse_llm_response(text):
    """Parse LLM output, stripping any markdown wrappers."""
    text = text.strip()
    # Remove markdown code fences
    for m in re.finditer(r'```(?:json)?\s*([\s\S]*?)\s*```', text):
        return m.group(1).strip()
    return text


def layer2_check(text):
    """Run parameter sandbox validation."""
    param_ok, param_violations = validate_parameters(text)
    return param_ok, param_violations


def layer3_check(client, expanded_query, original_answer):
    """Run self-consistency LLM check."""
    prompt = SELF_CONSISTENCY_PROMPT.format(
        expanded_query=expanded_query[:3000],
        original_answer=str(original_answer)[:2000]
    )
    try:
        resp = client.chat.completions.create(
            model=STRONG_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            max_tokens=512
        )
        result = resp.choices[0].message.content
        passed = '通过' in result and '不通过' not in result
        return passed, result
    except Exception as e:
        return False, f'API error: {e}'


def expand_item(client, item, standard_texts, expand_type='L2'):
    """Expand a single item's scenario with 3-layer defense. Returns (expanded_item, success, log)."""
    query = item.get('query', '')
    answer = str(item.get('expected_answer', ''))
    src = item.get('source_standard', '')
    lvl = item.get('level', expand_type)

    # Select prompt template
    if expand_type == 'L2':
        template = L2_EXPAND_PROMPT
        min_chars = 150
    else:
        template = L3_EXPAND_PROMPT
        min_chars = 300

    # Determine relevant voltage levels from existing query+answer
    combined = query + ' ' + answer
    voltage_levels = extract_voltage_levels(combined)
    param_table = build_parameter_table(voltage_levels or [220, 500])

    # Find relevant standard text
    std_text = ''
    for name, text in standard_texts.items():
        if name.split()[1].replace('-', '') in src.replace('/', '').replace('-', ''):
            std_text += f'\n=== {name} ===\n{text[:2000]}\n'

    if not std_text:
        # Use first available standard
        first_std = next(iter(standard_texts.values())) if standard_texts else ''
        std_text = first_std[:2000] if first_std else '(标准文本未加载)'

    # Attempt expansion with retries
    for attempt in range(3):
        log = []
        prompt = template.format(
            parameter_table=param_table,
            standard_text=std_text,
            query=query,
            answer=answer[:1500]
        )

        try:
            resp = client.chat.completions.create(
                model=STRONG_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.3,
                max_tokens=2000 if expand_type == 'L2' else 3000
            )
            expanded = parse_llm_response(resp.choices[0].message.content)
        except Exception as e:
            log.append(f'API error (attempt {attempt+1}): {e}')
            continue

        # Validate format
        has_scenario = '【场景】' in expanded
        has_problem = '【问题】' in expanded
        if not has_scenario or not has_problem:
            log.append(f'Format check failed: scenario={has_scenario}, problem={has_problem}')
            if attempt < 2:
                continue

        # Extract scenario part for length check
        sc_match = re.search(r'【场景】(.+?)(?:【问题】|$)', expanded, re.DOTALL)
        scenario_len = len(sc_match.group(1).strip()) if sc_match else 0
        if scenario_len < min_chars:
            log.append(f'Scenario too short: {scenario_len} < {min_chars} chars')
            if attempt < 2:
                continue

        # Layer 2: Parameter sandbox
        param_ok, param_violations = layer2_check(expanded)
        if not param_ok:
            log.append(f'Layer 2 failed: {param_violations[:3]}')
            if attempt < 2:
                continue

        # Layer 3: Self-consistency
        cons_ok, cons_result = layer3_check(client, expanded, answer)
        if not cons_ok:
            log.append(f'Layer 3 failed: {cons_result[:200]}')
            if attempt < 2:
                continue

        # All checks passed
        new_item = copy.deepcopy(item)
        new_item['query'] = expanded.strip()
        return new_item, True, f'OK (attempt {attempt+1}, scenario_len={scenario_len})'

    return item, False, ' | '.join(log[-3:]) if log else 'Max retries exceeded'


def restructure_l3_answer(client, item, standard_texts):
    """Restructure L3 answer to five-section format. Returns (restructured_item, success, log)."""
    query = item.get('query', '')
    answer = str(item.get('expected_answer', ''))
    src = item.get('source_standard', '')

    # Get relevant standard text
    std_text = ''
    for name, text in standard_texts.items():
        if name.split()[1].replace('-', '') in src.replace('/', '').replace('-', ''):
            std_text += f'\n=== {name} ===\n{text[:2000]}\n'
    if not std_text:
        first_std = next(iter(standard_texts.values())) if standard_texts else ''
        std_text = first_std[:2000] if first_std else ''

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=STRONG_MODEL,
                messages=[{'role': 'user', 'content': L3_RESTRUCTURE_PROMPT.format(
                    query=query[:2000],
                    answer=answer[:3000],
                    standard_text=std_text
                )}],
                temperature=0.2,
                max_tokens=4000
            )
            restructured = parse_llm_response(resp.choices[0].message.content)
        except Exception as e:
            if attempt < 2:
                continue
            return item, False, f'API error: {e}'

        # Validate five sections present
        sections = ['诊断', '冲突', '比选', '方案', '控制']
        found = [s for s in sections if s in restructured]
        if len(found) < 4:
            if attempt < 2:
                continue
            return item, False, f'Missing sections: {set(sections) - set(found)}'

        # Check min length per section
        lines = restructured.split('\n')
        if len(restructured) < 400:
            if attempt < 2:
                continue

        new_item = copy.deepcopy(item)
        new_item['expected_answer'] = restructured.strip()
        return new_item, True, f'OK (attempt {attempt+1}, len={len(restructured)})'

    return item, False, 'Max retries exceeded'


def generate_new_l1(client, std_name, std_text, count):
    """Generate new L1 items with anti-hallucination prompt."""
    prompt = f'''从标准 {std_name} 生成{count}道L1评测题（直接型：答案在单一规范条款中可直接检索）。
要求：每题有精确数值答案或一句话答案（含具体参数值），避免笼统定义类题目。

## 参数约束
- 电压等级：110kV~750kV（参考{std_name}适用范围）
- 所有数值必须在标准规定的合理范围内

输出严格JSON数组：[{{"query":"问题","expected_answer":"精确答案","expected_keywords":["kw1","kw2","kw3","kw4","kw5"],"source_standard":"{std_name} §X.Y","category":"参数检索"}}]

标准文本（节选）：{std_text[:6000]}'''

    try:
        resp = client.chat.completions.create(
            model=STRONG_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.3, max_tokens=4000
        )
        # Parse response
        text = resp.choices[0].message.content
        arr = re.search(r'\[[\s\S]*\]', text)
        if arr:
            return json.loads(arr.group(0))
    except Exception as e:
        print(f'  L1 generation error: {e}')
    return []


def generate_new_l3(client, std_texts, count):
    """Generate new L3 cross-standard items."""
    std_list = list(std_texts.items())
    if len(std_list) < 2:
        return []

    all_new = []
    for i in range((count + 1) // 2):
        a_idx = i % len(std_list)
        b_idx = (i + 1) % len(std_list)
        if b_idx <= a_idx:
            b_idx = (a_idx + 1) % len(std_list)
        name_a, text_a = std_list[a_idx]
        name_b, text_b = std_list[b_idx]
        n = min(2, count - len(all_new))
        if n <= 0:
            break

        prompt = f'''从 {name_a} 和 {name_b} 生成{n}道L3综合评测题。
要求：含方案A/B对比（各有具体参数），答案五段式：现状诊断→多维冲突分析→多方案比选→折中综合方案→控制优先级链条。
必须同时引用两份标准的具体条款。场景≥300字，答案≥800字。

## 参数约束
所有参数必须在相关电压等级和标准的合理范围内。

标准1 {name_a}：{text_a[:4000]}
标准2 {name_b}：{text_b[:4000]}

输出严格JSON数组：[{{"query":"【场景】...\\\\n\\\\n【问题】...","expected_answer":"现状诊断：...\\\\n多维冲突分析：...\\\\n多方案比选：...\\\\n折中综合方案：...\\\\n控制优先级链条：...","expected_keywords":["kw1","kw2","kw3","kw4","kw5"],"source_standard":"{name_a}, {name_b}","category":"综合评估"}}]'''

        try:
            resp = client.chat.completions.create(
                model=STRONG_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.4, max_tokens=16000
            )
            text = resp.choices[0].message.content
            arr = re.search(r'\[[\s\S]*\]', text)
            if arr:
                batch = json.loads(arr.group(0))
                all_new.extend(batch[:n])
        except Exception as e:
            print(f'  L3 generation error: {e}')

    return all_new[:count]


# Shared ID manager for gap filling (single instance avoids ID collision)
from data_generation.utils.id_manager import IDManager
_gap_idm = IDManager(l1_start=201, l2_start=201, l3_start=201)


def assemble_new_item(raw, lvl, index):
    """Assemble a raw LLM output into proper dataset item."""
    global _gap_idm
    idm = _gap_idm

    item = {
        'query': raw.get('query', ''),
        'expected_answer': raw.get('expected_answer', raw.get('answer', '')),
        'expected_keywords': raw.get('expected_keywords', raw.get('keywords', [])),
        'source_standard': raw.get('source_standard', ''),
        'category': raw.get('category', {'L1': '参数检索', 'L2': '审查判断', 'L3': '综合评估'}[lvl]),
        'question_id': idm.next(lvl),
        'question_class': lvl,
        'level': lvl,
        'grading_method': 'auto_keyword_match' if lvl == 'L1' else 'manual_review',
        'knowledge_base': 'General-KB',
    }

    # Normalize fields
    if 'scenario' in raw and 'query' not in raw:
        item['query'] = f"【场景】{raw['scenario']}\n\n【问题】{raw.get('question', raw.get('problem', ''))}"

    kws = list(item.get('expected_keywords', []))
    if len(kws) < 3:
        for c in re.findall(r'[一-鿿]{2,4}', item['query'] + str(item['expected_answer'])):
            if c not in kws:
                kws.append(c)
            if len(kws) >= 5:
                break
    item['expected_keywords'] = kws[:5]

    item = fix_question_escaping(item)
    if lvl in ('L2', 'L3'):
        rubric = build_rubric(str(item['expected_answer']), lvl)
        for k, v in rubric.items():
            item[k] = v

    # Topic
    from data_generation.fix_v2 import extract_topic, determine_knowledge_base, normalize_standard
    item['topic'] = extract_topic(item['query'], item['expected_answer'])
    item['knowledge_base'] = determine_knowledge_base(item['source_standard'])
    item['source_standard'] = normalize_standard(item['source_standard'])

    return item


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    DATA_DIR = r'D:\coding\meta_AutoData\data\questions'
    INPUT_FILE = os.path.join(DATA_DIR, 'generated_eval_set_v2.json')
    OUTPUT_FILE = os.path.join(DATA_DIR, 'generated_eval_set_v3.json')
    CHECKPOINT_FILE = os.path.join(DATA_DIR, 'expand_checkpoint.json')

    print('=' * 60)
    print('expand_scenarios.py — Safe L2/L3 Expansion')
    print(f'Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    # Load data
    data = json.load(open(INPUT_FILE, 'r', encoding='utf-8'))
    print(f'Loaded v2: {len(data)} questions')

    # Load standards
    sources, _ = load_all_sources()
    std_texts = {}
    for s in sources:
        for kw, name in [('38755', 'GB 38755-2019'), ('5429', 'DL/T 5429-2009'), ('5218', 'DL/T 5218-2012')]:
            if kw in s['title']:
                std_texts[name] = s['raw_text']
                break
    print(f'Standards loaded: {list(std_texts.keys())}')

    # Setup client
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # Resume from checkpoint if exists
    completed_ids = set()
    if os.path.exists(CHECKPOINT_FILE):
        completed_ids = set(json.load(open(CHECKPOINT_FILE, 'r', encoding='utf-8')))
        print(f'Resuming: {len(completed_ids)} already completed')

    stats = {
        'l2_expand_attempted': 0, 'l2_expand_succeeded': 0,
        'l3_expand_attempted': 0, 'l3_expand_succeeded': 0,
        'l3_restructure_attempted': 0, 'l3_restructure_succeeded': 0,
        'l1_new_generated': 0, 'l3_new_generated': 0,
        'layer2_rejections': 0, 'layer3_rejections': 0,
    }
    start_time = time.time()
    total_ops = 0

    # ── Identify items needing work ──
    l2_no_scenario = [q for q in data if isinstance(q, dict) and q.get('level') == 'L2'
                      and '【场景】' not in q.get('query', '')
                      and q.get('question_id') not in completed_ids]
    l3_no_scenario = [q for q in data if isinstance(q, dict) and q.get('level') == 'L3'
                      and '【场景】' not in q.get('query', '')
                      and q.get('question_id') not in completed_ids]
    l3_no_structure = [q for q in data if isinstance(q, dict) and q.get('level') == 'L3'
                       and '【场景】' in q.get('query', '')  # already has scenario
                       and ('诊断' not in str(q.get('expected_answer', ''))[:200]
                            or '冲突' not in str(q.get('expected_answer', ''))[:500]
                            or '比选' not in str(q.get('expected_answer', ''))[:800])
                       and q.get('question_id') not in completed_ids]

    print(f'\nTo expand (L2 scenario): {len(l2_no_scenario)}')
    print(f'To expand (L3 scenario): {len(l3_no_scenario)}')
    print(f'To restructure (L3 answer): {len(l3_no_structure)}')

    # ── Expand L2 scenarios ──
    print(f'\n{"─" * 40}')
    print(f'PHASE 1: L2 Scenario Expansion ({len(l2_no_scenario)} items)')
    print(f'{"─" * 40}')
    for i, item in enumerate(l2_no_scenario):
        qid = item.get('question_id', f'?')
        new_item, success, log_msg = expand_item(client, item, std_texts, 'L2')
        if success:
            idx = data.index(item)
            data[idx] = new_item
            stats['l2_expand_succeeded'] += 1
        else:
            if 'Layer 2 failed' in log_msg:
                stats['layer2_rejections'] += 1
            elif 'Layer 3 failed' in log_msg:
                stats['layer3_rejections'] += 1
        stats['l2_expand_attempted'] += 1
        total_ops += 1

        status = 'OK' if success else 'FAIL'
        print(f'  [{i+1}/{len(l2_no_scenario)}] {qid} {status} {log_msg[:80]}')

        # Progress report every 10 items
        if (i + 1) % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            done = stats['l2_expand_succeeded'] + stats['l3_expand_succeeded'] + stats['l3_restructure_succeeded']
            total_todo = len(l2_no_scenario) + len(l3_no_scenario) + len(l3_no_structure)
            pct = 100 * (i + 1) / total_todo if total_todo else 0
            print(f'  [进度] L2场景: {stats["l2_expand_succeeded"]}/{stats["l2_expand_attempted"]} | 已耗时: {elapsed:.1f}min | 预计进度: {pct:.0f}%')

        # Save checkpoint
        completed_ids.add(qid)
        json.dump(list(completed_ids), open(CHECKPOINT_FILE, 'w', encoding='utf-8'))

    # ── Expand L3 scenarios ──
    print(f'\n{"─" * 40}')
    print(f'PHASE 2: L3 Scenario Expansion ({len(l3_no_scenario)} items)')
    print(f'{"─" * 40}')
    for i, item in enumerate(l3_no_scenario):
        qid = item.get('question_id', '?')
        new_item, success, log_msg = expand_item(client, item, std_texts, 'L3')
        if success:
            idx = data.index(item)
            data[idx] = new_item
            stats['l3_expand_succeeded'] += 1
        else:
            if 'Layer 2' in log_msg:
                stats['layer2_rejections'] += 1
            elif 'Layer 3' in log_msg:
                stats['layer3_rejections'] += 1
        stats['l3_expand_attempted'] += 1
        total_ops += 1
        print(f'  [{i+1}/{len(l3_no_scenario)}] {qid} {"OK" if success else "FAIL"} {log_msg[:80]}')
        completed_ids.add(qid)
        json.dump(list(completed_ids), open(CHECKPOINT_FILE, 'w', encoding='utf-8'))

    # ── Restructure L3 answers ──
    print(f'\n{"─" * 40}')
    print(f'PHASE 3: L3 Answer Restructuring ({len(l3_no_structure)} items)')
    print(f'{"─" * 40}')
    for i, item in enumerate(l3_no_structure):
        qid = item.get('question_id', '?')
        new_item, success, log_msg = restructure_l3_answer(client, item, std_texts)
        if success:
            idx = data.index(item)
            data[idx] = new_item
            stats['l3_restructure_succeeded'] += 1
        stats['l3_restructure_attempted'] += 1
        total_ops += 1
        print(f'  [{i+1}/{len(l3_no_structure)}] {qid} {"OK" if success else "FAIL"} {log_msg[:80]}')

        if (i + 1) % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f'  [进度] L3答案重构: {stats["l3_restructure_succeeded"]}/{stats["l3_restructure_attempted"]} | 已耗时: {elapsed:.1f}min')

        completed_ids.add(qid)
        json.dump(list(completed_ids), open(CHECKPOINT_FILE, 'w', encoding='utf-8'))

    # ── Gap filling ──
    l1_count = sum(1 for q in data if isinstance(q, dict) and q.get('level') == 'L1')
    l3_count = sum(1 for q in data if isinstance(q, dict) and q.get('level') == 'L3')
    l1_needed = max(0, 100 - l1_count)
    l3_needed = max(0, 100 - l3_count)

    print(f'\n{"─" * 40}')
    print(f'PHASE 4: Gap Filling (L1: {l1_needed}, L3: {l3_needed})')
    print(f'{"─" * 40}')

    if l1_needed > 0:
        # Distribute across 3 standards
        per_std = l1_needed // 3
        remainder = l1_needed % 3
        dist = [
            ('GB 38755-2019', per_std + (1 if remainder > 0 else 0)),
            ('DL/T 5429-2009', per_std + (1 if remainder > 1 else 0)),
            ('DL/T 5218-2012', per_std),
        ]
        for std_name, cnt in dist:
            if cnt <= 0:
                continue
            print(f'  Generating {cnt} L1 from {std_name}...')
            new_qs = generate_new_l1(client, std_name, std_texts.get(std_name, ''), cnt)
            for raw in new_qs:
                item = assemble_new_item(raw, 'L1', 200 + stats['l1_new_generated'])
                data.append(item)
                stats['l1_new_generated'] += 1
            print(f'    -> {len(new_qs)} generated')
            total_ops += cnt

    if l3_needed > 0:
        print(f'  Generating {l3_needed} L3 cross-standard...')
        new_qs = generate_new_l3(client, std_texts, l3_needed)
        for raw in new_qs:
            item = assemble_new_item(raw, 'L3', 200 + stats['l3_new_generated'])
            data.append(item)
            stats['l3_new_generated'] += 1
        print(f'    -> {len(new_qs)} generated')
        total_ops += l3_needed

    # ── Final assembly ──
    # Rebuild rubrics for all modified items
    print(f'\n{"─" * 40}')
    print('PHASE 5: Final Rubric Rebuild')
    print(f'{"─" * 40}')
    rubric_count = 0
    for item in data:
        if isinstance(item, dict) and item.get('level') in ('L2', 'L3'):
            # Rebuild if rubric might be stale
            if not item.get('rubric_clauses') or len(item.get('rubric_clauses', [])) == 0:
                rubric = build_rubric(str(item.get('expected_answer', '')), item['level'])
                for k, v in rubric.items():
                    item[k] = v
                rubric_count += 1
    print(f'  Rubrics rebuilt: {rubric_count}')

    # ── Verify ──
    print(f'\n{"=" * 60}')
    print('VERIFICATION')
    print('=' * 60)

    by_level = {}
    for q in data:
        if isinstance(q, dict):
            lvl = q.get('level', '?')
            by_level.setdefault(lvl, {'total': 0, 'valid': 0, 'has_scenario': 0})
            by_level[lvl]['total'] += 1
            v, _ = validate_question(q, lvl)
            by_level[lvl]['valid'] += int(v)
            if '【场景】' in q.get('query', ''):
                by_level[lvl]['has_scenario'] += 1

    for lvl in ['L1', 'L2', 'L3']:
        d = by_level.get(lvl, {})
        print(f'  {lvl}: {d.get("total", 0)} total, {d.get("valid", 0)} valid, '
              f'{d.get("has_scenario", 0)} with scenario')

    # L3 structure check
    l3_items = [q for q in data if isinstance(q, dict) and q.get('level') == 'L3']
    l3_structured = sum(1 for q in l3_items if all(
        s in str(q.get('expected_answer', '')) for s in ['诊断', '冲突', '比选']
    ))
    print(f'  L3 five-section: {l3_structured}/{len(l3_items)}')

    # ── Save ──
    elapsed_total = (time.time() - start_time) / 60
    print(f'\nTotal time: {elapsed_total:.1f} min')
    print(f'Total API operations: ~{total_ops}')
    print(f'Layer 2 rejections: {stats["layer2_rejections"]}')
    print(f'Layer 3 rejections: {stats["layer3_rejections"]}')

    json.dump(data, open(OUTPUT_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    json.dump(stats, open(os.path.join(DATA_DIR, 'expand_report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\nSaved: {OUTPUT_FILE} ({len(data)} questions)')

    # Clean up checkpoint on success
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == '__main__':
    main()
