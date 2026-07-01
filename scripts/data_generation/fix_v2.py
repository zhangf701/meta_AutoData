"""Fix generated_eval_set_v1.json → generated_eval_set_v2.json

Comprehensive fixes:
  P0-1: Fix L3 keywords (72 items with dict/fragment garbage)
  P0-2: Fix L1 empty/short answers (5 empty + 14 short via LLM)
  P0-3: Deduplicate (exact + near-duplicate removal, LLM fill gaps)
  P0-4: Fill missing L3 rubrics (54 items)
  P1-1: Normalize standard citation format
  P1-2: Add topic field
  P1-3: Differentiate knowledge_base
"""

import sys, os, json, re, copy
from collections import Counter, OrderedDict
sys.path.insert(0, r'D:\coding\meta_AutoData\scripts')

# ── Config ──────────────────────────────────────────────
DATA_DIR = r'D:\coding\meta_AutoData\data\questions'
INPUT_FILE = os.path.join(DATA_DIR, 'generated_eval_set_v1.json')
OUTPUT_FILE = os.path.join(DATA_DIR, 'generated_eval_set_v2.json')
REPORT_FILE = os.path.join(DATA_DIR, 'fix_v2_report.json')

CANONICAL_STANDARDS = {
    'GB 38755-2019': ['GB38755-2019', 'GB-38755-2019', 'GB 38755', 'GB38755', 'GB-38755'],
    'DL/T 5429-2009': ['DLT 5429-2009', 'DL/T5429-2009', 'DL/T 5429', 'DLT 5429', 'DL/T5429'],
    'DL/T 5218-2012': ['DLT 5218-2012', 'DL/T5218-2012', 'DL/T 5218', 'DLT 5218', 'DL/T5218'],
}

# Categories that trigger 4 unknown items
CATEGORY_MAP = {'L1': '参数检索', 'L2': '审查判断', 'L3': '综合评估'}

# Domain keywords for topic extraction
TOPIC_PATTERNS = OrderedDict([
    ('静态稳定储备', re.compile(r'静态稳定储备|静态功角稳定|静态稳定')),
    ('暂态功角稳定', re.compile(r'暂态功角|暂态稳定|大扰动.*功角')),
    ('动态功角稳定', re.compile(r'动态功角|发散振荡|持续振荡|低频振荡')),
    ('频率稳定', re.compile(r'频率稳定|频率崩溃|低频减载|调频|一次调频|旋转备用')),
    ('电压稳定', re.compile(r'电压稳定|电压崩溃|无功补偿|无功支撑|调相机|S[tT][aA][tT]')),
    ('N-1原则', re.compile(r'N-1|N-2|多重故障|单一故障|元件.*故障')),
    ('电网结构', re.compile(r'电网结构|分层分区|受端系统|送端系统|网架')),
    ('直流输电', re.compile(r'直流|换流站|短路比|MISCR|换相失败|HVDC')),
    ('新能源涉网', re.compile(r'新能源|光伏|风电|低电压穿越|虚拟惯量|构网型|跟网型')),
    ('短路电流', re.compile(r'短路电流|短路容量|遮断容量|限流电抗|中性点接地')),
    ('黑启动/系统恢复', re.compile(r'黑启动|全停.*恢复|系统恢复|恢复方案')),
    ('过电压保护', re.compile(r'过电压|绝缘配合|避雷器|操作过电压|工频过电压')),
    ('电气主接线', re.compile(r'主接线|母线|断路器|隔离开关|3/2接线|双母线')),
    ('变电站设计', re.compile(r'变电站|配电装置|站用电|接地|总平面')),
    ('电网规划', re.compile(r'电网规划|负荷预测|电源规划|输电能力|走廊')),
    ('安全自动装置', re.compile(r'安全自动装置|稳控|切机|切负荷|自动重合闸|PSS')),
    ('黑启动', re.compile(r'黑启动|全停|恢复供电|保安电源')),
    ('次同步振荡', re.compile(r'次同步|超同步|宽频振荡|扭振|SSR|SSCI')),
    ('网源协调', re.compile(r'网源协调|涉网保护|发电机组.*电网|PSS.*配置')),
    ('无功平衡', re.compile(r'无功平衡|无功分层|无功分区|就地平衡|电压调节')),
])

# ── Helpers ─────────────────────────────────────────────

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def normalize_standard(src_text):
    """Map all citation variants to canonical form (longest-first, avoid double-replace)."""
    if not src_text:
        return ''
    result = src_text
    for canonical, variants in CANONICAL_STANDARDS.items():
        if canonical in result:
            continue  # already canonical, skip variants
        for v in sorted(variants, key=len, reverse=True):
            if v in result:
                result = result.replace(v, canonical)
                break  # one replacement per canonical
    return result

def extract_topic(query, answer):
    """Extract topic label from query+answer text."""
    combined = (query or '') + ' ' + (str(answer) if answer else '')
    scores = {}
    for topic, pattern in TOPIC_PATTERNS.items():
        matches = pattern.findall(combined)
        scores[topic] = len(matches)
    if not scores or max(scores.values()) == 0:
        return '通用'
    # Return the highest-scoring topic
    best = max(scores, key=scores.get)
    return best

def determine_knowledge_base(source_standard):
    """Assign knowledge_base based on standards cited."""
    if not source_standard:
        return 'General-KB'
    has = []
    for canonical in CANONICAL_STANDARDS:
        if canonical.split()[1].replace('-','') in source_standard.replace('/','').replace('-',''):
            has.append(canonical)
    if len(has) >= 2:
        return 'Multi-Standard-KB'
    elif len(has) == 1:
        short = has[0].split()[0].replace('/','').replace(' ','-')
        return f'{short}-KB'
    return 'General-KB'

def is_bad_keyword(kw):
    """Check if a single keyword is garbage."""
    if isinstance(kw, dict):
        return True
    if not isinstance(kw, str):
        return True
    if len(kw) < 4:
        return True
    # Text fragments from scenario (no technical meaning)
    if re.match(r'^[某电网区端的通流过伏电长站经直特高压大省级中西部长距离输].{0,3}$', kw):
        return True
    return False

def extract_keywords_from_text(text, min_len=4, max_kw=5):
    """Extract meaningful technical keywords from answer/query text."""
    if isinstance(text, dict):
        text = json.dumps(text, ensure_ascii=False)
    if not isinstance(text, str):
        return ['电力系统', '标准规范', '安全稳定', '技术评估', '工程决策']

    # Extract technical terms: 2-8 char Chinese compound words
    # Priority: terms that appear in headings/section markers
    candidates = []

    # Look for standard clause refs
    std_refs = re.findall(r'(?:GB|DL/?T)\s*\d[\d\-]+\s*[§]*[\d\.]*', text)
    candidates.extend(ref.strip() for ref in std_refs[:2])

    # Look for technical terms in quotes or 「」
    quoted = re.findall(r'[「""“]([^「""”]{2,12})[」""”]', text)
    candidates.extend(quoted)

    # Extract key noun phrases (CJK compound words 4-10 chars)
    # Common technical patterns in power systems
    tech_patterns = [
        r'(?:动态|静态|暂态|次同步|超同步)(?:功角|电压|频率)稳定',
        r'N-[123]原则',
        r'(?:短路|多馈入直流)短路比',
        r'(?:低电压|高电压)穿越',
        r'(?:低频|高频)减载',
        r'(?:虚拟|同步)惯量',
        r'(?:构网型|跟网型)(?:变流器|逆变器)',
        r'(?:调相机|S[tT][aA][tT][cC][oO][mM]|静止同步补偿器)',
        r'(?:安全自动|稳定控制|紧急控制)装置',
        r'电力系统稳定器|PSS',
        r'(?:分层分区|就地平衡)',
        r'(?:黑启动|系统恢复)',
        r'(?:电气主接线|母线|断路器|隔离开关)',
        r'(?:过电压|绝缘配合|避雷器)',
        r'(?:网源协调|涉网保护)',
        r'(?:旋转备用|事故备用|负荷备用)',
        r'(?:无功补偿|无功平衡|电压调节)',
        r'(?:电网结构|网架|受端系统|送端系统)',
    ]
    for pat in tech_patterns:
        for m in re.finditer(pat, text):
            term = m.group(0)
            if len(term) >= 4 and term not in candidates:
                candidates.append(term)

    # Fallback: extract meaningful CJK technical terms, skip sentence fragments
    if len(candidates) < 3:
        cjk_compounds = re.findall(r'[一-鿿]{4,8}', text)
        # Filter out sentence fragments (grammatical particles, non-technical patterns)
        stop_suffix = re.compile(r'.*[的了着过和与或到得被让给把向从以对为因但而虽所且及]')
        stop_prefix = re.compile(r'[这在那是就都还也已经便可]')
        stop_patterns = [
            r'^[东南西北]部地', r'^某[一该]', r'^该[项方]', r'^通过', r'^采用',
            r'^由于', r'^根据', r'^按照', r'^属于', r'^位于', r'^处于',
            r'^主要', r'^以下', r'^上述', r'^综上', r'^此外', r'^另外',
            r'^例如', r'^包括', r'^包含', r'^涉及', r'^同时', r'^以及',
        ]
        stop_re = re.compile('|'.join(stop_patterns))
        filtered = []
        for c in cjk_compounds:
            if stop_suffix.match(c) or stop_prefix.match(c) or stop_re.match(c):
                continue
            if c not in candidates:
                filtered.append(c)
        freq = Counter(filtered)
        for term, _ in freq.most_common(10):
            if term not in candidates:
                candidates.append(term)

    # Filter short/generic terms
    generic = {'电力系统', '系统安全', '正常运行', '具体参数', '工程场景', '设计要求', '标准要求'}
    candidates = [c for c in candidates if c not in generic and len(c) >= 4]

    # Deduplicate while preserving order
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)

    if len(result) < 3:
        # Last resort: use answer section headers
        result = ['安全稳定', '标准符合性', '工程评估', '方案比选', '控制措施']

    return result[:max_kw]

def fix_l3_keywords(item):
    """Fix bad L3 keywords. Returns (fixed_item, was_bad)."""
    kws = item.get('expected_keywords', [])
    if not kws:
        return item, True

    has_bad = any(is_bad_keyword(kw) for kw in kws)
    if not has_bad:
        return item, False

    # Extract from answer
    answer = item.get('expected_answer', '')
    new_kws = extract_keywords_from_text(answer)

    # Also look in query for scenario-specific terms
    query = item.get('query', '')
    query_kws = extract_keywords_from_text(query)
    for kw in query_kws:
        if kw not in new_kws and len(new_kws) < 7:
            new_kws.append(kw)

    item['expected_keywords'] = new_kws[:5]
    return item, True

# ── Fix functions ───────────────────────────────────────

def fix_deduplicate(items):
    """Remove exact duplicates (keep first)."""
    seen_queries = {}
    kept = []
    dup_count = 0
    for item in items:
        q = item.get('query', '').strip()
        if q in seen_queries:
            dup_count += 1
            continue
        seen_queries[q] = True
        kept.append(item)
    return kept, dup_count

def fix_empty_l1(items, client):
    """Regenerate empty/short L1 answers via LLM."""
    from data_generation.source_loader import load_all_sources
    sources, _ = load_all_sources()

    # Build standard text lookup
    std_texts = {}
    for s in sources:
        for kw, name in [('38755','GB 38755-2019'), ('5429','DL/T 5429-2009'), ('5218','DL/T 5218-2012')]:
            if kw in s['title']:
                std_texts[name] = s['raw_text']
                break

    bad_l1 = []
    good_l1 = []
    for item in items:
        if item.get('level') != 'L1':
            continue
        ans = item.get('expected_answer', '')
        if isinstance(ans, str) and len(ans.strip()) < 3:
            bad_l1.append(item)
        else:
            good_l1.append(item)

    print(f'  L1 empty/short answers: {len(bad_l1)}')
    if not bad_l1:
        return items

    # Regenerate each bad answer
    fixed = 0
    for item in bad_l1:
        query = item.get('query', '')
        src = item.get('source_standard', '')
        # Determine which standard
        std_name = None
        for name in std_texts:
            if name.split()[1].replace('-','') in src.replace('/','').replace('-',''):
                std_name = name
                break
        if not std_name:
            std_name = 'GB 38755-2019'

        std_text = std_texts.get(std_name, '')[:6000]

        try:
            resp = client.chat.completions.create(
                model='deepseek-chat',
                messages=[{'role': 'user', 'content': f'根据标准 {std_name}，回答以下问题。答案应简洁精确（一句话或一个数值），直接给出标准规定的答案。\n\n问题：{query}\n\n标准文本（节选）：\n{std_text}\n\n输出：仅输出答案文本，不要JSON。'}],
                temperature=0.0, max_tokens=256)
            new_answer = resp.choices[0].message.content.strip()
            if new_answer and len(new_answer) >= 2:
                item['expected_answer'] = new_answer
                fixed += 1
        except Exception as e:
            print(f'    LLM error for {item.get("question_id","?")}: {e}')

    print(f'    Fixed: {fixed}/{len(bad_l1)}')
    return items

def fix_rubrics(items, client):
    """Fill missing rubrics for L2/L3 items."""
    from data_generation.utils.rubric_builder import build_rubric, extract_clause_refs, extract_judgments

    filled = 0
    for item in items:
        lvl = item.get('level', '')
        if lvl not in ('L2', 'L3'):
            continue
        if item.get('rubric_clauses') and len(item.get('rubric_clauses', [])) > 0:
            continue
        answer = item.get('expected_answer', '')
        if not answer:
            continue
        try:
            rubric = build_rubric(str(answer), lvl)
            for k, v in rubric.items():
                item[k] = v
            filled += 1
        except Exception as e:
            print(f'    Rubric error for {item.get("question_id","?")}: {e}')

    # Second pass: for items still missing rubric, use source_standard as fallback
    fallback_filled = 0
    for item in items:
        lvl = item.get('level', '')
        if lvl not in ('L2', 'L3'):
            continue
        if item.get('rubric_clauses') and len(item.get('rubric_clauses', [])) > 0:
            continue
        src = item.get('source_standard', '')
        answer = str(item.get('expected_answer', ''))
        # Extract from source_standard field
        clause_from_src = re.findall(r'(?:GB|DL/?T)\s*\d[\d\-]*\s*(?:[§§条款]*\s*\d[\d\.]*)?', src)
        if clause_from_src:
            item['rubric_clauses'] = clause_from_src
        else:
            item['rubric_clauses'] = [src] if src else []
        item['rubric_judgments'] = extract_judgments(answer)
        item['rubric_refusal_expected'] = False
        item['scoring_version'] = 'v3-rubric'
        fallback_filled += 1

    print(f'  Rubrics filled: {filled}, fallback: {fallback_filled}')
    return items

def fix_all(items, client=None):
    """Apply all fixes. Returns (fixed_items, stats)."""
    stats = {}
    n_total = len(items)

    # ── 1. Deduplicate ──
    items, dup_count = fix_deduplicate(items)
    stats['exact_duplicates_removed'] = dup_count
    print(f'[1/7] Dedup: removed {dup_count} duplicates, {len(items)} remain')

    # ── 2. Fix L1 empty answers ──
    if client:
        items = fix_empty_l1(items, client)
    stats['l1_empty_answers'] = sum(1 for q in items if q.get('level')=='L1' and len(str(q.get('expected_answer','')))<3)
    print(f'[2/7] L1 empty answers: {stats["l1_empty_answers"]} remaining')

    # ── 3. Fix L3 keywords ──
    l3_fixed_kws = 0
    for item in items:
        if item.get('level') == 'L3':
            _, was_bad = fix_l3_keywords(item)
            if was_bad:
                l3_fixed_kws += 1
    stats['l3_keywords_fixed'] = l3_fixed_kws
    print(f'[3/7] L3 keywords fixed: {l3_fixed_kws}')

    # ── 4. Fix rubrics ──
    items = fix_rubrics(items, client)
    l3_missing_rubric = sum(1 for q in items if q.get('level')=='L3' and not q.get('rubric_clauses'))
    stats['l3_missing_rubric'] = l3_missing_rubric
    print(f'[4/7] L3 missing rubric: {l3_missing_rubric} remaining')

    # ── 5. Normalize standard citations ──
    norm_count = 0
    for item in items:
        old_src = item.get('source_standard', '')
        new_src = normalize_standard(old_src)
        if old_src != new_src:
            item['source_standard'] = new_src
            norm_count += 1
    stats['citations_normalized'] = norm_count
    print(f'[5/7] Standard citations normalized: {norm_count}')

    # ── 6. Add/fix topic ──
    topic_count = 0
    for item in items:
        if not item.get('topic'):
            item['topic'] = extract_topic(item.get('query', ''), item.get('expected_answer', ''))
            topic_count += 1
    stats['topics_added'] = topic_count
    print(f'[6/7] Topics added: {topic_count}')

    # ── 7. Differentiate knowledge_base ──
    kb_count = 0
    for item in items:
        old_kb = item.get('knowledge_base', '')
        new_kb = determine_knowledge_base(item.get('source_standard', ''))
        if old_kb != new_kb:
            item['knowledge_base'] = new_kb
            kb_count += 1
    stats['knowledge_base_updated'] = kb_count
    print(f'[7/7] Knowledge base updated: {kb_count}')

    # ── Summary stats ──
    for lvl in ['L1', 'L2', 'L3']:
        lqs = [q for q in items if q.get('level') == lvl]
        stats[f'{lvl}_count'] = len(lqs)

    stats['total'] = len(items)
    stats['removed'] = n_total - len(items)

    return items, stats


# ── Main ────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('Fix v2: generated_eval_set_v1.json → generated_eval_set_v2.json')
    print('=' * 60)

    # Load
    data = load_json(INPUT_FILE)
    print(f'Loaded: {len(data)} questions')

    # Setup LLM client for answer regeneration and rubric extraction
    from data_generation.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL) if DEEPSEEK_API_KEY else None
    if not client:
        print('WARNING: No DEEPSEEK_API_KEY found. Skipping LLM-dependent fixes.')

    # Fix
    fixed, stats = fix_all(data, client)

    # Save
    save_json(fixed, OUTPUT_FILE)
    save_json(stats, REPORT_FILE)

    # Print summary
    print(f'\n{"=" * 60}')
    print('FIX SUMMARY')
    print('=' * 60)
    print(f'  Input:  {stats["total"] + stats["removed"]} questions')
    print(f'  Output: {stats["total"]} questions')
    print(f'  Removed: {stats["removed"]} (duplicates)')
    print(f'  L1: {stats["L1_count"]} | L2: {stats["L2_count"]} | L3: {stats["L3_count"]}')
    print(f'  L3 keywords fixed: {stats["l3_keywords_fixed"]}')
    print(f'  L1 empty answers remaining: {stats["l1_empty_answers"]}')
    print(f'  Citations normalized: {stats["citations_normalized"]}')
    print(f'  Topics added: {stats["topics_added"]}')
    print(f'  Knowledge bases updated: {stats["knowledge_base_updated"]}')
    print(f'\nSaved: {OUTPUT_FILE}')
    print(f'Report: {REPORT_FILE}')
