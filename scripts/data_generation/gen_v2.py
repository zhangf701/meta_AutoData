"""V2: Balanced generation — L1/L2 per standard, L3 cross-standard pairs.

Target (per level, ±5):
  GB 38755-2019: 33-37 | DL/T 5429-2009: 28-32 | DL/T 5218-2012: 33-37

L3 coverage math (100 questions × 2 standards = 200 mentions):
  GB×DLT5429=30, GB×DLT5218=40, DLT5429×DLT5218=30
  → GB=70(35%), DLT5429=60(30%), DLT5218=70(35%)
"""
import sys, os, json, re
sys.path.insert(0, r'D:\coding\meta_AutoData\scripts')
os.environ['DEEPSEEK_API_KEY'] = os.environ.get('DEEPSEEK_API_KEY','')
from data_generation.source_loader import load_all_sources
from data_generation.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STRONG_MODEL
from openai import OpenAI


def parse(text):
    try: return json.loads(text)
    except:
        for m in re.finditer(r'```(?:json)?\s*([\s\S]*?)\s*```', text):
            try: return json.loads(m.group(1))
            except: pass
        arr = re.search(r'\[[\s\S]*\]', text)
        if arr:
            try: return json.loads(arr.group(0))
            except:
                fixed = re.sub(r',(\s*[}\]])', r'\1', arr.group(0))
                try: return json.loads(fixed)
                except: pass
    return []


sources, _ = load_all_sources()
STD = {}
for src in sources:
    for kw, name in [('38755','GB 38755-2019'), ('5429','DL/T 5429-2009'), ('5218','DL/T 5218-2012')]:
        if kw in src['title']: STD[name] = src['raw_text']; break

for n, t in STD.items():
    cjk = sum(1 for c in t if 0x4e00 <= ord(c) <= 0x9fff)
    print(f'{n}: {len(t):,} chars, {cjk:,} CJK')

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
OUT = r'D:\coding\meta_AutoData\data\questions'


def gen_batch(std_name, std_text, level, count, batch_size, extra_constraint=""):
    """Generate `count` L1 or L2 questions from a single standard."""
    all_qs = []
    label = '直接型：答案在单一规范条款中可直接检索' if level == 'L1' else '推理型：需条件判断或跨参数推理'
    cat = '参数检索' if level == 'L1' else '审查判断'
    max_tok = 8000 if level == 'L1' else 12000

    num_batches = (count + batch_size - 1) // batch_size
    for bn in range(num_batches):
        n = min(batch_size, count - len(all_qs))
        print(f'  batch {bn+1}/{num_batches} ({n}q)...', end=' ', flush=True)

        prompt = f'''从标准 {std_name} 生成{n}道{level}评测题。{label}
{extra_constraint}
输出严格JSON数组：[{{"query":"...","expected_answer":"...","expected_keywords":["kw1","kw2","kw3","kw4","kw5"],"source_standard":"{std_name} §X.Y","category":"{cat}"}}]
标准文本：{std_text[:8000]}'''

        resp = client.chat.completions.create(
            model=STRONG_MODEL, messages=[{'role':'user','content':prompt}],
            temperature=0.3, max_tokens=max_tok)
        batch = parse(resp.choices[0].message.content)
        all_qs.extend(batch[:n])
        print(f'{len(batch)} parsed ({len(all_qs)}/{count})')
    return all_qs[:count]


def gen_l3_pair(std_a, std_b, text_a, text_b, count):
    """Generate L3 cross-standard synthesis questions."""
    all_qs = []
    for bn in range((count + 4) // 5):
        n = min(5, count - len(all_qs))
        print(f'  {std_a.split()[0]}×{std_b.split()[0]} batch {bn+1} ({n}q)...', end=' ', flush=True)

        prompt = f'''从以下两份标准生成{n}道L3综合评测题。L3=需引用两份规范综合判断。
每题必须含方案A/B对比（各有具体参数），答案五段式：现状诊断→多维冲突分析→多方案比选→折中综合方案→控制优先级链条。
必须同时引用{std_a}和{std_b}的具体条款。

标准1 {std_a}：{text_a[:4000]}
标准2 {std_b}：{text_b[:4000]}

输出严格JSON数组：[{{"query":"【场景】...\\n\\n【问题】...","expected_answer":"现状诊断：...\\n多维冲突分析：...\\n多方案比选：...\\n折中综合方案：...\\n控制优先级链条：...","expected_keywords":["kw1","kw2","kw3","kw4","kw5"],"source_standard":"{std_a}, {std_b}","category":"综合评估"}}]'''

        resp = client.chat.completions.create(
            model=STRONG_MODEL, messages=[{'role':'user','content':prompt}],
            temperature=0.4, max_tokens=16000)
        batch = parse(resp.choices[0].message.content)
        all_qs.extend(batch[:n])
        print(f'{len(batch)} parsed ({len(all_qs)}/{count})')
    return all_qs[:count]


# ═══════════════════════════════════════════════
#  PLAN
#  L1 (100): GB=35, DLT5429=30, DLT5218=35
#  L2 (100): GB=35, DLT5429=30, DLT5218=35
#  L3 (100): GB×DLT5429=30, GB×DLT5218=40, DLT5429×DLT5218=30
# ═══════════════════════════════════════════════

L1_PLAN = [('GB 38755-2019', 35), ('DL/T 5429-2009', 30), ('DL/T 5218-2012', 35)]
L2_PLAN = [('GB 38755-2019', 35), ('DL/T 5429-2009', 30), ('DL/T 5218-2012', 35)]
L3_PLAN = [
    ('GB 38755-2019', 'DL/T 5429-2009', 30),
    ('GB 38755-2019', 'DL/T 5218-2012', 40),
    ('DL/T 5429-2009', 'DL/T 5218-2012', 30),
]

all_l1, all_l2, all_l3 = [], [], []

# ── L1 ──
print('\n' + '='*60 + '\nL1 (100)\n' + '='*60)
for std_name, count in L1_PLAN:
    print(f'{std_name}: {count}')
    qs = gen_batch(std_name, STD[std_name], 'L1', count, batch_size=10)
    all_l1.extend(qs)

# ── L2 ──
print('\n' + '='*60 + '\nL2 (100)\n' + '='*60)
for std_name, count in L2_PLAN:
    print(f'{std_name}: {count}')
    qs = gen_batch(std_name, STD[std_name], 'L2', count, batch_size=10,
                   extra_constraint="每题含150-300字工程场景+具体参数+推理问题")
    all_l2.extend(qs)

# ── L3 ──
print('\n' + '='*60 + '\nL3 (100, cross-standard)\n' + '='*60)
for std_a, std_b, count in L3_PLAN:
    print(f'{std_a} × {std_b}: {count}')
    qs = gen_l3_pair(std_a, std_b, STD[std_a], STD[std_b], count)
    all_l3.extend(qs)

# ── Verify distribution ──
print('\n' + '='*60 + '\nVERIFICATION\n' + '='*60)
for label, qs in [('L1', all_l1), ('L2', all_l2), ('L3', all_l3)]:
    std_cnt = {'GB 38755':0, 'DL/T 5429':0, 'DL/T 5218':0}
    for q in qs:
        src = q.get('source_standard','')
        for s in std_cnt:
            if s.split()[1] in src: std_cnt[s] += 1
    print(f'{label} ({len(qs)}): ', end='')
    for s, c in std_cnt.items():
        print(f'{s}={c}({100*c/max(len(qs),1):.0f}%) ', end='')
    print()

# ── Save ──
for name, data in [('l1',all_l1),('l2',all_l2),('l3',all_l3)]:
    with open(f'{OUT}/gen_{name}_batch.json','w',encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
print(f'\nSaved. L1={len(all_l1)}, L2={len(all_l2)}, L3={len(all_l3)}')
