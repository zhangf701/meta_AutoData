"""Complete dataset generation: L1(100) + L2(100) + L3(30) via DeepSeek."""
import sys,os,json,re,random
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
std = {s['title']: s['raw_text'][:8000] for s in sources if any(k in s['title'] for k in ['GB 38755','DLT 5429','DLT 5218'])}
combined = '\n\n'.join(f'=== {t} ===\n{tx}' for t,tx in std.items())
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ── L1: 5 batches of 20 ──
l1 = []
for bn in range(5):
    print(f'L1 batch {bn+1}/5...')
    p = f'从电力系统标准生成20道L1评测题（答案在单一规范条款中可直接检索，每题一个精确答案）。输出JSON数组：[{{"query":"...","expected_answer":"精确答案","expected_keywords":["kw1","kw2","kw3"],"source_standard":"标准编号 条款号","category":"参数检索","topic":"主题"}}]\n标准：{combined[:7000]}'
    r = client.chat.completions.create(model=STRONG_MODEL, messages=[{'role':'user','content':p}], temperature=0.3, max_tokens=8000)
    l1.extend(parse(r.choices[0].message.content))
    print(f'  -> {len(l1)} total')

# ── L2: 4 batches of 25 ──
l2 = []
for bn in range(4):
    print(f'L2 batch {bn+1}/4...')
    p = f'从电力系统标准生成25道L2评测题（需条件判断或跨参数推理，每题含150-300字工程场景+推理问题，答案引用标准条款）。输出JSON数组：[{{"query":"【场景】...\\n\\n【问题】...","expected_answer":"推理过程","expected_keywords":["kw1","kw2","kw3","kw4","kw5"],"source_standard":"标准编号 条款号","category":"审查判断"}}]\n标准：{combined[:7000]}'
    r = client.chat.completions.create(model=STRONG_MODEL, messages=[{'role':'user','content':p}], temperature=0.3, max_tokens=12000)
    l2.extend(parse(r.choices[0].message.content))
    print(f'  -> {len(l2)} total')

# ── L3: 3 batches of 6 (long answers, 5-part structure) ──
l3 = []
for bn in range(3):
    print(f'L3 batch {bn+1}/3...')
    p = f'从电力系统标准生成6道L3综合评测题（需引用2+标准，含方案A/B对比，答案五段式：诊断-冲突-比选-决策-控制链，每道约1500字）。输出JSON数组。\n标准：{combined[:7000]}'
    r = client.chat.completions.create(model=STRONG_MODEL, messages=[{'role':'user','content':p}], temperature=0.4, max_tokens=12000)
    l3.extend(parse(r.choices[0].message.content))
    print(f'  -> {len(l3)} total')

print(f'\nDone: L1={len(l1)}, L2={len(l2)}, L3={len(l3)}')

# Save raw batches
for name, data in [('l1',l1),('l2',l2),('l3',l3)]:
    path = f'D:\\coding\\meta_AutoData\\data\\questions\\gen_{name}_batch.json'
    with open(path,'w',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2)

print('Batches saved.')
