"""Improve dataset: expand short L3 queries + add specialized topic coverage."""
import sys,os,json,re,random
sys.path.insert(0,r'D:\coding\meta_AutoData\scripts')
os.environ['DEEPSEEK_API_KEY']=os.environ.get('DEEPSEEK_API_KEY','')
from data_generation.config import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL,STRONG_MODEL
from data_generation.utils.id_manager import IDManager
from data_generation.utils.format_validator import validate_question
from data_generation.utils.escaping_fixer import fix_question_escaping
from data_generation.utils.rubric_builder import build_rubric
from openai import OpenAI

def parse(text):
    try: return json.loads(text)
    except:
        for m in re.finditer(r'```(?:json)?\s*([\s\S]*?)\s*```',text):
            try: return json.loads(m.group(1))
            except: pass
        arr=re.search(r'\[[\s\S]*\]',text)
        if arr:
            try: return json.loads(arr.group(0))
            except:
                fixed=re.sub(r',(\s*[}\]])',r'\1',arr.group(0))
                try: return json.loads(fixed)
                except: pass
    return []

client=OpenAI(api_key=DEEPSEEK_API_KEY,base_url=DEEPSEEK_BASE_URL)
DATA=r'D:\coding\meta_AutoData\data\questions'
qs=json.load(open(f'{DATA}/generated_eval_set_v1.json',encoding='utf-8'))

# ═══════════════════════════════════════════════════════════
# TASK 1: Expand 60 short L3 queries
# ═══════════════════════════════════════════════════════════
l3=[q for q in qs if q['level']=='L3']
short_l3=[q for q in l3 if len(q.get('query',''))<300]
print(f'Task 1: Expanding {len(short_l3)} short L3 queries (batch=5)...')

expanded=0
for bn in range(0,len(short_l3),5):
    batch=short_l3[bn:bn+5]
    items='\n\n---\n\n'.join(
        f'[#{i+1}] Query: {q["query"]}\nAnswer: {q["expected_answer"][:200]}'
        for i,q in enumerate(batch))

    prompt=f'''以下5道电力系统L3评测题的场景描述过短（<300字），请为每题扩写query中的场景部分，增加：
- 具体的工程参数（电压等级、设备容量、线路长度等）
- 明确的约束条件（短路容量限制、走廊限制、环保限制等）
- 清晰的决策目标（方案比选的目标是什么）

保持原有问题核心不变，保持答案不变。输出严格JSON数组：
[{{"query":"扩写后的完整query（含≥300字场景+问题）","expected_answer":"保持原答案不变","expected_keywords":[...五个关键词],"source_standard":"原标准"}}]

原题：
{items}

输出严格的纯JSON数组（5个对象）：'''

    resp=client.chat.completions.create(model=STRONG_MODEL,messages=[{'role':'user','content':prompt}],temperature=0.3,max_tokens=12000)
    results=parse(resp.choices[0].message.content)
    for r in results:
        if bn+len(results)<=len(short_l3):
            idx=bn+results.index(r)
            if idx<len(short_l3):
                orig=short_l3[idx]
                orig['query']=r.get('query',orig['query'])
                expanded+=1
    print(f'  batch {bn//5+1}/{(len(short_l3)+4)//5}: expanded {len(results)}')

print(f'Expanded: {expanded}/{len(short_l3)}')

# Verify
still_short=sum(1 for q in l3 if len(q.get('query',''))<300)
print(f'Still <300 chars: {still_short}/100')

# ═══════════════════════════════════════════════════════════
# TASK 2: Generate topic supplement questions
# ═══════════════════════════════════════════════════════════
print(f'\nTask 2: Generating topic-specific supplement questions...')

# Topics needing more coverage
L2_TOPICS=[
    ('负载率计算与N-1校验','涉及线路/变压器负载率限额校核、N-1故障后剩余元件不过载的要求'),
    ('暂态稳定与动态稳定计算','涉及故障切除时间、临界切除时间、功角稳定裕度、低频振荡阻尼'),
    ('N-2及多重故障分析','涉及双回线同停、母线故障、断路器失灵等较严重故障场景'),
    ('过电压保护与绝缘配合','涉及工频过电压限值、操作过电压、避雷器配置、绝缘裕度'),
    ('短路电流限制措施','涉及断路器遮断容量、限流电抗器、中性点接地方式对零序电流的影响'),
]
L3_TOPICS=[
    ('负载率与热稳定极限的综合决策','当N-1后负载率超标时，增容导线vs新建线路vs需求侧管理的比选'),
    ('多重故障下的稳定控制策略','N-2同停后系统是否满足第二/三级安全稳定标准，切机/切负荷决策'),
]

# Generate L2 supplement (10 questions)
print('  L2 supplement (10 questions)...')
l2_prompt=f'''从三份电力标准(GB 38755-2019,DL/T 5429-2009,DL/T 5218-2012)生成10道L2推理型评测题，
专门覆盖以下5个主题（每主题2道）：
{chr(10).join(f'{i+1}. {t[0]}: {t[1]}' for i,t in enumerate(L2_TOPICS))}

每题含150-300字场景+推理问题。3标准各占~1/3。输出严格JSON数组：
[{{"query":"【场景】...\\n\\n【问题】...","expected_answer":"...","expected_keywords":["kw1","kw2","kw3","kw4","kw5"],"source_standard":"标准 §X.Y","category":"审查判断"}}]'''

resp=client.chat.completions.create(model=STRONG_MODEL,messages=[{'role':'user','content':l2_prompt}],temperature=0.3,max_tokens=12000)
l2_supp=parse(resp.choices[0].message.content)
print(f'    -> {len(l2_supp)} generated')

# Generate L3 supplement (4 questions)
print('  L3 supplement (4 questions)...')
l3_prompt=f'''从三份电力标准生成4道L3综合评测题，专门覆盖：
{chr(10).join(f'{i+1}. {t[0]}: {t[1]}' for i,t in enumerate(L3_TOPICS))}

每题含方案A/B对比(≥300字场景)，答案五段式。跨标准引用。输出严格JSON数组。'''

resp=client.chat.completions.create(model=STRONG_MODEL,messages=[{'role':'user','content':l3_prompt}],temperature=0.4,max_tokens=16000)
l3_supp=parse(resp.choices[0].message.content)
print(f'    -> {len(l3_supp)} generated')

# ═══════════════════════════════════════════════════════════
# TASK 3: Replace weakest existing questions
# ═══════════════════════════════════════════════════════════
print(f'\nTask 3: Replacing weakest questions with supplements...')

# Find weakest L2 questions (shortest queries, least keywords)
l2_qs=[q for q in qs if q['level']=='L2']
l2_weak=sorted(l2_qs,key=lambda q: len(q.get('query',''))+len(q.get('expected_keywords',[]))*10)[:len(l2_supp)]

for old,new in zip(l2_weak,l2_supp):
    new['question_id']=old['question_id']
    new['question_class']='L2'; new['level']='L2'
    new['grading_method']='manual_review'; new['knowledge_base']='General-KB'
    new=fix_question_escaping(new)
    rubric=build_rubric(new.get('expected_answer',''),'L2')
    for k,v in rubric.items(): new[k]=v
    qs[qs.index(old)]=new

# Find weakest L3 questions (shortest queries)
l3_qs=[q for q in qs if q['level']=='L3']
l3_weak=sorted(l3_qs,key=lambda q: len(q.get('query','')))[:len(l3_supp)]

for old,new in zip(l3_weak,l3_supp):
    new['question_id']=old['question_id']
    new['question_class']='L3'; new['level']='L3'
    new['grading_method']='manual_review'; new['knowledge_base']='General-KB'
    new=fix_question_escaping(new)
    rubric=build_rubric(new.get('expected_answer',''),'L3')
    for k,v in rubric.items(): new[k]=v
    qs[qs.index(old)]=new

print(f'  Replaced {len(l2_supp)} L2 + {len(l3_supp)} L3')

# ═══════════════════════════════════════════════════════════
# VERIFY & SAVE
# ═══════════════════════════════════════════════════════════
print(f'\n=== VERIFICATION ===')
for lvl in ['L1','L2','L3']:
    lqs=[q for q in qs if q['level']==lvl]
    v=sum(1 for q in lqs if validate_question(q,lvl)[0])
    print(f'{lvl}: {len(lqs)} total, {v} valid')

# L3 length check
l3=[q for q in qs if q['level']=='L3']
s=sum(1 for q in l3 if len(q.get('query',''))<300)
print(f'L3 <300 chars: {s}/100')

# L2/L3 topic coverage
for lvl in ['L2','L3']:
    lqs=[q for q in qs if q['level']==lvl]
    topics={'N-1':0,'N-2':0,'负载率':0,'潮流':0,'稳定计算':0,'短路':0,'过电压':0,'备用容量':0,'无功':0}
    for q in lqs:
        t=q.get('query','')+q.get('expected_answer','')
        for tp in topics:
            if tp in t: topics[tp]+=1
    print(f'{lvl} topics: {topics}')

# Standard distribution
print('Standard distribution:')
for lvl in ['L1','L2','L3']:
    lqs=[q for q in qs if q['level']==lvl]
    sc={'GB':0,'DL/T 5429':0,'DL/T 5218':0}
    for q in lqs:
        src=q.get('source_standard','')
        if '38755' in src: sc['GB']+=1
        if '5429' in src: sc['DL/T 5429']+=1
        if '5218' in src: sc['DL/T 5218']+=1
    gb = sc['GB']; dlt5 = sc['DL/T 5429']; dlt52 = sc['DL/T 5218']
    print(f'  {lvl}: GB={gb}, DLT5429={dlt5}, DLT5218={dlt52}')

with open(f'{DATA}/generated_eval_set_v1.json','w',encoding='utf-8') as f:
    json.dump(qs,f,ensure_ascii=False,indent=2)
print(f'\nSaved: {len(qs)} questions')
