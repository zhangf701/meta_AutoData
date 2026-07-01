"""Regenerate deduplicated questions with proper distribution."""
import sys,os,json,re
sys.path.insert(0,r'D:\coding\meta_AutoData\scripts')
os.environ['DEEPSEEK_API_KEY']=os.environ.get('DEEPSEEK_API_KEY','')
from openai import OpenAI
from data_generation.config import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL,STRONG_MODEL
from data_generation.source_loader import load_all_sources
from data_generation.utils.format_validator import validate_question
from data_generation.utils.escaping_fixer import fix_question_escaping
from data_generation.utils.rubric_builder import build_rubric

def parse(text):
    try: return json.loads(text)
    except:
        for m in re.finditer(r'```(?:json)?\s*([\s\S]*?)\s*```',text):
            try: return json.loads(m.group(1))
            except: pass
        arr=re.search(r'\[[\s\S]*\]',text)
        if arr:
            try: return json.loads(arr.group(0))
            except: return []

sources,_=load_all_sources()
STD={}
for s in sources:
    for kw,n in [('38755','GB 38755-2019'),('5429','DL/T 5429-2009'),('5218','DL/T 5218-2012')]:
        if kw in s['title']: STD[n]=s['raw_text']; break

client=OpenAI(api_key=DEEPSEEK_API_KEY,base_url=DEEPSEEK_BASE_URL)
qs=json.load(open(r'D:\coding\meta_AutoData\data\questions\generated_eval_set_v1.json',encoding='utf-8'))
print(f'Current: {len(qs)} questions')

# Collect existing answers per standard to dedup against
existing_answers={}
for q in qs:
    src=q.get('source_standard',''); lvl=q['level']
    for kw in ['38755','5429','5218']:
        if kw in src: existing_answers.setdefault(kw,set()).add(q.get('expected_answer','')[:80])

new_qs=[]

# L1: GB=16, DLT5429=14, DLT5218=16
for std_name,count in [('GB 38755-2019',16),('DL/T 5429-2009',14),('DL/T 5218-2012',16)]:
    kw=std_name.split()[1]
    generated=0
    for bn in range((count+9)//10):
        n=min(10,count-generated)
        if n<=0: break
        print(f'L1 {std_name}: {n}q...',end=' ',flush=True)
        existing=list(existing_answers.get(kw,set()))[:8]
        prompt=f'从{std_name}生成{n}道L1评测题（直接型，答案在单一规范条款中可直接检索）。要求每题有精确数值答案或一句话答案，避免笼统定义类题目。避免与以下答案重复：{chr(10).join(existing)}。输出严格JSON数组。标准文本：{STD[std_name][:6000]}'
        resp=client.chat.completions.create(model=STRONG_MODEL,messages=[{'role':'user','content':prompt}],temperature=0.3,max_tokens=8000)
        batch=parse(resp.choices[0].message.content)
        new_qs.extend(batch); generated+=len(batch)
        print(f'{len(batch)} parsed ({generated}/{count})')

# L2: 3 new DLT5429
print(f'L2 DLT5429: 3q...',end=' ',flush=True)
resp=client.chat.completions.create(model=STRONG_MODEL,messages=[{'role':'user','content':f'从DL/T 5429-2009生成3道L2评测题（推理型，150-300字工程场景+推理问题，答案引用标准条款）。电气专业。输出严格JSON数组。\n标准文本：{STD["DL/T 5429-2009"][:6000]}'}],temperature=0.3,max_tokens=8000)
l2_new=parse(resp.choices[0].message.content)
for q in l2_new: q['level']='L2'; q['question_class']='L2'
new_qs.extend(l2_new)
print(f'{len(l2_new)} parsed')

# L3: 4 new GBxDLT5429
print(f'L3 GBxDLT5429: 4q...',end=' ',flush=True)
resp=client.chat.completions.create(model=STRONG_MODEL,messages=[{'role':'user','content':f'从GB 38755-2019和DL/T 5429-2009生成4道L3综合评测题（方案A/B对比，含300+字场景，答案五段式：诊断-冲突-比选-决策-控制链）。输出严格JSON数组。\n标准1：{STD["GB 38755-2019"][:4000]}\n标准2：{STD["DL/T 5429-2009"][:4000]}'}],temperature=0.4,max_tokens=16000)
l3_new=parse(resp.choices[0].message.content)
for q in l3_new: q['level']='L3'; q['question_class']='L3'
new_qs.extend(l3_new)
print(f'{len(l3_new)} parsed')

# Assign IDs
ID_POOL={'L1':['L1-113','L1-114','L1-120','L1-121','L1-122','L1-123','L1-124','L1-125','L1-126','L1-127','L1-128','L1-132','L1-134','L1-135','L1-143','L1-147','L1-148','L1-149','L1-152','L1-155','L1-156','L1-157','L1-158','L1-159','L1-160','L1-161','L1-162','L1-164','L1-165','L1-177','L1-179','L1-183','L1-184','L1-186','L1-187','L1-188','L1-189','L1-191','L1-192','L1-194','L1-195','L1-196','L1-197','L1-198','L1-199','L1-200'],
         'L2':['L2-145','L2-156','L2-184'],
         'L3':['L3-185','L3-186','L3-187','L3-188']}

id_ptr={k:0 for k in ID_POOL}
for q in new_qs:
    lvl=q.get('level','L1')
    pool=ID_POOL.get(lvl,ID_POOL['L1'])
    if id_ptr[lvl]<len(pool):
        q['question_id']=pool[id_ptr[lvl]]; id_ptr[lvl]+=1
    else:
        q['question_id']=f'{lvl}-FIX{id_ptr[lvl]}'; id_ptr[lvl]+=1
    q['question_class']=lvl; q['level']=lvl
    q['grading_method']='auto_keyword_match' if lvl=='L1' else 'manual_review'
    q['knowledge_base']='General-KB'
    if not q.get('category'): q['category']={'L1':'参数检索','L2':'审查判断','L3':'综合评估'}[lvl]
    q=fix_question_escaping(q)
    if lvl in ('L2','L3'):
        rubric=build_rubric(q.get('expected_answer',''),lvl)
        for k,v in rubric.items(): q[k]=v
    if 'scenario' in q and 'question' in q and 'query' not in q:
        q['query']=f'【场景】{q.pop("scenario","")}\n\n【问题】{q.pop("question","")}'
    if 'answer' in q and 'expected_answer' not in q:
        ans=q.pop('answer',''); q['expected_answer']=str(ans)
    kws=list(q.get('expected_keywords',[]))
    if len(kws)<3:
        for c in re.findall(r'[一-鿿]{2,4}',q.get('query','')+q.get('expected_answer','')):
            if c not in kws: kws.append(c)
            if len(kws)>=5: break
        q['expected_keywords']=kws[:5]
    if not q.get('source_standard'): q['source_standard']=''

qs.extend(new_qs)
print(f'\nTotal: {len(qs)} ({len(new_qs)} new)')

# Final verification
for lvl in ['L1','L2','L3']:
    lqs=[q for q in qs if q['level']==lvl]
    v=sum(1 for q in lqs if validate_question(q,lvl)[0])
    gb=sum(1 for q in lqs if '38755' in q.get('source_standard',''))
    d5=sum(1 for q in lqs if '5429' in q.get('source_standard',''))
    d52=sum(1 for q in lqs if '5218' in q.get('source_standard',''))
    print(f'{lvl}: {v}/{len(lqs)} | GB={gb} DLT5429={d5} DLT5218={d52}')

with open(r'D:\coding\meta_AutoData\data\questions\generated_eval_set_v1.json','w',encoding='utf-8') as f:
    json.dump(qs,f,ensure_ascii=False,indent=2)
print('Saved')
