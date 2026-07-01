"""Replace DL/T 5218 questions with non-electrical content.

DL/T 5218 electrical chapters only: 电气主接线(5), 配电装置(6), 无功补偿(7),
电气二次(8), 过电压保护与绝缘配合(9), 接地(10), 站用电(11).
Exclude: 土建, 消防, 暖通, 环保, 建筑, 结构, 道路, 给排水.
"""
import sys,os,json,re
sys.path.insert(0,r'D:\coding\meta_AutoData\scripts')
os.environ['DEEPSEEK_API_KEY']=os.environ.get('DEEPSEEK_API_KEY','')
from openai import OpenAI
from data_generation.source_loader import load_all_sources
from data_generation.config import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL,STRONG_MODEL
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
            except:
                fixed=re.sub(r',(\s*[}\]])',r'\1',arr.group(0))
                try: return json.loads(fixed)
                except: pass
    return []

sources,_=load_all_sources()
dlt5218=[s['raw_text'] for s in sources if '5218' in s['title']][0]
gb38755=[s['raw_text'] for s in sources if '38755' in s['title']][0]
dlt5429=[s['raw_text'] for s in sources if '5429' in s['title']][0]

client=OpenAI(api_key=DEEPSEEK_API_KEY,base_url=DEEPSEEK_BASE_URL)
DATA=r'D:\coding\meta_AutoData\data\questions'
qs=json.load(open(f'{DATA}/generated_eval_set_v1.json',encoding='utf-8'))

ELECTRICAL_TOPICS='''仅使用DL/T 5218-2012的电气章节：第5章(电气主接线)、第6章(配电装置)、第7章(无功补偿)、第8章(电气二次/继电保护/监控)、第9章(过电压保护与绝缘配合)、第10章(接地)、第11章(站用电)。
严禁涉及：土建、建筑、结构、消防、暖通、给排水、环保、绿化、噪声、抗震、通风、空调、采暖、围墙、道路、大门、装修、地基、基础等非电气内容。'''

# ═══ Generate L1 replacements (2) ═══
print('L1 replacements (2 questions)...')
resp=client.chat.completions.create(model=STRONG_MODEL,messages=[{'role':'user','content':f'从DL/T 5218-2012生成2道L1评测题（直接型）。{ELECTRICAL_TOPICS}\n输出JSON数组。\n标准文本：{dlt5218[:8000]}'}],temperature=0.3,max_tokens=4000)
l1_new=parse(resp.choices[0].message.content)
print(f'  -> {len(l1_new)} generated')

# ═══ Generate L2 replacements (19) ═══
l2_new=[]
for bn in range(2):
    print(f'L2 batch {bn+1}/2...')
    resp=client.chat.completions.create(model=STRONG_MODEL,messages=[{'role':'user','content':f'从DL/T 5218-2012生成{10 if bn==0 else 9}道L2评测题（推理型，含150-300字工程场景）。{ELECTRICAL_TOPICS}\n输出严格JSON数组。\n标准文本：{dlt5218[bn*6000:(bn+1)*6000+2000]}'}],temperature=0.3,max_tokens=12000)
    batch=parse(resp.choices[0].message.content)
    l2_new.extend(batch)
    print(f'  -> {len(batch)} parsed ({len(l2_new)} total)')

# ═══ Generate L3 replacements (50) ═══
# Split into cross-standard pairs involving DL/T 5218 electrical content
l3_new=[]
for bn in range(10):
    n=5
    # Rotate the pairing standard
    if bn<5:
        other_std_name='GB 38755-2019'; other_text=gb38755[:4000]
    else:
        other_std_name='DL/T 5429-2009'; other_text=dlt5429[:4000]
    print(f'L3 batch {bn+1}/10: DL/T 5218 × {other_std_name} ({n}q)...')
    resp=client.chat.completions.create(model=STRONG_MODEL,messages=[{'role':'user','content':f'''从DL/T 5218-2012和{other_std_name}生成{n}道L3综合评测题（需引用两份标准，含方案A/B对比，答案五段式）。
{ELECTRICAL_TOPICS}
仅涉及电气一次、系统及电气二次内容。

标准1 DL/T 5218-2012：{dlt5218[bn*4000:(bn+1)*4000]}
标准2 {other_std_name}：{other_text}

输出严格JSON数组。'''}],temperature=0.4,max_tokens=16000)
    batch=parse(resp.choices[0].message.content)
    l3_new.extend(batch)
    print(f'  -> {len(batch)} parsed ({len(l3_new)} total)')

print(f'\nTotal: L1={len(l1_new)}, L2={len(l2_new)}, L3={len(l3_new)}')

# ═══ Replace affected questions ═══
non_elec=['土建','建筑','结构','消防','暖通','给排水','环保','绿化','噪声','抗震','通风','空调','采暖','围墙','道路','大门','装修','屋面','地基','基础','桩','混凝土','钢筋','砌体','门窗','排水沟','水保','生态','排污','废水','隔声','降噪']
replaced={'L1':0,'L2':0,'L3':0}
for lvl,new_qs in [('L1',l1_new),('L2',l2_new),('L3',l3_new)]:
    old_bad=[q for q in qs if q['level']==lvl and '5218' in q.get('source_standard','')
             and any(kw in q.get('query','')+q.get('expected_answer','') for kw in non_elec)]
    for old,new in zip(old_bad,new_qs):
        new['question_id']=old['question_id']
        new['question_class']=lvl; new['level']=lvl
        new['grading_method']='auto_keyword_match' if lvl=='L1' else 'manual_review'
        new['knowledge_base']='General-KB'
        if 'category' not in new: new['category']={'L1':'参数检索','L2':'审查判断','L3':'综合评估'}[lvl]
        new=fix_question_escaping(new)
        if lvl in ('L2','L3'):
            r=build_rubric(new.get('expected_answer',''),lvl)
            for k,v in r.items(): new[k]=v
        # Pad keywords
        kws=list(new.get('expected_keywords',[]))
        if len(kws)<3:
            for c in re.findall(r'[一-鿿]{2,4}',new.get('query','')+new.get('expected_answer','')):
                if c not in kws: kws.append(c)
                if len(kws)>=5: break
            new['expected_keywords']=kws[:5]
        qs[qs.index(old)]=new
        replaced[lvl]+=1

print(f'Replaced: {replaced}')

# ═══ Verify ═══
print('\n=== VERIFICATION ===')
for lvl in ['L1','L2','L3']:
    lqs=[q for q in qs if q['level']==lvl]
    v=sum(1 for q in lqs if validate_question(q,lvl)[0])
    # Standard dist
    gb=sum(1 for q in lqs if '38755' in q.get('source_standard',''))
    d5=sum(1 for q in lqs if '5429' in q.get('source_standard',''))
    d52=sum(1 for q in lqs if '5218' in q.get('source_standard',''))
    # Check remaining non-elec
    ne=sum(1 for q in lqs if '5218' in q.get('source_standard','') and any(kw in q.get('query','')+q.get('expected_answer','') for kw in non_elec))
    print(f'{lvl}: {v}/{len(lqs)} valid | GB={gb} DLT5429={d5} DLT5218={d52} | non-elec remaining: {ne}')

with open(f'{DATA}/generated_eval_set_v1.json','w',encoding='utf-8') as f:
    json.dump(qs,f,ensure_ascii=False,indent=2)
print('Saved.')
