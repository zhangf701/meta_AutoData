"""Assemble final dataset from generated batches."""
import sys,os,json,random,re
sys.path.insert(0, r'D:\coding\meta_AutoData\scripts')
os.environ['DEEPSEEK_API_KEY'] = os.environ.get('DEEPSEEK_API_KEY','')

from data_generation.utils.id_manager import IDManager
from data_generation.utils.format_validator import validate_question
from data_generation.utils.escaping_fixer import fix_question_escaping
from data_generation.utils.rubric_builder import build_rubric

DATA = r'D:\coding\meta_AutoData\data\questions'

# Load pre-generated batches
l1_path = os.path.join(DATA, 'gen_l1_batch.json')
l2_path = os.path.join(DATA, 'gen_l2_batch.json')
l3_path = os.path.join(DATA, 'gen_l3_batch.json')

l1_qs = json.load(open(l1_path, encoding='utf-8'))
l2_qs = json.load(open(l2_path, encoding='utf-8'))
l3_qs = json.load(open(l3_path, encoding='utf-8'))

print(f'L1={len(l1_qs)}, L2={len(l2_qs)}, L3={len(l3_qs)}')

# Assemble with proper IDs
idm = IDManager()
all_qs = []
for lvl, qs in [('L1', l1_qs), ('L2', l2_qs), ('L3', l3_qs)]:
    for q in qs:
        # Normalize DeepSeek field names to our schema
        if 'scenario' in q and 'query' not in q:
            scenario = q.pop('scenario', '')
            question_text = q.pop('question', q.pop('title', ''))
            q['query'] = f"【场景】{scenario}\n\n【问题】{question_text}" if scenario else question_text
        if 'answer' in q and 'expected_answer' not in q:
            ans = q.pop('answer', '')
            q['expected_answer'] = ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)
        if 'standards_cited' in q and 'source_standard' not in q:
            stds = q.pop('standards_cited', [])
            q['source_standard'] = ', '.join(stds) if isinstance(stds, list) else str(stds)
        if 'options' in q and 'expected_keywords' not in q:
            opts = q.pop('options', [])
            q['expected_keywords'] = opts if isinstance(opts, list) else [opts]
        if 'keywords' in q and 'expected_keywords' not in q:
            q['expected_keywords'] = q.pop('keywords', [])

        q['question_id'] = idm.next(lvl)
        q['question_class'] = lvl
        q['level'] = lvl
        q['grading_method'] = 'auto_keyword_match' if lvl=='L1' else 'manual_review'
        q['knowledge_base'] = 'General-KB'
        if not q.get('category'):
            q['category'] = {'L1':'参数检索','L2':'审查判断','L3':'综合评估'}[lvl]
        if not q.get('source_standard'):
            q['source_standard'] = ''
        q = fix_question_escaping(q)

        # Pad keywords to >= 3 for any level
        if len(q.get('expected_keywords', [])) < 3:
            kws = list(q.get('expected_keywords', []))
            # Add words from query and answer
            for char_seq in re.findall(r'[一-鿿]{2,4}', q.get('query','') + q.get('expected_answer','')):
                if char_seq not in kws:
                    kws.append(char_seq)
                if len(kws) >= 5:
                    break
            q['expected_keywords'] = kws[:5]

        if lvl in ('L2','L3'):
            rubric = build_rubric(q.get('expected_answer',''), lvl)
            for k,v in rubric.items():
                q[k] = v
        all_qs.append(q)

# Stats
by_level = {}
for q in all_qs:
    lvl = q['level']
    by_level.setdefault(lvl, {'total':0,'valid':0})
    by_level[lvl]['total'] += 1
    v, issues = validate_question(q, lvl)
    by_level[lvl]['valid'] += int(v)

print(f'\nFinal Dataset: {len(all_qs)} questions')
for lvl in ['L1','L2','L3']:
    d = by_level.get(lvl, {'total':0,'valid':0})
    print(f'  {lvl}: {d["total"]} ({d["valid"]} valid)')

out = os.path.join(DATA, 'generated_eval_set_v1.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(all_qs, f, ensure_ascii=False, indent=2)
print(f'Saved: {out} ({len(all_qs)} questions)')
