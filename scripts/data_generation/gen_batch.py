"""Direct L2/L3 generation via DeepSeek from standard full texts."""
import sys, os, re, json
sys.path.insert(0, r'D:\coding\meta_AutoData\scripts')
os.environ['DEEPSEEK_API_KEY'] = os.environ.get('DEEPSEEK_API_KEY','')

from data_generation.source_loader import load_all_sources
from data_generation.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STRONG_MODEL
from openai import OpenAI

__all__ = ['robust_parse']


def robust_parse(text):
    """Multiple strategies to extract valid JSON array from LLM response."""
    # Strategy 1: direct
    try: return json.loads(text)
    except: pass
    # Strategy 2: code block extraction
    for m in re.finditer(r'```(?:json)?\s*([\s\S]*?)\s*```', text):
        try: return json.loads(m.group(1))
        except: pass
    # Strategy 3: array extraction
    arr = re.search(r'\[[\s\S]*\]', text)
    if arr:
        raw = arr.group(0)
        try: return json.loads(raw)
        except:
            # Strategy 4: fix trailing commas
            fixed = re.sub(r',(\s*[}\]])', r'\1', raw)
            try: return json.loads(fixed)
            except: pass
    # Strategy 5: line-by-line fix
    if arr:
        try:
            # Replace unescaped newlines in strings
            lines = arr.group(0).split('\n')
            cleaned = []
            for line in lines:
                stripped = line.strip()
                if stripped:
                    cleaned.append(stripped)
            return json.loads('\n'.join(cleaned))
        except: pass
    return []


if __name__ == '__main__':
    sources, _ = load_all_sources()
    std_texts = {s['title']: s['raw_text'][:8000] for s in sources
             if any(kw in s['title'] for kw in ['GB 38755','DLT 5429','DLT 5218'])}
design_texts = {s['title']: s['raw_text'][:5000] for s in sources
                if any(kw in s['title'] for kw in ['方案','反馈单','规划'])}
combined_std = '\n\n'.join(f'=== {t} ===\n{tx}' for t,tx in std_texts.items())
combined_design = '\n\n'.join(f'=== {t} ===\n{tx}' for t,tx in design_texts.items())

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ── L2 Generation (batch 1 of 4) ──
all_l2 = []
for batch_num in range(4):
    print(f"Generating L2 batch {batch_num+1}/4 (25 questions)...")
    prompt_l2 = f"""从以下电力系统标准中，生成25道L2级别（推理型）评测题。

L2定义：需要条件判断或跨参数推理。每题包含具体工程场景（150-300字）和推理问题。
答案需引用标准条款编号。

输出严格的纯JSON数组（不要markdown包裹，确保JSON完整闭合）：
[{{"query":"...","expected_answer":"...","expected_keywords":["kw1","kw2","kw3","kw4","kw5"],"source_standard":"标准编号 条款号","category":"审查判断"}}]

标准文本（用于提取规范条款）：
{combined_std[:8000]}

工程文档（用于提取真实场景参数，可选）：
{batch_num * 3}道题的场景建议从以下文档中提取参数：
{combined_design[:3000]}"""

    resp = client.chat.completions.create(
        model=STRONG_MODEL, messages=[{'role':'user','content':prompt_l2}],
        temperature=0.3, max_tokens=12000,
    )
    text = resp.choices[0].message.content
    batch_qs = robust_parse(text)
    all_l2.extend(batch_qs)
    print(f"  Batch {batch_num+1}: {len(batch_qs)} parsed")
    if not batch_qs:
        print(f"  RAW: {text[-200:]}")
        break

l2_qs = all_l2
print(f"Total L2: {len(l2_qs)}")

# ── L3 Generation (2 batches of 6) ──
all_l3 = []
for batch_num in range(2):
    print(f"Generating L3 batch {batch_num+1}/2 (6 questions)...")
    prompt_l3 = f"""从以下电力系统标准中，生成6道L3级别（综合型）评测题。

L3定义：需引用两份以上规范综合判断。每道题必须包含：
1. 两个完整的技术方案对比（方案A和方案B，各有具体参数）
2. 问题要求跨标准综合论证
3. 答案包含五段：现状诊断、多维冲突分析、多方案比选、折中综合方案、控制优先级链条

输出严格的纯JSON数组（确保闭合），每题约1500-2000字：
[{{"query":"【场景】...（≥300字，含方案A和方案B完整描述）\\n\\n【问题】...","expected_answer":"现状诊断：...\\n多维冲突分析：...\\n多方案比选：...\\n折中综合方案：...\\n控制优先级链条：...","expected_keywords":["kw1","kw2","kw3","kw4","kw5"],"source_standard":"标准1, 标准2","category":"综合评估/方案对比/多标准协调"}}]

标准文本（提取跨标准冲突的条款）：
{combined_std[:8000]}"""

    resp = client.chat.completions.create(
        model=STRONG_MODEL, messages=[{'role':'user','content':prompt_l3}],
        temperature=0.4, max_tokens=12000,
    )
    text = resp.choices[0].message.content
    batch_qs = robust_parse(text)
    all_l3.extend(batch_qs)
    print(f"  Batch {batch_num+1}: {len(batch_qs)} parsed")
    if not batch_qs:
        print(f"  RAW ends: ...{text[-200:]}")
        # Try to salvage partial array
        partial = re.search(r'\[[\s\S]*?(?:"[^"]*"\s*\]|\])\s*$', text)
        if partial:
            print(f"  Partial match: {len(partial.group(0))} chars")

l3_qs = all_l3
print(f"Total L3: {len(l3_qs)}")
