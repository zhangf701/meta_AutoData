"""Generate LaTeX inspection PDF from generated_eval_set_v3.json — L1/L2/L3 x20."""
import json, os, subprocess, random, re

DATA = r'D:\coding\meta_AutoData\data\questions\generated_eval_set_v3.json'
OUT_DIR = r'D:\coding\meta_AutoData\scripts\data_generation\quality_reports'
TEX_FILE = os.path.join(OUT_DIR, 'sample_v3_60.tex')

qs = json.load(open(DATA, encoding='utf-8'))
by_level = {}
for q in qs:
    by_level.setdefault(q['level'], []).append(q)

samples = {}
for lvl in ['L1', 'L2', 'L3']:
    samples[lvl] = random.sample(by_level[lvl], 20)

def esc(text):
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    return (text
        .replace('\\', r'\textbackslash{}')
        .replace('&', r'\&').replace('%', r'\%').replace('$', r'\$')
        .replace('#', r'\#').replace('_', r'\_')
        .replace('{', r'\{').replace('}', r'\}')
        .replace('^', r'\^{}').replace('~', r'\textasciitilde{}')
        .replace('>', r'\textgreater{}').replace('<', r'\textless{}'))

LEVEL_NAMES = {
    'L1': 'L1 — 直接型（答案在单一规范条款中可直接检索）',
    'L2': 'L2 — 推理型（需条件判断或跨参数推理）',
    'L3': 'L3 — 综合型（需引用两份以上规范综合判断）',
}

doc = r'''\documentclass[12pt,a4paper]{ctexart}
\usepackage{geometry}
\geometry{left=2cm,right=2cm,top=1.8cm,bottom=2cm}
\usepackage{fancyhdr}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}

\fancyhf{}
\fancyhead[L]{generated\_eval\_set\_v3 抽样检查}
\fancyhead[R]{\thepage}
\pagestyle{fancy}

\title{\textbf{电力审查数据集 v3 — 抽样检查（60题）}}
\author{L1/L2/L3 各随机抽取20题}
\date{\today}

\begin{document}
\maketitle
\thispagestyle{fancy}

\section*{说明}
从 generated\_eval\_set\_v3.json（300题）中随机抽取 L1/L2/L3 各20题，供人工检查质量。
\vspace{6pt}

'''

for lvl in ['L1', 'L2', 'L3']:
    doc += f'\n\\section*{{{LEVEL_NAMES[lvl]}}}\n'
    for i, q in enumerate(samples[lvl], 1):
        qid = q.get('question_id', f'{lvl}-???')
        query = esc(q.get('query', ''))
        answer = esc(q.get('expected_answer', ''))
        keywords = '、'.join(esc(k) for k in q.get('expected_keywords', []))
        std = esc(q.get('source_standard', '未标注'))
        cat = esc(q.get('category', ''))
        topic = esc(q.get('topic', ''))

        doc += f'''
\\noindent\\textbf{{{qid}}} \\hfill \\textit{{{cat}}}
\\begin{{description}}[leftmargin=1.5em,style=nextline,font=\\normalfont]
  \\item[Q:] {query}
  \\item[A:] {answer}
  \\item[KW:] {keywords}
  \\item[Std:] {std}
'''
        if topic:
            doc += f'  \\item[Topic:] {topic}\n'
        doc += r'\end{description}' + '\n\\vspace{4pt}\n\\hrule\\vspace{6pt}\n'

doc += r'\end{document}'

with open(TEX_FILE, 'w', encoding='utf-8') as f:
    f.write(doc)
print(f'LaTeX: {TEX_FILE} ({len(doc):,} chars)')

# Compile
os.chdir(OUT_DIR)
pdf = TEX_FILE.replace('.tex', '.pdf')
for _ in range(2):
    subprocess.run(['xelatex', '-interaction=nonstopmode', '-jobname',
                    'sample_v3_60', TEX_FILE],
                   capture_output=True, timeout=90)
print(f'PDF: {pdf} ({os.path.getsize(pdf):,} bytes)')
