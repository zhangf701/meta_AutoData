"""Generate LaTeX inspection document from sampled questions."""
import json, os, subprocess, random

DATA = r'D:\coding\meta_AutoData\data\questions\generated_eval_set_v1.json'
OUT_DIR = r'D:\coding\meta_AutoData\scripts\data_generation\quality_reports'
TEX_FILE = os.path.join(OUT_DIR, 'sample_inspection.tex')

qs = json.load(open(DATA, encoding='utf-8'))
by_level = {}
for q in qs:
    by_level.setdefault(q['level'], []).append(q)

samples = {}
for lvl in ['L1', 'L2', 'L3']:
    samples[lvl] = random.sample(by_level[lvl], min(5, len(by_level[lvl])))

# LaTeX escaping
def esc(text):
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    return (text
        .replace('\\', r'\textbackslash{}')
        .replace('&', r'\&')
        .replace('%', r'\%')
        .replace('$', r'\$')
        .replace('#', r'\#')
        .replace('_', r'\_')
        .replace('{', r'\{')
        .replace('}', r'\}')
        .replace('^', r'\^{}')
        .replace('~', r'\textasciitilde{}')
        .replace('>', r'\textgreater{}')
        .replace('<', r'\textless{}')
    )

LEVEL_NAMES = {
    'L1': 'L1 — 直接型（答案在单一规范条款中可直接检索）',
    'L2': 'L2 — 推理型（需条件判断或跨参数推理）',
    'L3': 'L3 — 综合型（需引用两份以上规范综合判断）',
}

doc = r'''\documentclass[12pt,a4paper]{ctexart}
\usepackage{geometry}
\geometry{left=2.5cm,right=2.5cm,top=2cm,bottom=2cm}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}

\titleformat{\section}{\Large\bfseries}{}{0em}{}
\titlespacing{\section}{0pt}{18pt}{8pt}

\title{\textbf{电力审查数据集 — 抽样检查}}
\author{自动生成 · AutoData Pipeline}
\date{\today}

\begin{document}
\maketitle
\thispagestyle{fancy}
\fancyhf{}
\fancyhead[L]{电力审查数据集抽样检查}
\fancyhead[R]{\thepage}

\section*{说明}
以下从自动生成的213道电力系统设计审查数据集中，随机抽取L1/L2/L3各5道（共15道），供人工检查质量。
数据集由 DeepSeek-V4 从 GB~38755-2019、DL/T~5429-2009、DL/T~5218-2012 三份标准全文生成。

\tableofcontents
\newpage
'''

for lvl in ['L1', 'L2', 'L3']:
    doc += f'\n\\section{{{LEVEL_NAMES[lvl]}}}\n'
    for i, q in enumerate(samples[lvl], 1):
        qid = q.get('question_id', f'{lvl}-???')
        query = esc(q.get('query', ''))
        answer = esc(q.get('expected_answer', ''))
        keywords = '、'.join(esc(k) for k in q.get('expected_keywords', []))
        std = esc(q.get('source_standard', '未标注'))
        cat = esc(q.get('category', ''))
        clauses = '、'.join(esc(c) for c in q.get('rubric_clauses', []))

        doc += f'''
\\subsection*{{{qid}  \\hfill {cat}}}
\\begin{{description}}[leftmargin=2em,style=nextline]
  \\item[\\textbf{{问题}}] {query}
  \\item[\\textbf{{答案}}] {answer}
  \\item[\\textbf{{关键词}}] {keywords}
  \\item[\\textbf{{来源标准}}] {std}
'''
        if clauses:
            doc += f'  \\item[\\textbf{{条款引用}}] {clauses}\n'
        doc += r'\end{description}' + '\n\\vspace{6pt}\n'

doc += r'\end{document}'

with open(TEX_FILE, 'w', encoding='utf-8') as f:
    f.write(doc)
print(f'LaTeX written: {TEX_FILE} ({len(doc):,} chars)')

# Compile with xelatex
os.chdir(OUT_DIR)
for _ in range(2):  # Two passes for TOC
    result = subprocess.run(
        ['xelatex', '-interaction=nonstopmode', TEX_FILE],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f'xelatex error: {result.stderr[-500:]}')
        break

pdf = TEX_FILE.replace('.tex', '.pdf')
print(f'PDF: {pdf}')
print(f'Exists: {os.path.exists(pdf)}')
