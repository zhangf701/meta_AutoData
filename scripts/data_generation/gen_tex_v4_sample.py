"""Generate LaTeX inspection PDF from generated_eval_set_v4.json — L1/L2/L3 x10.

Displays ALL fields including new v4 additions:
clause_source, rubric_clauses, rubric_judgments, sub_category, and
LoopJudge verdict/score/gap for L2/L3 questions.
"""

import json
import os
import random
import subprocess
import sys
import time

DATA = r"D:\coding\meta_AutoData\data\questions\generated_eval_set_v4.json"
OUT_DIR = r"D:\coding\meta_AutoData\scripts\data_generation\quality_reports"
TEX_FILE = os.path.join(OUT_DIR, "sample_v4_30.tex")
PDF_FILE = os.path.join(OUT_DIR, "sample_v4_30.pdf")

sys.path.insert(0, r"D:\coding\meta_AutoData\scripts")
from data_generation.config import DATA_DIR


def esc(text):
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    return (text
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
        .replace("#", r"\#").replace("_", r"\_")
        .replace("{", r"\{").replace("}", r"\}")
        .replace("^", r"\^{}").replace("~", r"\textasciitilde{}")
        .replace(">", r"\textgreater{}").replace("<", r"\textless{}"))


def format_list(items, max_items=8):
    """Format a list of strings for LaTeX display."""
    if not items:
        return "（无）"
    items = items[:max_items]
    return "、".join(esc(str(i)) for i in items)


def format_loopjudge(q):
    """Format LoopJudge fields if present."""
    if "loopjudge_verdict" not in q:
        return ""
    verdict = q.get("loopjudge_verdict", "")
    ws = q.get("weak_score", 0)
    ss = q.get("strong_score", 0)
    gap = q.get("weak_strong_gap", 0)
    fb = q.get("loopjudge_feedback", "")
    verdict_map = {"accept": "通过", "rewrite": "需重写", "narrow": "需收窄", "reject": "拒绝"}
    v_cn = verdict_map.get(verdict, verdict)
    lines = [
        r"  \item[LJ判定:] " + f"{v_cn} (弱={ws:.2f}, 强={ss:.2f}, 差距={gap:.2f})",
    ]
    if fb:
        lines.append(r"  \item[LJ反馈:] " + esc(fb[:200]))
    return "\n".join(lines)


LEVEL_NAMES = {
    "L1": "L1 — 直接型（参数检索，答案在单一规范条款中可直接检索）",
    "L2": "L2 — 推理型（需条件判断或跨参数推理）",
    "L3": "L3 — 综合型（需引用两份以上规范综合判断，五段式结构）",
}


def main():
    print(f"Loading: {DATA}")
    qs = json.load(open(DATA, encoding="utf-8"))

    # Sample
    random.seed(123)
    by_level = {}
    for q in qs:
        by_level.setdefault(q["level"], []).append(q)

    samples = {}
    for lvl in ["L1", "L2", "L3"]:
        samples[lvl] = random.sample(by_level[lvl], 10)
        print(f"  {lvl}: {len(samples[lvl])} sampled")

    # ── Build LaTeX ─────────────────────────────────────────────────
    doc = r"""\documentclass[11pt,a4paper]{ctexart}
\usepackage{geometry}
\geometry{left=1.8cm,right=1.8cm,top=1.5cm,bottom=2cm}
\usepackage{fancyhdr}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}

\fancyhf{}
\fancyhead[L]{generated\_eval\_set\_v4 抽样检查}
\fancyhead[R]{\thepage}
\pagestyle{fancy}

\title{\textbf{电力审查数据集 v4 — 抽样检查（30题）}}
\author{L1/L2/L3 各随机抽取10题 \quad 完整字段展示}
\date{\today}

\begin{document}
\maketitle
\thispagestyle{fancy}

\section*{说明}
从 generated\_eval\_set\_v4.json（300题）中随机抽取 L1/L2/L3 各10题，
展示全部字段，供人工质量检查。

v4 相对于 v3 新增字段：
\textbf{clause\_source}（条款溯源）、
\textbf{sub\_category}（细粒度技术子类别）、
\textbf{LoopJudge verdict/score/gap}（弱/强模型评分闭环）。

\vspace{6pt}
"""

    for lvl in ["L1", "L2", "L3"]:
        doc += f"\n\\section*{{{LEVEL_NAMES[lvl]}}}\n"

        for i, q in enumerate(samples[lvl], 1):
            qid = q.get("question_id", f"{lvl}-???")
            query = esc(q.get("query", ""))
            answer = esc(q.get("expected_answer", ""))
            keywords = format_list(q.get("expected_keywords", []))
            std = esc(q.get("source_standard", "未标注"))
            clause_src = esc(q.get("clause_source", ""))
            cat = esc(q.get("category", ""))
            sub_cat = esc(q.get("sub_category", ""))
            topic = esc(q.get("topic", ""))
            grading = esc(q.get("grading_method", ""))
            knowledge = esc(q.get("knowledge_base", ""))

            # Rubric
            rubric_clauses_text = format_list(q.get("rubric_clauses", []))
            rubric_judgments_text = format_list(q.get("rubric_judgments", []), max_items=5)

            # Options (L1 multiple choice)
            options_block = ""
            opts = q.get("options", [])
            if opts:
                options_block = r"  \item[选项:] " + r" \\ ".join(esc(o) for o in opts) + "\n"
            ans_block = ""
            ans_raw = q.get("answer", "")
            if ans_raw:
                ans_block = r"  \item[正确选项:] " + esc(str(ans_raw)) + "\n"

            # LoopJudge (L2/L3 only)
            lj_block = format_loopjudge(q)

            doc += f"""
\\noindent\\textbf{{{qid}}} \\hfill \\textbf{{{cat}}} $|$ \\textit{{{sub_cat}}}
\\begin{{description}}[leftmargin=1.5em,style=nextline,font=\\normalfont]
  \\item[Q:] {query}
  \\item[A:] {answer}
  \\item[KW:] {keywords}
  \\item[标准:] {std}
  \\item[条款溯源:] {clause_src}
  \\item[主题:] {topic}
  \\item[评分方式:] {grading}
{options_block}{ans_block}  \\item[Rubric条款:] {rubric_clauses_text}
  \\item[Rubric判据:] {rubric_judgments_text}
{lj_block}
\\end{{description}}
\\vspace{{4pt}}
\\hrule
\\vspace{{6pt}}
"""

    doc += r"""
\end{document}
"""

    # Write tex file
    with open(TEX_FILE, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"\nTeX written: {TEX_FILE} ({len(doc):,} chars)")

    # Compile with xelatex (required by ctexart for Chinese support)
    print("Compiling with xelatex...")
    os.chdir(OUT_DIR)
    for _ in range(2):  # Two passes for TOC/cross-refs
        result = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", TEX_FILE],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            # Extract relevant errors
            errors = [l for l in result.stdout.split("\n") if l.startswith("!")]
            if errors:
                print(f"  LaTeX errors: {errors[:5]}")
            else:
                print(f"  xelatex exit code {result.returncode}")
            break
    else:
        print(f"  Compiled successfully")

    # Check PDF
    if os.path.exists(PDF_FILE):
        size_kb = os.path.getsize(PDF_FILE) / 1024
        print(f"\nPDF: {PDF_FILE} ({size_kb:.0f} KB)")
    else:
        print(f"\nPDF NOT generated. Check log: {TEX_FILE.replace('.tex', '.log')}")


if __name__ == "__main__":
    main()
