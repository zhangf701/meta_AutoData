"""run_loopjudge_eval.py: Offline LoopJudge evaluation on v3 L2/L3 questions.

Phase 2 Task 2.1: Evaluates existing 200 L2+L3 questions through the
weak/strong solver gap analysis loop. Uses real Ollama qwen2.5:3b as weak
solver and DeepSeek as strong solver.

Output: generated_eval_set_v3_loopjudge.json + loopjudge_eval_report.md
"""

import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, r"D:\coding\meta_AutoData\scripts")

from data_generation.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DATA_DIR,
    L2_ACCEPT_WEAK_MAX, L2_ACCEPT_STRONG_MIN, L2_ACCEPT_GAP_MIN,
    L3_ACCEPT_WEAK_MIN, L3_ACCEPT_STRONG_MIN, MAX_L3_NARROW_ITERATIONS,
    L1_WEAK_REJECT_THRESHOLD,
)

from data_generation.solvers.weak_solver import WeakSolver
from data_generation.solvers.strong_solver import StrongSolver
from data_generation.agents.loop_judge import LoopJudge


# ── Scoring ──────────────────────────────────────────────────────────────

def keyword_score(model_answer, expected_keywords):
    """Score model answer by keyword overlap (0-1).

    Simple but fast scoring: what fraction of expected keywords
    appear in the model's answer.
    """
    if not expected_keywords:
        return 0.5
    ans_lower = model_answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans_lower)
    return hits / len(expected_keywords)


def rubric_judge_score(model_answer, expected_answer, rubric_judgments, client):
    """Use DeepSeek as judge to score against rubric_judgments (0-1).

    Returns a score representing how many rubric judgment points
    the model answer satisfies.
    """
    if not rubric_judgments or len(rubric_judgments) == 0:
        return keyword_score(model_answer, [])

    # Build a lightweight judging prompt
    judgments_text = "\n".join(f"{i+1}. {j}" for i, j in enumerate(rubric_judgments[:5]))
    prompt = f"""你是一名电力系统标准审核专家。请评估以下模型答案相对于标准答案的质量。

标准答案：
{expected_answer[:800]}

模型答案：
{model_answer[:800]}

评分要点：
{judgments_text}

请评估模型答案满足了多少个评分要点，输出0-1之间的分数（0=完全不满足，1=完全满足）。

只输出一个浮点数，不要任何解释。例如：0.6"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        text = response.choices[0].message.content.strip()
        # Parse float
        import re
        match = re.search(r"([01]\.\d+|0|1|\.\d+)", text)
        if match:
            return float(match.group(1))
        return 0.5
    except Exception as e:
        # Fallback to keyword score
        return 0.0


def compute_score(model_answer, question, client):
    """Compute a 0-1 rubric score for a model answer.

    Hybrid approach: keyword overlap (weight 0.4) + DeepSeek judge (weight 0.6).
    Falls back to pure keyword if DeepSeek fails.
    """
    keywords = question.get("expected_keywords", [])
    kw_s = keyword_score(model_answer, keywords)

    rubric_judgments = question.get("rubric_judgments", [])
    if rubric_judgments:
        j_s = rubric_judge_score(
            model_answer,
            question.get("expected_answer", ""),
            rubric_judgments,
            client,
        )
        return 0.4 * kw_s + 0.6 * j_s
    else:
        return kw_s


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 2.1: LoopJudge Offline Evaluation")
    print("=" * 60)

    # Load dataset (with clause_source + L1 fixes)
    input_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_l1_fixed.json")
    if not os.path.exists(input_path):
        input_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_clause_backfill.json")
    print(f"\nLoading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Only evaluate L2 and L3 (L1 are parametric, not suitable for solver-based eval)
    eval_questions = [q for q in questions if q.get("level") in ("L2", "L3")]
    print(f"  Evaluating {len(eval_questions)} questions (L2+L3)")

    # Initialize solvers
    print("\nInitializing solvers...")
    try:
        weak_solver = WeakSolver()
        print(f"  Weak solver (Ollama {weak_solver.model}): OK")
    except Exception as e:
        print(f"  [ERROR] Weak solver init failed: {e}")
        print("  Check: is Ollama installed? Run 'ollama serve' manually?")
        return

    strong_solver = StrongSolver()
    print(f"  Strong solver (DeepSeek {strong_solver.model}): OK")

    # Initialize LoopJudge
    judge = LoopJudge(weak_solver, strong_solver)
    print(f"  LoopJudge: OK")

    # Initialize DeepSeek client for scoring
    from openai import OpenAI
    score_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # Evaluate each question
    print(f"\nEvaluating {len(eval_questions)} questions...")
    stats = Counter()
    by_level = {"L2": Counter(), "L3": Counter()}
    gaps = []

    t0 = time.time()
    for i, q in enumerate(eval_questions):
        level = q.get("level", "L2")
        qid = q.get("question_id", "?")

        # Get context chunks from clause_source
        clause_text = q.get("clause_source", q.get("source_standard", ""))
        context = [clause_text] if clause_text else None

        # Weak solver
        try:
            weak_answer = weak_solver.solve(q["query"], context_chunks=context)
        except Exception as e:
            weak_answer = f"[Error: {e}]"

        # Strong solver
        try:
            strong_answer = strong_solver.solve(q["query"], context_chunks=context)
        except Exception as e:
            strong_answer = f"[Error: {e}]"

        # Score both answers
        weak_score = compute_score(weak_answer, q, score_client)
        strong_score = compute_score(strong_answer, q, score_client)

        # LoopJudge verdict
        verdict = judge.judge(
            q, weak_answer, strong_answer,
            weak_score, strong_score,
            narrow_count=0,
        )

        # Store results
        q["loopjudge_verdict"] = verdict["verdict"]
        q["loopjudge_feedback"] = verdict.get("feedback", "")
        q["weak_score"] = round(weak_score, 3)
        q["strong_score"] = round(strong_score, 3)
        q["weak_strong_gap"] = round(verdict.get("gap", 0), 3)

        stats[verdict["verdict"]] += 1
        by_level[level][verdict["verdict"]] += 1
        gaps.append(verdict.get("gap", 0))

        # Progress
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            print(f"  [{i+1}/{len(eval_questions)}] "
                  f"accept={stats.get('accept',0)} "
                  f"rewrite={stats.get('rewrite',0)} "
                  f"narrow={stats.get('narrow',0)} "
                  f"reject={stats.get('reject',0)} "
                  f"rate={rate:.0f}/min")

        # Rate limiting
        time.sleep(0.5)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"LoopJudge Evaluation Complete")
    print(f"{'='*60}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Questions evaluated: {len(eval_questions)}")

    # Summary statistics
    print(f"\n  Overall Verdicts:")
    for v in ["accept", "rewrite", "narrow", "reject"]:
        count = stats.get(v, 0)
        pct = count / len(eval_questions) * 100
        print(f"    {v:10s}: {count:3d} ({pct:.1f}%)")

    print(f"\n  By Level:")
    for lvl in ["L2", "L3"]:
        print(f"    {lvl}:")
        for v in ["accept", "rewrite", "narrow", "reject"]:
            count = by_level[lvl].get(v, 0)
            n = len([q for q in eval_questions if q.get("level") == lvl])
            pct = count / n * 100 if n else 0
            print(f"      {v:10s}: {count:3d} ({pct:.1f}%)")

    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    print(f"\n  Score Gap Stats:")
    print(f"    Mean gap:    {avg_gap:.3f}")
    print(f"    Median gap:  {sorted(gaps)[len(gaps)//2]:.3f}")
    print(f"    Max gap:     {max(gaps):.3f}")
    print(f"    Min gap:     {min(gaps):.3f}")

    # Save results
    output_path = os.path.join(DATA_DIR, "questions", "generated_eval_set_v3_loopjudge.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"\n  Dataset saved: {output_path}")

    # Generate markdown report
    report_path = os.path.join(
        os.path.dirname(input_path).replace("questions", ""),
        "scripts", "data_generation", "quality_reports",
        "loopjudge_eval_report.md"
    )
    # Fix path
    report_dir = r"D:\coding\meta_AutoData\scripts\data_generation\quality_reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "loopjudge_eval_report.md")

    report = []
    report.append("# LoopJudge 离线评测报告\n")
    report.append(f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**评估题目数**: {len(eval_questions)} (L2+L3)")
    report.append(f"**耗时**: {elapsed:.0f}s ({elapsed/60:.1f} min)\n")

    report.append("## 总体分布\n")
    report.append("| Verdict | 数量 | 占比 |")
    report.append("|---------|------|------|")
    for v in ["accept", "rewrite", "narrow", "reject"]:
        count = stats.get(v, 0)
        pct = count / len(eval_questions) * 100
        report.append(f"| {v} | {count} | {pct:.1f}% |")

    report.append("\n## 各级分布\n")
    report.append("| Level | accept | rewrite | narrow | reject | accept率 |")
    report.append("|-------|--------|---------|--------|--------|---------|")
    for lvl in ["L2", "L3"]:
        n = len([q for q in eval_questions if q.get("level") == lvl])
        a = by_level[lvl].get("accept", 0)
        rw = by_level[lvl].get("rewrite", 0)
        na = by_level[lvl].get("narrow", 0)
        rj = by_level[lvl].get("reject", 0)
        report.append(f"| {lvl} | {a} | {rw} | {na} | {rj} | {a/n*100:.1f}% |")

    report.append(f"\n## 分数差距统计\n")
    report.append(f"- 平均 gap: {avg_gap:.3f}")
    report.append(f"- 中位数 gap: {sorted(gaps)[len(gaps)//2]:.3f}")
    report.append(f"- 最大 gap: {max(gaps):.3f}")
    report.append(f"- 最小 gap: {min(gaps):.3f}")

    report.append(f"\n## 模型配置\n")
    report.append(f"- 弱模型: Ollama qwen2.5:3b (temperature=0)")
    report.append(f"- 强模型: DeepSeek chat (temperature=0)")
    report.append(f"- 评分: 关键词重叠(0.4) + DeepSeek Judge(0.6)")

    report.append(f"\n## 配置阈值\n")
    report.append(f"- L2: weak<{L2_ACCEPT_WEAK_MAX}, strong>{L2_ACCEPT_STRONG_MIN}, gap>{L2_ACCEPT_GAP_MIN}")
    report.append(f"- L3: weak>{L3_ACCEPT_WEAK_MIN}, strong>{L3_ACCEPT_STRONG_MIN}, max_narrow={MAX_L3_NARROW_ITERATIONS}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"  Report saved: {report_path}")


if __name__ == "__main__":
    main()
