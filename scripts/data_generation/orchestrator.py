"""Orchestrator: End-to-end data generation pipeline.

Phases:
1. Load source materials
2. Extract/load clauses
3. Generate L1/L2/L3 questions
4. Build rubric + gold chunks
5. Generate quality report
"""

import json
import os
import sys
import time
from datetime import datetime

from data_generation.config import (
    TARGET_COUNTS, CLAUSES_OUTPUT, QUESTIONS_OUTPUT,
    GOLD_OUTPUT, QUALITY_REPORT_DIR,
)
from data_generation.source_loader import load_all_sources
from data_generation.clause_extractor import run_extraction
from data_generation.agents.challenger import Challenger
from data_generation.agents.quality_verifier import QualityVerifier
from data_generation.agents.loop_judge import LoopJudge
from data_generation.solvers.weak_solver import WeakSolver
from data_generation.solvers.strong_solver import StrongSolver
from data_generation.generators.l1_generator import L1Generator
from data_generation.generators.l2_generator import L2Generator
from data_generation.generators.l3_generator import L3Generator
from data_generation.utils.id_manager import IDManager
from data_generation.utils.rubric_builder import build_rubric
from data_generation.utils.format_validator import validate_question


def load_or_extract_clauses():
    """Load cached clauses or run extraction."""
    if os.path.exists(CLAUSES_OUTPUT):
        print(f"Loading cached clauses from {CLAUSES_OUTPUT}")
        with open(CLAUSES_OUTPUT, "r", encoding="utf-8") as f:
            data = json.load(f)
        clauses = data.get("clauses", [])
        print(f"  Loaded {len(clauses)} clauses")
        return clauses

    print("No cached clauses found. Running extraction...")
    return run_extraction()


def build_gold_entry(question, all_sections):
    """Build a gold_chunks entry for a generated question.

    Matches answer keywords against source sections to find
    candidate chunks (analogous to RAG retrieval).
    """
    answer = question.get("expected_answer", "")
    keywords = question.get("expected_keywords", [])

    # Simple keyword-based candidate matching
    candidates = []
    for i, sec in enumerate(all_sections[:200]):  # Limit to first 200 sections
        sec_text = sec.get("text", "")
        hits = [kw for kw in keywords if kw in sec_text]
        if hits:
            candidates.append({
                "chunk_index": i,
                "chunk_id": f"source_section_{sec.get('source_file','')[:30]}_{i}",
                "score": len(hits) / max(len(keywords), 1),
                "signals": {
                    "keyword_hits": hits,
                    "keyword_hit_count": len(hits),
                    "keyword_total": len(keywords),
                    "answer_match": "exact" if answer[:50] in sec_text else "none",
                    "source_match": any(s in sec_text for s in ["GB", "DL/T", "Q/GDW"]),
                },
                "preview": sec_text[:200],
            })

    # Sort by score
    candidates.sort(key=lambda x: -x["score"])
    top_candidates = candidates[:8]  # Keep top 8

    gold_indices = [c["chunk_index"] for c in top_candidates[:3]]  # Top 3 as gold

    # Build rubric
    rubric = build_rubric(answer, question.get("level", "L2"))

    return {
        "question_id": question.get("question_id", ""),
        "expected_answer": answer,
        "expected_keywords": question.get("expected_keywords", []),
        "source_standard": question.get("source_standard", ""),
        "candidates": top_candidates,
        "gold_chunk_indices": gold_indices,
        "gold_chunk_ids": [c["chunk_id"] for c in top_candidates[:3]],
        "annotator_notes": f"[auto-generated via Agentic Self-Instruct] clause_source={question.get('clause_source','')}",
        "status": "auto_generated",
        "gold_chunk_indices_minimal": gold_indices[:2],
        "rubric_clauses": rubric["rubric_clauses"],
        "rubric_judgments": rubric["rubric_judgments"],
        "rubric_refusal_expected": rubric["rubric_refusal_expected"],
        "scoring_version": rubric["scoring_version"],
    }


def generate_quality_report(all_questions, gold_entries, stats):
    """Generate markdown quality report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(QUALITY_REPORT_DIR, f"report_{timestamp}.md")

    by_level = {}
    for q in all_questions:
        lvl = q.get("level", "L2")
        by_level.setdefault(lvl, []).append(q)

    lines = [
        f"# 自动生成数据集质量报告",
        f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**总题目数:** {len(all_questions)}",
        f"",
        f"## 概览",
        f"| 层级 | 生成数 | 带rubric_clauses | 平均clause数 |",
        f"|------|--------|-----------------|-------------|",
    ]

    for lvl in ["L1", "L2", "L3"]:
        qs = by_level.get(lvl, [])
        with_clauses = sum(1 for q in qs if q.get("rubric_clauses"))
        avg_clauses = sum(len(q.get("rubric_clauses", [])) for q in qs) / max(len(qs), 1)
        lines.append(f"| {lvl} | {len(qs)} | {with_clauses} | {avg_clauses:.1f} |")

    lines.extend([
        f"",
        f"## 格式验证",
        f"| 层级 | 通过 | 失败 |",
        f"|------|------|------|",
    ])
    for lvl in ["L1", "L2", "L3"]:
        qs = by_level.get(lvl, [])
        passed = sum(1 for q in qs if validate_question(q, lvl)[0])
        lines.append(f"| {lvl} | {passed} | {len(qs) - passed} |")

    lines.extend([
        f"",
        f"## L3 场景内联检查",
    ])
    l3_qs = by_level.get("L3", [])
    l3_with_scenario = sum(1 for q in l3_qs if len(q.get("query", "")) >= 300)
    l3_with_both_schemes = sum(1 for q in l3_qs
                               if "方案A" in q.get("query", "") and "方案B" in q.get("query", ""))
    lines.append(f"- 场景≥300字: {l3_with_scenario}/{len(l3_qs)}")
    lines.append(f"- 含方案A和方案B: {l3_with_both_schemes}/{len(l3_qs)}")

    lines.extend([
        f"",
        f"## 统计详情",
        f"```json",
        json.dumps(stats, ensure_ascii=False, indent=2),
        f"```",
        f"",
        f"## 样本展示",
    ])

    import random
    for lvl in ["L1", "L2", "L3"]:
        qs = by_level.get(lvl, [])
        if qs:
            sample = random.choice(qs)
            lines.extend([
                f"### {lvl} 样本: {sample.get('question_id', '?')}",
                f"**Query:** {sample.get('query', '')[:300]}...",
                f"**Answer:** {sample.get('expected_answer', '')[:300]}...",
                f"**Keywords:** {sample.get('expected_keywords', [])}",
                f"**Clauses:** {sample.get('rubric_clauses', [])}",
                f"",
            ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Quality report saved to: {report_path}")
    return report_path


def main():
    print("="*60)
    print("Autodata Pipeline: Power Grid Q&A Dataset Generation")
    print("="*60)

    # Phase 1-2: Load sources and clauses
    print("\n[Phase 1-2] Loading source materials and clauses...")
    sources, all_sections = load_all_sources()

    # Load scenario materials (extracted from design/plan/feedback docs)
    scenario_path = CLAUSES_OUTPUT.replace("clauses_v1.json", "scenario_materials_v1.json")
    scenario_materials = []
    if os.path.exists(scenario_path):
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario_materials = json.load(f).get("materials", [])
        print(f"  Loaded {len(scenario_materials)} scenario materials")

    clauses = load_or_extract_clauses()

    if not clauses:
        print("ERROR: No clauses extracted. Aborting.")
        return

    # Phase 3-4: Initialize components
    print(f"\n[Phase 3-4] Initializing pipeline components...")
    weak_solver = WeakSolver()
    strong_solver = StrongSolver()
    id_manager = IDManager()

    # Phase 5: Generate questions
    print(f"\n[Phase 5] Generating questions...")
    print(f"  Target: L1={TARGET_COUNTS['L1']}, L2={TARGET_COUNTS['L2']}, L3={TARGET_COUNTS['L3']}")

    t0 = time.time()
    all_questions = []

    # L1 (template-based, fast — uses standard clauses only)
    print("\n  --- L1 Generation ---")
    l1_gen = L1Generator(clauses, id_manager)
    l1_questions = l1_gen.generate(TARGET_COUNTS["L1"])
    all_questions.extend(l1_questions)

    # L2 (LLM-based, slower — injects design doc scenario materials)
    print("\n  --- L2 Generation ---")
    l2_gen = L2Generator(clauses, id_manager, weak_solver, strong_solver, scenario_materials)
    l2_questions = l2_gen.generate(TARGET_COUNTS["L2"])
    all_questions.extend(l2_questions)

    # L3 (LLM-based + narrowing — injects conflict scenarios)
    print("\n  --- L3 Generation ---")
    l3_gen = L3Generator(clauses, id_manager, weak_solver, strong_solver, scenario_materials)
    l3_questions = l3_gen.generate(TARGET_COUNTS["L3"])
    all_questions.extend(l3_questions)

    elapsed = time.time() - t0
    print(f"\n  Generation completed in {elapsed:.0f}s "
          f"({len(all_questions)} questions total)")

    # Phase 5.3: Build gold entries
    print(f"\n[Phase 5.3] Building gold chunk entries...")
    gold_entries = []
    for q in all_questions:
        gold = build_gold_entry(q, all_sections)
        # Merge rubric fields back into question
        for key in ["rubric_clauses", "rubric_judgments", "rubric_refusal_expected", "scoring_version"]:
            q[key] = gold[key]
        gold_entries.append(gold)

    # Save outputs
    print(f"\n[Output] Saving datasets...")
    with open(QUESTIONS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"  Questions: {QUESTIONS_OUTPUT} ({len(all_questions)} items)")

    with open(GOLD_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(gold_entries, f, ensure_ascii=False, indent=2)
    print(f"  Gold chunks: {GOLD_OUTPUT} ({len(gold_entries)} items)")

    # Phase 6: Quality report
    print(f"\n[Phase 6] Generating quality report...")
    stats = {
        "total_questions": len(all_questions),
        "by_level": {
            "L1": len(l1_questions),
            "L2": len(l2_questions),
            "L3": len(l3_questions),
        },
        "total_clauses_used": len(clauses),
        "generation_time_seconds": int(elapsed),
    }
    report_path = generate_quality_report(all_questions, gold_entries, stats)

    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  Questions: {QUESTIONS_OUTPUT}")
    print(f"  Gold:      {GOLD_OUTPUT}")
    print(f"  Report:    {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
