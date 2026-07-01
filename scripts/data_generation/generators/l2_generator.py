"""L2 Generator: Scenario reasoning questions with Challenger + QV + LoopJudge.

Based on Autodata CS pipeline pattern: questions may be too easy,
LoopJudge rejects with "increase reasoning depth" feedback.

LoopJudge is ENABLED by default. Set fast_mode=True to skip solver evaluation
for rapid iteration (MVP behavior).
"""

from data_generation.agents.challenger import Challenger
from data_generation.agents.quality_verifier import QualityVerifier
from data_generation.agents.loop_judge import LoopJudge
from data_generation.utils.escaping_fixer import fix_question_escaping
from data_generation.utils.format_validator import validate_question


def _keyword_score(model_answer, expected_keywords):
    """Fast keyword-based scoring (0-1) for LoopJudge decisions."""
    if not expected_keywords:
        return 0.5
    ans_lower = model_answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans_lower)
    return hits / len(expected_keywords)


class L2Generator:
    """Generate L2 questions using agentic self-instruct loop.

    LoopJudge weak/strong solver gap analysis is ENABLED by default.
    Set fast_mode=True to skip solver eval (accepts if QV passes).
    """

    def __init__(self, clauses, id_manager, weak_solver, strong_solver,
                 scenario_materials=None, fast_mode=False):
        self.l2_clauses = [c for c in clauses if c.get("difficulty") == "L2"]
        self.id_manager = id_manager
        self.challenger = Challenger()
        self.verifier = QualityVerifier()
        self.judge = LoopJudge(weak_solver, strong_solver)
        self.weak_solver = weak_solver
        self.strong_solver = strong_solver
        self.scenario_materials = scenario_materials or []
        self.fast_mode = fast_mode

    def _pick_scenario_for_clause(self, clause, top_k=2):
        """Pick top-k relevant scenario materials with multi-factor scoring.

        Scoring dimensions:
        - Topic keyword match in scenario text
        - Key terms overlap between clause and scenario
        - Character overlap between clause text and scenario text

        Returns dict with merged top-k content, or None if no high-confidence
        match (best score < 3).
        """
        topic = clause.get("topic", "")
        clause_text = clause.get("clause_text", "")
        if not self.scenario_materials:
            return None

        scored = []
        clause_chars = set(clause_text)
        for sm in self.scenario_materials:
            score = 0
            sm_text = sm.get("preview", sm.get("content", ""))
            if not sm_text:
                continue

            # Topic match
            if topic in sm_text:
                score += 3
            # Keyword overlap
            for kw in clause.get("key_terms", []):
                if kw in sm_text:
                    score += 2
            # Character overlap bonus (0-1 normalized)
            sm_chars = set(sm_text)
            overlap = len(clause_chars & sm_chars)
            score += overlap / max(len(clause_chars), 1)

            if score > 0:
                scored.append((score, sm))

        scored.sort(key=lambda x: -x[0])

        # No high-confidence match: return None so generator invents scenario
        if not scored or scored[0][0] < 3:
            return None

        # Merge top-k: concatenate previews and combine params/conflicts
        top = [s for _, s in scored[:top_k]]
        merged_preview = "\n\n".join(
            t.get("preview", t.get("content", ""))[:400] for t in top
        )
        merged_conflicts = list(set(
            c for t in top for c in t.get("conflict_types", [])
        ))
        merged_params = list(dict.fromkeys(
            p for t in top for p in t.get("sample_params", [])
        ))[:20]

        return {
            "preview": merged_preview,
            "content": merged_preview,
            "conflict_types": merged_conflicts,
            "sample_params": merged_params,
            "type": "merged_scenario",
        }

    def _run_loopjudge(self, question, clause):
        """Run weak/strong solver evaluation and return verdict.

        Uses keyword-based scoring for speed. The gap between weak
        (qwen2.5:3b) and strong (DeepSeek) scores determines
        whether the question is accept/rewrite/reject.
        """
        clause_text = clause.get("clause_text", "")[:500]
        context = [clause_text] if clause_text else None
        query = question.get("query", "")

        # Solve with weak model
        try:
            weak_answer = self.weak_solver.solve(query, context_chunks=context)
        except Exception:
            weak_answer = ""

        # Solve with strong model
        try:
            strong_answer = self.strong_solver.solve(query, context_chunks=context)
        except Exception:
            strong_answer = ""

        # Score both
        keywords = question.get("expected_keywords", [])
        weak_score = _keyword_score(weak_answer, keywords)
        strong_score = _keyword_score(strong_answer, keywords)

        # LoopJudge verdict
        return self.judge.judge(
            question, weak_answer, strong_answer,
            weak_score, strong_score,
        )

    def generate(self, target_count=100, max_retries=3):
        """Generate up to target_count L2 questions.

        For each clause, up to max_retries rewriting attempts
        if LoopJudge rejects.

        Returns:
            list of accepted question dicts
        """
        accepted = []
        stats = {"accepted": 0, "qv_reject": 0, "judge_reject": 0, "llm_fail": 0}

        i = 0
        last_feedback = ""
        while len(accepted) < target_count and i < len(self.l2_clauses):
            clause = self.l2_clauses[i]
            i += 1

            for attempt in range(max_retries):
                # Step 1: Challenger generates (with scenario material if available)
                feedback = None if attempt == 0 else f"上次尝试被拒绝，原因：{last_feedback}"
                scenario = self._pick_scenario_for_clause(clause)
                question = self.challenger.generate(clause, "L2", narrow_feedback=feedback, scenario_material=scenario)

                if not question:
                    stats["llm_fail"] += 1
                    break

                question["question_id"] = self.id_manager.next("L2")
                question = fix_question_escaping(question)

                # Step 2: Quality Verifier
                qv_pass, qv_issues = self.verifier.verify(question, "L2")
                if not qv_pass:
                    stats["qv_reject"] += 1
                    last_feedback = "; ".join(qv_issues)
                    continue

                # Step 3: Validate structure
                valid, _ = validate_question(question, "L2")
                if not valid:
                    stats["qv_reject"] += 1
                    continue

                # Step 4: LoopJudge — weak/strong solver gap analysis
                if self.fast_mode:
                    # Fast path: accept if QV passes (MVP behavior)
                    accepted.append(question)
                    stats["accepted"] += 1
                else:
                    verdict = self._run_loopjudge(question, clause)
                    if verdict["verdict"] == "accept":
                        accepted.append(question)
                        stats["accepted"] += 1
                    elif verdict["verdict"] == "rewrite":
                        stats["judge_reject"] += 1
                        last_feedback = verdict.get("feedback", "增加推理深度")
                        continue  # Retry with feedback
                    elif verdict["verdict"] == "reject":
                        stats["judge_reject"] += 1
                        break  # Skip this clause entirely

                if len(accepted) % 10 == 0:
                    print(f"    L2: {len(accepted)}/{target_count} accepted "
                          f"(qv_reject={stats['qv_reject']} judge_reject={stats['judge_reject']})")
                break  # Move to next clause

        print(f"  L2 Generation: {stats}")
        return accepted
