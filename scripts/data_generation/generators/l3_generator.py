"""L3 Generator: Cross-standard synthesis questions with narrowing loop.

Based on Autodata Legal pipeline pattern: questions may be too hard,
LoopJudge sends "narrow" feedback to reduce complexity.

Key fix from data quality audit: query MUST inline full scenario description
(≥300 chars, including complete方案A and方案B descriptions).
"""

import random
from data_generation.agents.challenger import Challenger
from data_generation.agents.quality_verifier import QualityVerifier
from data_generation.agents.loop_judge import LoopJudge
from data_generation.utils.escaping_fixer import fix_question_escaping
from data_generation.utils.format_validator import validate_question
from data_generation.config import MAX_L3_NARROW_ITERATIONS


def _keyword_score(model_answer, expected_keywords):
    """Fast keyword-based scoring (0-1) for LoopJudge decisions."""
    if not expected_keywords:
        return 0.5
    ans_lower = model_answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans_lower)
    return hits / len(expected_keywords)


class L3Generator:
    """Generate L3 questions using agentic self-instruct with narrowing loop.

    LoopJudge weak/strong solver gap analysis is ENABLED by default.
    Set fast_mode=True to skip solver eval (accepts if QV passes).
    """

    def __init__(self, clauses, id_manager, weak_solver, strong_solver,
                 scenario_materials=None, fast_mode=False):
        # Combine L3 clauses with cross-standard L2 clause pairs
        self.l3_clauses = [c for c in clauses if c.get("difficulty") == "L3"]
        self.l2_clauses = [c for c in clauses if c.get("difficulty") == "L2"]
        self.id_manager = id_manager
        self.challenger = Challenger()
        self.verifier = QualityVerifier()
        self.judge = LoopJudge(weak_solver, strong_solver)
        self.weak_solver = weak_solver
        self.strong_solver = strong_solver
        self.scenario_materials = scenario_materials or []
        self.fast_mode = fast_mode

    def _pick_conflict_scenario(self):
        """Pick a scenario material with conflicts for L3 generation."""
        if not self.scenario_materials:
            return None
        # Prefer materials with explicit conflicts
        conflict_mats = [m for m in self.scenario_materials
                        if m.get("has_conflicts") or m.get("type") == "review_feedback"]
        import random
        pool = conflict_mats if conflict_mats else self.scenario_materials
        return random.choice(pool) if pool else None

    def _build_clause_pairs(self, count):
        """Build cross-standard clause pairs for L3 generation.

        Pairs clauses from different standards to force synthesis.
        """
        # Group by standard
        by_standard = {}
        for c in self.l2_clauses:
            std = c.get("standard", "unknown")
            by_standard.setdefault(std, []).append(c)

        stn_names = list(by_standard.keys())
        if len(stn_names) < 2:
            # Only one standard available, use all clauses
            pairs = []
            all_clauses = self.l3_clauses + self.l2_clauses
            for i in range(0, min(len(all_clauses), count * 2), 2):
                if i + 1 < len(all_clauses):
                    pairs.append([all_clauses[i], all_clauses[i + 1]])
            return pairs[:count]

        # Pair clauses from different standards
        pairs = []
        attempts = 0
        while len(pairs) < count and attempts < count * 3:
            attempts += 1
            s1, s2 = random.sample(stn_names, 2)
            if by_standard[s1] and by_standard[s2]:
                c1 = random.choice(by_standard[s1])
                c2 = random.choice(by_standard[s2])
                # Avoid duplicate pairs
                pair_ids = sorted([c1.get("clause_id", ""), c2.get("clause_id", "")])
                if pair_ids not in [sorted([p[0].get("clause_id", ""), p[1].get("clause_id", "")]) for p in pairs]:
                    pairs.append([c1, c2])

        # Add single L3 clauses as seed
        for l3c in self.l3_clauses[:count]:
            companion = random.choice(self.l2_clauses) if self.l2_clauses else None
            if companion:
                pairs.append([l3c, companion])

        return pairs[:count]

    def _run_loopjudge(self, question, pair, narrow_count):
        """Run weak/strong solver evaluation for L3 narrowing loop.

        Uses keyword-based scoring for speed. Returns verdict dict
        with accept/narrow/reject decision and feedback.
        """
        clause_text = " ".join(
            c.get("clause_text", "")[:300] for c in pair[:2]
        )
        context = [clause_text] if clause_text else None
        query = question.get("query", "")

        try:
            weak_answer = self.weak_solver.solve(query, context_chunks=context)
        except Exception:
            weak_answer = ""
        try:
            strong_answer = self.strong_solver.solve(query, context_chunks=context)
        except Exception:
            strong_answer = ""

        keywords = question.get("expected_keywords", [])
        weak_score = _keyword_score(weak_answer, keywords)
        strong_score = _keyword_score(strong_answer, keywords)

        return self.judge.judge(
            question, weak_answer, strong_answer,
            weak_score, strong_score,
            narrow_count=narrow_count,
        )

    def generate(self, target_count=100):
        """Generate up to target_count L3 questions with narrowing loop."""
        accepted = []
        stats = {"accepted": 0, "qv_reject": 0, "narrowed": 0, "rejected": 0}

        clause_pairs = self._build_clause_pairs(target_count * 2)  # Over-generate

        for pair in clause_pairs:
            if len(accepted) >= target_count:
                break

            feedback = None
            for narrow_iter in range(MAX_L3_NARROW_ITERATIONS + 1):
                # Step 1: Challenger generates
                scenario = self._pick_conflict_scenario()
                question = self.challenger.generate(pair, "L3", narrow_feedback=feedback, scenario_material=scenario)

                if not question:
                    stats["rejected"] += 1
                    break

                question["question_id"] = self.id_manager.next("L3")
                question = fix_question_escaping(question)

                # Step 2: Quality Verifier (critical: check scenario inlining)
                qv_pass, qv_issues = self.verifier.verify(question, "L3")
                if not qv_pass:
                    stats["qv_reject"] += 1
                    feedback = "; ".join(qv_issues)
                    # Don't waste narrowing iterations on structural issues
                    if narrow_iter == 0:
                        continue
                    else:
                        break

                # Step 3: Validate structure
                valid, issues = validate_question(question, "L3")
                if not valid:
                    stats["qv_reject"] += 1
                    break

                # Step 4: LoopJudge with solver gap analysis + narrowing
                if self.fast_mode:
                    accepted.append(question)
                    stats["accepted"] += 1
                else:
                    verdict = self._run_loopjudge(question, pair, narrow_iter)
                    if verdict["verdict"] == "accept":
                        accepted.append(question)
                        stats["accepted"] += 1
                        if narrow_iter > 0:
                            stats["narrowed"] += 1
                    elif verdict["verdict"] == "narrow":
                        feedback = verdict.get("feedback", "收窄题目范围")
                        continue  # Continue narrowing loop
                    elif verdict["verdict"] == "rewrite":
                        feedback = verdict.get("feedback", "重写题目")
                        continue  # Retry with feedback
                    elif verdict["verdict"] == "reject":
                        stats["rejected"] += 1
                        break

                if len(accepted) % 5 == 0:
                    print(f"    L3: {len(accepted)}/{target_count} accepted "
                          f"(narrowed={stats['narrowed']} rejected={stats['rejected']})")
                break

        print(f"  L3 Generation: {stats}")
        return accepted
