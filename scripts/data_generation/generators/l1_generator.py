"""L1 Generator: Template-based parameter and prescriptive requirement questions.

Uses both numerical-parameter clauses and pure-text prescriptive clauses.
QC verification is applied to every generated question.
No LoopJudge needed for L1 (scoring is deterministic keyword match).
"""

from data_generation.agents.challenger import generate_l1_from_clause
from data_generation.agents.quality_verifier import QualityVerifier
from data_generation.utils.escaping_fixer import fix_question_escaping
from data_generation.utils.format_validator import validate_question


class L1Generator:
    """Generate L1 questions from parameter and prescriptive clauses.

    Handles two clause types:
    - Numerical: clauses with extractable parameters (thresholds, percentages, voltages)
    - Prescriptive: pure-text regulatory requirements without numerical values

    Both types pass through QC verification before acceptance.
    """

    def __init__(self, clauses, id_manager, min_numerical_ratio=0.6):
        """Initialize L1Generator.

        Args:
            clauses: list of clause dicts with 'difficulty' == 'L1'
            id_manager: IDManager instance
            min_numerical_ratio: minimum fraction of questions that should be numerical
        """
        import re
        all_l1 = [c for c in clauses if c.get("difficulty") == "L1"]
        if not all_l1:
            raise ValueError("No L1 clauses provided to L1Generator")
        # Sort: numerical clauses first for priority generation
        self.numerical_clauses = [
            c for c in all_l1
            if c.get("clause_type") == "numerical"
            or re.search(r'\d+(?:\.\d+)?\s*[%~kMGVWVAHhzΩ℃倍年月日台回套米秒]',
                         c.get("clause_text", ""))
        ]
        self.prescriptive_clauses = [
            c for c in all_l1 if c not in self.numerical_clauses
        ]
        self.id_manager = id_manager
        self.verifier = QualityVerifier()
        self.min_numerical_ratio = min_numerical_ratio

    def generate(self, target_count=100):
        """Generate L1 questions with QC, prioritizing numerical clauses.

        Strategy:
        1. Generate numerical questions first (target: min_numerical_ratio * target_count)
        2. Fill remaining slots with prescriptive questions
        3. If not enough numerical, supplement from prescriptive pool

        Returns:
            list of accepted question dicts
        """
        import re
        accepted = []
        rejected = {"structure": 0, "qc": 0, "duplicate": 0}
        seen_queries = set()
        min_numerical = int(target_count * self.min_numerical_ratio)

        # ── Phase 1: Generate numerical questions ──
        print(f"    Phase 1: Generating numerical questions (target: {min_numerical})")
        num_accepted = self._generate_from_pool(
            self.numerical_clauses, min_numerical, accepted,
            seen_queries, rejected, "numerical"
        )
        print(f"    Phase 1 done: {num_accepted} numerical accepted")

        # ── Phase 2: Fill remaining with prescriptive ──
        remaining = target_count - len(accepted)
        print(f"    Phase 2: Generating prescriptive questions (target: {remaining})")
        pres_accepted = self._generate_from_pool(
            self.prescriptive_clauses, remaining, accepted,
            seen_queries, rejected, "prescriptive"
        )
        print(f"    Phase 2 done: {pres_accepted} prescriptive accepted")

        # ── Phase 3: If total < target, use any remaining clauses ──
        if len(accepted) < target_count:
            shortfall = target_count - len(accepted)
            print(f"    Phase 3: Filling shortfall ({shortfall}) from all clauses")
            all_clauses = self.numerical_clauses + self.prescriptive_clauses
            self._generate_from_pool(
                all_clauses, shortfall, accepted,
                seen_queries, rejected, "backup"
            )

        # Summary
        numerical_count = sum(
            1 for q in accepted
            if re.search(r'\d+(?:\.\d+)?\s*[%~kMGVWVAHhzΩ℃倍年月日台回套米秒]',
                         q.get("expected_answer", ""))
        )
        num_pct = 100 * numerical_count // max(len(accepted), 1)
        print(f"  L1 Generation: {len(accepted)} accepted "
              f"({numerical_count} numerical/{num_pct}%, "
              f"{len(accepted) - numerical_count} prescriptive)")
        print(f"    Rejected: {rejected}")

        return accepted

    def _generate_from_pool(self, clause_pool, target, accepted, seen_queries,
                            rejected, label):
        """Generate questions from a clause pool, cycling if needed."""
        import re
        count = 0
        if not clause_pool:
            return count

        # Cycle through clauses if needed
        pool = clause_pool[:]
        attempt = 0
        max_attempts = len(pool) * 3  # generous limit

        for clause in self._cycle_pool(pool):
            if count >= target or attempt >= max_attempts:
                break
            attempt += 1

            question = generate_l1_from_clause(clause)
            if not question:
                continue

            question["question_id"] = self.id_manager.next("L1")
            question = fix_question_escaping(question)

            valid, _ = validate_question(question, "L1")
            if not valid:
                rejected["structure"] += 1
                continue

            qc_passed, _ = self.verifier.verify(question, "L1")
            if not qc_passed:
                rejected["qc"] += 1
                continue

            query_key = question["query"][:60]
            if query_key in seen_queries:
                rejected["duplicate"] += 1
                continue
            seen_queries.add(query_key)

            accepted.append(question)
            count += 1

        return count

    @staticmethod
    def _cycle_pool(pool):
        """Yield clauses from pool, cycling indefinitely."""
        if not pool:
            return
        i = 0
        while True:
            yield pool[i % len(pool)]
            i += 1
