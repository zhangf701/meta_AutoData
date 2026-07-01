"""L1 Generator: Template-based parameter lookup questions.

Uses regex-extracted clauses with numerical thresholds.
No LoopJudge needed for L1 (scoring is deterministic keyword match).
"""

from data_generation.agents.challenger import generate_l1_from_clause
from data_generation.agents.quality_verifier import QualityVerifier
from data_generation.utils.escaping_fixer import fix_question_escaping
from data_generation.utils.format_validator import validate_question


class L1Generator:
    """Generate L1 questions from parameter-heavy clauses.

    Only uses clauses that actually contain numerical parameters
    (thresholds, percentages, voltages, etc.). Non-numerical clauses
    are skipped (they belong to L2).
    """

    def __init__(self, clauses, id_manager):
        import re
        all_l1 = [c for c in clauses if c.get("difficulty") == "L1"]
        # Filter: only keep clauses with actual numerical parameters
        self.l1_clauses = [
            c for c in all_l1
            if re.search(r'\d+(?:\.\d+)?\s*[%~kMGVWVAHhzΩ℃倍年月日台回套]', c.get("clause_text", ""))
        ]
        if not self.l1_clauses:
            # Fallback: use all clauses tagged L1
            self.l1_clauses = all_l1
        self.id_manager = id_manager
        self.verifier = QualityVerifier()

    def generate(self, target_count=100):
        """Generate up to target_count L1 questions.

        Returns:
            list of accepted question dicts
        """
        accepted = []
        rejected = {"too_simple": 0, "structure": 0}

        for i, clause in enumerate(self.l1_clauses):
            if len(accepted) >= target_count:
                break

            # Generate from clause
            question = generate_l1_from_clause(clause)
            if not question:
                continue

            # Assign ID
            question["question_id"] = self.id_manager.next("L1")

            # Fix escaping
            question = fix_question_escaping(question)

            # Validate structure
            valid, issues = validate_question(question, "L1")
            if not valid:
                rejected["structure"] += 1
                continue

            accepted.append(question)

            if (i + 1) % 20 == 0:
                print(f"    L1: {len(accepted)}/{target_count} accepted "
                      f"(processed {i+1}/{len(self.l1_clauses)} clauses)")

        print(f"  L1 Generation: {len(accepted)} accepted, "
              f"rejected: {rejected}")

        return accepted
