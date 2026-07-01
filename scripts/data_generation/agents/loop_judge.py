"""Loop Judge: Analyze weak/strong solver rollouts and decide accept/reject/narrow.

Based on Legal pipeline's Loop Judge from the Autodata paper.
Key insight: For L3 (too hard), send "narrow" feedback.
For L2 (may be too easy), demand deeper reasoning.
"""

import sys
import os

# Add scoring package to path
sys.path.insert(0, r"D:\coding\power_grid_rag\scripts")

from data_generation.config import (
    L1_WEAK_REJECT_THRESHOLD,
    L2_ACCEPT_WEAK_MAX, L2_ACCEPT_STRONG_MIN, L2_ACCEPT_GAP_MIN,
    L3_ACCEPT_WEAK_MIN, L3_ACCEPT_STRONG_MIN,
    MAX_L3_NARROW_ITERATIONS,
)


class LoopJudge:
    """Judge question quality based on weak/strong solver gap analysis."""

    def __init__(self, weak_solver, strong_solver):
        self.weak_solver = weak_solver
        self.strong_solver = strong_solver

    def judge(self, question, weak_answer, strong_answer, weak_score, strong_score,
              narrow_count=0):
        """Analyze solver performance gap and return verdict.

        Args:
            question: dict with query, expected_answer, etc.
            weak_answer: text response from weak solver
            strong_answer: text response from strong solver
            weak_score: rubric_score() result for weak solver (0-1)
            strong_score: rubric_score() result for strong solver (0-1)
            narrow_count: number of prior narrowing iterations (L3 only)

        Returns:
            dict: {"verdict": "accept"/"reject"/"narrow"/"rewrite",
                   "feedback": str,
                   "gap": float}
        """
        level = question.get("level", "L2")
        gap = strong_score - weak_score

        if level == "L1":
            return self._judge_l1(weak_score, strong_score, gap)

        elif level == "L2":
            return self._judge_l2(weak_score, strong_score, gap, weak_answer)

        elif level == "L3":
            return self._judge_l3(weak_score, strong_score, gap, narrow_count,
                                  weak_answer, strong_answer)

        return {"verdict": "accept", "feedback": "", "gap": gap}

    def _judge_l1(self, weak_score, strong_score, gap):
        """L1: Simple rejection if too easy for weak model."""
        if weak_score >= L1_WEAK_REJECT_THRESHOLD:
            return {
                "verdict": "reject",
                "feedback": f"L1题目过于简单：弱模型得分={weak_score:.2f}≥{L1_WEAK_REJECT_THRESHOLD}，弱模型可以轻松答对",
                "gap": gap,
            }
        if gap < 0.2:
            return {
                "verdict": "reject",
                "feedback": f"L1区分度不足：差距={gap:.2f}<0.2",
                "gap": gap,
            }
        return {"verdict": "accept", "feedback": "", "gap": gap}

    def _judge_l2(self, weak_score, strong_score, gap, weak_answer):
        """L2: Accept if weak struggles and strong succeeds with meaningful gap.

        Analogous to CS paper task in Autodata: questions may be too easy.
        """
        # Too easy: weak model already scores well
        if weak_score >= L2_ACCEPT_WEAK_MAX:
            return {
                "verdict": "rewrite",
                "feedback": f"题目推理深度不足：弱模型得分={weak_score:.2f}≥{L2_ACCEPT_WEAK_MAX}。请增加推理复杂度，加入更多条件判断和参数交互。",
                "gap": gap,
            }

        # Ideal: weak struggles, strong succeeds
        if weak_score < L2_ACCEPT_WEAK_MAX and strong_score >= L2_ACCEPT_STRONG_MIN and gap >= L2_ACCEPT_GAP_MIN:
            return {"verdict": "accept", "feedback": "", "gap": gap}

        # Too hard: even strong solver can't answer
        if strong_score < 0.4:
            return {
                "verdict": "rewrite",
                "feedback": f"题目可能有问题或过于模糊：强模型得分也仅={strong_score:.2f}。请明确场景参数，确保问题可基于标准条款回答。",
                "gap": gap,
            }

        # Marginal: gap too small
        if gap < L2_ACCEPT_GAP_MIN:
            return {
                "verdict": "rewrite",
                "feedback": f"题目区分度不足：差距={gap:.2f}<{L2_ACCEPT_GAP_MIN}。弱模型={weak_score:.2f}，强模型={strong_score:.2f}。请增加需要专业推理的判断点。",
                "gap": gap,
            }

        # Default accept
        return {"verdict": "accept", "feedback": "", "gap": gap}

    def _judge_l3(self, weak_score, strong_score, gap, narrow_count,
                   weak_answer, strong_answer):
        """L3: Manage difficulty via narrowing loop.

        Analogous to Legal task in Autodata: questions may be too hard,
        weak model gets all zeros, needs to be narrowed to provide
        useful GRPO gradient signal.
        """
        # Both models score zero: question is OBJECTIVELY too hard or broken
        if weak_score == 0.0 and strong_score < 0.3:
            if narrow_count >= MAX_L3_NARROW_ITERATIONS:
                return {
                    "verdict": "reject",
                    "feedback": f"已达最大收窄次数({MAX_L3_NARROW_ITERATIONS})，双方模型得分均接近零。题目可能超出知识范围。",
                    "gap": gap,
                }
            return {
                "verdict": "narrow",
                "feedback": (
                    f"题目过于困难：弱模型={weak_score:.2f}，强模型={strong_score:.2f}。"
                    f"请收窄范围：(1) 减少涉及的规范条款数量（从3-4条减到2条）"
                    f"(2) 简化场景参数，降低综合复杂度"
                    f"(3) 在query中加入更多引导性提示"
                    f"(4) 确保方案A和方案B的关键差异点清晰可辨"
                ),
                "gap": gap,
            }

        # Weak model zero but strong does well: VALID question but weak can't handle it
        if weak_score == 0.0 and strong_score >= L3_ACCEPT_STRONG_MIN:
            if narrow_count < MAX_L3_NARROW_ITERATIONS:
                return {
                    "verdict": "narrow",
                    "feedback": (
                        f"题目对弱模型过难：弱={weak_score:.2f}，强={strong_score:.2f}。"
                        f"强模型可回答，说明题目本身有效。请适度降低场景复杂度，"
                        f"或增加子问题引导弱模型推理。"
                    ),
                    "gap": gap,
                }
            # After max narrows, accept with caveat
            return {
                "verdict": "accept",
                "feedback": "经多次收窄后接受。弱模型得分仍为零，但强模型表现良好。标注为高难度题。",
                "gap": gap,
                "annotator_note": f"high_difficulty: weak={weak_score:.2f}, strong={strong_score:.2f}, narrows={narrow_count}",
            }

        # Weak has some signal: acceptable
        if weak_score >= L3_ACCEPT_WEAK_MIN and strong_score >= L3_ACCEPT_STRONG_MIN:
            return {"verdict": "accept", "feedback": "", "gap": gap}

        # Weak model has some signal but gap too small
        if gap < 0.2:
            return {
                "verdict": "rewrite",
                "feedback": f"L3区分度不足：差距={gap:.2f}。请增强方案间的技术冲突或增加工程决策的难度。",
                "gap": gap,
            }

        return {"verdict": "accept", "feedback": "", "gap": gap}
