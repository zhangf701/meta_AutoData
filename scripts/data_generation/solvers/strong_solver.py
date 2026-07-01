"""Strong Solver: DeepSeek API for quality benchmarking."""

import os
from openai import OpenAI
from data_generation.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STRONG_MODEL


class StrongSolver:
    """Calls DeepSeek API as the strong reference solver."""

    def __init__(self, api_key=None, base_url=None, model=None):
        self.client = OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL,
        )
        self.model = model or STRONG_MODEL

    def solve(self, query, context_chunks=None, temperature=0.0):
        """Send query to DeepSeek API, return response text.

        Args:
            query: The question text
            context_chunks: Optional list of reference standard texts
            temperature: Generation temperature

        Returns:
            str: Model response text
        """
        prompt = query
        if context_chunks:
            ctx = "\n\n".join(context_chunks[:8])
            prompt = f"参考以下标准条文和技术文档：\n\n{ctx}\n\n请基于上述参考资料，专业地回答以下电力系统设计审查问题：\n\n{query}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[StrongSolver Error: {e}]"

    def solve_with_scoring_rubric(self, query, expected_answer, rubric_judgments, context_chunks=None):
        """Solve and return response formatted for rubric scoring evaluation."""
        answer = self.solve(query, context_chunks)
        return answer
