"""Quality Verifier: Three rule-based checks on generated questions.

Based on CS pipeline's Quality Verifier from the Autodata paper.
1. Context Leakage Detection - answer must not be trivially extractable from source
2. Recall vs. Reasoning Classification - verify L2/L3 require genuine reasoning
3. Scenario Inlining + Rubric Formatting - ensure structural requirements met
"""

import re


class QualityVerifier:
    """Rule-based quality checks on generated Q&A pairs."""

    def __init__(self, source_sections=None):
        """source_sections: optional list of source text sections for leakage check."""
        self.source_sections = source_sections or []

    def verify(self, question, level):
        """Run all applicable checks and return (pass, issues) tuple.

        Args:
            question: dict with query, expected_answer, expected_keywords, etc.
            level: "L1", "L2", or "L3"

        Returns:
            (bool, list[str]): pass/fail and list of issue descriptions
        """
        issues = []

        # L1-specific checks
        if level == "L1":
            l1_issues = self._verify_l1(question)
            issues.extend(l1_issues)

        # Check 1: Context Leakage (L2/L3 only)
        if level in ("L2", "L3"):
            leakage = self._check_leakage(question)
            if leakage:
                issues.append(f"答案泄露: {leakage}")

        # Check 2: Recall vs Reasoning (L2/L3 only)
        if level in ("L2", "L3"):
            recall_issue = self._check_recall_vs_reasoning(question, level)
            if recall_issue:
                issues.append(f"推理深度不足: {recall_issue}")

        # Check 3: Structural requirements
        struct_issues = self._check_structure(question, level)
        issues.extend(struct_issues)

        return len(issues) == 0, issues

    def _verify_l1(self, question):
        """L1-specific quality checks.

        Checks:
        1. Answer must not be vague placeholder text
        2. Answer must not be circular (e.g., "不超过规定的事故过负荷能力")
        3. Query must not contain non-electrical keywords
        4. Answer must have meaningful content (numerical OR clear professional judgment)
        """
        issues = []
        answer = question.get("expected_answer", "")
        query = question.get("query", "")

        # Check 1: Answer must not be vague
        vague_pattern = re.compile(
            r'无具体|未明确|未给出|需参考其他|需查阅|标准中未|详见|参见'
        )
        if vague_pattern.search(answer):
            issues.append("答案模糊: 含'无具体数值/未明确给出/需参考其他条款'等占位文本")

        # Check 2: Answer must not be circular
        circular_pattern = re.compile(
            r'不超过规定(的)?(事故过负荷能力|过负荷能力)'
        )
        if circular_pattern.search(answer):
            issues.append("答案循环引用: 以'不超过规定的能力'回答'不超过多少'的问题")

        # Check 3: Query must be electrical engineering only
        non_elec_pattern = re.compile(
            r'防洪|防涝|土石方|飞机场|自然保护区|人文遗址'
            r'|消防|防火|环保|水土保持|绿化|噪声|噪音'
            r'|基础抗拔|抗倾覆|地基|桩基|混凝土|给排水|暖通'
        )
        if non_elec_pattern.search(query):
            issues.append("非电气内容: 题目含土建/防洪/消防/环保等非电气关键词")

        # Check 4: Answer must have meaningful content
        # Numerical check
        has_numerical = bool(re.search(
            r'\d+(?:\.\d+)?\s*[%~kMGVWVAHhzΩ℃倍年月日台回套米秒]', answer
        ))
        # Prescriptive check: answer contains normative judgment keywords
        prescriptive_kw = ['应', '必须', '宜', '可', '不应', '不得', '禁止', '不宜',
                          '直接接地', '全部接地', '经', '采用', '配置', '设置',
                          '接线', '母线', '保护', '接地', '变压器', '断路器']
        has_prescriptive = any(kw in answer for kw in prescriptive_kw)
        # Minimum answer length
        has_min_length = len(answer) >= 4

        if not has_numerical and not (has_prescriptive and has_min_length):
            issues.append(
                f"答案内容不足: 既无数值参数，也缺规定性判断 "
                f"(数值={has_numerical}, 规定性={has_prescriptive}, 长度={len(answer)})"
            )

        return issues

    def _check_leakage(self, question):
        """Verify answer is not trivially extractable from source sections.

        Returns None if OK, or description of leakage.
        """
        answer = question.get("expected_answer", "")
        if len(answer) < 30:
            return "答案过短，可能过于简单"

        if not self.source_sections:
            return None

        # Check each source section for significant overlap with answer
        answer_chars = set(answer)
        for sec in self.source_sections:
            sec_text = sec.get("text", "")
            if len(sec_text) < 100:
                continue
            overlap = len(answer_chars & set(sec_text))
            ratio = overlap / max(len(answer_chars), 1)
            if ratio > 0.6:
                return f"答案与源段落字符重叠率={ratio:.1%}，可能直接从原文复制"

        return None

    def _check_recall_vs_reasoning(self, question, level):
        """Verify question requires reasoning, not just recall.

        Returns None if OK, or description of issue.
        """
        answer = question.get("expected_answer", "")
        query = question.get("query", "")

        # L2: Must have causal/conditional reasoning patterns
        if level == "L2":
            reasoning_indicators = [
                "因此", "导致", "由于", "引发", "面临", "存在.*风险",
                "需.*采取措施", "根据.*要求", "当.*时", "因为",
                "否则", "若不", "一方面", "另一方面", "结论"
            ]
            hits = sum(1 for pat in reasoning_indicators if re.search(pat, answer))
            if hits < 2:
                return f"答案缺少推理连接词（命中{hits}/2+），可能是条款复述而非推理"

        # L3: Must have multi-standard comparison structure
        if level == "L3":
            l3_indicators = [
                "方案A", "方案B", "冲突", "对抗", "比选", "折中",
                "控制优先级", "综合", "诊断", "协调"
            ]
            hits = sum(1 for pat in l3_indicators if re.search(pat, answer))
            if hits < 3:
                return f"L3答案缺少综合比选结构（命中{hits}/3+）"

            # L3 must reference at least 2 different standards
            stds = set(re.findall(r"(?:GB\s*[0-9]+|DL/?T\s*[0-9]+|Q/GDW\s*[0-9]+)", answer))
            if len(stds) < 2:
                return f"L3答案仅引用{len(stds)}个标准，需要≥2个"

        return None

    def _check_structure(self, question, level):
        """Check structural requirements: scenario length, keywords, format."""
        issues = []
        query = question.get("query", "")
        answer = question.get("expected_answer", "")
        keywords = question.get("expected_keywords", [])

        # Scenario context check
        if level == "L2" and len(query) < 150:
            issues.append(f"L2 query仅{len(query)}字，场景描述不足（需≥150字）")

        if level == "L3" and len(query) < 300:
            issues.append(f"L3 query仅{len(query)}字，场景描述不足（需≥300字）")

        # L3 must describe both方案A and方案B
        if level == "L3":
            has_a = "方案A" in query or "方案 A" in query
            has_b = "方案B" in query or "方案 B" in query
            if not (has_a and has_b):
                issues.append("L3 query未完整描述方案A和方案B")

        # Keyword count check
        if len(keywords) < 3:
            issues.append(f"关键词仅{len(keywords)}个，需≥3个")

        # LaTeX escaping pollution check
        pollution_patterns = [
            (r"\\\\<", "转义'\\\\<'应替换为'<'"),
            (r"\\\\-\\\\->", "转义'\\\\-\\\\->'应替换为'→'"),
            (r"\\$S\\\\_n\\$", "LaTeX转义'\\\\_'应替换为'_'"),
        ]
        for pat, desc in pollution_patterns:
            if re.search(pat, answer) or re.search(pat, query):
                issues.append(f"格式污染: {desc}")

        # Answer minimum length
        if level == "L3" and len(answer) < 300:
            issues.append(f"L3答案仅{len(answer)}字，过短（需≥300字）")

        return issues
