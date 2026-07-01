"""Escaping Fixer: Fix LaTeX and JSON escaping pollution in generated text."""

import re


def fix_escaping(text):
    """Fix common LaTeX/JSON escaping issues."""
    if not text or not isinstance(text, str):
        return text

    # Fix < and > escaped as \\< \\>
    text = re.sub(r"\\\\<", "<", text)
    text = re.sub(r"\\\\>", ">", text)

    # Fix arrow \\-\\-> to →
    text = re.sub(r"\\\\-\\\\->", "→", text)

    # Fix LaTeX subscript escaping: $S\\_n$ -> $S_n$
    text = re.sub(r"\$([^$]*?)\\\\_([^$]*?)\$", r"$\1_\2$", text)

    # Fix double backslash in LaTeX commands
    text = re.sub(r"\\\\frac", r"\\frac", text)
    text = re.sub(r"\\\\times", r"\\times", text)
    text = re.sub(r"\\\\sqrt", r"\\sqrt", text)

    # Fix escaped quotes in JSON strings
    text = text.replace('\\"', '"')

    # Fix bare newline escaping
    text = text.replace("\\n", "\n")

    return text


def fix_question_escaping(question):
    """Fix escaping in all text fields of a question dict."""
    for field in ["query", "expected_answer"]:
        if field in question and question[field]:
            question[field] = fix_escaping(question[field])

    if "expected_keywords" in question:
        question["expected_keywords"] = [
            fix_escaping(kw) for kw in question["expected_keywords"]
        ]

    if "source_standard" in question and question["source_standard"]:
        question["source_standard"] = fix_escaping(question["source_standard"])

    return question
