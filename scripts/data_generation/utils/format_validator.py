"""Format Validator: Verify generated questions match reference schema."""

import json
import os


def load_reference_schema():
    """Load schema reference from existing eval_set_70_v3.

    Returns set of required fields and their expected types.
    """
    ref_path = r"D:\coding\meta_AutoData\data\questions\eval_set_70_v3.json"
    if not os.path.exists(ref_path):
        return None

    with open(ref_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return None

    # Extract required fields from first L1 entry
    l1_entry = None
    l2_entry = None
    l3_entry = None
    for entry in data:
        qc = entry.get("question_class", "")
        if qc == "L1" and not l1_entry:
            l1_entry = entry
        elif qc == "L2" and not l2_entry:
            l2_entry = entry
        elif qc == "L3" and not l3_entry:
            l3_entry = entry

    return {
        "L1": {k: type(v).__name__ for k, v in l1_entry.items()} if l1_entry else {},
        "L2": {k: type(v).__name__ for k, v in l2_entry.items()} if l2_entry else {},
        "L3": {k: type(v).__name__ for k, v in l3_entry.items()} if l3_entry else {},
    }


# Required fields per level
REQUIRED_FIELDS = {
    "L1": ["question_class", "level", "category", "query",
           "expected_answer", "expected_keywords", "source_standard",
           "grading_method", "knowledge_base"],
    "L2": ["question_class", "level", "query",
           "expected_answer", "expected_keywords", "source_standard",
           "grading_method", "knowledge_base"],
    "L3": ["question_class", "level", "query",
           "expected_answer", "expected_keywords", "source_standard",
           "grading_method", "knowledge_base"],
}


def validate_question(question, level):
    """Validate a single question against the required schema.

    Returns:
        (bool, list[str]): pass/fail and list of missing/invalid fields
    """
    issues = []
    required = REQUIRED_FIELDS.get(level, REQUIRED_FIELDS["L2"])

    for field in required:
        if field not in question:
            issues.append(f"缺少必需字段: {field}")
        elif question[field] is None:
            issues.append(f"字段{field}为空")
        elif isinstance(question[field], str) and question[field] == "" and field not in ("expected_answer", "source_standard"):
            issues.append(f"字段{field}为空字符串")

    # Type checks
    if "expected_keywords" in question:
        kws = question["expected_keywords"]
        if not isinstance(kws, list):
            issues.append(f"expected_keywords应为list，实际为{type(kws).__name__}")
        elif len(kws) < 3:
            issues.append(f"expected_keywords不足3个（仅{len(kws)}个）")

    return len(issues) == 0, issues
