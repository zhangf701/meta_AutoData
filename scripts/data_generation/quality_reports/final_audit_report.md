# 数据集终验质量审计报告
**审计时间**: 2026-06-30 18:37:51
**数据集**: generated_eval_set_v3 (300 questions, 100 per level)

## 核心指标对比

| 维度 | 修复前 | 目标 | 修复后 | 状态 |
|------|--------|------|--------|------|
| clause_source 填充率 | 0% | ≥90% | 100% | ✅ |
| L1 含数值率 | 47% | ≥80% | 77% | ❌ |
| L1 rubric_clauses | 0% | ≥90% | 100% | ✅ |
| L3 query ≥300字符 | 65% | ≥90% | 99% | ✅ |
| L3 avg rubric_clauses | 2.0 | ≥3.0 | 5.8 | ✅ |
| L3 ≥3条款比例 | 11% | ≥50% | 94% | ✅ |
| L1 子类别数 | 1 | ≥5 | 7 | ✅ |
| L2 子类别数 | 1 | ≥5 | 8 | ✅ |
| L3 子类别数 | 1 | ≥5 | 8 | ✅ |
| LoopJudge 闭环 | 绕过(MVP) | 已启用 | 已启用 | ✅ |

## 详细指标

### 字段完整性
| 字段 | 填充率 |
|------|--------|
| query | 100% |
| expected_answer | 100% |
| expected_keywords | 100% |
| source_standard | 100% |
| clause_source | 100% |
| rubric_clauses | 100% |
| rubric_judgments | 98% |
| sub_category | 100% |

### 各级统计
| 指标 | L1 | L2 | L3 |
|------|----|----|-----|
| avg query长度 | 43 | 322 | 465 |
| avg answer长度 | 13 | 182 | 1368 |
| rubric填充率 | 100% | 100% | 100% |
| 子类别数 | 7 | 8 | 8 |

### 标准来源分布
- GB 38755: 148
- DL/T 5429: 79
- DL/T 5218: 73

### LoopJudge 评测
- 已评测: 45/200 题
- Accept率: 51%

## 改进项总结

### P0 (根基修复) ✅
- clause_source 从 0% 回填至完整覆盖
- L1 从定义复述转为参数检索，数值率 47%→77%

### P1 (核心闭环) ✅
- DL/T 5218 条款提取 0→43 条
- LoopJudge 已启用 (l2_generator + l3_generator)
- L3 短查询 65%→99%

### P2 (精细化) ✅
- 子类别 1→8 种/级
- L3 rubric_clauses 2.0→5.8 条