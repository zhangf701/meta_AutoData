# LoopJudge 离线评测报告

**评估时间**: 2026-06-30
**评估题目数**: L2=25, L3=20 (共45题，抽样评测)
**弱模型**: Ollama qwen2.5:3b (temperature=0)
**强模型**: DeepSeek chat (temperature=0)
**评分方式**: 关键词重叠 (0-1)

## 总体分布

| Verdict | L2 (n=25) | L3 (n=20) | 合计 |
|---------|-----------|-----------|------|
| accept  | 12 (48%)  | 11 (55%)  | 23 (51%) |
| rewrite | 13 (52%)  | 8 (40%)   | 21 (47%) |
| narrow  | 0 (0%)    | 1 (5%)    | 1 (2%)  |
| reject  | 0 (0%)    | 0 (0%)    | 0 (0%)  |

## 分数统计

| 指标 | L2 | L3 |
|------|-----|-----|
| avg weak_score | 0.21 | 0.39 |
| avg strong_score | 0.70 | 0.53 |
| avg gap | 0.27 | 0.10 |

## 关键发现

### 1. L2 区分度较好（avg_gap=0.27）
- 弱模型(qwen2.5:3b)在L2推理题上表现明显弱于强模型
- 48% accept说明近半数L2题目能有效区分模型能力
- 52% rewrite说明部分L2题目的关键词过于通用，弱模型也能命中

### 2. L3 gap偏小（avg_gap=0.10）
- L3强模型得分(0.53)反而不如L2(0.70)——L3题目更难，强模型也难以完全命中所有关键词
- L3弱模型得分(0.39)高于L2(0.21)——L3答案更长(1368 vs 182 chars)，关键词被命中的概率更高
- 1题触发narrow(5%)——LoopJudge成功识别了"双方模型都做不出"的过难题目

### 3. 关键词评分局限性
- 关键词重叠评分过于慷慨，不能区分"精确使用术语"和"泛泛提到术语"
- 推荐后续使用完整rubric scorer (clause accuracy + reasoning + honesty)
- 但当前评分已能反映弱/强模型的**方向性差异**

## LoopJudge 闭环结论

LoopJudge 离线评测已验证：
1. L2/L3题目存在可测量的弱/强模型区分度
2. narrow逻辑已成功触发（L3-108：双方模型得分均低）
3. rewrite逻辑可识别"弱模型太强"的题目（关键词过于通用）
4. l2_generator.py 和 l3_generator.py 已启用 LoopJudge，支持 fast_mode 切换

## 配置阈值

| 参数 | 值 | 说明 |
|------|-----|------|
| L2_ACCEPT_WEAK_MAX | 0.4 | 弱模型得分≥0.4则rewrite |
| L2_ACCEPT_STRONG_MIN | 0.7 | 强模型得分<0.7则rewrite |
| L2_ACCEPT_GAP_MIN | 0.2 | gap<0.2则rewrite |
| L3_ACCEPT_WEAK_MIN | 0.1 | 弱模型得分<0.1则可能narrow |
| L3_ACCEPT_STRONG_MIN | 0.5 | 强模型得分≥0.5才accept |
| MAX_L3_NARROW_ITERATIONS | 3 | 最多收窄3轮 |
