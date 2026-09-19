# M10 / M11 step 150：离线机制分析

日期：2026-09-19。状态：**已完成，CPU only；没有启动训练、模型 forward、环境 rollout 或 GPU 任务。**

本分析固定现有第 150 步训练 batch，重放 estimator、比较反事实 readout。它回答同一批轨迹的 credit 如何变化，不是更换方法后的训练成功率预测，也不是用最终 checkpoint 重新采样的 evaluation。

## 数据与可重现性

| 数据 | M10 ALFWorld | M11 WebShop |
|---|---:|---:|
| 来源 | 原始 M10 H1，step 150 | 原始 M11 H1，step 150 |
| task batches / trajectories | 16 / 128 | 16 / 128 |
| 去掉 padding 副本后的 turn rows | 2,610 | 691 |
| 本训练 batch 成功轨迹 | 106 / 128，82.81% | 72 / 128，56.25% |
| terminal rows | 128，4.90% | 128，18.52% |
| observation group 不足、使用 task fallback 的 rows | 144，5.52% | 111，16.06% |

这些成功率是快照内训练 rollout 的成功率，**不是 step 150 validation success**。两份数据都是 H1；本地准备的 H2 ablations 不在此结果中。特征、目标和 reward 标签全部取自原始快照，WebShop item-option scorer 没有在此重新评估。

来源：

- [M10 step 150](../m10-ccpo-attncred-ctxadv-future-progress-alfworld-1.5b-2gpu-20260918/outputs/future_progress/step-0150.npz)
- [M11 step 150](../m11-ccpo-attncred-ctxadv-future-progress-webshop-1.5b-2gpu-20260918/outputs/future_progress/step-0150.npz)
- [完整机器可读结果与来源 SHA-256](summary.json)
- [分析脚本](../../scripts/analyse_step150_posthoc.py)
- [验证记录](VALIDATION.json)

从项目根目录重现：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/analyse_step150_posthoc.py \
  --output experiments/step150-posthoc-analysis-20260919 --bootstrap 512
```

首先用实际 `core_ccpo.ccpo_step_advantage` 重建历史和 potential baseline；两份数据的最大绝对误差都为 **0**。原始 edge 重建误差也为 **0**。NOCTX 使用保存的 processed phi 中的 hidden block，重新 L2 normalize，并与 raw hidden 重新 whitening 的结果独立校验。uniform、κ=4、λ=1、λ=0.5 的反事实公式分别与实际 core 配置对照通过。

## 1. Context 是否改善 return / potential 估计？

对 query 的整条 trajectory 做 leave-out；query 的 label 不进入 peer 均值或 task prior。保持观测 grouping、peer 集合、reward、目标、whitening 规则和 κ=2 不变，分别使用：

- `context_k2`：保存的 hidden + context statistics。
- `hidden_only_k2`：去掉 context statistics block，保留 frozen hidden representation。
- `uniform_k2`：保留整条 trajectory 排除和 credibility shrinkage，将 kernel 改为均匀权重。fallback 也使用其对应均匀 task readout。

主表限制在 **exact observation groups**，避免把 fallback 改变混进 context 比较。评估的是 noisy realized label 的预测误差，不是已知的真实 V 值，更不是 PPO 梯度方差。

对每个 trajectory 先平均平方误差，再对同一 task 的 trajectories 平均，最后对 16 个 tasks 平均。区间是对 task-wise paired differences 做 5,000 次 bootstrap；表示这 16 个 task batches 间的经验差异，不包含训练 seed 或新策略的不确定性。各子集单独做这种宏平均，因此子集均值不一定按 row 数加权后等于总均值。

历史目标为 Y；potential 目标为 `Z_t = gamma^(T-t) R_episode`。两者高度相关，不能把 Y 和 Z 上相似的结果当作两次独立验证。

**Potential Z 的 MSE，越低越好：**

| Readout | M10 ALFWorld | M11 WebShop |
|---|---:|---:|
| Context，κ=2 | 3.9546 | 6.2325 |
| Hidden only，κ=2 | 3.9194 | 6.3581 |
| Uniform，κ=2 | 4.3013 | 6.6232 |
| Context，κ=4 | 4.0074 | 6.6907 |
| Context，λ=1（no shrinkage） | 4.2144 | 5.6065 |
| Context，固定 λ=0.5 | 4.0145 | 6.3740 |

**关键配对差值：variant MSE − context κ=2 MSE；负数表示 variant 更好。**

| 对照 | M10 差值 [95% task-bootstrap CI] | M11 差值 [95% task-bootstrap CI] |
|---|---:|---:|
| Hidden only | −0.0353 [−0.1881, +0.1756] | +0.1256 [−0.0297, +0.3352] |
| Uniform | +0.3466 [−0.0836, +0.7791] | +0.3907 [+0.1256, +0.7341] |
| κ=4 | +0.0528 [−0.0539, +0.1515] | +0.4582 [+0.1754, +0.7772] |
| No shrinkage | +0.2597 [+0.1002, +0.4306] | −0.6260 [−1.1840, −0.1728] |

Interpretation：

- 整体 NOCTX 差异很小：M10 的 hidden-only MSE 低约 0.9%，M11 高约 2.0%；两者区间都跨 0。**这批数据不足以确认额外 context-stat vector 有整体收益或损害。**
- Context 相对 uniform 的收益更集中在重复 observation 子集。M10 exact-revisit 的 Z MSE：context 5.2464、uniform 6.6720；差值区间 [+0.6471, +2.4035]。M11：4.1481 对 4.9106；区间 [+0.2833, +1.3155]。
- M10 revisit 子集中 hidden-only MSE 为 8.5677，但差值区间较宽并跨 0 [−0.3704, +10.3756]，不能只凭该均值宣称统计向量已得到稳健验证。
- 上述区间为探索性多项比较，未做多重检验校正，也不能推广成训练效果的显著性结论。

历史 Y 的对应主表 MSE：M10 context / hidden-only / uniform = 4.3822 / 4.3432 / 4.7664；M11 = 6.9068 / 7.0460 / 7.3394。所有子集、readout 和区间见 [errors.csv](errors.csv)。

## 2. Shrinkage：预测误差和稳定性不一定同向

在 usable exact groups：

```
B = lambda * kernel + (1-lambda) * task_prior
lambda = J/(J+kappa), or a fixed lambda
```

No-shrinkage 保留原来的 fallback 行为；没有 exact peers 时，仍然用 contextual task fallback，不是强行构造不存在的 node baseline。

对每条 query trajectory，从其余 7 条 empirical peer trajectories 中有放回抽取 7 次，共 **512 组**。同一组 draw 同时用于该 query 的当前、未来 endpoint，保留协方差。几何、bandwidth、原始 label 和终局结果固定；bootstrap counts 改变 peer votes、task prior 和 support-based λ。重复抽中的 empirical trajectory 作为重复的 bootstrap draws 计数。此处只测条件敏感性，不是完整 estimator 或 policy gradient 的误差分布。

| 指标 | M10 κ=2 | M10 no shrink | M11 κ=2 | M11 no shrink |
|---|---:|---:|---:|---:|
| Z 预测 MSE，exact groups | 3.9546 | 4.2144 | 6.2325 | 5.6065 |
| 非终止 raw progress 的平均 bootstrap SD | 0.4526 | 0.6782 | 0.2133 | 0.3095 |
| 原始 raw progress 非零时，重采样后的平均同号率 | 77.23% | 77.83% | 91.82% | 93.15% |

去掉 shrinkage：

- M10 的预测 MSE **升高 6.6%**，future 差值的条件 SD **升高约 49.8%**。
- M11 的预测 MSE **降低 10.0%**，但 future 差值的条件 SD **升高约 45.1%**。
- 方差／SD 增大不必然意味着符号稳定性变差，因为差值本身的大小也变了。不同变体的“非零 raw progress”子集数量也略有不同；同号率不是完全相同样本集上的训练优劣指标。
- κ=4 的非终止 progress SD 为 M10 0.3610、M11 0.1833；更平稳没有转化成更低的该 batch 预测误差，尤其 M11 的 MSE 升到 6.6907。

当前 λ 随 J 增长是定义保证的，误差／敏感性却不是单调的。M10 的 J=1/3/7 平均 V bootstrap SD 为 0.8750 / 1.0186 / 0.5626；M11 的 J=1/3/5/7 为 1.1382 / 0.0574 / 1.0574 / 0.2826，其中 J=3 只有 **4 rows**。不同 J 混合了不同任务、回报分布和轨迹阶段，**J 不是经过校准的完整置信度**。

两个 endpoint 的协方差很重要。非终止 rows 上，实际 paired progress 的平均条件方差为 M10 0.3436、M11 0.1457；如果误把 endpoint 视为独立，方差相加会得到 1.3223、1.3705，分别约 **3.85 倍、9.40 倍**。

M11 有 4 个 task 的全部 8 条 trajectory 均失败；这种 empirical peer pool 可产生零方差，却不说明环境中不存在成功 continuation。因此不能把低 bootstrap SD 解释成 future 一定有用。

见 [support.csv](support.csv)、[stability.csv](stability.csv)。

## 3. Future credit 落在哪里？

| 指标 | M10 ALFWorld | M11 WebShop |
|---|---:|---:|
| Terminal row 占比 | 4.90% | 18.52% |
| Terminal 占 raw progress 绝对质量 | 16.20% | 47.59% |
| Terminal 占 task-normalized future 绝对质量 | 15.92% | 49.48% |
| Terminal 占实际 applied future 绝对质量 | **17.93%** | **49.48%** |
| 单个 terminal / nonterminal 的平均 applied future 绝对值之比 | **4.24 倍** | **4.31 倍** |

WebShop 的约一半 future credit 位于终止步骤；但它本身的 terminal row 比例更高，轨迹更短。两边的单步 terminal 放大倍数几乎相同，**不能把 49.48% 对 17.93% 直接解释成 WebShop 的终局偏置更强**。ALFWorld 的两次归一化改变了各 task 的相对质量，所以 raw、normalized、applied 三种尺度分别记录。

每条 trajectory 最后的 action 对应一个 terminal endpoint；其前置 observation 本身不必是终止画面。H1 主实验按距最终 action 的步数分组，详见 [temporal.csv](temporal.csv)。

## 4. History / future：如何改变实际 advantage？

这里直接使用保存的 `history_applied`、`future_applied`、`combined_applied`，其中两分量之和通过与实际 actor advantage 的数值校验。ALFWorld 两分量共享原始 combined normalization 的中心／尺度；WebShop 使用原来的 mean_norm 路径。

| 非终止步骤指标 | M10 | M11 |
|---|---:|---:|
| corr(H_applied, F_applied) | 0.2334 | 0.1878 |
| H、F 都非零的 rows 数 | 2,482 | 424 |
| 这些非零 pairs 中 H/F 异号比例 | **45.77%** | **34.43%** |
| 全部非终止 rows 中，合并后改变 H 符号的比例 | **9.67%** | **10.30%** |

两分量在中间步骤并不高度冗余；它们的 scalar credit 确实经常方向不同。**异号不是梯度冲突，符号翻转也不是自动判定的 credit 错误。** 一个失败 episode 可以包含合理中间动作，一个成功 episode 也可以包含多余动作。

成功 trajectory 的非终止 rows 中，combined credit 为负的比例，从仅保留记录的 H 到 H+F：M10 32.98% → 39.60%；M11 29.45% → 40.80%。这说明 future 改变了相对分配，无法仅依最终 episode 成败判断这些变化是否有害。

**Raw 0 → normalized negative 的具体案例：**

- M10 有 46 个非终止 rows，M11 有 31 个，raw progress 为 0 而 task-standardized future 为负。
- 其中 sole-success trajectory 的中间 rows：M10 **32** 个，M11 **4** 个。排除 query trajectory 后只剩失败 peers，当前和未来非终止 potential 都为 0；task 中正的终局 progress 使 centered future 为负。
- 这 32 / 4 个 rows 的最终 combined credit **全部仍为正**。M10 的该示例中间 F_applied 为 −0.01093，M11 为 −0.03138，均远小于正的 H。

固定原有 applied components，离线查看 `H + alpha F`：

| alpha | M10：全部 rows 中相对 H 的符号翻转 | M11：全部 rows 中相对 H 的符号翻转 |
|---|---:|---:|
| 0 | 0% | 0% |
| 0.25 | 1.57% | 3.33% |
| 0.5 | 4.29% | 6.37% |
| 1 | 9.46% | 8.39% |

这张表是固定原有 normalizer 的 scalar sensitivity；没有模拟换 alpha 后的新轨迹、PPO clipping 或参数更新，也不报告新 alpha 下重新归一化后的梯度幅度。见 [components.csv](components.csv)、[fusion.csv](fusion.csv)。

## 5. 与 M3/M5 原始 edge 的距离来自哪里？

固定 H1、同一批轨迹、相同 success/failure terminal convention 和 task standardization；只比较 future channel。参考 edge 使用 self-inclusive uniform node values，重放与原始记录精确一致。

| Future readout | M10 非终止 corr | M11 非终止 corr | M10 两端 exact corr | M11 两端 exact corr |
|---|---:|---:|---:|---:|
| 原始 self-inclusive uniform edge | 1.000 | 1.000 | 1.000 | 1.000 |
| Uniform，whole-trajectory LOO，无 shrinkage | 0.778 | 0.458 | **0.922** | **0.977** |
| Context kernel，LOO，无 shrinkage | 0.486 | 0.110 | 0.564 | 0.363 |
| 当前 Context kernel，LOO，κ=2 | **0.383** | **0.179** | **0.442** | **0.342** |
| Hidden-only kernel，LOO，κ=2 | 0.297 | 0.156 | 0.333 | 0.371 |

数值为 task-standardized future 与原始 edge 的 Pearson correlation。原始值相关和符号一致率同时保存在 [edge.csv](edge.csv)。

“两端 exact”只保留当前和未来 observation 都有跨 trajectory peers 的非终止 rows：M10 2,240 / 2,482；M11 377 / 563。即至少一个 endpoint 需要 task fallback 的非终止 rows 占 M10 **9.75%**、M11 **33.04%**。

在有充分分组支持的 subset 上，uniform LOO 与原始 edge 仍非常接近；引入 contextual weighting 后相似性下降更大。对整批 WebShop 的比较，还不能忽略 task fallback：原始 self-inclusive node 即使只出现于 query trajectory 仍有值，LOO 后可能没有 exact peer，必须改用 task readout。uniform ladder 在这部分也使用 uniform task fallback，因此全体行的第一步同时涉及 exclusion 和 fallback。

同一 task 内的排序也会改变。在“两端 exact”的非终止 rows 上：

| Readout | M10 平均 Spearman / 高绝对 credit 集合 Jaccard | M11 平均 Spearman / 高绝对 credit 集合 Jaccard |
|---|---:|---:|
| Uniform LOO，无 shrinkage | 0.936 / 0.800 | 0.975 / 0.971 |
| Context LOO，无 shrinkage | 0.562 / 0.317 | 0.346 / 0.357 |
| 当前 Context LOO，κ=2 | 0.502 / 0.212 | 0.397 / 0.393 |

各 task 等权平均；高绝对 credit 集合使用该 task/subset 的绝对值 80th-percentile 阈值，边界 ties 全保留，集合可能超过 20%。忽略信用无变化、无法定义相关或排序的 task：M10 有 16 个有效 task，M11 有 8 个。Jaccard 是交集 / 并集，不是“保留了百分之多少原始 top rows”。

Task standardization 仍然在各完整 task 上计算后再筛子集，没有为了提高 subset correlation 重新归一化。这条分解路径说明数值差异来源，**不意味着更接近原始 edge 就一定更好**；不同设计之间还可能有交互。

## 6. 行为层面的现有证据与缺口

本快照有 observation、trajectory ID、step、episode outcome 和 action-valid flag，可以分析长度、重复 observation 及信用轨迹。没有完整 actor token/logprob/action trace，因此没有伪造 search / option / purchase 的动作分类或声称确定 WebShop 选错 option 的原因。

| Trajectory outcome | 数量 | 平均长度 | 平均重复 observation 比例 |
|---|---:|---:|---:|
| M10 success | 106 | 14.25 | 16.86% |
| M10 failure | 22 | **50.00** | **63.64%** |
| M11 success | 72 | 5.53 | 44.34% |
| M11 failure | 56 | 5.23 | 41.80% |

ALFWorld 的失败轨迹都长达 50 步，并频繁回到相同 observation，值得后续针对循环／低进展行为做动作 trace 分析。WebShop 在这两项代理指标上没有类似的成功／失败区分。重复 observation 不自动等于环境状态完全相同或动作错误。

图中每个环境各选一个成功和失败示例：在各 outcome 内选 H/F 异号 turn 比例最高的 trajectory，长度作为 tie-break；**这些是刻意选取的诊断例子，不是随机代表性样本**。选择规则和完整行数据在 [cases.csv](cases.csv)，全部 trajectory 的统计在 [behavior.csv](behavior.csv)。

## 图表

- [机制总览 PNG](mechanisms.png) / [PDF](mechanisms.pdf)：credit 时间位置、readout 误差、support 与敏感性。
- [History/future 与 edge PNG](credit_and_edge.png) / [PDF](credit_and_edge.pdf)。
- [轨迹示例 PNG](trajectory_cases.png) / [PDF](trajectory_cases.pdf)。

## 对后续实验的含义

1. **保留 no-shrinkage ablation 的优先级，但不要预设两个环境同向。** 当前 ALFWorld 的 κ=2 在误差和 progress 稳定性上都有理由保留；WebShop 的 no-shrinkage 降低预测误差，同时放大 future 差值波动，需要训练实验决定净效应。
2. **NOCTX 有必要做，但该 batch 的总体差异还不足以宣称 context-stat vector 有用或无用。** context 相对 uniform 在 revisit 子集的差异更明显，值得在更多任务 batch 重复检查。
3. **Future 与 edge 的不同不是简单缩放。** 支持充分时 contextual weighting 改变了分配；WebShop 较多的 endpoint fallback 又增加了一层差异。history-only、uniform-future-LOO 和固定 fusion weight 是有信息量的对照。
4. **单个 step 的 snapshot 无法判断训练早期／晚期演变、真正的梯度冲突或最终训练收益。** 若继续采证，可以固定一个 checkpoint 多批 rollout；无需先重训，但需要重新采样并记录动作和必要的模型输入。

此次精确回放验证的是保存数组与 estimator 的一致性，不重新验证历史 GPU forward 的 token-to-feature 对应关系，也不代替远端 8-GPU 故障的实机诊断。所有结论限定在这两份本地 H1 快照。
