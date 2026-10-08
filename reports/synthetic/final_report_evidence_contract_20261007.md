# 最终汇报的证据合同

这是对实际源码、39 次注册训练、统一汇总器与当前运行状态的独立审计。它规定最终报告怎样从证据得出结论，**不是全部实验已经完成的报告**。机器可读版本为 [final_report_evidence_contract_20261007.json](final_report_evidence_contract_20261007.json)。审计时未读取新增模型的 2005 预测。

## 完成与选型的判据

最终数值必须来自 [53 统一汇总器](../../experiments/synthetic/53_matched_reconstruction_report.py)产生的 `outputs/synthetic_matched_20261007/metrics.json`。仅当 [57 完成守卫](../../experiments/synthetic/57_finalize_matched_campaign.py)核验 39 个不同 run、13 个完整三种子配置、实际每次 15,000 步、温盐各 351,895 个共同评分值，以及既有 learned OI 和固定 MLP 必需行，并生成 `completion.json`，才可称本轮完整实验已交付。报告、指标、冻结选择的 SHA256 必须与完成凭证一致。

最终方案按 **2004 验证集温盐两变量平均标准化 RMSE**选择；逐深度训练标准差用于标准化，注册架构使用三种子预测集成均值。经典方法和既有 learned OI 同样可以获选；固定单种子 MLP 仅用于比较。2005 已在此前工作中查看，本轮是共同 setting 的开发证据，不是新的独立最终测试。不能因为另一个模型在 2005 更好而改写已冻结的验证赢家。

注册学习 family 的获选方案是三模型集成；最终设计应区分每个模型的 backbone 和三模型推理成本。既有 learned OI 是单个固定检查点，不能隐藏集成与单模型的差别。标准化只拟合训练年份，但沿用旧协议使用那些年份的全部剖面，包括 held-out cohort；不得额外声称 held-out float 从未进入训练年预处理。

若既有 learned OI 获选，最终架构应写为保留该既有方案，新增 latent/MoE 为研究候选。若 `latent_off` 获选，不能把关闭的 latent 写成最终保留的 backbone。若 Dense 或 Local Transformer 获选，只能使用属于该 family 的实际配置和成绩，不能把 SoftMoE 的组件消融当作赢家的机制证明。

## 与旧方法的差别怎样解释

旧 64-slot 模型本来已有 Perceiver 风格共享 latent。旧 `ProfileEncoder` 先编码逐层温盐，再在本数据的 **4 个物理深度带**内对 embedding 做 masked mean；不是五个深度带，也不是直接平均原始温盐值。

新增系统保留逐层数值和掩码，增加冻结球面 OI 均值锚点、直接读原始新息的局部数值通路，再比较共享 latent 的 Dense/Soft MoE/Local Transformer 候选。均值 heads 与局部 gate 零初始化，因此初始均值等于 OI。这个实现事实不能推导出严格观测插值、latent 无损压缩、解析后验更新或重复证据的正确信息量计算。

旧→新同时改变观测表示、OI 锚点、局部数值通路、损失、学习率、schedule、target dropout、精度与部分容量，因此其误差差值是**完整系统**的变化。Dense64→Dense192 也改变宽度、latent 数、block 数、head 数和参数容量；Dense192→SoftMoE192 替换 FFN 并改变容量，不是只改变 attention。

新 46 使用 AdamW、峰值 LR 0.0003、变量均衡 MSE + 0.02 Gaussian NLL、bf16 AMP 神经计算；旧 49/52 沿用 AdamW、LR 0.001、pooled MSE、20% target-variable dropout、原 schedule 与 FP32。固定 OI 的几何和线性求解保持 float64。4DVarNet 使用作者 solver/prior/gradient 源码的本任务适配，优化和数据接口另列，不能称为原论文完全同协议复现。

必须报告更强的既有 `67 anc_kriging` learned OI：**120 个参数、1,500 步、固定 seed 1234**，而不是把它写成新的三种子 15k family。其检查点、历史误差和共同评分身份已通过现时 parity 核验。超过旧 64-slot 或固定 44 OI，不足以证明超过这个此前最好方法。其单种子结果不能补造三种子标准差。

## 组件、模态与不确定性结论的条件

| 要支持的结论 | 本批实际对应证据 | 解释边界 |
|---|---|---|
| 新完整系统相对旧版改善 | 同模式 `contrasts[mode:family].against_previous_token64`，配对种子结果另列 | 同口径比较种子均值或集成，承认多项系统改动 |
| 超过此前最好方案 | `selected_against_previous_best[mode].against_prior_learned_oi` | 温盐分别判断，保留单种子/预算差别 |
| 共享 latent 额外有用 | `argo:full_minus_latent_off` 与 `surface:full_minus_latent_off` | **仅 SoftMoE192**；无 Dense/Local 对应消融 |
| 新增局部通路额外有用 | `argo:full_minus_local_off` | **仅 Argo-only SoftMoE192**；固定 OI 并未移除；没有 surface local-off |
| 表层产品增加收益 | `multimodal_contrasts[family]` | 必须同 family 两模式，不能对两个不同赢家作模态归因 |
| 概率预测或校准更好 | NLL、CRPS、68%/95% coverage 与尺度共同比较 | 区间变宽造成覆盖率改善不够；只评边际 Gaussian |

所有差值均为“前者减后者”，负值表示前者 RMSE 较低。点估计小于零可称该变量误差下降；95% 月份 bootstrap 区间上界也小于零，才有对应的跨月份区间支持。区间覆盖零时应写为点估计下降但区间未排除无改善。三种子样本 SD（`ddof=1`）和 12 个评分月份的 bootstrap CI 描述不同变异来源，不能合称模型的全部不确定性。

相对 RMSE 降幅按 `100 × (参照 RMSE − 候选 RMSE) / 参照 RMSE` 计算。候选与参照必须同为种子均值口径或同为集成口径；不要挑选最优单种子与旧三种子均值比较。

表层组共享同一 CESM2 产品和缓存，但旧模型用 patch、新模型用剖面/查询位置特征、4DVarNet 用辅助整场。SST/SSS 是无噪声 5 m 真值，steric sea level 来自同一待重建温盐真值，因此收益不能直接外推到真实独立卫星观测。历史粘贴的多模态/4DVarNet 数字缺少可定位产物，本轮新 recipe 不能冒充它们的精确回放。

## 汇总器实际字段

| 用途 | JSON 字段 |
|---|---|
| 最终选择 | `selection.selected[mode].family / kind / validation_mean_standardized_rmse` |
| 三种子均值与 SD | `row.seed_mean_sd[TEMP\|SALT][metric].mean / sd` |
| 各种子实测 | `row.seed_metrics[i].physical_metrics[TEMP\|SALT][metric]` |
| 集成实测 | `row.ensemble.physical_metrics[TEMP\|SALT][metric]` |
| 固定基线实测 | `row.physical_metrics[TEMP\|SALT][metric]` |
| 逐深度实测 | 固定行的 `by_depth_physical_metrics`，或学习行的 `ensemble/seed_metrics[i].by_depth_physical_metrics` |
| 配对 CI | `contrast[TEMP\|SALT].delta_rmse / ci95_delta_rmse` |
| 组件的种子差值 | `contrast.paired_seed_rmse[TEMP\|SALT].per_seed_delta_rmse / mean_delta_rmse / sd_delta_rmse` |

学习行没有根级 `physical_metrics`。53 为统一表格把固定 MLP 和既有 learned OI 编码为 `kind=classical`，但这两个实际仍是单种子学习方法。冻结选择的 `kind=classical` 或 `existing_learned_oi` 行在 `rows` 中属于 `mode=argo`，即使它被 surface 候选组选中，也应按 Argo 固定行定位。

RMSE/MAE/偏差在物理距平与绝对状态上相同；R² 与相关系数不同，`r2/pearson_r` 为物理距平，`absolute_r2/absolute_pearson_r` 为恢复气候态后的状态。身份对齐将标准化 target 转为 float32；**最终全部物理指标使用 47 保存的原始 float64 真值**。不能把二者混淆。

OI 与既有 learned OI 的不确定性尺度分别由 2004 各深度残差 RMS 拟合，不是解析 OI 后验。三种子标准差由全方差公式合并；NLL/CRPS 用 moment-matched Gaussian，不是完整 Gaussian mixture 密度。没有不确定性输出的方法应留空。

## Contribution 与 novelty 的表达

最终报告应分开列“已实现设计”“本实验支持的实证贡献”“仍待证明的研究假设”。Perceiver IO、Soft MoE、背景场加观测修正和 learned assimilation 都有先例，不能单独作为原创性证明。应对最近文献逐项比较输入、观测算子、latent 更新、数值新息通路、相关证据处理与不确定性；若先例已有相同组合，原创性风险需明确说明。

本轮尚未证明任意深度/日期连续 3-D 推理、严格因果预报、transfer pretraining、跨海盆迁移、重复证据信息守恒、守恒约束、稀疏专家计算效率或独立 SOTA。Soft MoE 的全部 experts 都执行；注册参数数包括消融绕过的模块，不能当作有效 FLOPs。若新增网络没有超过强基线，仍可报告完整复现、对照与机制诊断，但不能写成新 backbone 的精度突破。

## 本次独立检查

审计快照时间与 SHA256 详见配套 JSON。快照时 39 次训练完成 3 次；selection、metrics、completion 尚未产生。主训练、固定 MLP 评价等待、进度和完成守卫四个 native user systemd 服务均为 `active/running`，`ExecMainStatus=0`。主训练内存约 14.8 GB，限制为 MemoryHigh 24G、MemoryMax 32G、MemorySwapMax 0。

39 个 manifest 命令形成 13 个完整三种子 family；已完成 Dense64 三次训练的源码指纹、4DVarNet/旧 surface ready 标记、既有 learned OI 检查点/摘要/源码 SHA256 均与现时文件一致。当前训练日志未检出 Traceback、OOM、CUDA error 或非有限错误。本快照可能随运行变旧，不代替最终完成凭证，也不保证后续所有训练成功。
