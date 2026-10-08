# 海洋多模态重建：文献相似性与贡献核查

核查时间：2026-10-07T11:43:19Z。文献截止：2026-10-07。依据当前[架构与实验协议](matched_architecture_protocol_20261007.md)及[实际模型源码](../../src/ocean_tokenizer/latent_ocean.py)，只核查已实现机制。结构化证据见[JSON](literature_novelty_audit_20261007.json)。

**当前不能把“卫星 + Argo、共享 latent、Transformer/MoE 或学习不确定性”本身写成核心原创贡献。** 可检验的候选贡献是：用冻结数值 OI 保留一个强初始解，再让原 query 深度的温盐新息直接参与局部修正，同时测试压缩的共享上下文是否带来额外精度或校准收益。方法已经实现，贡献强度仍取决于完整同设置对照；最终 backbone 不能在结果之前确定。

此处的“风险”是**研究主张与既有方法重叠造成的评审新颖性风险**。它不是抄袭、专利或法律判断。下面对我们机制的相似性评估属于基于论文和源码的分析，而不是论文作者对本项目的判断。

## 1. 最需要具体讨论的三篇工作

**CLOINet（2024）最接近“数值 OI 与学习表示协作”这条路线。** 它在 latent clusters 中改变相关性/协方差；我们本轮冻结球面 OI，另外学习 residual 和原数值 local candidate。这一区别可以准确描述，但只有 full/local_off/latent_off 和强 learned OI 的共同结果才能说明它是否有价值。[CLOINet 原文](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1151868/full)

**ADAF-Ocean（2025 preprint）最接近“直接编码异构观测，在共享表示中融合并按坐标解码”。** 必须纠正范围：该版本验证的是表面变量，明确把垂直剖面同化列为后续工作。我们的三维逐层 source Argo 与数值旁路是区别；“海洋多模态 latent + query”已有先例。[ADAF-Ocean 原文，4.2 与 Discussion](https://arxiv.org/html/2511.06041v1)

**TS-Cast（2026-07-16）最直接限制“温盐 latent + 不确定性”这一主张。** 它已有卫星共享 latent、FiLM 条件垂直 U-Net 与温盐误差尺度输出。我们的区别是把当前 source Argo 作为条件、保留 OI/数值新息路径，并以独立 query 请求输出；需要以数值收益支持这一区别。[TS-Cast 原文，2.3](https://os.copernicus.org/articles/22/2161/2026/os-22-2161-2026.html)

更早的基础先例是 **ARMOR3D 的卫星背景 + 剖面 OI 校正**，因此该总体流程也不是首次提出。[Guinehut et al., 2012](https://os.copernicus.org/articles/8/845/2012/)

## 2. 保留的十二项 primary-source 对照

表中的风险等级针对“把重叠部分称为原创”的主张，不表示这些论文采用同一实现或相同数据。

| 工作与来源 | 已有机制 | 我们的重要区别 | 主张重叠风险 |
|---|---|---|---|
| [High resolution 3-D temperature and salinity fields derived from in situ and satellite observations](https://os.copernicus.org/articles/8/845/2012/) (2012) | 先用测高/SST回归生成三维温盐背景，再用原位剖面进行OI校正。 | 本轮synthetic锚点是Argo-only冻结OI；新增共享Transformer和逐深度数值局部修正，并非首次采用背景加剖面校正。 | 高 |
| [Advancing Ocean State Estimation with efficient and scalable AI](https://arxiv.org/abs/2511.06041) (2025) | NP启发的分模态MLP编码与坐标解码，直接融合非规则观测和背景。 | 该版本只验证表面变量，垂直剖面同化列为后续工作；我们读取20层Argo并另保留冻结OI及原数值新息通路。 | 高 |
| [CLOINet: ocean state reconstructions through remote-sensing, in-situ sparse observations and deep learning](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1151868/full) (2024) | OI先验与卫星/原位信息经模糊聚类融合，在latent距离中构造协方差。 | CLOINet学习cluster covariance；我们冻结球面OI，独立query读取共享latent，再直接混合query深度数值新息。 | 高 |
| [TS-Cast: deep learning for subsurface ocean reconstruction from satellite observations in the northwestern Pacific](https://os.copernicus.org/articles/22/2161/2026/os-22-2161-2026.html) (2026) | SST/SSS/ADT编码为共享latent，通过FiLM调制垂直U-Net，输出温盐及深度误差尺度。 | 该模型以卫星和气候态预测整条剖面；我们还用当前source Argo作条件，并有数值OI/新息通路和独立query。 | 高 |
| [Inversion of Sea Surface Currents From Satellite-Derived SST-SSH Synergies With 4DVarNets](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS003609) (2024) | 可训练观测/先验项与展开梯度求解器融合SST和SSH，重建表面流。 | 本轮48使用作者模块适配20层T/S，并不是原论文表面流benchmark；主新模型则一次latent编码后解码。 | 中 |
| [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795) (2022) | 输入cross-attention进入固定latent，latent处理后用输出query解码不同结构。 | 海洋坐标及OI/原数值局部路径是任务设计；旧64-slot已经有Perceiver风格latent。 | 高 |
| [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951) (2024) | 用token到expert slots的软dispatch与反向combine实现可微专家层。 | 本轮是海洋任务应用，4个experts均执行；未提出新router，也未完成参数/FLOPs匹配对照。 | 高 |
| [Attentive Neural Processes](https://arxiv.org/abs/1901.05761) (2019) | 用context/query attention缓解平均聚合的欠拟合，并预测条件分布。 | 我们的额外local候选直接组合原温盐数值；latent是确定性slots，不是ANP的随机全局latent posterior。 | 中 |
| [GP-ConvCNP: Better generalization for conditional convolutional Neural Processes on time series data](https://proceedings.mlr.press/v161/petersen21a.html) (2021) | 以GP posterior替代ConvCNP确定性set embedding，再经CNN条件预测。 | 我们固定OI均值加residual/local correction；该文把GP后验作为函数表示并可采样，不是同一更新公式。 | 中 |
| [Convformer: A Model for Reconstructing Ocean Subsurface Temperature and Salinity Fields Based on Multi-Source Remote Sensing Observations](https://www.mdpi.com/2072-4292/16/13/2422) (2024) | ConvLSTM与时空/global/local attention从SST/SSS/SSH/风重建温盐。 | 该文输入为卫星栅格序列；我们融合当前稀疏剖面并按坐标查询，不只进行表层到深层回归。 | 中 |
| [Cross-scale 3-D thermohaline modeling via dual-residual swin transformer with multisource ocean observations](https://www.tandfonline.com/doi/abs/10.1080/17538947.2025.2607902) (2026) | Swin Transformer、U-Net和dual residual共同建模全球/海盆三维温盐。 | 该文是栅格多尺度one-shot重建；我们的原位逐层条件及OI数值通路是主要设计差别。 | 中 |
| [ReconMOST: Multi-Layer Sea Temperature Reconstruction with Observations-Guided Diffusion](https://arxiv.org/abs/2506.10391) (2025) | CMIP6预训练无条件diffusion，生成阶段用稀疏观测引导多层全球温度。 | 该版本只重建温度且逐步扩散生成；我们联合T/S，当前没有diffusion或transfer pretraining。 | 中 |

Perceiver IO 的 year 按 ICLR 2022；最早 preprint 为 2021。Soft MoE 按 ICLR 2024；最早 preprint 为 2023。ADAF-Ocean 和 ReconMOST 在这里明确标为作者 preprint，不混写为已审稿论文。SwinOcean3D 的 DOI 含 2025，期刊为 2026 Volume 19，论文记载 2025-12-15 接收。

访问限制：Convformer 和 SwinOcean3D 的出版社直接打开受限，本轮利用 web search 返回的**出版社可索引摘要及方法/讨论段**核查，未声称成功打开所有正文页面。其他核心三篇及 ARMOR3D 已读 primary full text。检索还筛查了 3DV-Unet、GraphDOP、OSnet 等，十二项表不是该方向的全部文献，也不构成不存在其他类似方法的证明。

## 3. 当前哪些贡献可以写，哪些仍需证明

| 研究主张 | 当前状态 | 判断理由 | 所需证据 |
|---|---|---|---|
| 首次融合卫星与Argo重建三维温盐 | known_prior_art | 总体任务和背景后校正流程已有直接工作。 | 写成任务背景，精确陈述新的机制与同设置结果。 |
| 观测编码器→共享latent→坐标query decoder是新backbone | known_prior_art | 结构已有通用来源和海洋应用；旧64-slot也已有latent。 | 引用来源，用同系统latent_off对照证明当前任务价值。 |
| Soft MoE或Transformer本身是核心方法创新 | known_prior_art | 当前expert routing沿用已有设计；时空温盐Transformer已有多篇。 | 三seed收益、成本和容量控制；若无稳定收益不列核心贡献。 |
| OI锚点+逐深度原数值新息旁路+共享上下文的具体设计 | implemented_candidate_contribution_not_yet_demonstrated | 公式与旧band embedding refiner有明确差别，但混合核/学习方法已存在，不能只凭组合声称首创。 | full vs local_off、full vs latent_off、fixed OI和既有learned OI的同设置结果；误差及校准改善和失败区域。 |
| 共享latent提供独立信息与不可替代收益 | pending_matched_ablation | old-vs-new同时改观测粒度、OI、局部数值和loss；不能单独归因于latent。 | 固定OI/输入/训练配置下full相对latent_off三seed和月block统计。 |
| 保留原数值旁路意味着latent compression无信息损失或严格插值 | not_established | OI与local都读取有限邻居；learned attention和signed gate没有无损或测点严格重现保证。 | 专门扰动/测点回放及可证明性质；当前只可描述为逐深度数值可直接参与校正。 |
| 新模型DFS估计独立信息量、去重并保留证据质量 | not_implemented_in_current_model | 新46仅有模态总prior平衡；不是DFS或相关观测information accounting，复制/partition不变性未保证。 | 若研究此方向，需新的模型规则与复制观测、独立噪声及相关产品专项实验。 |
| 精确Kalman后验更新和完整不确定性 | not_implemented | 当前latent是learned cross-attention，scale是独立边际Gaussian head，没有分析协方差更新。 | 数学更新规则、相关误差模型及完整posterior检验；现稿使用learned conditional reconstruction。 |
| 学习温盐不确定性head本身有新颖性 | known_prior_art | 相关海洋重建已有latent及深度方差输出。 | 报告CRPS/NLL/coverage/区间宽度相对validation-calibrated OI的实际增益。 |
| 连续任意深度、未来预测、迁移或物理一致性已验证 | outside_current_validated_scope | 本轮只评分20层、月尺度回顾重建；没有独立cutoff forecast、transfer、守恒或稳定性约束实验。 | 新增深度留出/跨年份海盆/严格forecast cutoff/物理诊断，分别验证后才提出相应主张。 |
| 本轮达到全领域SOTA或已超过所有类似论文 | not_established_by_registered_comparisons | 39任务有官方4DVarNet任务适配及强OI，但没有对CLOINet/TS-Cast等逐项同setting重跑；2005为已使用development。 | 限定为本协议注册方法中的最好结果；更广的SOTA需要共同benchmark、相近方法及独立未使用测试。 |
| 共同数据/评分/三seed组件消融与可重算预测产物是科学贡献 | implemented_evaluation_contribution_results_pending | 可支持可审计的比较和机制判断，但不自动成为首个benchmark或方法突破。 | 全部完成、协议校验、验证先选型、公开物理指标与完整失败/负结果。 |

这些状态是本次审查时点的结论。训练结果完成后，只能用已完成的实证更新相应“pending”项；已有组件来源不会因为结果较好而变为本项目原创。

## 4. 与之前64-slot方法的真实差异

旧模型**已经有 Perceiver 风格共享 latent**。旧版本先编码每层，再对 embedding 做 masked depth-band pooling；本轮20层最深984.7 m，对应4个 bands。不能把旧模型描述成“原始温盐直接平均”，也不能把这次 old-vs-new 写成“第一次加入 latent”。

本轮的主要系统改动是：独立保留各实际深度的观测行；给均值加入冻结球面 OI；local candidate 直接组合当前 query 深度原温盐数值；使用零初始化 residual/gate 使**初始均值**等于 OI；再比较更大 Dense、Soft MoE 与局部 Transformer，并学习边际 Gaussian scale。新 loss、目标 dropout、optimizer trajectory 与容量也有变化，详见共同协议。因此 old-vs-new 说明完整系统变化；共享 latent 与 local path 的独立贡献要分别看 full/latent_off、full/local_off。

“直接读原数值”是可核对的实现差别，**不是无损压缩、重复证据守恒或测点严格插值的数学保证**。OI和local仍选择有限邻居；gate可带符号；新模型没有解析 Kalman latent posterior，没有新的 DFS 信息量规则，也没有 token partition/copy invariance 保证。

本轮表层组是从同一 CESM2 truth 生成无噪声 SST/SSS 和 steric SSH proxy。它既不是独立真实测高观测，也没有模拟产品共享原位资料造成的误差相关性。这个理想化 OSSE 中的提升不能直接证明真实观测的额外独立信息，更不能证明预测未来的能力。

## 5. 结果不同，最终贡献与backbone就应不同

若 full 相对 latent_off、local_off 及既有 learned OI 都有稳定额外收益，可以将核心贡献写为**“数值锚定的逐层条件重建：共享上下文与原数值新息协作”**。正文给出具体公式、使用范围、三seed物理指标与校准结果；同时承认通用 backbone 和 FFN 的来源。只有共享 latent 的对照成立，它才应成为性能解释的核心。

若 latent_off 获选或二者差异不稳健，应采用验证选中的局部条件修正设计，清楚报告共享 latent 没有独立收益。Soft MoE 相对 Dense 的比较还要说明容量差异；同宽度和steps不等于参数/FLOPs匹配，expert 使用率也不等于学到了特定海洋物理机制。

若既有 learned OI 获选，应按 OI 作为当前 setting 的最终设计汇报。科学贡献可以是严谨的共同协议、对旧压缩模型的诊断和负结果，不能为了保留新架构而改选型规则，或者把对旧slot模型的大幅提升归因于Transformer。

## 6. 最终报告的数值和措辞边界

最终主表以温度 **RMSE（°C）**、盐度 **RMSE（PSU）** 为主，补 MAE/bias/R²/相关、逐深度/区域结果；能提供尺度的方法才报告 CRPS/NLL/coverage/区间宽度。各seed均值±样本SD与ensemble分开列。用验证集冻结最终选择，2005明确称development，不能改写为全新独立测试。

最关键的对比应同时回答：相对旧64-slot改进多少；相对fixed OI和120参数既有learned OI改进多少；full/latent_off及full/local_off哪条机制提供收益；satellite arm的收益出现在哪些深度；增加容量和MoE的代价是否值得。不因初始解等于OI就宣称训练后所有点评分不会退化。

39个注册训练任务加辅助强基线能支持**“本协议下注册方法中的最好结果”**。它没有对表中的全部相关论文做同setting适配，特别是最接近的 CLOINet/TS-Cast；官方4DVarNet也是作者代码的本任务适配。因此它不能单独建立全领域SOTA。类似论文的原文RMSE来自不同区域、尺度、深度、真值及输入，不能直接拼到本项目表中排高低。

最终完整报告应把“已实现的机制”“实验支持的贡献”“已有组件”“仍未验证的假设”分别写清。贡献标题、最终backbone与性能提升比例以正式选型和统一预测数组为准。
