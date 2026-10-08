# CESM2 历史版本与同设置基线核验

本报告核对用户粘贴的历史表格，并记录可重放的模型、评分点和缺失的产物。新的架构必须在这一 CESM2 设置下重新训练和评估；真实 Argo 的结果不能直接作为这张表的改进值。

## 已核实的实验设置

| 项目 | 核实结果 |
|---|---|
| 数据 | `data/synthetic_argo/cesm2_uniform.nc`，CESM2 单成员模拟真值，连续均匀海洋位置处双线性采样 |
| 数据 SHA-256 | `773c5b3f3f16749e0c89570c34e57cf29e59c85902dd61a407c27af7645f0b83`，本次重放已与历史基线记录核对 |
| 年份 | 训练 2000–2003；验证 2004；开发评估 2005（旧报告称 test；该年已被多次使用） |
| 每月剖面 | 6,080 条输入，1,520 条留出查询；72 个月固定剖面池；每条剖面独立 float ID |
| 深度 | 20 个 CESM2 原生层，5–984.7 m，无垂向插值 |
| 目标 | T/S 减去该剖面自身位置处的训练年月气候态；训练年份统计量归一化 |
| 固定评分点 | 验证每月最多 8,000 个值，月份种子 `20260918`；2005 所有查询值完整评分 |
| 有效值数 | 验证每个变量 92,517；2005 每个变量 351,895 |
| 数据处理 | 29 个沿海深层 T=S=0 的重网格填充值柱在空间插值前移除 |

共同数据与评分接口是 `src/ocean_tokenizer/synth_argo_eval.py` 的 `load`、`eval_month` 和 `eval_set`。剖面来源、月份、剖面索引、深度索引与目标数组均需匹配。相同有效值总数本身不足以证明评分点相同。

## 历史结果产物

下表采用物理单位 RMSE。三种子行显示均值 ± 样本标准差，种子为 1234、1235、1236。

| 方法/设置 | 温度 RMSE (°C) | 盐度 RMSE (PSU) | 核实产物与状态 |
|---|---:|---:|---|
| 气候态 | 0.652775 | 0.111907 | `fixed_baselines/summary.json`；本次完整预测重放一致 |
| 最近邻剖面 | 0.172770 | 0.037145 | 同上；本次完整预测重放一致 |
| 验证集调参 OI | 0.085980 | 0.022639 | 同上及 `tuning.json`；本次使用冻结的 2004 设置完整重放一致 |
| Pointwise MLP，固定种子 1234 | 0.1745 | 0.0372 | `pointwise_mlp/summary_seed1234.json` 存在，模型权重缺失，尚不能计算新的 MAE/R² |
| 旧 token 模型，64 latent slots，15,000 步，500 km / gate 1 | 0.229254 ± 0.002254 | 0.041723 ± 0.000370 | `k15_l64_syn_r500_g1/summary_seed1234..1236.json` 存在，三种子权重均缺失；需要重新训练后导出预测 |
| 注册初始 refiner，cell-centre 目标 | 0.311160 ± 0.001620 | 0.059836 ± 0.000304 | `k15_l64_syn_reg_cell/summary_seed1234..1236.json` |
| 注册初始 refiner，Argo-position 目标 | 0.230924 ± 0.005333 | 0.044652 ± 0.000379 | `k15_l64_syn_reg/summary_seed1234..1236.json` |

历史目录均位于 `outputs/audit/synthetic/`。可比较历史 RMSE 汇总，但不能从这些汇总数反推出 MAE、bias、R²、相关系数或置信区间。

用户粘贴的多模态旧模型 0.2124 / 0.0371、4DVarNet 剖面输入 0.2219 / 0.0457、多模态输入 0.2269 / 0.0371，暂未在 main、anchored-fusion 或相邻旧 checkout 中定位到相应 runner、权重、预测或汇总文件。当前旧 `62_sanity_train.py` 显式拒绝 synthetic `--surface`，旧 synthetic 报告也明确是 profiles only。因此这些数字保留为**用户提供、尚未核实的历史参照**，不能写成这次已经重现的结果。

粘贴表格最后一句称 4DVarNet “trained here on Argo profiles only”，但表中另列 Argo + satellite 一行；两者存在表述矛盾。在找到对应运行配置前，不推断该多模态行究竟如何训练。

## 已完成的固定基线重放

新增 `experiments/synthetic/47_replay_previous_models.py` 直接使用历史 OI 验证集选择，并核验六份输出的原始 RMSE 和有效值数：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /DATA2/zihao/projects/ocean_project/.venv/bin/python \
  experiments/synthetic/47_replay_previous_models.py
```

产物位于 `outputs/synthetic_matched_20261007/previous_baselines/`：

- `climatology_{validation,development}.npz`
- `nearest_profile_{validation,development}.npz`
- `oi_{validation,development}.npz`
- `replay_manifest.json`，包含数据、OI 设置、runner 与输出 SHA-256。

所有方法共享验证评分指纹 `eea02a59f13e72a8151c87c9d228d7fa4a882c46bf108ff98b1effc3e3ef1abe` 和开发评分指纹 `0ebdfec1ca327fbbd5948fd47a5c4410a5ba14b9cdb96a4c141a1ae5a2f3099a`。

NPZ 中 `target`/`mean` 为标准化异常；`target_physical`/`mean_physical` 为物理单位异常；另存 `*_absolute_physical` 用于明确计算绝对海洋状态的 R² 和相关系数。RMSE、MAE 和 bias 在预测及真值加回同一气候态时不变，R² 和相关系数则会改变，报告必须注明采用哪一种目标。确定性基线没有分布预测，不为其编造不确定性。

## 旧模型的严格同设置重跑

新增 `experiments/synthetic/49_previous_token_matched.py` 执行旧 `62_sanity_train.py` 的原始 AST 模型、训练采样、损失、AdamW 与调度器代码；旧文件没有被修改。仅增加完整优化器/RNG 检查点、严格参数检查与公共预测数组导出。

正式配置是 407,111 参数、64 个 latent slots、宽度 64、2 个 latent self-attention block、2 个 decoder block、4 heads、FP32、每步 1 个月 / 1,024 查询、target dropout 0.2、15,000 步、warmup 300、LR 0.001、weight decay 0.01、refiner 初始水平尺度 500 km / 垂直尺度 100 m / gate 1.0。模型看见每月完整的输入剖面池，剖面编码器内部仍是旧深度带压缩路径。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=<gpu> \
  /DATA2/zihao/projects/ocean_project/.venv/bin/python \
  experiments/synthetic/49_previous_token_matched.py \
  --region synthetic --mode train --backbone d4rt --mass-mode dfs \
  --ablation anomaly_exact --seed 1234 --n-latent 64 \
  --refiner-km 500 --refiner-gate 1.0 --steps 15000 \
  --tag previous_token_64slots_15k_s1234 \
  --out-root outputs/synthetic_matched_20261007/previous_token_s1234
```

种子 1235、1236 使用各自独立输出目录。中断后在完全相同命令中增加 `--resume-training`，恢复模型、AdamW、学习率调度器、CPU/CUDA RNG、NumPy RNG 和验证选中的 best state。`--evaluation-only` 从已保存的最佳权重导出验证/开发预测。诊断运行必须显式使用 `--diagnostic`，不能计入正式比较。

CPU 续训诊断已通过：在固定 8 步学习率计划下，先训练 4 步再恢复至 6 步，与连续训练 6 步的模型、AdamW、调度器、NumPy/Torch RNG、最佳模型和损失记录完全一致。证据保存于 `outputs/synthetic_matched_20261007/previous_token_continuation_validation_cpu.json`。此处只验证续训工程行为，没有产生新的正式精度结论。

## 4DVarNet 复现来源与多模态边界

本次已取得作者仓库 `external/4dvarnet-starter`，commit `20f1b5f34b201342cde6dd21a30419d07541db54`。仓库提供 `src/models.py` 的 `GradSolver`、`BilinAEPriorCost`、`ConvLstmGradModel`，以及 `contrib/three_d` 和 `contrib/multimodal`。后续 runner 必须清楚记录三维温盐观测算子、监督目标和卫星观测算子适配，不能将重新实现的泛用网络称为已复现作者方法。

仓库已有 `ocean_tokenizer.ssh` 的 TEOS-10 steric/dynamic-height 实现，参考压力 990 dbar；它从同一 TEMP/SALT 真值推导，缺少 barotropic 成分，不能称为独立 altimetry。用户粘贴版本的 exact 5 m SST/SSS 和真值导出 sea level 设置在完成相应数据处理核验前，仍是用户指定的待复现配方。新的与旧的多模态方法必须使用同一个明确记录的合成表面场来源与空间采样规则。

## 比较解释

从旧模型改到当前架构会同时增加逐层数值观测、新息/OI 初猜、独立局部修正、共享 latent 或 Soft MoE，以及新的不确定性头。完整系统的改进需要先在相同数据和输入设置下成立；共享 latent、MoE 和局部通路各自的贡献还需组件消融。参数量、训练步数和单模型/集成口径必须展示，不能把容量、初猜和新增信息的收益全部归因于 MoE。
