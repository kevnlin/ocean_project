# 多模态海洋共享潜在状态实验

正式 search 已完成 7/7；开发集结果只能在模型和配置冻结后生成。

目标：从 SST、SSS、SLA 和稀疏 Argo 逐深度观测重建连续海洋温盐状态，再通过独立位置、深度、日期查询得到温度、盐度与不确定性。当前数据与评估仅覆盖 20 个固定深度层；尚未证明任意深度的连续重建能力。

## 架构与对照

固定已保存的卫星初猜、训练归一化、输入剖面池和评分身份。个体深度观测编码器读入卫星特征、初猜、测量误差及有效性，并保留数值新息；共享 latent 通过观测 cross-attention 与 latent self-attention 交换信息，独立查询解码器输出温盐残差和高斯尺度，同时保留原始局部逐层新息通路。

比较 dense latent Transformer、latent Soft MoE 和带每查询局部 Transformer 的 latent backbone；每类包含 base/large 两档。Soft MoE 使用连续 dispatch/combine，不能据此宣称稀疏计算加速。卫星 token 来自上下文剖面位置的卫星特征，尚未直接编码完整稠密卫星栅格。

新模型默认同时训练 OI 参数，因此增加 analysis-only 对照。latent 的收益应同时相对固定初猜加 OI anchor，以及同设置重新训练的 OI-only 对照评估。当前新增模块并不单独保证重复记录证据不变性。

## 2021 验证集架构搜索

仅以验证集标准化温盐 RMSE 的平均值 macro_z 选取 checkpoint 与 base/large 配置；J 为 RMSE / 目标 RMS，越小越好。step 0 可成为最佳 checkpoint，表示当前训练没有超越初始化 anchor。

| 配置 | 状态 | 神经参数 | 最佳 step | TEMP J | SALT J | macro_z |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dense_base_s1234 | 完成 | 1,506,838 | 3000 | 0.62052 | 0.64944 | 0.67174 |
| dense_large_s1234 | 完成 | 4,386,840 | 5000 | 0.61771 | 0.64741 | 0.66919 |
| soft_moe_base_s1234 | 完成 | 2,696,730 | 6000 | 0.62020 | 0.65028 | 0.67203 |
| soft_moe_large_s1234 | 完成 | 13,727,262 | 5000 | 0.61751 | 0.64790 | 0.66936 |
| local_transformer_base_s1234 | 完成 | 1,738,774 | 5500 | 0.61925 | 0.65038 | 0.67159 |
| local_transformer_large_s1234 | 完成 | 4,906,776 | 4500 | 0.61859 | 0.64913 | 0.67058 |
| analysis_only_s1234 | 完成 | 0 | 1000 | 0.62407 | 0.65478 | 0.67646 |

冻结选择：analysis_only_s1234, dense_large_s1234, local_transformer_large_s1234, soft_moe_large_s1234

## 多种子与已使用开发集

训练为 2016–2020，验证为 2021。2022–2023 早已用于先前开发，属于 development，不能作为独立 test 确认或跨论文 SOTA 证据。同月上下文可包含查询日期之后的观测，因此这里是回顾性重建；预测实验需要严格另设时间截断。归一化沿用全部训练年份剖面的既定协议，包含划为上下文之外的训练 WMO，不能宣称这些浮标未参与预处理。

SST/SSS L4 产品可能同化原位观测，其中的 Argo 证据可能回流到卫星输入；沿用 62_sanity_train.py 的既定协议，这一设置不是严格 source-independent 证据评估。所有方法使用相同产品仍可支持本协议下的匹配比较，但不能宣称已排除跨产品复用信息。

| 冻结配置 | split | seeds | TEMP J mean ± std | SALT J mean ± std | macro_z mean ± std |
| --- | --- | ---: | --- | --- | --- |
| analysis_only | validation | 3 | 0.62405 ± 0.00004 | 0.65479 ± 0.00002 | 0.67646 ± 0.00002 |
| analysis_only | development | 3 | 0.62297 ± 0.00002 | 0.75865 ± 0.00007 | 0.76391 ± 0.00003 |
| dense_large | validation | 3 | 0.61793 ± 0.00080 | 0.64841 ± 0.00091 | 0.66985 ± 0.00079 |
| dense_large | development | 3 | 0.61686 ± 0.00049 | 0.75117 ± 0.00019 | 0.75640 ± 0.00028 |
| local_transformer_large | validation | 3 | 0.61890 ± 0.00051 | 0.64925 ± 0.00023 | 0.67080 ± 0.00039 |
| local_transformer_large | development | 3 | 0.61700 ± 0.00033 | 0.75192 ± 0.00118 | 0.75690 ± 0.00060 |
| soft_moe_large | validation | 3 | 0.61771 ± 0.00158 | 0.64807 ± 0.00065 | 0.66954 ± 0.00117 |
| soft_moe_large | development | 3 | 0.61690 ± 0.00064 | 0.75153 ± 0.00038 | 0.75662 ± 0.00054 |

## 所选架构的固定 OI 与成分消融

消融单独收录，不参与 base/large 配置选择。

| 配置 | 固定 OI | latent | 局部通路 | 最佳 step | validation TEMP J | validation SALT J | development TEMP J | development SALT J |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| dense_large_fixed_oi_full_s1234 | True | True | True | 4500 | 0.61861 | 0.64864 | 0.61822 | 0.75286 |
| dense_large_fixed_oi_full_s1235 | True | True | True | 5000 | 0.61850 | 0.64990 | 0.61777 | 0.75237 |
| dense_large_fixed_oi_full_s1236 | True | True | True | 6000 | 0.61983 | 0.64999 | 0.61826 | 0.75204 |
| dense_large_fixed_oi_latent_off_s1234 | True | False | True | 5000 | 0.61841 | 0.64959 | 0.61787 | 0.75306 |
| dense_large_fixed_oi_latent_off_s1235 | True | False | True | 5000 | 0.61817 | 0.64964 | 0.61733 | 0.75158 |
| dense_large_fixed_oi_latent_off_s1236 | True | False | True | 6000 | 0.61998 | 0.65093 | 0.61752 | 0.75157 |
| dense_large_fixed_oi_local_off_s1234 | True | True | False | 5000 | 0.62081 | 0.65120 | 0.62124 | 0.75559 |
| dense_large_fixed_oi_local_off_s1235 | True | True | False | 4000 | 0.62064 | 0.65129 | 0.62150 | 0.75669 |
| dense_large_fixed_oi_local_off_s1236 | True | True | False | 6000 | 0.62246 | 0.65174 | 0.62199 | 0.75701 |


## 不确定性与配对统计

anchor 的温盐 Gaussian sigma 按 level/variable 仅在 2021 验证残差上拟合 RMS，不修正均值；冻结后应用于 development。其在验证集上的校准结果为拟合内结果，不能视作独立校准验证。模型 sigma 为标准化异常值单位，不是物理单位。

多种子预测只有 month/profile/level/target/baseline 完全一致才可平均；ensemble variance = mean(sigma² + mean²) − ensemble_mean²。报告使用整月配对 bootstrap，差值为模型减匹配 anchor，负值代表改进；此区间未包含训练种子或长周期相关性的不确定性。

### analysis_only / seed1234

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.62299 | 0.62326 | [-0.00033, -0.00020] | 1.01816 | 1.00535 | 0.77297 | 0.93335 |
| SALT | 0.75856 | 0.75846 | [0.00003, 0.00016] | 1.44733 | 1.29464 | 0.75816 | 0.92514 |

### analysis_only / seed1235

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.62295 | 0.62326 | [-0.00041, -0.00022] | 1.01806 | 1.00535 | 0.77300 | 0.93340 |
| SALT | 0.75868 | 0.75846 | [0.00003, 0.00036] | 1.44764 | 1.29464 | 0.75797 | 0.92516 |

### analysis_only / seed1236

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.62297 | 0.62326 | [-0.00037, -0.00020] | 1.01812 | 1.00535 | 0.77284 | 0.93337 |
| SALT | 0.75871 | 0.75846 | [0.00008, 0.00037] | 1.44771 | 1.29464 | 0.75798 | 0.92516 |

### analysis_only / ensemble

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.62295 | 0.62326 | [-0.00038, -0.00023] | 1.01797 | 1.00535 | 0.77296 | 0.93339 |
| SALT | 0.75863 | 0.75846 | [0.00003, 0.00028] | 1.44682 | 1.29464 | 0.75805 | 0.92519 |

### dense_large / seed1234

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.61727 | 0.62326 | [-0.00691, -0.00507] | 0.74435 | 1.00535 | 0.74799 | 0.95150 |
| SALT | 0.75132 | 0.75846 | [-0.01071, -0.00437] | 0.98065 | 1.29464 | 0.75342 | 0.95177 |

### dense_large / seed1235

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.61632 | 0.62326 | [-0.00804, -0.00586] | 0.74046 | 1.00535 | 0.73886 | 0.94823 |
| SALT | 0.75124 | 0.75846 | [-0.01056, -0.00478] | 1.08793 | 1.29464 | 0.75185 | 0.95122 |

### dense_large / seed1236

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.61700 | 0.62326 | [-0.00720, -0.00534] | 0.74280 | 1.00535 | 0.74573 | 0.95122 |
| SALT | 0.75095 | 0.75846 | [-0.01037, -0.00551] | 0.88227 | 1.29464 | 0.74257 | 0.94696 |

### dense_large / ensemble

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.61629 | 0.62326 | [-0.00791, -0.00605] | 0.73707 | 1.00535 | 0.74759 | 0.95243 |
| SALT | 0.75060 | 0.75846 | [-0.01131, -0.00526] | 0.91643 | 1.29464 | 0.75334 | 0.95240 |

### local_transformer_large / seed1234

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.61728 | 0.62326 | [-0.00691, -0.00510] | 0.74424 | 1.00535 | 0.73650 | 0.94686 |
| SALT | 0.75214 | 0.75846 | [-0.00985, -0.00368] | 0.93652 | 1.29464 | 0.73158 | 0.94212 |

### local_transformer_large / seed1235

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.61664 | 0.62326 | [-0.00774, -0.00548] | 0.74608 | 1.00535 | 0.76081 | 0.95672 |
| SALT | 0.75297 | 0.75846 | [-0.00909, -0.00270] | 0.99141 | 1.29464 | 0.75714 | 0.95254 |

### local_transformer_large / seed1236

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.61709 | 0.62326 | [-0.00726, -0.00511] | 0.74177 | 1.00535 | 0.74861 | 0.95264 |
| SALT | 0.75064 | 0.75846 | [-0.01011, -0.00627] | 1.02862 | 1.29464 | 0.73811 | 0.94519 |

### local_transformer_large / ensemble

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.61610 | 0.62326 | [-0.00808, -0.00626] | 0.73668 | 1.00535 | 0.75342 | 0.95505 |
| SALT | 0.75109 | 0.75846 | [-0.01067, -0.00484] | 0.94588 | 1.29464 | 0.74783 | 0.94999 |

### soft_moe_large / seed1234

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.61753 | 0.62326 | [-0.00657, -0.00491] | 0.74694 | 1.00535 | 0.74781 | 0.95137 |
| SALT | 0.75197 | 0.75846 | [-0.01051, -0.00327] | 0.89203 | 1.29464 | 0.75731 | 0.95268 |

### soft_moe_large / seed1235

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.61626 | 0.62326 | [-0.00812, -0.00591] | 0.74209 | 1.00535 | 0.73535 | 0.94659 |
| SALT | 0.75129 | 0.75846 | [-0.01061, -0.00463] | 1.10808 | 1.29464 | 0.74820 | 0.94933 |

### soft_moe_large / seed1236

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.61691 | 0.62326 | [-0.00732, -0.00536] | 0.74551 | 1.00535 | 0.74471 | 0.95044 |
| SALT | 0.75134 | 0.75846 | [-0.01106, -0.00403] | 0.84067 | 1.29464 | 0.73854 | 0.94568 |

### soft_moe_large / ensemble

| 变量 | J | anchor J | ΔJ 95% month CI | NLL | anchor calibrated NLL | 68% coverage | 95% coverage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| TEMP | 0.61630 | 0.62326 | [-0.00789, -0.00605] | 0.73892 | 1.00535 | 0.74627 | 0.95190 |
| SALT | 0.75093 | 0.75846 | [-0.01148, -0.00436] | 0.84753 | 1.29464 | 0.75228 | 0.95170 |

逐深度指标、calibration sigma/counts、完整配对差值与种子列表保存在同名 JSON。未完成训练、未生成数组或缺少开发集时不填造数字。

方法来源：[Perceiver IO](https://arxiv.org/abs/2107.14795)、[Soft MoE](https://arxiv.org/abs/2308.00951)、[ARMOR3D](https://os.copernicus.org/articles/8/845/2012/)。已有卫星初猜加剖面修正先例；这里的潜在贡献需由共享状态更新、准确率、校准和泛化实验支持。
