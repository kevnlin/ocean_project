# CESM2 同设置实验：运行与交付审计

审计时间：2026-10-07 11:34 UTC（06:34 CDT）。本文件是这一时刻的状态快照；持续更新见 [运行进度](matched_progress_20261007.md)。

**实现与公共基线已就绪，完整架构比较仍在运行。** 注册的 39 次训练中，Dense Transformer 64 的三个种子均已训练到 15000 步；6 次训练正在运行，30 次等待调度。尚未生成冻结选择、任何新增架构的 2005 预测或最终指标，不能宣布新架构超过此前最佳方案。

| 部分 | 已核验事实 | 尚待完成 |
|---|---|---|
| 正式训练 | 13 个架构/输入配置 × 1234/1235/1236；每次 15000 步；当前 3/39 完成 | 其余 36 次训练与验证导出 |
| 旧 64-slot 模型 | 原训练 AST、FP32 配方、500 km/gate 1 与恢复状态保留；三个 Argo 种子正在训练 | Argo 三种子及新注册 surface 三种子完整结果 |
| 4DVarNet | 固定作者源码版本；适配代码测试与真实协议短运行通过；ready 源码指纹一致 | Argo/surface 各三种子完整训练 |
| 旧模型 surface 适配 | 缓存身份、训练期归一化、字段完整性、通道映射及原模型前后向测试通过；ready 指纹一致 | 该组正式训练尚未启动，不能把单测写成正式结果 |
| Pointwise MLP | 原 seed 1234、30 epoch 配方完成；公共验证预测与检查点契约通过 | 冻结选择后导出 2005 预测 |
| 既有 learned OI | 已有 120 参数、1500 步检查点重放；公共验证/2005 预测与历史误差完全一致 | 作为固定强对照参与完整报告；不新增训练 |
| 选择与报告 | 仅验证集选型，允许经典/既有方法胜出；指标、身份、预算与完整性交付守卫通过 | 全部训练结束后冻结选择、共同评分、生成 completion.json |

此前最好可重放方案为既有 learned OI：2005 温度 RMSE **0.076809 °C**、盐度 RMSE **0.021750 PSU**。相比之下，44 经典 OI 为 **0.085980 °C / 0.022639 PSU**。最终报告必须与更强的既有方案比较，不能只超过后者便宣布超过此前最好结果。2005 已在历史实验中使用，属于开发证据。

## 流程与统计口径

51 队列不会在全部注册训练完成前启动选型。53 读取公共 2004 验证身份、训练预算、最佳权重与源代码/数据指纹；要求每组三个独立种子、同一评分点和训练期逐深度归一化。冻结后，新增模型才导出 2005 预测。真实现有的 Dense 64 三个检查点、MLP 和 learned OI 已通过这些验证契约。

46 的完成证据是保存历史的最后一步 15000；它没有独立 `completed_steps` 字段。51/53/56 与 57 均正确读取实际训练历史。写着 nominal 15000、历史只有 14000 的伪完整产物会被拒绝。

最终主表使用物理单位的温度/盐度 RMSE，同时报告 MAE、平均偏差、异常与绝对状态的 R²/相关性，以及可用的不确定性指标。三次训练的均值 ± **样本标准差**与三种子预测集成分开列出。OI 与既有 learned OI 各自仅使用 2004 的逐深度残差 RMS 校准尺度；该尺度不是精确 OI 后验。确定性模型不补造不确定性。

所有 surface 方法共享 SST/SSS/steric SSH 产品与缓存，但编码形式不同：新模型在剖面/查询位置采样，旧模型使用 3×3 栅格 patch，4DVarNet 使用整张辅助观测场。它们比较的是完整系统；合成产品来自无噪声 CESM2 真值，不能直接推出真实卫星条件下的优势。

## 收尾风险与处理

发现的风险是：已启动的旧版 51 进程没有新增的“最终指标必须存在”守卫，而 53 对尚未齐全的报告可以返回 pending。单看旧版队列 `phase=complete` 不足以证明交付完成。

已由独立 [57 交付守卫](../../experiments/synthetic/57_finalize_matched_campaign.py)补齐：它等待冻结选择、39 次开发预测和 MLP 预测全部存在，必要时重新生成报告，核验 39 个不同 run、13 个完整三种子配置、实际 15000 步训练证据、351895 个温/盐评分值和必需强基线，随后才写 `completion.json`。该服务已采用加强后的代码运行。**最终完成证据为 `completion.json` 与其中的报告/指标/选择 SHA256。**

当前主训练、独立 MLP 评价、进度服务和最终交付守卫均在 user systemd 下持续运行；本次读取未发现重试、NaN、OOM 或服务异常。主训练内存约 14 GiB，设置 `MemoryHigh=24G`、`MemoryMax=32G`、`MemorySwapMax=0`；不会因聊天回合结束而退出。以上是当前状态，不是剩余训练全部成功的保证。

## 已执行验证

- 当前 51/53/52/55/54/4DVarNet 相关 **71 个 CPU 测试通过**。
- 独立交付测试 [test_matched_finalization.py](../../tests/test_matched_finalization.py) **12 个通过**，包含漏配置、漏种子、缩短预算、46 nominal-only 完成证据、缺强基线、缺评分值、选择前不检查开发产物，以及 pending/退出 0 不得生成完成凭证。
- 新版 51 的 39-run 编排模拟通过：完整训练 → 冻结验证选择 → 开发评价 → MLP 评价 → 报告；缺最终指标的模拟必须报错。该模拟验证流程，不计入科学训练结果。
- 当前 campaign 的全部命令与 51 重新生成的注册命令一致；4DVarNet/52 ready 标记包含的源码 SHA256 与实际文件一致。

## 查询与故障后续跑

读取当前状态：

```bash
systemctl --user status ocean-synthetic-matched-20261007 ocean-synthetic-fixed-evaluation-20261007 ocean-synthetic-progress-20261007 ocean-synthetic-finalize-20261007 --no-pager
```

只有主队列已失败或停止、且没有同批训练子进程时，才使用下面的恢复命令。现存服务仍处于 active 时无需重启。队列根据各 run 的 `last` 完整状态恢复模型、优化器、学习率和 RNG；完成的 15000 步产物会被跳过，注册 manifest 必须保持一致。

```bash
OCEAN_WORKTREE=/DATA2/zihao/projects/ocean_project/.claude/worktrees/anchored-fusion
systemd-run --user --unit=ocean-synthetic-matched-recovery-20261007 \
  --property=MemoryHigh=24G --property=MemoryMax=32G --property=MemorySwapMax=0 \
  --working-directory="$OCEAN_WORKTREE" \
  --setenv=OMP_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4 \
  --setenv=PYTHONPATH="$OCEAN_WORKTREE/src" \
  /DATA2/zihao/projects/ocean_project/.venv/bin/python -u \
  "$OCEAN_WORKTREE/experiments/synthetic/51_matched_campaign.py" \
  --gpus 0,1,2,3,6,7 --phase all
```

若训练与评分已全部结束、仅交付守卫失败，查看其 `finalization_status.json` 与服务日志后单独恢复它；无需重新训练。报告仍缺产物时，不删除冻结选择，也不把 pending 当作完整结果。
