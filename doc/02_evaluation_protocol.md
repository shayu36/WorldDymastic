# 评估协议

## 1. 评估对象

主比较包含：

- 本地复现 BaseLine：epoch 56；
- DSQE：epoch 21、25、30、32。

DSQE 检查点位于：`work_dirs/dsqe-ddp-32-baseline56-b2/`。每个检查点均有一份完整 validation 日志和一份轨迹输出 pickle。

## 2. 语义占用评估

语义评估由 `tools/test.py --eval segm` 完成。最终日志中的数组包含 `[0s, 1s, 2s, 3s]` 四个时间位置。本报告只使用后三项：

- `IoU_1s / IoU_2s / IoU_3s`；
- `mIoU_1s / mIoU_2s / mIoU_3s`。

因此，报告表格不会再增加一个跨时域平均列，也不把当前帧 0s 指标混入 1s/2s/3s 结果。

原始机器可读结果：

- `work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/summary.csv`
- `work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/summary.json`

评估日志末尾的结果字典还包含 dynamic_mIoU 和 static_mIoU，本报告在详细表中一并列出。

## 3. 规划评估

规划结果由 `tools/evaluate_planning_temporal.py` 按项目 ST-P3 `PlanningMetric` 的轨迹部分计算。默认标签为：

- occupancy：`/data/jxy/projects/admlp/stp3_val/stp3_occupancy.pkl`；
- future trajectory：`/data/jxy/projects/admlp/stp3_val/stp3_traj_gt.pkl`；
- token filter：项目 ST-P3 validation token 子集。

本次每个 DSQE 轨迹 pickle 与 occupancy/trajectory 标签共有 4219 个有效 token，预测轨迹为 6 个未来点。

### 3.1 L2

对每个有效 token，计算预测点与 GT 点的欧氏距离，然后分别对前 2、4、6 个点求样本平均：

```text
L2_1s = mean(distance[0:2])
L2_2s = mean(distance[0:4])
L2_3s = mean(distance[0:6])
```

单位为米，越低越好。

### 3.2 plan_obj_box_col

使用 1.85m x 4.084m 的自车盒，将预测轨迹投影到 0.5m 分辨率的 200 x 200 BEV occupancy 上，统计预测盒碰撞且 GT 盒未碰撞的比例。输出乘以 100，单位为百分比，越低越好。

规划结果文件：

- `work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/planning_epoch_21.json`
- `work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/planning_epoch_25.json`
- `work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/planning_epoch_30.json`
- `work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/planning_epoch_32.json`

## 4. BaseLine 标准

语义占用 BaseLine 来自：

`/data/jxy/projects/work_dirs/sparseworld-traj-memory-only/eval_epoch56_memory_off.log`

规划 BaseLine 按用户指定的本地复现标准记录为：

| 指标 | 1s | 2s | 3s |
|---|---:|---:|---:|
| mIoU (%) | 14.96 | 13.18 | 11.53 |
| IoU (%) | 23.15 | 22.27 | 21.21 |
| L2 (m) | 0.1596 | 0.1945 | 0.2393 |
| 碰撞率 plan_obj_box_col (%) | 0.0948 | 0.1007 | 0.1106 |

规划预测文件为 `/data/jxy/projects/admlp/output_data.pkl`。该文件与 ST-P3 标签共有 4219 个有效 token。

## 5. 口径差异说明

使用当前通用脚本直接复算 BaseLine 时，L2 与上述标准完全一致；碰撞率得到的 1s/2s 为 `0.0948%/0.1007%`，3s 为 `0.1067%`，而用户指定的对照标准为 `0.1106%`。为保证报告与本地复现实验的判定标准一致，本文所有 BaseLine 对比均使用用户指定的 `0.1106%` 作为 3s 标准，并保留这一差异说明，不静默替换基准值。

## 6. 评估完整性

- epoch 21、25、30、32 的语义日志均以完整 IoU/mIoU 结果字典结束；
- 四个 checkpoint 均有对应 `output_data_epoch_*.pkl`；
- 四个 checkpoint 均有规划 JSON/CSV/TXT 汇总；
- 不使用 oracle pose、oracle role 或 legacy semantics 评估开关；
- 主表只报告 1s、2s、3s，不报告 Avg。

## 7. 其他运行的处理

仓库中还保留了 `logs/dsqe-ddp-six-step-v4.log` 等早期 IterBased/短程 smoke 运行。这些运行使用了不同的迭代预算、batch/累积配置或验证入口，不能与本次 `dsqe-ddp-32-baseline56-b2` 的 epoch 21/25/30/32 结果直接横向比较，因此不混入主结果表；相关原始日志仍保留在 `logs/` 目录中。
