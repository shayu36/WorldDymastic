# DSQE Oracle 诊断实验记录

## 1. 实验范围

本记录用于定位 DSQE 未来预测指标下降的原因。实验使用同一个 DSQE 检查点和同一批验证样本，只替换一个推理条件。

- 配置：`configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py`
- 检查点：`work_dirs/dsqe-ddp-32-baseline56-b2/epoch_32.pth`
- 检查点类型：最后检查点（final/last checkpoint），不是 best checkpoint
- 训练：32 epochs，初始权重为本地 BaseLine `epoch_56.pth`
- 训练硬件：2 张 RTX 3090
- 评估硬件：2 张 GPU，`torchrun --nproc_per_node=2`
- 验证子集：验证集前 500 个样本（`--max-samples 500`）
- 评估命令：`tools/test.py --eval segm --deterministic`
- 报告时域：仅报告 1s、2s、3s，不增加平均列

实验 A（关闭 DSQE 与 BaseLine 恒等复现）按要求不执行。

## 2. 四个对照设置

| 实验 | 自车位姿 | 动静态角色 | 未来语义 | 目的 |
|---|---|---|---|---|
| B0 | 模型预测 | 模型预测 | `base_semantics + semantic correction` | DSQE 普通推理基线 |
| B1 | GT 自车相对位姿 | 模型预测 | `base_semantics + semantic correction` | 检查自车位姿/坐标变换 |
| C1 | 模型预测 | GT 动静态角色 | `base_semantics + semantic correction` | 检查角色路由和角色传播 |
| D1 | 模型预测 | 模型预测 | legacy `cls_branch` | 检查未来语义输出分支 |

所有实验的当前帧 0s 结果均为 `IoU=24.79%`、`mIoU=17.67%`；下面只列未来 1s/2s/3s。

## 3. 语义占用结果

### 3.1 总体 IoU 与 mIoU

| 实验 | IoU 1s (%) | IoU 2s (%) | IoU 3s (%) | mIoU 1s (%) | mIoU 2s (%) | mIoU 3s (%) |
|---|---:|---:|---:|---:|---:|---:|
| B0 普通 DSQE | 21.44 | 20.78 | 20.01 | 9.15 | 7.51 | 6.47 |
| B1 GT pose | 21.56 | 20.96 | 20.33 | 9.27 | 7.61 | 6.57 |
| C1 GT role | 21.41 | 20.68 | 19.89 | 9.04 | 7.43 | 6.40 |
| D1 legacy semantics | 0.85 | 0.16 | 0.06 | 0.35 | 0.05 | 0.02 |

### 3.2 dynamic/static mIoU

| 实验 | dynamic 1s (%) | dynamic 2s (%) | dynamic 3s (%) | static 1s (%) | static 2s (%) | static 3s (%) |
|---|---:|---:|---:|---:|---:|---:|
| B0 普通 DSQE | 7.13 | 5.37 | 4.37 | 12.26 | 10.57 | 9.39 |
| B1 GT pose | 7.29 | 5.48 | 4.46 | 12.37 | 10.67 | 9.50 |
| C1 GT role | 6.89 | 5.21 | 4.23 | 12.27 | 10.55 | 9.36 |
| D1 legacy semantics | 0.19 | 0.00 | 0.00 | 0.54 | 0.10 | 0.04 |

### 3.3 逐类别 IoU：1s

| 类别 | B0 | B1 | C1 | D1 |
|---|---:|---:|---:|---:|
| others | 0.32 | 0.35 | 0.33 | 0.00 |
| barrier | 12.03 | 12.25 | 12.01 | 0.00 |
| bicycle | 3.14 | 3.26 | 3.21 | 0.00 |
| bus | 28.64 | 29.27 | 28.27 | 0.18 |
| car | 5.01 | 5.05 | 4.52 | 1.03 |
| construction_vehicle | 7.36 | 7.66 | 6.85 | 0.00 |
| motorcycle | 6.59 | 6.79 | 6.23 | 0.00 |
| pedestrian | 0.79 | 0.80 | 0.75 | 0.00 |
| traffic_cone | 1.41 | 1.33 | 1.49 | 0.00 |
| trailer | 0.00 | 0.00 | 0.00 | 0.00 |
| truck | 5.55 | 5.49 | 5.32 | 0.33 |
| driveable_surface | 24.01 | 24.08 | 23.84 | 4.29 |
| other_flat | 9.90 | 10.16 | 9.83 | 0.00 |
| sidewalk | 13.55 | 13.65 | 13.61 | 0.00 |
| terrain | 13.68 | 13.73 | 13.75 | 0.00 |
| manmade | 9.40 | 9.54 | 9.40 | 0.00 |
| vegetation | 14.14 | 14.22 | 14.23 | 0.06 |

### 3.4 逐类别 IoU：2s

| 类别 | B0 | B1 | C1 | D1 |
|---|---:|---:|---:|---:|
| others | 0.17 | 0.17 | 0.16 | 0.00 |
| barrier | 10.16 | 10.42 | 10.25 | 0.00 |
| bicycle | 2.76 | 2.83 | 2.62 | 0.00 |
| bus | 19.39 | 19.92 | 19.45 | 0.03 |
| car | 3.26 | 3.31 | 3.06 | 0.00 |
| construction_vehicle | 7.54 | 7.98 | 7.05 | 0.00 |
| motorcycle | 5.19 | 5.14 | 4.83 | 0.00 |
| pedestrian | 0.49 | 0.49 | 0.45 | 0.00 |
| traffic_cone | 0.56 | 0.49 | 0.67 | 0.00 |
| trailer | 0.00 | 0.00 | 0.00 | 0.00 |
| truck | 4.33 | 4.20 | 4.24 | 0.00 |
| driveable_surface | 21.68 | 21.77 | 21.45 | 0.80 |
| other_flat | 8.15 | 8.32 | 8.09 | 0.00 |
| sidewalk | 11.64 | 11.73 | 11.65 | 0.00 |
| terrain | 12.20 | 12.25 | 12.16 | 0.00 |
| manmade | 8.11 | 8.23 | 8.10 | 0.01 |
| vegetation | 12.06 | 12.14 | 12.07 | 0.02 |

### 3.5 逐类别 IoU：3s

| 类别 | B0 | B1 | C1 | D1 |
|---|---:|---:|---:|---:|
| others | 0.02 | 0.01 | 0.01 | 0.00 |
| barrier | 8.69 | 8.81 | 8.72 | 0.00 |
| bicycle | 2.53 | 2.66 | 2.36 | 0.00 |
| bus | 13.83 | 13.85 | 14.39 | 0.00 |
| car | 2.43 | 2.47 | 2.33 | 0.00 |
| construction_vehicle | 7.78 | 8.26 | 6.82 | 0.00 |
| motorcycle | 4.19 | 4.15 | 3.96 | 0.00 |
| pedestrian | 0.37 | 0.38 | 0.34 | 0.00 |
| traffic_cone | 0.27 | 0.31 | 0.25 | 0.00 |
| trailer | 0.00 | 0.00 | 0.00 | 0.00 |
| truck | 3.79 | 3.90 | 3.67 | 0.00 |
| driveable_surface | 20.24 | 20.35 | 20.04 | 0.29 |
| other_flat | 6.66 | 6.83 | 6.80 | 0.00 |
| sidewalk | 10.52 | 10.61 | 10.36 | 0.00 |
| terrain | 11.16 | 11.18 | 11.11 | 0.00 |
| manmade | 7.15 | 7.30 | 7.14 | 0.00 |
| vegetation | 10.43 | 10.58 | 10.45 | 0.01 |

## 4. 500 样本规划指标

规划值由 `tools/evaluate_planning_temporal.py` 对四个 oracle 输出文件计算。由于这里只使用 500 个样本，碰撞率步长较粗；这些数值不能替代完整 4219 样本的 BaseLine 标准。

| 实验 | L2 1s (m) | L2 2s (m) | L2 3s (m) | plan_obj_box_col 1s (%) | plan_obj_box_col 2s (%) | plan_obj_box_col 3s (%) |
|---|---:|---:|---:|---:|---:|---:|
| B0 普通 DSQE | 0.2101 | 0.2546 | 0.3119 | 0.3000 | 0.2500 | 0.2000 |
| B1 GT pose | 0.2101 | 0.2548 | 0.3123 | 0.3000 | 0.2500 | 0.2000 |
| C1 GT role | 0.2079 | 0.2508 | 0.3088 | 0.2000 | 0.2000 | 0.1667 |
| D1 legacy semantics | 0.2103 | 0.2550 | 0.3114 | 0.3000 | 0.2500 | 0.2333 |

## 5. 诊断结论

1. **B1：GT 自车位姿没有明显恢复指标。** 相对 B0，IoU 仅变化 `+0.12/+0.18/+0.32` 个百分点，mIoU 仅变化 `+0.12/+0.10/+0.10` 个百分点。因此当前下降不能主要归因于预测自车位姿。
2. **C1：GT 动静态角色没有改善指标。** C1 的 IoU/mIoU 略低于 B0，单独修正角色标签不能恢复动态类和静态类表现，角色路由不是唯一主因。
3. **D1：legacy `cls_branch` 替换后语义几乎崩溃。** 1s/2s/3s mIoU 变为 `0.35/0.05/0.02`。该结果说明当前 legacy 分支不能直接接收 DSQE 演化后的 `next_feat`，但不能简单解释为“Baseline 语义分支本身错误”：DSQE 路径中的 `cls_branch` 与 DSQE 特征分布、输出分支训练状态并不等价。若要把 D1 作为严格语义分支诊断，需要进一步使用 Baseline 特征路径或单独加载并验证兼容的 legacy 语义头。
4. **当前最明显的问题仍在 DSQE 主路径的未来点/语义联合演化。** B1、C1 都没有恢复，而 D1 证明简单替换语义头也不可直接使用；后续应优先检查 DSQE `next_feat` 到未来语义 logits 的接口、点位置演化和语义监督匹配。

## 6. 原始日志和输出文件

| 实验 | 日志 | exit 文件 | 规划输出 |
|---|---|---|---|
| B0 | `logs/dsqe-oracle-base500.log` | `logs/dsqe-oracle-base500.exit` | `work_dirs/dsqe-ddp-32-baseline56-b2/eval_oracle/output_base500.pkl` |
| B1 | `logs/dsqe-oracle-pose500.log` | `logs/dsqe-oracle-pose500.exit` | `work_dirs/dsqe-ddp-32-baseline56-b2/eval_oracle/output_pose500.pkl` |
| C1 | `logs/dsqe-oracle-role500.log` | `logs/dsqe-oracle-role500.exit` | `work_dirs/dsqe-ddp-32-baseline56-b2/eval_oracle/output_role500.pkl` |
| D1 | `logs/dsqe-oracle-semantics500.log` | `logs/dsqe-oracle-semantics500.exit` | `work_dirs/dsqe-ddp-32-baseline56-b2/eval_oracle/output_semantics500.pkl` |

四个最终评估日志的 `exit_code` 均为 `0`。此前 B1 和 D1 各有一次因 shell 工作目录/换行错误导致的失败尝试，不计入实验结果。
