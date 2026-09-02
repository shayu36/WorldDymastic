# 实验结果

## 1. 语义占用结果

所有数值来自 `work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/summary.csv/json`。BaseLine 的 1s/2s/3s 取自 epoch 56 日志，DSQE 取自对应 checkpoint 日志。

### 1.1 IoU 与 mIoU

| 模型 / checkpoint | IoU 1s (%) | IoU 2s (%) | IoU 3s (%) | mIoU 1s (%) | mIoU 2s (%) | mIoU 3s (%) |
|---|---:|---:|---:|---:|---:|---:|
| BaseLine / epoch 56 | 23.15 | 22.27 | 21.21 | 14.96 | 13.18 | 11.53 |
| DSQE / epoch 21 | 21.78 | 20.98 | 20.01 | 9.96 | 8.38 | 7.22 |
| DSQE / epoch 25 | 21.63 | 20.73 | 19.86 | 10.25 | 8.59 | 7.43 |
| DSQE / epoch 30 | 21.74 | 20.99 | 20.16 | 10.32 | 8.64 | 7.51 |
| DSQE / epoch 32 | 21.72 | 20.95 | 20.12 | 10.30 | 8.63 | 7.51 |

### 1.2 DSQE dynamic/static mIoU

| checkpoint | dynamic mIoU 1s (%) | dynamic mIoU 2s (%) | dynamic mIoU 3s (%) | static mIoU 1s (%) | static mIoU 2s (%) | static mIoU 3s (%) |
|---|---:|---:|---:|---:|---:|---:|
| epoch 21 | 7.39 | 5.85 | 4.87 | 13.47 | 11.70 | 10.26 |
| epoch 25 | 7.54 | 5.93 | 4.94 | 13.91 | 12.05 | 10.60 |
| epoch 30 | 7.65 | 6.00 | 5.03 | 13.95 | 12.08 | 10.67 |
| epoch 32 | 7.63 | 5.99 | 5.02 | 13.91 | 12.07 | 10.68 |

BaseLine 日志没有按 dynamic/static 子集输出可直接对应的四项数组，因此这里只列出 DSQE 的拆分结果。

## 2. 规划结果

规划碰撞率使用用户指定的 BaseLine 标准进行比较。数值单位分别为米和百分比。

| 模型 / checkpoint | L2 1s (m) | L2 2s (m) | L2 3s (m) | plan_obj_box_col 1s (%) | plan_obj_box_col 2s (%) | plan_obj_box_col 3s (%) |
|---|---:|---:|---:|---:|---:|---:|
| BaseLine / epoch 56 | 0.1596 | 0.1945 | 0.2393 | 0.0948 | 0.1007 | 0.1106 |
| DSQE / epoch 21 | 0.1651 | 0.2064 | 0.2625 | 0.2607 | 0.2074 | 0.2054 |
| DSQE / epoch 25 | 0.1739 | 0.2213 | 0.2831 | 0.0711 | 0.0770 | 0.1067 |
| DSQE / epoch 30 | 0.1750 | 0.2226 | 0.2899 | 0.2489 | 0.2429 | 0.2844 |
| DSQE / epoch 32 | 0.1763 | 0.2252 | 0.2928 | 0.1067 | 0.1067 | 0.1501 |

## 3. 相对 BaseLine 的差值

差值定义为 `DSQE - BaseLine`。对于 IoU/mIoU，正值更好；对于 L2 和碰撞率，负值更好。

### 3.1 语义占用差值

| checkpoint | ΔIoU 1s | ΔIoU 2s | ΔIoU 3s | ΔmIoU 1s | ΔmIoU 2s | ΔmIoU 3s |
|---|---:|---:|---:|---:|---:|---:|
| epoch 21 | -1.37 | -1.29 | -1.20 | -5.00 | -4.80 | -4.31 |
| epoch 25 | -1.52 | -1.54 | -1.35 | -4.71 | -4.59 | -4.10 |
| epoch 30 | -1.41 | -1.28 | -1.05 | -4.64 | -4.54 | -4.02 |
| epoch 32 | -1.43 | -1.32 | -1.09 | -4.66 | -4.55 | -4.02 |

epoch 30 的语义占用差距最小：3s IoU 差距为 `-1.05` 个百分点，3s mIoU 差距为 `-4.02` 个百分点。epoch 32 没有进一步改善。

### 3.2 规划差值

| checkpoint | ΔL2 1s | ΔL2 2s | ΔL2 3s | Δcollision 1s | Δcollision 2s | Δcollision 3s |
|---|---:|---:|---:|---:|---:|---:|
| epoch 21 | +0.0055 | +0.0119 | +0.0232 | +0.1659 | +0.1067 | +0.0948 |
| epoch 25 | +0.0143 | +0.0268 | +0.0438 | -0.0237 | -0.0237 | -0.0039 |
| epoch 30 | +0.0154 | +0.0281 | +0.0506 | +0.1541 | +0.1422 | +0.1738 |
| epoch 32 | +0.0167 | +0.0307 | +0.0535 | +0.0119 | +0.0060 | +0.0395 |

epoch 21 的轨迹误差距离 BaseLine 最近；epoch 25 的碰撞率优于 BaseLine（1s/2s/3s 分别为 `0.0711%/0.0770%/0.1067%`）。epoch 30 的碰撞率明显回升，epoch 32 虽恢复但仍高于 BaseLine。

## 4. 训练时序与 checkpoint 覆盖范围

训练脚本的保存策略设置为保留全部 checkpoint；但按当前文件系统检查，实验目录实际保留的是 epoch 21 至 epoch 32：

```text
epoch_21.pth, epoch_22.pth, ..., epoch_32.pth
```

本轮已完成完整语义和规划评估的 checkpoint 为 epoch 21、25、30、32；epoch 22–24、26–29、31 目前只有权重文件，未纳入本报告的完整评估表。epoch 1–20 的权重文件当前不在该实验目录中。

## 5. 结果文件索引

- 语义汇总：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/summary.csv`
- 语义差值：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/delta_vs_baseline.csv`
- epoch 21 规划：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/planning_epoch_21.csv/json/txt`
- epoch 25 规划：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/planning_epoch_25.csv/json/txt`
- epoch 30 规划：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/planning_epoch_30.csv/json/txt`
- epoch 32 规划：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/planning_epoch_32.csv/json/txt`
- epoch 21/25/30/32 语义日志：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/epoch_*_eval.log`
