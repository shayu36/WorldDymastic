# DSQE Residual Stage-1 训练与评估结果

## 1. 实验范围

本记录对应残差建模修正后的正式 Stage-1 实验：

- 工作目录：`work_dirs/dsqe-residual-32-baseline56-b2/`
- 训练日志：`logs/dsqe-residual-32-baseline56-b2.log`
- Work-dir 日志：`work_dirs/dsqe-residual-32-baseline56-b2/20260903_035437.log`
- BaseLine 初始化 checkpoint：`/data/jxy/projects/ckpts/epoch_56.pth`
- 训练配置：2 GPU，`samples_per_gpu=2`，`optimizer_config.cumulative_iters=4`
- 模型阶段：`dsqe_training_stage='residual_stage1'`
- BaseLine 本地标准：SparseWorld/BaseLine epoch 56

当前文件系统显示本次训练保存到 `epoch_23.pth`，`latest.pth -> epoch_23.pth`。日志最后一条训练记录为 `Epoch [24][1300/4933]`，`logs/dsqe-residual-32-baseline56-b2.exit` 中记录 `exit_code=130`，因此本文按“实验停止于 epoch 24 中途，完整 checkpoint 至 epoch 23”记录。

## 2. 已评估 checkpoint

完整 4219 样本评估已完成：

```text
epoch_7, epoch_9, epoch_11, epoch_13, epoch_19, epoch_21
```

`epoch_15_eval.log` 只运行到 `10/4219`，没有最终指标字典，未纳入结果表。

规划指标由已保存的 `output_data_epoch_*.pkl` 使用仓库脚本重新汇总：

```bash
/data/jxy/projects/env/bin/python tools/evaluate_planning_temporal.py \
  --prediction work_dirs/dsqe-residual-32-baseline56-b2/eval_results/output_data_epoch_9.pkl \
  --label DSQE_residual_epoch_9
```

该步骤只读取已保存预测，不重新运行模型。

## 3. 主指标

只比较 1s、2s、3s，不计算平均列。

| 模型 / checkpoint | mIoU 1s (%) | mIoU 2s (%) | mIoU 3s (%) | IoU 1s (%) | IoU 2s (%) | IoU 3s (%) | L2 1s (m) | L2 2s (m) | L2 3s (m) | plan_obj_box_col 1s (%) | plan_obj_box_col 2s (%) | plan_obj_box_col 3s (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BaseLine / epoch 56 | 14.96 | 13.18 | 11.53 | 23.15 | 22.27 | 21.21 | 0.1596 | 0.1945 | 0.2393 | 0.0948 | 0.1007 | 0.1106 |
| DSQE residual / epoch 7 | 14.58 | 12.99 | 11.44 | 23.12 | 22.26 | 21.16 | 0.1596 | 0.1945 | 0.2393 | 0.0948 | 0.1007 | 0.1106 |
| DSQE residual / epoch 9 | 14.58 | 13.01 | 11.46 | 23.12 | 22.27 | 21.18 | 0.1596 | 0.1945 | 0.2393 | 0.0948 | 0.1007 | 0.1106 |
| DSQE residual / epoch 11 | 14.47 | 12.93 | 11.40 | 23.11 | 22.27 | 21.19 | 0.1596 | 0.1945 | 0.2393 | 0.0948 | 0.1007 | 0.1106 |
| DSQE residual / epoch 13 | 14.53 | 12.94 | 11.40 | 23.11 | 22.27 | 21.18 | 0.1596 | 0.1945 | 0.2393 | 0.0948 | 0.1007 | 0.1106 |
| DSQE residual / epoch 19 | 14.53 | 12.97 | 11.42 | 23.11 | 22.27 | 21.19 | 0.1596 | 0.1945 | 0.2393 | 0.0948 | 0.1007 | 0.1106 |
| DSQE residual / epoch 21 | 14.52 | 12.95 | 11.42 | 23.10 | 22.25 | 21.18 | 0.1596 | 0.1945 | 0.2393 | 0.0948 | 0.1007 | 0.1106 |

## 4. 相对 BaseLine 差值

差值定义为 `DSQE residual - BaseLine`。IoU/mIoU 越高越好，L2/碰撞率越低越好。

| checkpoint | delta mIoU 1s | delta mIoU 2s | delta mIoU 3s | delta IoU 1s | delta IoU 2s | delta IoU 3s |
|---|---:|---:|---:|---:|---:|---:|
| epoch 7 | -0.38 | -0.19 | -0.09 | -0.03 | -0.01 | -0.05 |
| epoch 9 | -0.38 | -0.17 | -0.07 | -0.03 | 0.00 | -0.03 |
| epoch 11 | -0.49 | -0.25 | -0.13 | -0.04 | 0.00 | -0.02 |
| epoch 13 | -0.43 | -0.24 | -0.13 | -0.04 | 0.00 | -0.03 |
| epoch 19 | -0.43 | -0.21 | -0.11 | -0.04 | 0.00 | -0.02 |
| epoch 21 | -0.44 | -0.23 | -0.11 | -0.05 | -0.02 | -0.03 |

规划 L2 和 `plan_obj_box_col` 在已评估 checkpoint 上均与 BaseLine 数值一致；这是 Stage-1 当前保持 BaseLine trajectory 输出的预期表现。

## 5. dynamic/static mIoU

| checkpoint | dynamic mIoU 1s (%) | dynamic mIoU 2s (%) | dynamic mIoU 3s (%) | static mIoU 1s (%) | static mIoU 2s (%) | static mIoU 3s (%) |
|---|---:|---:|---:|---:|---:|---:|
| epoch 7 | 12.50 | 10.19 | 8.32 | 18.01 | 16.96 | 15.55 |
| epoch 9 | 12.48 | 10.20 | 8.36 | 18.03 | 16.99 | 15.57 |
| epoch 11 | 12.26 | 10.01 | 8.19 | 18.02 | 16.99 | 15.60 |
| epoch 13 | 12.37 | 10.06 | 8.22 | 18.02 | 16.98 | 15.57 |
| epoch 19 | 12.38 | 10.10 | 8.25 | 18.02 | 17.00 | 15.59 |
| epoch 21 | 12.36 | 10.07 | 8.26 | 18.01 | 16.99 | 15.58 |

## 6. epoch 9 逐类别 IoU

epoch 9 是当前已评估 checkpoint 中 mIoU 最好的结果之一。

| class | 1s IoU (%) | 2s IoU (%) | 3s IoU (%) |
|---|---:|---:|---:|
| others | 3.81 | 3.65 | 3.49 |
| barrier | 18.68 | 17.24 | 15.34 |
| bicycle | 7.75 | 6.52 | 5.10 |
| bus | 21.19 | 15.27 | 11.59 |
| car | 17.70 | 14.41 | 11.93 |
| construction_vehicle | 11.61 | 11.10 | 10.34 |
| motorcycle | 9.41 | 8.20 | 6.47 |
| pedestrian | 5.76 | 4.36 | 3.23 |
| traffic_cone | 6.48 | 5.20 | 3.97 |
| trailer | 9.25 | 7.81 | 6.68 |
| truck | 17.12 | 13.92 | 11.52 |
| driveable_surface | 31.61 | 30.24 | 28.23 |
| other_flat | 20.38 | 19.31 | 17.59 |
| sidewalk | 19.32 | 18.10 | 16.52 |
| terrain | 18.32 | 17.29 | 15.89 |
| manmade | 12.01 | 11.63 | 10.96 |
| vegetation | 17.47 | 16.89 | 16.02 |

## 7. 训练日志快照

最后一条训练 JSON 记录：

```text
epoch=24
iter=1300
lr=2e-05
loss=15.12019
grad_norm=9.1428
```

关键未来步指标：

| step | loss_cls | loss_pts | loss_role | loss_static | loss_dynamic | loss_ego | dynamic_f1 | gt_dynamic_ratio | pred_dynamic_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fu1 | 0.14953 | 0.39829 | 0.00573 | 0.61909 | 0.08500 | 0.05519 | 0.00626 | 0.00044 | 0.00916 |
| fu3 | 0.16066 | 0.39143 | 0.00262 | 0.26747 | 0.09140 | 0.04713 | 0.00107 | 0.00036 | 0.00504 |
| fu6 | 0.19626 | 0.42048 | 0.01108 | 0.37029 | 0.13496 | 0.07176 | 0.00058 | 0.00031 | 0.17277 |

残差尺度快照：

| step | dynamic abs mean x | dynamic abs mean y | static abs mean x | static abs mean y |
|---|---:|---:|---:|---:|
| fu1 | 0.73062 | 0.65094 | 0.03153 | 0.03334 |
| fu3 | 0.72136 | 0.55927 | 0.02543 | 0.02587 |
| fu6 | 0.84374 | 0.51210 | 0.02785 | 0.02543 |

## 8. 当前判断

在残差建模修正后，1s/2s/3s 的 IoU 已基本贴近 BaseLine；mIoU 仍略低于 BaseLine，但差距显著小于旧 DSQE epoch 32。当前已评估结果中，epoch 9 的综合语义指标最好：

- mIoU：`14.58 / 13.01 / 11.46`
- IoU：`23.12 / 22.27 / 21.18`
- L2：`0.1596 / 0.1945 / 0.2393`
- `plan_obj_box_col`：`0.0948 / 0.1007 / 0.1106`

## 9. 结果文件索引

- 完整训练日志：`work_dirs/dsqe-residual-32-baseline56-b2/20260903_035437.log`
- 训练 JSON 日志：`work_dirs/dsqe-residual-32-baseline56-b2/20260903_035437.log.json`
- 已保存 checkpoint：`work_dirs/dsqe-residual-32-baseline56-b2/epoch_1.pth` 到 `epoch_23.pth`
- 完整评估日志：`work_dirs/dsqe-residual-32-baseline56-b2/eval_results/epoch_{7,9,11,13,19,21}_eval.log`
- 未完成评估日志：`work_dirs/dsqe-residual-32-baseline56-b2/eval_results/epoch_15_eval.log`
- 规划输出：`work_dirs/dsqe-residual-32-baseline56-b2/eval_results/output_data_epoch_{7,9,11,13,19,21}.pkl`
- 规划指标 JSON/CSV：`work_dirs/dsqe-residual-32-baseline56-b2/eval_results/planning_epoch_{7,9,11,13,19,21}.{json,csv}`
