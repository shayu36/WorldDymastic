# 分析与建议

## 1. 语义占用表现

DSQE 在 epoch 21 到 epoch 30 之间有小幅改善，但提升主要发生在 1s/2s/3s 的 IoU 和 mIoU 绝对值上，仍没有达到本地复现 BaseLine：

- epoch 30 是 DSQE 语义指标最优点：IoU 为 `21.74 / 20.99 / 20.16`，mIoU 为 `10.32 / 8.64 / 7.51`；
- epoch 32 的 mIoU 与 epoch 30 在 3s 持平，但 IoU 和前两个 horizon 的 mIoU 略降；
- epoch 30 相对 BaseLine 的 3s IoU 差距为 `-1.05` 个百分点，3s mIoU 差距为 `-4.02` 个百分点，为当前 DSQE 检查点中最小差距。

这说明在当前 32 epoch 预算内，继续训练到 32 epoch 没有带来可见的语义占用收益，epoch 30 已经进入平台区间。

## 2. 规划表现

规划指标与语义指标没有同步单调变化：

- epoch 21 的 L2 最低，说明其平均轨迹位置误差最接近 BaseLine；
- epoch 25 的碰撞率最低，并且按指定 BaseLine 标准在三个时域都不高于 BaseLine；
- epoch 30 的碰撞率显著恶化，尽管其语义占用最好；
- epoch 32 的碰撞率较 epoch 30 恢复，但 3s 仍为 `0.1501%`，高于 BaseLine 的 `0.1106%`。

因此不能只根据 mIoU 选择规划 checkpoint，也不能只根据碰撞率推断语义占用质量。当前结果显示 DSQE 的语义演化分支、轨迹分支和碰撞敏感性之间仍存在明显解耦。

## 3. 当前 checkpoint 选择建议

| 使用目标 | 建议 checkpoint | 理由 |
|---|---|---|
| 语义占用 / mIoU 优先 | epoch 30 | DSQE 中 IoU/mIoU 综合最好，epoch 32 未继续提升 |
| 轨迹 L2 优先 | epoch 21 | 1s/2s/3s L2 均为四个 DSQE 检查点中最低 |
| 规划碰撞率优先 | epoch 25 | 三个 horizon 的碰撞率最低，且低于指定 BaseLine 标准 |
| 统一下游模型 | 暂不单独定稿 | 需要按固定验证集重新跑多目标选择，避免单指标选择造成偏差 |

如果当前论文或报告只能选择一个 checkpoint，建议主语义占用结果使用 epoch 30，规划结果单独报告 epoch 25，并明确两者不是同一个最优点。

## 4. 需要保留的评估说明

用户指定的 BaseLine 3s 碰撞率为 `0.1106%`。通用脚本直接复算项目 BaseLine 得到 `0.1067%`，L2 和 1s/2s 碰撞率完全一致。本文按用户指定的 `0.1106%` 作为正式比较标准，同时在评估协议中记录该复算差异。

这项差异不影响当前语义占用结论，但在对外发布规划碰撞率前，建议使用项目原始 ST-P3 评估入口复核一次 3s 统计口径。

## 5. 后续实验建议

1. 固定 epoch 25、30、32 的相同输出和标签集合，使用项目原始 PlanningMetric 再复核碰撞率，尤其是 3s。
2. 对 epoch 25 与 epoch 30 分别做动态/静态类别的碰撞和轨迹误差拆分，确认碰撞率波动是否主要来自动态目标。
3. 如果目标是提升 mIoU，应优先检查 DSQE semantic correction、dynamic/static role routing 和长时域监督权重，而不是继续单纯增加 epoch。
4. 如果目标是规划，应以 `mIoU + L2 + plan_obj_box_col` 的固定加权或 Pareto 规则选 checkpoint，并锁定同一评估脚本。
5. 对外报告时保留 `summary.csv/json` 和各 checkpoint 的 planning JSON，避免只保留手工整理后的表格。
