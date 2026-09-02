# DSQE 实验文档

本目录记录当前 DSQE-SCF 实验的建模配置、训练过程、评估口径、完整结果和与本地复现 BaseLine 的对比。

## 文档结构

- [实验建模](01_experiment_modeling.md)：模型结构、输入输出、损失和训练配置。
- [评估协议](02_evaluation_protocol.md)：数据来源、指标定义、样本范围和口径说明。
- [实验结果](03_experiment_results.md)：BaseLine 与 DSQE epoch 21/25/30/32 的完整结果和差值表。
- [分析与建议](04_analysis_and_recommendations.md)：结果解读、checkpoint 选择建议和后续复核项。
- [Oracle 诊断实验](05_oracle_diagnostics.md)：B0/B1/C1/D1 的 500 样本对照结果、逐类别 IoU、规划指标和诊断结论。
- [DSQE 残差建模修正](06_dsqe_residual_modeling_repair.md)：BaseLine carrier、DSQE 残差路径、运动状态角色 GT、TASS 固定同步和 Stage 1 冻结边界。
- [语义结果 CSV](results_summary.csv)：只保留 1s/2s/3s 的语义占用结果。
- [规划结果 CSV](planning_results.csv)：只保留 1s/2s/3s 的 L2 和碰撞率结果。

## 当前结论摘要

- 语义占用指标在 DSQE 检查点中以 epoch 30 最好；epoch 32 基本持平，但 1s/2s 的 IoU 和 mIoU 略低。
- 规划 L2 以 epoch 21 最好，规划碰撞率以 epoch 25 最好，因此语义占用、轨迹误差和碰撞率的最优 checkpoint 不一致。
- 在 1s、2s、3s 三个预测时域上，DSQE 的 mIoU 和 IoU 仍低于本地复现 BaseLine；延长到 epoch 32 没有缩小这一差距。
- 本报告严格按要求只展示 `1s / 2s / 3s`，不展示平均列。原始评估日志中的 0s 当前帧指标仍保留在机器可读的 `summary.csv/json` 中。

## 主要实验资产

- 训练目录：`work_dirs/dsqe-ddp-32-baseline56-b2/`
- 检查点：当前目录实际保留 `epoch_21.pth` 至 `epoch_32.pth`，另有 `latest.pth`
- 语义评估汇总：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/summary.csv`
- 语义评估 JSON：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/summary.json`
- 规划评估：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/planning_epoch_*.{json,csv,txt}`
- 训练日志：`logs/dsqe-ddp-32-baseline56-b2.log`、`logs/dsqe-resume-epoch25-to-32.log`
