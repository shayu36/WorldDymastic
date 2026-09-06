# DSQE 实验文档

本目录记录当前 DSQE-SCF 实验的建模配置、评估口径和实验归档。

## 文档结构

- [实验建模](01_experiment_modeling.md)：模型结构、输入输出、损失和训练配置。
- [评估协议](02_evaluation_protocol.md)：数据来源、指标定义、样本范围和口径说明。
- [实验记录归档](08_experiment_archive.md)：原 `03~07` 实验记录的合并归档，只保留问题、证据和结论。
- [语义结果 CSV](results_summary.csv)：只保留 1s/2s/3s 的语义占用结果。
- [规划结果 CSV](planning_results.csv)：只保留 1s/2s/3s 的 L2 和碰撞率结果。

## 当前结论摘要

- 零残差实验与 BaseLine 完全一致，说明 carrier 主路径正常；问题集中在 DSQE 残差侧，尤其是动态 / motion 相关路径。
- 当前 debug 链路显示 `feature residual` 不是主因；`point-only` 和 `semantic-only` 只能局部改善，不能把未来指标拉回 BaseLine。
- oracle 诊断进一步排除了单纯的 pose / role 单点错误，问题更像未来点 / 语义联合演化链路不稳定。
- Stage-1 残差实验中，epoch 9 是当前评估集合里语义最好的 checkpoint，但仍略低于本地 BaseLine；规划 L2 和碰撞率保持 BaseLine 数值。

## 主要实验资产

- 训练目录：`work_dirs/dsqe-ddp-32-baseline56-b2/`
- 检查点：当前目录实际保留 `epoch_21.pth` 至 `epoch_32.pth`，另有 `latest.pth`
- 语义评估汇总：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/summary.csv`
- 语义评估 JSON：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/summary.json`
- 规划评估：`work_dirs/dsqe-ddp-32-baseline56-b2/eval_results/planning_epoch_*.{json,csv,txt}`
- 训练日志：`logs/dsqe-ddp-32-baseline56-b2.log`、`logs/dsqe-resume-epoch25-to-32.log`
- 残差 Stage-1 训练目录：`work_dirs/dsqe-residual-32-baseline56-b2/`
- 残差 Stage-1 已评估日志：`work_dirs/dsqe-residual-32-baseline56-b2/eval_results/epoch_{7,9,11,13,19,21}_eval.log`
- 残差 Stage-1 结果与诊断：`doc/08_experiment_archive.md`
