# DSQE-PreSCF 文档索引

本目录只把当前 `mode='prescf'` 实现视为有效的新模型路径。历史
DSQE-Residual 指标仅保留作实验档案，不能用于宣称 PreSCF 的效果。

- [建模与训练契约](01_experiment_modeling.md)：Query 状态、坐标系、角色路由、
  动静态演化、规划共享状态、损失和分阶段训练。
- [评估协议](02_evaluation_protocol.md)：BaseLine 对照、1s/2s/3s 指标口径、正式训练
  与评估命令，以及历史 residual 结果的隔离说明。
- [规划结果表](planning_results.csv)：历史实验记录；使用前须按评估协议确认模型类型
  和 checkpoint，不能默认视作 PreSCF 结果。

当前源码验收只覆盖短时静态检查、单元测试以及合成六步 forward/backward。正式结论还
必须由真实数据单卡 smoke、双卡 DDP 200 iterations 和完整 validation 产生。
