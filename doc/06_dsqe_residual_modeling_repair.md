# DSQE 残差建模修正

## 1. 修正范围

本次修改针对 DSQE epoch 32 诊断中发现的结构问题，不修改 SparseWorld BaseLine 的内部算法，不重新训练，也不运行 4219 样本正式评估。目标是把 DSQE 从“独立替换未来预测”改为：

```text
SparseWorld BaseLine 完整未来预测
        +
DSQE 动态/静态残差修正
```

## 2. 修改前问题

旧路径由 DSQE 重新生成 `next_feat`、`next_points` 和 `next_semantics`，因此 BaseLine 的未来 carrier 被 DSQE 状态覆盖。位置还同时叠加了 BaseLine `vel_branch`、DSQE `query_motion`、动态 residual、`reg_branch` correction；语义则把上一时刻最终 logits 带入下一时刻，形成递归累加。

## 3. 修改后数据流

每个未来时刻先执行 `_baseline_future_step`。该函数抽取并复用原 `_forward_baseline_scf` 的单步逻辑：新 Query 激活、timestamp/position embedding、ego cross attention、trajectory head、`reg_branch`、`cls_branch`、`vel_branch`、原有 moving mask 和点细化。

随后 DSQE 只读取该时刻 BaseLine 结果：

```python
base_next_feat       = baseline_step['feat']
base_next_points     = baseline_step['points']
base_next_semantics  = baseline_step['semantics']

next_feat      = base_next_feat + delta_feat
next_points    = base_next_points + delta_points
next_semantics = base_next_semantics + delta_semantics
```

BaseLine carrier 使用独立的 `baseline_feat/baseline_points/baseline_timestamp` 链；DSQE 的残差状态使用 `state_feat/state_points/state_role` 链。语义残差不会作为下一时刻 BaseLine 的语义输入，DSQE 修正后的结果也不会再作为下一时刻 BaseLine 的输入。

## 4. 逐时刻绝对语义

未来语义现在严格按当前时刻计算：

\[
Z_t = Z_t^{base} + \Delta Z_t^{DSQE}
\]

`cls_branch` 只接收 `_baseline_future_step` 产生的 `base_next_feat`，不会接收 DSQE 修正后的特征。禁止 `state_semantics = previous_next_semantics` 形式的跨时刻 logits 累加。语义 residual head 的最终投影零初始化；没有额外零门控，因此 residual 投影本身仍可获得梯度。

## 5. 运动状态角色 GT

角色目标不再由语义类别直接决定。数据管线现在将 `temporal_agent_boxes` 和 `temporal_agent_feats` 同时提供给训练和测试路径。模型使用当前 actor box、速度和未来轨迹属性，先用 GT ego 相对变换补偿自车运动，再计算：

\[
v^{rel}=\frac{\|P_{t+1}^{GT}-Warp(P_t^{GT})\|}{\Delta t},\qquad
r^{GT}=\sigma((v^{rel}-\tau)/T)
\]

数据中的未来轨迹按相邻帧增量解释：第 `k` 个未来框中心使用前 `k` 段位移的累计和，速度使用当前段瞬时速度或累计位移的时间平均，不能把第 `k` 段增量直接当作绝对偏移。actor 中心先经过统一的 GT 自车变换补偿，再用于匹配。对于完整 box 字段，匹配使用带 yaw 的旋转矩形/长方体内部判断；缺少尺寸或朝向字段时才回退到兼容半径。无法匹配的动态点 `role_valid=False`。明确静态类点角色为 0 并保持有效。没有 actor 元数据时，动态类别不会伪造为有效动态 GT。

`temporal_agent_boxes` 和 `temporal_agent_feats` 在格式化阶段使用
`DataContainer(stack=False)`，模型侧通过 ragged per-sample sequence 消费，因此每卡
`samples_per_gpu=2` 时不同样本的 actor 数量不会触发 `torch.stack` 形状错误。

角色日志扩展为 `dynamic_precision`、`dynamic_recall`、`dynamic_f1`、`gt_dynamic_ratio`、`pred_dynamic_ratio`、`mean_role_on_moving_gt`、`mean_role_on_static_gt`、`role_saturation_low/high`，并保留原 `role_accuracy`、`dynamic_ratio` 和交互门控值。
普通推理只使用预测 pose；未来 GT pose/occupancy 仅在训练或显式
`eval_oracle_pose` / `eval_oracle_role` 诊断开关打开时读取。

## 6. 单一位置残差

`DSQEDualEvolution` 在 residual 模式下只返回相对于 `base_next_points` 的两套点残差：

\[
\Delta P_{ik}=r_{ik}\Delta P^{dyn}_{ik}+(1-r_{ik})\Delta P^{sta}_{ik}
\]

动态分支的 z 分量固定为 0；静态分支乘以小的 `static_point_delta_scale`。点级 `point_role` 是最终位置混合权重，Query 级角色仅参与上下文、双流交互和角色一致性损失。旧的 `query_motion` 不再作为完整位移叠加，`vel_branch` 只在 BaseLine carrier 单步中执行一次。

动态几何损失同时保留预测→GT项和真正的 GT→预测覆盖项。后者对每个有效动态 GT
点寻找最近预测点；两项均使用带下限的连续角色权重，不依赖 `pred_role > 0.5`，并以
GT角色质量归一化，确保低角色概率时仍有稳定几何梯度。

每个未来步还记录动态/静态 residual 的 xyz 均值、绝对均值和 95% 分位数。
同时记录 `query_motion` 的 xyz 均值/P95，以及基于有效运动状态标签的
`static_warp_error_{x,y,z}` 和 `dynamic_displacement_error_{x,y,z}`，用于在短
验证集上区分自车变换误差与真实动态残差误差。

## 7. Stage 1 冻结边界

默认配置新增：

```python
dsqe_training_stage='residual_stage1'
freeze_baseline=True
freeze_tass=True
planning_gradient_to_dsqe=False
```

Stage 1 冻结图像 backbone/FPN、OPUSHead/BaseLine transformer、BaseLine SCF 的 ego/position/reg/vel/cls/traj 模块；只训练 DSQE role router、双流演化/交互、joint refine、feature residual 和 semantic residual。规划输出使用 BaseLine trajectory，规划损失在 `planning_gradient_to_dsqe=False` 时从预测轨迹 detach，避免规划目标反向改变 DSQE 占用主路径。初始化时记录总参数量、可训练参数量和按顶层模块分组的冻结/可训练数量。

DSQE 开启时不再使用 `finetune_epoch` 重新进入 BaseLine pretrain 流程；未来步数由独立的 `forecast_curriculum_enabled` / `forecast_curriculum_start_epoch` 控制，默认直接执行全部未来时刻。

## 8. TASS 固定与多卡同步

DSQE Stage 1 不再在 `set_epoch()` 中执行 `num_stamps_all[:] = 1`，也不按 epoch 重新计算 `ind_stamps_all`。从 checkpoint 恢复后，首次需要时根据已有 `num_stamps_all` 生成一次固定分配，槽位数量仍为 `[720, 60, 60, 60, 60, 40, 40]`。后续 epoch 只复用该分配。

分布式运行时从 rank 0 广播 `num_stamps_all` 和 `ind_stamps_all`，广播前比较本地副本；任一 rank 的 checksum 或槽位内容不同都会立即抛出异常。单卡、未初始化 distributed 和 CPU 路径仍可执行。DSQE 关闭时保留原 BaseLine 的 TASS 课程行为。

## 9. 已执行测试

使用项目环境 `/data/jxy/projects/env/bin/python` 执行：

```bash
/data/jxy/projects/env/bin/python -m pytest -q tests/test_models
```

结果：`35 passed`。新增测试覆盖相邻帧 actor 位移累计、旋转 box footprint、ragged actor batch=2 的 mmcv collate、GT→预测覆盖、TASS rank divergence 检测和真实模型的 Baseline-only state dict 兼容；同时通过已有 DSQE 两步 CUDA smoke test（已传入 moving/stationary actor 字段），以及此前已有的 BaseLine 残差恒等性、零初始化梯度、逐时刻语义独立性、运动状态角色匹配、混合 Query 点级路由、动态损失软死区和重复位移防止测试。

另外已执行相关文件 `py_compile` 检查通过。尚未执行完整训练、4219 样本验证、真实多进程 rank divergence 注入测试和需要完整 nuScenes 数据的端到端指标验证。

## 10. 后续短验证方案

代码确认后再进行短实验：固定同一批 500 个验证样本，先运行 200 iteration 的 residual stage-1 smoke test；若 BaseLine 恒等性和残差梯度日志正常，再进行 2～3 epoch 短训练。短实验只检查 1s/2s/3s 的语义指标、L2、碰撞率、角色 F1 和 residual P95，不替代正式训练或 BaseLine 复现。
