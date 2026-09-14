# SparseWorld BaseLine 与 DSQE-PreSCF

## 1. 建模边界

仓库保留两条可以独立复现的路径：

- `sparseworld-traj-baseline.py`：纯 SparseWorld/BaseLine SCF，对照路径；
- `sparseworld-traj-prescf.py`：DSQE-PreSCF，直接替换未来 Query 的单流递推。

旧的“BaseLine 未来结果 + residual”方案已经废弃。PreSCF 不维护一个继续递归的
`baseline_state`，也不使用 `base_points + residual` 或跨时间累加语义 logits。

## 2. 输入与坐标约定

模型使用 6 路相机、`num_frames=5` 历史上下文和 6 个未来时间槽；最终 Query 点为
`[B, N, 48, 3]`，绝对语义为 `[B, N, 48, 17]`。

数据集中的 `temporal2ego[k]` 是累计变换：

```text
A_k = T(E_k -> E_0)
```

PreSCF 每一步使用相邻变换：

```text
B_0 = A_1
B_k = inv(A_k) @ A_(k+1)
```

数据管线同时导出 `temporal_adjacent2ego`。训练时 actor box、future trajectory、
occupancy voxel 和 Query 点都被转换到对应 future ego frame。Temporal actor boxes keep
native nuScenes ``(w,l,h)`` sizes and are explicitly converted to the ego footprint
ordering ``(l,w,h)`` before yaw-aware matching；推理阶段不读取未来 GT pose。

## 3. PreSCF Query 状态

每个时间步维护同一个 Query 的两种软视图：

```text
E_t       [B, N_t, C]       Query feature
P_t       [B, N_t, 48, 3]   metric/encoded Query points
Z_t       [B, N_t, 48, 17]  当前时刻绝对语义 logits
r_t       [B, N_t, 48, 1]   点级 dynamic probability
rho_t     [B, N_t, 1]       Query 级 dynamic probability
h_t       [B, N_t, 1]       carried/new 来源标识
```

每轮严格执行：

```text
当前状态
 -> carried/new 组织
 -> 软角色路由
 -> ego pose 预测与坐标对齐
 -> dynamic/static 点演化
 -> local-k 双流交互
 -> joint correction
 -> 绝对语义头重新分类
 -> 下一轮状态
```

`carried` Query 使用上一时刻状态；`new` Query 从 RAP/TASS 的 E0 点经过累计 ego
变换进入当前 frame，不能复制上一时刻角色。

## 4. 角色、演化和交互

动态类别为 `[2,3,4,5,6,7,9,10]`，静态类别为
`[1,8,11,12,13,14,15,16]`。角色路由全程使用 sigmoid 概率，不使用 `argmax` 或
`moving_mask`。训练数据提供变长 actor boxes、future trajectories、validity mask 和
occupancy label；box footprint 支持适度膨胀，`others` 不作为动态候选。

`DSQEDualEvolution` 对 48 个点逐点继承和更新：动态流包含 Query 级整体运动与点级
局部修正，静态流执行 ego warp 后的小幅修正，最后按 `rho_t` 软融合。最终点会真实
改变，`query_motion` 不再只是 diagnostics。

`DSQEDualInteraction` 对 D←D、S←S、D←S、S←D 均使用基于 evolved metric points 的
nearest `local_k` 邻域，且 `lambda_SD < lambda_DS`。

## 5. 监督与训练阶段

保留 BaseLine occupancy/point loss，并加入：

```text
L = L_occ + λ_role L_role + λ_ego L_ego
    + λ_static L_static + λ_dynamic L_dynamic + λ_smooth L_smooth
```

- `L_role`：按有效 GT 动/静比例自适应加权的 Focal BCE，记录 precision/recall/F1 和
  dynamic valid ratio；
- `L_ego`：相邻 ego 平移/偏航监督；
- `L_static`：future ego frame 中静态 voxel 的双向 coverage；
- `L_dynamic`：动态 voxel 与预测点的双向 nearest-neighbor coverage，保留梯度；
- `L_smooth`：相邻时间 Query motion 增量连续性，不把合法运动拉回零位移。

Stage 1（epoch 0–4）冻结 Backbone、RAP/TASS 及 BaseLine SCF/Planning heads，只训练
PreSCF 模块并保持单步预测；这里不复用 BaseLine 的 `pretrain=True`，因此 future OCC
和 absolute semantic head 从第一个 epoch 就有梯度。Stage 2 从 epoch 5 开始增加预测步数，
role/pose teacher forcing 同步线性衰减到预测闭环。Stage 3 可通过
`stage3_start_epoch` 解冻 TASS decoder 最后若干层，
其 optimizer learning rate 为主学习率的 0.1 倍。
absolute semantic head 在加载 BaseLine checkpoint 时从对应分类头 warm-start。

## 6. 运行入口与限制

```bash
# BaseLine 对照
CONFIG=configs/sparseworld/nuscenes-temporal/sparseworld-traj-baseline.py \
  bash tools/train_dsqe_project.sh

# DSQE-PreSCF（默认）
bash tools/train_dsqe_project.sh
```

验证命令：

```bash
/data/jxy/projects/env/bin/python3.9 -m py_compile \
  mmdet3d/models/sparsedetectors/*.py
/data/jxy/projects/env/bin/python3.9 -m pytest -q \
  tests/test_models/test_dsqe_prescf_contract.py

# 两张 GPU、默认 200 iteration 的无数据集 DDP smoke
/data/jxy/projects/env/bin/torchrun --nproc_per_node=2 \
  tools/test_prescf_ddp_smoke.py
```

测试套件包含真实 PyTorch/CUDA 六步 Query-state rollout、递归恒等与各 PreSCF 模块
反向梯度检查；没有 CUDA 时该项会明确 skip。完整数据集单卡训练 smoke、双卡 200 iter
DDP、显存上限和正式指标对比仍须在训练数据完整的运行环境执行。
