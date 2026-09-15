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

数据管线同时导出 `temporal_adjacent2ego`。Temporal actor boxes/trajectories are
stored by the VAD converter in the current LiDAR frame; the dataset exports
`temporal_agent_lidar2ego`, and the model converts centers, vectors, yaw and
footprints to E0 ego before applying the E0->Et warp. The
`temporal_agent_boxes` field is VAD `gt_boxes`; its dimensions come directly
from nuScenes `Box.wlh`. SECOND yaw 的局部 x/y 轴与该 ``(width,length)`` 顺序配对，
footprint matching 因而保持这一成对约定，不单独交换尺寸轴。VAD yaw deltas
are scalar raw adjacent increments and are converted to SECOND yaw with the
appropriate sign. Occupancy voxels and Query points stay in the same future
ego frame; inference never reads future GT pose.

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
nearest `local_k` 邻域，并加入连续 role log-gate 与距离 bias；参数化保证
`lambda_SD < lambda_DS`。

`DSQEJointRefine` 先在融合后的统一 Query bank 上执行共享的局部空间 self-attention
（位置编码和 nearest-k 均来自 evolved points 的中心），再展开到每个 Query 的 48 个点
输出 point-wise 坐标与角色纠错；不会把 48 点压缩成单点后重建。四路双流交互复用同一份
nearest-k 拓扑，避免六步 rollout 重复构造四张全局距离矩阵。

## 5. 监督与训练阶段

保留 BaseLine occupancy/point loss，并加入：

```text
L = L_occ + λ_role L_role + λ_ego L_ego
    + λ_static L_static + λ_dynamic L_dynamic + λ_smooth L_smooth
    + λ_leak L_leak
```

- `L_role`：按有效 GT 动/静比例自适应加权的 Focal BCE，记录 precision/recall/F1 和
  dynamic valid ratio；
- `L_ego`：相邻 ego 平移/偏航监督；
- `L_static`：只对 carried-static Query 约束上一状态经 GT ego warp 后的时序一致性，
  无历史的 new Query 不参与；
- `L_dynamic`：GT-confirmed dynamic prediction → dynamic GT 与 dynamic GT →
  nearest prediction 的双向 coverage，采用 metric-space SmoothL1/Huber
  （`dynamic_huber_beta=0.2`）；正向不吸引静态预测点，反向不依赖预测角色，且全程保留点梯度；
- `L_smooth`：相邻时间 Query motion 增量连续性，并用 `motion_valid` 排除 new Query
  首次激活时的人为零运动；
- `L_leak`：对 GT-static Query 抑制动态 Query-level motion 泄漏；dynamic semantic
  auxiliary BCE 直接读取未来绝对语义头的 dynamic logit。

Stage 1（epoch 0–4）冻结 Backbone、RAP/TASS 及 BaseLine SCF/Planning heads，训练
role、pose、双流演化、dual interaction、joint refine 和 absolute semantic head，并保持
单步预测。interaction/joint correction 从 `stage_gate_floor=0.1` 的正值开始执行，
不会用精确 0 的 gate 切断预期模块梯度，同时仍接近 BaseLine 恒等状态。这里不复用
BaseLine 的 `pretrain=True`，因此 future OCC 和 absolute semantic head 从第一个 epoch
就有梯度。Stage 2 从 epoch 5 开始按显式 `forecast_curriculum=[1,2,3,6]` 增加预测步数，
role/pose teacher forcing 使用独立比例并线性衰减到预测闭环。Stage 3 可通过
`stage3_start_epoch` 解冻 TASS decoder 最后若干层，
其 optimizer learning rate 为主学习率的 0.1 倍。为兼容 DDP 的一次性 reducer，待解冻层
从初始化开始就加入 DDP/optimizer；Stage 1/2 通过梯度门置零且关闭该参数组 weight decay，
Stage 3 再打开梯度门，因此冻结期参数不会被 AdamW 暗中更新。
interaction 与 joint feature path 的正门控在阶段边界平滑升到完整输出，避免随机新分支
造成递归状态突变。`DSQEDualEvolution` 的 motion head 从 ego warp 前的 `P_t` 做
Query-level pooling；点级动态修正默认输出完整 xyz，只有显式配置
`planar_motion_only=True` 才会约束 z 分量。

Planning 与 ego pose 不再使用两套平移头。pose head 递归输出相邻 ego 变换，随后通过
LiDAR 外参将累计 `T(E_t→E_0)` 共轭为 `T(L_t→L_0)`；相邻 LiDAR 原点之差作为 VAD
planning 输出。这样共享同一个平移状态，同时不会把 ego-frame xy 直接与 E0-LiDAR
坐标下的 planning GT 错配。
absolute semantic head 在加载 BaseLine checkpoint 时从对应分类头 warm-start。它逐层复制
BaseLine 的 `Linear→ReLU→Linear→ReLU→Linear(48×17)`，不平均最后一层；坐标 adapter
的最终投影零初始化，保持初始函数等价但仍可在训练中获得梯度。

## 6. 运行入口与限制

```bash
# Stage 0：固定 epoch-56 BaseLine，只评估、不继续训练
/data/jxy/projects/env/bin/python3.9 tools/test.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-baseline.py \
  --checkpoint /data/jxy/projects/ckpts/epoch_56.pth \
  --eval segm

# DSQE-PreSCF（默认）
bash tools/train_dsqe_project.sh
```

验证命令：

```bash
/data/jxy/projects/env/bin/python3.9 -m py_compile \
  mmdet3d/models/sparsedetectors/*.py
/data/jxy/projects/env/bin/python3.9 -m pytest -q \
  tests/test_models/test_dsqe_prescf_contract.py

# 两张 GPU、默认 200 iteration 的真实 forward_train DDP smoke（合成 batch，非 surrogate loss）
/data/jxy/projects/env/bin/torchrun --nproc_per_node=2 \
  tools/test_prescf_ddp_smoke.py

# 分别检查 Stage 2 和 Stage 3 的 teacher-forcing/解冻路径
/data/jxy/projects/env/bin/torchrun --nproc_per_node=2 \
  tools/test_prescf_ddp_smoke.py --stage 2
/data/jxy/projects/env/bin/torchrun --nproc_per_node=2 \
  tools/test_prescf_ddp_smoke.py --stage 3
```

测试套件包含真实 PyTorch/CUDA 六步 Query-state rollout、递归恒等与各 PreSCF 模块
反向梯度检查；没有 CUDA 时该项会明确 skip。完整数据集单卡训练 smoke、双卡 200 iter
DDP、显存上限和正式指标对比仍须在训练数据完整的运行环境执行。
