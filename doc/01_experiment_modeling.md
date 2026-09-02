# 实验建模

## 1. 实验目标

本实验在本地复现的 BaseLine 权重基础上加入 DSQE-SCF 时序演化模块。该模块在项目中以动态/静态查询角色、场景坐标变换和未来查询演化为核心，联合优化：

1. 多视角图像到 BEV/体素语义占用的预测；
2. 未来查询点的时序位置和类别语义演化；
3. 动态/静态角色路由；
4. 自车未来位姿和 1s 至 3s 轨迹预测。

配置文件为 `configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py`，模型类型为 `SparseWorld4DTraj`。

## 2. 输入与时序表示

- 数据集：nuScenes occupancy 4D trajectory 数据集。
- 输入模态：仅使用 6 路相机，不使用 LiDAR、Radar、地图或外部特征。
- 图像输入：6 个相机视角，训练管线读取当前帧及历史 sweep；图像最终增强尺寸为 `256 x 704`。
- 时序上下文：`num_frames=5`，即当前帧加 4 个历史时序输入。
- 未来预测：`num_fu_frames=6`，每个时间步对应约 0.5s，评估时按 2/4/6 个点汇总为 1s/2s/3s。
- 3D 体素范围：`[-40, -40, -1, 40, 40, 5.4]`，体素尺寸为 `[0.4, 0.4, 0.4]`。
- 占用类别：17 个非空语义类别，另有 empty/free 类用于输出和评估流程。

训练和测试管线同时携带 `temporal_semantics`、`temporal_trajs`、`temporal_ego_states`、`temporal_ego2global` 等时序监督信息。

## 3. 主干与 OPUS 查询解码器

- 图像主干：ResNet-50，冻结第一个 stage，启用 checkpoint 以降低显存。
- 特征颈部：FPN，4 个输出层，统一通道数 `embed_dims=256`。
- OPUS 查询：当前查询数 `num_query=720`。
- 未来查询：`[60, 60, 60, 60, 40, 40]`，对应六个未来时间步。
- Transformer：6 层 OPUSTransformer decoder，8 头注意力，4 个图像特征层，每个采样点使用 4 个采样位置。
- 逐层点细化数：`[1, 4, 16, 24, 32, 48]`。

每个 decoder layer 对查询执行位置编码、4D 多视角采样、特征混合、自注意力和 FFN，然后同时输出语义分类分数和细化后的 3D 查询点。

## 4. DSQE-SCF 结构

DSQE 在 OPUS 当前帧查询的基础上，逐步生成未来查询状态。每个未来步的处理顺序为：

```text
当前/历史查询
    -> 自车位姿预测与坐标系变换
    -> 动态/静态角色路由
    -> 动态/静态双流点演化
    -> 双流局部空间交互
    -> 联合特征细化与角色校正
    -> 位置、语义和未来轨迹输出
```

### 4.1 角色路由

`DSQERoleRouter` 使用查询特征、点坐标、语义 logits 和来源标识构造 query-level / point-level 动态概率。动态类别 ID 为 `[2,3,4,5,6,7,9,10]`，静态类别 ID 为 `[1,8,11,12,13,14,15,16]`。路由结果不是硬切分，而是作为动态流和静态流的连续门控权重。

训练中启用 role/ego teacher forcing，起始 epoch 为 5，结束 epoch 为 17；之后逐步转为模型自身的角色和位姿预测。

### 4.2 双流点演化

`DSQEDualEvolution` 对 carried queries 和 new queries 分开处理：

- 动态流通过 `motion_head` 预测位移，并叠加动态 residual；
- 静态流使用较小的静态 residual，默认 `static_alpha=0.1`，上限 `0.2`；
- 两个流按查询角色和点角色加权融合，`beta=0.7`；
- 新出现的未来查询使用 `new_update_head` 更新；
- 自车位姿变换通过 `DSQEEgoWarp` 统一到 `future_ego` 坐标模式。

### 4.3 双流交互与联合细化

`DSQEDualInteraction` 建立四类局部注意力：动态-动态、静态-静态、动态从静态、静态从动态。局部邻域最多取 `local_k=16` 个查询，使用 8 头空间注意力和可学习温度。初始跨流门控为：

- dynamic-from-static：1.0；
- static-from-dynamic：0.25。

`DSQEJointRefine` 将基础特征、动态特征、静态特征和点中心编码融合，再进行一次局部空间注意力、FFN 和角色校正，得到下一步查询特征。

## 5. 监督与损失

模型同时优化 OPUS 语义/点监督与 DSQE 时序监督。配置中的主要项为：

- OPUS 分类：sigmoid Focal Loss，`gamma=2.0`、`alpha=0.25`、权重 `2.0`；
- OPUS 点回归：Smooth L1，`beta=0.2`、权重 `0.5`；
- DSQE 角色损失：`lambda_role=0.5`；
- DSQE 静态流损失：`lambda_static=0.2`；
- DSQE 动态流损失：`lambda_dynamic=0.5`；
- 自车位姿损失：`lambda_ego=1.0`；
- 时序平滑和跨流泄漏约束：分别为 `0.05` 和 `0.05`；
- 六个未来预测槽位均输出轨迹损失，训练日志字段为 `loss_traj_1s` 至 `loss_traj_6s`；规划评估严格按前 2/4/6 个采样点聚合为 1s/2s/3s，日志字段名不改变这一评估映射。

DSQE 开启时，旧的逐步 `cls_branch` 不参与训练，未来语义由 semantic correction 分支在基础语义上进行残差修正。

## 6. 训练配置与初始化

本次结果对应的运行配置如下：

| 项目 | 设置 |
|---|---|
| 初始化权重 | `/data/jxy/projects/ckpts/epoch_56.pth`，本地 BaseLine |
| GPU | 2 张 RTX 3090 |
| samples/GPU | 2 |
| 梯度累积 | 4 iter |
| 有效 batch size | `2 GPUs x 2 samples/GPU x 4 accumulation = 16` |
| 优化器 | AdamW |
| 学习率 | `1e-4`，由项目训练脚本覆盖配置默认值 |
| weight decay | `1e-2` |
| 学习率策略 | linear warmup + CosineAnnealing |
| 梯度裁剪 | max norm 5 |
| 最大训练轮数 | 32 epochs |
| checkpoint | 运行参数设置为每 epoch 保存、`max_keep_ckpts=-1`；当前目录实际保留 epoch 21–32 和 `latest.pth` |
| 随机种子 | 17 |

训练分两段完成：第一段从 BaseLine 权重训练到 epoch 25，第二段从 `latest.pth` 恢复并继续到 epoch 32。第二段日志记录了 `resumed epoch 25, iter 123325`，最终保存 `epoch_32.pth`。虽然训练日志中出现过更早 epoch 的保存记录，但当前实验目录实际可用的权重文件从 epoch 21 开始。

## 7. 代码与日志依据

- 模型/数据/损失配置：`configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py`
- 项目式训练入口：`tools/train_dsqe_project.sh`
- DSQE 主流程：`mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py`
- 角色路由：`mmdet3d/models/sparsedetectors/dsqe_role_router.py`
- 双流演化：`mmdet3d/models/sparsedetectors/dsqe_dual_evolution.py`
- 双流交互：`mmdet3d/models/sparsedetectors/dsqe_dual_interaction.py`
- 联合细化：`mmdet3d/models/sparsedetectors/dsqe_joint_refine.py`
- 自车坐标变换：`mmdet3d/models/sparsedetectors/dsqe_ego_warp.py`
- 训练日志：`logs/dsqe-ddp-32-baseline56-b2.log` 和 `logs/dsqe-resume-epoch25-to-32.log`
