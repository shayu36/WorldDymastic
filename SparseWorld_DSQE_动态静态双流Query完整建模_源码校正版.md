# SparseWorld 动态—静态双流 Query 完整建模（源码校正版）

> 版本定位：重新建模，不沿用此前汇报版流程。  
> Baseline：上传的 **SparseWorld(2).pdf**（SparseWorld）。  
> 重点源码参考：[SparseWorld 官方实现](https://github.com/MSunDYY/SparseWorld) 与 [GaussianWorld 官方实现](https://github.com/zuosc19/GaussianWorld)。  
> 本文将内容明确分为：**Baseline 已有机制**、**从 GaussianWorld 借鉴的原则**、**本文新增的动态—静态双流 Query 建模**，避免把已有代码误写成创新。

---

## 1. 结论先行

在 SparseWorld 上，正确的“双流 Query”不应被实现为两套完全独立、固定数量的 Query bank，也不应把 TASS 的时间分配矩阵从 `N×(F+1)` 粗暴扩成“时间×动态/静态”。推荐模型是：

1. **RAP 与 TASS 保持 Baseline 语义**：RAP 仍学习带时间归属的稀疏 4D Query；TASS 仍只负责 Query 的时间槽分配。
2. **每个样本、每个点预测软动态概率**，再聚合为 Query 级动态概率；不使用类别 `argmax` 形成不可导硬门控。
3. **SCF 内按“来源 + 运动属性”更新 Query**：
   - 来源：历史继承 Query / 本时间步新激活 Query；
   - 属性：动态概率 / 静态概率。
4. **先做可解释的运动分解，再做统一纠错**：静态流主要做自车坐标变换，动态流在自车变换上叠加局部运动；随后用共享的 Query 交互层纠正角色、遮挡和位姿误差。
5. **保留每个 Query 的全部 48 个点**，不再像 Baseline `refine_points()` 那样先压成均值点再生成下一帧点集。
6. **规划头分别读取动态流与静态流**：动态流提供碰撞参与体，静态流提供可行驶区域、边界和道路结构。

本文将该模块称为 **DSQE-SCF：Dynamic–Static Query Evolution for Scene Conditioned Forecasting**。

---

## 2. 先还原 Baseline：SparseWorld 实际做了什么

### 2.1 总体任务

给定当前帧及历史多帧多视角图像、历史自车状态，SparseWorld 同时输出：

- 当前语义占用；
- 未来 `F` 帧语义占用；
- 未来自车轨迹。

论文结构由四部分组成：图像 Backbone、RAP、SCF、并行占用/规划解码；TASS 是训练与 Query 时间分配策略，而不是额外的推理网络。

### 2.2 RAP：稀疏 4D Query 感知

设 Query 总数为 `N`，特征维为 `C`，第 `i` 个 Query 在最后一层包含 `K=48` 个 3D 点：

\[
E\in\mathbb{R}^{B\times N\times C},\qquad
P\in\mathbb{R}^{B\times N\times K\times 3},\qquad
Z\in\mathbb{R}^{B\times N\times K\times C_{cls}}.
\]

RAP 的可学习点坐标首先受历史自车运动驱动的 Adaptive Scaling 调整。论文写作：

\[
\gamma=\operatorname{MLP}([w^{-p},\ldots,w^0]),\qquad
p_i'=\gamma\odot p_i.
\]

源码中 `points_scale_branch` 将缩放限制在约 `[0.8,1.5]`。六层 Query 解码器依次执行图像特征采样、Adaptive Mixing、带时空偏置的自注意力、FFN 与点集细化；每层每 Query 的点数为 `(1,4,16,24,32,48)`。

论文的时空自注意力为：

\[
A_{ij}=q_i^\top q_j-\tau_i\lVert p_i-p_j\rVert_2^2+M_{ij},
\]

其中因果掩码：

\[
M_{ij}=
\begin{cases}
0,&t_i\ge t_j,\\
-\infty,&t_i<t_j.
\end{cases}
\]

### 2.3 TASS：只分时间，不分运动角色

官方配置中 `N=1040`，七个时间槽的 Query 数量为：

\[
(N_0,N_1,\ldots,N_6)=(720,60,60,60,60,40,40).
\]

TASS 维护：

\[
M^{stamp}\in\mathbb{N}^{N\times(F+1)},
\]

再依据 Query 与多时间 GT 点的匹配统计，把每个 Query 分到一个时间槽。源码中的 `num_stamps_all` 确实是 `1040×7`，`ind_stamps_all` 是每个 Query 唯一的时间标签。

必须注意：源码更新 TASS 统计时使用

```python
fore_mask = (label < 2) | (label > 10)
```

也就是类别 `2...10` 的匹配点不进入时间统计。这个历史实现细节可以在后续做消融，但第一版双流模型应先保持它不变，避免同时破坏 Query 时间容量分配和未来解码。

### 2.4 SCF：Baseline 的递归未来预测

TASS 将 RAP 输出切分成：

\[
Q^0,Q^1,\ldots,Q^F.
\]

在第 `t` 次递归中，活动场景集合由已累积 Query 与下一时间槽的新 Query 拼接：

\[
\mathcal Q_{act}^{t+1}=[\widehat{\mathcal Q}^{t},\mathcal Q^{t+1}].
\]

自车 Query 先对活动场景 Query 做空间交叉注意力，预测下一轨迹点；活动场景特征再叠加自车特征与时空位置编码，输出类别、通用点偏移和额外速度偏移。

### 2.5 Baseline 已经存在“动态类速度分支”，但它不是双流 Query

`sparseworld_4d_traj.py` 的核心逻辑是：

```python
pred_labels = cls_score.argmax(-1)
pred_moving_mask = (pred_labels >= 2) & (pred_labels <= 10)
reg_xy = reg_xy + vel_offset * pred_moving_mask
```

因此不能声称 Baseline 完全没有动态建模。它已有**点级、类别硬掩码的额外速度修正**。但仍存在五个关键问题：

1. `argmax` 后的硬掩码不可导，运动分支不能反向改善角色判断；
2. 类别区间 `2...10` 把 `traffic_cone` 也视作 moving；
3. 同一个 Query 的 48 个点可以属于不同语义，但没有 Query 级一致运动；
4. 动态点和静态点共享同一特征演化与通用位置回归；
5. `refine_points()` 先对上一帧 48 点取均值，再用新 offset 生成点集，破坏了已学习的局部几何形状。

所以本文的新增点不是“再加一个 velocity head”，而是把 SCF 改成**可导、来源感知、运动分解、双流交互、统一纠错**的 Query 演化模型。

---

## 3. GaussianWorld 应该借鉴什么，不应该照搬什么

### 3.1 可借鉴的源码原则

GaussianWorld 流式代码中包含三个重要思想：

1. `warp_anchor()` 使用相邻帧位姿把历史 Gaussian anchor 对齐到当前帧；
2. 历史 anchor 与新补充 anchor 使用不同更新规则；
3. 第一次 refinement 对历史 anchor 施加语义动态门控，后续 refinement 再允许更充分的统一特征/属性修正。

这对应一种稳定的顺序：

\[
\text{确定性/可解释演化}\rightarrow\text{学习式统一纠错}.
\]

### 3.2 不能原样复制的部分

GaussianWorld 是“过去帧到当前帧”的流式感知：历史位姿已知；SparseWorld 是“当前帧到未来帧”的预测：未来自车位姿必须预测或由轨迹推导。因此：

- 训练时可用 GT 位姿做辅助监督或早期 teacher forcing；
- 推理主路径必须使用预测位姿，不能偷用未来 GT；
- SparseWorld 的 TASS 新激活 Query 不等同于 GaussianWorld 的几何填充 anchor；它携带由多帧图像和 4D 预训练学到的未来先验。

GaussianWorld 也不是两套完全隔离的 Gaussian bank；其核心是“旧/新来源标记 + 动态语义门控 + 后续共享 refinement”。本文沿用这个原则，而不是复制数据结构。

---

## 4. DSQE-SCF 的完整变量定义

对未来递归步 `t→t+1`，定义：

| 符号 | 形状 | 含义 |
|---|---:|---|
| `E^t` | `B×N_t×C` | 当前活动 Query 特征 |
| `P^t` | `B×N_t×K×3` | 当前活动 Query 的 K 点几何 |
| `Z^t` | `B×N_t×K×C_cls` | 点级语义 logits |
| `r^t` | `B×N_t×K×1` | 点级动态软概率 |
| `ρ^t` | `B×N_t×1` | Query 级动态软概率 |
| `h^t` | `B×N_t×1` | 来源标记；继承 Query 为 1，新激活 Query 为 0 |
| `ξ^t` | `B×4` | 自车相对运动：`Δx,Δy,sinΔψ,cosΔψ` |
| `m^t` | `B×N_t×3` | 动态 Query 的局部刚体平移 |
| `δP_D,δP_S` | `B×N_t×K×3` | 动态/静态点级残差 |

语义类别集合建议按 NuScenes 配置明确写出，而不是继续使用连续编号：

\[
\mathcal C_D=\{\text{bicycle,bus,car,construction\_vehicle,motorcycle,pedestrian,trailer,truck}\},
\]

\[
\mathcal C_S=\{\text{barrier,traffic\_cone,driveable\_surface,other\_flat,sidewalk,terrain,manmade,vegetation}\}.
\]

`others` 作为不确定/共享类，不强制给动态或静态标签。这样可明确修正 Baseline 将 `traffic_cone` 纳入 `2...10` 运动区间的问题。

---

## 5. 第一步：软动态—静态角色估计

### 5.1 点级角色：语义先验 + 可学习残差

先从点级语义概率构造动态先验。官方分类头/`get_occ()` 使用逐类 sigmoid，因此先把非空类别分数归一化，再求动态类别质量；若后续把分类头改为单标签 softmax，则直接使用 softmax 概率即可：

\[
p_{ikc}^{t}=\frac{\sigma(Z_{ikc}^{t})}
{\sum_{c'=1}^{C_{cls}}\sigma(Z_{ikc'}^{t})+\epsilon},
\qquad
s_{ik}^{D,t}=\sum_{c\in\mathcal C_D}p_{ikc}^{t}.
\]

角色头读取 Query 特征、点位置编码、语义 logits 和来源嵌入：

\[
\Delta a_{ik}^t=\operatorname{MLP}_{role}
([E_i^t,\operatorname{PE}(P_{ik}^t),Z_{ik}^t,e(h_i^t)]).
\]

最终点级动态概率：

\[
r_{ik}^t=\sigma\left(\operatorname{logit}(\operatorname{clip}(s_{ik}^{D,t}))+\Delta a_{ik}^t\right).
\]

这不是重复做语义分类：语义概率给出类别先验，残差项可以利用运动上下文纠正语义不确定或类别—运动不完全等价的问题。

### 5.2 Query 级角色：保留混合点，但保证主体运动一致

每个 Query 内的点权重：

\[
\alpha_{ik}^t=\operatorname{softmax}_k
(\operatorname{MLP}_{pool}([E_i^t,\operatorname{PE}(P_{ik}^t),Z_{ik}^t])).
\]

Query 级动态概率：

\[
\rho_i^t=\sum_{k=1}^{K}\alpha_{ik}^t r_{ik}^t.
\]

`ρ_i` 用于 Query 主体的整体局部运动和跨 Query 注意力路由；`r_ik` 用于点级语义/残差融合。这样既允许一个 Query 覆盖物体边缘与背景，又能避免 48 个点完全独立漂移。

### 5.3 角色监督不需要实例轨迹或光流标注

SparseWorld 原有 KNN/Chamfer 匹配已经为每个预测点产生最近 GT 标签 `y_ik`。直接得到：

\[
y_{ik}^{D}=\mathbb{1}[y_{ik}\in\mathcal C_D],
\qquad
\bar y_i^D=\frac{\sum_k\alpha_{ik}y_{ik}^D}{\sum_k\alpha_{ik}}.
\]

`others` 匹配点从角色损失中忽略。无需引入新的人工标注。

---

## 6. 第二步：来源感知的 Query 激活

第 `t+1` 步活动集合：

\[
\mathcal Q_{act}^{t+1}
=\underbrace{\widehat{\mathcal Q}^{t}}_{h=1,\text{carried}}
\cup
\underbrace{\mathcal Q^{t+1}}_{h=0,\text{newly activated}}.
\]

两类 Query 的差异是：

- **继承 Query**已有上一时刻几何与语义，应以演化为主，避免无约束重建；
- **新激活 Query**只有 RAP/TASS 提供的时空先验，应允许几何、语义、特征完整更新，用于新显现区域与未来新增结构。

来源嵌入 `e_carried/e_new` 加到特征中：

\[
\widetilde E_i^t=E_i^t+e(h_i^t)+\operatorname{PE}(t_i).
\]

这一步对应 GaussianWorld 对 historical anchor 与 fill anchor 的区分，但仍保留 SparseWorld TASS 的未来 Query 先验。

---

## 7. 第三步：自车运动与局部动态运动分解

### 7.1 先明确坐标系，禁止重复 warp

官方训练流程中，TASS 预训练会通过 `temporal2ego` 把多时刻稀疏点对齐到当前参考帧；而 `loss_future()` 又直接把每个未来预测与相应未来 voxel grid 比较。因此实现时必须设置一个明确的 `frame_mode`：

- `future_ego`：未来监督位于各自未来自车坐标系。每步需要把上一状态变换到下一自车坐标系；这是官方未来解码最合理的建模方式，也是本文默认设置。
- `t0_aligned`：若数据管线已把所有未来监督统一对齐到 `t=0`，则几何 warp 设为恒等映射，只保留自车运动用于规划和特征条件化。

在 dataloader 与 `loss_future()` 前打印一组静态 voxel 的坐标变化即可完成一次性验证。绝对不能一边在数据预处理对齐，一边在模型内再次 warp。

### 7.2 预测未来自车相对位姿

原规划头只输出 `xy` 轨迹。增加 yaw 分支，并将平移与轨迹头共享，得到：

\[
\hat\xi^t=(\Delta\hat x^t,\Delta\hat y^t,
\sin\Delta\hat\psi^t,\cos\Delta\hat\psi^t).
\]

若 `T_{E_t\to G}` 表示从时刻 `t` 自车坐标到全局坐标，则从 `t` 坐标表达转换到 `t+1` 坐标表达的正确矩阵为：

\[
T_{t\rightarrow t+1}=
(T_{E_{t+1}\to G})^{-1}T_{E_t\to G}.
\]

训练时由数据中的自车位姿计算 GT 相对变换，监督 `ξ`；主预测路径使用 `\hat ξ`。早期训练可按衰减概率用 GT 变换 teacher forcing，最终必须完全切到预测变换。

对所有继承点先做：

\[
\bar P_{ik}^{t+1}=\mathcal W(P_{ik}^{t},\hat T_{t\rightarrow t+1}).
\]

对于 TASS 新激活 Query，如果其 RAP 几何仍处于 `t=0` 对齐坐标，则用累计预测变换 `\hat T_{0\rightarrow t+1}` 只转换一次；如果数据/实现已经在激活前完成转换，则不再重复。

### 7.3 静态流：自车变换 + 小残差

静态世界的主要位置变化来自坐标系变化：

\[
P_{ik,S}^{t+1}=\bar P_{ik}^{t+1}+\alpha_S\delta P_{ik,S}^{t},
\qquad 0\le\alpha_S\ll1.
\]

建议 `α_S` 从 `0.05` 或 `0.1` 起步并可学习上限。它用于吸收位姿预测误差、离散 voxel 误差和轻微非刚性边界变化，而不是让静态 Query 自由漂移。

### 7.4 动态流：自车变换 + Query 级刚体运动 + 点级形变

先预测 Query 主体的局部平移：

\[
\Delta m_i^t=\operatorname{MLP}_{motion}
([E_i^t,e_{ego}^{t+1},\operatorname{Pool}(P_i^t),\rho_i^t]).
\]

必要时可再预测对象 yaw；第一版只用 `xy` 平移最稳健。动态点演化为：

\[
P_{ik,D}^{t+1}
=\bar P_{ik}^{t+1}+\Delta m_i^t+\delta P_{ik,D}^{t}.
\]

`Δm_i` 保证同一 Query 的整体运动一致，`δP_ik,D` 只描述形状变化和局部误差。

### 7.5 继承 Query 的软融合与新 Query 的完整更新

继承 Query：

\[
P_{ik,evo}^{t+1}
=(1-\rho_i^t)P_{ik,S}^{t+1}
+\rho_i^tP_{ik,D}^{t+1}.
\]

若希望边界点更细，可使用层级门控：

\[
g_{ik}^t=\beta\rho_i^t+(1-\beta)r_{ik}^t,
\quad \beta\in[0.5,0.8],
\]

再以 `g_ik` 代替 `ρ_i`。

新激活 Query 没有可信的上一状态，采用独立完整更新：

\[
P_{ik,new}^{t+1}=\mathcal W(P_{ik,RAP}^{t+1},\hat T_{0\rightarrow t+1})
+\delta P_{ik,new}^{t}.
\]

最后按来源融合：

\[
P_{ik,src}^{t+1}
=h_iP_{ik,evo}^{t+1}+(1-h_i)P_{ik,new}^{t+1}.
\]

关键实现要求：`P_i^t` 的 48 点逐点保留并变换，不能再调用 Baseline 的“先 `mean(dim=2)` 再生成点”的 `refine_points()`。

---

## 8. 第四步：动态流与静态流的特征交互

### 8.1 不是物理拆成两份 Query，而是软路由两个视图

\[
E_{i,D}^t=\rho_i^t(\widetilde E_i^t+e_D),
\qquad
E_{i,S}^t=(1-\rho_i^t)(\widetilde E_i^t+e_S).
\]

这样 Query 数量仍是 `N_t`，显存不会因复制两套 bank 直接翻倍；不同流的投影层和 FFN 参数可以独立。

### 8.2 角色与空间共同调制的注意力

从流 `b` 到流 `a` 的注意力 logits：

\[
A_{ij}^{a\leftarrow b}
=\frac{(W_q^aE_{i,a})^\top(W_k^bE_{j,b})}{\sqrt d}
-\tau_i^a\lVert c_i-c_j\rVert_2^2
+\log(g_i^a+\epsilon)+\log(g_j^b+\epsilon),
\]

其中：

\[
g_i^D=\rho_i,\qquad g_i^S=1-\rho_i,
\qquad c_i=\frac1K\sum_kP_{ik}.
\]

四种交互的推荐关系：

1. `D←D`：学习交通参与体之间的交互；
2. `S←S`：维持道路、地形、建筑等空间连续性；
3. `D←S`：强交互，道路与边界约束动态体运动；
4. `S←D`：弱门控交互，只补充遮挡和接触边界，防止动态噪声污染静态地图。

可写为：

\[
\widehat E_D=E_D+\operatorname{Attn}_{D\leftarrow D}
+\lambda_{DS}\operatorname{Attn}_{D\leftarrow S},
\]

\[
\widehat E_S=E_S+\operatorname{Attn}_{S\leftarrow S}
+\lambda_{SD}\operatorname{Attn}_{S\leftarrow D},
\qquad \lambda_{SD}<\lambda_{DS}.
\]

推荐初值 `λ_DS=1.0, λ_SD=0.25`，之后设为 sigmoid 门控的可学习标量。

### 8.3 TASS 因果性保持不变

RAP 内部仍使用原 `ind_mask`，当前时间 Query 不能看未来时间 Query；SCF 每一步只对已经激活的集合做双流交互。双流角色不能绕过 TASS 的时间因果掩码。

---

## 9. 第五步：共享的统一纠错层

仅靠动态/静态先验会放大角色误判，因此模仿 GaussianWorld“先受限演化、后统一 refinement”的顺序：

\[
E_{joint}=\operatorname{LN}
(W_DE_D+W_SE_S+E_{base}).
\]

随后执行一层或两层共享的空间自注意力 + FFN，得到：

\[
\Delta P_{ik,corr},\quad \Delta Z_{ik,corr},\quad \Delta r_{ik,corr}.
\]

位置纠错仍受角色约束：

\[
P_{ik}^{t+1}=P_{ik,src}^{t+1}
+[\alpha_{corr}(1-\rho_i)+\rho_i]\Delta P_{ik,corr},
\]

其中 `α_corr` 建议不超过 `0.2`。语义和特征则允许对所有 Query 完整更新：

\[
Z^{t+1}=Z^t+\Delta Z_{corr},\qquad
r^{t+1}=\operatorname{clip}(r^t+\Delta r_{corr},0,1).
\]

该层负责：

- 纠正动态/静态角色误判；
- 修正预测自车位姿误差；
- 处理遮挡与新显现区域；
- 恢复双流分解中暂时丢失的全局一致性。

---

## 10. 第六步：规划 Query 对双流分别读取

Baseline 中 ego Query 对全部活动场景做一次 cross-attention。改为：

\[
e_D^{t+1}=\operatorname{CrossAttn}_D(e_{ego}^t,E_D^{t+1},P^{t+1}),
\]

\[
e_S^{t+1}=\operatorname{CrossAttn}_S(e_{ego}^t,E_S^{t+1},P^{t+1}),
\]

\[
e_{ego}^{t+1}=\operatorname{LN}
(e_{ego}^{t}+W_De_D^{t+1}+W_Se_S^{t+1}).
\]

再由共享规划头输出下一轨迹增量与 yaw：

\[
(\Delta\hat x,\Delta\hat y,\sin\Delta\hat\psi,\cos\Delta\hat\psi)
=H_{plan}(e_{ego}^{t+1}).
\]

动态读取承担潜在碰撞与交互，静态读取承担可行驶区域、路沿和固定障碍约束。为防止规划头只依赖某一路，训练前半程可对两路分别做小概率 stream dropout，但不能同时丢弃。

---

## 11. 输出语义融合与原占用接口兼容

### 11.1 保持 SparseWorld 输出格式

最终仍输出：

\[
P^{t+1}\in\mathbb{R}^{B\times N_{t+1}\times48\times3},
\qquad
Z^{t+1}\in\mathbb{R}^{B\times N_{t+1}\times48\times17}.
\]

因此原 `OPUSHead.get_occ()`、voxel scatter、评估协议都可复用。

### 11.2 可选的双语义头

动态与静态各自输出 logits：`Z_D,Z_S`。融合时不要对错误类别组产生高分：

\[
Z_{ikc}=
\begin{cases}
Z_{ikc}^{D}+\log(r_{ik}+\epsilon),&c\in\mathcal C_D,\\
Z_{ikc}^{S}+\log(1-r_{ik}+\epsilon),&c\in\mathcal C_S,\\
\operatorname{LSE}(Z_{ikc}^{D},Z_{ikc}^{S})-\log2,&c=\text{others}.
\end{cases}
\]

第一版也可只保留原 `cls_branch`，把双流用于运动与特征；待几何收益稳定后再加入双语义头，便于逐项消融。

---

## 12. 完整损失函数

### 12.1 保留 Baseline 损失

\[
\mathcal L_{SW}
=\mathcal L_{RAP/TASS}
+\sum_{t=0}^{F}(\mathcal L_{pts}^{t}+\mathcal L_{cls}^{t})
+\lambda_{traj}\mathcal L_{traj}.
\]

论文用 Chamfer + focal 描述点集/分类监督；官方实现通过双向 KNN 匹配配合 `SmoothL1` 点损失和 focal 类损失完成相应集合监督。复现实验应以当前代码配置为准，并在论文叙述中说明实现差异。

### 12.2 角色损失

\[
\mathcal L_{role}
=\operatorname{FocalBCE}(r_{ik},y_{ik}^{D})
+\lambda_q\operatorname{BCE}(\rho_i,\bar y_i^D).
\]

若动态/静态不平衡严重，对点级损失使用类别均衡权重；不建议一开始加入固定动态 Query 比例约束，因为场景真实比例变化很大。

### 12.3 静态一致性损失

仅对**继承且 GT 为静态**的 Query/点施加：

\[
\mathcal L_{static}
=\frac{1}{|\Omega_S|}\sum_{(i,k)\in\Omega_S}
\lVert P_{ik}^{t+1}-\mathcal W(P_{ik}^{t},T_{t\to t+1}^{gt})\rVert_1.
\]

新激活 Query 不施加该约束，否则会阻碍新显现结构的学习。

### 12.4 动态集合损失

对动态匹配点单独计算集合距离与分类：

\[
\mathcal L_{dynamic}
=\operatorname{CD}(P_D^{t+1},G_D^{t+1})
+\lambda_{Dcls}\operatorname{Focal}(Z_D,Y_D).
\]

该损失不需要实例 ID；若后续能获得实例轨迹，再增加 Query 级速度监督，而不是把它作为第一版必要条件。

### 12.5 自车相对位姿损失

\[
\mathcal L_{ego}
=\lVert\Delta\hat t-\Delta t^{gt}\rVert_1
+\lambda_\psi\left[1-cos(\Delta\hat\psi-\Delta\psi^{gt})\right].
\]

轨迹头与相对位姿头共享平移输出，可避免两套 `xy` 互相矛盾。

### 12.6 运动平滑与静态抑制

\[
\mathcal L_{smooth}
=\sum_t\lVert\Delta m_i^{t+1}-\Delta m_i^t\rVert_1,
\]

\[
\mathcal L_{leak}
=\sum_{i,k}(1-y_{ik}^{D})\lVert\Delta m_i^t\rVert_1.
\]

### 12.7 总损失

\[
\boxed{
\mathcal L
=\mathcal L_{SW}
+\lambda_r\mathcal L_{role}
+\lambda_s\mathcal L_{static}
+\lambda_d\mathcal L_{dynamic}
+\lambda_e\mathcal L_{ego}
+\lambda_m\mathcal L_{smooth}
+\lambda_l\mathcal L_{leak}}
\]

推荐从小权重开始：`λ_r=0.5, λ_s=0.2, λ_d=0.5, λ_e=1.0, λ_m=0.05, λ_l=0.05`；最终以各损失梯度量级而不是数值表面大小调权。

---

## 13. 单步前向传播伪代码

```python
def dsqe_scf_step(carried, newly_activated, ego_feat, t, frame_mode):
    # carried: E_c [B,Nc,C], P_c [B,Nc,K,3], Z_c [B,Nc,K,Ccls]
    # new:     E_n [B,Nn,C], P_n [B,Nn,K,3], Z_n [B,Nn,K,Ccls]

    E = concat(E_c, E_n, dim=1)
    P = concat(P_c, P_n, dim=1)
    Z = concat(Z_c, Z_n, dim=1)
    h = concat(ones(Nc), zeros(Nn), dim=1)  # carried/new source flag

    # 1) point-level + query-level soft roles
    semantic_prior = semantic_dynamic_probability(Z, dynamic_class_ids)
    r_point = sigmoid(logit(semantic_prior) + role_head(E, P, Z, h))
    rho_query = attentive_pool(r_point, E, P, Z)

    # 2) ego-conditioned planning and relative-pose prediction
    E_dyn = rho_query * (E + dynamic_embed)
    E_sta = (1 - rho_query) * (E + static_embed)
    ego_dyn = ego_cross_attn_dyn(ego_feat, E_dyn, P)
    ego_sta = ego_cross_attn_sta(ego_feat, E_sta, P)
    ego_next = fuse_ego(ego_feat, ego_dyn, ego_sta)
    rel_pose, pred_traj = plan_pose_head(ego_next)

    # 3) source-aware geometric prior
    P_carried_prior = warp(P_c, rel_pose) if frame_mode == "future_ego" else P_c
    P_new_prior = warp_from_t0_once(P_n, t + 1) if needs_t0_warp else P_n

    # 4) dual evolution; preserve all K points
    motion_query = motion_head(E_c, ego_next, P_carried_prior, rho_query[:, :Nc])
    dP_dyn = dynamic_residual_head(E_c, P_carried_prior)
    dP_sta = static_residual_head(E_c, P_carried_prior)
    P_dyn = P_carried_prior + motion_query[:, :, None, :] + dP_dyn
    P_sta = P_carried_prior + alpha_static * dP_sta
    P_carried_evo = soft_mix(P_sta, P_dyn, rho_query[:, :Nc], r_point[:, :Nc])
    P_new_evo = P_new_prior + new_full_update_head(E_n, P_new_prior)
    P_evo = concat(P_carried_evo, P_new_evo, dim=1)

    # 5) asymmetric dual-stream interaction
    E_dyn = attn_DD(E_dyn, P_evo) + gate_DS * attn_DS(E_dyn, E_sta, P_evo)
    E_sta = attn_SS(E_sta, P_evo) + gate_SD * attn_SD(E_sta, E_dyn, P_evo)

    # 6) shared correction
    E_joint = joint_refine(E, E_dyn, E_sta, P_evo)
    dP_corr, Z_next, role_corr = output_heads(E_joint, P_evo)
    P_next = P_evo + correction_scale(rho_query) * dP_corr
    r_next = clamp(r_point + role_corr, 0, 1)

    return E_joint, P_next, Z_next, r_next, ego_next, pred_traj, rel_pose
```

---

## 14. 与现有源码的最小侵入式改造位置

### 14.1 `sparseworld_4d_traj.py`

替换 SCF 循环内部的以下逻辑：

- 删除 `argmax → pred_moving_mask → vel_offset` 硬门控；
- 删除/绕开对 48 点先取均值的 `refine_points()`；
- 在每个 interval 拼接 Query 后，显式构造 `source_flag`；
- 插入 `RoleRouter`、`EgoWarp`、`DualEvolution`、`DualInteraction`、`JointCorrection`；
- `pred_traj` 与相对自车位姿平移共享输出；
- 保持 `forecast_points_list`、`forecast_semantics_list`、`pred_trajs_list` 的外部接口不变。

### 14.2 `opus_head.py`

- 保持 `num_stamps_all: [1040,7]` 和 `ind_stamps_all`；
- 保持原 KNN 匹配，额外返回匹配标签作为 role target；
- 在 `loss_future()` 中加入动态/静态分组损失；
- 第一版保持原 TASS `fore_mask`，另做“按新静态集合定义修正 fore_mask”的消融。

### 14.3 `opus_transformer.py`

- RAP 的采样、Adaptive Mixing、TS-MHSA、FFN 先不改；
- 若后续希望角色从感知阶段提前形成，可只在最后一层增加 role head；
- 不得破坏 `ind_mask` 的时序因果性。

### 14.4 建议新增文件

```text
mmdet3d/models/sparsedetectors/
├── dsqe_role_router.py
├── dsqe_ego_warp.py
├── dsqe_dual_evolution.py
├── dsqe_dual_interaction.py
└── dsqe_joint_refine.py
```

---

## 15. 训练策略

### Stage 0：复现 Baseline

- 使用官方权重或严格复现官方训练；
- 记录当前帧、1–6 秒未来占用和规划指标；
- 保存 `ind_stamps_all` 与每时刻活动 Query 数，用作后续一致性检查。

### Stage 1：只训练角色头与相对位姿头

- 主干、RAP、原 SCF 大部分参数冻结；
- 角色头使用 KNN 匹配 GT 标签；
- 位姿头使用 GT 自车相对变换；
- 几何主路径暂时仍用 Baseline，先验证 `r_point/ρ_query` 与 yaw 输出有效。

### Stage 2：启用来源感知双流演化

- 继承 Query 使用静态/动态运动分解；
- 新激活 Query 使用完整更新；
- GT 角色与预测角色混合：

\[
r_{route}=\eta r_{gt}+(1-\eta)r_{pred},
\]

其中 `η` 从 `1` 线性衰减到 `0`；自车位姿 teacher forcing 同理。

### Stage 3：启用双流交互与共享纠错

- 解冻 SCF；
- 沿用 Baseline 的未来步 curriculum：训练初期只预测 1 帧，逐 epoch 增长到 6 帧；
- 观察长时滚动中的静态漂移和动态运动衰减。

### Stage 4：端到端微调

- 完全使用预测角色与预测自车位姿；
- 逐步解冻 RAP 最后 1–2 层；
- 不建议一开始重训全部 RAP，因为角色路由和时间分配同时漂移容易导致 Query collapse。

---

## 16. 必做消融实验

| 编号 | 模型 | 验证问题 |
|---|---|---|
| B0 | 官方 SparseWorld | 真实 Baseline |
| B1 | 用软角色替换硬 `argmax` 速度门控 | 可导门控本身是否有效 |
| B2 | B1 + 显式自车坐标变换 | 静态漂移是否来自隐式 ego motion |
| B3 | B2 + carried/new 来源感知 | 新激活 Query 是否需要独立更新 |
| B4 | B3 + Query 级主体运动 + 点级残差 | 是否优于 48 点独立速度 |
| B5 | B4 + 动/静双流非对称交互 | 道路约束与动态交互是否有效 |
| B6 | B5 + 共享统一纠错 | 是否能修复硬分解误差 |
| B7 | B6 + 双语义头 | 语义解耦是否额外增益 |
| C1 | 48 点保留 vs. 均值后重建 | 局部几何是否被 Baseline 破坏 |
| C2 | hard role vs. soft role | 训练稳定性与边界混合 |
| C3 | GT pose / predicted pose / no warp | 排查训练—推理位姿差距 |
| C4 | `traffic_cone` 动态 vs. 静态 | 验证类别集合修正 |
| C5 | 保持 TASS fore_mask vs. 新静态集合 | 区分时间分配与运动角色的影响 |

报告指标至少包括：

- 总体 IoU/mIoU 与每个未来时刻指标；
- 动态类 mIoU、静态类 mIoU；
- car、pedestrian、truck 等关键动态类别；
- 经 GT ego warp 后的静态点漂移；
- 动态点位移误差；
- 规划 L2 与 collision rate；
- 参数量、显存、FPS。

---

## 17. 实现自检清单

1. `ind_stamps_all` 是否仍是一维时间标签，而非动态/静态标签？
2. RAP 内的未来因果 mask 是否仍有效？
3. `traffic_cone` 是否按静态类处理，而不是继续使用 `2...10` 连续区间？
4. 角色路由是否使用概率而非 `argmax`？
5. 一个 Query 的 48 点是否在相邻未来帧之间逐点保留？
6. 自车 warp 是否与 future label 的坐标系一致，且只执行一次？
7. 推理是否完全不读取未来 GT pose？
8. 静态一致性损失是否只约束 carried-static，而不约束 new Query？
9. `D←S` 是否强于 `S←D`，避免动态噪声污染静态流？
10. 最终输出形状是否仍兼容 `get_occ()`？
11. 角色 GT 是否来自现有 KNN 匹配，而不是另造不一致的匹配？
12. 动态/静态双流是否增加了明确的单元测试：全静态场景、全动态合成场景、无新 Query、全部新 Query、零 ego motion、纯 ego rotation？

---

## 18. 创新边界与可写入论文的准确表述

不能写：

- “SparseWorld 没有动态运动分支”；它已有按预测类别硬掩码的 `vel_branch`。
- “GaussianWorld 使用完全独立的动态/静态双流”；其实现本质是来源感知与语义门控后的共享 refinement。
- “首次将 Query 分成动态和静态”；仅按类别拆分不是足够强的新意。

可以准确写成：

> We introduce a source-aware dynamic–static query evolution module for SparseWorld. Instead of the baseline's non-differentiable point-wise class mask, our model performs hierarchical soft role routing at both point and query levels. It explicitly decomposes future evolution into ego-coordinate transformation, query-level local motion, and point-level deformation; applies asymmetric dynamic–static interaction; and finally uses a shared correction stage to recover from role and pose uncertainty. Temporal query allocation remains governed by the original TASS mechanism.

中文对应：

> 本文提出来源感知的动态—静态 Query 演化模块。不同于 SparseWorld Baseline 基于点级预测类别 `argmax` 的不可导速度硬门控，模型在点级和 Query 级进行层级软角色路由，并将未来场景演化显式分解为自车坐标变换、Query 级局部运动和点级形变；通过非对称动静交互建模道路—交通体约束，最后用共享纠错层吸收角色与位姿不确定性，同时完整保留原 TASS 的时间 Query 分配。

---

## 19. 参考源码与论文

- SparseWorld 论文：上传文件 `SparseWorld(2).pdf`，重点为第 3.2–3.4 节及 Figure 2。
- [SparseWorld 官方仓库](https://github.com/MSunDYY/SparseWorld)
- [SparseWorld SCF 与规划源码](https://github.com/MSunDYY/SparseWorld/blob/master/mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py)
- [SparseWorld TASS、匹配和未来损失源码](https://github.com/MSunDYY/SparseWorld/blob/master/mmdet3d/models/sparsedetectors/opus_head.py)
- [SparseWorld RAP Transformer 源码](https://github.com/MSunDYY/SparseWorld/blob/master/mmdet3d/models/sparsedetectors/opus_transformer.py)
- [GaussianWorld 官方仓库](https://github.com/zuosc19/GaussianWorld)
- [GaussianWorld 历史 anchor warp/fill 源码](https://github.com/zuosc19/GaussianWorld/blob/main/model/decoder/gaussian_decoder/gaussian_decoder_stream.py)
- [GaussianWorld 来源感知与动态语义 refinement 源码](https://github.com/zuosc19/GaussianWorld/blob/main/model/encoder/gaussian_encoder/refine_layer.py)
- [GaussianWorld 流式配置](https://github.com/zuosc19/GaussianWorld/blob/main/config/nusc_surroundocc_stream_eval.py)
- [GaussianWorld 论文](https://arxiv.org/html/2412.10373v1)
