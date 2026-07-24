# Reconstruct_and_Retarget 下游项目:RL 修正 SharpaHand 抓取轨迹

> 归档笔记。recon 之后的核心项目结构 + 设计共识 + MVP 范围。
> 相关:`Reconstruction/`(已分析)、`Reconstruction/fp_pose_improvement_notes.md`。

## 目标
第一人称视频 recon 出**带噪**结果(egocentric → **手估计准、物体估计误差大**),
训一个 **RL correction 模型**,输入 Object Geometry(知道"抓什么/怎么抓"),
**修正 SharpaHand 动作轨迹**,输出在 IsaacSim 里能成功的轨迹,最终 **0/1**。
RL 经 **optimization** 学习(细节待定,参考 https://github.com/Shen626/Articulation_Bodex,BODex 系抓取优化)。

## 端到端结构
```
① Reconstruction(已分析完)
   视频 → world_fused.npz(noisy: 物体mesh + 逐帧物体6DoF + 手MANO轨迹) + object mesh
② Retarget
   MANO 手轨迹 → SharpaHand(+臂)轨迹   [Jiakai 的 retarget_pipeline 已有一版]
③ Object Geometry(新模块,待细化)
   recon mesh → {水密mesh, 碰撞mesh, 表面点云, 法向, SDF, (后期)shape latent}
④ RL Correction Model
   输入(noisy: 物体初始位姿 + geometry + 手/Sharpa轨迹先验) → 修正后的 SharpaHand 动作轨迹
⑤ IsaacSim 验证
   驱动修正后的 SharpaHand → 物理 rollout → 任务判据 → 0/1(同时给 ④ 当 reward)
```

## 设计共识
- **只驱动手,物体被动**:sim 里物体运动是接触物理的*结果*,不 set 物体轨迹。
- **单手 grasp**:只修手基本够(初始物体位姿准 + geometry 好);RL 主要补 MANO→SharpaHand 的 embodiment gap(抓稳)。
- **双手协作(handover/倒水)**:仍只驱动手,但需 ③ geometry 保证两抓取在**同一刚体一致** + 用 sim rollout 的物体状态协调双手 + 任务判据依赖物体状态。
- **recon 物体轨迹 = 置信度加权参考**:信遮挡前/关键帧,遮挡段靠物理。直接用 fp_pose 的 valid/置信度(见 fp_pose_improvement_notes A1)。
- **视频给"复现什么交互",RL+sim 找"SharpaHand 怎么做才物理可行"**(imitation + physics-based correction)。

## 范围
- ✅ 先**刚体**(SAM3D 只出刚体);铰接后期下载 asset。
- ✅ **全局形状 + 局部接触**都要。
- ⏳ 跨物体泛化(shape latent)放后期。
- 🎯 **当前 MVP:单物体 · 简单 Grasp · 刚体**,先不做 shape latent/泛化。

## MVP 切片
| 环节 | MVP 做法 |
|---|---|
| ① recon | 一条已跑通序列的 `world_fused.npz` + 物体 mesh |
| ② retarget | 复用 Jiakai 的 retarget → SharpaHand 抓取先验 |
| ③ Object Geometry | 水密mesh + 凸分解碰撞体 + 表面点云+法向 + SDF;**不做 shape latent** |
| ④ RL | 修 SharpaHand 抓取动作:给定 geometry+初始位姿,抓稳并抬起 |
| ⑤ sim | IsaacSim:reward = 抬起成功 + 抓取稳定 + 模仿可靠段手轨迹;输出 0/1 |

MVP 里 Object Geometry 为**静态**(重建一次),位姿取 recon 初始帧 + 物理。

## 待定 / TODO
- ④ optimization 学习方式(reward含优化目标 / loop挂优化器 / model-based)——处理完 ③ 再定,参考 Articulation_Bodex。
- 铰接物体支持(多部件 mesh + 关节)——后期。
- 跨物体泛化的 shape encoder——后期。

## 概念备忘
- **水密(watertight)**:闭合、无洞、流形的 mesh;SDF 判内外/物理碰撞的前提。SAM3D mesh 常带洞需水密化。
- **shape latent**:形状编码向量(PointNet++/DeepSDF 等),泛化时给 policy 当形状观测。
