# `datasets/` — 复现 Grasp3 训练所需的最小数据集

**目的**：让外部协作者 `git clone` 完就能直接开一次训练，不必先拿到上游的
`Reconstruct_and_Retarget` / `AffordanceModel` 两个仓库。

**只够跑 `--clip Grasp3` 这一条。** 别的 clip 仍需完整上游产物，见
`docs/DEPLOY_NEW_MACHINE.md` §2B ②。

---

## 内容（88MB，全部走 Git LFS）

| 文件 | 大小 | 是什么 |
|---|---|---|
| `RR/Output/RetargetOutput/egodex/part2/basic_pick_place/3/replay_world.npz` | 0.95MB | **人手参考轨迹**（世界系）+ 物体逐帧位姿。残差策略叠加的那条参考 |
| `RR/Output/ReconstructOutput/egodex/part2/basic_pick_place/3/object_mesh_scaled_final.obj` | 65MB | 重建出的**物体网格**（已缩放到真实尺度）。用于接触判定、分数图 FPS 采样、GraspPose 筛选 |
| `RR/Output/RetargetOutput/egodex/part2/basic_pick_place/3/object.usd` | 9MB | 物体视觉 USD（物理属性运行时贴上，见 `clips.runtime_object_physics`）。已验证自包含，无外部引用 |
| `AffordanceModel/outputs/pred_egodex_part2_all20/obj_03/affordance.npz` | 0.06MB | 逐点 affordance 热图（物体系），设定 B 用 |
| `vega_urdf/vega_1p_sharpa/` | 16MB | DexMate URDF + 54 个 STL。`ArmIK` 的正/逆运动学靠它 |

机器人本体 USD 不在这里，在 `assets/vega_1p_sharpa_fixedtorso.usd`。

## 路径是怎么找到的

`rl_rebuild/correction/paths.py` 和 `kinematics.py` 按这个优先级解析：

```
环境变量 (RR_ROOT / AFFORDANCE_ROOT / VEGA_URDF)
  ↓ 没设
本机上游仓的默认路径 —— **存在才用**
  ↓ 不存在 (= 外部协作者的机器)
本仓库的 datasets/          ← 自动回落到这里
```

所以在原作者机器上行为完全不变（仍读上游仓的最新产物），在别人机器上自动用这份快照。
想强制用快照：`export RR_ROOT=$PWD/datasets/RR AFFORDANCE_ROOT=$PWD/datasets/AffordanceModel`。

## ⚠ 这是快照，不是数据源

上游重跑重建/retarget 后这里**不会自动更新**。要同步就重新拷一次：

```bash
CLIP=egodex/part2/basic_pick_place/3
cp $RR_ROOT/Output/RetargetOutput/$CLIP/{replay_world.npz,object.usd} \
   datasets/RR/Output/RetargetOutput/$CLIP/
cp $RR_ROOT/Output/ReconstructOutput/$CLIP/object_mesh_scaled_final.obj \
   datasets/RR/Output/ReconstructOutput/$CLIP/
```

已知数据缺陷（不是 bug，是重建本身的）：手↔物 XY 存在 14~16cm 的系统偏差，
详见 `docs/DESIGN_LOOP.md` 和摆放约定相关章节。
