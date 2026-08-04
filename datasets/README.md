# `datasets/` — 复现 Grasp3 训练所需的最小数据集

**目的**：让外部协作者 `git clone` 完就能直接开一次训练，不必先拿到上游的
`Reconstruct_and_Retarget` / `AffordanceModel` 两个仓库。

**只够跑 `--clip Grasp3` 这一条。** 别的 clip 仍需完整上游产物，见
`docs/DEPLOY_NEW_MACHINE.md` §2B ②。

---

## 内容（88MB，全部走 Git LFS）

| 文件 | 大小 | 是什么 |
|---|---|---|
路径里的 `<C>` = `egodex/part2/basic_pick_place/3`。

| 文件 | 大小 | 是什么 | 缺了会怎样 |
|---|---|---|---|
| `RR/Output/RetargetOutput/<C>/replay_world.npz` | 0.95MB | **人手参考轨迹**（世界系）+ 物体逐帧位姿。残差策略叠加的那条参考 | 起不来 |
| `RR/Output/RetargetOutput/<C>/ref_qpos.npz` | 11KB | 离线 retarget 的 **22 关节手指参考 qpos**（`export_qpos.py` 产物） | `correction_env.py` 硬断言 `缺 ref_qpos.npz` |
| `RR/Output/RetargetOutput/<C>/object.usd` | 9MB | 物体视觉 USD（物理属性运行时贴，见 `clips.runtime_object_physics`）。已验证自包含 | 场景建不出物体 |
| `RR/Output/ReconstructOutput/<C>/object_mesh_scaled_final.obj` | 65MB | 重建的**物体网格**（真实尺度）。接触判定 / 分数图 FPS 采样 / GraspPose 筛选 | 起不来 |
| `RR/Output/ReconstructOutput/<C>/world_fused.npz` | 106KB | 重建的**世界系元数据**：`c2w` 相机轨迹 + `world_xy_alignment_mode` | ⚠ **静默降级**：`camera_anchor_shift` 返回 None → 退回 palm 锚定（§2.8 之前的旧约定），只打一行 WARN，摆放和我们的不一样 |
| `RR/Output/ReconstructOutput/<C>/stable_poses.json` | 1.5KB | `trimesh.compute_stable_poses` 的**缓存** | 在 Isaac 进程里现算 1~2 分钟 → 阻塞主循环触发中止回调 → **段错误 exit 139** |
| `AffordanceModel/outputs/pred_egodex_part2_all20/obj_03/affordance.npz` | 0.06MB | 逐点 affordance 热图（物体系），设定 B 用 | 抓取目标点退化到质心 |
| `vega_urdf/vega_1p_sharpa/` | 16MB | DexMate URDF + 54 个 STL。`ArmIK` 的正/逆运动学靠它 | ArmIK 起不来 |

### ⚠ 两处 mtime 判定已为快照放宽

git **不保留 mtime** —— clone 后所有文件的时间戳都是克隆时刻。上游有两处按 mtime
判新鲜度的检查，在别人机器上会必然误判：

| 位置 | 原判定 | 对快照的处理 |
|---|---|---|
| `load_replay.py` | `ref_qpos.npz` 的 `source_mtime` 必须与 `replay_world.npz` 的 mtime 差 <1s，否则断言"ref_qpos 过期" | 跳过（两份是一起提交的，必然配套） |
| `bimanual_align.py` | `stable_poses.json` 缓存 key = `mesh名:mesh的mtime` | 只比 mesh 名 |

判据是 `paths.is_bundled(path)`（路径是否在 `datasets/` 下），**上游仓路径仍走原来的严格判定**。

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
cp $RR_ROOT/Output/RetargetOutput/$CLIP/{replay_world.npz,object.usd,ref_qpos.npz} \
   datasets/RR/Output/RetargetOutput/$CLIP/
cp $RR_ROOT/Output/ReconstructOutput/$CLIP/{object_mesh_scaled_final.obj,world_fused.npz,stable_poses.json} \
   datasets/RR/Output/ReconstructOutput/$CLIP/
```

已知数据缺陷（不是 bug，是重建本身的）：手↔物 XY 存在 14~16cm 的系统偏差，
详见 `docs/DESIGN_LOOP.md` 和摆放约定相关章节。
