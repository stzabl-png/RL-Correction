# SharpaWave 抓取生成管线使用手册

物体 mesh → Dexonomy 合成 GraspPose（SharpaWave 右手，22 DOF）→ Isaac PhysX 物理验证 → 只保留成功且姿态自然的抓取。

## 一键运行（推荐）

```bash
conda activate dexonomy && cd /home/lyh/Project/Dexonomy
tools/grasp_pipeline.sh <mesh.obj路径> [物体名]
```

- 物体名缺省为 `obj_<mesh父目录名>`；重建物体惯例命名 `pp<编号>`（如 pp0, pp5）。
- 环境变量调参（写在命令前）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `TMPL` | fingertip_mid | 模板（可选 `1_Large_Diameter` `3_Medium_Wrap` `fingertip_small` 及 Isaac 回灌的 `isaac_*`，见 `assets/hand/sharpa_wave/init_tmpl/`） |
| `EPOCH` | 50 | init 采样轮数（每轮 ~10 个初始化；不够就翻倍） |
| `PLANE_THRE` | 0.012 | init 阶段手骨架离桌面最小距离 [m] |
| `PLANE_MARGIN` | 0.008 | grasp 精修阶段桌面斥力缓冲 [m]（矮扁物体用小值） |
| `MASS` | 0.1 | Isaac 中物体质量 [kg] |
| `UP` | 自动 | 手动指定静置朝上方向（物体 mesh 系），如 `UP="0 1 0"` |
| `KEEP_SERVER` | 0 | 1=跑完保留 Isaac 服务器（连续处理多物体时用，最后一个用 0 收尾） |

**最终交付物**：`output/<名>_sharpa_wave/isaac_succ/`
```
<抓取名>.npy          # 抓取数据（字段见下）
<抓取名>_video.mp4    # Isaac 物理回放（接近→合指→加力→抬升8cm）
<抓取名>_report.json  # 物理指标（metrics.grasp_success 等）
summary.json          # 本物体全部成功项汇总
```

## 管线内部步骤（脚本自动执行；也可单独调用排查）

| 步 | 工具 | 输入 → 输出 |
|---|---|---|
| 1 导入 | `tools/import_object.py --mesh X --oid N --auto-up` | mesh → `assets/object/custom/{processed_data,scene_cfg}/N/`（质心居中 + **旋转到静置姿态 z-up 规范系** + CoACD 凸分解 + tabletop 场景带 virtual_plane） |
| 2 初始化 | `dexrun op=init hand=sharpa_wave exp_name=N tmpl_name=T 'init_gpu=[0]' op.object.cfg_path=... op.epoch=E op.filter.collision.plane_thre=0.012` | GPU 采样物体表面匹配模板接触点 → `init_data/`（硬平面约束：桌下姿态直接删，见 `filter.collision.hard_plane`） |
| 3 精修 | `dexrun op=grasp hand=sharpa_wave exp_name=N 'op.grasp.plane_margin=0.008'` | MuJoCo 虚拟接触力精修 + 四道过滤（限位/碰撞/接触组/力闭合QP）→ `grasp_data/` |
| 4 净距过滤 | `tools/filter_plane_clearance.py --exp-dir E --data grasp_data --check-pregrasp` | FK 全手碰撞 mesh，剔除穿桌 >2mm 的 → 移入 `grasp_data_below_plane/` |
| 4.5 姿态过滤 | `tools/filter_hand_orientation.py --exp-dir E --data grasp_data` | **剔除虎口朝下**（手根 +Y 轴世界 z < -0.25）→ 移入 `grasp_data_thumb_down/` |
| 5 导出 | `tools/export_isaac_traj.py --exp-dir E --data grasp_data` | 每个 npy → `isaac_traj/<名>/trajectory.{npz,json}`（ocir Stage B 契约） |
| 6 服务器 | ocir `scripts/sim/start_isaacsim_server.py --mode local`（**必须 --mode local**，本机无 streaming kit） | 常驻 Isaac，HTTP 127.0.0.1:8765 |
| 7 批量验证 | `tools/run_isaac_batch.py --traj-root E/isaac_traj` | 每条 ~35 秒 → `isaac_traj/<名>/isaac_sim/{report.json,video.mp4}` + `isaac_summary.json` |
| 8 收尾 | `tools/keep_isaac_success.py --exp-dir E` | 成功项收进 `isaac_succ/`；失败轨迹删除、失败 npy 移 `grasp_data_isaac_failed/` |

后处理（手动，可选）：`tools/promote_templates.py --exp-dir E --prefix isaac_N` 把成功抓取回灌为模板（后续 `TMPL=isaac_N_xxx` 生成同风格）。渲染检查：`tools/render_grasps.py --exp-dir E --data grasp_data`（真实桌面自动染半透明粉色）。

## 数据格式

**抓取 npy**（`np.load(f, allow_pickle=True).item()`）关键字段：
- `grasp_qpos (1,29)`：`[x,y,z, qw,qx,qy,qz, 22关节角]`。**物体规范系**（物体质心在原点、z-up 静置、桌面在 z=zmin 水平面）。四元数 wxyz。
- `pregrasp_qpos (K,29)`：张开→合拢的接近序列（第 0 帧最张开）；`squeeze_qpos (1,29)`：施力目标。
- 关节顺序 = ocir `sharpa_wave_right.yml` 的 `joint_order`（thumb 5 + index/middle/ring 各 4 + pinky 5），**与 MJCF/Isaac 恒等映射**。

**映射回真实场景**：`T_场景_手 = T_场景_物体输入系 × Trans(com_offset) × Rot(canonical_from_input_rot_wxyz)⁻¹ × T_规范系_手`；两个补偿量存在 `assets/object/custom/processed_data/<名>/info/simplified.json`。物体在场景中绕竖直轴旋转/平移时，抓取直接跟随变换。

**Isaac 成功判定**（`report.json → metrics`）：`grasp_success` = carry 段连续 ≥5 帧抬升 ≥2cm 且结束时未掉落。其余指标：`max_lift_m`、`max_consecutive_lifted_steps`、`object_dropped`。

**contact 评估模式**（2026-08-06 新增，只评接触点质量、不做提起测试）：
- 导出：`export_isaac_traj.py --no-carry --squeeze-frames 60`（approach→close→2s 缓慢收紧，无抬升段）
- 验证：`run_isaac_batch.py --eval-mode contact`（自动开 contact-aware governor 做有界缓慢收紧）
- 判定（ocir `simulate_grasp_traj.py --eval-mode contact`，物体挂 PhysX 接触报告）：`grasp_success` = 收紧+settle 末段持续接触（≥80% 帧）包含**拇指+至少一根对指**，且物体位移 ≤3cm（`--max-object-drift`）、倾倒 ≤30°（`--max-object-tilt-deg`）。新指标：`contact_fingers_sustained`、`object_drift_m`、`object_tilt_deg`；逐帧接触存 `isaac_sim/contact_track.npz`。
- **左手三件套缺一不可**：导出 `--side left` + batch `--asset-config $OCIR/assets/robots/hands/sharpa_wave/sharpa_wave_left.yml`；漏传会用右手资产仿左手位姿，全部失败且不报错（2026-08-06 踩过）。

## 关键约定与教训（改动前必读）

1. **规范系 = 静置姿态**：导入时物体被旋转到"躺/立在桌上"的真实姿态。`--auto-up` 会读 mesh 同目录的 `world_fused.npz`，用视频位姿在 trimesh 稳定姿态（p≥0.02）中消歧——**不要只信最大概率稳定姿态**（教训：pp0 瓶子视频里立着，最大概率姿态却是躺倒）。选择与视频差 >45° 会打 WARNING，此时人工确认 `--up`。
2. **虚拟平面 vs 真实平面**：init 用 virtual_plane 过滤（`hard_plane=True` 硬约束，修复过 quota 补齐漏洞）；grasp 用真实 plane geom + 小 margin；eval/Isaac 用真实桌面零 margin。三者缺一不可（历史 bug：无桌 eval 时物体自由落体，系统性偏爱从桌下抓的姿态）。
3. **MuJoCo eval（op=eval）不在主管线里**：其 15° 转动阈值对扁平/环状物过严，仅作快速预估用（`op.rot_thre=45` 放宽）。最终裁判是 Isaac。
4. Dexonomy 重跑某个 op 前要删 `output/<exp>/log/<op>_*/`（skip_done 机制）；覆盖 n_worker 用顶层 `n_worker=1` 不是 `op.n_worker`。
5. warp 的 `CUDA error 36` 报错无害（走 CPU 回退，torch CUDA 正常）。
6. 命令必须在 `/home/lyh/Project/Dexonomy` 目录、`dexonomy` 环境下执行；给用户的命令用**单行**（多行反斜杠粘贴易断）。

## 已处理物体（basic_pick_place 系列）

| 物体 | 说明 | 自然抓取成功数 |
|---|---|---|
| pp0 | 瓶子（立） | 35 |
| pp1 | — | 1 |
| pp2 | 环状 | 4 |
| pp3 | — | 16 |
| pp5 | 环状（螺母状） | 9 |

批量处理脚本模式（服务器复用）见 `output/batch_pp_all.log` 的调用方式；剩余待处理：pp4, pp6~pp19, pp55。

## 相关代码位置

- 手部资产与生成脚本：`assets/hand/sharpa_wave/`、`tools/sharpa_wave/build_assets.py`（URDF→MJCF 全自动重建）
- 对 Dexonomy 核心的修改：`dexonomy/op/gen_init.py`（hard_plane + 平面检查物体系变换）、`dexonomy/op/gen_grasp.py`（plane_margin 可配）、`dexonomy/config/op/{init,grasp}.yaml`
- **Isaac 验证栈已内置（2026-08-06 起，不再依赖 ocir 仓库）**：`isaac/` 目录 = 从 ocir 移植的 `dexisaac` 包（仿真 `src/dexisaac/isaac/simulate_grasp_traj.py`、轨迹契约 `grasp_traj/trajectory_schema.py`、服务器 `sim/isaac_server.py`）+ 手部 USD 资产 `isaac/assets/robots/hands/sharpa_wave/`（左右手 yml 都在这）+ manifest `isaac/data/testing/identity_manifest/`。启动：`cd isaac && scripts/run_isaacsim_conda.sh scripts/start_isaacsim_server.py --mode local`（仍用 `env_isaacsim` conda 环境）。改仿真代码后必须重启 server。ocir 仓库对应文件此后只作历史参考，两边不再同步。
