# Dexonomy 抓取合成（**当前默认路线**）

物体 mesh → Dexonomy 合成 GraspPose（SharpaWave 右手，22 DOF）→ Isaac PhysX 物理验证
→ 只保留成功且姿态自然的抓取。

BODex 路线（`docs/anchored_bodex.md`、`docs/bodex_curobo_v2.md`）**保留为备选**，
两条路线的取舍见 [§0](#0-为什么默认换成-dexonomy)。

---

## 0. 为什么默认换成 Dexonomy

| | BODex（原路线，仍可用） | **Dexonomy（现默认）** |
|---|---|---|
| 抓取类型 | 由力闭合优化自由涌现，类型不可控 | **分类学模板驱动**——38 种人类抓取类型（Large Diameter / Tripod / Lateral / Ring …）各自成为一个初值模板 |
| 想要某种抓法 | 只能靠改约束间接引导 | 直接指定 `TMPL=<模板名>` |
| 成功抓取的风格 | 单一 | 可回灌：成功抓取用 `promote_templates.py` 变成新模板，后续生成同风格 |
| 桌面约束 | 后处理惩罚 | init 阶段**硬约束**（桌下姿态直接删），grasp 阶段真实 plane geom |
| 物理验证 | 同 | 同（Isaac PhysX，最终裁判） |

一句话：**BODex 回答"这个物体能不能抓稳"，Dexonomy 回答"用哪一种人类抓法抓稳"。**
Step4 的 RL 需要后者——参考轨迹里的人手是有类型的，抓取先验也应该有。

---

## 1. 这个目录装的是**适配层**，不是完整仓库

上游 Dexonomy（[JYChen18/Dexonomy](https://github.com/JYChen18/Dexonomy)，RSS 2025）
连物体数据集有 8.4GB，不入库。本仓库只放**把 SharpaWave 接进去所需的那一层**：

```
assets/robots/hands/sharpa_wave/grasp_synthesis/dexonomy/
    hand/                       SharpaWave 右手在 Dexonomy 里的定义
        right.xml               MJCF（由 URDF 自动重建，见 build_assets.py）
        keypoint.yaml           关键点标注
        skeleton.yaml           骨架层级
        body_group.yaml         接触组划分（力闭合过滤要用）
        meshes/*.STL            碰撞/视觉网格（col0..colN 为凸分解件）
        raw_anno/*.yaml         38 种抓取类型的人手接触点标注 + fingertip_* 自定义
        init_tmpl/*.npy         上面那些标注**重定向到 SharpaWave 后**的初值模板
    config/sharpa_wave.yaml     Dexonomy 的 hand 配置项
    patches/
        sharpa_wave-adaptation.patch    对上游 4 个文件的改动

scripts/grasp_synthesis/dexonomy/       管线脚本（在 Dexonomy 仓库根目录下运行）
```

### 安装到一个上游 Dexonomy 克隆里

```bash
git clone --recursive https://github.com/JYChen18/Dexonomy.git
cd Dexonomy && git checkout d561ff9        # patch 基于这个 commit

R=<RL-Correction 仓库路径>
A=$R/assets/robots/hands/sharpa_wave/grasp_synthesis/dexonomy
cp -r $A/hand                assets/hand/sharpa_wave
cp    $A/config/sharpa_wave.yaml  dexonomy/config/hand/
cp -r $R/scripts/grasp_synthesis/dexonomy  tools
git apply $A/patches/sharpa_wave-adaptation.patch
```

patch 改了四处（共 42 行）：

| 文件 | 改了什么 | 为什么 |
|---|---|---|
| `dexonomy/op/gen_init.py` | `hard_plane` 硬平面约束 + 平面检查的物体系变换 | 原实现只做软过滤，且 quota 补齐时会把桌下姿态放回来 |
| `dexonomy/op/gen_grasp.py` | `plane_margin` 可配 | 矮扁物体需要更小的桌面斥力缓冲 |
| `dexonomy/config/op/{init,grasp}.yaml` | 暴露上面两个开关 | — |

---

## 2. 一键运行

```bash
conda activate dexonomy && cd <Dexonomy 仓库>
tools/grasp_pipeline.sh <mesh.obj 路径> [物体名]
```

物体名缺省 `obj_<mesh 父目录名>`；重建物体惯例命名 `pp<编号>`（pp0、pp5…）。

环境变量调参（写在命令前）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `TMPL` | `fingertip_mid` | 抓取类型模板。可选 `1_Large_Diameter`、`3_Medium_Wrap`、`fingertip_small`，以及 Isaac 回灌的 `isaac_*`。全部见 `hand/init_tmpl/` |
| `EPOCH` | 50 | init 采样轮数（每轮约 10 个初始化；不够就翻倍） |
| `PLANE_THRE` | 0.012 | init 阶段手骨架离桌面最小距离 [m] |
| `PLANE_MARGIN` | 0.008 | grasp 精修阶段桌面斥力缓冲 [m]（矮扁物体用小值） |
| `MASS` | 0.1 | Isaac 中物体质量 [kg] |
| `UP` | 自动 | 手动指定静置朝上方向（物体 mesh 系），如 `UP="0 1 0"` |
| `KEEP_SERVER` | 0 | 1 = 跑完保留 Isaac 服务器（连续处理多物体时用，最后一个用 0 收尾） |

**交付物**：`output/<名>_sharpa_wave/isaac_succ/`

```
<抓取名>.npy          抓取数据（字段见 §4）
<抓取名>_video.mp4    Isaac 物理回放（接近 → 合指 → 加力 → 抬升 8cm）
<抓取名>_report.json  物理指标（metrics.grasp_success 等）
summary.json          本物体全部成功项汇总
```

---

## 3. 管线八步（脚本自动执行，也可单独调用排查）

| 步 | 工具 | 输入 → 输出 |
|---|---|---|
| 1 导入 | `import_object.py --mesh X --oid N --auto-up` | mesh → `assets/object/custom/{processed_data,scene_cfg}/N/`（质心居中 + **旋转到静置姿态 z-up 规范系** + CoACD 凸分解 + tabletop 场景带 virtual_plane） |
| 2 初始化 | `dexrun op=init hand=sharpa_wave exp_name=N tmpl_name=T 'init_gpu=[0]' op.object.cfg_path=… op.epoch=E op.filter.collision.plane_thre=0.012` | GPU 采样物体表面匹配模板接触点 → `init_data/` |
| 3 精修 | `dexrun op=grasp hand=sharpa_wave exp_name=N 'op.grasp.plane_margin=0.008'` | MuJoCo 虚拟接触力精修 + 四道过滤（限位 / 碰撞 / 接触组 / 力闭合 QP）→ `grasp_data/` |
| 4 净距过滤 | `filter_plane_clearance.py --exp-dir E --data grasp_data --check-pregrasp` | FK 全手碰撞 mesh，穿桌 >2mm 的移入 `grasp_data_below_plane/` |
| 4.5 姿态过滤 | `filter_hand_orientation.py --exp-dir E --data grasp_data` | **剔除虎口朝下**（手根 +Y 轴世界 z < −0.25）→ `grasp_data_thumb_down/` |
| 5 导出 | `export_isaac_traj.py --exp-dir E --data grasp_data` | 每个 npy → `isaac_traj/<名>/trajectory.{npz,json}`（ocir Stage B 契约） |
| 6 服务器 | ocir `scripts/sim/start_isaacsim_server.py --mode local` | 常驻 Isaac，HTTP 127.0.0.1:8765 |
| 7 批量验证 | `run_isaac_batch.py --traj-root E/isaac_traj` | 每条约 35 秒 → `isaac_traj/<名>/isaac_sim/{report.json,video.mp4}` + `isaac_summary.json` |
| 8 收尾 | `keep_isaac_success.py --exp-dir E` | 成功项收进 `isaac_succ/`；失败轨迹删除、失败 npy 移 `grasp_data_isaac_failed/` |

**第 6 步必须 `--mode local`**——本机没有 streaming kit。

可选后处理：

- `promote_templates.py --exp-dir E --prefix isaac_N` —— 把成功抓取回灌成模板，后续 `TMPL=isaac_N_xxx` 生成同风格
- `render_grasps.py --exp-dir E --data grasp_data` —— 渲染检查（真实桌面自动染半透明粉色）

---

## 4. 数据格式

抓取 npy（`np.load(f, allow_pickle=True).item()`）关键字段：

- `grasp_qpos (1,29)` = `[x, y, z, qw, qx, qy, qz, 22 个关节角]`，**物体规范系**
  （物体质心在原点、z-up 静置、桌面在 z=zmin 水平面），四元数 wxyz
- `pregrasp_qpos (K,29)` —— 张开→合拢的接近序列（第 0 帧最张开）
- `squeeze_qpos (1,29)` —— 施力目标

关节顺序 = ocir `sharpa_wave_right.yml` 的 `joint_order`（thumb 5 + index/middle/ring 各 4
+ pinky 5），**与 MJCF / Isaac 恒等映射**。

**映射回真实场景**：

```
T_场景_手 = T_场景_物体输入系 × Trans(com_offset) × Rot(canonical_from_input_rot_wxyz)⁻¹ × T_规范系_手
```

两个补偿量存在 `assets/object/custom/processed_data/<名>/info/simplified.json`。
物体在场景中绕竖直轴旋转 / 平移时，抓取直接跟随变换。

**Isaac 成功判定**（`report.json → metrics`）：`grasp_success` = carry 段连续 ≥5 帧
抬升 ≥2cm 且结束时未掉落。其余指标 `max_lift_m`、`max_consecutive_lifted_steps`、
`object_dropped`。

---

## 5. 关键约定与教训（改动前必读）

1. **规范系 = 静置姿态。** 导入时物体被旋转到"躺 / 立在桌上"的真实姿态。`--auto-up`
   读 mesh 同目录的 `world_fused.npz`，用视频位姿在 trimesh 稳定姿态（p ≥ 0.02）中消歧
   ——**不要只信最大概率稳定姿态**。教训：pp0 瓶子在视频里立着，最大概率姿态却是躺倒。
   选择与视频差 >45° 会打 WARNING，此时人工确认 `--up`。
2. **三种平面约束缺一不可。** init 用 virtual_plane 过滤（`hard_plane=True` 硬约束，
   修过 quota 补齐漏洞）；grasp 用真实 plane geom + 小 margin；eval / Isaac 用真实桌面零 margin。
   历史 bug：无桌 eval 时物体自由落体，系统性偏爱从桌下抓的姿态。
3. **MuJoCo eval（`op=eval`）不在主管线里。** 它的 15° 转动阈值对扁平 / 环状物过严，
   只作快速预估（`op.rot_thre=45` 放宽）。**最终裁判是 Isaac。**
4. 重跑某个 op 前要删 `output/<exp>/log/<op>_*/`（skip_done 机制）。
   覆盖 worker 数用顶层 `n_worker=1`，不是 `op.n_worker`。
5. warp 的 `CUDA error 36` 无害（走 CPU 回退，torch CUDA 正常）。
6. 命令必须在 Dexonomy 仓库根目录、`dexonomy` conda 环境下执行。

**用户已确认的硬约束**：不要"虎口朝下"的抓取——管线 step 4.5 自动过滤。

---

## 6. 已处理物体（`basic_pick_place` 系列）

| 物体 | 说明 | 自然抓取成功数 |
|---|---|---|
| pp0 | 瓶子（立） | 35 |
| pp1 | — | 1 |
| pp2 | 环状 | 4 |
| pp3 | — | 16 |
| pp5 | 环状（螺母状） | 9 |

待处理：pp4、pp6~pp19、pp55。

---

## 7. 与 Step4 的接口

Step4 的 `tasks/pregrasp/screen_prior.py` 对这些候选做四道筛（Gate 0/1/1b/2），
选出的那个存成 `tasks/pregrasp/priors/<clip>_candidates/<名>.npz` 供训练用
（当前冠军配方用的是 `Grasp3_candidates/8_5.npz`，`--prior_yaw 215`）。
筛选口径见 Step4 分支的 `docs/GRASPPOSE_SCREENING.md`。

## 8. 手部资产怎么重建

`scripts/grasp_synthesis/dexonomy/sharpa_wave/build_assets.py` —— URDF → MJCF 全自动重建。
换手或改 URDF 后重跑它，再用 `render_check.py` 目视确认，`search_thumb.py` 标定拇指。
`retarget_shadow_templates.py` 把上游 Shadow 手的 38 个分类学标注重定向到 SharpaWave
（`raw_anno/_shadow_origin.json` 记着来源）。
