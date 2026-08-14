# RL-Correction 仓库整理规划 (2026-08-07)

目标形态 = 一条 **Data Engine**:输入 video,输出仿真验证过的人手-物体交互轨迹;
把"重建出来不可用"的数据修正为可用。

```
1 感知    video → 自动接触帧 + 自动双手/物体 mask
2 选帧    交互 episode / 物体实例 / 最佳重建帧
3 重建    深度+相机 → 物体 mesh → 人手轨迹 → 物体位姿 → 统一坐标系
4 抓取    mesh(重建 or retrieval) → GraspPose 合成 + 优化
5 修正    重建数据 + prior → RL 残差修正
6 输出    仿真验证过的交互轨迹 (D_verified → 去重 → D_high-quality)
```

## 一、现状盘点(基于 8 个分支实际内容,非 README 声称)

| 分支 | 覆盖步骤 | 状态 |
|---|---|---|
| `main` | 总纲 | 只有 README(8 步公式框架) |
| `Step1_DataInput` | 数据契约 | 只有 README,契约已过时(见 §五) |
| `Step2_NoisyRecon` | 1(半) + 2(隐式) + 3 | 重建全套;**mask/phase 默认仍人工** |
| `agent/v17a-static-reconstruction-adapter` | 1 的接线补丁 | Step2 的**严格超集**(基点=Step2 HEAD,+1 提交) |
| `Step3_GraspPose_Optimization` | 4 | Dexonomy(默认)+ BODex(备选)两套并存 |
| `Step4_RL_Correction` | 5 + 6(部分) | 冠军存档稳定版 |
| `RL_scene_ready_to_use` | 4(retrieval)+ 5(摆放) | 职责混杂,摆放部分已被本地超越 |
| `task5/pour-bimanual` | 5/6 任务实例 | Kailang 负责,本规划不动 |

**三个真实缺口**:步骤 2 没有选帧器;步骤 4 没有 retrieval mesh(开发中);步骤 6 没有任何 Part。

---

## 二、步骤 1-2:自动化闭环(最高优先级)

### 关键发现:v17A 分支不是"另一版自动 mask",是 Step2 缺的那根线

Step2 里**已经有**三块自动化零件,但一块都没接上:
- `experimental/hoi_detr_v17a/` — v17A 自动交互物体分割本体(README 明写"未接入主流程")
- `phase/detect.py` — 2D mask 邻接的自动接触检测,天然分左右手
- `phase/comotion_gate.py` — 手物协同运动闸,判「真抓 vs 靠近/搁着」

v17A 分支新增的正是**导入适配器** `recon_kailang/v17_mask_adapter/import_v17a_masks.py`:
把 v17A 的 manifest + 逐帧 mask 写进 `sam2_object` 的既有布局(含 `label_prompt.json`
和完成标记),让下游一步都不用改。外加三个原位补丁(sam3_hands 支持本地 checkpoint、
sam3d_scale 固定 visible-surface 尺度、.gitattributes 修 LF)。

**结论:合回 Step2 是快进合并,零冲突,合完自动化零件就齐了。**

### 人工标注只有两处,都已有自动替代

| 人工入口 | 产物 | 自动替代 | 接线状态 |
|---|---|---|---|
| `label.sh` / `--web` 点标物体 | `label_prompt.json` | v17A → `import_v17a_masks.py` | 适配器就绪,**缺上游选帧器** |
| `tools/annotate_grasp_frames.py` | `grasp_annotation.json` | `phase/detect.py` → `contact_auto.json` | ✅ 已就绪,`provider.py` 默认 `prefer="auto"`,**文件一存在就自动切换** |

双手 mask(`sam3_hands`/SAM3)本来就是自动的,不涉及人工。

### 唯一要新写的代码:选帧器(= 你的步骤 2)

适配器自己声明 `selection_policy: "upstream_explicit"`,三个参数必须外部给:
`--manifest`(哪个 episode)、`--source-object-id`(哪个物体实例)、`--reconstruction-frame`(哪一帧重建)。

**这个上游选择器目前不存在**,是全自动化的唯一硬缺口。输入已齐备(v17A 输出带固定实例 ID
的逐帧 mask + interaction 关键帧),判据建议:接触前的静置帧 + mask 面积/完整度最大 +
无手部遮挡 + 物体实例在该 episode 内 ID 连续。

### 行动项

- **A1** `agent/v17a-...` 快进合回 `Step2_NoisyRecon`,删除该分支
- **A2** 新写选帧器 `recon_kailang/frame_selector/`,产出 `(episode, object_id, recon_frame)`
- **A3** `reconstruct.sh` 加 `--auto` 路径:v17A → import → 跳过 `sam2_object` → 其余步骤照旧;
  重建后跑 `phase/detect` 生成 `contact_auto.json`
- **A4** 人工入口**降级为 fallback,不删**。理由:v17A 的 hand-object link 是"关联/靠近"
  而非"抓取",实测有假阳性(手搁旁边也高置信),`comotion_gate` 就是为此写的闸门。
  自动路径跑通并在多条视频上对拍后,再决定是否彻底移除。

---

## 三、步骤 4:GraspPose 收敛到 Dexonomy 单路线

### 删除范围

```
src/ocir/grasp_synthesis/anchored_bodex/          人手锚定 BODex
src/ocir/grasp_synthesis/bodex_curobo_v2/         纯 BODex 内核(已冻结)
src/ocir/grasp_synthesis/synthesize_sharpa_bodex_curobo_v2.py
docs/anchored_bodex.md  docs/bodex_curobo_v2.md
assets/robots/hands/sharpa_wave/grasp_synthesis/bodex/**
assets/robots/hands/sharpa_wave/usd/right/bodex_reference/**
```

### 两个必须先确认的连带影响

1. **cuRobo ≠ BODex,不要一起删**。`third_party/curobo` 子模块同时被 Dexonomy 的轨迹合成
   使用;`assets/.../robot_configs/curobo_*.yml`、`collision/curobo/*.yml` 也是 cuRobo 侧配置。
   删 BODex 合成路线,保留 cuRobo。
2. **两个遗留 clip 会失效**:`clips.py` 的 `pp0_human` / `pp55_human` 走
   `_td_ocir(...)` → `{base}/grasp_pose/grasp_pose_6.json` + `{base}/curobo_traj`(设定 A 遗留)。
   活跃路径(`tasks/pregrasp` + 冠军存档 `Approach_Pick_冠军seed42`)走的是
   `ref_builders/replay_grasp.py`,**不依赖这条链**。删前确认这两个 clip 是否还要复现;
   若不要,连同 `ref_builders/ocir.py` 一并退役。

- **B1** 删上述 BODex 合成路线;**B2** 保留 cuRobo;**B3** 确认后退役 `_td_ocir` 两个 clip
- **B4** `affordance_seed.py` / `object_surface.py` / `clearance.py` 属约束层,Dexonomy 路线
  暂未使用但与 affordance 先验相关 —— **单独评估,不随 BODex 删**

---

## 四、步骤 5:Step4 / RL_scene 去重

RL_scene 的本意是 **retrieval mesh 替换**(把重建效果差的 mesh、尤其 articulated object
换成检索到的精细 mesh),开发中;它顺带做了仿真场景摆放,而摆放已被本地终版超越。

- **C1** RL_scene **拆两半**:
  - retrieval mesh 部分 → 独立 Part(建议 `Step3b_MeshRetrieval`),定位是**步骤 4 的 mesh 来源二**
    (来源一 = SAM3D 重建 mesh)
  - 场景摆放部分 → **作废**,以本地 `Step4_RL_Correction` 为唯一权威
- **C2** ⚠ **本地 25 个文件未提交**,其中就是摆放的终版(palm 旧锚删除、物体听手下沉 builder):
  `replay_grasp.py`(-83 行)、`clips.py`(+185)、`DESIGN_LOOP.md`(+259)、`load_replay.py`(+8)。
  远端 RL_scene 仍保留"掌心对准 affordance 区(旧锚,会破坏相机对齐)"分支。
  **先提交推送,再谈去重** —— 否则同事拉到的是带旧锚的错版本。
  (`RECON_TO_ENV_ALIGNMENT.md`、`place_camera.py`、`recon_pose.py` 两边已一致,无需处理。)
- **C3** `task5/pour-bimanual` 不动。

---

## 五、数据契约:统一到 RL_scene 实际格式

现存**三套**并行布局,以 RL_scene 实际在用的为准:

**权威格式**(`datasets/<source>/<clip>/`):
```
meta.json                     schema_version + 每个部件: mesh/mass_kg/friction/hand/
                              interaction_frame/placement_frame + 任务专用参数
reconstruction/<part>.obj     + 同名 .obj.json
retarget/replay_world.npz     轨迹(joints/verts/obj_pose/phase/valid/confidence)
retarget/ref_qpos.npz         机器人手参考构型
VALIDATION.md  SHA256SUMS  THIRD_PARTY_LICENSES/
```

**废弃**:
- Step1 README 的 HF 契约(`human_demo.npz` / `affordance.npz` / `object_coacd_parts.npz`)—— 改写 README 指向权威格式
- `_td_ocir` 布局(`ocir_sequence/object.obj` + `grasp_pose/*.json` + `curobo_traj`)—— 随 §三 B3 退役

- **D1** 改写 `Step1_DataInput` README 为权威契约
- **D2** **加 schema 校验器**。实证:`clips.py:72` 记录 `water_bottle_twist_static` 的
  `meta.json`/`VALIDATION.md` 把左右手记反了(视频是左手拧盖右手扶瓶),靠代码注释绕过。
  没有校验,这类错误只能靠人肉发现。校验项:字段完备、mesh 文件存在、
  帧号在 `[0, num_frames)`、`hand` 与 `replay_world.npz` 的 `valid_*`/`phase_*` 一致。

---

## 六、步骤 6:新建 `Step5_SuccessTracker`(**尚未设计**)

`main` README 的第 5-8 步(Isaac 验证 → `D_verified` → 去重 → `D_high-quality`)**没有任何分支对应**。
验证逻辑现散在两处:Step3 的 GraspPose PhysX 验证(lift / contact 双模式)、Step4 的 RL eval。

**定位**:根据 **RL 的输出**判定「训练是否正确、动作是否合理」,而不只是仿真里跑没跑通。
这是本规划中唯一**还没有设计**的 Part,待 Step1-4 整理完后单独设计。

- **E1** 新建 `Step5_SuccessTracker`:统一判定入口(轨迹级,非单抓取级)+ `D_verified` 产出契约
  + 按几何/接触模式/抓姿/轨迹相似度/结果多样性去重
- **E2** 判定维度待设计,已知素材:Step3 的 contact 模式思路(不强求蛮力提起,判接触点合理性)、
  Step4 的 RL eval 成功率、与原视频意图的 fidelity

---

## 七、执行顺序(依赖关系)

```
0. [立刻] C2 提交推送本地终版摆放改动          ← 防丢失,且是后续去重的基准
1. A1 v17a 快进合回 Step2                     ← 零风险
2. D1+D2 契约统一 + 校验器                    ← 后续所有 Part 的公共地基
3. A2 选帧器 → A3 接线 → 全自动跑通一条视频     ← 步骤 1-2 闭环
4. B1-B3 删 BODex(先确认遗留 clip 去留)
5. C1 RL_scene 拆分
6. E1 新建 Step5_SimVerified
```

风险最高的是第 4 步(删代码),放在契约和自动化稳定之后;第 0 步无风险且防数据丢失,先做。
