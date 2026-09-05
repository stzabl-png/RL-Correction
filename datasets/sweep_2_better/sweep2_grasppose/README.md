# sweep_2（扫地任务）双手 GraspPose 交付包

生成：Dexonomy（SharpaWave 定制版）合成管线，2026-08-19/20。
物体来自 egodex `sweep_dustpan/2` 视频重建；左手持簸箕、右手持扫帚（手别绑定来自
contact 自动判定，非人工指定）。

## 目录结构（两个物体同构）

```
sweep2_broom_right/          扫帚 × SharpaWave v2 右手
sweep2_dustpan_left/         簸箕 × SharpaWave v2 左手
  grasp_data/                ★功能池：虎口对功能头的候选（推荐从这里选）
  all_candidates/            全量候选池
  region_rank.json           全量排序表（见下）
  render/                    排名 Top-6 渲染图（双视角）
demo_replay/                 人手示范逐字回放渲染（0° 基准参照，判断"像不像人"跟它比）
object_info/                 两物体的 info json（含 canonical_from_input_rot_wxyz）
```

- 扫帚：全量 3331 / 功能池 117（示范夹角 <60°，最佳 23.5°）
- 簸箕：全量 216 / 功能池 50（<90°；把手贴地物理下限约 50°，两轮独立合成一致）

## 候选 npy 字段（np.load(f, allow_pickle=True).item()）

| 键 | 含义 |
|---|---|
| grasp_qpos | (1,29)=[腕位置3 + 腕四元数 wxyz4 + 22 手指关节]，**规范系**（物体静置桌面，Z 竖直向上） |
| squeeze_qpos / pregrasp_qpos | 加压构型 / 6 步接近轨迹（同系同格式） |
| obj_cpn_w / hand_cpn_w | 物体侧/手侧接触点+法向（规范系） |
| ho_c | MuJoCo 落盘时真实接触（bn1=接触 body 名，pos/dist/wrench） |
| ext_center | 物体质心（规范系） |
| in_region_frac | 接触落在视频接触带内的比例（1.0=全在带内） |
| demo_angle_deg | 候选腕姿与**人手示范腕姿**的旋转夹角[deg]，越小越像人 |
| func_align | 几何代理分（仅参考，语义标定见管线文档） |

## region_rank.json

按 demo_angle_deg 升序的全量排序：`name / demo_angle_deg / in_region_frac / functional`。
选 prior 建议：从头部取（demo 夹角小 + in_region 高），再按力分配均匀度复核
（ho_c.wrench 逐指求和，避免单指独吞 >70% 的候选）。

## 坐标系与换算

- 规范系 = 物体静置桌面姿态（簸箕=视频开头静置；扫帚=释放后静置，因其开场已在手中——
  "手持开局取释放窗"公约）。渲染图中坐标轴 X红 Y绿 Z蓝（右手系，原点在质心）。
- 规范系 ↔ 原始 mesh 系：`object_info/*.json` 的 `canonical_from_input_rot_wxyz`。
- 原始 mesh：`Reconstruct_and_Retarget/results/sweep_2_better/objects/object_{0,1}/
  object_mesh_scaled_final.obj`（object_0=簸箕 15.8×2.9×21.6cm，object_1=扫帚 7.8×11.6×29.2cm，
  真实尺度、水密）。

## 口径声明

- 候选已过：桌面净距、虎口朝向、区域（视频接触带）排序、功能朝向（示范夹角）。
- **未过 Isaac/RL 平台验证**——接触标定在 DP 指节壳（Dexonomy 口径），elastomer 实际
  吃力与否以下游零动作回放为准。
- 已知注意项：簸箕把手贴地（柄下净空 ~2cm），臂可达性需按各自机器人自查。
