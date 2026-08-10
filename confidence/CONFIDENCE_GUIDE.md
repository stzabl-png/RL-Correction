# 物体轨迹可信度链路指南

> 2026-08-10 定稿。回答三个问题：**分数怎么打的？smooth/filtered/σ 怎么用？哪些数据能用？**
> 立论（全项目的锚）：我们不修正视频先验的错误——我们**判定哪些信息可信**；
> 高可信的给 RL 当参考，不可信的显式放开让 RL 自己探索。宁缺毋假。

## 0. 一图流

```
重建 take (world_fused.npz + masks/ + objects/*.obj)          原视频 mp4
        │                                                        │
        ▼                                                        ▼
  pose_audit.py  ◄──────── cc_*.json ◄──────── cotracker_consistency.py
  9判据逐帧打分              CT 第二观察员(纯2D比对, 不解PnP)
        │
        ▼
  pose_audit.json   逐帧 conf_pos / conf_rot (0~100) + rot_refuted 标志
        │
        ├──► take_manifest.py ──► TAKE_MANIFEST.json   take级裁决(下游取数入口)
        │
        ▼
  rts_smoother.py   conf→测量噪声R; 位置KF+RTS, 旋转MEKF前后向
        │
        ▼
  rts_<take>.npz    smooth(非因果) + filtered(因果) + σ_reported + refuted剔除
```

数据根目录：`/home/lyh/Project/Reconstruct_and_Retarget/Data/VideoPrior/poseqa/`

```
pose_audit.json      全库逐帧打分 (33 takes)
TAKE_MANIFEST.json   ★下游从这里取数据, 不要自己扫目录
heldout_report.json  29 条未调参 take 的验收成绩
rts/                 rts_<take>.npz + 摘要 json (+验收视频 rts_*.mp4)
cc/                  CT 一致性 cc_<take>.json (+可视化 cc_*.mp4)
cc_early/            个别 take 的前 60 帧 CT 可视化(GPU 受限时的替代)
confviz/             可信度叠加视频 conf_<take>.mp4 (轮廓按分上色+双曲线)
calib/               人眼校准表(高/低可信帧对照图) —— 阈值的出处证据
```

## 1. 评分标准（pose_audit.py）

每帧输出**两路**分数，绝不混成一个：`conf_pos`（位置）与 `conf_rot`（旋转）。
苹果案例是原因：位置对(90)旋转错(0)，单一总分要么高估要么低估。
（npz 里也存了 `conf = 0.6·pos + 0.4·rot`，但 0.6/0.4 是全链路唯一没标定的数——
**RL 侧永远直接用两个分量，别用合成分。**）

### conf_pos 的构成

```
s_pos  = 100·exp(−(d_cent_norm/0.35)²)      质心偏移 —— 主判据(唯一经人眼验证全可靠)
s_cov  = explained 在 [0.40, 0.85] 线性拉分   只抓"完全没盖住", 不当硬线
s_img  = (s_pos + w·s_cov)/(1+w),  w=1−occl  遮挡越重, explained 权重越低
conf_pos = min(s_img, 100·(1−occl), scale_cap)   证据量封顶:看不见就不能说可信
           × 0.30 若瞬移(>50mm/25°每帧)
           × 0.40 若滞后(mask动>8px而投影动<2px)
           × 0.85 若穿透(软信号,支撑面是估的)
           × 0.6/0.3 若 CT flow_err > 5px / 10px
```

关键设计：**explained = |mask∩proj|/|mask| 而非 IoU**。v17A/SAM2 的 mask 是 modal
（只含可见部分），mesh 投影是 amodal（完整轮廓），IoU 在遮挡下必然崩，explained 不会。
检错必须用**原始尺度** mesh 投影——偏大的 mesh 会把误差盖住（27 号 46-114 帧曾借此假及格）。

### conf_rot 的构成

```
CT 在场(该帧有≥6个可见追踪点):
    conf_rot = conf_pos × min(tr_r_err, tr_spread)     纹理可观测, CT 裁决
    rot_refuted = trust < 0.20                          ★被反驳 ≠ 低分
CT 不在场:
    conf_rot = conf_pos × rot_factor                    退回轮廓可观测性
```

`rot_factor` 来自网格几何：绕各主轴转 30° 后投影轮廓 IoU 掉多少。掉得少 = 转了也看不出
= **不可观测**。实测：苹果 0.036 / 瓶长轴 0.019 / 手机 0.281。低于 0.08 的轴直接判 0。

### 三种"低分"的语义不同，下游处理必须不同

| 语义 | 例子 | 下游处理 |
|---|---|---|
| **不可信**（证据差） | 遮挡重、质心偏 | σ 放大（弱测量），不剔除 |
| **不可观测**（结构性） | 苹果绕轴转、瓶绕长轴转 | 该轴永久自由，RL 自己探索 |
| **被反驳**（有反证） | CT trust<0.20 | 旋转测量**剔除**——一段"一致地错"的测量会让滤波器自信地跟着错 |

### 判据演化史（别走回头路，详见 pose_audit.py 顶部注释）

删「静置长轴应竖直」（950 次假阳性——手机本来就平躺）；删「离稳定姿态距离」（物理可行≠正确）；
删 unsupportable、降级 penetrating（人眼校准确认假阳性）；explained 降为辅助（遮挡下判别力
单调衰减）。**阈值 v3 已冻结**：在 4 条调参 take（bpp/13、bpp/6、27_scene、bpp/11）上迭代过
三轮，继续调 = 过拟合背答案。改阈值必须走新的 held-out。

### CT 第二观察员的纪律（cotracker_consistency.py）

- **不解 PnP**。三版参考帧策略互有胜负 = 方法对参考帧过敏感；纯 2D 比对（FP 位姿反投影的
  预测点 vs CoTracker 追踪点）把问题整个消掉。
- 输入必须 **RGB + 原分辨率**（官方规格；曾喂 BGR+手动降采样，修正后 57.7°→4.4°）。
- 种子按参考帧**分组**评估，不混池（坏参考帧的种子会把误差抹平）。
- **ct_t_err 禁止用于罚 conf_pos**（坏种子污染平移分量；帧级定位只用 flow_err）。
- CT 资格门：FP 在静止段稳(<0.5°/帧)而 CT 自己 r_err>0.08 → 整条取消 CT 资格。
  全库开火 0 次，留作换数据集的保险。

## 2. 平滑层（rts_smoother.py）

conf 是**对原始轨迹的逐帧证据分**，单向喂给平滑器定测量噪声，自己不被平滑：

```
σ_meas_pos(conf) = 5mm + 95mm·(1−conf/100)²      conf 100→5mm, 0→10cm
σ_meas_rot(conf) = 2°  + 10°·(1−conf/100)²       conf 100→2°,  0→12°
```

**永不因低 conf 剔除测量**（"没证据"="弱测量"≠"无测量"；v1 剔除曾导致好测量段自由漂移
explained 0.883→0.489）。唯一的剔除是 `rot_refuted` 帧的旋转测量。

位置：线性 KF [p,v] + 精确 RTS 后向。旋转：MEKF（初始姿态先验 45°）前后向切空间融合。
三个正确性机制（都是修过的 bug，参数别乱动）：

- `CHI2_GATE=16.27` + `RECOVER_AFTER=5`：连续 5 帧被 NIS 门拒收 → 错的是滤波器不是测量，
  膨胀协方差强制收敛回测量（修"门控锁死"：2_scene 曾把正确测量拒了一整段，倾角 90° 还自报 σ 2.4°）。
- `VEL_DECAY=0.8`：无测量/被拒帧速度衰减——桌面物体不会自己持续转/滑
  （修"证据真空脑补"：bpp/2 曾匀速外推出 8.6°/帧的虚构旋转）。
- **σ 报告封底**：`σ_reported = max(σ_滤波, σ_meas(conf)/2)`。滤波器内部 σ 在偏差段会撒谎
  （bpp/2 错着转还自报 2.3°，封底后 7.2°）。**对外只认 reported。**

### rts_<take>.npz 字段

| 字段 | 含义 |
|---|---|
| `object_ob_in_world_smooth` | ★非因果 RTS 输出（前向 Kalman + 后向修正），默认参考轨迹 |
| `object_ob_in_cam_smooth` | 同上，相机系 |
| `object_ob_in_world_filtered` | 因果 Kalman 前向输出（零预判） |
| `sigma_pos_reported_m` / `sigma_rot_reported_deg` | ★对外唯一认可的 σ（conf 封底后） |
| `sigma_pos_m` / `sigma_rot_deg` | RTS 内部 σ（诊断用，勿直接消费） |
| `sigma_*_filtered` | filtered 版内部 σ（诊断用） |
| `innov_pos_nis` / `innov_rot_nis` | 逐帧 innovation（>16.27 = 测量与预测打架，瞬移探针） |
| `conf_pos` / `conf_rot` | 原样透传的证据分 |
| `measurement_used` | 该帧测量是否被采纳（门拒/refuted 为 False） |

### smooth vs filtered 怎么选

| 场景 | 用哪个 | 原因 |
|---|---|---|
| RL 参考轨迹、RSI 锚点、回放 | `smooth` | 遮挡段插值质量好，两端锚定中间放开（P4） |
| 时序敏感（接触时刻对齐、事件检测、在线场景） | `filtered` | smooth 非因果，动作起始可提前 1~2 帧；filtered 零预判 |

RTS = 前向 Kalman **filter** + 后向修正（smoother）。非因果是**设计选择**不是错误；
预判超过 1~2 帧才是 bug（那是门控锁死，已修）。

## 3. RL 消费规则

**入口：先查 `TAKE_MANIFEST.json`，只用 `status=="active"` 的 take。** 然后逐帧：

```
位置 (σ = sigma_pos_reported_m):
  σ ≤ 10mm        硬用: 锚点 / RSI / 强 tracking reward
  10 ~ 30mm       软用: 死区 reward(在 σ 球内不罚)
  > 30mm          放开: 不给参考, RL 自由探索
旋转 (双门, 缺一不可):
  conf_rot ≥ 30 且 σ_rot_reported ≤ 5°   → 可用
  否则                                     → 该帧旋转不给参考
  manifest.rotation_usable == false        → 整条旋转通道弃用(宁缺毋假)
  manifest.rotation_free_axes 里的轴       → 永久自由(σ_rot 是标量, 装不下对称轴)
```

阈值出处：RL 实测"2.5cm 参考噪声训死 / 12mm 救活"。**当超参暴露**，待消融，不是圣旨。

为什么旋转要双门：σ 继承 Kalman 对**偏差**的盲区（测量一致地错时 σ 照样收缩），
conf_rot 是独立证据分，两者互补。为什么整条弃用而不是逐帧：B 类验收（轮廓 explained）
对**近对称物体大角度转错**全瞎——2_bottle 底朝相机投影成圆、3 号瓶 180° 颠倒，B 都"及格"。
平滑会把"吵闹的错"变成"安静的错"，迷惑性更大，所以低分段宁可不给。

## 4. take 级裁决（take_manifest.py）

三层：① 硬排除——**透明物体整条过滤**（用户规则：FP 在透明物体上系统性失灵，且该失效对
轮廓验收不可见，必须前置排除）；② 同视频同物体多次重建→按 conf 择优其余落选
（2_bottle 输给 2_scene：同一条 2.mp4 两次独立重建，一次 FP 开头倾角 85~88° 一次基本正确，
conf 排序与人眼一致）；③ active 内部分级（见文件头注释）。

当前战果：33 takes → **29 active（12 条旋转可用），2 透明淘汰，2 落选**。

## 5. 验收协议（改任何东西都要重跑）

- **A 静止段真值**：无接触帧物体必静止——全链路唯一有真值的验收，红线。
- **B 好段不变差**：高可信段 explained 平滑后 ≥ 原始 − 0.03。注意 B 对对称物体旋转错误**失明**（见上），及格≠旋转对。
- **C innovation 尖峰**：应命中已知瞬移帧（bpp/11 的 f49 是基准）。
- **held-out 纪律**：29 条未调参 take，口径与调参期一字不改。当前成绩 A 27/29 (93%)，
  双过 20/29 (69%)。剩 7 个失败清一色情形 C（双证人都不可靠 + refuted 真空段有真实运动），
  是结构问题不是阈值问题。

驱动脚本：`heldout_eval.py --out <poseqa> --stage {ct|audit|rts|all}`。

## 6. 新数据接入 SOP

**正常情况下不用手跑**：2026-08-10 起 confidence 已是重建 pipeline 的第 9 步
（fuse 之后自动执行，接入方案见 `PIPELINE_INTEGRATION.md`），`reconstruct.sh` 跑完
一条视频就自带评分+平滑+manifest 更新。下面的手动流程只用于离线补跑/调试：

```bash
PY=/home/lyh/anaconda3/envs/hawor/bin/python
P=/home/lyh/Project/Reconstruct_and_Retarget/Data/VideoPrior/poseqa
S=<take目录>   # 含 world_fused.npz + masks/ + objects/*/*.obj
$PY cotracker_consistency.py --scene $S --video <原视频> --out $P/cc \
    --audit $P/pose_audit.json --no-viz          # GPU ~1-2 分钟
$PY pose_audit.py --root <ReconstructOutput> --include egodex \
    --out $P --ct-dir $P/cc                       # 全库重打分
$PY rts_smoother.py --scene $S --audit $P/pose_audit.json --out $P/rts
$PY take_manifest.py --audit $P/pose_audit.json --out $P/TAKE_MANIFEST.json
# 可视化(可选): conf_viz.py / cotracker_consistency.py 不加 --no-viz
```

接入前先过两个人工判定：物体是否透明（是→不进库）；同视频是否已有 take（是→manifest
的 DESELECTED 规则里裁一次）。

## 7. 工具清单

| 脚本 | 作用 |
|---|---|
| `pose_audit.py` | 9 判据逐帧打分 → pose_audit.json |
| `cotracker_consistency.py` | CT 第二观察员（2D 比对 + 可视化 cc_*.mp4） |
| `rts_smoother.py` | conf→R 的 KF/RTS/MEKF，出双版本轨迹 + σ |
| `take_manifest.py` | take 级裁决 → TAKE_MANIFEST.json |
| `heldout_eval.py` | held-out 三阶段驱动 + 验收汇总 |
| `conf_viz.py` | 可信度叠加视频 |
| `pose_calib_sheet.py` | 人眼校准表（含 `resolve_video()`：take→原视频路径） |
| `pose_projection_check.py` | 单条投影诊断（尺度拟合出处） |
| `view_object_traj.py` | Isaac 里回放物体轨迹 |
| `cotracker_rotation.py` | 旧 PnP 路线（**弃用**，仅 `static_check`/网格工具被复用） |

## 8. 已知局限（诚实清单）

1. B 类验收对近对称物体旋转错误失明 → held-out 69% 对旋转类失败近视，含金量打折。
2. 情形 C（FP、CT 双双失灵 + 真空段真实运动）无解，7/29。候选方向：CT 反驳时用 CT 自身
   相对旋转顶上当替代测量（信息在点轨迹里，现在整个扔掉）。金属瓶家族 CT 实测可靠
   （2 号瓶静止段漂移 0.2°），此方向对该家族可行——可选研究，未排期。
3. σ_rot 是标量，表达不了"绕对称轴自由、其他轴可信"——靠 manifest 的 `rotation_free_axes` 补。
4. 支撑面高度是 EM 估计出来的，穿透判据因此只是软信号。
5. conf 的 0.6/0.4 合成权重无标定——别用合成分。
