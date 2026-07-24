# OCIR 使用手册 / Handbook

> 每次开新终端先读这个文件。命令都从**仓库根目录**跑(下文记作 `$REPO`)。
>
> ```bash
> export REPO=/path/to/GraspPose_Optimization      # 你 clone 的位置
> export RR_ROOT=/path/to/Reconstruct_and_Retarget  # Step2 重建/retarget 输出(见 §6)
> ```
> 命令/路径/参数是英文原样;说明用中文。

---

## 0. 一分钟上手(最常用的一条命令)

**从一条已重建+retarget 的 take,一条命令生成 GraspPose 并用 cuRobo 驱动出"接近→抓取→抬起"验证视频:**

```bash
cd $REPO
export OCIR_DATA_ROOT=$PWD/data
scripts/run_grasp_synthesis_conda.sh scripts/grasp_traj/reconstructed_grasp_video.py \
  --recon-dir    $RR_ROOT/Output/ReconstructOutput/egodex/part2/<...>/<take> \
  --retarget-dir $RR_ROOT/Output/RetargetOutput/egodex/part2/<...>/<take> \
  --sequence-id  <名字> \
  --anchored --squeeze-overclose 0.2
```

它会自动串起:抽简网格 → 预测 affordance → 合成+优化 GraspPose → 选出"过桌面"的最优抓取 → 生成轨迹(重建初始手姿 + cuRobo 接近 + 抓紧 + 抬10cm)→ Isaac 物理仿真出视频。

**输出**(`<名字>` = sequence-id):
- 验证视频:`data/testing/grasp_traj/<名字>/isaac_sim/video.mp4`
- GraspPose:`data/testing/grasp_synthesis/<名字>/{grasp_pose_N,failed_grasp_N}.json`
- 手部轨迹:`data/testing/grasp_traj/<名字>/trajectory.npz`

跑一条大概 8–12 分钟(合成 500 迭代 + Isaac 启动 + 仿真)。

---

## 1. 环境 / Environments

| 用途 | conda env | 说明 |
|---|---|---|
| 合成 + 轨迹 + Isaac 仿真 | **`env_isaacsim`** | Python 3.11、torch 2.7.0+cu128、cuRobo v2、coal、coacd;Isaac Sim 5.1 通过 activate.d 钩子挂进来 |
| Affordance 模型推理 | **`deximit`** | 需要 sonata + spconv-cu128 + torch_scatter;由 affordance 步骤**子进程**调用,不用手动进 |

**永远用这两个包装脚本跑,别直接调 python**(否则 Isaac 的 numpy/torch 会污染,见 §9):
- `scripts/run_grasp_synthesis_conda.sh <script.py> [args]` — 在 env_isaacsim 里跑,自动设 PYTHONPATH。
- `scripts/run_isaacsim_conda.sh <script.py> [args]` — 同上 + 自动接受 Isaac EULA(启物理/渲染用这个)。

**每次都先 `export OCIR_DATA_ROOT=$PWD/data`**(否则默认指向一个不存在的路径)。

覆盖 affordance 环境/checkpoint 的环境变量:`OCIR_AFFORDANCE_ENV`(默认 deximit)、`OCIR_AFFORDANCE_CKPT`(默认 `assets/affordance/model.pt`)。

---

## 2. 管线全貌

```
重建 take(物体网格 + replay_world.npz 里的人手/物体轨迹)
        │
        ├─ [1] Affordance 预测(deximit 子进程)→ 期望抓取区域热图
        │
        ├─ [2] 合成 GraspPose(env_isaacsim, cuRobo v2/BODex)
        │       ├─ anchored:  用人手位姿锚定(修虎口朝下),需要人手接触物体
        │       └─ affordance: 只用区域播种(gappy take 的回退),已做上表面过滤防穿桌
        │
        ├─ [3] Stage A 轨迹(cuRobo 驱动):从重建初始手姿 → 接近 → 闭合 → 抓紧 → 抬升
        │
        └─ [4] Stage B 物理仿真(Isaac Sim 5.1 + PhysX)→ video.mp4 + 抬升/掉落指标
```

`reconstructed_grasp_video.py` = 把 [1][2][3][4] 全自动串起来的编排器。

---

## 3. 常用工作流(逐条命令)

> 下面都假设已经 `cd` 到仓库根 + `export OCIR_DATA_ROOT=$PWD/data`。

### A. 一条命令全流程(见 §0)
`scripts/grasp_traj/reconstructed_grasp_video.py` — 参数见 §4。

### B. 只合成 GraspPose(不出轨迹视频)

**affordance 播种(区域,无需人手接触;有虎口朝下风险):**
```bash
scripts/run_grasp_synthesis_conda.sh scripts/grasp_synthesis/synthesize_affordance_seeded.py \
  --sequence-dir data/testing/sequences/<名字> \
  --out-dir      data/testing/grasp_synthesis/<名字> \
  --seeds 40 --top-k 8 --opt-iters 500 \
  --table-up <ux> <uy> <uz>          # 可选:物体系里的"上"方向,过滤掉朝下的播种点(防穿桌)
```
(序列目录里要先有 `object.obj`。`--table-up` 一般由编排器自动算好;单独跑可省略。)

**anchored 人手锚定(修虎口朝下;需要 human_demo.npz):**
```bash
# 先从 replay_world.npz 建 human_demo.npz
scripts/run_grasp_synthesis_conda.sh scripts/grasp_synthesis/export_replay_human_demo.py \
  --replay-npz <RetargetOutput>/.../replay_world.npz \
  --out        data/testing/sequences/<名字>/human_demo.npz
# 再跑 anchored 合成
scripts/run_grasp_synthesis_conda.sh scripts/grasp_synthesis/synthesize_sharpa_anchored_bodex.py \
  --sequence-dir data/testing/sequences/<名字> \
  --out-dir      data/testing/grasp_synthesis/<名字> \
  --seeds 40 --top-k 8 --opt-iters 500 \
  --pose-weight 1.0 --table-penalty-weight 500 --no-isaac-visualize
```

### C. 在 Isaac 里"看"一个抓取

**静态看 GraspPose(弹窗,窗口一直开):**
```bash
scripts/run_isaacsim_conda.sh scripts/isaac/visualize_grasp.py --mode local \
  --sequence-dir data/testing/sequences/<名字> \
  --grasp-json   data/testing/grasp_synthesis/<名字>/grasp_pose_1.json \
  --out-dir      data/testing/grasp_synthesis/<名字>/viz \
  --show-table --hold-open --hold-open-until-closed
```
加 `--turntable-frames 120` 出一个绕圈转台 mp4(`viz/grasp_turntable.mp4`);加 `--headless` 则不弹窗只出图/视频。

**看完整过程(接近→抓取→抬起,实时物理):**
```bash
scripts/run_isaacsim_conda.sh scripts/isaac/simulate_grasp_traj.py --mode local \
  --trajectory-dir data/testing/grasp_traj/<名字> \
  --out-dir        data/testing/grasp_traj/<名字>/isaac_sim \
  --sequence-id    <名字> \
  --manifest       $PWD/data/testing/identity_manifest/manifest.json \
  --tabletop-z 0.85 --object-mass 0.2
```
(不加 `--headless` 就弹窗;加了就无头只录 `video.mp4`。远程无显示器用 `--headless`。)

### D. 单独重跑 Stage A / Stage B
- Stage A(轨迹生成,cuRobo):`scripts/grasp_traj/generate_reconstructed_grasp_traj.py`,参数 `--sequence-dir --synthesis-dir <合成目录> --replay-npz --recon-mesh --out-dir --carry-lift-height 0.10 --squeeze-overclose 0.2`。用 `--synthesis-dir` 时会自动选"过桌面且力封闭最优"的抓取。
- Stage B(物理+视频)= §C 的 `simulate_grasp_traj.py`。

---

## 4. `reconstructed_grasp_video.py` 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--recon-dir` | — | ReconstructOutput/.../<take>(含 `object_mesh_scaled_final.obj`)|
| `--retarget-dir` | — | RetargetOutput/.../<take>(含 `replay_world.npz`,`ref_qpos.npz` 可选)|
| `--sequence-id` | — | 输出名字 |
| `--anchored` | off | 人手锚定(修虎口朝下)。**没有人手接触帧时自动回退到 affordance**(见 §5 空隙)|
| `--squeeze-overclose` | 0.0 | 到位后手指多闭的弧度(抓得更紧、少下滑),0.2 较实 |
| `--pose-weight` | 1.0 | anchored:人手位姿先验强度 |
| `--table-penalty-weight` | 500 | anchored:手-桌碰撞惩罚权重 |
| `--cone-halfangle-deg` | 80 | **approach-cone** 半角:把抓取区域收窄到人手接近的那一侧(见 §8)|
| `--no-approach-cone` | off | 关掉 approach-cone,用完整 affordance 区域 |
| `--opt-table-penalty-weight` | **0(关)** | **①** affordance 路径:优化时的手-桌惩罚。厚物体建议 500;薄物体慎用(见 §8)|
| `--fingertip-contacts` | off | **②** 指尖捏取(实测有害,仅实验用,见 §8)|
| `--seeds` / `--opt-iters` / `--top-k` | 40 / 500 / 8 | 合成搜索规模 |
| `--carry-lift-height` | 0.10 | 抬升高度(m)|
| `--object-mass` | 0.2 | 物体质量(kg,Stage B)|
| `--table-z` | 0.85 | 桌面高度(m)|
| `--reuse-synthesis` | off | 已合成过就跳过合成,只重跑轨迹+视频(省时)|

---

## 5. 核心概念 / 设计决定(读懂这些就懂为什么这么跑)

- **anchored vs affordance**:两条合成分支。anchored 从**人手抓取帧**播种 + 位姿先验 → 抓取方向自然(**修虎口/大拇指朝下**);affordance 只从 AffordanceModel 的**区域**播种(不需要人手碰到物体),是 anchored 用不了时的回退。
- **重建空隙(gap)**:有些 take 物体位姿(FoundationPose)在被手遮挡时会漂,导致人手和物体差很远(0 接触帧)→ anchored 没法播种 → **自动回退 affordance**。判据看日志 `contact_frames`。**裁剪原视频到只有 grasp 动作,通常能让人手全程接触 → anchored 生效**(这是让 anchored 工作的推荐做法)。空隙本身不归 OCIR 处理,交给 RL Correction。
- **场景 = 和 retarget 一致**:物体保持**重建出来的朝向**(不做 stable-pose 投影)、落到桌上;手和物体用同一变换,保证"手在物体上方"。z-up 世界系,桌面 0.85,scene-rot 默认 identity(egodex 本就 z-up)。
- **防穿桌(两条分支都做了)**:①affordance 播种时**剔除朝下的表面点**(手只从上/侧接近);②轨迹生成时**只选手部球不扎进桌面的抓取**。所以 gappy take 回退到 affordance 也不会"手从桌下伸上来"。
- **overclose(抓紧)**:到 GraspPose 后把屈曲手指再多闭一点、抬升全程保持,让软驱动持续顶住、减少下滑。握力被手指力矩上限兜着,不会挤飞。
- **ref_qpos 可选**:轨迹的"初始手姿"优先用 `ref_qpos.npz`;没有就从 `replay_world.npz` 的 MANO 关节推(所以缺 ref_qpos 也能跑)。初始手腕取**最长连续有效手段的起始帧**(`t_init`),手指从张开开始。
- **identity manifest**:轨迹是在世界系里写的,Stage B 本来要 DexYCB 相机外参;我们用一个"恒等 manifest"让"相机系=世界系",无需真外参。`data/testing/identity_manifest/`,按 sequence-id 自动生成。

---

## 6. 数据与输出布局

**输入(Step2 `Reconstruct_and_Retarget` 产出,别改):**
```
Output/ReconstructOutput/egodex/part2/<子类>/<take>/object_mesh_scaled_final.obj
Output/RetargetOutput/egodex/part2/<子类>/<take>/replay_world.npz   (+ 可选 ref_qpos.npz, object.usd)
```

**OCIR 工作区(`data/testing/`,已 gitignore):**
```
sequences/<id>/        object.obj(抽简), human_demo.npz(anchored 用), affordance_pred/
grasp_synthesis/<id>/  grasp_pose_N.json / failed_grasp_N.json, summary.json, affordance_seed_info.json
grasp_traj/<id>/       trajectory.npz(手轨迹:hand_pos/quat + finger_targets + segment)
                       isaac_sim/video.mp4, report.json(指标), object_track.npz(物体实际轨迹), scene.usd
identity_manifest/     Stage B 用的恒等 manifest
```
`report.json` 指标:`grasp_success` / `lifted` / `max_lift_m` / `final_object_position_error_m` / `lifted_carry_fraction` 等。

---

## 7. Affordance 模型(换更好的模型)

- 代码已 vendored 进 `src/ocir/affordance/`;checkpoint 放 `assets/affordance/model.pt`(434MB)。
- **⚠️ checkpoint 不随仓库分发**:`assets/` 已在 `.gitignore` 里(体积 + MANO 许可限制),clone 后需**自行放置**。用 `OCIR_AFFORDANCE_CKPT` 指到别处也行。同理 `assets/mano/mano_right_vertex_part_ids.npy`(anchored/export 用)也需自备。
- **换模型 = 直接替换 `assets/affordance/model.pt`**(加载器从 checkpoint 自带的 cfg 读架构,其他不用动)。详见 `assets/affordance/README.md`。
- 单独跑一次预测:`conda run -n deximit env PYTHONPATH=$PWD/src python -m ocir.affordance.predict --mesh <obj> --out <dir>`。
- 环境搭建:`envs/affordance-requirements.txt`。

---

## 8. 新增约束项:实测结论与已知问题

> 这一节记录 approach-cone / ① 桌面惩罚 / ② 指尖 三个**最外层新增约束**的实测表现。
> 结论来自 **18 条右手 egodex `basic_pick_place` take × 4 组配置 = 72 次完整流程**(合成→轨迹→Isaac 物理)。
> **接手前务必读完**——有两项是「默认关且不建议开」。

### 实测总表(18 物体,纯 affordance 路径)

| 组 | 抓起并保持 | 没抬起 | 抬起又掉 | 物理异常 |
|---|---|---|---|---|
| base(仅 approach-cone) | 6 | 11 | 1 | 0 |
| **+ ① 桌面惩罚** | **10** | 5 | 3 | 0 |
| + ② 指尖 | 4 | 13 | 1 | 0 |
| + ①+② | 5 | 11 | 2 | 0 |

按物体形态拆开(**厚度**是关键,不是大小):

| 组 | 薄物体(厚度≤2.6cm,6 个) | 厚物体(≥3.5cm,11 个) |
|---|---|---|
| base | 1 | ~4 |
| **① 桌面惩罚** | **1(没帮上)** | **~8(明显变好)** |

### ① 优化时桌面惩罚 —— 有效,但**只对厚物体**
- **默认关**(`--opt-table-penalty-weight 0`)。开:传 `500`。
- **原理**:冻结核心优化时**完全不知道桌子**(只有播种前剔除朝下法线 + 事后选过桌抓取两道被动过滤)。① 把 `relu(低于桌面深度)²` 加进优化成本,让优化过程主动把手推离桌面。
- **实测收益**:过桌候选数普遍从 `0~2/8` 升到 `3~8/8`,桌面间隙从 -26mm/-17mm 拉回 -2/-3mm;厚物体 OK 数约 4→8。
- **⚠️ 薄物体上会反噬**:包络与「不碰桌」几何互斥,惩罚一推就只能松手。典型 idx12(厚 1.3cm):穿桌从 **-33mm 修到 +12mm**,但 **grasp_err 从 0.013 飙到 0.849**(力封闭被打崩)→ 仍然抓不起。**它把失败模式从「穿桌」换成「抓不住」,不是修好。**
- **建议**:厚物体开;薄物体别指望它。

### ② 指尖捏取 —— **实测有害,默认关,别开**
- OK 数 6→4,**在它本该帮的扁物体上 0/8**。
- **原因(物理真实,非 bug)**:TopDown 用 5 指尖压扁物体顶面,接触法线同向、**无对握** → 力封闭不成立(扁物体 grasp_err 飙到 1.68;圆物体尚可 0.0127,但仍不如 11 点包络的 0.0015)。
- **副作用**:配合下面的「无自碰惩罚」缺口,手指会**蜷成一团、自碰**,物理里出现关节跟踪误差 53/80 rad 的失稳(base 全在 0.3–1.8)。
- 保留仅为实验对照。**「扁物体该用指尖捏」这个假设已被证伪**(至少 top-down 这么做不行)。

### approach-cone —— 默认开,但方向可能不可信
- 方向由**接触前手腕 path**估(自适应回溯到累计位移 ≥5cm),**不用**重建的手腕朝向(pose 不可信)。
- 兜底已做:位移 < 3cm → 不收窄;锥后存活点 < min_points → 不收窄。
- **⚠️ 残留风险**:重建噪声大的 take 方向可能偏(20 条里 take4/take10 估出的接近方向朝上,明显不合理)。80° 宽锥 + 兜底能压住大部分,但**噪声 take 仍可能收偏、滤掉好区域**。日志里核对 `approach-cone: ... N -> M points` 和方向向量。

### 已知缺口:冻结核心**没有自碰惩罚**
- `bodex_curobo_v2`(affordance 路径走的)优化成本里只有 grasp_energy / contact_distance / regularization / quat_norm —— **不管手指互插**。anchored 有 `w_selfcol=1000`,affordance 路径**没有**。
- 后果:② 尤其明显(手蜷成团);base/① 目视尚可但同样缺这道防线。
- **未修**。要修就照 ① 的方式把 anchored 的 `_self_collision_cost` 以 monkeypatch 挂进 affordance 路径。

### 仿真侧:两条已验证的**反面**结论(别再踩)
- **`--object-collision sdf` 对这类物体有害**:SDF「穿模处理更软」,深穿模会沉更深、反弹更猛。实测把爆炸从 jtrk 80 rad 推到 **153 rad**、物体角速度冲到 187 rad/s。**用默认 convex decomposition。**
- **缩小接触 offset 同样帮倒忙**(margin 更小 → 插得更深再弹)。
- 那个 187 rad 爆炸的根因是 **②的自碰位姿**,不是碰撞体类型——**仿真调参救不了一个自碰的目标位姿**。

---

## 9. 坑 / 排错

- **一定用包装脚本 + `export OCIR_DATA_ROOT=$PWD/data`**。直接 `python` 会因 Isaac 的 numpy/torch shadow 出错。
- **视频里物体偏小**:相机按整张大桌子自动取景。看细节可以用 §C 的转台或 GUI。
- **`contact_frames=0` + anchored 失败**:重建空隙大,已自动回退 affordance;想用 anchored 就把原视频裁到只有 grasp 动作再重建。
- **下滑**:力封闭临界(grasp_error 高于 0.001)+ retarget 手指碰撞体轻微重叠;加 `--squeeze-overclose 0.2`、或更多 `--seeds`、或 Stage B `--friction`(默认 3.0,手物有效 9.0)。
- **穿桌**:已修(§5 防穿桌)。若换了新物体又出现,查日志 `table-clearing grasp selection: N/8 candidates clear`——若 0/8,说明所有候选都扎桌,需要更强的上表面过滤或换 anchored。
- **`assets/` 不在仓库里**:已 gitignore(415MB checkpoint + MANO 许可限制)。clone 后 affordance/anchored 报找不到文件,就是缺这些——见 §7 自行放置。
  (例外:`assets/robots/hands/sharpa_wave/` 的手模型是上游历史里就跟踪的,**会**随 clone 下来。)
- `data/` 已 gitignore,里面的产物不会误提交。

---

## 10. 文件地图(改哪找哪)

| 文件 | 作用 |
|---|---|
| `scripts/grasp_traj/reconstructed_grasp_video.py` | **一条命令编排器**(§0)|
| `scripts/grasp_traj/generate_reconstructed_grasp_traj.py` | Stage A 轨迹生成 + 过桌面抓取选择 |
| `scripts/grasp_synthesis/synthesize_affordance_seeded.py` | affordance 播种合成(+ 上表面过滤)|
| `scripts/grasp_synthesis/synthesize_sharpa_anchored_bodex.py` | anchored 人手锚定合成 |
| `scripts/grasp_synthesis/export_replay_human_demo.py` | replay_world.npz → human_demo.npz 适配器 |
| `scripts/isaac/simulate_grasp_traj.py` | Stage B 物理仿真 + 录像 |
| `scripts/isaac/visualize_grasp.py` | 静态抓取可视化 + 转台 |
| `src/ocir/grasp_synthesis/affordance_seed.py` | affordance 预测封装 + 区域加载(上表面过滤)+ 播种注入 |
| `src/ocir/grasp_synthesis/anchored_bodex/` | anchored 管线(rollout 里有手-桌碰撞惩罚)|
| `src/ocir/affordance/` | vendored 的 Affordance 模型 + `predict.py` |
| `src/ocir/isaac/identity_manifest.py` | 恒等 manifest 生成 |
| `assets/affordance/model.pt` | Affordance checkpoint(换模型只替换它)|
| `docs/affordance_seeding.md` / `docs/grasp_traj.md` | 更详细的子系统文档 |

> 冻结、别改:`src/ocir/grasp_synthesis/bodex_curobo_v2/`(核心 BODex 优化器)。新功能都是在它之上做加法。
