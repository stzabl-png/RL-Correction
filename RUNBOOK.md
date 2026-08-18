# 端到端复刻手册：一条视频 → 重建 → 抓取位姿

给要复刻这条链的同事。**每条命令都是从代码里核出来的，不是凭记忆写的**；有不确定的地方我会明说。

---

## 0. 先搞清楚：两条链在同一个仓的两个分支上

```
Step2_NoisyRecon    重建(相机/双手/物体 mesh+6DoF) + 可信度       ← 本分支
Step3_Dexonomy      抓取位姿生成(Dexonomy)                        ← 另一个分支
```

同一个 GitHub 仓 `stzabl-png/RL-Correction`，**必须分别 clone 到两个目录**，不能在一个目录里来回切分支（两边都有各自的 `third_party` 与生成物）。

```bash
git clone git@github.com:stzabl-png/RL-Correction.git recon      && (cd recon && git checkout Step2_NoisyRecon)
git clone git@github.com:stzabl-png/RL-Correction.git grasppose  && (cd grasppose && git checkout Step3_Dexonomy)
```

---

## 1. 环境

重建链**一条命令跨 4 个 conda 环境**，脚本自己 `conda run` 切换，你不用手动激活。权威表在
`ego_pipeline/Reconstruction/recon_pipeline/run_batch_queue.py` 的 `STEP_ENVS`，**别凭记忆**：

| env | 负责的步骤 |
|---|---|
| `cu128` | vipe（相机位姿 + 深度）|
| `codetr` | sam3_hands / sam2_object / HOI-DETR v17A / label.sh |
| `biv2ap` | sam3d / sam3d_scale / fp_pose（物体建模、尺度、6DoF）|
| `hawor` | hawor / vlm_gate / retrieval / fuse / confidence / contact |

**显存**：`sam3d` 30G、`sam3d_scale` 34G、`fp_pose` 28G、`confidence` 12G、`fuse` 2G。
16G 的卡跑不动物体那三步——**本地只适合开发，跑数据要上大显存机器**。

---

## 2. 重建

### EgoDex（有设备自带相机/内参/重力，走专用入口）

```bash
cd recon/ego_pipeline
./reconstruct_egodex.sh pour/17                    # 单条
./reconstruct_egodex.sh pour                       # 整个任务
./reconstruct_egodex.sh pour/17 --variant prod     # 手改用 HaWoR 估(默认 dev 用设备手)
```

> ⚠ **EgoDex 必须走这个入口**。走通用的 `reconstruct.sh` 会拿 HaWoR 估的手，实测手会跑到离相机
> 2.2m 的地方，而且三项体检全过——**错得没有任何征兆**。

`--variant` 两个值的差 = HaWoR 的误差对下游的影响，是白捡的量化结果：

| | 相机/内参/重力 | 双手 |
|---|---|---|
| `dev`（默认）| EgoDex | EgoDex |
| `prod` | EgoDex | HaWoR |

### 其它数据集 / 任意 mp4

```bash
cd recon/ego_pipeline
./reconstruct.sh 10                                          # HOI4D 前 10 条
./reconstruct.sh <视频.mp4> --dataset X --root /path/to/X
./reconstruct.sh <目录> --dataset egodex                      # 目录下所有 mp4
./reconstruct.sh ... --gpu-ids 2,3                            # 选卡
```

物体标注**默认全自动**（v17A 出 mask，无需人工）。人工点选只是 fallback（`--web` 或先跑 `./label.sh`）。

> ⚠ 多物体必须用**视频级** manifest：`auto_label_v17a.py --instance all`（或 `AUTO_LABEL_INSTANCE=all`）。
> 用 episode 级会让多物体静默塌成单物体。

### 产物

```
Output/ReconstructOutput/<dataset>/<task>/<take>/
    world_fused.npz                  相机/双手/物体位姿, 下游都读它
    confidence_complete.json         该 take 的可信度裁决(见第 5 节)
    objects/<oid>/object_mesh_scaled_final.obj
    contact/grasp_prompt.json        给 Step3 的工作清单
```

### 远程批量

八卡机器上跑批见 `ego_pipeline/Reconstruction/docs/REMOTE_RUNBOOK.md`。
**共享机器上不要把 GPU 写死**——起跑时空的卡半小时后可能被别人吃掉，实测 8 条里 7 条
`sam3d_scale` OOM。选卡要按**利用率**优先再按空闲显存，且留 5G 余量：只看显存会挑中
"显存空、算力满"的卡（实测 GPU 用 4G 但 SM 已 75%）。

---

## 3. Retarget（转仿真格式）

```bash
cd recon/ego_pipeline
./retarget.sh                       # 全部
./retarget.sh <take目录>            # 指定
./retarget.sh --skip-usd            # 只出 replay_world.npz, 不启 Isaac
```

产物：`Output/RetargetOutput/<dataset>/<take>/replay_world.npz` + `object.usd`

---

## 4. 抓取位姿生成（Step3，抓取组维护）

新机器先初始化一次（做两件不可省的事：软链物体资产目录、`pip install -e` 装 `dexrun` 命令）：

```bash
# ★ 强烈建议先显式指定物体资产放哪, 理由见下面的坑
export STEP3_OBJ_ROOT=/你的大盘/step3_object
cd grasppose/steps/step3_grasppose && ./setup_workdir.sh
```

然后：

```bash
python steps/step3_grasppose/run_take.py --take <重建take的绝对路径>
python steps/step3_grasppose/run_take.py --dataset egodex_auto --task pour --take-id 17
python steps/step3_grasppose/run_take.py ... --dry-run     # 只打印计划
python steps/step3_grasppose/run_take.py ... --force       # 重做
python steps/step3_grasppose/run_take.py ... --only ...    # 只跑指定组合
```

**不用先激活 conda**：`run_take.py` 只 import 标准库，系统 `python3` 就能跑；重活在
`take_grasp_pipeline.sh` 里，那个脚本自己找 dexonomy 环境。

它会自己读 Step2 的产物决定跑什么：`grasp_prompt.json` 给工作清单，
`confidence_complete.json` 的裁决决定跳过/降级，`vlm_grasp.json` 给形状先验。

### ⚠ 两个会让新机器卡住的坑

**① `assets/object` 软链指向一个空目录是正常的。**
物体数据由 `run_take.py` **逐 take 生成**（约 1~3GB/物体），不需要预先存在。
看到软链指向空目录**不要以为装错了**，也不要去别处找这些文件。

**② 没设 `STEP3_OBJ_ROOT` 时的默认值对新机器是个陷阱。**
`paths.py` 的解析顺序是：

```
环境变量 STEP3_OBJ_ROOT  >  /home/lyh/Project/Dexonomy/assets/object(存在才用)  >  <仓父目录>/_step3_work/object
```

中间那项是为了在原开发机上复用已生成的资产才留的。**如果你机器上恰好也有
`~/Project/Dexonomy/assets/object`（比如你 clone 过旧的 Dexonomy），它会被静默优先使用**——
不报错，只是你以为在用新目录、实际在用旧的。所以上面那句 `export` 建议照做。

> 这一节由抓取组（`Step3_Dexonomy` 分支）复核过（2026-08-17，`--dry-run` 与
> `setup_workdir.sh` 均实测通过）。**以他们的 `steps/step3_grasppose/README.md` 为准。**

---

## 5. ★ 怎么读产物的可信度（最容易用错的地方）

权威文件：`Data/VideoPrior/poseqa/TAKE_MANIFEST.json`。**不要自己扫目录判断数据好坏。**

每条记录的关键字段：

```
position_grade    good(≥70) / mixed(≥40) / poor      位置准不准
rotation_grade    good(≥70) / mixed(≥50) / poor      朝向准不准  ← 2026-08-17 新增
rotation_usable   布尔                                存量口径, 与上面并存
rotation_free_axes  该轴永久自由                      ★见下
```

### ⚠ 三件必须知道的事

**① `rotation_grade` 是排序，不是合格证。**
用 ARCTIC 真值标定（51 条 take / 16889 帧）：

| 档 | 真实旋转误差中位 | p75 |
|---|---|---|
| good | 20.4° | 23.6° |
| mixed | 34.9° | 86.2° |
| poor | 103.0° | 128.0° |

但 **good 档大约 1/7 会翻车**（7 条里 1 条真实误差 113°），反向也有（poor 档 30 条里 4 条其实只有 9~12°）。
这些数字**跟着 manifest 的 `rotation_grade_caveats` 字段一起下发**，不用去翻文档。

**② `rotation_free_axes` 不是缺陷，是物理限制。**
我们 89 个物体里 **78 个（88%）恰好有一条自由轴**——瓶子、杯子绕自己中轴转，画面上完全一样，
**任何算法都测不出来**。这些 take 的位置轨迹可能非常好（有一条位置 94 分、旋转 0 分）。
⇒ **别把 `rotation_grade=poor` 读成"这条数据废了"**，先看 `position_grade` 和自由轴。

**③ 逐帧用法要用增量误差验，别用累计误差。**
`conf_rot≥30 且 σ_rot≤5° → 该帧可用` 这条逐帧规则**是成立的**（低分帧每帧多错 2.0°，基线 3.69°/帧）。
但如果你拿"相对第 0 帧的累计误差"去验它，会得出"完全无效"的错误结论——因为序列中途一次翻面
会让之后每一帧都爆表，与那些帧各自的分数无关。**我 2026-08-17 就这么误判过一次并差点写进代码。**

### 标定的适用边界

标定用的全是 ARCTIC 的大件（微波炉、笔记本、盒子），而我们要用在瓶杯上。
**跨物体类别能不能搬，未验证**，且 ARCTIC 里没有瓶杯——重算 ARCTIC 也解决不了这个。

---

## 5.5 重跑数据前必须知道的三件事

### ⚠ 重跑是赌博，而且赌输了会覆盖原数据

`fp_pose` 是**非确定性**的：同配置重跑，`conf_pos`/`conf_rot` 各能飘 25~36 分
(实测 screw/13 同配置三次: 46 / 16 / 31)。重建链会**覆盖**产物，闭环搜帧只在本轮
搜出的配置里择优，**不知道重跑前的旧成绩**。已经因此弄坏过两条
(screw/4 53.5→38.0、screw/7 30.0→29.0 掉出可用线)，旧产物找不回。

⇒ **跑前存快照，跑后比，变差回滚**：

```bash
ego_pipeline/bin/take_snapshot.sh save    <take目录>   # 跑之前
ego_pipeline/bin/take_snapshot.sh diff    <take目录>   # 跑之后(自动判是否变差)
ego_pipeline/bin/take_snapshot.sh restore <take目录>   # 变差了回滚
```

约 26MB/条。`diff` 会区分"物体数变多(目标达成，不建议回滚)"和"物体数变少(真的坏了)"。

### ⚠ 重跑标注必须设 `AUTO_LABEL_FORCE=1`，否则整轮白跑

`auto_label_v17a.py` 有两道跳过闸：`label_prompt.json` 已存在就直接返回。
不设这个变量，跑满 25 分钟出来的结果和上次一模一样，**而且不会有任何提示**。

### 服务没起时可以跳过 VLM 检索

```bash
NO_VLM_RETRIEVAL=1 ./reconstruct_egodex.sh ...     # 服务不可用时
NO_FRAMESCAN=1     ./reconstruct_egodex.sh ...     # 跳过闭环精修(留到最后统一做)
NO_CONTACT=1       ./reconstruct_egodex.sh ...     # 跳过接触提取
```

`vlm_retrieval` 给的是**检索先验**，与几何重建无关，事后可单独补跑。
但**透明门也用 VLM**(在 auto_label 里，另一条路)——服务没起时它会 fail-open **全部放行**，
等于过滤规则没生效。批量跑之前要过滤空透明物体的话，先把服务拉起来。

### 起 VLM 服务

```bash
~/framescan_ab/start_vlm.sh <GPU号> [端口]     # UCB 上; util 按空闲显存自动反算
```

需要 **≥30GB 空闲**(权重就要 27.6 GiB)。六个已知启动坑都已在脚本里规避
(其中两个是 2026-08-17 新踩的: 只用绝对路径调 `vllm` 会找不到同环境的 `ninja`;
默认 256K 上下文光 KV cache 就要 16 GiB 会直接启动失败)。

## 6. 已知走不通的路

**换帧重建救不了烂数据。** 试过 10 条，净收益 0；而且**重跑会把本来合格的数据弄坏**
（重建链覆盖 `confidence_complete.json`，闭环搜帧只在本轮搜出的配置里择优，不知道重跑前的旧成绩）。
最后一条 `pour/19` 跑了 469 分钟，结果仍不合格。
⇒ **批量重跑前必须先筛掉已合格的 take。**

---

## 7. 复刻一条的最小例子

```bash
# ① 重建
cd recon/ego_pipeline && ./reconstruct_egodex.sh pour/17

# ② 看裁决
python - <<'EOF'
import json
d=json.load(open("../Output/ReconstructOutput/egodex_auto/pour/17/confidence_complete.json"))
for k,v in (d.get("objects") or {}).items():
    print(k, v["position_grade"], v.get("rotation_grade"), v["rotation_usable"])
EOF

# ③ 转仿真格式
./retarget.sh ../Output/ReconstructOutput/egodex_auto/pour/17

# ④ 抓取位姿(另一个 checkout)
cd ../../grasppose && python steps/step3_grasppose/run_take.py \
    --dataset egodex_auto --task pour --take-id 17
```

---

*维护：Step2 部分由重建侧维护；Step3 部分以 `Step3_Dexonomy` 分支的 README 为准。
最后核对 2026-08-17。*
