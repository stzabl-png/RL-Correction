# Reconstruct_and_Retarget 使用手册

> 每次开新 Terminal 先 `cat /home/lyh/Project/Reconstruct_and_Retarget/USAGE.md`。
> 这个项目做的事:**人类视频 → 重建(手+物体的世界轨迹 + 物体 mesh)→ retarget 到 SharpaWave 机器人手 → Isaac Sim 回放**,产物给下游 RL correction(`/home/lyh/Project/RL_Correction`,即 GitHub `Shen626/GR00T-VisualSim2Real`)。

---

## 0. 三步全流程(EgoDex 一条视频)

以 `Data/Egodex_Part2/part2/basic_pick_place/55.mp4` 为例(通用数据集,视频相对 `--root` 的路径决定输出嵌套:`part2/basic_pick_place/55.mp4` → 输出 `.../egodex/part2/basic_pick_place/55`)。

```bash
# 1) 重建(--web 打开浏览器标注物体)—— 必须在你自己的终端跑,不要放后台
cd /home/lyh/Project/Reconstruct_and_Retarget/ego_pipeline
./reconstruct.sh \
  /home/lyh/Project/Reconstruct_and_Retarget/Data/Egodex_Part2/part2/basic_pick_place/55.mp4 \
  --dataset egodex --root /home/lyh/Project/Reconstruct_and_Retarget/Data/Egodex_Part2 \
  --web --keep-interim
#   打开 http://127.0.0.1:8765/ → 左键点目标物体(绿点) 右键点背景(红点) → 按 s 保存 → 自动重建
#   远程: ssh -L 8765:127.0.0.1:8765 user@host

# 2) retarget(recon → replay + object.usd)。⚠️ 路径必须跟上,见「坑 2」
cd /home/lyh/Project/Reconstruct_and_Retarget/ego_pipeline
python bridge/retarget.py ../Output/ReconstructOutput/egodex/part2/basic_pick_place/55

# 3) 在 Isaac Sim 里看效果(SharpaWave 手 + 物体回放)
cd /home/lyh/Project/Reconstruct_and_Retarget/third_party/MagicDexMate
./run_retarget.sh ../../Output/RetargetOutput/egodex/part2/basic_pick_place/55
#   弹窗后按 ENTER 播放,再 ENTER 重播,q+ENTER 退出;自动连播加 --auto --loop
```

HOI4D(内置数据集,不用 `--root`,标注用本地 `./label.sh`):
```bash
cd /home/lyh/Project/Reconstruct_and_Retarget/ego_pipeline
./label.sh 10                    # 本地窗口标前 10 条(或给 Data 路径/父目录)
./reconstruct.sh 10              # 重建前 10 条(默认 --skip-label 用已标好的)
./retarget.sh 10                 # retarget 前 10 条(retarget.sh 仅支持 hoi4d)
```

---

## 1. 关键路径

| 用途 | 路径 |
|---|---|
| 项目根 | `/home/lyh/Project/Reconstruct_and_Retarget` |
| 脚本 | `ego_pipeline/{reconstruct.sh, retarget.sh, label.sh}`、`ego_pipeline/bridge/retarget.py` |
| GUI 回放 | `third_party/MagicDexMate/run_retarget.sh` |
| 重建输出 | `Output/ReconstructOutput/<dataset>/<嵌套take>/world_fused.npz (+ object_mesh_scaled_final.obj)` |
| retarget 输出 | `Output/RetargetOutput/<dataset>/<嵌套take>/replay_world.npz + object.usd` |
| 中间产物 | `Output/ReconstructOutput/interim/...`(默认跑完删,加 `--keep-interim` 保留) |
| 管线代码(重建) | `/home/lyh/Project/HumanVideo2RobotData/recon_pipeline`(git repo) |
| 内置数据集 root | hoi4d=`Data/HOI4D`;egodex(旧)=`/home/lyh/Project/V2AP/data/egocentric/egodex` |

---

## 2. 命令与常用 flag

**`reconstruct.sh <视频/目录/id...> [flags]`**(8 步:vipe→sam3_hands→sam2_object→hawor→sam3d→sam3d_scale→fp_pose→fuse)
- `--dataset X --root /path`:通用数据集(egodex 等)必给;hoi4d 不用
- `--web`:浏览器标注+重建一条龙(通用数据集首次必用,**在自己终端跑**)
- 默认无 `--web` = 本地 `--skip-label`(需已标注/有缓存)
- `--force`:重做所有步骤(否则已完成的跳过);`--keep-interim`:保留中间产物
- 其它 flag 透传 run_batch_queue

**`bridge/retarget.py <take目录> [--force] [--skip-usd]`**(通用数据集用这个,不是 retarget.sh)
- 参数是 **ReconstructOutput 下的 take 目录**;`--skip-usd` 只出 replay 不启 Isaac;`--force` 重做
- 内部:`recon_to_replay.py`(hawor 环境跑 MANO)+ `obj_to_usd.py`(.venv-isaac 出 USD)

**`run_retarget.sh <take目录> [flags]`**(在 `third_party/MagicDexMate` 下跑)
- 自动找 take 目录里的 `replay_*world*.npz` + `*.usd`
- `--phys`:物理抓取测试(物体变刚体,靠接触);`--mano`:看人手 MANO 网格(不做机器人 retarget)
- `--auto --loop`:自动连续播放

---

## 3. 输出内容(`world_fused.npz` 关键字段)

- 世界系为 `gravity_z_up_world`(+Z 朝上)。
- `hand_trans (2,T,3)` `hand_rot (2,T,3)` `hand_pose (2,T,45)` `hand_valid (2,T)`:**index 0=左手, 1=右手**。
- `object_ob_in_world (T,4,4)`、`object_ob_in_world_all (n,T,4,4)`、`object_*_by_frame`。
- `hand_confidence (2,T)` / `object_confidence (T,)`:**逐帧置信度**(1 原始 / 0.5 插值 / 0.3 夹住 / 0 忽略),给 RL 用。
- `replay_world.npz` 额外有 `joints_left/right`、`valid_left/right`、`confidence_left/right`、`obj_confidence`、`obj_pose (T,7)`。
- **抓取阶段(grasp phase,可选)**:`hand_phase (2,Th)`(world_fused)/ `phase_left/right (Tv,)`(replay)+ `phase_obj`、`phase_source`,int8:`-1 未知 / 0 未接触 / 1 接触抓取中`。见第 8 节。

---

## 4. conda 环境(哪步用哪个)

| 步骤 | env |
|---|---|
| vipe | `cu128`(+ `third_party/vipe/.venv`,用 uv) |
| sam3_hands / sam2_object | `HV2RD` |
| hawor / fuse | `hawor`(**需装 OpenEXR**,已装) |
| sam3d / sam3d_scale / fp_pose | `biv2ap` |
| retarget: recon→replay | `hawor` python |
| retarget: obj→usd | `third_party/MagicDexMate/.venv-isaac` python |

sam3_hands 默认要 `sam3.1`;5090/Blackwell 显卡自动回退 `sam3`。检查 mesh/npz 等杂事可用 `hawor` 环境的 python(有 numpy/trimesh/h5py/scipy)。

---

## 5. 常见坑(务必看)

1. **长路径粘贴被终端折断** → bash 把它拆成两条命令执行(经常中招)。
   - 对策:先 `cd` 到父目录用**短相对路径**;或先 `T=.../take` 单独一行再 `... "$T"`;不要贴超长绝对路径。
2. **`bridge/retarget.py` 不带 take 路径 = 默认处理全部 take**(会去重跑 HOI4D 整批)。**永远把路径跟上**。
3. **Web 标注要在你自己的终端跑**(前台)。放后台 label 服务会掉、卡在空等标注。
4. **物体 mesh 质量看 SAM3D 单视图**:标注帧那一刻物体视角/mask 不好 → mesh 会塌(如圆柱塌成截锥)。SAM3D **seed 固定=42,原样重跑结果一样**。换更好的帧重标,或换视频。
5. **输出嵌套按视频相对 `--root` 的路径**:`part2/Grasp/0.mp4` 和 `part2/basic_pick_place/0.mp4` 是**两个不同输出目录**,不会互相覆盖。
6. **重跑覆盖**:已完成的 take 会被判「complete 跳过」,要真重算得加 `--force`(recon)/ `--force`(retarget)。
7. hawor 报 `Decoded EXR depth is all zero` = `hawor` 环境缺 `OpenEXR`:`conda run -n hawor python -m pip install OpenEXR`(已装,换机器才需再装)。

---

## 6. 管线里已做的改动(2026-07,分支 `docs/hawor-openexr` @ recon_pipeline)

1. **SAM3 证据门控补帧**:hawor 打开 infiller,但**只在 SAM3 mask 看得见手的帧才接受补出的帧**(不幻觉)。修了「交互手缺帧/被丢」。
2. **fuse 轻量轨迹清理**:对手+物体世界轨迹做稳健离群检测(偏离中值基线 + 速度跳变「飞出又回来」判据),短缺口前后插值、长/端点夹住或忽略,并写逐帧 `*_confidence`。参数在 `recon_pipeline/fuse/trajectory_cleaning.py` 的 `CleanConfig`。
3. Reconstruct_and_Retarget 侧:`recon_to_replay.py` 无效帧写 NaN + 透传置信度;`retarget_isaacsim.py` 按 valid 保留手(仅「<2 有效帧」才丢)。

**未做(下一步候选)**:相机/SLAM 世界漂移治本(静止手被重建出 ~0.5m 漂移);SAM3D 差 mesh 的「拟合 primitive」兜底。

---

## 7. GT 离线校验(仅 EgoDex,不进管线)

每条 EgoDex `mp4` 配有 `.hdf5`(Vision Pro GT):`transforms/leftHand|rightHand (T,4,4)`、`confidences/*`、attrs 里 `environment` 含 `hand:right/left` 与任务描述。**只用来离线核对**重建的手/物体轨迹相似度(Sim3 对齐后比手腕轨迹),**绝不进重建管线**(要的是 noisy reconstruction)。你自己裁剪的视频(如 `part2/Grasp/*`)没有 hdf5,无 GT。

---

## 8. 抓取/接触阶段(grasp phase)

在**人手轨迹上加逐帧阶段标签**(未接触/接触抓取中),给下游 RL correction 用。做成**可替换 provider**:现在=人工标注,同学的自动接触检测就绪后实现 `ego_pipeline/phase/auto.py` 即可无缝替换,下游零改动。详见 `ego_pipeline/phase/README.md`。

```bash
# 1) 人工标注(浏览器,Space 打 抓/放 点,交替)
python tools/annotate_grasp_frames.py <mp4或帧目录>       # -> 视频目录/grasp_annotation.json
# 2a) 事后贴到已有重建产物(不重跑管线;HOI4D 标注在 Data,需 --annotation)
python tools/attach_grasp_phase.py <ReconstructOutput/take目录> [--annotation <json>] [--dry-run]
# 2b) 或重跑 retarget 时自动写入(能定位到标注就写,向后兼容)
python ego_pipeline/bridge/retarget.py <take目录>
```

- 规范时间线=视频帧 Tv(标注帧号即 Tv);手 `Th=2*Tv` 写入时自动上采样对齐。
- 切 provider:`BIV2AP_PHASE_PROVIDER=auto|manual|none`,或 `--provider`。auto 未就绪时自动回退人工。
- 字段/标签见第 3 节末行。**同学接入点**:`ego_pipeline/phase/auto.py` 顶部契约。
