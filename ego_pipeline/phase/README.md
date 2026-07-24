# 抓取/接触阶段(grasp phase)

在**人手轨迹上添加逐帧语义标签**:每一帧、每只手当前是「未接触」还是「接触/抓取中」。
设计成**可替换的 provider**:现在接**人工标注**,同学的**自动接触检测**就绪后,
只实现一个类即可无缝替换,下游(attach 工具、retarget、RL correction)零改动。

## 数据流与时间线

```
标注来源(二选一,同一个 PhaseTracks 接口)          注入落点
─────────────────────────────────────────         ────────────────────────
人工: tools/annotate_grasp_frames.py                world_fused.npz
      -> grasp_annotation.json {left/right:[[s,e]]}   hand_phase (2,Th)
自动: 同学模块 (auto.py 待接入)          ──►          obj_phase  (Tv,)
      -> PhaseTracks(left,right @Tv)                replay_world.npz  ← 下游 RL 直接用
                                                      phase_left/right (Tv,)
                                                      phase_obj (Tv,)
```

- **规范时间线 = 视频/物体帧 Tv**。人工标注的帧号本身就是 Tv 帧号(实测 `num_frames`==`world_fused.num_frames`)。
- 手轨迹在 `world_fused` 是 `Th=2*Tv`(30fps vs 15fps),写入时用 `to_hand_timeline(Th)` 最近邻上采样对齐 `hand_trans`。
- `replay_world.npz` 全在 `Tv`,阶段 `(Tv,)` 与 `joints_*` / `confidence_*` 天然对齐(照 `confidence` 的模式)。

## 标签语义(int8)

| 值 | 名称 | 含义 |
|---|---|---|
| -1 | UNKNOWN | 无标注/未知(完全没有标注文件) |
| 0 | FREE | 未接触(自由移动、接近、松手后) |
| 1 | CONTACT | 接触/抓取中 |
| 2+ | 预留 | APPROACH/MANIPULATE/RELEASE… 由自动版扩展,在 `types.PHASE_NAMES` 登记 |

`index 0=左手, 1=右手`(与全项目一致)。

## 怎么用

**人工(现在)**:
```bash
# 1) 标注(浏览器,Space 打抓/放点)
python tools/annotate_grasp_frames.py <mp4或帧目录>          # -> 视频目录/grasp_annotation.json
# 2a) 事后贴到已有重建产物(不重跑管线)
python tools/attach_grasp_phase.py <take目录> [--annotation <json>]
# 2b) 或重跑 retarget 时自动写入(take 内/可定位到标注即写)
python ego_pipeline/bridge/retarget.py <take目录>
```

**自动(同学就绪后)**:实现 `auto.AutoGraspPhaseProvider` 后,上面命令**原样不变**——
`load_phase(prefer="auto")` 会优先自动、回退人工。也可用 `BIV2AP_PHASE_PROVIDER=auto|manual|none` 强制。

## 同学的接入点 —— `auto.py`

只需实现两个方法(契约见 `auto.py` 顶部注释):
- `available(take_dir, ...) -> bool`:结果就绪吗(建议文件契约 `take_dir/contact_auto.json`)。
- `load(take_dir, num_frames=Tv, ...) -> PhaseTracks`:返回逐帧 `left/right` @Tv,`source="auto"`。

输入可用:`take_dir/world_fused.npz`(手轨迹 @Th、物体轨迹 @Tv、mesh)、原始视频。
若模型在 `Th` 上出结果,用 `resample_labels(labels_Th, Tv)` 降到 `Tv`。

## 文件清单

| 文件 | 作用 |
|---|---|
| `types.py` | `PhaseTracks`、标签常量、时间线重采样(`resample_labels` / `to_hand_timeline`) |
| `provider.py` | `load_phase()` 工厂:auto→manual 自动回退 |
| `manual.py` | 人工 provider:读 `grasp_annotation.json`,`segments_to_dense` |
| `auto.py` | **自动 provider 占位**(同学在此接入) |
| `../../tools/attach_grasp_phase.py` | 就地把阶段贴到 world_fused / replay npz(不重跑) |
| `../bridge/recon_to_replay.py` | retarget 时可选写入 `phase_*`(向后兼容) |
