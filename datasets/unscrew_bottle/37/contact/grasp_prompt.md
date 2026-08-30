# 抓取接触先验 · 给 GraspPose Agent

来源 take: `/home/yanghong/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_part4/screw_unscrew_bottle_cap/37`

## 这份东西是什么

单目 egocentric 视频重建 -> 逐帧手(MANO)与物体(6DoF)在世界系对齐 -> 在'手物相对位姿稳定 且 贴合 且 对生'的窗口内, 统计物体表面每个采样点被手碰到的帧数比例。判贴合时**丢弃沿相机视线方向的距离分量**(单目深度近乎不可观测), 只用图像平面内的分量。

## ⚠ 使用前必读 · 已知误差

1. **接触区可能整体偏移。** 物体位姿是重建出来的, 实测物体投影与实测 mask 的 IoU 只有 0.45~0.69。接触区的**相对形状**(在物体上的高度/方位)比**绝对坐标**可信。
2. **接触区常常只覆盖一侧圆弧而非整圈。** 手相对物体有偏移时只有近的那半边登记上, 所以 `azimuth_span_deg` 是**下界**, 真实包裹范围只会更大不会更小。
3. **`opposition` < 0.40 的不是抓握**(可能是推/扶/蹭), 已被过滤不会出现在下面。
4. 每条都给了 `alternative_windows` —— 同一段视频里手常有多次抓握, 选窗只取了最长的一次。若最长那次的接触区不合理, 换一个候选窗重跑即可。

## 抓握 1: object_0 × left手

- 物体网格: `/home/yanghong/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_part4/screw_unscrew_bottle_cap/37/objects/object_0/object_mesh_scaled_final.obj`
- 尺寸: [0.065, 0.065, 0.197] m, 主轴 z
- 抓握发生在第 10~14 帧(5 帧)

**接触区在物体上的位置**

- 沿主轴高度: **15% ~ 30%**(中位 24%, 0=最低端)
- 绕主轴方位跨度: **90°**
- 离轴距离: 物体最大半径的 100%
- 热点数 295(占表面 0.5%)

**可信度**

- 对生度 0.7(≥0.40 才算抓握; 0.5≈环抱180°)
- 手部体检 0.354(正常 0.55~0.63)
- 物体体检 0.683(健康 0.45~0.69)
- 手到物体表面 0.1 mm
- 物体位姿可信度 conf_pos=83.0 conf_rot=57.0
- **只覆盖相机可见的一面**: 自遮挡门删掉 91% 的背面点; 方位跨度 90° 是**下界**, 真实包裹只多不少
- 同一段视频里还有 4 个不同的稳定抓握
- 精确点云: `/home/yanghong/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_part4/screw_unscrew_bottle_cap/37/contact/contact_v2_object_0_left.ply`(可直接拖进 MeshLab)

## 被拒的(物体×手)组合

这些**试过了但不可用**, 不是没试 —— 别当成缺数据。

- `object_0 × right`: **no_stable_contact** —— 没有既稳又贴的帧段
- `object_1 × left`: **no_contact_interval**
- `object_1 × right`: **no_stable_contact** —— 窗口 1 帧 < 门槛 5 帧, 判无稳定抓握

## 语义对照

| 状态 | 含义 |
|---|---|
| `hand_unreliable` | 手被重建到了错的位置, 该 take 的接触全部不可信 |
| `no_contact_interval` | 这只手压根没碰过这个物体 |
| `no_stable_contact` | 碰了但不构成抓握(不稳/不贴/不对生) |
| `no_contact_verts` | 有窗口但没有点通过 2D 否证, 可疑 |
