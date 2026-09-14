# 数据来源 · egodex_auto/clean_tableware__8

帧数 300 · variant `dev` · 生成于本目录的 `world_fused.npz`

**⚠ 本条数据不是纯重建。** 下表逐项说明每个量是我们估出来的，还是数据集直接给的。
拿它做精度结论前请先看这张表。

| 内容 | 来源 | 说明 |
|---|---|---|
| 相机位姿 c2w | **数据集给定** | EgoDex `transforms/camera`（设备 SLAM），非 ViPE 估计 |
| 相机内参 K | **数据集给定** | EgoDex `camera/intrinsic`（设备标定），非 ViPE 估计 |
| 世界系 / 重力 | **数据集给定** | EgoDex 本身重力对齐米制，仅做 Y-up→Z-up 换轴；原点在地面而非首帧相机 |
| 深度 | ****重建**** | ViPE 估计，再按 depth_scale=0.3823212313361898 缩放到真实米制 |
| 双手轨迹 | **数据集给定** | EgoDex ARKit 手部追踪（腕位）；有效帧比例 left 100% right 100%；手指 45 维由 ARKit 25 关节拟合（arkit_to_mano，腕系逐帧拟合），仅当拟合失败才退回零占位（重建日志有醒目警告） |
| 物体 mask | ****重建**** | HOI-DETR 找交互 + SAM2 传播（v17A 全自动） |
| 物体网格 | ****重建**** | SAM3D 单帧重建 |
| 物体尺度 | ****重建**** | sam3d_scale（单帧深度 + 单帧 FoundationPose） |
| 物体位姿 | ****重建**** | FoundationPose 逐帧跟踪 |
| 接触区间 | ****重建**** | mask 重叠检测；已产出 contact_auto.json |
| 材质判定 object_0 | **VLM 推断** | ? conf=high 建议过滤=False（只记录，未删数据） |
| 材质判定 object_0001 | **VLM 推断** | ? conf=high 建议过滤=False（只记录，未删数据） |
| 材质判定 object_0002 | **VLM 推断** | ? conf=high 建议过滤=False（只记录，未删数据） |
| conf_pos / conf_rot | ****估计**** | 66.0 / 10.0，判定 mixed |

## 不在本条数据里的东西

* **Retrieval（资产库检索）** —— 本条**未使用**资产（物体网格是 SAM3D 重建的）。原因: VLM 判 part_change=none, 不需要分件资产。资产库在 `ego_pipeline/Retargeting/assets/retrieval/`。
* **物体真值** —— EgoDex 只给物体的文字名（`llm_objects`），没有位姿/网格/尺寸。物体那条线的精度只能靠 ARCTIC 验。

任务描述：Clean plates, forks, spoons, and knives placed on a wooden table while sitting against a white background.
物体（LLM 标注）：['plates', 'fork', 'spoon', 'knife', 'table']

---

## 2026-09-10 类别先验尺度

盘子锚定 object_0=⌀24cm(标准餐盘先验), 整条 ×1.9124(相机中心规范变换, 投影不变)。
⚠ 尺度来源=**类别常识先验非重建**(此任务非抓握, 手锚失效; 重建内无可靠尺度锚)。原件在 `_prescale_backup/`。

与 take 3(×1.9884) / take 18(×1.85) 同一条先验。变换实现经回归验证: 用 take 3 的 `_prescale_backup`
原件复现其当前产物, max|diff| 1.9e-8。变换内容 = `obj_verts_local` 与网格顶点 ×k;
`object_ob_in_cam` 平移 ×k(旋转不动); world/vipe_world 系位姿由各自 c2w 重算; **手与相机完全不动**。

投影不变性自检(本地同一打分器, 变换前→后): occl 0.324→0.324 / 0.351→0.352, 拟合尺度 1.05→1.05,
conf_pos 66→65 / 56→54(光栅化精度 + 尺度搜索网格挪一格, 与 take 3 远程/本地 0.5 分漂移同量级)。

## 2026-09-10 抹布网格单独缩小(用户裁定"方案 3")

全局 k 保场景布局不变的前提下, `object_1`(海绵/抹布)网格**绕自身局部原点** ×0.65
(局部原点即网格 bbox 中心, 偏差 <0.1mm ⇒ 只缩不移), 24.5cm -> **16.0cm**。
**位姿数组完全未动**(`obj_pose` / `object_ob_in_*` 原样), 只改 `obj_verts_local[1]` 与网格文件。
原件在 `_meshfix_backup/`。

倍率依据: 本条自己的 mask 拟合尺度 0.65(take 3 的同一物体 mask 推出 14.5cm, 两条独立估计吻合);
**不锚到 take 3 的网格 13.2cm**, 因为那份网格自己也偏小(其 mask 要求 ×1.1)。

效果: `scale_iou` 0.365 -> **0.455**(+25%); `explained` 0.702 -> 0.585(构造性下降 ——
网格越大越容易"解释" modal mask, 不代表变差); conf_pos 54 -> 53(本来就被遮挡上限压着)。
拟合尺度 0.65 -> 0.875 未收敛到 1.0, **有意不再迭代**: CONFIDENCE_GUIDE 明示 mask 是 modal、
投影是 amodal, 遮挡下 IoU 系统性偏向更小的网格; 本物体遮挡 0.35, 继续追就是追偏差。

盘子 object_0 在这一步逐位未动(pos 65 / rot 4 / 证伪 0 / 拟合尺度 1.05 全部不变), 是本步的正对照。

## 2026-09-10 contact 重算(尺度变换后)

`--force` 全量重跑: `ref_qpos_{left,right}` 重生成(300/300 有效, 最大帧间跳变 0.06/0.04 rad),
4 组(物体×手)对齐+热度图重算, recon 渲染重出。
**四条 attempt 全部 `ok`**(重算前是 4 条 `failed`)。

⚠ 两处工具不覆盖旧产物, 已手工隔离到 `contact/_stale_pre_rescale_20260910/`:
- `affordance_object_*.png` —— `affordance_viz` 见文件存在就跳过, 已删除后强制重画。
- `contact_mano_object_*.npz` + `contact_v2_summary.json` —— `mano_contact` 只在 `ok` 分支写 npz,
  本次四条都是 `no_grasp_core` 故不写, 旧的 4 个 npz 会留成孤儿(`tools/arctic_eval/*` 会读它们)。

`contact_fingers.json` 由变换前的"四条全 ok(4~5 指)"变为**四条全 `no_grasp_core`**(max_regions=0),
与 take 3 / take 18 的结果一致 —— 变换前网格只有真实尺寸的一半, 手指落在 35mm 阈值内属假阳性。

## 2026-09-10 贴图重烘(两个物体)

`tools/sam3d_texture_bake.py`(远程 UCB, biv2ap, GPU 2), 2048² UV 贴图, 几何一根顶点不动。

| | 重建帧 | scale_raw_to_final | chamfer | 几何守恒 | 耗时 |
|---|---|---|---|---|---|
| object_0 盘 | f237 | 0.2401 | 1.02mm (p95 2.09) | 7.0e-06 mm | uv 255s + bake 32s |
| object_1 抹布 | f103 | 0.1600 | 0.89mm (p95 1.76) | 3.4e-06 mm | uv 140s + bake 31s |

**为什么必须重烘**: 原有的 `object_0/textured/` 是 **09-04 那次重建**的产物 —— 元数据里
`reconstruction_frame: 3` / `existing_vertices: 140818`, 而 09-10 重跑后网格是 143362 顶点、
重建帧 f237(framescan 换过帧)。顶点集不同源, UV 挂不上去。object_1 则从来没有贴图。

烘之前把**变换后**的网格推到了远程(工具的尺度是推导的 `s = mean(ext_final/ext_raw)`, 再
`fin.vertices / s` 转回 raw 系, 任何均匀缩放都能除回去), 所以产物直接是正确米制, 无需二次缩放。
校验: 贴图网格 extents 与当前网格逐项一致(21.0×3.3×24.0 / 9.0×4.7×15.9)。

通道: `input rgba_correct (post-fix take), no swap` —— 本 take 在 R/B 修复之后重建, 不需补偿
(09-04 旧贴图走的是 legacy swap 路线)。正对照: 视频里盘子是蓝色, 贴图 RGB 均值 (59,150,175) 也是蓝色。

## 2026-09-10 本地/远程一致性

变换后的 `objects/*/object_mesh_scaled_final.obj` + `world_fused.npz` + `replay_world.npz` 已推到远程,
两边网格校验一致。远程 take 目录下的 contact/ 与 poseqa/ 仍是变换前的(远程只是批量工作区,
take 裁决以本地 TAKE_MANIFEST 为准)。

---

## 2026-09-10 尺寸口径统一到 take 3(用户裁定)

用户裁定: **"Take3 里面训练的物体大小和 GraspPose 挺合适的, 我都想用 Take3 的大小"** ——
不论 take 3 的尺度是手动设计还是与 SAM3D 有多大偏差, 一律以它为基准。

**实证: 三条 take 的两个物体确实是同一批物件**(ICP, 6000 顶点采样):

| | take8 → take3 | take18 → take3 |
|---|---|---|
| 盘 object_0 | 尺度 1.006, 残差中位 3.06mm, 旋转 1.1°, det +1 | 尺度 0.997, 残差 4.62mm, 旋转 90.0°, **det −1(镜像)** |
| 抹布 object_1 | 尺度 0.829, 残差中位 2.36mm, 旋转 0.7°, det +1 | 尺度 1.032, 残差 3.14mm, 旋转 2.1°, det +1 |

**take 8 的网格局部系本来就与 take 3 对齐(≤1.1°)**, 所以换用 take 3 的资产**不需要任何朝向改写**。
⚠ take 18 的盘 ICP 落到了**反射解**(det=−1, 近似对称圆盘的已知陷阱), 它要对齐必须单独处理。

**本次动作**: object_1 网格再 ×0.829(绕自身局部原点, 位姿数组未动) -> **7.5×3.9×13.2cm**,
与 take 3 的 7.6×3.8×13.2 毫米级一致。原件 `_meshfix_backup/object_1_mesh_before_take3align.obj`。
(下午那次用 mask 拟合的 0.65 判断错了; 按 take 3 对齐应为 0.65×0.829=0.54。打分器的拟合尺度
由 0.65 变为 1.050, 反向印证 take 3 的尺寸是对的。) 盘 1.006 差 0.6%, 不动。

**训练链怎么用**(与 take 18 同一规矩): 物体资产与 GraspPose 一律用 **take 3 的交付件**
(`Dexonomy/output/DELIVER/clean3_{plate_left,sponge_right}/`, 导入物体 `clean3_plate18_ped`
obb 2.27×17.73×17.75cm = take 目录网格 ×0.75)。**本条自己的 `objects/` 只为溯源与目检,
训练链不读它** —— 所以 take 8 的盘那 14.5% 椭圆度不进训练。GraspPose **无需重新合成**。

⚠ 记录(不影响上述裁定): 用 ARKit 手真值当尺子实测盘直径约 11~13cm(与 SAM3D 原始重建吻合),
即 take 目录的 ⌀24 与交付的 ⌀18 都大于实物(玩具塑料餐具)。用户已知悉并选择沿用 take 3 口径。
