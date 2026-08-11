# ARCTIC 真值校准 confidence —— 实验台账

目的:拿 ARCTIC 真值回答"confidence 准不准、多少分能用",并尝试**新增判据**让它更准。
所有结论、改动、可还原信息都记在这里。**改动一律走 git,不改就地覆盖。**

## 纪律(先写在最前面,防止自己越界)

1. **v3 阈值已冻结**(见 `traj-confidence-pipeline` 记忆:同 4 条 take 调三轮 = 过拟合)。
   本次**只新增判据,不重调既有阈值**。若必须动既有阈值,单独记录并在留出集上验证。
2. **训练/留出集必须划分**。在 n=1~2 上"改进"没有意义。
   dev = 用来设计判据的序列;held-out = 设计完全冻结后才碰。
3. **任何我做的预处理都要记录**(如丢弃开头暗帧),它们不是数据本身的属性。
4. 报告误差时**分方向**(深度 vs 横向),因为本次主误差在深度而判据只看图像。

## 数据与口径

- 源:`/media/lyh/DATA2/arctic/repo/data/arctic_data/data/`(第一人称 = view 0,
  `crop_images.py` 对它是 `resize(0.3×)` 纯缩放,**不裁不 warp**;其余 8 机位是裁剪+拉伸,不可用)
- 造视频:`scratchpad/arctic_ego_video.py` —— 去畸变(dist8,角点偏 26–40px)、
  丢开头连续暗帧(阈值 40,自动曝光未稳)、落 `video_frame → arctic_vidx` 映射表
- 真值:`object.npy` = [铰接角, axis-angle(3), 平移(3,**毫米**)];
  `world2ego = [R_k_cam_np | T_k_cam_np]`;帧号 `vidx = 文件名 - ioi_offset`(逐 subject,实测有 1 和 2)
- 误差口径(**零拟合**):两边网格顶点按各自位姿变换后的**质心距**,在相机系内比较。
  不用 Umeyama/13参数联合拟合 —— 见下方"踩过的坑"。
- 旋转口径:**对称感知**,在 {I,Rx180,Ry180,Rz180} 上取最小角误差。

## 踩过的坑(会给出完全错误的结论)

1. 别把世界变换 X 和网格变换 M 一起解(13参数非凸)。实测落局部极小,报"尺度0.224/旋转142°",
   与相机独立测得 1.1995 矛盾。**X 是两个世界的属性 → 用相机轨迹 Umeyama 闭式解定住**。
2. **对称物体必须对称感知**。不处理时 conf_rot 相关 +0.302("无预测力"),处理后 -0.543 单调。
   **投影贴合 ≠ 朝向正确**。

---

## 进度日志

### 2026-08-11 00:0x —— take #1 `s05/laptop_grab_01`(基线,已完成)

十步全链全自动,`EXIT=0`。
- ViPE 相机:位置残差中位 **12.1mm**、朝向 **1.94°**(首次真值验证);**尺度偏小 17%**
- 物体网格:主尺寸差 6–7%(350 vs 326mm),**厚度多估 36%**(参考帧 f642 正上方俯视,厚度不可观测)
- fp_pose:828/828 帧、1 次配准、**fallback 0**
- **误差:质心中位 239mm(深度 236 / 横向 38),投影质心只差 33px** —— 图像里看不出来
- 深度比 我们/真值 = **1.233**(焦距高估 3.7% × 网格大 7.8% 叠加)
- confidence 判 `conf_pos 84 / grade good / rotation_usable=True`
- **排序有效**:conf_pos 五档误差 703/648/472/486/**269** mm,相关 -0.696
  conf_rot vs 对称感知朝向误差 67/66/64/50/**36°**,相关 -0.543,**单调**
- **绝对值不可信**:被判 "good" 的重建,物体中心偏了自身长度的 73%
- ★ 结论:判据(`explained`/`d_cent_norm`/`lag`)全是图像内轮廓比对,
  **对深度/尺度结构性失明**,而本次主误差正在该维度
- 可视化:`Output/arctic_vis/arctic_gt_vs_recon.mp4`

### 2026-08-11 00:58 —— 批量 10 条启动(dev + held-out 池)

11 类物体各取平移最大的一条 grab 序列(铰接 ≤6°,近似刚体)。laptop 已跑,其余 10 条批量:
ketchup(s07) box(s05) capsulemachine(s04) espressomachine(s05) microwave(s05)
mixer(s05) notebook(s06) phone(s10) scissors(s02) waffleiron(s05)

跳过 contact 步(纯 CPU 944s/条,与 confidence 无关)。GPU 5,tmux `batch10`,日志 `/tmp/batch10.log`。

**ketchup 是重点观察对象**:番茄酱瓶旋转对称 → conf_rot **应该**主动报低分。
若它反而给高分,说明判据会"自信地错",比分数偏低严重得多。

### 2026-08-11 01:0x —— 判据设计的关键前提被查清

**① `fp_pose` 是 RGB-D 配准, 深度来自 ViPE**(`fp_common._load_depth_frames` →
`est.register(k, rgb, depth, mask)`, 深度存在 `vipe/depth/<id>.zip` 的 EXR 里)。
后果一:"拿 ViPE 深度校验 fp_pose"这条判据**不成立**(天然一致)。
后果二:**23% 的深度误差很可能是从 ViPE 深度继承的**, 不是 fp_pose 造成的。
→ 必须保住 interim 才能验证, 批量已改用 `--keep-interim`(laptop 那条的 interim 已被清掉)。

**② 手的真值审计(首次)** —— `NOISY_WORLD_FRAMEWORK` 三大缺口第一条:
| | 深度比(我们/真值) | 总误差 | 深度分量 | 横向分量 |
|---|---|---|---|---|
| 左手 | 1.115 | 185mm | 86mm | 164mm |
| 右手 | 1.058 | 178mm | 48mm | 169mm |
| 物体 | 1.233 | 239mm | 236mm | 38mm |
**手的深度比物体准得多**(6–12% vs 23%), 手的误差以横向为主(可能含 MANO trans 约定/betas 偏移)。
→ 同一次重建内部, 手与物体的深度**不自洽 11–17%**, 这个矛盾**不需要真值就能观测**。

**③ 手的米制尺寸是合理的**(掌长 9.29cm, 成人 9–11cm; 全长 15.8cm 略偏小)。
→ 人手是每条第一人称视频天然自带的**外部米制尺子**, 这是"通用判据"的关键锚点。
   (纯内部一致性检查**永远发现不了**全局尺度错误 —— 这是原理性的, 必须有外部参照。)

**候选信号(只用我们自己的产物, 真值仅事后验证)**, laptop 实测:
- `hand_palm_cm` 9.29 / `hand_total_cm` 15.83 —— 人手先验锚点
- `contact_gap_mm` 162.8 / `contact_gap_min_mm` 55.1 —— 2D 判接触(与深度无关)时的 3D 间隙, 物理上应≈0
- `obj_over_hand_depth` **1.189** —— 物体比手深 19%(真实深度误差 1.358, 量级吻合)
⚠ n=1 不能下任何结论。评测脚手架 `tools/arctic_eval/arctic_eval.py` 已写好并在 laptop 上验证。

**踩坑记录**:
- 接触间隙的第一版太粗糙:左手接触帧占 90%(判据太宽松), 接触/非接触间隙几乎一样
  (左 283 vs 302, 右 43 vs 31)。已改为**只用五个指尖 + 抽样**, 仍需在多条上验证。
- 我 kill 掉第一次批量, 在 ketchup 的 `instance_pipeline_v17a` 留下半截目录 →
  v17A 拒绝写入非空目录 `FileExistsError` → **整批 10 条全部中止**。
  已修 `reconstruct.sh`:批量里单条失败跳过并记账, 全败才 exit 1(commit 5c59763)。
- 另一个 agent 的**VLM 透明门已接进 reconstruct.sh**, 还多了 preflight 硬链接镜像
  (它也会丢开头暗帧, 与我的 `arctic_ego_video.py` 重复但无害)。
