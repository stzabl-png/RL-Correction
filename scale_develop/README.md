# scale_develop — 选帧 & 尺度估计优化工作区

目标:优化 recon 管线里 (1) 标 mask/重建参考帧的自动选择,(2) mesh 尺度估计。
分支 `agent/v17a-auto-mask-reconstruction`,背景见仓库外调研:该分支 adapter 的
`--reconstruction-frame` 无自动生产者,现成基线是 `ego_pipeline/utils/object_io.py:61`
的 `pick_best_frame`(旧管线在用,未接入 v17A 链路)。该帧号同时决定 SAM3D 重建帧、
sam3d_scale 尺度参考帧、FoundationPose 注册帧。

## 当前选帧器 v2.1(select_frame_v2.py --hand-mode pixel)

输入 = stage B 的 video_mask_sequence.json(逐帧 accepted/rejected mask) + stage A 的
detections.json(手框) + 视频。两段式:

**第 0 层(上游天然门,不在本脚本)**: v17A 时序质量门,601→~150-220 帧 accepted。
**第一段 几何粗筛**(全部尺度不变/相对量,硬门+相对门,~200→60 帧):
  贴边剔除 | frag≥0.98 | 时间窗(±15帧)相对面积∈[0.7,1.3] | 凸度≥本物体中位×0.90 |
  填充率(area/diag²)≥本物体中位×0.75 | Laplacian 清晰度剔除最糊 25%(crop 归一 512 宽)
  幸存者按干净度(凸度rel×填充率rel×清晰度rel)取前 60。
**第二段 像素级遮挡精排**:
  SAM2(图像模式)以 HOI-DETR 手框为 prompt 逐帧出手 mask(缓存 runs/*/hand_masks/);
  occ = |手mask∩物体凸包|/|凸包| + 0.25×|手mask∩边界膨胀环(25px)|/|环|;
  凸包项天然区分手前/手后(凸物体):手在物体后≠遮挡。
  排序 = occ 按 0.05 分桶升序,桶内按干净度降序;取 top-8 出对比图。

3 视频验证(2026-08-10): ketchup f514→f162, phone f234→f278, scissors f514→f446,全部
避开抓握重遮挡帧。已知待改:top-K 时间 NMS/贴边软化/角落畸变扣分/GT 尺度闭环(见台账)。

## 第一阶段:基线选帧评测(2 条 ARCTIC 视频)

- `run_arctic_video.sh <s01/box_grab_01>` — 单视频端到端:
  ego 帧→mp4 → HOI-DETR 探测(hoidetr env) → 实例分割/SAM2 传播(sam3 env)
  → `eval_baseline_frame_selection.py`。默认 GPU 1(`GPU=0 bash ...` 可改),避开训练。
- `eval_baseline_frame_selection.py` — 对 manifest 每个物体,把 accepted mask 序列
  原样喂给 `pick_best_frame`(零改动),输出:
  - `runs/<id>/frame_selection_baseline/report.json` — 选中帧 + 逐帧 area/frag/border
  - `<obj>_chosen_f*.jpg` — 选中帧全分辨率 overlay
  - `<obj>_contact_sheet.jpg` — 均匀采样 12 帧 overlay(绿=选中,红=其他),人工判读用

评测视频(多样性:铰接大盒 vs 小瓶):`s01/box_grab_01`、`s01/ketchup_grab_01`。

## 资产位置

| 资产 | 路径 |
|---|---|
| ARCTIC 数据 | `/media/msc-auto/HDD/dataset/arctic/`(下载脚本 `download_10seq.sh`,需 ARCTIC 账号) |
| HOI-DETR 源码+ckpt | `/home/bangdu/HOI-DETR`(revision `1b36729` ✓,epoch_5.pth 大小校验 ✓) |
| SAM2 | `/home/bangdu/HumanVideo2RobotData/third_party/sam2`(sam2.1_hiera_large.pt ✓) |
| conda env | stage A→`hoidetr`,stage B/评测→`sam3`(torch 2.10,sam2 可 import ✓) |

## 之后的优化方向(调研结论,待第一阶段数据验证)

- 把 `pick_best_frame`(面积分位带+碎裂+贴边)与本分支两段**未接线**的评分合成:
  `_hand_link_occlusion_fraction`(run_instance_seed_discovery.py:188,手遮挡分,全仓库唯一)
  和 `select_component_anchor_proposals`(multi_box_components.py:415,proposal_score)。
- 在 adapter 之前打分产出 `--reconstruction-frame`,不动 adapter(它声明 selection_policy=
  upstream_explicit,本就设计为上游选帧)。
- 尺度:`sam3d_scale/run_sequence.py:66` `_estimate_scale_given_orientation`,现为单帧
  单标量 PCA 主轴比;优化空间 = 多帧融合/多轴/鲁棒回归,选帧质量直接影响它的参考帧。

## 尺度第 1 层原型:多帧跨度比(scale_extent_v1.py,2026-08-11)

去朝向化:stage1 干净帧上 mask 反投影点云的 PCA 鲁棒跨度(P2-P98)直接当物体最长边,
不信 SAM3D/FP 朝向(扫把案例证明朝向错=尺度错:obs_len 18.4cm 是对的,朝向歪导致 40cm)。
跑 foundationpose env(要 OpenEXR 读 vipe 深度)。两种聚合口径的三视频验证:

| 视频 | 原版几何 | median_all | median_topclean(npts门槛+干净度top10) | 参照 |
|---|---|---|---|---|
| 扫把 | 40.2cm (2.2×) | 18.9cm | **22.1cm** | 伪GT≥18cm,口头~25cm,Qwen 22-30cm |
| ketchup | 34.2cm (1.60×) | **20.5cm (0.96×)** | 14.3cm (0.67×) | GT 21.3cm |
| laptop | 100.3cm (3.07×) | 69.2cm | 68.6cm | GT(合拢模板)32.6cm;视频中摊开,口径失配 |

结论:跨度比全面优于几何路径,但聚合口径未定——扫把赢在 topclean(远帧截断拉低 median_all),
ketchup 赢在 median_all(topclean 选中的干净帧物体偏小)。待做:可见完整度门控(mask 是否
触边/被手截断)代替单纯点数门槛;铰接物体(laptop)最长边口径需按构型定义;第 4 条 microwave
重建完后进 compare_scale_gt.py 闭环。

## 尺度最终层:三路共识融合(scale_fuse_qwen.py,2026-08-11)

L_extent(点云跨度) x L_hand(手长锚点) x L_typical(类别常识),一次 Qwen 调用。
锚点仲裁 extent 双口径(解决上表僵局);三路一致->consensus 用 extent;extent 离群且
锚点互洽->corrected_by_prior 用 geomean(锚点);identity confidence 低->弃 typical。
已接入 chain_recon.sh(sam3d_scale 之后),修正 mesh 落 runs/<id>/scale_fuse/。

| 视频 | 原版 | 旧护栏 | 融合 | verdict |
|---|---|---|---|---|
| ketchup | 1.60x | 1.60x(pass放行) | **0.96x** | consensus(口径仲裁选 median_all) |
| laptop | 3.07x | 2.71x | **1.15x** | corrected_by_prior(extent 68.6cm 被否) |
| 扫把 | ~2x | ~2x | **22.1cm≈1x** | consensus(口径仲裁选 topclean) |

注:laptop 被 Qwen 认成"木质砧板"(conf high)但尺寸反而准——手长锚点不依赖类别认对,
这正是双锚点设计的容错。前置 track 过滤(filter_tracks.py A+B 层)与终审主体仲裁见
qwen_final_arbiter.py 头注。
