# 相机外参（手眼）标定 Procedure

> 适用脚本：`extr_calib.py`
> 靶标生成：`make_apriltag_grid_pdf.py`（16 格）、`make_apriltag_pdf.py`（单 tag）
> 场景：AprilTag 装在 Dexmate 的手/腕 link 上，头部 ZED X Mini 观察，求 `X_CammountCam`
> 内参标定见 [MW-intrinsics-measure.md](MW-intrinsics-measure.md)。本文假定内参已有。

---

## 0. 和内参的关键区别

**外参的绝对尺度是真值。** 内参那套「方格边长填个大概就行」在这里**完全反过来**：

`X_CamTag` 的平移量与 tag 尺寸成正比，而手眼方程的另一半（机器人正运动学）自带真实米制尺度。
两边尺度不一致时，求解器给出的不是"缩放过的正确答案"，而是一个**有偏且自相矛盾的拟合**。

| | 内参 | 外参（本文） |
|---|---|---|
| tag / 方格尺寸填错 | 无害 | **直接等比例污染平移** |
| 印后必做 | 量两个方向的比例 | 量绝对跨距，且要准 |
| 板子平整度 | 头号误差源 | 同样重要 |

---

## 1. 靶标

三张，全部 A4、100% 实际大小、激光打印、贴刚性板。

```bash
# 单 tag —— 正式标定用这两张，优先 80 mm
python make_apriltag_pdf.py --tag-id 7 --tag-size-mm 80 --page a4
python make_apriltag_pdf.py --tag-id 7 --tag-size-mm 60 --page a4

# 16 格板 —— 备用：单 tag 结果不稳时换这张，可根治平面二义性
python make_apriltag_grid_pdf.py --page a4 --tag-size-mm 40
```

**正式用单 tag**，代价是精度低一个数量级（§2），必须靠 §4.1 的采样纪律补回来：
**靠近（≤0.4 m）、倾斜（≥25°）、多采（N ≥ 40）**。这三条不是建议，是这条路线成立的前提。

| | 16 格 | 单 60 mm | 单 80 mm |
|---|---|---|---|
| PDF | `assets/apriltag_grid/compact_apriltag_grid_4x4_tag40mm_a4.pdf` | `assets/apriltag_single/apriltag_tag36h11_id7_tag60mm_a4_portrait.pdf` | `..._tag80mm_...pdf` |
| family | tag36h11 | tag36h11 | tag36h11 |
| id | 0–15 | 7 | 7 |
| tag 边长 | 40 mm | 60 mm | 80 mm |
| 靶标外廓 | **173.9 × 173.9 mm** | 90 × 90 mm（含白边） | 120 × 120 mm（含白边） |
| 页边距 | 18.0 / 61.5 mm | 60.0 / 103.5 mm | 45.0 / 88.5 mm |
| `board_min_tags` | 4（默认） | **必须设 1** | **必须设 1** |
| 点数 | 64 | 4 | 4 |

**为什么 16 格选 40 mm**：A4 上 4×4 最大能到 44 mm（外廓 191.3 mm），但页边距只剩 9.35 mm，
贴胶带固定就会盖到最外圈黑框。40 mm 留出 18 mm 边距，精度只差 17%（下表），换来能上手装夹。

### 1.1 印后必做：量跨距

| 靶标 | 量什么 | 标称 |
|---|---|---|
| 16 格 | 最外圈**黑边到黑边**，横竖各一次 | `3 × 47.826 + 40 = 183.48 mm` |
| 单 tag | 黑框方块边长 | 60.00 / 80.00 mm |

16 格板的跨距比单个 tag 长 4.6 倍，同样的卡尺读数误差下**相对精度高 4.6 倍** —— 这是选 16 格的另一个理由。

- 偏差 <0.1% → 直接用
- 偏差更大 → 按实测值重新生成 **yaml**（PDF 不要重印，只取 yaml）：

```bash
python make_apriltag_grid_pdf.py --page a4 --tag-size-mm 39.85 --stem grid_measured
# 然后把 tag_board_yaml_path 指向 assets/apriltag_grid/grid_measured.yaml
```

### 1.2 单 tag 的两个额外注意

- **白边不能裁。** 单 tag 没有相邻格子提供 quiet zone，PDF 上画了虚线，**沿虚线或更外面裁**。
- **平面位姿二义性。** 四个共面点的 PnP 有两个互为镜像的解，两者重投影误差几乎一样，
  所以 `max_reproj_px` **抓不到它**。为此加了独立的 `min_ambiguity_ratio` 门禁，见 §3.3。

---

## 2. 精度预期

以下都是用仓库真实求解器跑出来的，不是估计值。
相机参数按 ZED X Mini：1920×1080，110° HFOV → fx ≈ 672 px；角点噪声按 tag 像素尺寸建模。

### 2.1 单帧 `X_CamTag` 误差

| 靶标 | 外廓 | @0.4 m | @0.6 m |
|---|---|---|---|
| 单 60 mm | 90 mm | 0.515° / 0.68 mm | 0.971° / 1.90 mm |
| 单 80 mm | 120 mm | 0.323° / 0.44 mm | 0.620° / 1.23 mm |
| **16 格 40 mm** | 174 mm | **0.069° / 0.10 mm** | **0.136° / 0.29 mm** |
| （16 格 44 mm，供对比） | 191 mm | 0.059° / 0.09 mm | 0.116° / 0.25 mm |

### 2.2 手眼解出的 `X_CammountCam` 误差

`calibrate_cammount_and_tag_prob` 的输出，15 次中位数。零噪声下该求解器精确复原真值（`0.0 deg / 6.6e-10 m`），所以下表全是噪声传播。

| 靶标 | 距离 | N=10 | N=20 | N=40 |
|---|---|---|---|---|
| 单 80 mm | 0.4 m | 0.08° / 1.36 mm | 0.06° / 0.85 mm | 0.04° / 0.54 mm |
| 单 80 mm | 0.6 m | 0.18° / 3.09 mm | 0.13° / 1.92 mm | 0.09° / 1.22 mm |
| 16 格 | 0.4 m | **0.01° / 0.15 mm** | 0.01° / 0.13 mm | 0.00° / 0.09 mm |
| 16 格 | 0.6 m | 0.02° / 0.44 mm | 0.02° / 0.26 mm | 0.01° / 0.19 mm |

**三条结论：**

1. **16 格板 N=10 已经比单 tag N=40 好 4–8 倍。** 单 tag 要追平得采几百帧。
2. **误差按 √N 降。** 单 tag 从 N=10 到 N=40，3.09 → 1.22 mm。
3. **距离比张数更有效。** 单 tag 从 0.6 m 挪到 0.4 m，同样 N=10，3.09 → 1.36 mm，直接对折。

> ⚠ 上表假定采样时有**大幅姿态变化**（仿真用 ±50° 旋转）。只在一小块区域平移，N 再大也解不出旋转分量。

---

## 3. 配置 `extr_calib.py`

`ExtrinsicsCalibConfig`（`extr_calib.py:351`）：

```python
K, dist, image_size = load_intrinsics_yaml(Path("outputs/<你的内参>.yaml"))

config = ExtrinsicsCalibConfig(
    robot_urdf_path=Path("../assets/vega_1_sharpa.urdf"),
    cammount_link_name="vega_1_head_l3",   # 相机装在头部末端 link
    tagmount_link_name="right_hand_flange", # tag 装在手/腕 link
    K=K,
    dist=dist,                              # 必传，见 §3.1
    image_size=image_size,
    tag_board_yaml_path=Path(
        "assets/apriltag_single/apriltag_tag36h11_id7_tag80mm_a4_portrait.yaml"),
    board_min_tags=1,                       # 单 tag 必须是 1，默认 4 永远满足不了
    max_reproj_px=2.0,
    min_ambiguity_ratio=3.0,                # 单 tag 专用，见 §3.3
    output_file_path=Path("outputs/extrinsics_vega_head.yaml"),
    ransac_sample_size=8,                   # 攒够这么多才开始解算
)
```

| 字段 | 行 | 说明 |
|---|---|---|
| `robot_urdf_path` | :353 | 必须和真机一致 |
| `cammount_link_name` | :356 | 解出的 `X_CammountCam` 是**对 URDF 标称值的修正** |
| `tagmount_link_name` | :357 | |
| `dist` | :368 | 不传会告警并按已校正处理 |
| `image_size` | :372 | 实时帧分辨率不同会自动缩放 K |
| `board_min_tags` | :383 | **单 tag 必须设 1**，默认 4 永远满足不了 |
| `max_reproj_px` | :387 | 超过就拒收该帧 |
| `min_ambiguity_ratio` | :393 | 单 tag 专用，多 tag 板自动失效。见 §3.3 |
| `ransac_sample_size` | :396 | 8 |

Vega URDF 里相机的标称外参在 `zed_left_camera_mount`：相对 `vega_1_head_l3`
`xyz = 0.0365 -0.023 -0.0489`，`rpy = 1.5708 0 3.1416`。
注意 `zed_depth_frame` 沿 x 少 11.5 mm（`0.025`），只在融合深度时才用后者。

### 3.1 dist 一定要传

`extr_calib.py` 在**畸变原图**上检角点，配正确的 `dist` 走 `solvePnP` 才是对的，不需要先 undistort。
不传 `dist` 的代价实测过：600 px 焦距、k1=-0.28 的镜头，板子偏画面一角时 **33.5 mm / 5.28° 的系统性偏差**。

ZED SDK 默认输出已校正，所以这份内参的 `dist` 应该 ≈ 0（`|k1| < 0.01`）。
**如果 k1 出来是 -0.1 这种量级，说明取流链路和你以为的不是同一条**，先去查链路再标外参。

### 3.2 Vega 不是 xArm6，三处硬门禁要绕开

仓库自带的两个示例（`:692`、`:734`）是 xArm6 的，Vega 有 **70 个 actuated joint**，直接用会抛异常：

| 位置 | 内容 | 处理 |
|---|---|---|
| `:639` `assert_xarm6_example_model` | 断言 `ndof == 6` | **不要调用** |
| `:680` `xarm6_joint_values` | 断言传入正好 6 个关节值 | **不要调用** |
| `:474` `vis_step(..., input_joint_names=None)` | 70 自由度必须显式传 | **必须传 `input_joint_names`**，让它按 URDF 顺序重排 |

### 3.3 单 tag 的二义性门禁

`apriltag_board.square_tag_pose_ambiguity()` 用 `SOLVEPNP_IPPE_SQUARE` 一次取出**两个**解，
返回**次优解重投影误差 / 最优解重投影误差**这个比值。比值接近 1 = 两个解拟合得一样好 = 报出来的位姿是抛硬币。
低于 `min_ambiguity_ratio` 的帧直接拒收。

对**多 tag 板它返回 `None`，门禁自动失效** —— 只在"恰好 4 个点、且构成以原点为中心的正方形"时才启用，
栅格板里的 tag 相对板心有偏移，不满足条件。已验证 16 格板走这条路返回 `None`。

**实测（80 mm tag，1920×1080，fx≈672，0.3 px 角点噪声，各 300 次）：**

| 倾角 | 距离 | 比值中位数 | 单帧旋转误差 | 门禁@3.0 放行率 |
|---|---|---|---|---|
| 3° | 0.4 m | 2.24x | 1.55° | **34%** |
| 3° | 0.6 m | 1.94x | 3.35° | **23%** |
| 10° | 0.4 m | 10.75x | 1.02° | 100% |
| 10° | 0.6 m | 5.24x | 1.79° | 86% |
| 25° | 0.4 m | 31.18x | 0.52° | 100% |
| 40° | 0.4 m | 51.72x | 0.35° | 100% |

**它实际起的作用**：在你的工作距离上，把"正对"的帧挡掉大半（那些误差是倾斜帧的 3–4 倍），
倾斜 ≥10° 的帧全部放行。等于把 §4.1「要倾斜」那条纪律**自动强制执行**，不靠人记。

> ⚠ **诚实的边界：它救不了"又远又正对"。** 实测 1.2–1.5 m + 4° 倾角时，
> 放行的 11–18% 帧翻转率仍有 39–46%，和不加门禁一样。噪声大到这个程度，
> IPPE 自己对两个解的排序就不可靠了。
> **那个工况唯一的解法是不要采它** —— 靠近到 0.4 m 就不会出现。
>
> 顺带验证过：把求解器换成 `IPPE_SQUARE` 最优解并不比现在的 `ITERATIVE + refineLM` 好
> （近距离反而略差），所以求解器保持不变。

---

## 4. 采集

```bash
python extr_calib.py    # 改 __main__ 调你自己的 Vega example
```

浏览器开 `http://localhost:8080/`。界面上有：

| 控件 | 作用 |
|---|---|
| `detection` 文本 | 当前帧状态，`OK 1 tags, reprojection 0.28 px, ambiguity 18.4x` 或 `REJECT ...` |
| `click_and_append` | 追加当前观测；攒够 `ransac_sample_size` 后自动解算 |
| `click_and_save` | 存当前估计到 `output_file_path` |

**先看 `detection` 再点 append。** 三道门禁任一不过，`X_CamTag` 就是 `None`，append 被拒绝并在终端打
`Not appended: <原因>` —— 不会把上一帧的陈旧位姿混进数据。三道分别是：

1. 检测/PnP 本身失败（tag 数不够 `board_min_tags`）
2. 重投影误差 > `max_reproj_px`
3. **二义性比值 < `min_ambiguity_ratio`**（仅单 tag）—— 看到这条就是在提醒你**把 tag 转过来一点**

### 4.1 采样配方

**单 tag 目标 N ≥ 40**（16 格板 20–40 即可）。

1. **距离**：尽量把 tag 送到 **0.4 m 以内**。这是收益最大的一条。
2. **姿态**：让腕关节**绕不同轴大幅转**（±30°~50°）。两个理由叠加：手眼求解需要**旋转轴不平行**的
   多组运动（只平移不转，方程秩不足，旋转分量解不出来）；单 tag 还额外需要倾斜来压掉二义性
   —— 倾斜 25° 时单帧误差 0.52°，正对时是 1.55°，差 3 倍。**正对的帧会被门禁直接挡掉。**
3. **覆盖画面**：tag 要出现在画面中心、四角、上下左右，别只在一个位置。
4. **头部/躯干也动一动**：相机侧的链路（torso 3 + head 3 关节）也要覆盖到不同构型。
5. **每次点 append 前让机器人停稳**，运动模糊会直接抬高重投影误差被拒收。

### 4.2 输出

`output_file_path` 写入：

```yaml
X_CammountCam:  # 4x4，相机相对 cammount_link 的位姿
X_TagmountTag:  # 4x4，tag 相对 tagmount_link 的位姿
```

---

## 5. 验收

| 检查 | 怎么做 | 合格线 |
|---|---|---|
| 每帧重投影 | 看 `detection` 文本 | 稳定 < 1.0 px（门禁是 2.0） |
| 解的收敛性 | 继续采样，看 `X_CammountCam` 是否稳定 | 后 10 个样本带来的变化 < 0.5 mm |
| **与 URDF 标称值比对** | `X_CammountCam` 对比 `zed_left_camera_mount` 的 `xyz/rpy` | 差值应在**几毫米、几度**内 |
| 卷尺验证 | tag 摆在量准的距离，看 `X_CamTag` 平移模长 | 与实测一致 |

**第三项最重要。** 如果解出来的 `X_CammountCam` 和 URDF 标称值差了几厘米或几十度，
不是标定不准，是**坐标系或 link 名搞反了** —— 回去查 `cammount_link_name` / `tagmount_link_name`。

---

## 6. Troubleshooting

| 症状 | 原因 | 处理 |
|---|---|---|
| `detection` 一直 `REJECT no pose: Only 1 board tags detected; need at least 4` | 用了单 tag 但 `board_min_tags` 还是 4 | 设成 1 |
| `detection` 一直 `REJECT no pose`，且用的是 16 格板 | family 不对 / 白边被裁 / 板子太远 | `tag_family` 必须是 `tag36h11`；先拿 `_board_only.png` 自检 |
| `REJECT reprojection X px > 2.00 px` | 运动模糊、板子翘曲、内参不对 | 停稳再采；查板子平整度；回查内参 `rms` |
| 检出 tag 数忽多忽少 | 光照 / 反光 | 漫射光，别用光面纸 |
| 点 append 没反应 | 当前帧无有效位姿 | 看终端 `Not appended: ...` |
| 攒了 8 个样本还没解算 | `ransac_sample_size` | 继续采，或调小 |
| 解出来一直在跳 | 姿态多样性不够 | 按 §4.1 第 2 条，绕不同轴大幅转 |
| 解与 URDF 标称值差很远 | link 名搞反 | 见 §5 |
| 抛 `xArm6, not xArm7` | 调了 xArm6 的断言 | 见 §3.2 |
| 抛 `Expected 5 plumb_bob ... got 4` | 内参是 fisheye 模型 | 本脚本只支持 pinhole |
| `REJECT ambiguous pose, runner-up fits 1.9x ...` | tag 太正对相机 | **把 tag 转过来**，倾斜 25° 以上；或再靠近些 |
| 单 tag 结果时好时坏，但门禁全过 | 又远又正对，门禁在这个工况失效（§3.3） | 靠近到 0.4 m 以内重采；或换 16 格板 |

---

## 7. Checklist

打印后：

- [ ] 三张都是 A4、100% 实际大小、激光
- [ ] 单 tag 沿虚线或更外面裁，白边完整
- [ ] 全部贴刚性平板，目视无翘曲
- [ ] 已量跨距（16 格 183.48 mm；单 tag 60.00 / 80.00 mm），偏差 <0.1%
- [ ] 偏差超标的已按实测值重生成 yaml

配置：

- [ ] `dist` 已从内参 yaml 传入，量级与「是否 rectified」预期一致
- [ ] `image_size` 已传
- [ ] `tag_board_yaml_path` 指向实际打印的那张
- [ ] 单 tag 时 `board_min_tags = 1`、`min_ambiguity_ratio = 3.0`
- [ ] Vega 的 xArm6 断言已绕开，`input_joint_names` 已传

采集：

- [ ] tag 送到 0.4 m 以内
- [ ] 腕关节绕不同轴大幅转过（±30°~50°），没有一帧是正对采的
- [ ] tag 覆盖过画面中心和四角
- [ ] 头部/躯干构型也变过
- [ ] N ≥ 40（单 tag）/ N ≥ 20（16 格）
- [ ] 每次 append 前 `detection` 显示 `OK`，且 `ambiguity` ≥ 10x

结果：

- [ ] `X_CammountCam` 与 URDF `zed_left_camera_mount` 标称值差在几毫米/几度内
- [ ] 卷尺距离验证通过
- [ ] 已 `click_and_save`
