# 相机内参标定 Procedure

> 适用脚本：`intr_calib_charuco.py`（ChArUco，主用）
> 靶标生成：`make_charuco_pdf.py`
> 本文只描述**仓库里实际存在的代码路径和常量**；建议值标了「建议」，硬性门禁标了行号。

---

## 0. 一句话前提：什么能标出来，什么标不出来

单目标定里 **绝对尺度是不可观测的**。把 objpoints 整体乘 s，投影时 s 在除以 z 时原地约掉，
`(K, dist, R, t)` 和 `(K, dist, R, s·t)` 产生**完全相同的像素坐标和残差**。

推论，直接决定下面每一步的优先级：

| | 影响 K / dist 吗 | 怎么办 |
|---|---|---|
| 方格边长填错（整体） | **不影响** | 填个大概就行，不用卡尺 |
| 打印 x/y 缩放比例不一致 | **影响 `fx/fy`** | 激光打印 + 印后量两个方向 |
| 板子不平（翘曲） | **影响，且最大** | 贴刚性板，这是 A1 尺寸下的头号误差源 |
| 位姿多样性不够 | **影响，尤其 dist** | 见 §3，这是最容易做砸的一步 |

> ⚠ **外参标定的结论和这里完全相反** —— 那边 tag 尺寸是真值，必须量准。别把这张表带过去。

畸变系数 `k1,k2,k3` 的信息量**几乎全部集中在画面边缘和四角**。板子只在画面中间晃，会得到一个
RMS 很漂亮但 dist 基本没被约束的 K，拿去对画面边缘的目标做 PnP 就系统性地偏。

---

## 1. 靶标准备

### 1.1 生成

三块板已生成在 `assets/charuco/`，参数只有 square 边长不同（10×7 / DICT_5X5_50 保持一致）：

```bash
python make_charuco_pdf.py                          # A1，square 75 mm
python make_charuco_pdf.py --page a2 --square-mm 55 # A2，square 55 mm
python make_charuco_pdf.py --page a3 --square-mm 38 # A3，square 38 mm
```

每条产出 3 个文件：`.pdf`（打印）、`_board_only.png`（自检 / 仿真贴图）、`.yaml`（配置记录）。

| 板 | 板面 | 页边距 | square | marker | 建议工作距离 |
|---|---|---|---|---|---|
| A1 | 750×525 mm | 45.5 / 34.5 mm | 75 mm | 56.25 mm | 0.35–0.7 m |
| A2 | 550×385 mm | 22.0 / 17.5 mm | 55 mm | 41.25 mm | 0.25–0.5 m |
| A3 | 380×266 mm | 20.0 / 15.5 mm | 38 mm | 28.5 mm | 0.18–0.35 m |

三块板全部 54 内角点 / 35 marker，已 round-trip 验证检出 54/54、35/35。

选板依据：**最近的那批视角，板子要横跨画面宽度的 50%~70%**。视场角 θ 的相机在距离 d 处
画面宽 `W = 2·d·tan(θ/2)`，ZED X Mini 约 110° → `W ≈ 2.86·d`。

> ⚠ **三块板的 marker id 完全相同（0–34）。绝对不要让两块板同时进画面**，检测器会把它们
> 当成同一块板的角点，解出来的位姿是垃圾。

### 1.2 打印

1. **激光打印机。** 不是因为尺度更准（激光 ~0.1–0.2% vs 喷墨 ~0.2–0.5%，同一数量级，
   而且都能靠测量修掉），而是因为标定板黑色覆盖率约 50%，喷墨在重墨量下会产生
   **cockle（局部起皱）** —— 这是局部、非均匀的形变，**全局测量修不掉，贴刚性板也压不回去**。
2. **打印对话框选「实际大小 / 100%」，绝对不要选「缩放以适合可打印区域」。**
   这条比用什么打印机重要。三块板的页边距都远超打印机 4–5 mm 的硬件极限，可以放心选 100%。
3. **贴到刚性平面上**：铝塑板、泡沫板、亚克力。A1 尺寸下 1 mm 的纸张翘曲，在 0.5 m 斜视角下
   角点重投影偏约 **2 px**，比亚像素噪声（~0.2 px）大一个数量级。**这是这个尺寸下的单一最大误差源。**

### 1.3 印后验证（3 分钟，必做）

量**横向 10 格总长**和**纵向 7 格总长**（A1 标称 750 / 525 mm）：

| 结果 | 判断 |
|---|---|
| 两个方向缩放比差 <0.1% | ✅ 直接用 |
| 差 0.1%–0.3% | ⚠️ 能用，但这就是 `fx/fy` 比值的误差下限，记录在案 |
| 差 >0.5% | ❌ 重打，多半是打印机偷偷缩放了 |

绝对值偏了不用管（见 §0）。

---

## 2. 脚本配置

改 `intr_calib_charuco.py` 顶部的宏，或用命令行覆盖。

### 2.1 靶标（`intr_calib_charuco.py:31-36`）

当前默认已是 A1 板（`:31-36`）：

```python
CHARUCO_SQUARES_X = 10
CHARUCO_SQUARES_Y = 7
CHARUCO_SQUARE_LENGTH = 0.075     # 米
CHARUCO_MARKER_LENGTH = 0.05625
CHARUCO_DICTIONARY = "DICT_5X5_50"
CHARUCO_LEGACY_PATTERN = False
```

换用 A2 / A3 板时**只改 `CHARUCO_SQUARE_LENGTH`**（0.055 / 0.038），其余不动。
或命令行：`--square-length 0.055 --marker-length 0.04125`。

> 填错方格数或 dictionary → 一个角点都检不出来，会立刻发现。
> 填错边长 → 不报错，但对 K/dist 无害（§0）。

### 2.2 相机（`:71-79`）

```python
DEFAULT_CV2_SOURCE   = "0"          # 或 "/dev/video4"
DEFAULT_CV2_PORT     = "5-3:1.0"    # USB port id，覆盖 --src
DEFAULT_CV2_WIDTH    = 2592
DEFAULT_CV2_HEIGHT   = 1944
DEFAULT_CV2_FPS      = 50
DEFAULT_CV2_FOURCC   = "MJPG"
DEFAULT_DISPLAY_SCALE= 0.4          # 只影响显示窗口，不影响存盘分辨率
CAMERA_MODEL         = "pinhole"    # 鱼眼镜头改 "fisheye"
```

**采集分辨率优先级最高的两件事：**

1. **FOURCC 能用未压缩就用未压缩。** `MJPG` 意味着相机吐出来的已经是 JPEG 压过的，
   存成 PNG 只是把有损数据无损地抄一遍。文件里注释掉的 D435 预设用 `YUYV` 就是为此。
2. **不要在降采样后的流上标定。** 亚像素角点定位的精度直接随分辨率走。

`configure_capture()`（`:336`）只设 fourcc/宽/高/fps/buffersize，**不碰曝光、增益、白平衡**。
标定前请手动锁死曝光并压短快门（宁可加环境光），否则转板子时自动曝光来回跳 + 运动模糊，
采到的帧会被 §4 的清晰度筛选砍掉，等于白采。

---

## 3. 采集

```bash
python intr_calib_charuco.py
```

窗口按键（`:2531-2549`）：

| 键 | 作用 |
|---|---|
| `s` | 手动存当前帧 |
| `p` | 暂停 |
| `c` | 清空缓冲 |
| `h` | 显示/隐藏帮助 |
| `q` / `ESC` | 结束并进入标定流水线 |

**不用一张张按快门** —— `AUTO_SAVE_VALID_IMAGES = True`（`:95`），检测合格就自动存，
冷却 0.8 s。举着板子走一遍位姿即可，**3–5 分钟**。

### 3.1 单帧准入门禁（`charuco_detection_quality`，`:619`）

| 常量 | 值 | 含义 |
|---|---|---|
| `CHARUCO_MIN_CORNERS_PER_SAMPLE` | 24 | 至少 24/54 个角点 |
| `CHARUCO_MIN_GRID_ROWS_PER_SAMPLE` | 2 | 至少跨 2 行 |
| `CHARUCO_MIN_GRID_COLS_PER_SAMPLE` | 4 | 至少跨 4 列 |
| `CHARUCO_MIN_BOARD_BBOX_FRACTION` | 0.35 | 角点包围盒 ≥ 板子内角点网格的 35% |

> ⚠ `BBOX_FRACTION` 算的是**角点占板子自身网格的比例，跟画面大小无关**。
> **脚本里没有任何一条约束在管「板子在画面里够不够大」。** 站 2 米外拍 A4，
> 只要解出 24 个角点它照样欢天喜地地存。板子大小只能靠 §1.1 选对，脚本不兜底。

### 3.2 位姿配方（这一步决定 dist 的质量）

目标 **100–200 张有效检测**。低于 100 张，经过 §4 的漏斗后可能就贴着 `MIN_SAMPLES = 20` 的下限了。

**关键是位姿多样性，不是张数。** 同一个角度拍 200 张对解 fx/dist 毫无帮助。按下面走一遍：

1. **正对，远近各一批** —— 覆盖建议工作距离的两端
2. **画面四角各一批** —— 板子的角要**压到画面的角**。这批数据决定 `k1,k2,k3`，**最重要、最容易漏**
3. **绕两个水平轴倾斜 ±30°~45°** —— 解耦 fx/fy 与 z，斜视角是分离焦距和距离的唯一手段
4. **绕光轴旋转 90° / 180°** —— 抵消打印各向异性和板子非平面的系统性偏差
5. **上下左右平移扫一遍** —— 约束主点 cx/cy

倾斜别超过约 60°，角点会开始被自身遮挡且定位精度掉得很快。

---

## 4. 按 `q` 之后：标定流水线

6 个阶段，终端有进度条（`FINAL_PIPELINE_STAGE_COUNT = 6`）：

| # | 阶段 | 做什么 | 相关常量 |
|---|---|---|---|
| 1 | Accelerator setup | 探测 OpenCL/GPU | `FINAL_USE_GPU_FOR_SHARPNESS` |
| 2 | Sharpness analysis | 在**校正后的板子裁剪区**算 Laplacian 方差 + Tenengrad 梯度，按板子表观尺度分箱归一化（避免"远而清晰"被误判为糊），砍掉最糊的 15% | `FINAL_BLUR_REJECT_FRACTION = 0.15` |
| 3 | Pose diversity | 最远点采样，截到 72 个视角 | `FINAL_MAX_CALIBRATION_VIEWS = 72` |
| 4 | Robust calibration | `cv2.calibrateCamera`，剔除单视角重投影 >0.8 px 的，最多 5 轮 | `FINAL_MAX_VIEW_ERROR_PX = 0.8`<br>`FINAL_MAX_REJECTION_ROUNDS = 5` |
| 5 | Cross-validation | 2-fold，分别训练并在 holdout 上跑 PnP | `FINAL_CROSS_VALIDATE = True` |
| 6 | Save outputs | 写 yaml + 诊断目录 | |

漏斗形状：

```
采 100–200 张有效检测
  → 砍最糊的 15%
  → 截到 72 张
  → 剔除重投影 >0.8 px（最多 5 轮）
  → 最终 40–70 张参与标定（硬下限 MIN_SAMPLES = 20）
```

### 4.1 产物

- **内参 yaml**：`outputs/intrinsics_{camera}_{model}_{target}_{W}x{H}_{时间戳}.yaml`
- **采集原图**：`outputs/intrinsics_charuco_samples/<MMDD_HHMMSS>/*.png`
  （**无损 PNG，且存的是原始帧不是标注帧**；`--display-scale` 不影响存盘）
- **诊断目录**：`selection_report.csv` + 三张 contact sheet
  （selected / rejected_blur / rejected_reprojection）

---

## 5. 验收标准

打开产出的 yaml，按顺序查：

| 项 | 合格线 | 不合格说明 |
|---|---|---|
| `rms` | **< 0.5 px** | >1.0 px：板子不平、运动模糊、或方格数填错 |
| `num_samples` | **40–70** | <25：采太少或被筛太狠，回去补采 |
| 交叉验证 `fx_range_px` / `fy_range_px` | **< 2 px** | 大：位姿多样性不够，两折看到的是不同分布 |
| `mean_holdout_pnp_rmse` | 与 `rms` 接近 | 明显更大 = 过拟合到训练视角 |
| `all_view_pnp.pnp_p90` | 与 `pnp_rmse` 同量级 | 长尾说明有一批坏视角混进来了 |
| `cx, cy` | 接近图像中心 ±5% | 偏太多：多半是分辨率/裁剪没对上 |

**参考基线**（本仓库跑过的最好一次，`outputs/intrinsics_charuco_offline_eval_0730_162254_0730_162618/`）：
259 有效视角 → 筛出 52 张，`rms = 0.502 px`，2-fold `fx_range = 1.62 px`。

### 5.1 `dist` 的量级 —— 免费的链路自检

**本该约等于 0 的就应该约等于 0。**

| 观测到的 `k1` | 结论 |
|---|---|
| \|k1\| < 0.01 | 图像是 **rectified** 的（ZED SDK 默认输出就是这样），符合预期 |
| \|k1\| ~ 0.05–0.3 | 图像是**原始未校正**的 |

如果你以为拿的是 rectified 流、结果 k1 出来 -0.1，说明**取流链路和你以为的不是同一条**，
或者你标的这一路和之后消费的那一路不是同一路。这一步不花任何额外成本，标完第一件事就看它。

`extr_calib.py` 里也内建了这个检查：`load_intrinsics_yaml()` 在 `max|coeff| > 0.01` 时会打日志。

---

## 6. 怎么用这份内参

**K 是有分辨率单位的。** 1920×1080 标定、640×360 消费时必须缩放：

```python
sx, sy = new_w / calib_w, new_h / calib_h
K[0, :] *= sx      # fx, skew, cx
K[1, :] *= sy      # fy, cy
# dist 定义在归一化坐标上，与分辨率无关，原样不动
```

仓库里现成的实现：
- `extr_calib_fingertip.py:158` `scale_intrinsics()`
- `extr_calib.py` `CamTagCalibrator.intrinsics_for_frame()`（分辨率不一致时自动缩放并告警）

下游消费必须同时带上 **K + dist + image_size + camera_model** 四样，缺一不可：
`extr_calib_d435_cube_cv2_apriltag_grid.py:743` 会直接断言 `camera_model == "pinhole" and dist.size == 5`。

---

## 7. Troubleshooting

### 采集阶段

| 症状 | 原因 | 处理 |
|---|---|---|
| 一个角点都检不出来 | `CHARUCO_SQUARES_X/Y` 或 `CHARUCO_DICTIONARY` 填错 | 对照 `assets/charuco/*.yaml` 核对；先拿 `_board_only.png` 喂进去自检 |
| 检出的 marker 数忽多忽少 | 光照不均 / 反光 | 换漫射光源，板子别用光面相纸，避开正对光源的镜面反射 |
| 只有部分区域检得出 | 板子翘了，或景深不够 | 贴刚性板；缩小光圈换景深 |
| HUD 一直显示 `board bbox 0.22 < 0.35` | 只看到一小块板 | 靠近，或换小一号的板 |
| HUD 显示 `corners N < 24` | 同上，或对焦不准 | 同上 |
| 自动存图不触发，原因显示 `cooldown` | 正常，0.8 s 冷却 | 慢一点移动即可 |
| 自动存图不触发，原因显示 `same frame` | 相机没出新帧 | 检查 fps/fourcc 组合相机是否真支持 |
| 相机打不开 | 见下 §7.1 | |

### 7.1 相机打不开

`start_capture()`（`:366`）只走 `cv2.VideoCapture(..., cv2.CAP_V4L2)`。

- **USB / UVC 相机**：用 `--port` 指定 USB port id（如 `5-3:1.0`）比 `--src 0` 稳，
  设备号会随插拔变化，port id 不会。
- **GMSL / 专有 SDK 相机（如 ZED X Mini）**：**cv2 打不开**。
  ZED 走 ZS01 TCP 流或 zenoh，必须先把帧存成 PNG，再走离线路线：

  ```bash
  git show fd30f43:offline_intr_calib_charuco.py > offline_intr_calib_charuco.py
  python offline_intr_calib_charuco.py --image-dir <存图目录> --source-yaml <参考内参yaml>
  ```

  默认 `--min-views 40 --max-views 80`，与 §3.2 的建议一致。
  采图时务必起全分辨率（`zed_streamer --resolution HD1080`），**不要用默认缩到 640×360 的
  JPEG 流去标** —— 压缩伪影 + 3 倍降采样直接废掉一半亚像素精度。

### 标定阶段

| 症状 | 原因 | 处理 |
|---|---|---|
| `rms` > 1.0 px | 板子不平 | 贴刚性板重来。这是最常见的原因 |
| | 运动模糊没被筛干净 | 锁曝光、压快门；或调大 `FINAL_BLUR_REJECT_FRACTION` |
| | 方格边长填错到离谱 | 不影响 K，但检查一下有没有别的填错 |
| `fx/fy` 明显偏离 1.0（方像素传感器） | 打印各向异性 | 回 §1.3 量板子 |
| | 图像被非等比缩放过（如 1920×1200 压成 1920×1080） | 查取流链路 |
| 交叉验证 fx 两折差很多 | 位姿多样性不够 | 按 §3.2 补采，重点补四角和斜视角 |
| `dist` 里 k3 很大且 k1/k2 反号 | 高阶项在硬凑，画面边缘数据不足 | 补四角视角；或考虑固定 k3 |
| 报 `MIN_SAMPLES` 不足 | 采太少 / 被筛太狠 | 看 `selection_report.csv` 和 rejected contact sheet，确认是糊还是重投影超标 |
| 鱼眼镜头标定不收敛 | `CAMERA_MODEL` 还是 `pinhole` | 改成 `"fisheye"`（`:93`）。注意 dist 会变成 4 个系数，下游 pinhole 的 solvePnP 用不了 |

### 结果可疑

| 症状 | 检查 |
|---|---|
| PnP 出来的距离系统性偏大/偏小 | `fx` 错。**拿卷尺量准距离摆一次板子跑 PnP** —— 这是整条链上唯一真值来自物理世界的检查 |
| 画面中心准、边缘偏 | `dist` 没标好。补四角视角重标；或确认消费端有没有把 dist 一起用上 |
| 换了分辨率后全错 | K 没缩放。见 §6 |
| 和别人给的内参差 1.8 倍 | 两份内参对应的分辨率/裁剪不同，或其中一份是标称兜底值不是标定值 |

---

## 8. Checklist

采集前：

- [ ] 板子已印，激光，100% 实际大小
- [ ] 已量横纵跨距，两方向缩放比一致（<0.1%）
- [ ] 已贴刚性平面，目视无翘曲
- [ ] `CHARUCO_SQUARE_LENGTH` 与所用板子匹配
- [ ] 分辨率拉满、FOURCC 尽量未压缩
- [ ] 曝光/白平衡已手动锁死，快门够短
- [ ] 画面里只有一块板

采集中：

- [ ] 有效检测 100–200 张
- [ ] 板子的角压到过画面的**四个角**
- [ ] 有远、有近、有 ±30°~45° 倾斜、有绕光轴 90°/180°

标定后：

- [ ] `rms < 0.5 px`
- [ ] `num_samples` 落在 40–70
- [ ] 交叉验证 `fx_range_px < 2`
- [ ] `dist` 量级与「是否 rectified」的预期一致
- [ ] 卷尺距离检查通过
- [ ] yaml 里 K / dist / image_size / camera_model 四样齐全
