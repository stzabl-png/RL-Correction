# v17a_debug — 自动标注(HOI-DETR + SAM2 传播)的排查工具

排查"自动认物体"失败时用。**先跑 `overlay_video.py`** —— 它一眼能看出传播在哪帧跟丢，
比读 manifest 的 status 快得多（status 会把"没数据"报成"时序离群"，误导排查方向）。

| 脚本 | 回答什么 |
|---|---|
| `overlay_video.py <manifest> <mp4> <out.mp4>` | mask 叠回原视频；空 mask 帧打红字 |
| `viz_v17a.py <mp4> <instance_pipeline_v17a 目录> <out.jpg>` | 逐 episode 抽帧对照，看同一 id 在不同 episode 指的是不是同一物体 |
| `count_raw.py <manifest>` | raw_mask 的缺失/全空/极小/可用 计数 |
| `rej_stats.py <manifest>` | accepted vs rejected 的几何指标中位数对比 |
| `bypass_gate.py <manifest> <out>` | 把 rejected 全提为 accepted，用于验证"门是不是太严" |

⚠ 一律用 manifest 的 **`raw_mask`** 字段，不要用 `mask` —— 后者在被质量门拒掉的帧上是空的，
用它会看不出传播到底有没有跟住。

实测样例(2026-08-12)：
* `s07/ketchup_grab_01`：703 帧中 manifest 只覆盖 301，其中 **265 帧 raw_mask 全空**
  → 不是门太严，是 SAM2 传播整段失效（瓶子红盖+绿身+深色标签，多色物体）
* `s05/laptop_grab_01`：862 个条目**无一为空**，面积中位 73710px → 传播稳定
