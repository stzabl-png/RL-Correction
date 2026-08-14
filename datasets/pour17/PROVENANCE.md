# pour17 训练数据 stage —— 来源与**人为改动**记录

**源 take**: `R&R/Output/ReconstructOutput/egodex_auto/pour/17`(重建)
+ `.../RetargetOutput/egodex_auto/pour/17`(replay_world.npz + 两个 USD)
**任务**: egodex `test/pour/17`,142 帧 @30fps;物体 `graycup` + `proud`(Proud Source 铝瓶)

## ⚠ 本目录里的东西**不是纯重建**,含以下人为改动

| 改动 | 内容 | 为什么 | 何时还 |
|---|---|---|---|
| **杯等比缩放 0.4516** | 15.5×23.2×15.1 → **7.0×10.5×6.8 cm** | 重建杯超出 SharpaWave 抓握包络(~7.9cm),原样开训会因**现实中不存在的原因**失败;7.0cm 取自 `fingertip_middle` 包络(5.3×9.2×6.3)与已验证可抓的瓶身(6.7cm) | 重建尺度/可抓性链路成熟后 |
| **减面到 80k 面** | 杯 688k→80k、瓶 312k→80k(尺寸偏差 ≤0.21mm) | 原网格在 1024 env 下做碰撞跑不动;参照已验证可跑的 CAD(135k 面) | 不用还(工程必需) |
| 抓握模板手写 | 两手都用 `fingertip_middle` | 绕过尚在优化的 Dexonomy 合成与 selector v3 | Dexonomy 优化完成后 |

**总述**:本轮是**手动调整过、不完全自动化**的训练准备,上述每条都是"上游还没好、先手动顶上",
**不得当作设计结论**。完整欠账清单见 `docs/POUR_TRAINING_DESIGN.md` §12.2。

## 自动产出(无人为干预)

- `scene_layout.json` —— 逐物体"听手"摆放;姿态由**稳定候选→筛直立→筛开口朝上→取概率最高**裁定,
  **完全不用重建的旋转**(杯 conf_rot 仅 5.5,是 GUI 里看到倒置的根因)。
  锚帧用**物体运动起始**精化(杯 f0→f28,挪 4.2cm;瓶 f8→f10)。
- `contact_auto_object_*.json` / `confidence_complete.json` —— 从源 take 原样拷贝。
- `world_fused.npz` —— 软链到源 take(大文件不复制)。
