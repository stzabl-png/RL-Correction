# Clean/3 判据与奖励合约 (A4, 2026-09-07 初版; 数字待 A0 放音标定)

判据全部在**盘规范系** (z-up, 原点盘心) 里算, 实现 = `progress_batch.py` (无 Isaac 依赖, `selftest_progress.py` CPU 自检)。

## 几何
- 盘: ⌀18cm 浅碟, 顶面径向剖面 top(r): r<4.5cm 平底 −0.73cm, 4.5~8.5cm 斜坡升到缘 +1.22cm (mesh 实测, 与母带构建器同源)。
- 海绵: 规范系 7.5(x)×13.3(y)×3.8(z)cm, 擦盘面 = −z 面, 离原点 1.91cm。刚体海绵落座高度 = 足迹最高支撑点 + 1.91cm
  (13.3cm 长跨在 9cm 碟心平底与斜坡缘之间; `CleanSignals.seat_height`)。

## 信号 (`CleanSignals.__call__`)
- 足迹采样点 (7×13 @1cm) 沿海绵 −z 偏 1.91cm 后, 逐点离盘面高度 gap = z − top(r); **擦到** = gap < 5mm ∧ r ≤ 8.5cm。
- `contact` = 任一足迹点擦到; `plate_tilt` = 盘规范 z 轴与世界 z 夹角。

## 进度与成功 (`CleanProgressBatch.step`)
- 覆盖率 = 被"擦到"的 1cm 格 ∩ 盘面圆 / 盘面圆格数 (earn-only, 单调)。
- 行程 = 接触中的海绵中心面内位移累计。
- 保持 = 盘倾角 < 15° ∧ 盘心离参考位 < 3cm ∧ 双物未掉 (掉 = 相对手漂移 >5cm 或落到桌面)。
- gates: [首次接触, 覆盖 ≥ ½X, 覆盖 ≥ X, **成功** = 覆盖 ≥ X ∧ 行程 ≥ Y ∧ 保持]。
- 奖励: 10×Δ覆盖 + 5×Δ行程/Y (接触中) + 里程碑 0.5/1/4/12。出盘沿部分不计分不罚 (用户裁定 §5.0③)。
- X (`cover_min`) = **0.45**, Y (`travel_min`) = **0.70m** (2026-09-07 定案): 参考自身几何成绩 覆盖 0.565 / 接触中行程 87cm 的 80%
  (L5-27 铁则)。碟心 (r<4.5cm) 先不算 (用户裁定 §5.6): 刚体 13.3cm 海绵平放碰不到 9cm 碟心平底, 覆盖率分母仍是整个盘面圆,
  参考能达的上限即 0.565。"擦到"的力条件 (海绵↔盘法向力 ≥2N) 待 A5 接 aux 接触传感器后加, 阈值用零动作放音标定。

## 与训练 env 的接口 (A5 接线时)
env 每步: `sig = S(plate_pos, plate_quat, sponge_pos, sponge_quat)`; `out = P.step(sig, plate_pos, dropped)`;
`out["task_reward"]` 进 ep_rew/task, `out["success"]` 是唯一成功口径; 盘 HOLD_POSE 渐进罚与 D1 掉落终止在 env 层。
