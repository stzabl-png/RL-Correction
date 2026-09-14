# Clean3_hold_s42 —— Clean/3 Stage-1 抓稳段 (Denso GPU0, 2026-09-07 23:04 UTC 发车)

本地镜像自 Denso `~/RL_Correction/logs/Clean3_hold_s42/`; 训练仍在跑, 这里是 **2026-09-08 04:50 UTC 的快照** (10M 步 / 655 epoch, 目标 50M)。
再拉一次: `rsync -az Denso:RL_Correction/logs/Clean3_hold_s42/ logs/Clean3_hold_s42/ && rsync -az Denso:Clean3_hold_s42.log logs/Clean3_hold_s42/train.log`

| 文件 | 内容 |
|---|---|
| `train.log` | 训练标准输出 (Isaac 启动横幅 / `[phys] 难度覆写` / `[CleanHoldEnv]` 维度 / 每 epoch `Agent Steps... FPS... Mean Rewards` / `[课程] release_row ->` 档位推进) |
| `stage1_tb/events.*` | TensorBoard 账本, 每 epoch 一点: `sr/success sr/cert term/drop_*  hold/dev_* ep_rew/{contact,hold,bonus,action,cross} shape/cross_frac curr/release_row_cur contact/*` |
| `stage1_nn/` | ckpt: `last.pth` (=ep_640, 10M 步) / `best.pth` / `ep_*_step_*_reward_*.pth` 每 40 epoch 一个 |
| `world.json` | 出生世界指纹 (物理 0.3kg/μ1/指垫1, stage1 常量, IO 维, 臂下垂补偿) —— 回放脚本核对用 |
| `videos/` | 见下 |

## videos/ —— `last.pth` 回放 (`tasks/Clean/3/C_Wiring/record_clean.py`, 确定性 mu, 4 env 并行, env0 录像)
两组课程档位各一份: `last_r35_*` = 录制时训练所在档 (release_row_cur=35, 抖动 [31,35]); `last_r50_*` = 课程起点档 (50)。
- `*_front.mp4` (正前上方, 两手+盘+海绵居中, **主看这个**) / `*_side.mp4` (侧面看整机) / `*_top.mp4` (俯视); 1280×720, 20 fps = 1 帧 1 控制行, 110 帧。
- `*_steps.csv` / `*_steps.npz`: **逐步奖惩记录**, 4 env × 110 行。列:
  `env step row release_row pinned released reward | r_contact r_hold r_bonus r_act r_cross cross cross_any c_plate c_sponge |
   dp_plate_cm dr_plate_deg dp_sponge_cm dr_sponge_deg within plate_ok sponge_ok | cert cert_run new_cert dropped drop_plate drop_sponge new_drop success done |
   act_abs_mean res_arm_max_rad res_fin_max_rad | F_L_{thumb..pinky} F_R_{thumb..pinky} (N) | gapL/gapR_{idx_mid,mid_ring,ring_pinky}_cm (相邻指尖掌系侧向有符号间距, <1cm=交叉, 负=翻转) d3L/d3R_*_cm (指尖 3D 距, <1.5cm=重叠)`
  `reward = r_contact + r_hold + r_bonus + r_act + r_cross` 逐行可核。终止行 (done=1) 的 gap/d3 列为 NaN (IsaacLab 已自动重置)。
- `*_summary.json`: 每回合 汇总 (return / 五项分量和 / 认证行 / 掉落行 / 放手后均漂移 / cross_frac / 各指对最小间距) + 均值。
- `rec_r35.log` / `rec_r50.log`: 录制日志 (含 `[world] ✅` 核对与 `[record]` 汇总行)。

### 读数 (2026-09-08 04:50 UTC, last.pth = 10M 步)
| 档位 | success | cert | drop | return | 放手后均漂移 盘 / 海绵 | cross_frac |
|---|---|---|---|---|---|---|
| r35 (当前档) | 4/4 | 4/4 | 0 | +21.6 | 0.6cm/3.1° / 0.24cm/3.2° | 0.52 |
| r50 (起点档) | 0/4 | 4/4 | 0 | +0.7 | 0.5cm/3.8° / 0.25cm/5.0° | 0.48 |
- r50 的 0/4 不是掉, 是回合末海绵转角 5~7.5° 漂出 5° 认证窗 (`within=0`), 拿不到 +20 成功奖; 策略已适应当前档的放手时刻。
- **交叉罚来源 = 左手(盘) 中指-无名指对** (gapL_mid_ring 最小 -0.3cm, 翻转; 约 50% 步交叉), 不是海绵手; 右手三对间距 ≥1.7cm 干净。
  每回合 r_cross ≈ -8 (W_CROSS=1), 相对 +25 认证/成功奖压不住。

重录: `SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY tasks/Clean/3/C_Wiring/record_clean.py --checkpoint logs/Clean3_hold_s42/stage1_nn/last.pth --out logs/Clean3_hold_s42/videos/last_r35 --release_row 35 --headless --enable_cameras`
(档位看 `grep "\[课程\]" train.log | tail -1`; 每条约 2 分钟, Denso 每卡最多并跑 2 条)
