"""DP 数据源·变体生成器 (2026-08-28, owner拍板: act58联合/触觉真实读数)。

原理: 对每个抖动初值(物体 xy±2cm / yaw±10°), 把 Approach 的腕轨迹按该物体的刚体
变换搬移(转移段线性混入, 成形段全量), 逐行 ArmIK 重解 —— **动作真响应状态**,
DP 测试因此"能失败"(固定轨迹回放N次是不可能失败的测试, RL_Training 同判)。
手指行不动: 抓形在物体系下不变, 物体刚体变换恰由腕搬移吸收。
IK 超差/超限位的变体整条丢弃(报数)。

输出: dp_variants.npz
  rows58   (M, T, 58) f32   [R臂7,L臂7,R指22,L指22] 绝对关节角 (act58_bimanual_v1 同序)
  obj_dxy  (M, 2, 2)  f32   [杯,瓶] 的 xy 抖动 (m)
  obj_dyaw (M, 2)     f32   [杯,瓶] 的 yaw 抖动 (rad)
  meta     json 字符串
"""
from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=24, help="变体数")
p.add_argument("--jit_xy", type=float, default=0.02)
p.add_argument("--jit_yaw", type=float, default=10.0, help="度")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--out", default="tasks/Pour/17/B_SmokeTest/dp_variants.npz")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("dp_variants")
app = AppLauncher(args).app

import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
import pour_env as PE  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, R_to_quat, quat_to_R  # noqa: E402

cfg = PE.build_cfg(num_envs=1)
E = PE.PourEnv(cfg)
E.reset()

APP = np.load("tasks/Pour/17/A_Design/L1_Data/Motion_Planning/Approach_pour17.npz",
              allow_pickle=True)
T = len(APP["right_q"])
segl = [int(v) for v in APP["seg_lens"]]
rows0 = np.concatenate([APP["right_q"], APP["left_q"],
                        APP["right_f"], APP["left_f"]], axis=1).astype(np.float64)
# 混入权重: 转移段 0->1 (起点必须钉在站姿), 成形段恒 1
w = np.ones(T)
w[:segl[0]] = np.linspace(0.0, 1.0, segl[0])

# 物体静置位 (env 系 = 变体的变换中心); 0=杯(左手) 1=瓶(右手)
c_obj = {oi: E.rest_pose[oi][:3].cpu().numpy() for oi in (0, 1)}
ik = {"right": ArmIK("right", anchor_link="arm_center", anchor_T=E._anchor_T),
      "left": ArmIK("left", anchor_link="arm_center", anchor_T=E._anchor_T)}
side_obj = {"right": 1, "left": 0}          # 右手跟瓶, 左手跟杯

rng = np.random.default_rng(args.seed)
ok_rows, dxys, dyaws, n_fail = [], [], [], 0
while len(ok_rows) < args.n:
    dxy = rng.uniform(-args.jit_xy, args.jit_xy, size=(2, 2))
    dyaw = rng.uniform(-np.radians(args.jit_yaw), np.radians(args.jit_yaw), size=2)
    rows = rows0.copy()
    bad = False
    for s, sl in (("right", slice(0, 7)), ("left", slice(7, 14))):
        oi = side_obj[s]
        cz, sz = np.cos(dyaw[oi]), np.sin(dyaw[oi])
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        q_prev = rows0[0, sl].copy()
        for t in range(T):
            q0 = rows0[t, sl]
            pw, Rw = ik[s].fk(q0)
            # 目标 = 绕物体中心的刚体变换, 按 w[t] 混入
            p_full = Rz @ (pw - c_obj[oi]) + c_obj[oi]
            p_full[:2] += dxy[oi]
            p_t = pw + w[t] * (p_full - pw)
            # 姿态: 分数次幂近似 —— 小角度下用 w*dyaw 的 Rz
            th = w[t] * dyaw[oi]
            c2, s2 = np.cos(th), np.sin(th)
            Rz_w = np.array([[c2, -s2, 0.0], [s2, c2, 0.0], [0.0, 0.0, 1.0]])
            R_t = Rz_w @ Rw
            r = ik[s].solve(p_t, R_t, q0=q_prev, iters=60)
            if r["pos_err"] > 0.008:
                bad = True
                break
            rows[t, sl] = r["q"]
            q_prev = np.asarray(r["q"], np.float64)
        if bad:
            break
    if bad:
        n_fail += 1
        if n_fail > args.n * 4:
            raise SystemExit("IK 失败率过高, 检查抖动幅度")
        continue
    ok_rows.append(rows.astype(np.float32))
    dxys.append(dxy.astype(np.float32))
    dyaws.append(dyaw.astype(np.float32))
    print(f"[variants] {len(ok_rows)}/{args.n} (IK废弃 {n_fail})", flush=True)

meta = {"schema": "act58_bimanual_v1", "source": "Approach_pour17.npz + ik_warp_v0",
        "jit_xy_m": args.jit_xy, "jit_yaw_deg": args.jit_yaw, "seed": args.seed,
        "blend": "transit 0->1, forming 1", "ik_pos_tol_m": 0.008,
        "note": "手指行不变(抓形物体系不变); 碰撞未重检, 由采集端成功判据过滤"}
np.savez_compressed(args.out, rows58=np.stack(ok_rows), obj_dxy=np.stack(dxys),
                    obj_dyaw=np.stack(dyaws), meta=json.dumps(meta))
print(f"[variants] ✅ {args.out}: {len(ok_rows)} 条 × {T} 行 (IK废弃 {n_fail})")
try:
    _slot.release()
except Exception:
    pass
app.close()
