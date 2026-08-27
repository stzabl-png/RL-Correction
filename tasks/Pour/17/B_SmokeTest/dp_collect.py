"""DP staging 采集器: 批量回放变体, 记录 (state116_bimanual_v1, act58_bimanual_v1)@20Hz。

- 每 env 一个变体: 物体按抖动刚体变换摆放, 播放该变体的 IK 重解行(15fps 行线性
  重采样到 20Hz 时间轴), 动作=当步下发的 58 维绝对关节目标(★不是残差)。
- 成功判据(只导成功回合): 全程播完 & 物体位移<3cm & 倾角<15° & 不掉桌。
- 退化通道如实入 meta: Approach 期触觉≈0(真实读数)、物体位姿≈常量。

输出: dp_staging/episode_XXXX.npz + manifest.json
  state  (T,116) f32 = 臂R7+臂L7+指R22+指L22 + 腕R7+腕L7 + 杯7+瓶7 + 触R15+触L15
  action (T, 58) f32 = 绝对关节角 (act58_bimanual_v1)
  四元数 wxyz, w>=0; 位置 env 系(米); 触觉 /squeeze_f0, clamp±3。
"""
from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--variants", default="tasks/Pour/17/B_SmokeTest/dp_variants.npz")
p.add_argument("--out_dir", default="tasks/Pour/17/B_SmokeTest/dp_staging")
p.add_argument("--hz", type=float, default=20.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("dp_collect")
app = AppLauncher(args).app

import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
import pour_env as PE  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul  # noqa: E402

V = np.load(args.variants, allow_pickle=True)
rows_all = V["rows58"]                       # (M,T,58)
dxy_all, dyaw_all = V["obj_dxy"], V["obj_dyaw"]
M, T0, _ = rows_all.shape
# 15fps 行轴 -> 20Hz 时间轴 线性重采样
T1 = int(round(T0 * args.hz / 15.0))
tsrc = np.linspace(0.0, 1.0, T0)
tdst = np.linspace(0.0, 1.0, T1)
rows20 = np.stack([np.stack([np.interp(tdst, tsrc, rows_all[m, :, j])
                             for j in range(58)], axis=1)
                   for m in range(M)]).astype(np.float32)     # (M,T1,58)

cfg = PE.build_cfg(num_envs=M)
E = PE.PourEnv(cfg)
E.reset()
dev = E.device
org = E.scene.env_origins


def qz(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], np.float32)


def qmulnp(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


# ---- 摆物体 (刚体变换后的静置位) + 机器人到首行 ----
placed = {}
for oi, art in ((0, E.aux), (1, E.object)):
    rest = E.rest_pose[oi].cpu().numpy()
    pose = np.zeros((M, 7), np.float32)
    for m in range(M):
        pose[m, :3] = rest[:3]
        pose[m, :2] += dxy_all[m, oi]
        pose[m, 3:7] = qmulnp(qz(float(dyaw_all[m, oi])), rest[3:7])
    placed[oi] = torch.tensor(pose, device=dev)
    w_pose = placed[oi].clone()
    w_pose[:, :3] += org
    art.write_root_pose_to_sim(w_pose)
    art.write_root_velocity_to_sim(torch.zeros(M, 6, device=dev))
q0 = E.hand.data.default_joint_pos.clone()
q0[:, E.map_ids_t] = torch.tensor(rows20[:, 0], device=dev)
E.hand.write_joint_state_to_sim(q0, torch.zeros_like(q0))
E.hand.set_joint_position_target(q0)
DECI = int(getattr(E.cfg, "decimation", 12))  # 20Hz 控制 = 每步 12 物理子步
for _ in range(20 * DECI):                    # 静置落定 (1s)
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())

SQ = float(E.cfg.squeeze_f0)
S_buf = np.zeros((M, T1, 116), np.float32)
A_buf = rows20.copy()                        # 动作=下发的绝对目标 (act58_bimanual_v1)
alive = np.ones(M, bool)

for t in range(T1):
    tgt = torch.tensor(rows20[:, t], device=dev)
    full = E.hand.data.joint_pos.clone()
    full[:, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    for _ in range(DECI):                     # 一个 20Hz 控制步 = DECI 个物理子步
        E.scene.write_data_to_sim()
        E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())
    # ---- state116 ----
    q58 = E.hand.data.joint_pos[:, E.map_ids_t]
    wr = []
    for s in ("R", "L"):
        wp = E.hand.data.body_pos_w[:, E.wid[s]] - org
        wq = E._qsign(E.hand.data.body_quat_w[:, E.wid[s]])
        wr.append(torch.cat([wp, wq], dim=1))
    obj = []
    for oi, art in ((0, E.aux), (1, E.object)):
        op = art.data.root_pos_w - org
        oq = E._qsign(art.data.root_quat_w)
        obj.append(torch.cat([op, oq], dim=1))
    F = E._pads_f()                          # (M,10,3) 世界系
    qinR = quat_conjugate(E._qsign(E.hand.data.body_quat_w[:, E.wid["R"]]))
    qinL = quat_conjugate(E._qsign(E.hand.data.body_quat_w[:, E.wid["L"]]))
    tacR = quat_apply(qinR.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4),
                      F[:, :5].reshape(-1, 3)).reshape(M, 15) / SQ
    tacL = quat_apply(qinL.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4),
                      F[:, 5:].reshape(-1, 3)).reshape(M, 15) / SQ
    st = torch.cat([q58, wr[0], wr[1], obj[0], obj[1],
                    tacR.clamp(-3, 3), tacL.clamp(-3, 3)], dim=1)
    S_buf[:, t] = st.float().cpu().numpy()
    # 中途掉桌即废
    for oi, art in ((0, E.aux), (1, E.object)):
        alive &= (art.data.root_pos_w[:, 2] > 0.87 - 0.05).cpu().numpy()

# ---- 成功判据: 物体没被碰跑/碰歪 ----
ok = alive.copy()
for oi, art in ((0, E.aux), (1, E.object)):
    cur_p = art.data.root_pos_w - org
    dp = (cur_p[:, :3] - placed[oi][:, :3]).norm(dim=1).cpu().numpy()
    up = E.PB.up if oi == 1 else torch.tensor([0.0, 1.0, 0.0], device=dev)
    qn = art.data.root_quat_w / art.data.root_quat_w.norm(dim=1, keepdim=True)
    v = quat_apply(qn, up.unsqueeze(0).expand(M, 3))
    tilt = torch.acos((v[:, 2] / v.norm(dim=1).clamp(min=1e-9))
                      .clamp(-1, 1)).cpu().numpy()
    ok &= (dp < 0.03) & (tilt < np.radians(15))

os.makedirs(args.out_dir, exist_ok=True)
manifest = {"schema_state": "state116_bimanual_v1", "schema_action": "act58_bimanual_v1",
            "hz": args.hz, "quat": "wxyz,w>=0", "units": "m/rad",
            "tactile": "wrist-frame, /squeeze_f0, clamp±3, 真实读数",
            "task": "pour17_approach_mp", "success_rule": "播完&物移<3cm&倾角<15°&不掉桌",
            "degenerate_channels": {"tactile": "Approach 接触前≈0(真实读数)",
                                    "obj_pose": "接近期≈常量(仅随抖动初值变)"},
            "scope_note": "仅测'DP能否拟合状态相关的双手动作流', 不外推任务能力",
            "episodes": []}
n_saved = 0
for m in range(M):
    if not ok[m]:
        continue
    st, ac = S_buf[m], A_buf[m]
    assert st.shape == (T1, 116) and ac.shape == (T1, 58)
    assert st.dtype == np.float32 and ac.dtype == np.float32
    for sl in (slice(61, 65), slice(68, 72), slice(75, 79), slice(82, 86)):
        n = np.linalg.norm(st[:, sl], axis=1)
        assert np.all(np.abs(n - 1) < 1e-3), f"四元数未归一 ep{m} {sl}"
    ep = {"success": True, "variant": m,
          "obj_dxy": dxy_all[m].tolist(), "obj_dyaw": dyaw_all[m].tolist(),
          "len": int(T1)}
    np.savez_compressed(os.path.join(args.out_dir, f"episode_{n_saved:04d}.npz"),
                        state=st, action=ac, meta=json.dumps(ep))
    manifest["episodes"].append(ep)
    n_saved += 1
with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)
print(f"[collect] ✅ {n_saved}/{M} 成功入库 -> {args.out_dir} "
      f"(废弃 {M - n_saved}: 碰跑/碰歪/掉桌)")
try:
    _slot.release()
except Exception:
    pass
app.close()
