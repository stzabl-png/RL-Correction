"""参考延长: 站姿→5cm PreGrasp (现有全障碍规划) + 5→1cm 直线段 (IK 插值) 拼接。

裁定 (2026-08-18 晚): PreGrasp 挪到 1cm —— 腕由前馈送到门口, 手指由 RL 学。
5→1cm 段物体接触在所难免(近场豁免已开), 且 leg2 探针证明该段是纯直线(1.00x),
无须 cuRobo, IK 两端 + 关节线性插值即可。

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.extend_ref_1cm --headless
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--grasp_prior", default="tasks/pregrasp/priors/Pour17_bottle.npz")
p.add_argument("--prior_yaw", type=float, default=19.5)
p.add_argument("--prior_b", default="tasks/pregrasp/priors/Pour17_cup.npz")
p.add_argument("--prior_b_yaw", type=float, default=90.0)
p.add_argument("--src", default="tasks/pregrasp/priors/curobo_pour17_pregrasp0_ref.npz")
p.add_argument("--out", default="tasks/pregrasp/priors/curobo_pour17_pregrasp1_ref.npz")
p.add_argument("--leg_frames", type=int, default=20)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("extend_ref")
app = AppLauncher(args).app

import os  # noqa: E402

import numpy as np  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
from tasks.pregrasp import bimanual as BM  # noqa: E402
from tasks.pregrasp.bimanual_env import BimanualApproachEnv  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
# ⚠ 必须在 apply_grasp_prior 之前 (与 suite/record 同因: 观测宽度按此算死);
#   也是 2026-08-18 双臂防雷断言的要求 (approach_only 自动开退避起点族)
cfg.approach_only = True
cfg.action_space = 7
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.direct_grasp_prob = 0.0
cfg.approach_t0_max = 0.0
cfg.prior_b_npz = os.path.abspath(args.prior_b)
cfg.prior_b_yaw_deg = float(args.prior_b_yaw)
_a1, _o1 = cfg.action_space, cfg.observation_space
cfg.action_space, cfg.observation_space = 2 * _a1, 2 * _o1
cfg._obs_single = _o1
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0

env = BimanualApproachEnv(cfg)
env.reset()
pcm = float(getattr(cfg, "pregrasp_palm_cm", 1.0))
print(f"[extend] 当前掌心锚点 = {pcm}cm (cfg.pregrasp_palm_cm)")
z = np.load(args.src)
out = {}
K = int(args.leg_frames)
for sn, side in ((env._A_name, env._A), (env._B_name, env._B)):
    with BM.use_side(env, side):
        pw = env._pregrasp_w[0]
        anchor_T = env._anchor_T
    pos = np.asarray(pw[0], np.float64)
    quat = np.asarray(pw[1], np.float64)
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    q_src = np.asarray(z[f"{sn}_q"], np.float64)
    seed = q_src[-1]
    ik = ArmIK(sn, anchor_link="arm_center", anchor_T=anchor_T)
    r = ik.solve(pos, quat_to_R(quat), q0=seed, iters=300)
    print(f"[extend] {sn}: IK@1cm err {r['pos_err']*100:.2f}cm | "
          f"末帧关节位移 {np.degrees(np.abs(r['q']-seed)).max():.1f}°")
    assert r["pos_err"] < 0.01, f"{sn} 1cm 终点 IK 不达标"
    al = np.linspace(0.0, 1.0, K + 1)[1:, None]
    leg = (1 - al) * seed[None] + al * r["q"][None]
    out[f"{sn}_q"] = np.concatenate([q_src, leg], 0).astype(np.float32)
np.savez(args.out, right_q=out["right_q"], left_q=out["left_q"],
         seg_frames=np.array(["both|pregrasp", "both|to1cm"], dtype=object),
         seg_lens=np.array([len(z["right_q"]), K]),
         joint_mode=1, source="extend_ref_1cm", palm_cm=pcm)
print(f"[extend] ✅ 已存 {args.out} (右 {out['right_q'].shape} 左 {out['left_q'].shape})")
try:
    _slot.release()
    print("[extend] GPU 槽位已主动释放")
except Exception:
    pass
os._exit(0)            # 硬退 (2026-08-20 三入口同修)
