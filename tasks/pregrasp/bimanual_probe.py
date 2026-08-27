"""双臂接近 —— **第 0 步探针**: 在真场景里把两只手的靶点各自算出来并对拍。

## 为什么先做这个, 而不是直接写训练

`bimanual.py` 的架构是"env 单边逻辑不动, 外面换入换出侧状态", 它自己的注释警告:

    换入换出必须覆盖**全部**单边量, 漏一个就是**静默串台**(A 手的动作写进 B 手的状态)。

静默 = 不报错。所以第一件事必须是**证明两侧的量真的是分开的、且各自正确**, 而不是
先写训练再debug。判据(全部可证伪):

  ① 两侧关节索引不重叠 (R_arm_j* vs L_arm_j*)
  ② 两侧 GraspPose 靶点**不同**, 且各自离**自己那个物体**近、离对方远
  ③ 两侧靶点 IK 各自有解 (err < 2cm) —— 够不到就没得训
  ④ assert_covered 扫不出漏网的单边量

## 用法

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.bimanual_probe --headless \\
        --clip Pour17_bottle --prior_a tasks/pregrasp/priors/Pour17_bottle.npz --yaw_a 19.5 \\
        --prior_b tasks/pregrasp/priors/Pour17_cup.npz --yaw_b 180
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--prior_a", required=True, help="主手(clip 的交互手)的 GraspPose")
p.add_argument("--yaw_a", type=float, default=-1.0)
p.add_argument("--prior_b", required=True, help="另一只手的 GraspPose")
p.add_argument("--yaw_b", type=float, default=-1.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("bimanual_probe")
app = AppLauncher(args).app

import numpy as np  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
from tasks.pregrasp import bimanual as BM  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

BAR = "=" * 74
cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.approach_only, cfg.action_space = True, 7
apply_grasp_prior(cfg, args.prior_a, args.yaw_a, approach=True)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()

sideA_name = cfg.hand_side
other = "left" if sideA_name == "right" else "right"
print(f"\n{BAR}\n主手 A = {sideA_name} (clip {args.clip} 的交互手) | 另一只手 B = {other}\n{BAR}")

# ---- ① 快照 A 侧, 然后按 B 侧重新解析 ----
sideA = BM.SideState(sideA_name, E)
A_arm, A_gp = list(E.arm_jids), E._grasp_pos_w.cpu().numpy().astype(float)
A_gq = E._grasp_quat_w.cpu().numpy().astype(float)
A_obj = E.obj_init_pos.cpu().numpy().astype(float)

# aux(第二个物体)的摆放 —— B 手的目标物体
aux_off = getattr(E, "aux_rel_offset_np", None)
assert aux_off is not None, "这条 clip 没有第二个物体 (aux), 双臂无从谈起"
B_obj = A_obj + np.asarray(aux_off, float)

E._resolve_joint_ids(force_side=other)          # 只换关节索引, 场景一字不动
B_arm = list(E.arm_jids)
print(f"① 关节索引")
print(f"   A({sideA_name:5s}) 臂关节 {A_arm}")
print(f"   B({other:5s}) 臂关节 {B_arm}")
_ov = set(A_arm) & set(B_arm)
print(f"   重叠 = {sorted(_ov) if _ov else '空'}  {'❌ 两侧共用关节!' if _ov else '✅ 完全分开'}")

# ---- ② B 侧的 GraspPose 靶点 ----
zb = np.load(args.prior_b)
canon_b = np.asarray(zb["canon_rot"], float)
a = np.radians(args.yaw_b)
yq = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


oq_b = qmul(yq, canon_b)
B_gp = quat_to_R(oq_b) @ np.asarray(zb["grasp"][:3], float) + B_obj
B_gq = qmul(oq_b, np.asarray(zb["grasp"][3:7], float))

print(f"\n② 两只手的靶点 (env 局部系, cm)")
print(f"   A 靶点 {np.round(A_gp*100, 1)}   A 物体 {np.round(A_obj*100, 1)}")
print(f"   B 靶点 {np.round(B_gp*100, 1)}   B 物体 {np.round(B_obj*100, 1)}")
d_tgt = np.linalg.norm(A_gp - B_gp) * 100
print(f"   两靶点相距 {d_tgt:.1f}cm  {'✅ 明显不同' if d_tgt > 5 else '❌ 几乎重合, 八成串台了'}")
for nm, gp, o_own, o_oth in (("A", A_gp, A_obj, B_obj), ("B", B_gp, B_obj, A_obj)):
    d_own, d_oth = np.linalg.norm(gp - o_own)*100, np.linalg.norm(gp - o_oth)*100
    ok = d_own < d_oth
    print(f"   {nm} 靶点离**自己**物体 {d_own:5.1f}cm, 离对方 {d_oth:5.1f}cm  "
          f"{'✅' if ok else '❌ 靶点离对方物体更近, 配对错了'}")

# ---- ③ 两侧 IK 各自有解? ----
print(f"\n③ 两侧 IK (锚点 = env 实测 arm_center, 与训练同口径)")
for nm, side, gp, gq in ((f"A({sideA_name})", sideA_name, A_gp, A_gq),
                         (f"B({other})", other, B_gp, B_gq)):
    ik = ArmIK(side, anchor_link="arm_center", anchor_T=E._anchor_T)
    r = ik.solve(gp, quat_to_R(gq), iters=200)
    print(f"   {nm:10s} ok={r['ok']!s:5s} 位置误差 {r['pos_err']*100:6.2f}cm  "
          f"{'✅' if r['pos_err'] < 0.02 else '❌ >2cm, 这只手够不着自己的目标'}")

# ---- ④ 单边量覆盖 ----
print(f"\n④ 单边量覆盖扫描 (漏一个 = 两手静默共用它)")
sus = BM.assert_covered(E)
print(f"   可疑字段 {len(sus)} 个" + (f": {sus}" if sus else " —— ✅ 无"))

E._resolve_joint_ids(force_side=sideA_name)     # 还原, 免得影响后续
print(f"\n{BAR}")
app.close()
