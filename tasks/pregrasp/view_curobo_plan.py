"""Pour17 双手 cuRobo 规划可视化 —— GUI 一次性脚本。

**先规划右手到瓶的 GraspPose, 再规划左手到杯的 GraspPose**(左手会避开已就位的右臂),
在 Isaac GUI 回放, **绿线=右手腕轨迹 / 红线=左手腕轨迹**。

## 与 RL 训练环境对齐的三件事 (用户 2026-08-17 定, 全部用**实测值**不许猜)

1. **桌面 + 两个物体都是障碍**, 物体用 **Mesh**(不是包围盒 —— 瓶/杯是回转体,
   包围盒会把外接柱体整个封死, 手贴不到表面)。位姿取 env 里物体的**实际**位姿。
2. **起点 = env 的站姿**(`hand.data.default_joint_pos`), 不是 cuRobo 默认值。
3. **靶点 link = `{side}_hand_C_MC`**, 与 RL 的 `ee_body` 同一个 —— 不是 R_ee/L_ee。
   (2026-08-17 实测: 用错 link 会穿桌且到不了 GraspPose)

## 用法

    cd /home/lyh/Project/RL_Correction
    SHARPA_WANDB=0 PYTHONPATH=. /home/lyh/luhr/MagicSim/.venv/bin/python \
        -m tasks.pregrasp.view_curobo_plan --loop

    --headless   只规划不开 GUI (快速验可行性)
    --act_dist   碰撞激活距离(m), 默认 0.015; 调小更容易贴近物体
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--prior_a", default="tasks/pregrasp/priors/Pour17_bottle.npz")
p.add_argument("--yaw_a", type=float, default=19.5)
p.add_argument("--prior_b", default="tasks/pregrasp/priors/Pour17_cup.npz")
p.add_argument("--yaw_b", type=float, default=180.0)
p.add_argument("--act_dist", type=float, default=0.015)
p.add_argument("--table_pad", type=float, default=0.0,
               help="规划用桌面垫高(m), 验收仍按真实桌面; 治'贴桌太近指尖擦桌'")
p.add_argument("--fps", type=float, default=20.0)
p.add_argument("--loop", action="store_true")
p.add_argument("--save_plan", default="",
               help="两道验收都过后, 把逐侧臂关节参考存到此 npz (给 RL 当前馈)")
p.add_argument("--joint", type=int, default=0,
               help="1=双手联合规划(同时移动); 左手终点退到近点, 蹭杯段不进参考")
p.add_argument("--left_short_cm", type=float, default=5.0,
               help="joint 模式左手终点后退距离(cm), 应与 RL 的 ff_freeze_cm 一致")
p.add_argument("--leg2", type=int, default=0,
               help="1=只规划接触短腿: 起点=双臂 IK(掌心PreGrasp), 目标=真 GraspPose, "
                    "物体排除/桌保留 —— 验证 cuRobo 能否直接驱动 PreGrasp→GraspPose")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("view_curobo_plan")
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import quat_to_R  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import (GENERIC_JOINT_ORDER as _GJO,  # noqa: E402
                                GENERIC_OPEN as _GOPEN)
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

BAR = "=" * 74


def qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


# ---------------------------------------------------------------- 1) 场景 + 靶点
cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.approach_only, cfg.action_space = True, 7
apply_grasp_prior(cfg, args.prior_a, args.yaw_a, approach=True)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()
W = E.scene.env_origins[0].cpu().numpy().astype(np.float64)

A_name = cfg.hand_side
B_name = "left" if A_name == "right" else "right"
A_gp = E._grasp_pos_w.cpu().numpy().astype(np.float64)
A_gq = E._grasp_quat_w.cpu().numpy().astype(np.float64)
A_obj = E.obj_init_pos.cpu().numpy().astype(np.float64)
A_oq = E.obj_init_quat.cpu().numpy().astype(np.float64)

aux_off = getattr(E, "aux_rel_offset_np", None)
assert aux_off is not None, "这条 clip 没有第二个物体"
B_obj = A_obj + np.asarray(aux_off, np.float64)
zb = np.load(args.prior_b)
_a = np.radians(args.yaw_b)
oq_b = qmul(np.array([np.cos(_a/2), 0.0, 0.0, np.sin(_a/2)]),
            np.asarray(zb["canon_rot"], np.float64))
B_gp = quat_to_R(oq_b) @ np.asarray(zb["grasp"][:3], np.float64) + B_obj
B_gq = qmul(oq_b, np.asarray(zb["grasp"][3:7], np.float64))

# 裁定B3 (2026-08-18): 规划唯一目标 = 掌心锚点 PreGrasp (抓姿沿"接触质心→腕"平移 3cm)。
# A 侧读 env 算好的(含校准), B 侧与 B_gp 同一条 raw 链(与 bimanual_env 口径一致)。
A_pre0_p = np.asarray(E._pregrasp_w[0][0], np.float64)
A_pre0_q = np.asarray(E._pregrasp_w[0][1], np.float64)
_zbg = np.asarray(zb["grasp"], np.float64)
_zbc = np.asarray(zb["contact_centroid"], np.float64)
_ub = _zbg[:3] - _zbc
_ub = _ub / max(np.linalg.norm(_ub), 1e-9)
_pcm = float(getattr(cfg, "pregrasp_palm_cm", 5.0)) / 100.0
B_pre0_p = quat_to_R(oq_b) @ (_zbg[:3] + _pcm * _ub) + B_obj
B_pre0_q = qmul(oq_b, _zbg[3:7])

_leg2_arms = {}
if int(args.leg2):
    # 接触短腿探针 (2026-08-18晚): 起点 = 双臂在掌心 PreGrasp 的 IK 位形
    from rl_rebuild.correction.kinematics import ArmIK as _AIK
    for _sn, _pp, _qq in ((A_name, A_pre0_p, A_pre0_q),
                          (B_name, B_pre0_p, B_pre0_q)):
        _qn = np.asarray(_qq, np.float64)
        _qn = _qn / max(np.linalg.norm(_qn), 1e-12)
        _ik = _AIK(_sn, anchor_link="arm_center", anchor_T=E._anchor_T)
        _q0s = E.hand.data.default_joint_pos[0][
            [list(E.hand.joint_names).index(
                f"{'R' if _sn == 'right' else 'L'}_arm_j{i}")
             for i in range(1, 8)]].cpu().numpy().astype(np.float64)
        _r = _ik.solve(np.asarray(_pp, np.float64), quat_to_R(_qn),
                       q0=_q0s, iters=300)
        print(f"[leg2] {_sn} PreGrasp IK: err {_r['pos_err']*100:.2f}cm")
        _leg2_arms[_sn] = _r["q"]

print(f"\n{BAR}\ncuRobo 双手规划 (joint=1 双手同时联合规划 | joint=0 先右后左) | clip={args.clip}\n{BAR}")
print(f"  右手(瓶) 靶点 {np.round(A_gp*100,1)} cm  物体 {np.round(A_obj*100,1)}")
print(f"  左手(杯) 靶点 {np.round(B_gp*100,1)} cm  物体 {np.round(B_obj*100,1)}")

# ---------------------------------------------------------------- 2) 交给子进程规划
# ⚠ cuRobo 与 Isaac 的 warp 版本冲突, 必须**独立进程**(见 worker 的 docstring)。
_e = clips.clip_entry(args.clip)
_sec = _e.get("secondary") or {}
_sx, _sy, _sz = cfg.table_size
_objs = [{"name": "obj_primary", "mesh": _e["mesh"],
          "pos": (A_obj + W).tolist(), "quat": A_oq.tolist()}]
if _sec.get("mesh"):
    _bq = E.aux.data.root_quat_w[0].cpu().numpy().astype(float).tolist() \
        if getattr(E, "aux", None) is not None else [1.0, 0.0, 0.0, 0.0]
    _objs.append({"name": "obj_secondary", "mesh": _sec["mesh"],
                  "pos": (B_obj + W).tolist(), "quat": _bq})

# ★★ 坐标系: cuRobo 的机器人模型以**自己的底座**为原点, 而 Isaac 里机器人站在
#    dexmate_pos=(-0.5,0,0)。直接喂世界坐标, cuRobo 在自己坐标系里"精确到达"
#    (它的 FK 验收 0.00cm), 但关节角放回 Isaac 后手整体偏 -X ≈ 0.5m
#    (2026-08-17 用户 GUI 实测: 双手都停在各自靶球的 -X 方向)。
#    ⟹ 所有位置量减去**实测**的机器人根位置转到底座系, 画线时再加回来。
_root_p = (E.hand.data.root_pos_w[0].cpu().numpy().astype(np.float64) - W)
print(f"[frame] 机器人根(实测, env 系) = {np.round(_root_p, 4)} —— "
      f"喂给 cuRobo 的所有位置都减它")

_tmp = tempfile.mkdtemp(prefix="curobo_plan_")
_tgt, _out = os.path.join(_tmp, "targets.json"), os.path.join(_tmp, "plan.npz")
with open(_tgt, "w") as f:
    json.dump({
        "table_pose": [float(W[0] - _root_p[0]), float(W[1] - _root_p[1]),
                       float(W[2] + cfg.table_top_z - _sz / 2 - _root_p[2])],
        "table_dims": [float(_sx), float(_sy), float(_sz)],
        "objects": [dict(o, pos=(np.asarray(o["pos"]) - _root_p).tolist())
                    for o in _objs],
        # ★ 躯干/头在 env 里是**锁死**的(TORSO_FIXED, 用的是躯干锁死 USD, 关节表里
        #   根本没有这几个关节), 而 cuRobo 以为它们可动 —— 规划器会靠转躯干去够目标,
        #   而真机转不了。所以把锁死角度显式传过去并要求 cuRobo 也锁住。
        #   角度取自 dexmate_env_cfg 的非锁死分支(躯干锁死 USD 烘的就是这组值)。
        "start_joints": dict(
            {n: float(v) for n, v in zip(
                E.hand.joint_names,
                E.hand.data.default_joint_pos[0].cpu().numpy().astype(float))},
            # 2026-08-27 Pour 新站姿 (必须与 make_fixed_torso_usd LOCK 表一致)
            **{"torso_j1": 0.7072,
               "torso_j2": 1.2856,
               "torso_j3": 0.0068,
               "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0},
            # ★ 手指锁在 GENERIC_OPEN(裁定B2 回放/训练的伸直指型), 不是站姿指型 ——
            #   两者不一致时规划器避的是另一只手的形状 (2026-08-18 左手撞杯排查)
            **{n.replace("right_", f"{_sd}_"): float(v)
               for _sd in ("right", "left")
               for n, v in zip(_GJO, _GOPEN)}),
        # ★ 手指也锁死在站姿(伸直张开): 用户设计 2026-08-18 —— PreGrasp 阶段手指
        #   不弯曲, 碰撞检查必须按伸直几何做, 否则躲的是假碰撞、漏的是真碰撞。
        "lock_joints": ["torso_j1", "torso_j2", "torso_j3",
                        "head_j1", "head_j2", "head_j3"]
                       + [n for n in E.hand.joint_names
                          if n.startswith(("right_", "left_"))],
        "tool_frames": [f"{A_name}_hand_C_MC", f"{B_name}_hand_C_MC"],
        # 每只手的目标物体 —— 规划该手时把它从障碍中排除(终点要贴上去)
        "hand_targets": {f"{A_name}_hand_C_MC": "obj_primary",
                         f"{B_name}_hand_C_MC": "obj_secondary"},
        "goals": {f"{A_name}_hand_C_MC": {"pos": (A_gp + W - _root_p).tolist(),
                                          "quat": A_gq.tolist()},
                  f"{B_name}_hand_C_MC": {"pos": (B_gp + W - _root_p).tolist(),
                                          "quat": B_gq.tolist()}},
        # 裁定B: joint 模式唯一规划目标 (leg2 探针: 目标改为真抓姿)
        "pregrasp_goals": ({
            f"{A_name}_hand_C_MC": {"pos": (A_gp + W - _root_p).tolist(),
                                    "quat": A_gq.tolist()},
            f"{B_name}_hand_C_MC": {"pos": (B_gp + W - _root_p).tolist(),
                                    "quat": B_gq.tolist()}} if int(args.leg2) else {
            f"{A_name}_hand_C_MC": {"pos": (A_pre0_p + W - _root_p).tolist(),
                                    "quat": A_pre0_q.tolist()},
            f"{B_name}_hand_C_MC": {"pos": (B_pre0_p + W - _root_p).tolist(),
                                    "quat": B_pre0_q.tolist()}}),
    }, f)
if int(args.leg2):
    # 起点臂关节改写为 PreGrasp 位形 (start_joints 在 json 里, 重写后重存)
    with open(_tgt) as _fj:
        _T2 = json.load(_fj)
    for _sn, _qarm in _leg2_arms.items():
        _pfx = "R" if _sn == "right" else "L"
        for _i in range(7):
            _T2["start_joints"][f"{_pfx}_arm_j{_i+1}"] = float(_qarm[_i])
    with open(_tgt, "w") as _fj:
        json.dump(_T2, _fj)
    print("[leg2] 起点=双臂 PreGrasp 位形 | 目标=真 GraspPose | 物体排除/桌保留")
print(f"  障碍: 桌 + 物体 ×{len(_objs)} (Mesh) | 起点 = env 站姿")
print(f"\n[plan] 起 cuRobo 子进程 (无 Isaac) ...")
subprocess.run([sys.executable, "-u", "-m", "tasks.pregrasp.curobo_plan_worker",
                "--targets", _tgt, "--out", _out, "--act_dist", str(args.act_dist),
                "--joint", str(int(args.joint)),
                "--left_short_cm", str(args.left_short_cm),
                "--table_pad", str(args.table_pad),
                "--exclude_objects", str(int(args.leg2))],
               cwd=os.getcwd(), env=dict(os.environ, PYTHONPATH=os.getcwd()),
               timeout=2400)
if not os.path.exists(_out):
    print("[plan] ❌ 子进程没有产出, 见上面的报错")
    app.close(); raise SystemExit(1)
_z = np.load(_out, allow_pickle=True)
if not bool(_z["ok"]):
    print(f"\n❌ 规划失败 (卡在 {_z.get('failed_frame')}) —— 在**桌+双物体都不可碰**"
          f"的约束下无解。可试 --act_dist 0.008 (更贴近表面)。")
    app.close(); raise SystemExit(0)

traj = _z["traj"]
_jn = [str(x) for x in _z["joint_names"]]
tf = [str(x) for x in _z["tool_frames"]]
wp = {f: _z[f"wp_{f}"].astype(np.float64) + _root_p for f in tf}   # 底座系 -> env/世界系
_segs = list(zip([str(x) for x in _z["seg_frames"]], _z["seg_lens"].tolist()))
print(f"\n✅ 规划成功 | 合计 {traj.shape[0]} 帧 | 分段 {_segs}")

# ---------------------------------------------------------------- 3) 关节映射 + 画线
sim_names = list(E.hand.joint_names)
col = {n: _jn.index(n) for n in _jn if n in sim_names}
print(f"[map] cuRobo 的 {len(_jn)} 个关节里, {len(col)} 个能对上仿真")
for f in tf:
    d = np.linalg.norm(np.diff(wp[f], axis=0), axis=1)
    st = np.linalg.norm(wp[f][-1] - wp[f][0])
    print(f"[traj] {f}: 行程 {d.sum()*100:6.1f}cm | 直线 {st*100:6.1f}cm | "
          f"绕路系数 {d.sum()/max(st,1e-9):.2f}x")

for f in tf:                       # worker 给的 wp 已经是**世界系**, 不要再加 W
    c = (0.15, 1.0, 0.15) if f.startswith("right") else (1.0, 0.15, 0.15)
    E._draw_curve(f"/World/Plan_{f}", wp[f], c, 0.004)
E._draw_sphere("/World/TgtR", (A_gp if A_name == "right" else B_gp) + W,
               (0.15, 1.0, 0.15), 0.015)
E._draw_sphere("/World/TgtL", (B_gp if B_name == "left" else A_gp) + W,
               (1.0, 0.15, 0.15), 0.015)
print("[draw] 绿=右手腕轨迹+瓶靶点 | 红=左手腕轨迹+杯靶点")

# ------------------------------------------------- 3.5) Isaac 实测验收 (地面真值)
# ⚠ worker 的验收是在 **cuRobo 自己的坐标系**里做的 FK —— 坐标系错了它照样全绿
#   (2026-08-17 就是这么漏掉整体 -X 偏移的, 用户肉眼才发现)。
#   这里把规划的**最终帧关节角真的摆进仿真**, 量真实的 hand_C_MC 离靶球多远。
_qf = E.hand.data.default_joint_pos.clone()
for _n, _ci in col.items():
    _qf[:, sim_names.index(_n)] = float(traj[-1, _ci])
E.hand.write_joint_state_to_sim(_qf, torch.zeros_like(_qf))
E.hand.set_joint_position_target(_qf)
for _ in range(5):
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())
_bn = list(E.hand.body_names)
_fail = False
for _side, _tgt_p in ((A_name, A_gp), (B_name, B_gp)):
    _bi = _bn.index(f"{_side}_hand_C_MC")
    _real = E.hand.data.body_pos_w[0, _bi].cpu().numpy().astype(np.float64) - W
    # joint 模式: worker 会给出各手的**预期终点**(左手=近点, 底座系) —— 优先用它,
    # 否则左手对着抓握位姿差 5cm 会误判失败
    _ek = f"end_expect_{_side}_hand_C_MC"
    if _ek in _z:
        _tgt_p = np.asarray(_z[_ek], float) + _root_p
    _err = np.linalg.norm(_real - _tgt_p) * 100
    _ok = _err < 2.5
    print(f"[isaac验收] {'✅' if _ok else '❌'} {_side} 手实测 {np.round(_real*100,1)}cm "
          f"vs 预期终点 {np.round(_tgt_p*100,1)}cm | 差 {_err:.2f}cm", flush=True)
    _fail |= not _ok
if _fail:
    print("[isaac验收] ❌❌ 仿真实测没有落在靶点上 —— 不要看回放, 先修坐标", flush=True)
    app.close(); raise SystemExit(1)
print("[isaac验收] ✅✅ 仿真实测双手都在靶点上", flush=True)

if args.save_plan:
    # 把逐侧臂关节参考存给 RL 当前馈 (只在**两道验收都过**之后才存)。
    # 存全部 324 帧: 右手 0-161 运动/162-323 保持, 左手相反 —— 保持段保留了
    # "先右后左"的时序, 训练时两侧各按自己的整条参考走, 复现规划的顺序语义。
    _sides = {"right": [f"R_arm_j{i}" for i in range(1, 8)],
              "left": [f"L_arm_j{i}" for i in range(1, 8)]}
    _sv = {}
    for _sd, _names in _sides.items():
        _ci = [_jn.index(n) for n in _names]
        _sv[f"{_sd}_q"] = traj[:, _ci].astype(np.float32)
        _sv[f"{_sd}_names"] = np.array(_names, dtype=object)
    np.savez(args.save_plan, seg_frames=_z["seg_frames"], seg_lens=_z["seg_lens"], **_sv)
    print(f"[save] 逐侧臂参考已存 -> {args.save_plan} "
          f"(右 {_sv['right_q'].shape} 左 {_sv['left_q'].shape})", flush=True)

if getattr(args, "headless", False):
    # ⚠ headless 下 app.is_running() 恒真, 进回放循环就永远不退 ——
    #   2026-08-17 实测留了一个抱着 5.1GB 显存的僵尸, 把用户的 GUI 挤到 OOM。
    print("[view] headless 验证完成, 退出 (回放只在 GUI 下进行)")
    # ★ 活干完立刻放槽位 —— Isaac 退出流程常挂死, 不放的话排队方(GUI/训练)会被
    #   僵尸进程堵住 (2026-08-18 一天撞了七次)
    try:
        _slot.release()
        print("[view] GPU 槽位已主动释放")
    except Exception:
        pass
    app.close(); raise SystemExit(0)

# ---------------------------------------------------------------- 4) 回放
def apply(k):
    q = E.hand.data.joint_pos.clone()
    for n, ci in col.items():
        q[:, sim_names.index(n)] = float(traj[k, ci])
    E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
    E.hand.set_joint_position_target(q)
    E.hand.write_data_to_sim()


print(f"\n{BAR}\n回放 {traj.shape[0]} 帧 @ {args.fps:.0f}fps"
      f"{' 循环' if args.loop else ''} —— 关窗口或 Ctrl-C 退出\n{BAR}")
dt_f, k = 1.0 / max(args.fps, 1e-3), 0
try:
    while app.is_running():
        apply(min(k, traj.shape[0] - 1))
        E.sim.step(render=True)
        k = (k + 1) if (k + 1 < traj.shape[0] or args.loop) else k
        if args.loop and k >= traj.shape[0]:
            k = 0
        time.sleep(dt_f)
except KeyboardInterrupt:
    pass
app.close()
