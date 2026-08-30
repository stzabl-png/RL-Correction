"""L5-25 禁碰传感器冒烟: 验证 D6 绑定已修 + 禁碰传感器真的读得到力。

★为什么必须跑这个: collide_flags 的单元自检只证明"给它力它会红";
证明不了"传感器真的把力送进来"。D6 恰恰是栽在这一步 —— 逻辑没错,
过滤器从未绑上, 5 条线 7900 万步恒为 0 而无人发现。
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

_CW = os.path.dirname(os.path.abspath(__file__)).replace("B_SmokeTest", "C_Wiring")
sys.path.insert(0, _CW)
import pour_env as PE  # noqa: E402

cfg = PE.build_cfg(num_envs=8)
print(f"[probe] 传感器区段: {cfg.sensor_slices}", flush=True)
E = PE.PourEnv(cfg)
print(f"[probe] 实际传感器数={len(E._all_sensors)} 区段={E._slices}", flush=True)

# 逐段报告形状, 空段/零形状即为"没绑上"
for k in ("pad_r", "pad_l", "d6", "noc_arm", "obj_obj"):
    ss = E._seg(k)
    if not ss:
        print(f"[probe] {k:8s} 段为空", flush=True)
        continue
    s0 = ss[0]
    nf = getattr(s0.data, "net_forces_w", None)
    fm = getattr(s0.data, "force_matrix_w", None)
    print(f"[probe] {k:8s} n={len(ss)} net={tuple(nf.shape) if nf is not None else None}"
          f" mat={tuple(fm.shape) if fm is not None else None}", flush=True)

def sweep(tag, action_fn, steps):
    """跑一段, 返回 (触发计数, 各通道观测到的最大力)。"""
    acc = {k: 0 for k in ("arm", "pad", "objobj", "d6")}
    mx = {k: 0.0 for k in ("arm", "pad", "objobj", "d6", "pad_net")}
    perbody = [[0, 0.0] for _ in range(len(E._seg("noc_arm")))]
    E.reset()
    trace = []
    for t in range(steps):
        E.step(action_fn(t))
        c = E._collide()
        if t % 50 == 0 or t == steps - 1:
            pn = 0.0
            ps = E._seg("pad_r") + E._seg("pad_l")
            if ps:
                pn = float(torch.cat([s.data.net_forces_w.reshape(E.num_envs, -1, 3)
                                      for s in ps], dim=1).nan_to_num(0.0)
                           .norm(dim=-1).max())
            trace.append((t, int(E.row.float().mean()), pn,
                          int(E.PB.g1.sum()), int(E.PB.g2.sum())))
        for k in acc:
            acc[k] += int(c[k].sum())
        for _i, _sn in enumerate(E._seg("noc_arm")):
            _nn = _sn.data.net_forces_w.reshape(E.num_envs, -1, 3) \
                .nan_to_num(0.0).norm(dim=-1)
            perbody[_i][0] += int((_nn > 1.0).any(dim=1).sum())
            perbody[_i][1] = max(perbody[_i][1], float(_nn.max()))
        for k, ss, at in (("arm", E._seg("noc_arm"), "net_forces_w"),
                          ("d6", E._seg("d6"), "force_matrix_w"),
                          ("objobj", E._seg("obj_obj"), "force_matrix_w"),
                          ("pad_net", E._seg("pad_r") + E._seg("pad_l"),
                           "net_forces_w"),
                          ("pad", E._seg("pad_r") + E._seg("pad_l"),
                           "force_matrix_w")):
            if ss:
                v = torch.cat([getattr(s.data, at).reshape(E.num_envs, -1, 3)
                               for s in ss], dim=1).nan_to_num(0.0).norm(dim=-1).max()
                mx[k] = max(mx[k], float(v))
    print(f"\n[probe] === {tag} ({steps} 步 × {E.num_envs} env) ===", flush=True)
    for k in ("arm", "pad", "objobj", "d6"):
        print(f"[probe]   {k:8s} 触发={acc[k]:5d}  最大力={mx[k]:.3f} N", flush=True)
    print(f"[probe]   [正对照] 手垫净接触力最大 = {mx['pad_net']:.3f} N "
          f"(抓着物体时必须 >0, 否则整条接触链路是死的)", flush=True)
    # 逐垫报 净力 vs 对自物体的力 的差 —— pad 判据就是拿这个差和阈值比
    PN = ["R_idx", "R_mid", "R_rng", "R_pky", "R_thb",
          "L_idx", "L_mid", "L_rng", "L_pky", "L_thb", "R_palm", "L_palm"]
    _ps = E._seg("pad_r") + E._seg("pad_l") + E._seg("palm")
    if _ps:
        print("[probe]   逐垫 净力/对自物体力/差 (差>1N 即判禁碰):", flush=True)
        for _i, _sn in enumerate(_ps):
            _nn = float(_sn.data.net_forces_w.reshape(E.num_envs, -1, 3)
                        .nan_to_num(0.0).norm(dim=-1).max())
            _ff = float(_sn.data.force_matrix_w.reshape(E.num_envs, -1, 3)
                        .nan_to_num(0.0).norm(dim=-1).max())
            _nm2 = PN[_i] if _i < len(PN) else f"pad{_i}"
            print(f"[probe]     {_nm2:8s} 净={_nn:8.3f}N 对自物={_ff:8.3f}N "
                  f"差={_nn-_ff:8.3f}N {'★判禁碰' if _nn-_ff>1.0 else ''}", flush=True)
    NAMES = ["R_arm_l5", "R_arm_l7", "R_arm_l8", "right_hand_C_MC",
             "L_arm_l5", "L_arm_l7", "L_arm_l8", "left_hand_C_MC"]
    print("[probe]   逐体分解 (到底是哪个体在碰):", flush=True)
    for _i, _nm in enumerate(NAMES[:len(perbody)]):
        print(f"[probe]     {_nm:18s} 触发={perbody[_i][0]:5d} "
              f"最大力={perbody[_i][1]:7.3f} N", flush=True)
    print("[probe]   轨迹 (步/参考行/垫净力/G1数/G2数):", flush=True)
    for t, r, pn, g1, g2 in trace:
        print(f"[probe]     t={t:4d} row={r:4d} padN={pn:7.3f} g1={g1} g2={g2}",
              flush=True)
    return acc, mx


Z = torch.zeros(E.num_envs, PE.ACT_DIM, device=E.device)
a0, m0 = sweep("A 零动作(参考回放)", lambda t: Z, args.steps)

# ★必然该红的输入: 把两条手臂的残差打满并相向, 强行制造碰撞。
# 若这样都读不到力, 说明通道是死的 —— 这是本探针的核心判据。
BIG = torch.zeros(E.num_envs, PE.ACT_DIM, device=E.device)
BIG[:, 0:7] = 1.0        # 右臂 7 关节残差打满
BIG[:, 7:14] = -1.0      # 左臂 7 关节反向打满
a1, m1 = sweep("B 双臂残差打满相向(强行制造碰撞)", lambda t: BIG, args.steps)

# ★C 段: 残差被限幅在 0.05~0.10 rad, 动作打满也只有 ±5.7°, 手臂根本撞不到一起
# —— B 段"强行撞"其实没撞。绕过残差, 直接改写关节状态, 把手臂扫进桌面/彼此。
print("\n[probe] === C 直接改写关节(绕过残差限幅), 逐档扫描 ===", flush=True)
E.reset()
for _ in range(3):
    E.step(Z)
q0 = E.hand.data.joint_pos.clone()
qd0 = torch.zeros_like(q0)
armR = E.hand.find_joints(["R_arm_j[1-7]"])[0]
armL = E.hand.find_joints(["L_arm_j[1-7]"])[0]
best = {"arm": 0.0, "d6": 0.0, "hit": 0}
for j in range(7):
    for amp in (-1.2, -0.6, 0.6, 1.2):
        q = q0.clone()
        q[:, armR[j]] += amp
        q[:, armL[j]] -= amp
        E.hand.write_joint_state_to_sim(q, qd0)
        for _ in range(6):
            E.step(Z)
        c = E._collide()
        na = torch.cat([sn.data.net_forces_w.reshape(E.num_envs, -1, 3)
                        for sn in E._seg("noc_arm")], dim=1).nan_to_num(0.0) \
            .norm(dim=-1).max()
        nd = (torch.cat([sn.data.force_matrix_w.reshape(E.num_envs, -1, 3)
                         for sn in E._seg("d6")], dim=1).nan_to_num(0.0)
              .norm(dim=-1).max() if E._seg("d6") else torch.tensor(0.0))
        h = int((c["arm"] | c["d6"]).sum())
        if float(na) > best["arm"] or float(nd) > best["d6"] or h > best["hit"]:
            print(f"[probe]   关节R_arm_j{j+1} ±{abs(amp):.1f}rad -> "
                  f"臂净力={float(na):7.3f}N  D6={float(nd):7.3f}N  触发={h}", flush=True)
        best["arm"] = max(best["arm"], float(na))
        best["d6"] = max(best["d6"], float(nd))
        best["hit"] = max(best["hit"], h)
print(f"[probe]   C 段最大: 臂净力={best['arm']:.3f}N  D6={best['d6']:.3f}N  "
      f"最多触发={best['hit']}", flush=True)
m1["arm"] = max(m1["arm"], best["arm"])
m1["d6"] = max(m1["d6"], best["d6"])

print("\n[probe] ★判读表", flush=True)
live_pad = m0["pad_net"] > 0 or m1["pad_net"] > 0
live_arm = m1["arm"] > 0 or m0["arm"] > 0
print(f"[probe]   接触链路是否活的(垫净力>0)      : {'✅ 是' if live_pad else '★否 —— 整条链路死的'}",
      flush=True)
print(f"[probe]   臂传感器能否读到力(B段)          : {'✅ 能' if live_arm else '★不能'}",
      flush=True)
print(f"[probe]   A段全零 + B段非零 ⟹ A的0是真的没碰, 不是没绑上", flush=True)
print(f"[probe]   A段全零 + B段也全零 ⟹ ★通道有问题, 判据无意义", flush=True)
sys.stdout.flush()
app.close()
os._exit(0)
