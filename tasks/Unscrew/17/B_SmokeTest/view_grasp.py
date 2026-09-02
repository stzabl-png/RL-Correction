"""Unscrew/17 GraspPose 目检 (§10 拍板①, 照 Pour L1-3 口径):
右臂+右手 = 初始站姿不动; 左臂+左手 = 直接摆到 LD227 站位 GraspPose (母带 IA0 行)。
瓶默认钉在静置位 (看抓法几何; --free 放开看物理), 每 2s 打印五垫力/瓶系径向/手最低点。
用法: bash tasks/Unscrew/17/B_SmokeTest/view_grasp.sh   (加 --squeeze 1.0 看加剂量后的合拢)
"""
import argparse, os, sys, time
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--squeeze", type=float, default=0.0, help="βL squeeze 剂量 (0=纯 GraspPose)")
p.add_argument("--free", action="store_true", help="瓶不钉 (看真实物理推挤)")
p.add_argument("--close", type=int, default=40, help="合拢动画步数 (0=旧口径: 直接瞬移成抓姿)")
p.add_argument("--path", default="", help="接近路径 npz (build_left_approach 产物): 循环=站姿->cuRobo->六级梯->squeeze->保持")
p.add_argument("--hold", type=int, default=60, help="合拢后保持步数 (循环: 保持完复位瓶重来)")
p.add_argument("--selftest", type=int, default=0, help=">0: 无头跑 N 个控制步打读数退出")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "part4", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
import task_config as TC
import task_env as PE

cfg = PE.build_cfg(num_envs=1)
E = PE.UnscrewEnv(cfg); E.force_entry = [0]; E.reset()
dev = E.device; DECI = int(getattr(E.cfg, "decimation", 12))
org = E.scene.env_origins[0]
# ★钉位姿必须在任何自由物理步之前抓 (2026-09-02 修: 静置期没走螺纹装配, 盖自由落体掉进瓶口)
_pinb = torch.cat([E.object.data.root_pos_w, E.object.data.root_quat_w], dim=1).clone()
_pinc = torch.cat([E.aux.data.root_pos_w, E.aux.data.root_quat_w], dim=1).clone()
_zero6 = torch.zeros(1, 6, device=dev)
for _ in range(30):
    if not args.free:
        E.object.write_root_pose_to_sim(_pinb); E.object.write_root_velocity_to_sim(_zero6)
        E.aux.write_root_pose_to_sim(_pinc); E.aux.write_root_velocity_to_sim(_zero6)
    E.scene.write_data_to_sim(); E.sim.step(render=False); E.scene.update(E.sim.get_physics_dt())

# 目标位形: 右 = 站姿行0; 左臂/左指 = 站位行 IA0 (= GraspPose 落位)
tgt = E.ref58[0].clone()
tgt[7:14] = E.ref58[E.IA0][7:14]
tgt[36:58] = E.ref58[E.IA0][36:58]
if args.squeeze > 0:
    sql = np.asarray(np.load(TC.PRIOR_AUX)["squeeze"], np.float64).reshape(-1)[7:29]
    dsq = np.clip(args.squeeze * (sql - E.ref58[E.IA0, 36:58].cpu().numpy()),
                  -TC.SQUEEZE_DELTA_CAP, TC.SQUEEZE_DELTA_CAP)
    tgt[36:58] += torch.tensor(dsq, dtype=torch.float32, device=dev)
# 预张开杯状手 (缝1 同款配方: 抓握指值沿合拢方向反推, 逐关节封顶 25°)
_sql22 = np.asarray(np.load(TC.PRIOR_AUX)["squeeze"], np.float64).reshape(-1)[7:29]
_fin_g = tgt[36:58].cpu().numpy().copy()
_fin_open = _fin_g - np.clip(2.0 * (_sql22 - _fin_g), -np.radians(25), np.radians(25))
tgt_open = tgt.clone(); tgt_open[36:58] = torch.tensor(_fin_open, dtype=torch.float32, device=dev)
PATH = None
if args.path:
    _zp2 = np.load(args.path, allow_pickle=True)
    _fn2 = [str(n) for n in _zp2["fin_names"]]
    _jl = [E.hand.joint_names.index(f"L_arm_j{i}") for i in range(1, 8)]
    _jr = [E.hand.joint_names.index(f"R_arm_j{i}") for i in range(1, 8)]
    _jfl = [E.hand.joint_names.index(n.replace("right_", "left_")) for n in _fn2]
    _jfr = [E.hand.joint_names.index(n) for n in _fn2]
    PATH = dict(lq=torch.tensor(np.asarray(_zp2["left_q"]), dtype=torch.float32, device=dev),
                lf=torch.tensor(np.asarray(_zp2["left_f"]), dtype=torch.float32, device=dev),
                rq=torch.tensor(np.asarray(_zp2["right_q"]), dtype=torch.float32, device=dev),
                rf=torch.tensor(np.asarray(_zp2["right_f"]), dtype=torch.float32, device=dev),
                ids=(_jl, _jfl, _jr, _jfr), T=len(_zp2["left_q"]))
    print(f"[grasp目检] 接近路径: {os.path.basename(args.path)} {PATH['T']} 行 (段 {list(_zp2['seg_lens'])})", flush=True)
full = E.hand.data.default_joint_pos.clone()
full[0, E.map_ids_t] = tgt_open if args.close > 0 else tgt
if PATH is not None:
    full = E.hand.data.default_joint_pos.clone()      # 路径模式: 从站姿行 0 出发
    _jl, _jfl, _jr, _jfr = PATH["ids"]
    full[0, _jl] = PATH["lq"][0]; full[0, _jfl] = PATH["lf"][0]
    full[0, _jr] = PATH["rq"][0]; full[0, _jfr] = PATH["rf"][0]
E.hand.write_joint_state_to_sim(full, torch.zeros_like(full))   # 直接摆位 (无扫掠), 合拢模式下手是张开的
E.hand.set_joint_position_target(full)
print(f"[grasp目检] 模式: {'合拢动画 ' + str(args.close) + ' 步 (张开->抓姿' + ('->squeeze' if args.squeeze > 0 else '') + '), 保持 ' + str(args.hold) + ' 步后复位循环' if args.close > 0 else '静态瞬移摆位'}", flush=True)
print(f"[grasp目检] 左先验 = {os.path.basename(TC.PRIOR_AUX)} | 右手 = 初始站姿 | "
      f"瓶 {'自由' if args.free else '钉住'} | squeeze βL={args.squeeze}", flush=True)

_PADS = ("thumb", "index", "middle", "ring", "pinky")

def readout():
    f = E._pads_f().norm(dim=-1)[0]
    bp = E.object.data.root_pos_w[0] - org
    bq = E.object.data.root_quat_w[0]
    from isaaclab.utils.math import quat_apply
    up = quat_apply(bq.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=dev))[0]
    tilt = float(torch.rad2deg(torch.acos(up[2].clamp(-1, 1))))
    lb = [i for i in E.hand_bids if E.hand.body_names[i].startswith("left")]
    hz = float(E.hand.data.body_pos_w[0, lb, 2].min() - org[2]) - 0.87
    pads = []
    for i, nm in enumerate(_PADS):
        pw = E.hand.data.body_pos_w[0, E._pad_bids[i]] - org
        r = float(((pw[:2] - bp[:2])).norm())
        pads.append(f"{nm} r{r*100:4.1f}cm F{float(f[i]):5.2f}N")
    dzc = float(E.aux.data.root_pos_w[0, 2] - E.object.data.root_pos_w[0, 2])
    print(f"[grasp目检] 左垫 {int((f[:5] > 0.5).sum())}/5 | " + " | ".join(pads)
          + f" | 瓶倾 {tilt:4.1f}° | 盖-瓶Δz {dzc*100:+.1f}cm | 左手最低-桌 {hz*100:+.1f}cm", flush=True)

_prev_tgt = [None]
def drive_to(full_new):
    """目标在 DECI 子步间线性插值 (消卡顿); 每子步走真实螺纹副 (盖骑瓶, 自由模式不再掉)。"""
    f0 = _prev_tgt[0] if _prev_tgt[0] is not None else full_new
    for i in range(DECI):
        u = (i + 1) / DECI
        E.hand.set_joint_position_target(f0 * (1 - u) + full_new * u)
        if not args.free:
            E.object.write_root_pose_to_sim(_pinb); E.object.write_root_velocity_to_sim(_zero6)
        E._SA.apply_screw(E, integrate_angle=False)
        E.scene.write_data_to_sim()
        E.sim.step(render=not args.headless)
        E.scene.update(E.sim.get_physics_dt())
        if not args.selftest:
            time.sleep(0.004)
    _prev_tgt[0] = full_new.clone()

def reset_cycle():
    E.object.write_root_pose_to_sim(_pinb); E.object.write_root_velocity_to_sim(_zero6)
    E.aux.write_root_pose_to_sim(_pinc); E.aux.write_root_velocity_to_sim(_zero6)
    if PATH is not None:
        _jl, _jfl, _jr, _jfr = PATH["ids"]
        full[0, _jl] = PATH["lq"][0]; full[0, _jfl] = PATH["lf"][0]
        full[0, _jr] = PATH["rq"][0]; full[0, _jfr] = PATH["rf"][0]
    else:
        full[0, E.map_ids_t] = tgt_open
    E.hand.write_joint_state_to_sim(full, torch.zeros_like(full))
    E.hand.set_joint_position_target(full)
    _prev_tgt[0] = full.clone()

t0 = time.time(); n = 0
STEPS = args.selftest if args.selftest else 10**9
N1 = PATH["T"] if PATH is not None else args.close          # ① 段长: 路径行数 或 合拢步数
CYCLE = (N1 + max(args.close // 2, 1) + args.hold) if (args.close > 0 or PATH is not None) else 0
while n < STEPS:
    if CYCLE:
        k = n % CYCLE
        if k == 0 and n > 0:
            print("[grasp目检] —— 复位瓶, 重新合拢 ——", flush=True)
        if k == 0:
            reset_cycle()
        if PATH is not None and k < N1:          # ① 路径: 站姿 -> cuRobo -> 六级梯 -> 抓姿
            _jl, _jfl, _jr, _jfr = PATH["ids"]
            full[0, _jl] = PATH["lq"][k]; full[0, _jfl] = PATH["lf"][k]
            full[0, _jr] = PATH["rq"][k]; full[0, _jfr] = PATH["rf"][k]
        elif PATH is None and k < N1:            # ① 合拢: 张开 -> 抓姿
            u = (k + 1) / N1
            cur = (1 - u) * tgt_open + u * tgt
            full[0, E.map_ids_t] = cur
        elif args.squeeze > 0 and k < N1 + max(args.close // 2, 1):   # ② squeeze 渐入
            u2 = (k - N1 + 1) / max(args.close // 2, 1)
            cur = tgt.clone()
            cur[36:58] += u2 * torch.tensor(np.clip(args.squeeze * (_sql22 - _fin_g),
                          -TC.SQUEEZE_DELTA_CAP, TC.SQUEEZE_DELTA_CAP), dtype=torch.float32, device=dev)
            full[0, E.map_ids_t] = cur
        drive_to(full)
        if k == CYCLE - 1:
            readout()
    else:
        drive_to(full)
    n += 1
    if time.time() - t0 > 2.0:
        readout(); t0 = time.time()
readout()
print("[grasp目检] done", flush=True)
app.close(); os._exit(0)
