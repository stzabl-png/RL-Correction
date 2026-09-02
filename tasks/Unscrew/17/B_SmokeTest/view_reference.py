"""Unscrew/17 全链参考活样机 (照 Pour/17 view_reference 物理体制, 2026-09-01 §10 U-P2):
  Approach/Retreat: 机器人碰撞开(真实物理), 物体钳在静置位;
  缝1/交互/缝2:      机器人碰撞关(编舞语义), 物体逐帧钳到母带物轨行, 手走 ref58 行。
用法 (GUI):
  bash tasks/Unscrew/17/B_SmokeTest/view_ref.sh            # 默认 v2 母带, 15fps, 循环
自检 (无头抽帧):  ... view_ref.sh --selftest --headless
"""
import argparse, os, sys, time, select
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--fps", type=float, default=15.0)
p.add_argument("--start_row", type=int, default=0)
p.add_argument("--loop", type=int, default=1, help="1=播完静置3s重来; 0=播完定格")
p.add_argument("--squeeze", type=float, default=0.0, help=">0 = 缝1起加 squeeze 剂量 βL (默认0=纯母带行)")
p.add_argument("--selftest", action="store_true", help="无头抽帧冒烟: 各段跑通+钳位误差")
p.add_argument("--record", default="", help="录 mp4 路径 (需 --enable_cameras, 可无头)")
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
for _ in range(30):
    E.scene.write_data_to_sim(); E.sim.step(render=False); E.scene.update(E.sim.get_physics_dt())
org = E.scene.env_origins[0]

# ---- 母带与分段 ----
Z = np.load(PE.MASTER, allow_pickle=True)
T, APP, IA0, IA1, RET0 = E.T_ROW, E.APP_END, E.IA0, E.IA1, E.RETREAT0
SEG = [("approach", 0, APP, True), ("seam1", APP, IA0, False),
       ("interact", IA0, IA1 + 1, False), ("seam2", IA1 + 1, RET0, False),
       ("retreat", RET0, T, True)]
print(f"[view] 母带 {os.path.basename(PE.MASTER)} 全链 {T} 行 | " +
      " ".join(f"{n}[{a},{b})碰撞{'开' if c else '关'}" for n, a, b, c in SEG), flush=True)
OBJ = {oi: (torch.tensor(np.asarray(Z[f"obj_pos_{oi}"]), dtype=torch.float32, device=dev),
            torch.tensor(np.asarray(Z[f"obj_quat_{oi}"]), dtype=torch.float32, device=dev))
       for oi in (0, 1)}

# ---- 机器人碰撞开关 (Pour 定案同款: UsdPhysics.CollisionAPI 逐体切换) ----
import omni.usd
from pxr import UsdPhysics
_stage = omni.usd.get_context().get_stage()
_COLS = [UsdPhysics.CollisionAPI(pr).CreateCollisionEnabledAttr() for pr in _stage.Traverse()
         if "/Robot" in str(pr.GetPath()) and pr.HasAPI(UsdPhysics.CollisionAPI)]
_col_state = [None]
def set_collision(on):
    if _col_state[0] == bool(on):
        return
    for a in _COLS:
        a.Set(bool(on))
    _col_state[0] = bool(on)
    print(f"[view] 机器人碰撞: {'开' if on else '关'} ({len(_COLS)} 体)", flush=True)

# ---- 轨迹曲线 (conf 三档着色, 画交互窗) ----
try:
    for oi, tag in ((0, "瓶"), (1, "盖")):
        P = np.asarray(Z[f"obj_pos_{oi}"])[IA0:IA1 + 1] + org.cpu().numpy()
        c = np.asarray(Z.get(f"conf_pos_{oi}", np.full(T, np.nan)))[IA0:IA1 + 1]
        tier = np.where(np.isnan(c), 1, np.where(c >= 70, 2, np.where(c >= 40, 1, 0)))
        COL = {2: (0.15, 0.85, 0.15), 1: (0.95, 0.85, 0.15), 0: (0.95, 0.15, 0.15)}
        s = 0
        for e in range(1, len(P) + 1):
            if e == len(P) or tier[e] != tier[s]:
                if e - s >= 2:
                    E._draw_curve(f"/World/RefTraj_{tag}_{s}", P[s:e], COL[int(tier[s])], 0.003)
                s = e
        print(f"[view] {tag} 交互轨迹画好: 绿{int((tier==2).sum())} 黄{int((tier==1).sum())} 红{int((tier==0).sum())} 行")
except Exception as e:
    print(f"[view] 轨迹曲线跳过 ({e})")

# ---- squeeze 剂量 (可选) ----
dsq = torch.zeros(22, device=dev)
if args.squeeze > 0:
    sql = np.asarray(np.load(TC.PRIOR_AUX)["squeeze"], np.float64).reshape(-1)[7:29]
    ref0 = E.ref58.cpu().numpy()
    dsq = torch.tensor(np.clip(args.squeeze * (sql - ref0[IA0, 36:58]),
                               -TC.SQUEEZE_DELTA_CAP, TC.SQUEEZE_DELTA_CAP),
                       dtype=torch.float32, device=dev)

def pin_objs(row):
    for oi, ob in ((0, E.object), (1, E.aux)):
        P, Q = OBJ[oi]
        st = torch.cat([P[row] + org, Q[row]]).unsqueeze(0)
        ob.write_root_pose_to_sim(st)
        ob.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))

def drive(row, sL, spf=0.0):
    r0, r1 = min(row, T - 1), min(row + 1, T - 1)
    full = E.hand.data.joint_pos.clone()
    for i in range(DECI):
        u = (i + 1) / DECI
        tgt = (1 - u) * E.ref58[r0] + u * E.ref58[r1]      # 子步插值 -> 不卡顿
        tgt = tgt.clone(); tgt[36:58] += sL * dsq
        full[0, E.map_ids_t] = tgt
        E.hand.set_joint_position_target(full)
        pin_objs(r0 if u < 0.5 else r1)
        E.scene.write_data_to_sim()
        E.sim.step(render=not args.headless)
        E.scene.update(E.sim.get_physics_dt())
        if spf > 0:
            time.sleep(spf)

paused = [False]
def poll_pause():
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        paused[0] = not paused[0]
        print(f"[view] {'⏸ 暂停 (回车继续)' if paused[0] else '▶ 继续'}", flush=True)

def play(abbr=False):
    err_max = 0.0
    for name, a, b, col in SEG:
        set_collision(col)
        rng = range(max(a, args.start_row), b, max(1, (b - a) // 8) if abbr else 1)
        for row in rng:
            spf = 0.0 if abbr else max(0.0, (1.0 / args.fps) / DECI - 0.004)
            drive(row, 1.0 if (args.squeeze and row >= APP) else 0.0, spf)
            if not abbr:
                poll_pause()
                while paused[0]:
                    time.sleep(0.1); poll_pause()
        # 段末钳位误差 (自检读数)
        bp = E.object.data.root_pos_w[0] - org
        e_ = float((bp - OBJ[0][0][min(b - 1, T - 1)]).norm())
        err_max = max(err_max, e_)
        print(f"[view] 段 {name:8s} 完 (行 {a}..{b - 1}) | 瓶钳位误差 {e_*100:.2f}cm", flush=True)
    return err_max

if args.record:
    import imageio
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom
    _st2 = omni.usd.get_context().get_stage()
    _camp = UsdGeom.Camera.Define(_st2, "/World/RecCam")
    _camp.CreateFocalLengthAttr().Set(16.0)
    _m = Gf.Matrix4d()
    _m.SetLookAt(Gf.Vec3d(0.85, -1.15, 1.60), Gf.Vec3d(-0.15, 0.10, 0.95), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(_camp).AddTransformOp().Set(_m.GetInverse())
    _rp = rep.create.render_product("/World/RecCam", (1280, 720))
    _annot = rep.AnnotatorRegistry.get_annotator("rgb")
    _annot.attach(_rp)
    _frames = []
    _rec_n = [0]
    def _grab_rows():
        pass
    # 逐行播 (每行抓一帧): 用 abbr=False 全行, 但把 spf=0 抓帧
    for name, a, b, col in SEG:
        set_collision(col)
        for row in range(a, b):
            drive(row, 0.0, 0.0)
            E.sim.render()
            d = _annot.get_data()
            if d is not None and getattr(d, "size", 0):
                _frames.append(np.asarray(d)[..., :3].astype(np.uint8))
    imageio.mimsave(args.record, _frames, fps=15)
    print(f"[view] 录像 {args.record} | {len(_frames)} 帧 @15fps", flush=True)
    app.close(); os._exit(0)
if args.selftest:
    e = play(abbr=True)
    print(f"[view] 自检: 五段跑通, 瓶钳位误差峰 {e*100:.2f}cm {'✅' if e < 0.01 else '⚠'}", flush=True)
else:
    print("[view] ▶ 播放中 (终端回车=暂停/继续)", flush=True)
    while True:
        play()
        if not args.loop:
            print("[view] 播完定格 (Ctrl+C 退出)", flush=True)
            while True:
                time.sleep(1)
        print("[view] 3s 后重播…", flush=True)
        time.sleep(3)
app.close(); os._exit(0)
