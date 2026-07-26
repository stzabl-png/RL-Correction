"""环境查看器 — 只看场景, 不跑策略, 不写日志.

用途: 调训练环境(桌子/物体/相机/手的摆位)时开着它, 改一处看一处.

  # 开 GUI, 单 env, 零残差循环回放参考轨迹
  MAGICSIM_PY=/home/lyh/luhr/MagicSim/.venv/bin/python
  $MAGICSIM_PY -m rl_rebuild.correction.env_viewer --clip Grasp2

两种"改了马上看到"的方式:

  【A. 热更新】不重启, 改 tweak JSON 即时生效 —— 相机/桌子/物体可视位姿
     默认文件: <仓库>/.env_viewer.json (不存在会自动生成一份带注释的模板)
     ⚠️ 桌子挪动只保证**视觉**更新; 物理碰撞是否跟随取决于 PhysX 的 USD 同步,
        要确认就把物体放上去看它落在哪 (tweak 里设 "drop_object": true)

  【B. 自动重启】改 cfg/env 代码后自动重建整个场景 (全部生效, 约 30s)
     加 --watch, 它会盯住 correction_env_cfg.py / correction_env.py 的 mtime,
     一变就退出(退出码 42); 配 viewer_loop.sh 就能自动拉起来.

不需要 SHARPA_WANDB=0 —— 本脚本根本不建 writer.
"""
import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", type=str, default="Grasp2")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--tweak", type=str, default=None,
                    help="热更新 JSON 路径 (默认 <仓库>/.env_viewer.json)")
parser.add_argument("--realtime", action="store_true",
                    help="按源帧率节拍播放, 让 GUI 速度和录像/原视频一致")
parser.add_argument("--play", action="store_true",
                    help="启动即播放 (忽略 tweak 里的 paused, 不用去控制器输 play)")
parser.add_argument("--follow", action="store_true",
                    help="用 IK 让 DexMate 的臂+Sharpa手跟踪参考轨迹 (代替飞手)")
parser.add_argument("--watch", action="store_true",
                    help="盯住 env cfg/代码, 一改就退出(码 42), 交给 viewer_loop.sh 重启")
parser.add_argument("--eye", type=str, default="0.9,0.9,1.35")
parser.add_argument("--lookat", type=str, default="0.0,0.0,0.90")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# GPU 独占槽位: 同一时刻只允许一个 Isaac 进程占 GPU (见 utils/gpu_guard.py).
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("viewer")

app = AppLauncher(args).app          # 不传 --headless => 开 GUI

import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from rl_rebuild.correction.env.correction_env import SharpaCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.correction_env_cfg import SharpaCorrectionEnvCfg  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TWEAK = os.path.abspath(args.tweak or os.path.join(_REPO, ".env_viewer.json"))
WATCHED = [os.path.join(os.path.dirname(__file__), "env", "correction_env_cfg.py"),
           os.path.join(os.path.dirname(__file__), "env", "correction_env.py"),
           os.path.abspath(__file__)]          # 改查看器自身也自动重建

LOGF = os.path.join(_REPO, ".env_viewer.log")


def _log(msg):
    """查看器自己的日志: 终端 + 文件双写 (Isaac 刷屏太厉害, 单看终端debug不了)."""
    line = f"[viewer] {msg}"
    print(line, flush=True)
    try:
        with open(LOGF, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


_TEMPLATE = {
    "_说明": "改这个文件保存即生效(约0.5s)。不需要的键可以删。",
    "_需要重启才生效": ["table_top_z(物理)", "num_envs", "机器人资产", "接触传感器"],
    "camera_eye": [0.9, 0.9, 1.35],
    "camera_lookat": [0.0, 0.0, 0.90],
    "table_translate": None,       # [x,y,z] 桌子中心 (None=不动)
    "table_scale": None,           # [sx,sy,sz] 相对原尺寸的缩放 (None=不动)
    "paused": False,               # true=冻住不推进参考帧
    "loop": True,                  # 回合结束自动重来
    "step_every": 1,               # >1 = 放慢 (每 N 个渲染帧才推进一步物理)
    "show_axes": False,            # 在物体原点画一个坐标轴 (需重启生效)
}


def _load_tweak():
    if not os.path.exists(TWEAK):
        with open(TWEAK, "w") as f:
            json.dump(_TEMPLATE, f, indent=2, ensure_ascii=False)
        print(f"[viewer] 已生成模板: {TWEAK}")
        return dict(_TEMPLATE)
    try:
        with open(TWEAK) as f:
            return json.load(f)
    except Exception as e:
        print(f"[viewer] tweak 解析失败 ({e}), 沿用上一份")
        return None


def _set_xform(prim, translate=None, scale=None, yaw_deg=None):
    """就地改 USD xform op (已存在就赋值, 不存在才新建 — 重复 Add 会抛异常)."""
    from pxr import UsdGeom, Gf
    xf = UsdGeom.Xformable(prim)
    ops = {o.GetOpType(): o for o in xf.GetOrderedXformOps()}
    if translate is not None:
        op = ops.get(UsdGeom.XformOp.TypeTranslate) or xf.AddTranslateOp()
        op.Set(Gf.Vec3d(*[float(v) for v in translate]))
    if scale is not None:
        op = ops.get(UsdGeom.XformOp.TypeScale) or xf.AddScaleOp()
        op.Set(Gf.Vec3f(*[float(v) for v in scale]))
    if yaw_deg is not None:
        h = math.radians(float(yaw_deg)) * 0.5
        op = ops.get(UsdGeom.XformOp.TypeOrient)     # spawn 用的是 orient(四元数)
        if op is not None:
            for Q, V in ((Gf.Quatd, Gf.Vec3d), (Gf.Quatf, Gf.Vec3f)):
                try:
                    op.Set(Q(math.cos(h), V(0.0, 0.0, math.sin(h)))); break
                except Exception:
                    continue
        else:
            (ops.get(UsdGeom.XformOp.TypeRotateZ) or xf.AddRotateZOp()).Set(float(yaw_deg))


def _apply_dexmate(base, tw, prev):
    """整机摆位 + 逐关节角度 —— 走 env 的 Articulation 张量 API.

    ⚠ 不能用 USD 的 DriveAPI.targetPosition: GPU 仿真开着
    PxSceneFlag::eENABLE_DIRECT_GPU_API, PhysX 会拒绝那条路径
    ("illegal to call this method if eENABLE_DIRECT_GPU_API is enabled"),
    属性写进去了但完全不生效 —— 这就是之前 GUI 里怎么点都不动的原因.
    """
    dm, pdm = tw.get("dexmate") or {}, prev.get("dexmate") or {}
    if not dm or dm == pdm:
        return []
    if getattr(base, "dexmate", None) is None:
        _log("⚠ 场景里没有 DexMate (SHOW_DEXMATE=1 了吗?)")
        return []
    hit = []
    bp, by = dm.get("base_pos"), dm.get("base_yaw_deg")
    if bp != pdm.get("base_pos") or by != pdm.get("base_yaw_deg"):
        try:
            if base.set_dexmate_base(pos=bp, yaw_deg=by):
                hit.append(f"底座 pos={bp} yaw={by}")
        except Exception as e:
            _log(f"⚠ 底座位姿失败: {type(e).__name__}: {e}")
    js, pjs = dm.get("joints") or {}, pdm.get("joints") or {}
    changed = {k: v for k, v in js.items() if pjs.get(k) != v}
    if changed:
        try:
            ok = base.set_dexmate_joints(changed)
            if ok:
                hit.append("关节 " + ", ".join(ok))
        except Exception as e:
            _log(f"⚠ 关节设置失败: {type(e).__name__}: {e}")
    return hit


_BASE = None


def _apply(env, tw, prev):
    """把 tweak 里能热更新的部分应用到场景. 返回实际生效的项."""
    hit = []
    # --- 相机 ---
    eye, la = tw.get("camera_eye"), tw.get("camera_lookat")
    if eye and la and (eye != prev.get("camera_eye") or la != prev.get("camera_lookat")):
        try:
            env.sim.set_camera_view(tuple(eye), tuple(la))
            hit.append("相机")
        except Exception as e:
            print(f"[viewer] 相机设置失败: {e}")
    # --- 桌子 ---
    tt, ts = tw.get("table_translate"), tw.get("table_scale")
    if (tt or ts) and (tt != prev.get("table_translate") or ts != prev.get("table_scale")):
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            n = 0
            for i in range(env.num_envs):
                p = stage.GetPrimAtPath(f"/World/envs/env_{i}/Table")
                if p.IsValid():
                    _set_xform(p, translate=tt, scale=ts)
                    n += 1
            if n:
                hit.append(f"桌子×{n}")
        except Exception as e:
            print(f"[viewer] 桌子变换失败: {e}")
    # --- 隐藏飞手 (纯视觉, 物理照常; 看 DexMate 时不挡视线) ---
    hh = tw.get("hide_hand")
    if hh != prev.get("hide_hand"):
        try:
            import omni.usd
            from pxr import UsdGeom
            stage = omni.usd.get_context().get_stage()
            n = 0
            for i in range(env.num_envs):
                pr = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
                if pr.IsValid():
                    im = UsdGeom.Imageable(pr)
                    im.MakeInvisible() if hh else im.MakeVisible()
                    n += 1
            if n:
                hit.append(f"飞手{'隐藏' if hh else '显示'}×{n}")
        except Exception as e:
            _log(f"⚠ 隐藏飞手失败: {e}")
    # --- DexMate 整机位姿 + 逐关节 ---
    try:
        hit += _apply_dexmate(_BASE, tw, prev)
    except Exception as e:
        _log(f"DexMate 控制失败: {type(e).__name__}: {e}")
    return hit


def _watched_sig():
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in WATCHED)


# ---------------- 构建环境 ----------------
env_cfg = SharpaCorrectionEnvCfg()
clips.configure_cfg(env_cfg, args.clip)
env_cfg.scene.num_envs = args.num_envs
env_cfg.rsi_prob = 0.0                       # 从头完整走一遍, 看得最全
_eye = tuple(float(x) for x in args.eye.split(","))
_lookat = tuple(float(x) for x in args.lookat.split(","))
env_cfg.viewer = ViewerCfg(eye=_eye, lookat=_lookat, origin_type="world")

base = SharpaCorrectionEnv(env_cfg)
_BASE = base
env = GymStyleEnvWrapper(base, clip_actions=env_cfg.clip_actions)

print(f"\n{'=' * 68}")
print(f"[viewer] clip={args.clip}  envs={args.num_envs}  episode={base.ep_total} 步")
print(f"[viewer] 桌面 z={env_cfg.table_top_z}  桌子尺寸={env_cfg.table_size}")
print(f"[viewer] 物体静置 z={base.obj_rest_z:.4f}")
print(f"[viewer] 热更新文件: {TWEAK}")
if args.watch:
    print(f"[viewer] --watch 已开: 改 correction_env_cfg.py / correction_env.py 会自动重启")
print(f"[viewer] Ctrl+C 退出")
print(f"{'=' * 68}\n")

# 导出 Articulation 的**真实**关节名 (URDF 里有的 USD 未必有, 以这份为准)
if getattr(base, "dexmate", None) is not None:
    _names = list(base.dexmate.joint_names)
    _p = os.path.join(_REPO, "DEXMATE_ARTICULATION_JOINTS.txt")
    with open(_p, "w") as _f:
        _f.write("\n".join(_names) + "\n")
    _log(f"DexMate Articulation 实际关节 {len(_names)} 个 -> {_p}")

# 导出 DexMate 各 body 的世界位姿 (给"人手轨迹对齐到机器人双手"用)
if getattr(base, "dexmate", None) is not None:
    import json as _json
    _bn = list(base.dexmate.body_names)
    _pw = base.dexmate.data.body_pos_w[0].tolist()
    _out = {n: [round(v, 5) for v in p] for n, p in zip(_bn, _pw)}
    _p2 = os.path.join(_REPO, "DEXMATE_BODY_POSES.json")
    with open(_p2, "w") as _f:
        _json.dump({"base_pos": list(base.cfg.dexmate_pos),
                    "joints_deg": base.cfg.dexmate_joints,
                    "bodies": _out}, _f, indent=1, ensure_ascii=False)
    _log(f"DexMate {len(_bn)} 个 body 世界位姿 -> {_p2}")

# ---------------- DexMate 臂+手 IK 跟踪 (--follow) ----------------
# 实现在 dexmate_follow.py, 和 record_replay.py 共用, 避免两边行为漂移.
_FOLLOW = None
if args.follow:
    from rl_rebuild.correction.dexmate_follow import DexmateFollower
    _f = DexmateFollower(base, log=_log)
    _FOLLOW = _f if _f.ok else None


def _follow_step(k):
    if _FOLLOW is not None:
        _FOLLOW.step(k)


zero = torch.zeros(args.num_envs, 28, device=base.device)
obs = env.reset()
# 启动时先加载一次 (文件不存在就在这里生成模板 —— 之前放在循环里被
# os.path.exists() 挡住, 模板永远生成不出来)
prev_tw = _load_tweak() or dict(_TEMPLATE)
prev_mtime = os.path.getmtime(TWEAK) if os.path.exists(TWEAK) else 0.0
# 启动时把 JSON 里的设定**应用一遍**. 之前只把它读进 prev_tw 就完事, 导致
# hide_hand / paused / 相机 这类"开机就该生效"的项必须等文件再改一次才生效.
if args.follow:
    prev_tw = dict(prev_tw); prev_tw["hide_hand"] = True   # 跟踪模式下飞手会挡视线
if args.play:
    prev_tw = dict(prev_tw); prev_tw["paused"] = False     # 启动即播放
try:
    _hit0 = _apply(base, prev_tw, {})
    _log(f"启动应用 -> {'; '.join(_hit0) if _hit0 else '(无)'}")
except Exception as e:
    _log(f"启动应用失败: {type(e).__name__}: {e}")
sig0 = _watched_sig()
t, frame = 0, 0
# 实时节拍: 一个参考帧一拍, 拍长 = 1/源帧率
_RT_DT = 0.0
if args.realtime:
    try:
        import numpy as _np
        _RT_DT = 1.0 / float(_np.load(clips.clip_entry(args.clip)["npz"],
                                      allow_pickle=True)["fps"])
        _log(f"--realtime: 按源帧率 {1/_RT_DT:.0f}fps 节拍播放")
    except Exception as _e:
        _log(f"⚠ 读源帧率失败({_e}), 不做节拍")
_rt_last = time.time()

try:
    while app.is_running():
        # --- 热更新检查 (每 30 帧 ≈ 0.5s) ---
        if frame % 30 == 0:
            if os.path.exists(TWEAK) and os.path.getmtime(TWEAK) != prev_mtime:
                prev_mtime = os.path.getmtime(TWEAK)
                tw = _load_tweak()
                if tw is not None:
                    hit = _apply(env.unwrapped if hasattr(env, "unwrapped") else base, tw, prev_tw)
                    _log(f"tweak 变更 -> {'; '.join(hit) if hit else '(无可应用项)'}")
                    prev_tw = tw
            if args.watch and _watched_sig() != sig0:
                _log("检测到 env 代码改动 -> 退出重建 (码 42)")
                try:
                    env.close()
                except Exception:
                    pass
                # 不能走 app.close()+SystemExit: Isaac 关闭流程自己会把进程以 0 退出,
                # 退出码丢了 -> viewer_loop.sh 认为是正常退出就不重启了.
                # os._exit 绕过所有清理, 保证 42 传出去 (flock 由内核回收, 不会泄漏).
                os._exit(42)
        frame += 1

        tw = prev_tw or _TEMPLATE
        if tw.get("paused"):
            # 关键: 只 render 不 step 的话 PhysX 完全不跑, 驱动目标改了也不会动.
            # 所以照常 step (物理活着, 飞手的 wrench-PD 也才有人施加),
            # 再把 episode 计数退回去 -> 参考帧原地不动 = "回放暂停".
            _follow_step(t)
            obs, _r, done, _i = env.step(zero)
            try:
                base.episode_length_buf -= 1
            except Exception:
                pass
            continue
        if frame % max(int(tw.get("step_every", 1)), 1) != 0:
            base.sim.render()
            continue

        _follow_step(t)
        obs, _r, done, _i = env.step(zero)
        t += 1
        if args.realtime and _RT_DT > 0:      # 按源帧率节拍, 否则 GUI 跑多快看多快
            _el = time.time() - _rt_last
            if _el < _RT_DT:
                time.sleep(_RT_DT - _el)
            _rt_last = time.time()
        if bool(done[0]) or t >= base.ep_total:
            if tw.get("loop", True):
                obs = env.reset(); t = 0
            else:
                base.sim.render()
except KeyboardInterrupt:
    print("\n[viewer] 退出")

env.close()
app.close()
