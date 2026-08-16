"""从重建里读出"物体在抓握时刻是怎么立着的", 供 Dexonomy 导入时摆正。

Dexonomy 默认 `--auto-up` 靠 trimesh 猜最稳静置姿态 —— 对细长物体几乎必然选"躺倒"
(pour/17 实测: 22cm 高的瓶子和 23cm 的杯子都被摆平了), 与视频里的现实相反, 也与
RL 训练场景的摆放对不上。

正确来源: world_fused 里抓握窗内的物体位姿。世界系是 gravity_z_up, 所以
    up_local = R_world←object^T @ [0,0,1]
就是"物体自身坐标系里朝上的方向"。把它传给 `import_object --up`, 规范系就复刻了
视频里的立姿; Dexonomy 随后把物体落在虚拟桌面上, 与 RL 场景的 drop-on-table 同构。

⚠ 朝向不可信时不要用它。`grasp_prompt.json` 的 `trust.object_conf_rot` 低(实测
pour/17 杯子 5.5 vs 瓶子 50.0)意味着重建的朝向本身就是错的 —— 此时退回几何先验
(容器类: 最长轴朝上), 并在输出里标明来源, 别让"看起来精确"的坏数字流到下游。

用法:
  python tools/upright_from_recon.py --recon <take目录> --object object_1 \
      [--window 31 47] [--min-conf-rot 20]
  # 输出一行可直接喂给 grasp_pipeline 的 UP="x y z"
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True, help="重建 take 目录(含 world_fused.npz)")
    ap.add_argument("--object", default="object_0")
    ap.add_argument("--window", type=int, nargs=2, default=None,
                    help="抓握窗 [起 止]; 缺省从 contact/grasp_prompt.json 读")
    ap.add_argument("--min-conf-rot", type=float, default=20.0,
                    help="朝向置信度门槛; 低于此退回几何先验(长轴朝上)")
    ap.add_argument("--use-grasp-window", action="store_true",
                    help="用抓握窗定立姿(旧行为)。**默认改用开头 --head-frames 帧的中位数**。")
    ap.add_argument("--head-frames", type=int, default=10,
                    help="取开头多少帧的中位数当立姿(视频开头物体还静置在桌上)")
    ap.add_argument("--stable-only", action="store_true",
                    help="完全不用重建朝向, 直接取**概率最高**的稳定静置姿态。"
                         "由 run_take.py 在 Step2 判 rotation_usable=false 时传入 —— "
                         "此时重建的朝向本身是错的, 拿它去'吸附最近的稳定姿态'只会把错误"
                         "四舍五入成一个看起来精确的错误。老老实实退回纯几何, 并在 source "
                         "里标 stable_pose_only, 让下游知道这个摆放没有视频依据。")
    ap.add_argument("--no-snap", action="store_true",
                    help="不吸附到稳定静置姿态(诊断用)。默认**吸附**: 桌上的刚体物理上只能处于"
                         "有限几个稳定姿态(重心投影落在支撑多边形内), 所以取与重建估计最接近的"
                         "那一个 —— 既保住'本来就横躺/倾斜'的物体(躺姿也是稳定姿态之一, 吸附角很小), "
                         "又自动抹掉重建的系统性偏差(pour/17 实测瓶 7.9°、杯 21.6°)。"
                         "吸附角本身是诊断量: >45° 说明重建离谱或物体原本靠外部支撑。")
    a = ap.parse_args()

    D = Path(a.recon)
    w = np.load(D / "world_fused.npz", allow_pickle=True)
    ids = [str(x) for x in np.atleast_1d(w["object_ids"])] if "object_ids" in w.files else ["object_0"]
    oi = ids.index(a.object) if a.object in ids else 0
    T = w["object_ob_in_world_all"][oi] if "object_ob_in_world_all" in w.files else w["object_ob_in_world"]

    conf_rot, win = None, a.window
    gp = D / "contact" / "grasp_prompt.json"
    if gp.is_file():
        d = json.loads(gp.read_text())
        for g in d.get("grasps", []):
            if g["object_id"] == a.object:
                conf_rot = (g.get("trust") or {}).get("object_conf_rot")
                win = win or g.get("grasp_window_frames")
    if not win:
        win = [0, len(T) - 1]

    lo, hi = int(win[0]), min(int(win[1]), len(T) - 1)
    allup = np.array([Rm.T @ np.array([0.0, 0.0, 1.0]) for Rm in T[:, :3, :3]])
    # ★重建世界系的 +X 已经有明确语义: place_camera.py 写明 "+X = 中间帧相机前向的水平投影",
    #   也就是"从人看向物体"。RL 场景里机器人在 -X 侧、双手伸向 +X, 两边一致。
    #   把它一并换算到物体自身坐标系, 导入时用来把规范系的**方位角**也钉死 ——
    #   否则 import_object 只对齐竖轴, 绕竖轴那一转是 trimesh 随便给的, 这条信息就丢了。
    allfx = np.array([Rm.T @ np.array([1.0, 0.0, 0.0]) for Rm in T[:, :3, :3]])
    src = "recon_grasp_window"

    if not a.use_grasp_window:
        # ★立姿一律取**开头 n 帧的中位数**(用户 2026-08-14 裁定)。理由: 视频开头物体还静置
        #   在桌上, 就是我们要复刻的初始状态; 而提取器给的抓握窗是"手物相对位姿最稳"的那段,
        #   人把东西拿起来之后才最稳 —— pour/17 实测那个窗里瓶已离桌 12.7cm、杯 6.1cm, 拿它
        #   定立姿等于把桌子摆歪。中位数而非均值: 开头几帧跟踪常有跳变(杯实测抖动 p90 56.7°)。
        head = allup[:a.head_frames]
        headx = allfx[:a.head_frames]
        headx = headx[np.isfinite(headx).all(1)]
        if len(headx) >= 3:
            fx = np.median(headx, axis=0)
            fx /= max(np.linalg.norm(fx), 1e-9)
        head = head[np.isfinite(head).all(1)]
        if len(head) >= 3:
            u = np.median(head, axis=0)
            u /= max(np.linalg.norm(u), 1e-9)
            spread = float(np.degrees(np.arccos(np.clip(head @ u, -1, 1))).max())
            src = f"recon_head{len(head)}f"
            lo, hi = 0, a.head_frames - 1
    if src.startswith("recon_grasp_window"):
        ups = allup[lo:hi + 1]
        ups = ups[np.isfinite(ups).all(1)]
        u = ups.mean(0) / max(np.linalg.norm(ups.mean(0)), 1e-9)
        spread = float(np.degrees(np.arccos(np.clip(ups @ u, -1, 1))).max()) if len(ups) else 999.0
    if conf_rot is not None and conf_rot < a.min_conf_rot:
        # 朝向不可信 -> 几何先验: 网格最长轴朝上(容器/瓶罐类的通用假设)
        import trimesh
        mp = next((p for p in (D / "objects" / a.object / "object_mesh_scaled_final.obj",
                               D / "object_mesh_scaled_final.obj") if p.is_file()), None)
        if mp is not None:
            m = trimesh.load(mp, force="mesh", process=False)
            ext = m.bounds[1] - m.bounds[0]
            u = np.eye(3)[int(np.argmax(ext))]
            src = f"prior_longest_axis(conf_rot={conf_rot} < {a.min_conf_rot})"
    fx = locals().get("fx", np.array([1.0, 0.0, 0.0]))
    snap_deg = None
    if a.stable_only:
        # 朝向不可信: 取概率最高的稳定静置姿态, 方位角也随之失去依据(front 退回 +X)
        import trimesh
        mp = next((p for p in (D / "objects" / a.object / "object_mesh_scaled_final.obj",
                               D / "object_mesh_scaled_final.obj") if p.is_file()), None)
        Ts, probs = trimesh.poses.compute_stable_poses(
            trimesh.load(mp, force="mesh", process=False).convex_hull, threshold=0.01)
        best = int(np.argmax(probs))
        u = Ts[best][:3, :3].T @ np.array([0.0, 0.0, 1.0])
        u /= max(np.linalg.norm(u), 1e-9)
        fx = np.array([1.0, 0.0, 0.0])
        src = f"stable_pose_only(p={probs[best]:.2f}, rotation_usable=false)"
    elif not a.no_snap:
        # ★吸附到最近的稳定静置姿态。只用凸包算 —— 稳定性只取决于凸包, 而原网格
        #   (杯 66 万面)直接算慢到不可用。
        import trimesh
        mp = next((p for p in (D / "objects" / a.object / "object_mesh_scaled_final.obj",
                               D / "object_mesh_scaled_final.obj") if p.is_file()), None)
        if mp is not None:
            try:
                mesh = trimesh.load(mp, force="mesh", process=False)
                Ts, probs = trimesh.poses.compute_stable_poses(mesh.convex_hull, threshold=0.01)
                cands = [(float(np.degrees(np.arccos(np.clip(
                            (Tf[:3, :3].T @ np.array([0.0, 0.0, 1.0])) @ u, -1, 1)))),
                          float(p), Tf[:3, :3].T @ np.array([0.0, 0.0, 1.0]))
                         for Tf, p in zip(Ts, probs)]
                if cands:
                    snap_deg, _p, u_snap = min(cands, key=lambda c: c[0])
                    u = u_snap / max(np.linalg.norm(u_snap), 1e-9)
                    src += f"+snap_stable({snap_deg:.1f}deg)"
            except Exception as e:
                src += f"(吸附失败:{type(e).__name__})"

    fx = fx - (fx @ u) * u                      # 与 up 正交化
    if np.linalg.norm(fx) < 1e-6:               # front 与 up 平行(退化), 随便取一个正交方向
        fx = np.cross(u, [0.0, 0.0, 1.0] if abs(u[2]) < 0.9 else [1.0, 0.0, 0.0])
    fx /= max(np.linalg.norm(fx), 1e-9)
    print(json.dumps({"object": a.object, "window": [lo, hi], "snap_deg": None if snap_deg is None else round(snap_deg, 1),
                      "up": [round(float(x), 4) for x in u],
                      "source": src, "conf_rot": conf_rot, "window_spread_deg": round(spread, 1),
                      "front": [round(float(x), 4) for x in fx],
                      "UP_ENV": " ".join(f"{x:.4f}" for x in u),
                      "FRONT_ENV": " ".join(f"{x:.4f}" for x in fx)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
