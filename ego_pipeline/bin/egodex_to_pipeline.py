#!/usr/bin/env python3
"""EgoDex HDF5 → 重建管线的 interim 产物（覆盖 ViPE 的相机/内参/重力，可选覆盖 HaWoR 的手）。

============================ 为什么这么做 ============================

EgoDex（Apple Vision Pro 录制）自带三样我们本来要**估**的东西：

| 我们原本 | EgoDex 提供 | 实测差别 |
|---|---|---|
| ViPE 估相机位姿 | `transforms/camera` 逐帧 4×4 | 对齐残差中位 **1mm**（12 条 EgoDex） |
| ViPE 估内参 | `camera/intrinsic` | ViPE 740.18 vs 真值 736.63（差 0.5%） |
| ViPE 估重力 + fuse 做 xy 对齐 | 世界系本身就是**重力对齐米制** | 我们的世界尺度实测跨 **0.15–1.94**，EgoDex 是设备直出真米制 |
| HaWoR 估双手 | `transforms/{left,right}Hand` 等 25 关节/手 | HaWoR 腕位锚后误差 74mm（ARCTIC mocap 实测） |

**尺度那条可能比"省掉 HaWoR"更值钱** —— 我们 σ 的运行时口径从 18mm 掉到 40mm，
那 22mm 全部来自世界尺度不确定（见 docs/CONFIDENCE_REDESIGN.md §1.7）。

============================ 坐标系对账（易错） ============================

|  | 我们 `gravity_z_up_world` | EgoDex |
|---|---|---|
| 竖直轴 | **Z 向上** | **Y 向上** |
| 原点 | **首帧相机** | **地面/人体**（相机 y≈1.63m, hip≈0.955m） |
| 水平朝向 | `middle_camera_forward_projected_xy`（中间帧相机朝向投影）| 设备建图时确定 |

本脚本**保留 EgoDex 的原点**（地面在 y=0），只做 Y-up→Z-up 换轴。这比我们原来的约定更好：
地面高度变成已知量，不再依赖手给的 `table_height=0.85` 常数。
⚠ 下游若依赖"原点在首帧相机"或那套 xy 朝向约定，需要另行适配。

============================ 变体 ============================

* `--variant dev`  —— 相机 + 内参 + 重力 + **手**全用 EgoDex。开发/评测用。
* `--variant prod` —— 只用相机 + 内参 + 重力，**手仍跑 HaWoR**。

两者之差 = **HaWoR 的误差对下游的影响**，是白捡的量化结果。
用户 2026-08-12 裁定：开发阶段除 mocap 外的数据都可用；EgoDex 手部是设备追踪不是 mocap。
⚠ 但设备手部追踪**精度未知、无标定报告**：实测 40 条 take 上腕部置信度中位 0.99（可用），
**指尖中位仅 0.33、71% 的帧 <0.5**。所以 dev 变体的手指不可当真值。

============================ 用法 ============================

    python egodex_to_pipeline.py --hdf5 <x.hdf5> --dataset egodex_auto \\
        --video-id <task>__<n> --variant dev [--interim-root ...]

先跑 vipe（拿深度），再跑本脚本覆盖 pose/intrinsics/gravity，再跑其余步骤。
顺序不能反 —— vipe 会写自己的 pose/intrinsics。

============================ ★ 深度尺度必须校正 ============================

**深度是 ViPE 在它自己的尺度下估的，而位姿换成了 EgoDex 的真实米制 —— 两者不同尺度，
直接混用会把物体放在错误的距离上。** 实测三条 EgoDex：

| clip | ViPE/真值 尺度比 | ViPE 相机移动 | 真值 |
|---|---|---|---|
| add_remove_lid__0 | 0.857 | 51mm | 45mm |
| basic_pick_place__0 | 0.769 | 116mm | 86mm |
| pick_place_food__0 | 0.722 | 49mm | 36mm |

三条一致偏大 17–39%。

★★ 2026-08-12 更正：**上面这三条的 Umeyama 结论不能外推，`screw_unscrew_bottle_cap`
上它是退化解。** 那三条相机移动 49–116mm；而本任务 29 条 clip 实测：

| | 中位 | 最大 |
|---|---|---|
| 相机包围盒对角 | **7mm** | 30mm |
| 相机路径长 | 22mm | 105mm |
| 最大转角 | 1.3° | 5.0° |

29/29 条的对角都 < 5cm —— 人坐着几乎不动头。**两条几乎静止的轨迹之间尺度不可观测**，
Umeyama 拟合的是噪声：clip0 它给 0.032，而真实值约 0.29，差 9 倍。
残差小（1mm）**不能证明 s 可靠** —— 点云退化成一团时，任何 s 配上合适的 R,t 残差都小。

现在改用 **`depth_scale_from_hands`**：EgoDex 的手是米制的，把手关节投到图像上，
拿"手到相机的真实距离"去除"ViPE 在该像素的深度"。这条路不依赖相机运动。
已验证：投影点精确落在双手上（OpenCV 约定），瓶身 CAD 长 19.7cm 在画面里约 500px，
`736.6×0.197/500 ≈ 0.29m`，与 EgoDex 说的手距 0.22–0.32m 相符。

⚠ 对比 ARCTIC：那边 s 跨 0.15–1.94 且**无从校正**（没有设备位姿也没有米制手），
是 σ 精度从 18mm 掉到 40mm 的元凶。EgoDex 有米制手当参照，这个问题在这里是可解的。
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

RR = Path(__file__).resolve().parents[2]

# EgoDex 是 Y-up，我们是 Z-up。绕 X 轴 +90°：(x, y, z) → (x, -z, y)
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]])


def h5_read(path: Path):
    import h5py
    with h5py.File(path, "r") as f:
        cam = f["transforms/camera"][:].astype(np.float64)
        K = f["camera/intrinsic"][:].astype(np.float64)
        hands, conf = {}, {}
        for side, key in (("left", "leftHand"), ("right", "rightHand")):
            hands[side] = f[f"transforms/{key}"][:].astype(np.float64)
            # ⚠ 有的 clip 没有 confidences 分组(clip3 实测) —— 缺就当全可信, 不要整条挂掉
            ck = f"confidences/{key}"
            conf[side] = (f[ck][:].astype(np.float64) if ck in f
                          else np.ones(len(hands[side]), np.float64))
        # 全部手部关节(两手 25x2), 供 depth_scale 标定用 —— 关节越多样本越稳
        jn = [k for k in f["transforms"]
              if ("Finger" in k or "Thumb" in k or k.endswith("Hand")) and "Arm" not in k]
        joints = np.stack([f[f"transforms/{k}"][:, :3, 3] for k in sorted(jn)],
                          axis=1).astype(np.float64)
        attrs = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.attrs.items()}
    return cam, K, hands, conf, attrs, joints


def depth_scale_from_hands(itm: Path, vid: str, cam: np.ndarray, K: np.ndarray,
                           joints: np.ndarray, *, patch: int = 5,
                           lo_pct: float = 20.0) -> dict:
    """用 **EgoDex 的米制手** 标定 ViPE 深度的尺度。

    ★ 为什么不能用相机轨迹(Umeyama): 本任务 29 条 clip 的相机包围盒对角**中位仅 7mm**、
      最大转角 1.3deg —— 两条几乎静止的轨迹之间尺度**不可观测**, 解出来的是噪声
      (clip0 实测 Umeyama 给 0.032, 手法给 0.29, 差 9 倍)。

    做法: EgoDex 手关节(米制,世界系) --EgoDex 相机--> 像素 + 真实深度 Z_m;
          读 ViPE 深度同像素 Z_v;  scale = median(Z_m / Z_v)。

    两个必须的细节:
      * ViPE 深度是 **HALF Z 通道 EXR**, `cv2.imdecode` 读出来**全零且不报错** ——
        必须用 `_common.io.read_depth_exr_bytes`(2026-08-12 在这上面绕了一圈)。
      * 相机是 **OpenCV 约定**(+z 前 +y 下), 已用"投影点画到真实帧上"验证:
        点精确落在双手上; y/xy 翻转会落到墙上。
      * 取邻域低分位而非单点: 关节投影常压在手的轮廓边上, 单点会采到背景(更远),
        把 scale 系统性拉小。手是画面里最近的东西, 故取低分位。
    """
    import sys as _sys
    import zipfile
    _sys.path.insert(0, str(RR / "ego_pipeline/Reconstruction/recon_pipeline"))
    from _common.io import read_depth_exr_bytes            # noqa: E402

    zpath = itm / "vipe/depth" / f"{vid}.zip"
    if not zpath.is_file():
        return {"depth_scale": None, "reason": f"缺 ViPE 深度: {zpath}"}
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    zf = zipfile.ZipFile(zpath)
    names = sorted(n for n in zf.namelist() if n.lower().endswith(".exr"))
    T = min(len(names), len(cam), len(joints))
    per_frame = []
    for t in range(T):
        d = read_depth_exr_bytes(zf.read(names[t]))
        H, W = d.shape
        Rw, tw = cam[t, :3, :3], cam[t, :3, 3]
        pc = (Rw.T @ (joints[t] - tw).T).T
        zc = pc[:, 2]
        ok = zc > 0.05
        if not ok.any():
            continue
        u = (fx * pc[:, 0] / zc + cx) * W / (2 * cx)
        v = (fy * pc[:, 1] / zc + cy) * H / (2 * cy)
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        r = patch // 2
        ok &= (ui >= r) & (ui < W - r) & (vi >= r) & (vi < H - r)
        if not ok.any():
            continue
        vals = []
        for uu, vv, zm in zip(ui[ok], vi[ok], zc[ok]):
            win = d[vv - r:vv + r + 1, uu - r:uu + r + 1]
            win = win[np.isfinite(win) & (win > 1e-6)]
            if win.size:
                vals.append(zm / np.percentile(win, lo_pct))
        if vals:
            per_frame.append(float(np.median(vals)))
    if not per_frame:
        return {"depth_scale": None, "reason": "没有可用的手/深度样本"}
    pf = np.array(per_frame)
    med = float(np.median(pf))
    return {"depth_scale": med,
            "depth_scale_method": "hand_metric_vs_vipe_depth",
            "depth_scale_frames": int(len(pf)),
            "depth_scale_spread": float(np.median(np.abs(pf - med)) / med),
            "depth_scale_p25_p75": [float(np.percentile(pf, 25)),
                                    float(np.percentile(pf, 75))]}


def umeyama_scale(P: np.ndarray, Q: np.ndarray) -> tuple[float, float]:
    """P->Q 的相似变换尺度 + 对齐残差中位(m)。

    ⚠ **不要再用它求 depth_scale** —— EgoDex 的相机几乎不动, 尺度不可观测。
    保留仅用于报告"我们的位姿与 ViPE 位姿的对齐残差"这类诊断量。
    """
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(X.T @ Y)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = float((S * np.diag(D)).sum() / (X ** 2).sum())
    t = mq - s * R @ mp
    res = np.linalg.norm((s * (R @ P.T).T + t) - Q, axis=1)
    return s, float(np.median(res))


def to_z_up(T: np.ndarray) -> np.ndarray:
    """(T,4,4) Y-up → Z-up。旋转部分左乘换轴矩阵，平移同样换轴。"""
    out = np.tile(np.eye(4), (len(T), 1, 1))
    out[:, :3, :3] = Y_UP_TO_Z_UP @ T[:, :3, :3]
    out[:, :3, 3] = (Y_UP_TO_Z_UP @ T[:, :3, 3].T).T
    return out


def write_vipe(step_dir: Path, vid: str, c2w: np.ndarray, K: np.ndarray) -> None:
    """覆盖 vipe 的 pose / intrinsics / gravity（格式与 ViPE 自己的产物一致）。"""
    n = len(c2w)
    inds = np.arange(n, dtype=np.int64)

    (step_dir / "pose").mkdir(parents=True, exist_ok=True)
    np.savez(step_dir / "pose" / f"{vid}.npz", data=c2w.astype(np.float32), inds=inds)

    # intrinsics 的 data 是 (T,4) = [fx, fy, cx, cy]（实测格式）
    (step_dir / "intrinsics").mkdir(parents=True, exist_ok=True)
    row = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], np.float32)
    np.savez(step_dir / "intrinsics" / f"{vid}.npz",
             data=np.tile(row, (n, 1)), inds=inds)

    # 换轴后世界系已是 Z-up，重力就是 -Z（不确定度给 0：这是设备 IMU 的量，不是估计）
    g = step_dir / "gravity"
    g.mkdir(parents=True, exist_ok=True)
    doc = {"schema_version": "vipe_gravity_v2", "source": "egodex_device_imu",
           "sample_frame_indices": [0], "gravity_world": [0.0, 0.0, -1.0],
           "up_world": [0.0, 0.0, 1.0], "gravity_uncertainty": [0.0, 0.0, 0.0],
           "note": "EgoDex 世界系本身重力对齐(Y-up)，换轴到 Z-up 后重力恒为 -Z"}
    (g / f"{vid}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    # ★ 必须写全 `_common/gravity.py: load_vipe_gravity` 要求的字段, 否则 fuse 直接抛
    #   "ViPE gravity artifact missing keys"。必需集: schema_version / gravity_world /
    #   up_world / sample_frame_indices / gravity_cam  (2026-08-12 只写前两个 -> fuse 全挂)
    n = len(c2w)
    idx = np.unique(np.clip([0, n // 2, n - 1], 0, max(n - 1, 0))).astype(np.int32)
    g_world = np.array([0.0, 0.0, -1.0])
    # 重力在**相机系**下的方向: R_c2w^T @ g_world
    g_cam = np.stack([c2w[i, :3, :3].T @ g_world for i in idx])
    np.savez(g / f"{vid}.npz",
             schema_version=np.array("vipe_gravity_v2"),
             source=np.array("egodex_device"),
             sample_frame_indices=idx,
             gravity_world=g_world, up_world=-g_world,
             gravity_cam=g_cam, up_cam=-g_cam,
             gravity_world_samples=np.tile(g_world, (len(idx), 1)),
             up_world_samples=np.tile(-g_world, (len(idx), 1)),
             # 设备重力对齐, 不确定度记 0 —— 与 ViPE 的 GeoCalib 估计不同, 这是已知量
             gravity_uncertainty=np.zeros(len(idx)))


def write_hawor(step_dir: Path, vid: str, hands: dict, conf: dict, thr: float,
                h5path: Path | None = None, mano_fit: bool = True,
                device: str = "cuda", iters: int = 600) -> dict:
    """写 HaWoR 等价产物。

    格式（实测 `world_space_res.pth` 是**纯 pickle**，不是 torch 归档）：
        list[5] = [trans(2,T,3), rot(2,T,3), pose(2,T,45), betas(2,T,10), valid(2,T)]
        顺序: index 0=left, 1=right

    ★ 三个必须与 HaWoR 对齐的语义（2026-08-12 逐条实测确定，之前全都错）：

    1. `pose45` = **15 关节轴角**，经 `aa_to_rotmat` 后喂 `MANOLayer(pose2rot=False)`，
       **不加 pose_mean、不走 PCA**。由 `arkit_to_mano` 拟合得到（拟合而非直接搬角度：
       MANO 的 pose 是相对**它自己的静止骨架**，与 ARKit 静止骨架朝向不同）。
    2. `rot` = MANO 的 `global_orient` = `R_arkit_wrist @ R_off`，其中 `R_off` 是
       ARKit 腕系→MANO 根系的常量旋转，一并拟合出来。**不是 ARKit 腕部朝向原样。**
    3. `trans` = MANO 的 `transl` = **腕位 − joint0**。MANO 的 joint0 不在原点
       （右 87mm / 左 95mm），直接写腕位会让手整体偏 ~9cm。

    `mano_fit=False` 时退回全零手指（旧行为），并把 `rot/trans` 保持为 ARKit 原样 ——
    只在拟合失败时使用，`run_meta.json` 里会记明。
    """
    from scipy.spatial.transform import Rotation
    n = hands["left"].shape[0]
    trans = np.zeros((2, n, 3), np.float32)
    rot = np.zeros((2, n, 3), np.float32)
    pose45 = np.zeros((2, n, 45), np.float32)
    betas = np.zeros((2, n, 10), np.float32)
    valid = np.zeros((2, n), np.float32)

    fit = {}
    if mano_fit and h5path is not None:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from arkit_to_mano import fit_side          # noqa: E402
            for side in ("left", "right"):
                fit[side] = fit_side(h5path, side, device=device, iters=iters,
                                     verbose=False)
                print(f"[egodex] MANO 拟合 {side}: 关节误差 "
                      f"{fit[side]['err_mm_overall']:.1f}mm")
        except Exception as e:                          # 拟合失败不该拖垮注入
            print(f"[egodex] ⚠ MANO 拟合失败({type(e).__name__}: {e}), 手指退回零占位")
            fit = {}

    for i, side in enumerate(("left", "right")):
        T = to_z_up(hands[side])
        valid[i] = (conf[side] >= thr).astype(np.float32)
        if side in fit:
            f = fit[side]
            m = min(n, len(f["pose45"]))
            pose45[i, :m] = f["pose45"][:m]
            betas[i, :m] = f["betas"][None, :]
            # rot_world / joint0 是在 ARKit 的 Y-up 世界系里算的 -> 换轴到 Z-up
            Rw = Rotation.from_rotvec(f["rot_world"][:m]).as_matrix()
            rot[i, :m] = Rotation.from_matrix(Y_UP_TO_Z_UP @ Rw).as_rotvec()
            # ⚠ joint0 **不换轴**: 它是 MANO 自身模板坐标里的常量(只随 betas 变),
            #   LBS 里根关节的旋转是**绕 J0 施加**的, J0 本身不被 global_orient 旋转。
            #   误当成世界系向量去换轴会让腕位偏 |J0 - M·J0| = 11.3mm(往返验证抓到)。
            trans[i, :m] = T[:m, :3, 3] - f["joint0"][:m]
            if m < n:      # 拟合帧数不足时尾部退回原样, 免得留一段零
                rot[i, m:] = Rotation.from_matrix(T[m:, :3, :3]).as_rotvec()
                trans[i, m:] = T[m:, :3, 3]
        else:
            trans[i] = T[:, :3, 3]
            rot[i] = Rotation.from_matrix(T[:, :3, :3]).as_rotvec()

    payload = [trans, rot, pose45, betas, valid]
    out = step_dir / vid
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "world_space_res.pth", "wb") as f:
        pickle.dump(payload, f, protocol=4)
    (out / "run_meta.json").write_text(json.dumps({
        "sequence": vid, "camera_source": "egodex", "hand_source": "egodex_arkit",
        "conf_threshold": thr,
        "valid_fraction": {s: float(valid[i].mean()) for i, s in enumerate(("left", "right"))},
        "warning": "pose(45)/betas 为零占位，尚未做 ARKit 25 关节 → MANO 15 关节映射",
    }, ensure_ascii=False, indent=2))
    return {s: float(valid[i].mean()) for i, s in enumerate(("left", "right"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--interim-root", type=Path,
                    default=Path.home() / "Reconstruct_and_Retarget/Output/ReconstructOutput/interim")
    ap.add_argument("--no-mano-fit", action="store_true",
                    help="不拟合 ARKit->MANO 手指, 退回零占位(旧行为)")
    ap.add_argument("--mano-iters", type=int, default=600)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variant", choices=("dev", "prod"), default="dev")
    ap.add_argument("--hand-conf", type=float, default=0.5)
    a = ap.parse_args()

    cam, K, hands, conf, attrs, joints = h5_read(a.hdf5)
    base = a.interim_root / a.dataset / a.video_id
    c2w = to_z_up(cam)

    # ★ 求 depth_scale: ViPE 的深度在它自己的尺度下, 位姿要换成真实米制 -> 必须缩放。
    #   在**换轴前**的原始系里比(两边都用各自的原始坐标), 尺度与坐标系朝向无关。
    ds = depth_scale_from_hands(base, a.video_id, cam, K.reshape(3, 3), joints)
    depth_scale = ds.get("depth_scale") or 1.0
    if ds.get("depth_scale") is None:
        print(f"[egodex] ⚠ depth_scale 求不出({ds.get('reason')}), 退回 1.0 —— "
              "物体距离会系统性偏大, 该条数据的尺度不可信")
    # 诊断: 相机轨迹的运动幅度。太小说明**任何基于轨迹的尺度估计都不可用**。
    cam_span = float(np.linalg.norm(cam[:, :3, 3].max(0) - cam[:, :3, 3].min(0)))

    write_vipe(base / "vipe", a.video_id, c2w, K)
    print(f"[egodex] vipe 覆盖: {len(c2w)} 帧  K=[fx {K[0,0]:.1f} fy {K[1,1]:.1f} "
          f"cx {K[0,2]:.0f} cy {K[1,2]:.0f}]")
    print(f"         相机高度(Z-up) {c2w[:, 2, 3].min():.2f}~{c2w[:, 2, 3].max():.2f} m  "
          f"水平跨度 {np.linalg.norm(c2w[:, :2, 3].ptp(0)):.2f} m")
    print(f"[egodex] ★ depth_scale = {depth_scale:.4f}  (手的米制距离 vs ViPE 深度; "
          f"{ds.get('depth_scale_frames','?')} 帧, 帧间离散 "
          f"{ds.get('depth_scale_spread', float('nan')):.3f})")
    print(f"         相机运动包围盒 {cam_span*1000:.0f}mm —— "
          f"{'太小, 轨迹法求尺度不可用(已改用手法)' if cam_span < 0.3 else '足够, 轨迹法也可交叉验证'}")

    if a.variant == "dev":
        fr = write_hawor(base / "hawor", a.video_id, hands, conf, a.hand_conf,
                         h5path=a.hdf5, mano_fit=not a.no_mano_fit,
                         device=a.device, iters=a.mano_iters)
        print(f"[egodex] hawor 覆盖(dev): 有效帧比例 左 {fr['left']:.0%} 右 {fr['right']:.0%}"
              f"  (conf>={a.hand_conf})")
    else:
        print("[egodex] variant=prod: 手部保留 HaWoR，不覆盖")

    (base / "egodex_source.json").write_text(json.dumps({
        "hdf5": str(a.hdf5), "variant": a.variant,
        "frames": int(len(c2w)), "task_attrs": attrs,
        "depth_scale": depth_scale,
        "camera_span_m": cam_span,
        **{k: v for k, v in ds.items() if k != "depth_scale"},
        "depth_scale_note": "ViPE 深度在其自身尺度下; 位姿已换成真实米制, 故 fp_pose/"
                            "sam3d_scale 必须用 --depth-scale=该值, 否则物体距离系统性偏大",
        "frame_convention": "y_up_to_z_up, origin kept at EgoDex world origin (ground)",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
