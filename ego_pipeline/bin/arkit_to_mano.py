#!/usr/bin/env python3
"""EgoDex 的 ARKit 25 关节 -> MANO 15x3 轴角（`pose45`）+ `betas`。

**为什么不能直接搬角度**：ARKit 每个关节给的是世界系 4x4，看似可以取相邻关节的相对
旋转直接当 MANO 的 pose。不行 —— MANO 的 pose 是"相对**它自己的静止骨架**"的旋转，
而 ARKit 的静止骨架朝向与 MANO 不同（两边的腕坐标系轴向都不一样）。直接搬会系统性错。
所以这里**拟合**：让 MANO 前向出来的关节位置去贴 ARKit 的关节位置，误差用 mm 报出来。

拟合量（每只手）：
    R_off   (3)      常量：ARKit 腕坐标系 -> MANO 根坐标系 的固定旋转
    betas   (10)     常量：手的形状/尺寸（ARKit 是真人尺寸，MANO 默认手偏小）
    pose    (T,45)   逐帧：15 个手指关节的轴角

★ 在**腕坐标系**里拟合（把 ARKit 关节减去腕位再转进腕系），于是与世界系朝向无关 ——
Y-up/Z-up 换不换轴都不影响结果，也不需要逐帧优化 global_orient。

对应关系（MANO 侧用 HaWoR wrapper 的 OpenPose-21 顺序）：
    0 腕 | 1-4 拇指 | 5-8 食指 | 9-12 中指 | 13-16 无名指 | 17-20 小指
ARKit 的 *Metacarpal 没有 MANO 对应物（MANO 掌部是刚性的），**不参与拟合**。

输出 `pose45` 的语义必须与 HaWoR 一致（下游 retarget 按这个解读）：
    15 关节轴角 -> aa_to_rotmat -> MANOLayer(pose2rot=False)
    **不加 pose_mean、不走 PCA**（MANOLayer.forward 本来就绕过这两者）。

用法:
    python arkit_to_mano.py --hdf5 X.hdf5 --out fit.npz [--device cuda] [--iters 400]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RR = Path(__file__).resolve().parents[2]
MANO_DIR = {"right": RR / "third_party/hawor/_DATA/data/mano",
            "left": RR / "third_party/hawor/_DATA/data_left/mano_left"}

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]
# OpenPose-21 顺序里每根手指的 4 个关节, 对应 ARKit 的后缀。
# ⚠ 拇指 ARKit 只有 4 个(无 Metacarpal), 其余四指有 5 个, 丢掉 Metacarpal。
ARKIT_SUFFIX = ["Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]


def arkit_joint_names(side: str) -> list[str]:
    """→ 21 个 ARKit 键名, 顺序 = OpenPose-21。"""
    names = [f"{side}Hand"]
    for f in FINGERS:
        fname = f"{f}Finger" if f != "Thumb" else "Thumb"
        names += [f"{side}{fname}{s}" for s in ARKIT_SUFFIX]
    return names


def load_arkit(h5path: Path, side: str):
    """→ pts(T,21,3) 世界系, R_wrist(T,3,3), conf(T,)"""
    import h5py
    with h5py.File(h5path, "r") as f:
        tr = f["transforms"]
        names = arkit_joint_names(side)
        missing = [n for n in names if n not in tr]
        if missing:
            raise KeyError(f"{h5path.name} 缺关节: {missing}")
        M = np.stack([np.asarray(tr[n]) for n in names], axis=1)   # (T,21,4,4)
        key = f"{side}Hand"
        conf = None
        if "confidences" in f and key in f["confidences"]:
            conf = np.asarray(f["confidences"][key]).astype(np.float32).reshape(-1)
    pts = M[:, :, :3, 3].astype(np.float64)
    R_wrist = M[:, 0, :3, :3].astype(np.float64)
    if conf is None:
        conf = np.ones(len(pts), np.float32)
    return pts, R_wrist, conf


def to_wrist_local(pts: np.ndarray, R_wrist: np.ndarray) -> np.ndarray:
    """世界系关节 → 腕坐标系。与全局朝向/换轴无关。"""
    rel = pts - pts[:, :1, :]                       # (T,21,3)
    return np.einsum("tij,tkj->tki", np.transpose(R_wrist, (0, 2, 1)), rel)


def bone_length_report(pts: np.ndarray) -> dict:
    """自检: 若 ARKit 给的确实是世界系 4x4, 每根骨的长度应当逐帧恒定。

    ⚠ 必须**逐骨**统计: 一根手指的 3 段骨长本来就不同(近节~5cm 远节~2cm),
    混在一起算 std 会得到 ~10mm 的假象, 看起来像"骨长在变"(2026-08-12 踩过)。
    实测每根骨 std = 0.00mm —— ARKit 用的是固定尺寸的标准手骨架, 左右手逐位相同。
    """
    out = {}
    for fi, f in enumerate(FINGERS):
        i = 1 + fi * 4
        seg = np.linalg.norm(pts[:, i + 1:i + 4] - pts[:, i:i + 3], axis=-1)  # (T,3)
        out[f] = {"mean_mm": [round(float(x) * 1000, 1) for x in seg.mean(0)],
                  "std_mm": [round(float(x) * 1000, 3) for x in seg.std(0)]}
    return out


def kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """求 R 使 R@A ≈ B（A,B 都是 (N,3) 且已去心）。"""
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def build_mano(side: str, device):
    import warnings
    warnings.filterwarnings("ignore")
    import smplx
    import torch
    m = smplx.MANOLayer(model_path=str(MANO_DIR[side]), gender="neutral",
                        num_hand_joints=15, create_body_pose=False,
                        is_rhand=(side == "right"))
    # ★★ MANO_LEFT 的已知 bug (smplx issue #48): 左手模型 shapedirs 的 **x 分量符号是反的**。
    #   下游 `hawor.utils.process.run_mano_left` 默认 `fix_shapedirs=True`, 会做这一行修正;
    #   本拟合器从前不修 —— 于是拟合出的 betas 是"在**未修**模型下正确"的, 喂给下游的
    #   **已修**模型就反向解释, 左手整体缩水。
    #
    #   实测 (2026-08-15, screw_unscrew_bottle_cap/0):
    #     ARKit 真值      左 89.82 / 右 89.84 cm  -> 1.000  (同一个人, 本就该对称)
    #     修之前的产物    左 71.6  / 右 85.7      -> 0.835  (左手比真值小 20%)
    #     两实现对拍      右手任何 betas 下比值恒定; 左手 betas 非零即偏离, 且
    #                     下游(-β) ≈ 常数 × 拟合(+β)  <- 反向解释的可测等式
    #
    #   ⚠ 为什么不是"把 betas 取反": 该 bug 只反了 shape 位移的 x 分量, 不是整个 betas
    #     向量取反, 没有任何 betas 变换能等价补偿。**必须在模型层修**, 与下游对齐。
    #   ⚠ 为什么 pour/17 一直没暴露: 那条被试的手接近 MANO 平均手, 拟合出的
    #     betas ≈ 0.08, 对零向量做符号翻转没有任何影响 —— 纯属运气, 不是链路正确。
    if side == "left":
        with torch.no_grad():
            m.shapedirs[:, 0, :] *= -1
    return m.to(device).eval()


def mano_joints21(mano, global_orient_aa, pose_aa, betas, device):
    """MANO 前向 → OpenPose-21 关节。复刻 HaWoR wrapper 的 joint_map + 指尖顶点。"""
    import torch
    from smplx.vertex_ids import vertex_ids
    B = pose_aa.shape[0]
    go = aa_to_rotmat(global_orient_aa).view(B, 1, 3, 3)
    hp = aa_to_rotmat(pose_aa.reshape(B * 15, 3)).view(B, 15, 3, 3)
    o = mano(global_orient=go, hand_pose=hp, betas=betas.expand(B, -1))
    tips = torch.as_tensor(list(vertex_ids["mano"].values()), dtype=torch.long, device=device)
    joints = torch.cat([o.joints, o.vertices[:, tips, :]], dim=1)     # (B,16+5,3)
    jmap = torch.as_tensor([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19,
                            7, 8, 9, 20], dtype=torch.long, device=device)
    return joints[:, jmap, :]


def aa_to_rotmat(aa):
    """轴角 → 旋转矩阵（Rodrigues, 与 HaWoR 的 aa_to_rotmat 同一约定）。"""
    import torch
    theta = torch.norm(aa, dim=-1, keepdim=True).clamp(min=1e-8)
    k = aa / theta
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def fit_side(h5path: Path, side: str, *, device="cuda", iters=600, conf_thr=0.5,
             w_pose=2e-4, w_smooth=5e-4, verbose=True):
    """默认正则由实测扫描定(clip 0 右手, 整体关节误差):
        w_pose=2e-3 w_sm=5e-3  it=400  -> 10.92mm   (过度正则)
        w_pose=2e-4 w_sm=5e-4  it=400  ->  8.28mm
        w_pose=2e-4 w_sm=5e-4  it=1200 ->  8.20mm   (600 步后基本收敛)
        w_pose=0    w_sm=5e-4  it=1200 ->  7.76mm   (再好 0.4mm, 但没有姿态先验)
    留一点姿态先验换自然度, 代价 0.5mm。
    """
    import torch

    pts, R_wrist, conf = load_arkit(h5path, side)
    local = to_wrist_local(pts, R_wrist)              # (T,21,3) 目标
    T = len(local)
    valid = conf >= conf_thr
    if valid.sum() < 5:
        raise RuntimeError(f"{side}: 有效帧太少 ({int(valid.sum())}/{T})")

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    mano = build_mano(side, dev)
    tgt = torch.as_tensor(local, dtype=torch.float32, device=dev)

    # ---- R_off 解析初值: 用 MANO 静止手 vs ARKit 平均手做 Kabsch ----
    with torch.no_grad():
        rest = mano_joints21(mano, torch.zeros(1, 3, device=dev),
                             torch.zeros(1, 45, device=dev),
                             torch.zeros(1, 10, device=dev), dev)[0].cpu().numpy()
    rest_rel = rest - rest[0]
    mean_tgt = local[valid].mean(axis=0)
    R0 = kabsch(rest_rel[1:], mean_tgt[1:])
    from scipy.spatial.transform import Rotation
    r_off = torch.tensor(Rotation.from_matrix(R0).as_rotvec(), dtype=torch.float32,
                         device=dev, requires_grad=True)
    betas = torch.zeros(1, 10, device=dev, requires_grad=True)
    pose = torch.zeros(T, 45, device=dev, requires_grad=True)

    vmask = torch.as_tensor(valid, device=dev)
    opt = torch.optim.Adam([{"params": [pose], "lr": 0.02},
                            {"params": [r_off, betas], "lr": 0.01}])
    for it in range(iters):
        opt.zero_grad()
        j = mano_joints21(mano, r_off.expand(T, 3), pose, betas, dev)
        j = j - j[:, :1, :]
        err = ((j - tgt) ** 2).sum(-1)[vmask]                       # (Tv,21)
        loss = err.mean()
        loss = loss + w_pose * (pose ** 2).mean() + 1e-4 * (betas ** 2).mean()
        if T > 1:
            loss = loss + w_smooth * ((pose[1:] - pose[:-1]) ** 2).mean()
        loss.backward()
        opt.step()
        if verbose and (it % 100 == 0 or it == iters - 1):
            print(f"    [{side}] it{it:4d}  关节误差 {err.mean().sqrt().item()*1000:6.1f}mm")

    with torch.no_grad():
        j = mano_joints21(mano, r_off.expand(T, 3), pose, betas, dev)
        j = j - j[:, :1, :]
        per_joint = ((j - tgt) ** 2).sum(-1).sqrt()                 # (T,21)
        pj = per_joint[vmask].mean(0).cpu().numpy() * 1000
        overall = float(per_joint[vmask].mean().item() * 1000)
        # 世界系下的 MANO 根朝向: R_arkit_wrist @ R_off
        R_off_m = Rotation.from_rotvec(r_off.detach().cpu().numpy()).as_matrix()
        rot_world = Rotation.from_matrix(R_wrist @ R_off_m).as_rotvec()
        # ★ MANO 的 joint0 **不在原点**(右手 87mm / 左手 95mm)。HaWoR 存的 `trans`
        #   是 MANO 的 `transl`, 真实腕位 = trans + joint0。所以要写回管线时必须
        #   trans = 腕位 - joint0, 否则手整体偏 ~9cm(比 HaWoR 自身 74mm 误差还大)。
        j0 = mano_joints21(mano,
                           torch.as_tensor(rot_world, dtype=torch.float32, device=dev),
                           pose, betas, dev)[:, 0, :].cpu().numpy()

    return {
        "joint0": j0.astype(np.float32),        # (T,3) ARKit 世界系(Y-up)
        "pose45": pose.detach().cpu().numpy().astype(np.float32),
        "betas": betas.detach().cpu().numpy().reshape(10).astype(np.float32),
        "r_off": r_off.detach().cpu().numpy().astype(np.float32),
        "rot_world": rot_world.astype(np.float32),
        "valid": valid,
        "err_mm_overall": overall,
        "err_mm_per_joint": pj.tolist(),
        "bones": bone_length_report(pts),
    }


JOINT_LABELS = (["wrist"] +
                [f"{f}{i}" for f in ["thumb", "index", "middle", "ring", "little"]
                 for i in range(1, 5)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--conf-thr", type=float, default=0.5)
    a = ap.parse_args()

    res, report = {}, {"hdf5": str(a.hdf5)}
    for side in ("left", "right"):
        print(f"  拟合 {side} ...")
        r = fit_side(a.hdf5, side, device=a.device, iters=a.iters, conf_thr=a.conf_thr)
        res[side] = r
        report[side] = {
            "err_mm_overall": round(r["err_mm_overall"], 2),
            "err_mm_per_joint": {k: round(v, 1)
                                 for k, v in zip(JOINT_LABELS, r["err_mm_per_joint"])},
            "valid_frames": int(r["valid"].sum()),
            "bone_len_mm": {k: v["mean_mm"] for k, v in r["bones"].items()},
            "bone_len_std_mm": {k: v["std_mm"] for k, v in r["bones"].items()},
        }
        print(f"    → 整体 {r['err_mm_overall']:.1f}mm")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out,
                 pose_left=res["left"]["pose45"], pose_right=res["right"]["pose45"],
                 betas_left=res["left"]["betas"], betas_right=res["right"]["betas"],
                 rot_left=res["left"]["rot_world"], rot_right=res["right"]["rot_world"],
                 r_off_left=res["left"]["r_off"], r_off_right=res["right"]["r_off"],
                 valid_left=res["left"]["valid"], valid_right=res["right"]["valid"])
        (a.out.with_suffix(".json")).write_text(
            json.dumps(report, ensure_ascii=False, indent=1))
        print(f"  → {a.out}")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
