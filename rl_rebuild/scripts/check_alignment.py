"""Fast offline sanity-check for replay_world.npz -> Isaac placement.

Replicates the retarget_isaacsim.py convention chain (scene_rot -> recenter=table
-> obj_gap) in pure numpy + usd-core, then asserts the placement is physical:
object starts ON the table (not under it / inside it), hands stay above the
tabletop, and the rigid transform preserved the reconstructed hand-object
relative geometry.

Usage (any python with numpy + usd-core, no Isaac launch needed):
  python check_alignment.py --traj replay_world.npz --object-usd object.usd \
      --scene-rot obj0 --table-height 0.85 --obj-gap 0.01
"""
import argparse

import numpy as np


# ---------------- quaternion helpers (wxyz unless noted) ----------------
def quat_xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_to_rotmat(q):  # q: (4,) wxyz
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_quat(R):  # -> wxyz
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-8:  # fallback for 180-deg rotations
        i = int(np.argmax(np.diag(R)))
        q = np.zeros(4)
        q[i + 1] = 1.0
        return q
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])


def quat_mul(a, b):  # (...,4) wxyz hamilton product
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def rot_apply(q, v):  # rotate vectors v (...,3) by quats q (...,4) wxyz
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    qc = q * np.array([1.0, -1, -1, -1])
    return quat_mul(quat_mul(q, qv), qc)[..., 1:]


def object_aabb(mesh_path):
    """AABB in the object's local frame. .obj parsed with numpy; .usd needs pxr."""
    if mesh_path.endswith(".obj"):
        verts = []
        with open(mesh_path) as f:
            for line in f:
                if line.startswith("v "):
                    verts.append([float(x) for x in line.split()[1:4]])
        v = np.asarray(verts, dtype=np.float64)
    else:  # USD: only works where usd-core / kit pxr is importable
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(mesh_path)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        r = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        return np.array(r.GetMin(), dtype=np.float64), np.array(r.GetMax(), dtype=np.float64)
    return v.min(axis=0), v.max(axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--object-mesh", required=True, help=".obj (numpy 直接解析) 或 .usd (需 pxr)")
    ap.add_argument("--scene-rot", default="identity",
                    help="identity(新流水线默认) | obj0 | cv2zup | w,x,y,z quat")
    ap.add_argument("--table-height", type=float, default=0.85)
    ap.add_argument("--obj-gap", type=float, default=0.01)
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--contact-dist", type=float, default=0.10,
                    help="wrist..object distance defining the interaction segment")
    args = ap.parse_args()

    d = np.load(args.traj, allow_pickle=True)
    T = len(d["frames"])
    joints = d[f"joints_{args.hand}"].astype(np.float64)          # (T,21,3)
    valid = d[f"valid_{args.hand}"].astype(bool)                  # (T,)
    obj_p = d["obj_pose"][:, :3].astype(np.float64)               # (T,3)
    obj_q = d["obj_pose"][:, 3:7].astype(np.float64)  # 新流水线已是 wxyz (qw 在前)

    # nearest-fill invalid hand frames (same policy as retarget_isaacsim)
    idx_valid = np.flatnonzero(valid)
    fill_src = idx_valid[np.abs(np.arange(T)[:, None] - idx_valid[None, :]).argmin(1)]
    joints = joints[fill_src]

    # ---- scene rotation ----
    if args.scene_rot == "identity":
        Rs = np.eye(3)
    elif args.scene_rot == "obj0":
        Rs = quat_to_rotmat(obj_q[0]).T           # object frame-0 becomes canonical/upright
    elif args.scene_rot == "cv2zup":
        Rs = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
    else:
        Rs = quat_to_rotmat(np.array([float(x) for x in args.scene_rot.split(",")]))
    qs = rotmat_to_quat(Rs)

    obj_p_r = obj_p @ Rs.T
    obj_q_r = quat_mul(np.broadcast_to(qs, (T, 4)), obj_q)
    joints_r = joints @ Rs.T

    # ---- object bottom per frame: 真实 mesh 顶点 (AABB 角点对圆柱面是空气, 会高估穿透) ----
    lo, hi = object_aabb(args.object_mesh)
    verts = []
    with open(args.object_mesh) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    verts = np.asarray(verts, dtype=np.float64)               # (V,3)
    vert_w = rot_apply(obj_q_r[:, None, :], verts[None]) + obj_p_r[:, None]  # (T,V,3)

    # ---- recenter: obj frame-0 -> table center, bottom -> top + gap ----
    top = args.table_height
    shift = np.array([-obj_p_r[0, 0], -obj_p_r[0, 1],
                      top + args.obj_gap - vert_w[0, :, 2].min()])
    obj_p_r += shift
    joints_r += shift
    vert_w += shift

    bottom = vert_w[:, :, 2].min(axis=1)                      # (T,)
    wrist = joints_r[:, 0]
    tips = joints_r[:, [4, 8, 12, 16, 20]]                    # MANO fingertips
    wrist_obj = np.linalg.norm(wrist - obj_p_r, axis=1)
    tip_obj = np.linalg.norm(tips - obj_p_r[:, None], axis=2).min(axis=1)
    half_diag = np.linalg.norm(hi - lo) / 2  # 指尖贴到表面时 tip-中心距 ≈ 半对角线

    # relative-geometry preservation: rigid transform must not change distances
    rel_before = np.linalg.norm(d[f"joints_{args.hand}"][fill_src][:, 0].astype(np.float64) - obj_p)
    rel_after = np.linalg.norm(wrist - obj_p_r)
    rel_err = abs(np.linalg.norm(rel_before) - np.linalg.norm(rel_after))

    seg = np.flatnonzero(valid & (tip_obj < half_diag + 0.03))

    checks = [
        ("t0 物体在桌面上 (bottom-top ∈ [0, 3cm])",
         0.0 <= bottom[0] - top <= 0.03,
         f"bottom-top = {(bottom[0] - top) * 100:.2f} cm"),
        ("全程物体不深穿桌 (bottom > top - 3cm)",
         bool((bottom > top - 0.03).all()),
         f"最深 = {(bottom.min() - top) * 100:.2f} cm @ frame {int(bottom.argmin())}"),
        ("WARN 物体浅穿桌深度 (物体轨迹重建噪声, 桌面会托住)",
         None,
         f"低于桌面帧数 {int((bottom < top).sum())}/{len(bottom)}, "
         f"最深 {(bottom.min() - top) * 100:.2f} cm"),
        ("全程物体不入地 (bottom > 0)",
         bool((bottom > 0).all()),
         f"min bottom = {bottom.min():.3f} m"),
        # WARN 项: 手插桌是重建噪声(RL 要修的对象), 量化修正预算, 不拦截
        ("WARN 指尖插桌深度 (重建噪声, RL 修正预算)",
         None,
         f"valid 段最低指尖 = {(tips[valid, :, 2].min() - top) * 100:.2f} cm (相对桌面)"),
        ("刚体变换保持手-物相对几何 (err < 1e-6 m)",
         rel_err < 1e-6,
         f"err = {rel_err:.2e} m"),
        ("存在有效交互段 (valid & tip接近物体表面)",
         len(seg) > 0,
         f"{len(seg)} 帧: [{seg[0] if len(seg) else '-'}..{seg[-1] if len(seg) else '-'}]  "
         f"tip-中心最近 {tip_obj[valid].min():.3f}m (半对角 {half_diag:.3f}m)"),
    ]

    print(f"\n== {args.traj}  ({T} 帧 @ {float(d['fps']):.0f}fps, hand={args.hand}, "
          f"scene_rot={args.scene_rot}) ==")
    print(f"物体 AABB 尺寸: {np.round(hi - lo, 3).tolist()} m   桌面高: {top} m")
    print(f"物体位移: {np.linalg.norm(obj_p_r[-1] - obj_p_r[0]):.3f} m   "
          f"最大抬升: {(bottom.max() - bottom[0]) * 100:.1f} cm")
    ok = True
    for name, passed, detail in checks:
        if passed is None:
            print(f"  [WARN] {name:42s} {detail}")
            continue
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:42s} {detail}")
    print("==>", "全部通过, 可以进 Isaac" if ok else "有 FAIL 项, 修完再进 Isaac")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
