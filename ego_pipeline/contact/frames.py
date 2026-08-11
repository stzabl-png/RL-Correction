"""Load one take and put every input into ONE coordinate frame.

Three frames are in play:
  recon world   `world_fused.npz` gravity_z_up_world -- object mesh + pose + camera live here
  table  scene  `ref_qpos.npz`    -- RL_Correction/rl_rebuild/correction/export_qpos.py runs
                                     frames.align_replay() (scene_rot + drop-on-table recenter)
                                     before retargeting, so the SharpaWave wrist is shifted
  camera        per-frame, OpenCV convention (+z forward), K for the full-res video

Rather than re-deriving the table transform from export_qpos's arguments (table_height,
obj_gap, scene_rot -- which we would have to guess), we RECOVER it from the data: the
wrist point exists in both frames, so a rigid (scale-1) Umeyama fit of
`replay_world.joints_<side>[:,0]` -> `ref_qpos.wrist_pos` gives T_scene<-world exactly.
Its residual doubles as a self-check: it must be ~0, otherwise the two files are not
from the same take/run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIDES = ("left", "right")


# ---------------------------------------------------------------- rigid fits
def umeyama_rigid(A: np.ndarray, B: np.ndarray) -> tuple:
    """Least-squares rigid transform with scale fixed at 1: B ~= R @ A + t."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def invert_rt(R: np.ndarray, t: np.ndarray) -> tuple:
    return R.T, -R.T @ t


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------- take data
@dataclass
class Take:
    recon_dir: Path
    retarget_dir: Path
    side: str
    Tv: int
    # object
    object_id: str                   # which object of the take (multi-object: object_N)
    obj_mesh_path: Path
    obj_T_world: np.ndarray          # (Tv,4,4) object-local -> recon world
    obj_confidence: np.ndarray       # (Tv,)
    obj_valid: np.ndarray            # (Tv,) bool
    # camera (recon world)
    c2w: np.ndarray                  # (Tv,4,4)
    K: np.ndarray                    # (3,3)
    # SharpaWave hand, ALREADY mapped into recon world
    finger_qpos: np.ndarray | None   # (Tv,22)  None when ref_qpos.npz is absent
    joint_names: list
    wrist_pos: np.ndarray | None     # (Tv,3)
    wrist_quat_wxyz: np.ndarray | None  # (Tv,4)
    hand_valid: np.ndarray           # (Tv,) bool
    # MANO reference (recon world), for cross-checks
    mano_joints: np.ndarray          # (Tv,21,3)
    mano_verts: np.ndarray | None    # (Tv,778,3)
    # annotation
    intervals: list                  # [[start,end], ...] inclusive, video-frame indices
    # bookkeeping
    scene_from_world: tuple = field(default=(None, None))
    fit_residual_mm: float = 0.0
    video_path: Path | None = None

    @property
    def f0(self) -> int:
        """First annotated contact frame."""
        if not self.intervals:
            raise RuntimeError(f"no '{self.side}' interval in grasp_annotation.json")
        return int(self.intervals[0][0])

    def contact_frames(self) -> np.ndarray:
        idx = []
        for a, b in self.intervals:
            idx.extend(range(max(0, int(a)), min(self.Tv - 1, int(b)) + 1))
        return np.array(sorted(set(idx)), dtype=int)

    def qpos_dict(self, f: int) -> dict:
        return {n: float(v) for n, v in zip(self.joint_names, self.finger_qpos[f])}

    def mask_path(self, kind: str, f: int) -> Path:
        name = f"{self.side}_hand_0.png" if kind == "hand" else f"{self.object_id}.png"
        sub = "hands" if kind == "hand" else "objects"
        return self.recon_dir / "masks" / sub / "frames" / f"frame_{f:06d}_masks" / name


def load_take(recon_dir, retarget_dir=None, side="right", annotation=None,
              video=None, require_qpos=True, object_id="object_0") -> Take:
    recon_dir = Path(recon_dir).resolve()
    if retarget_dir is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from repo_paths import RECON_OUTPUT, RETARGET_OUTPUT
        retarget_dir = Path(RETARGET_OUTPUT) / recon_dir.relative_to(Path(RECON_OUTPUT).resolve())
    retarget_dir = Path(retarget_dir).resolve()

    w = np.load(recon_dir / "world_fused.npz", allow_pickle=True)
    p = np.load(retarget_dir / "replay_world.npz", allow_pickle=True)
    # A two-handed task (unscrewing a cap: one hand holds, the other twists) needs BOTH
    # hands' qpos in the same take dir, so prefer the side-specific file. The unsuffixed
    # name stays as a fallback for takes exported before this split.
    qpath = retarget_dir / f"ref_qpos_{side}.npz"
    if not qpath.exists():
        qpath = retarget_dir / "ref_qpos.npz"
    if not qpath.exists() and require_qpos:
        raise FileNotFoundError(
            f"{qpath} missing -- it holds the SharpaWave 22-DoF qpos. Generate it with\n"
            f"  {retarget_dir.parents[4]/'third_party/MagicDexMate/.venv-isaac/bin/python'} "
            f"-m contact.make_ref_qpos {retarget_dir} --hand {side} "
            f"--out {retarget_dir/f'ref_qpos_{side}.npz'}\n"
            f"(run it from ego_pipeline/; it needs the Isaac venv for dex_retargeting)\n"
            f"or pass require_qpos=False to run the observation-only checks.")
    q = np.load(qpath, allow_pickle=True) if qpath.exists() else None

    Tv = int(w["num_frames"])
    # -- which object (multi-object takes store *_all arrays; singular fields = object_0,
    #    verified in RL_Correction object_select.py on screw 4) --
    obj_ids = ([str(x) for x in np.asarray(w["object_ids"]).tolist()]
               if "object_ids" in w.files else ["object_0"])
    if object_id not in obj_ids:
        raise RuntimeError(f"object {object_id} not in take (has {obj_ids})")
    oi = obj_ids.index(object_id)
    if "object_ob_in_world_all" in w.files:
        obj_T = np.asarray(w["object_ob_in_world_all"][oi], dtype=np.float64)
    else:
        obj_T = np.asarray(w["object_ob_in_world"], dtype=np.float64)
    mesh_p = None
    cand = sorted((recon_dir / "objects" / object_id).glob("*.obj"))         if (recon_dir / "objects" / object_id).is_dir() else []
    if cand:
        mesh_p = cand[0]
    elif oi == 0:
        mesh_p = recon_dir / str(w["mesh_filename"])
    else:
        raise RuntimeError(f"no mesh for {object_id} under {recon_dir}/objects/")
    for name, arr in (("object_ob_in_world", obj_T),
                      ("joints_" + side, p[f"joints_{side}"])):
        if len(arr) != Tv:
            raise RuntimeError(f"{name} has {len(arr)} frames, expected Tv={Tv}")

    fq = jnames = wpos_w = wquat_w = None
    R_sw, t_sw = np.eye(3), np.zeros(3)
    resid = np.zeros(1)
    hand_valid = np.isfinite(p[f"joints_{side}"]).all((1, 2)) & (p[f"valid_{side}"] > 0.5)
    if q is not None:
        if len(q["finger_qpos"]) != Tv:
            raise RuntimeError(f"finger_qpos has {len(q['finger_qpos'])} frames, expected Tv={Tv}")
        if not str(q["joint_names"][0]).startswith(side):
            raise RuntimeError(f"ref_qpos.npz is for the other hand ({q['joint_names'][0]}), not {side}")

        # ---- table scene -> recon world, recovered from the shared wrist point ----
        wrist_world = np.asarray(p[f"joints_{side}"][:, 0], dtype=np.float64)
        wrist_scene = np.asarray(q["wrist_pos"], dtype=np.float64)
        valid = np.asarray(q["valid"], dtype=bool) & np.isfinite(wrist_world).all(1)
        if valid.sum() < 3:
            raise RuntimeError(f"only {valid.sum()} valid frames for the {side} hand -- cannot fit frames")
        R_sw, t_sw = umeyama_rigid(wrist_world[valid], wrist_scene[valid])
        resid = np.linalg.norm(wrist_world[valid] @ R_sw.T + t_sw - wrist_scene[valid], axis=1)
        R_ws, t_ws = invert_rt(R_sw, t_sw)

        from .urdf_fk import quat_wxyz_to_matrix
        wpos_w = np.asarray(q["wrist_pos"], dtype=np.float64) @ R_ws.T + t_ws
        wquat_w = np.stack([matrix_to_quat_wxyz(R_ws @ quat_wxyz_to_matrix(qq))
                            for qq in np.asarray(q["wrist_quat_wxyz"], dtype=np.float64)])
        fq = np.asarray(q["finger_qpos"], dtype=np.float64)
        jnames = [str(n) for n in q["joint_names"]]
        hand_valid = np.asarray(q["valid"], dtype=bool)

    # ---- annotation ----
    ann_path = Path(annotation) if annotation else recon_dir / "grasp_annotation.json"
    intervals = []
    if ann_path.exists():
        intervals = json.load(open(ann_path))["annotations"].get(side, [])

    # older recon schemas (HOI4D takes) lack the per-frame confidence/validity fields
    obj_valid = np.ones(Tv, dtype=bool)
    if "object_pose_valid_by_frame" in w.files:
        pv = np.asarray(w["object_pose_valid_by_frame"])
        obj_valid = np.asarray(pv[oi] if pv.ndim > 1 and len(pv) > oi else pv[0], dtype=bool)
    obj_conf = np.asarray(w["object_confidence"], dtype=np.float64) \
        if "object_confidence" in w.files else np.ones(Tv)

    return Take(
        recon_dir=recon_dir, retarget_dir=retarget_dir, side=side, Tv=Tv,
        object_id=object_id,
        obj_mesh_path=mesh_p,
        obj_T_world=obj_T,
        obj_confidence=obj_conf,
        obj_valid=obj_valid,
        c2w=np.asarray(w["c2w"], dtype=np.float64),
        K=np.asarray(w["K"], dtype=np.float64),
        finger_qpos=fq, joint_names=jnames or [],
        wrist_pos=wpos_w, wrist_quat_wxyz=wquat_w,
        hand_valid=hand_valid,
        mano_joints=np.asarray(p[f"joints_{side}"], dtype=np.float64),
        mano_verts=np.asarray(p[f"mano_verts_{side}"], dtype=np.float64)
        if f"mano_verts_{side}" in p.files else None,
        intervals=[[int(a), int(b)] for a, b in intervals],
        scene_from_world=(R_sw, t_sw),
        fit_residual_mm=float(np.max(resid) * 1000),
        video_path=Path(video) if video else None,
    )


# ---------------------------------------------------------------- sanity check
def object_projection_iou(take: Take, f: int, mesh=None) -> dict:
    """Project the object mesh at frame f and compare with the SAM2 object mask.

    This validates that {mesh, object_ob_in_world, c2w, K, OpenCV convention} are all
    mutually consistent. Everything downstream (2D contact region, alignment) is
    meaningless if this fails, so it runs before anything else.
    """
    import cv2
    from .observe2d import project_points

    if mesh is None:
        import trimesh
        mesh = trimesh.load(take.obj_mesh_path, force="mesh", process=False)
    V = np.asarray(mesh.vertices)
    Vw = V @ take.obj_T_world[f][:3, :3].T + take.obj_T_world[f][:3, 3]

    mask = cv2.imread(str(take.mask_path("object", f)), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(take.mask_path("object", f))
    H, W = mask.shape[:2]
    uv, z, inb = project_points(Vw, take.c2w[f], take.K, (H, W))

    proj = np.zeros((H, W), np.uint8)
    ok = inb & (z > 0)
    proj[uv[ok, 1], uv[ok, 0]] = 1
    proj = cv2.dilate(proj, np.ones((5, 5), np.uint8))          # vertices are points, mask is filled
    proj = cv2.morphologyEx(proj, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = (mask > 0).astype(np.uint8)

    # Score only where the object is ACTUALLY OBSERVABLE. SAM2's mask already has the
    # hand-covered part cut out of it, so comparing against the full projected mesh
    # punishes exactly the frames we want -- the ones deep in the grasp, where occlusion
    # is high. Excluding the hand region measures the object pose, not the occlusion.
    hand = cv2.imread(str(take.mask_path("hand", f)), cv2.IMREAD_UNCHANGED)
    seen = np.ones((H, W), bool) if hand is None else ~(hand > 0)
    pv, mv = proj.astype(bool) & seen, m.astype(bool) & seen
    inter_v, union_v = int((pv & mv).sum()), int((pv | mv).sum())
    inter, union = int((proj & m).sum()), int((proj | m).sum())
    return dict(frame=int(f), iou=inter_v / max(union_v, 1),
                iou_including_occluded=inter / max(union, 1),
                n_projected=int(ok.sum()), mask_px=int(m.sum()), proj_px=int(proj.sum()),
                mean_depth_m=float(np.median(z[ok])) if ok.any() else float("nan"))


def hand_reprojection_health(take: Take, frames=None) -> dict:
    """Do the RECONSTRUCTED hand joints land on the hand the video actually shows?

    A hand whose depth is merely wrong still reprojects onto the observed hand (moving
    along a viewing ray does not change the pixel). So a low score here means the hand
    is displaced ACROSS the ray too -- i.e. the reconstruction is off in a way that no
    single global fix can absorb, and per-frame alignment is the only option.
    """
    import cv2
    from .observe2d import project_points

    frames = range(take.Tv) if frames is None else frames
    fracs, used = [], []
    for f in frames:
        mask = cv2.imread(str(take.mask_path("hand", f)), cv2.IMREAD_UNCHANGED)
        if mask is None or not take.hand_valid[f]:
            continue
        hw = mask.shape[:2]
        uv, z, inb = project_points(take.mano_joints[f], take.c2w[f], take.K, hw)
        ok = inb & (z > 0)
        if not ok.any():
            continue
        fracs.append(float((mask[uv[ok, 1], uv[ok, 0]] > 0).mean()))
        used.append(int(f))
    if not fracs:
        return dict(mean=float("nan"), median=float("nan"), n_frames=0, good_frames=0)
    a = np.array(fracs)
    return dict(mean=float(a.mean()), median=float(np.median(a)), n_frames=len(a),
                good_frames=int((a > 0.5).sum()), per_frame=dict(zip(used, np.round(a, 3).tolist())))


def egodex_gt_wrist(hdf5_path, side: str, f: int) -> dict | None:
    """EgoDex Vision Pro ground truth, camera-frame wrist at frame f.

    OFFLINE VALIDATION ONLY (USAGE.md section 7): never feed this into the pipeline --
    the whole point of the pipeline is a NOISY reconstruction. It is loaded here purely
    to tell "the reconstruction is wrong" apart from "my maths is wrong".
    """
    try:
        import h5py
    except ImportError:
        return None
    p = Path(hdf5_path)
    if not p.exists():
        return None
    with h5py.File(p, "r") as h:
        key = f"transforms/{side}Hand"
        if key not in h or "transforms/camera" not in h:
            return None
        cam = h["transforms/camera"][f]
        hand = h[key][f]
        K = h["camera/intrinsic"][:]
        conf = float(h[f"confidences/{side}Hand"][f]) if f"confidences/{side}Hand" in h else float("nan")
    Pc = (np.linalg.inv(cam) @ hand)[:3, 3]
    if Pc[2] <= 0:
        return None
    return dict(cam_xyz=Pc, depth_m=float(Pc[2]), confidence=conf,
                px=np.array([K[0, 0] * Pc[0] / Pc[2] + K[0, 2],
                             K[1, 1] * Pc[1] / Pc[2] + K[1, 2]]))
