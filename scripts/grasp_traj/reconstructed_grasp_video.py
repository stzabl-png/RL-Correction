#!/usr/bin/env python3
"""One command: a reconstructed take -> affordance-seeded grasp -> full
approach/grasp/lift physics video.

Given a Reconstruct_and_Retarget-style reconstruction (object mesh) + retarget (replay_world.npz,
with the reconstructed MANO hand + object trajectory; ref_qpos.npz optional),
this runs the whole pipeline end to end:

  1. build an OCIR sequence dir (decimate the object mesh)
  2. affordance-seeded grasp synthesis  (predicts the expected grasp area,
     seeds only there; picks the best force-closure record)
  3. Stage-A trajectory: object placed at its stable pose on the table, the hand
     starting from the RECONSTRUCTED initial pose, cuRobo driving it to the
     grasp, then a +Z lift
  4. Stage-B PhysX simulation -> video.mp4 (via an auto-generated identity
     DexYCB manifest, since the trajectory is authored in world frame)

Run it through the grasp-synthesis conda wrapper:

  scripts/run_grasp_synthesis_conda.sh scripts/grasp_traj/reconstructed_grasp_video.py \
    --recon-dir    <ReconstructOutput>/.../<take> \
    --retarget-dir <RetargetOutput>/.../<take> \
    --sequence-id  <name>
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def sh(cmd, env):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def table_up_in_object(replay_path: Path):
    """World 'up' (+Z) expressed in the object frame, from the reconstructed object
    orientation at the resting (t_init) frame. Used to keep grasps above the table."""
    import numpy as np
    d = np.load(str(replay_path))
    valid = np.asarray(d["valid_right"]).astype(bool)
    # longest contiguous valid run start
    best = bs = cur = s = 0
    for i, v in enumerate(valid):
        if v:
            if cur == 0:
                s = i
            cur += 1
            if cur > best:
                best, bs = cur, s
        else:
            cur = 0
    q = np.asarray(d["obj_pose"][bs, 3:7], dtype=float)
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    up = R.T @ np.array([0.0, 0.0, 1.0])
    return [float(v) for v in up]


def _quat_wxyz_to_R(q):
    import numpy as np
    w, x, y, z = np.asarray(q, dtype=float) / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def approach_dir_in_object(replay_path: Path, annotation_path: Path,
                           *, target_stroke: float = 0.05, max_back: int = 30,
                           min_stroke: float = 0.03):
    """Human hand approach direction in the OBJECT frame, from the pre-contact
    wrist PATH. Returns ``(unit_dir_list, side)`` or ``(None, side)``.

    Reads the contact-onset frame ``t_c`` (start of the first annotated contact
    range for whichever hand is annotated) from ``grasp_annotation.json``, then
    takes the net displacement of that hand's WRIST over the 'approach stroke'
    just before ``t_c``: walk backward from ``t_c`` over valid, confident frames
    until the wrist has moved >= ``target_stroke`` (or ``max_back`` frames),
    then end-minus-start, rotated into the object frame by the object pose at
    ``t_c``. Only the wrist POSITION path is used -- never the unreliable
    reconstructed hand orientation. If the hand barely moved before contact
    (stroke < ``min_stroke``) the direction is ill-defined and ``None`` is
    returned so the caller leaves the grasp area un-narrowed."""
    import numpy as np
    ann = json.loads(Path(annotation_path).read_text())
    a = ann.get("annotations", {})
    side = "right" if a.get("right") else ("left" if a.get("left") else None)
    if side is None:
        return None, None
    t_c = int(a[side][0][0])

    d = np.load(str(replay_path))
    frames = np.asarray(d["frames"]).astype(int)
    where = np.flatnonzero(frames == t_c)
    if where.size == 0:
        return None, side
    ix = int(where[0])
    wrist = np.asarray(d[f"joints_{side}"], dtype=float)[:, 0, :]        # world wrist path
    valid = np.asarray(d[f"valid_{side}"]).astype(float) > 0.5
    conf = (np.asarray(d[f"confidence_{side}"], dtype=float)
            if f"confidence_{side}" in d.files else np.ones(len(frames)))
    obj = np.asarray(d["obj_pose"], dtype=float)

    R = _quat_wxyz_to_R(obj[ix, 3:7])
    p = obj[ix, :3]
    wobj = (wrist - p) @ R                                               # wrist path in object frame

    ts = ix
    for t in range(ix - 1, max(0, ix - max_back) - 1, -1):
        if not (valid[t] and conf[t] >= 0.5):
            continue
        ts = t
        if np.linalg.norm(wobj[ix] - wobj[ts]) >= target_stroke:
            break
    disp = wobj[ix] - wobj[ts]
    stroke = float(np.linalg.norm(disp))
    if stroke < min_stroke:
        return None, side
    return [float(v) for v in (disp / stroke)], side


def decimate(mesh_in: Path, mesh_out: Path, target_faces: int = 40000):
    import trimesh
    m = trimesh.load(str(mesh_in), force="mesh", process=False)
    if len(m.faces) > target_faces:
        import fast_simplification
        v, f = fast_simplification.simplify(m.vertices, m.faces, target_reduction=1 - target_faces / len(m.faces))
        m = trimesh.Trimesh(vertices=v, faces=f)
        m.remove_unreferenced_vertices()
    mesh_out.parent.mkdir(parents=True, exist_ok=True)
    m.export(str(mesh_out))
    print(f"[pipeline] sequence mesh: {len(m.faces)} faces -> {mesh_out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon-dir", required=True, help="ReconstructOutput/.../<take> (has object_mesh_scaled_final.obj)")
    ap.add_argument("--retarget-dir", required=True, help="RetargetOutput/.../<take> (has replay_world.npz [+ ref_qpos.npz])")
    ap.add_argument("--sequence-id", required=True)
    ap.add_argument("--work-root", default=str(REPO / "data" / "testing"))
    ap.add_argument("--mesh-name", default="object_mesh_scaled_final.obj")
    ap.add_argument("--affordance-threshold", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--opt-iters", type=int, default=500)
    ap.add_argument("--carry-lift-height", type=float, default=0.10)
    ap.add_argument("--squeeze-overclose", type=float, default=0.0,
                    help="Radians the flexion fingers grip past the grasp pose (reduces slip); ~0.12 = firm.")
    ap.add_argument("--object-mass", type=float, default=0.2)
    ap.add_argument("--table-z", type=float, default=0.85)
    ap.add_argument("--reuse-synthesis", action="store_true", help="Skip synthesis if grasp records already exist.")
    ap.add_argument("--anchored", action="store_true",
                    help="Anchor grasp orientation to the reconstructed human hand (fixes thumb-down): "
                         "builds human_demo.npz from replay_world.npz and runs anchored_bodex instead of "
                         "the region-only affordance seeding.")
    ap.add_argument("--pose-weight", type=float, default=1.0, help="Anchored: annealed human-pose prior weight.")
    ap.add_argument("--table-penalty-weight", type=float, default=500.0,
                    help="Anchored: hand-table collision penalty weight (0 disables).")
    ap.add_argument("--cone-halfangle-deg", type=float, default=80.0,
                    help="Affordance-seeded: approach-cone half-angle (deg) narrowing the grasp area to "
                         "the side the human hand approached from (via grasp_annotation.json). Larger = looser.")
    ap.add_argument("--no-approach-cone", action="store_true",
                    help="Disable approach-direction narrowing (use the full affordance region).")
    ap.add_argument("--opt-table-penalty-weight", type=float, default=0.0,
                    help="Affordance-seeded opt-①: hand-table collision penalty weight added to the optimizer "
                         "(0 disables; e.g. 500). Keeps the hand from diving through the table during optimization.")
    ap.add_argument("--fingertip-contacts", action="store_true",
                    help="Affordance-seeded opt-②: restrict the optimizer to fingertip contacts (pinch grasp).")
    args = ap.parse_args()

    recon = Path(args.recon_dir)
    ret = Path(args.retarget_dir)
    mesh = recon / args.mesh_name
    replay = ret / "replay_world.npz"
    refq = ret / "ref_qpos.npz"
    for p, what in [(mesh, "object mesh"), (replay, "replay_world.npz")]:
        if not p.exists():
            print(f"error: {what} not found at {p}", file=sys.stderr)
            return 2

    work = Path(args.work_root)
    seq_dir = work / "sequences" / args.sequence_id
    synth_dir = work / "grasp_synthesis" / args.sequence_id
    traj_dir = work / "grasp_traj" / args.sequence_id

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # 1. sequence dir
    decimate(mesh, seq_dir / "object.obj")

    # 2. synthesis: anchored (human-pose-anchored, fixes thumb-down) or affordance-seeded (region only)
    def glob_records():
        return sorted(synth_dir.glob("grasp_*.json")) + sorted(synth_dir.glob("failed_grasp_*.json"))

    up_obj = table_up_in_object(replay)  # table 'up' in object frame (keeps grasps above the table)

    # approach direction (object frame) from the pre-contact wrist path, used to
    # narrow the affordance grasp area to the side the human approached from.
    approach_dir = None
    ann_path = recon / "grasp_annotation.json"
    if not args.no_approach_cone and ann_path.exists():
        approach_dir, approach_side = approach_dir_in_object(replay, ann_path)
        if approach_dir is not None:
            print(f"[pipeline] approach-cone: side={approach_side} "
                  f"dir_object={[round(v, 3) for v in approach_dir]} "
                  f"halfangle={args.cone_halfangle_deg:.0f}deg", flush=True)
        else:
            print("[pipeline] approach-cone: no reliable approach direction "
                  "(hand static before contact or annotation unusable); grasp area not narrowed.", flush=True)
    elif not args.no_approach_cone:
        print(f"[pipeline] approach-cone: no grasp_annotation.json at {ann_path}; grasp area not narrowed.",
              flush=True)

    def run_affordance():
        cmd = [sys.executable, REPO / "scripts/grasp_synthesis/synthesize_affordance_seeded.py",
               "--sequence-dir", seq_dir, "--out-dir", synth_dir,
               "--affordance-threshold", args.affordance_threshold,
               "--seeds", args.seeds, "--top-k", args.top_k, "--opt-iters", args.opt_iters,
               "--table-up", *up_obj]
        if approach_dir is not None:
            cmd += ["--approach-dir", *approach_dir, "--cone-halfangle-deg", args.cone_halfangle_deg]
        if args.opt_table_penalty_weight > 0.0:
            cmd += ["--table-penalty-weight", args.opt_table_penalty_weight]
        if args.fingertip_contacts:
            cmd += ["--fingertip-contacts"]
        sh(cmd, env)

    def run_anchored():
        sh([sys.executable, REPO / "scripts/grasp_synthesis/export_replay_human_demo.py",
            "--replay-npz", replay, "--out", seq_dir / "human_demo.npz"], env)
        cmd = [sys.executable, REPO / "scripts/grasp_synthesis/synthesize_sharpa_anchored_bodex.py",
               "--sequence-dir", seq_dir, "--out-dir", synth_dir,
               "--seeds", args.seeds, "--top-k", args.top_k, "--opt-iters", args.opt_iters,
               "--pose-weight", args.pose_weight, "--table-penalty-weight", args.table_penalty_weight,
               "--no-isaac-visualize"]
        if approach_dir is not None:
            cmd += ["--approach-dir", *approach_dir, "--cone-halfangle-deg", args.cone_halfangle_deg]
        sh(cmd, env)

    records = glob_records()
    if args.reuse_synthesis and records:
        print(f"[pipeline] reusing existing synthesis ({len(records)} records)", flush=True)
    elif args.anchored:
        # anchored needs the reconstructed hand to actually contact the object; a big
        # reconstruction gap (object drift) leaves 0 contact frames -> no seeds. Fall back
        # to region-only affordance seeding so we still generate a GraspPose.
        try:
            run_anchored()
            if not glob_records():
                raise RuntimeError("anchored produced no records")
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"[pipeline] anchored synthesis unusable ({exc}); likely no hand-object contact "
                  f"(reconstruction gap). Falling back to affordance-seeded.", flush=True)
            run_affordance()
        records = glob_records()
    else:
        run_affordance()
        records = glob_records()

    if not records:
        print("error: synthesis produced no grasp records", file=sys.stderr)
        return 3

    # 3. Stage-A trajectory. The generator auto-selects the best force-closure grasp that
    #    CLEARS the table (given the object's on-table placement), so the hand never reaches
    #    up from below the tabletop -- across BOTH synthesis paths.
    cmd = [sys.executable, REPO / "scripts/grasp_traj/generate_reconstructed_grasp_traj.py",
           "--sequence-dir", seq_dir, "--synthesis-dir", synth_dir,
           "--replay-npz", replay, "--recon-mesh", mesh,
           "--out-dir", traj_dir, "--carry-lift-height", args.carry_lift_height, "--table-z", args.table_z,
           "--squeeze-overclose", args.squeeze_overclose]
    if refq.exists():
        cmd += ["--refqpos-npz", refq]
    sh(cmd, env)

    # 4. identity manifest (auto) + Stage-B video
    from ocir.isaac.identity_manifest import ensure_identity_manifest
    manifest = ensure_identity_manifest(work / "identity_manifest", args.sequence_id)

    benv = env.copy()
    benv["OMNI_KIT_ACCEPT_EULA"] = "YES"
    sh([sys.executable, REPO / "scripts/isaac/simulate_grasp_traj.py", "--mode", "local", "--headless",
        "--trajectory-dir", traj_dir, "--out-dir", traj_dir / "isaac_sim",
        "--sequence-id", args.sequence_id, "--manifest", manifest,
        "--tabletop-z", args.table_z, "--object-mass", args.object_mass], benv)

    video = traj_dir / "isaac_sim" / "video.mp4"
    report = traj_dir / "isaac_sim" / "report.json"
    print("\n[pipeline] DONE")
    print(f"[pipeline] video  -> {video}")
    if report.exists():
        m = json.loads(report.read_text()).get("metrics", {})
        print(f"[pipeline] metrics: grasp_success={m.get('grasp_success')} lifted={m.get('lifted')} "
              f"max_lift_m={m.get('max_lift_m')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
