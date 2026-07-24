"""把四类源数据 (Reconstruction / Retargeting / GraspPose / cuRoboTrajectory + OCIR序列)
按 数据集/物体 复制进 RL_Correction/TrainingData, 使训练项目自包含.

  python -m rl_rebuild.correction.stage_training_data egodex pp0
  python -m rl_rebuild.correction.stage_training_data egodex pp0 --dry-run
  python -m rl_rebuild.correction.stage_training_data --all

管线: RawData -> Reconstruction -> Retarget -> OCIR(grasp+traj) -> RL.
四阶段共享同一物体 mesh (reconstruct 原始 mesh, OCIR 内部降采样, 几何一致).
新物体: 在 REGISTRY 加一条 (dataset, obj) -> 五个源目录 + mass/friction.
"""
from __future__ import annotations

import argparse
import json
import os

from rl_rebuild.correction import paths
import shutil

_BIV2AP = paths.RR_OUTPUT
_OCIR = "/home/lyh/Project/ocir-grasp-synthesis/data/testing"
_TD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../TrainingData")
_TD = os.path.abspath(_TD)


def _egodex(clip: int, ocir: str) -> dict:
    """egodex/part2/basic_pick_place/<clip> + OCIR egodex_pp<clip> 的标准五源."""
    rec = f"{_BIV2AP}/ReconstructOutput/egodex/part2/basic_pick_place/{clip}"
    ret = f"{_BIV2AP}/RetargetOutput/egodex/part2/basic_pick_place/{clip}"
    return dict(
        reconstruction=rec,
        retarget=ret,
        grasp_pose=f"{_OCIR}/grasp_synthesis/{ocir}",
        curobo_traj=f"{_OCIR}/grasp_traj/{ocir}",
        ocir_sequence=f"{_OCIR}/sequences/{ocir}",
        src_clip=f"egodex/part2/basic_pick_place/{clip}",
        src_ocir=ocir,
        # 摩擦=验证器 SuperGrip 真值 (isaac_sim/report.json: 3.0/multiply); mass 同 CLI
        mass_kg=0.2, friction=3.0, friction_combine_mode="multiply",
    )


# (dataset, object) -> 源目录集. 新物体加一条即可.
REGISTRY = {
    ("egodex", "pp0"): _egodex(0, "egodex_pp0"),
    ("egodex", "pp11"): _egodex(11, "egodex_pp11"),
    ("egodex", "pp55"): _egodex(55, "egodex_pp55"),
}

# 每类只搬这些文件 (None=整目录递归复制)
PICK = {
    "reconstruction": ["object_mesh_scaled_final.obj", "world_fused.npz",
                       "world_summary.json", "reconstruction_complete.json"],
    "retarget": ["replay_world.npz", "object.usd", "ref_qpos.npz"],
    "grasp_pose": None,      # 全部 grasp_pose_*.json + summary.json + viz/
    "curobo_traj": None,     # trajectory.* + isaac_sim/
    "ocir_sequence": ["human_demo.npz", "object.obj", "affordance.npz",
                      "object_coacd_parts.npz"],
}


def _copy(src_dir: str, dst_dir: str, pick, dry: bool) -> list[str]:
    done = []
    if not os.path.isdir(src_dir):
        print(f"  [缺] 源目录不存在: {src_dir}")
        return done
    os.makedirs(dst_dir, exist_ok=True) if not dry else None
    names = pick if pick is not None else sorted(os.listdir(src_dir))
    for n in names:
        s = os.path.join(src_dir, n)
        if not os.path.exists(s):
            if pick is not None:
                print(f"  [缺] {n}  (源无此文件, 跳过)")
            continue
        d = os.path.join(dst_dir, n)
        print(f"  {'[dry] ' if dry else ''}{n}"
              f"{'/' if os.path.isdir(s) else ''}")
        if not dry:
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        done.append(n)
    return done


def stage(dataset: str, obj: str, dry: bool = False):
    key = (dataset, obj)
    if key not in REGISTRY:
        raise KeyError(f"未注册物体 {key}; 已注册: {list(REGISTRY)}")
    e = REGISTRY[key]
    base = os.path.join(_TD, dataset, obj)
    print(f"\n=== stage {dataset}/{obj} -> {base} ===")
    manifest = {}
    for kind in ["reconstruction", "retarget", "grasp_pose", "curobo_traj", "ocir_sequence"]:
        print(f"[{kind}]  <- {e[kind]}")
        manifest[kind] = dict(source=e[kind],
                              files=_copy(e[kind], os.path.join(base, kind), PICK[kind], dry))
    meta = dict(
        dataset=dataset, object=obj,
        src_clip=e["src_clip"], src_ocir=e["src_ocir"],
        mass_kg=e["mass_kg"], friction=e["friction"],
        friction_combine_mode=e["friction_combine_mode"],
        note="mesh: reconstruct原始 == OCIR降采样(同AABB/同物体). "
             "grasp_pose json 内 object_mesh 指向历史 _anchored 序列(已并入, 几何一致).",
        manifest=manifest,
    )
    if not dry:
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  -> meta.json 写入 {base}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("object", nargs="?")
    ap.add_argument("--all", action="store_true", help="归置注册表里全部物体")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.all:
        for ds, ob in REGISTRY:
            stage(ds, ob, a.dry_run)
    elif a.dataset and a.object:
        stage(a.dataset, a.object, a.dry_run)
    else:
        ap.error("给出 <dataset> <object> 或 --all")
