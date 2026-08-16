"""Keep only Isaac-validated grasps as the final deliverable set.

  python tools/keep_isaac_success.py --exp-dir output/pp5_sharpa_wave

- Collects every grasp_success grasp into <exp-dir>/isaac_succ/:
    <name>.npy  (grasp data: 29-D grasp/pregrasp/squeeze qpos)
    <name>_video.mp4, <name>_report.json
- Deletes the isaac_traj/<name>/ dirs of FAILED grasps (large, regenerable via
  tools/export_isaac_traj.py).
- Moves failed grasp_data npys to <exp-dir>/grasp_data_isaac_failed/ (reversible).
"""

import argparse
import glob
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--keep-failed-traj", action="store_true",
                    help="do not delete failed isaac_traj dirs")
    args = ap.parse_args()
    exp = args.exp_dir.rstrip("/")

    summary = json.load(open(f"{exp}/isaac_traj/isaac_summary.json"))
    succ = set(summary["success"])
    out = f"{exp}/isaac_succ"
    os.makedirs(out, exist_ok=True)

    for name in sorted(succ):
        srcs = glob.glob(f"{exp}/grasp_data/**/{name}.npy", recursive=True)
        if srcs:
            shutil.copy(os.path.realpath(srcs[0]), f"{out}/{name}.npy")
        for fn, dst in (("video.mp4", f"{name}_video.mp4"),
                        ("report.json", f"{name}_report.json")):
            src = f"{exp}/isaac_traj/{name}/isaac_sim/{fn}"
            if os.path.exists(src):
                shutil.copy(src, f"{out}/{dst}")

    n_del, n_moved = 0, 0
    for d in glob.glob(f"{exp}/isaac_traj/*/"):
        name = os.path.basename(d.rstrip("/"))
        if name not in succ and not args.keep_failed_traj:
            shutil.rmtree(d)
            n_del += 1
    for f in glob.glob(f"{exp}/grasp_data/**/*.npy", recursive=True):
        name = os.path.basename(f).replace(".npy", "")
        if name not in succ and os.path.isfile(f):
            dst = f.replace("/grasp_data/", "/grasp_data_isaac_failed/")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(os.path.realpath(f), dst)
            if os.path.islink(f):
                os.unlink(f)
            n_moved += 1

    json.dump({"n_success": len(succ), "success": sorted(succ),
               "results": {n: summary["results"][n] for n in sorted(succ)}},
              open(f"{out}/summary.json", "w"), indent=1)
    print(f"[keep] {len(succ)} success -> {out}")
    print(f"[keep] deleted {n_del} failed isaac_traj dirs, "
          f"moved {n_moved} failed grasp npys -> grasp_data_isaac_failed/")


if __name__ == "__main__":
    main()
