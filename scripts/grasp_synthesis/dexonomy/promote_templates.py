"""Promote Isaac-validated grasps into the Dexonomy template library.

  python tools/promote_templates.py --exp-dir output/pp2_sharpa_wave \
      --hand sharpa_wave [--prefix isaac] [--names 45_7_grasp ...]

Reads <exp-dir>/isaac_traj/isaac_summary.json (grasp_success list) unless --names
is given, finds the source grasp npy for each, resets n_evo to 0 (fresh evolution
headroom) and writes assets/hand/<hand>/init_tmpl/<prefix>_<name>.npy.
Afterwards these can be used directly:  dexsyn hand=<hand> 'tmpl_name=[isaac_45_7]' ...
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--hand", default="sharpa_wave")
    ap.add_argument("--prefix", default="isaac")
    ap.add_argument("--names", nargs="*", default=None)
    args = ap.parse_args()

    names = args.names
    if not names:
        summary = json.load(open(f"{args.exp_dir}/isaac_traj/isaac_summary.json"))
        names = summary["success"]
    tmpl_dir = f"assets/hand/{args.hand}/init_tmpl"
    os.makedirs(tmpl_dir, exist_ok=True)

    for name in names:
        srcs = glob.glob(f"{args.exp_dir}/grasp_data/**/{name}.npy", recursive=True)
        assert srcs, f"source grasp npy not found for {name}"
        d = np.load(srcs[0], allow_pickle=True).item()
        d["n_evo"] = np.array([0])
        tmpl_name = f"{args.prefix}_{name.replace('_grasp', '')}"
        d["tmpl_name"] = tmpl_name
        out = f"{tmpl_dir}/{tmpl_name}.npy"
        np.save(out, d)
        print(f"promoted {srcs[0]} -> {out}")
    print(f"\nusable via: dexrun op=init hand={args.hand} tmpl_name=<name> ...")


if __name__ == "__main__":
    main()
