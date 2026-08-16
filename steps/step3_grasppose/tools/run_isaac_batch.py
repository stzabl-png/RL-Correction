"""Submit all exported trajectory dirs to the running ocir Isaac server, poll each,
collect report.json metrics, and write a summary.

  python tools/run_isaac_batch.py --traj-root output/pp2_sharpa_wave/isaac_traj \
      [--server http://127.0.0.1:8765] [--object-mass 0.1] [--tabletop-z 0.85]
"""

import argparse
import glob
import json
import os
import time
import urllib.request

# ⚠ Isaac PhysX 验证栈(isaac/)**尚未随 Step3 迁入本仓**, 仍在旧检出里, 下一轮迁。
#   在那之前这个脚本只在有旧检出的机器上能跑。
MANIFEST = os.environ.get(
    "DEXO_ISAAC_MANIFEST",
    "/home/lyh/Project/Dexonomy/isaac/data/testing/identity_manifest/manifest.json")


def http_json(url, payload=None, timeout=30, retries=5):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-root", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--object-mass", type=float, default=0.1)
    ap.add_argument("--tabletop-z", type=float, default=0.85)
    ap.add_argument("--sequence-id", default="egodex_grasp0")
    ap.add_argument("--job-timeout", type=float, default=300)
    ap.add_argument("--asset-config", default=None,
                    help="手资产 yml (左手传 .../sharpa_wave_left.yml). server 的 "
                         "normalize_task_args 是通用透传, 任何与 CLI 同名的键都会生效")
    ap.add_argument("--hand-usd", default=None, help="直接覆盖手 USD")
    ap.add_argument("--eval-mode", default="lift", choices=("lift", "contact"),
                    help="contact: 只评接触点质量(拇指对指、不推翻物体), 不做提起测试; "
                         "配 export_isaac_traj --no-carry --squeeze-frames 60 使用")
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(f"{args.traj_root}/*/")
                  if os.path.exists(os.path.join(d, "trajectory.npz")))
    print(f"{len(dirs)} trajectories")
    results = {}
    for i, d in enumerate(dirs):
        name = os.path.basename(d.rstrip("/"))
        out_dir = os.path.join(d, "isaac_sim")
        rep_path = os.path.join(out_dir, "report.json")
        if os.path.exists(rep_path):
            rep = json.load(open(rep_path))
            results[name] = rep.get("metrics", rep)
            print(f"[{i+1}/{len(dirs)}] {name}: cached")
            continue
        sub = http_json(f"{args.server}/run", {
            "task": "grasp_traj_simulation",
            "params": {
                "trajectory_dir": os.path.abspath(d),
                "out_dir": os.path.abspath(out_dir),
                "sequence_id": args.sequence_id,
                "manifest": MANIFEST,
                "tabletop_z": args.tabletop_z,
                "object_mass": args.object_mass,
                **({"asset_config": os.path.abspath(args.asset_config)}
                   if args.asset_config else {}),
                **({"hand_usd": os.path.abspath(args.hand_usd)} if args.hand_usd else {}),
                **({"eval_mode": "contact", "contact_aware_finger_targets": True}
                   if args.eval_mode == "contact" else {}),
            },
            "wait": False,
        })
        job = sub["job_id"]
        t0 = time.time()
        while time.time() - t0 < args.job_timeout:
            st = http_json(f"{args.server}/jobs/{job}")
            if st.get("done"):
                break
            time.sleep(3)
        metrics = None
        if os.path.exists(rep_path):
            rep = json.load(open(rep_path))
            metrics = rep.get("metrics", rep)
        results[name] = metrics or {"error": st.get("response", {}).get("error", "no report")}
        m = results[name]
        if args.eval_mode == "contact":
            print(f"[{i+1}/{len(dirs)}] {name}: success={m.get('grasp_success')} "
                  f"fingers={m.get('contact_fingers_sustained')} "
                  f"drift={m.get('object_drift_m')} tilt={m.get('object_tilt_deg')}")
        else:
            print(f"[{i+1}/{len(dirs)}] {name}: success={m.get('grasp_success')} "
                  f"lift={m.get('max_lift_m')} dropped={m.get('object_dropped')}")

    summary_path = os.path.join(args.traj_root, "isaac_summary.json")
    succ = sorted(n for n, m in results.items() if m.get("grasp_success"))
    with open(summary_path, "w") as f:
        json.dump({"n_total": len(results), "n_success": len(succ),
                   "success": succ, "results": results}, f, indent=1)
    print(f"\n==== {len(succ)}/{len(results)} grasp_success ====")
    for n in succ:
        print("  ", n)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
