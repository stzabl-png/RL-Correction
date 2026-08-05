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

MANIFEST = "/home/lyh/Project/ocir-grasp-synthesis/data/testing/identity_manifest/manifest.json"


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
