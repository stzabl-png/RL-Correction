#!/usr/bin/env bash
# One-shot pipeline: object mesh -> Dexonomy synthesis -> Isaac physics validation.
#
#   tools/grasp_pipeline.sh <mesh.obj> [oid]
#
#   oid defaults to "obj_<parent dir name>" of the mesh path.
#   Tunables via env vars (defaults in parens):
#     TMPL (fingertip_mid)  template name, e.g. isaac_45_7 / 3_Medium_Wrap
#     EPOCH (50)            init sampling epochs (~10 candidates each)
#     PLANE_THRE (0.012)    init hand-skeleton min table clearance [m]
#     PLANE_MARGIN (0.008)  grasp-stage table repulsion cushion [m]
#     MASS (0.1)            object mass in Isaac [kg]
#     UP ("")               override resting-up dir, e.g. "0 1 0" (default: auto stable pose)
#     KEEP_SERVER (0)       1 = leave the Isaac server running after the run (faster next run);
#                           0 = shut it down at the end to free ~4GB GPU memory
#
# Results: output/<oid>_sharpa_wave/isaac_traj/isaac_summary.json (+ per-grasp video.mp4)

set -euo pipefail

MESH="${1:?usage: tools/grasp_pipeline.sh <mesh.obj> [oid]}"
OID="${2:-obj_$(basename "$(dirname "$MESH")")}"
TMPL="${TMPL:-fingertip_mid}"
EPOCH="${EPOCH:-50}"
PLANE_THRE="${PLANE_THRE:-0.012}"
PLANE_MARGIN="${PLANE_MARGIN:-0.008}"
MASS="${MASS:-0.1}"
UP="${UP:-}"
KEEP_SERVER="${KEEP_SERVER:-0}"

DEXO=/home/lyh/Project/Dexonomy
OCIR=/home/lyh/Project/ocir-grasp-synthesis
PY=/home/lyh/anaconda3/envs/dexonomy/bin/python
DEXRUN=/home/lyh/anaconda3/envs/dexonomy/bin/dexrun
SERVER=http://127.0.0.1:8765
export MUJOCO_GL=egl
cd "$DEXO"

[ -f "$MESH" ] || { echo "mesh not found: $MESH"; exit 1; }
EXP="output/${OID}_sharpa_wave"
echo "=== [$OID] mesh=$MESH tmpl=$TMPL epoch=$EPOCH ==="

echo "--- step 1/7 import"
if [ -n "$UP" ]; then
  # shellcheck disable=SC2086
  "$PY" tools/import_object.py --mesh "$MESH" --oid "$OID" --up $UP 2>&1 | grep -v CoACD | tail -2
else
  "$PY" tools/import_object.py --mesh "$MESH" --oid "$OID" --auto-up 2>&1 | grep -v CoACD | tail -3
fi

echo "--- step 2/7 init (epoch=$EPOCH)"
"$DEXRUN" op=init hand=sharpa_wave exp_name="$OID" tmpl_name="$TMPL" 'init_gpu=[0]' \
  "op.object.cfg_path=assets/object/custom/scene_cfg/$OID/tabletop/*.npy" \
  op.object.n_cfg=1 op.epoch="$EPOCH" op.filter.collision.plane_thre="$PLANE_THRE" 2>&1 | tail -1
N_INIT=$(find "$EXP/init_data" -name '*.npy' 2>/dev/null | wc -l)
echo "    init candidates: $N_INIT"
[ "$N_INIT" -gt 0 ] || { echo "no initializations — object may be too large/small for template $TMPL"; exit 1; }

echo "--- step 3/7 grasp refine"
"$DEXRUN" op=grasp hand=sharpa_wave exp_name="$OID" "op.grasp.plane_margin=$PLANE_MARGIN" 2>&1 | tail -1
N_GRASP=$(find "$EXP/grasp_data" -name '*.npy' 2>/dev/null | wc -l)
echo "    QP-passed grasps: $N_GRASP"
[ "$N_GRASP" -gt 0 ] || { echo "no grasps survived refinement — try EPOCH=$((EPOCH*2)) or another TMPL"; exit 1; }

echo "--- step 4/7 table clearance filter"
"$PY" tools/filter_plane_clearance.py --exp-dir "$EXP" --data grasp_data --check-pregrasp 2>&1 | tail -1
N_CLEAR=$(find "$EXP/grasp_data" -name '*.npy' 2>/dev/null | wc -l)
echo "    above-table grasps: $N_CLEAR"
[ "$N_CLEAR" -gt 0 ] || { echo "all grasps dipped below the table"; exit 1; }

echo "--- step 4.5 hand orientation filter (reject thumb-down grasps)"
"$PY" tools/filter_hand_orientation.py --exp-dir "$EXP" --data grasp_data 2>&1 | tail -1
N_NAT=$(find "$EXP/grasp_data" -name '*.npy' 2>/dev/null | wc -l)
echo "    natural-orientation grasps: $N_NAT"
[ "$N_NAT" -gt 0 ] || { echo "all grasps were thumb-down"; exit 1; }

echo "--- step 5/7 export Isaac trajectories"
"$PY" tools/export_isaac_traj.py --exp-dir "$EXP" --data grasp_data 2>&1 | tail -2

echo "--- step 6/7 Isaac server"
if ! curl -s -m 2 "$SERVER/status" >/dev/null 2>&1; then
  echo "    starting Isaac server (takes 1-2 min)..."
  ( cd "$OCIR" && export OCIR_DATA_ROOT=$PWD/data && \
    nohup scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py --mode local \
      > /tmp/isaac_server_pipeline.log 2>&1 & )
  for _ in $(seq 1 60); do
    curl -s -m 2 "$SERVER/status" >/dev/null 2>&1 && break
    sleep 5
  done
  curl -s -m 2 "$SERVER/status" >/dev/null 2>&1 || {
    echo "Isaac server failed to start; see /tmp/isaac_server_pipeline.log"; exit 1; }
fi
echo "    server ready"

echo "--- step 7/7 Isaac batch validation ($N_CLEAR trajectories, ~35s each)"
"$PY" -u tools/run_isaac_batch.py --traj-root "$EXP/isaac_traj" --object-mass "$MASS"

echo "--- step 8/8 keep only Isaac successes"
"$PY" tools/keep_isaac_success.py --exp-dir "$EXP"

if [ "$KEEP_SERVER" != "1" ]; then
  echo "--- shutting down Isaac server (KEEP_SERVER=1 to keep it)"
  ( cd "$OCIR" && scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py --shutdown-server >/dev/null 2>&1 ) || true
fi

echo ""
echo "=== DONE. deliverables: $EXP/isaac_succ/  summary: $EXP/isaac_traj/isaac_summary.json ==="
"$PY" - "$EXP" <<'EOF'
import json, sys
exp = sys.argv[1]
d = json.load(open(f"{exp}/isaac_traj/isaac_summary.json"))
print(f"Isaac 通过 {d['n_success']}/{d['n_total']}")
for n in d["success"]:
    m = d["results"][n]
    print(f"  ✅ {n}: lift={m['max_lift_m']*100:.1f}cm 持续={m['max_consecutive_lifted_steps']}帧"
          f"  视频: {exp}/isaac_traj/{n}/isaac_sim/video.mp4")
if d["n_success"] == 0:
    near = sorted((m.get('max_lift_m') or 0, n) for n, m in d['results'].items())[-3:]
    print("  无通过。最接近的候选（可看视频找原因）:")
    for lift, n in reversed(near):
        print(f"     {n}: max_lift={lift*100:.1f}cm  {exp}/isaac_traj/{n}/isaac_sim/video.mp4")
    print("  建议: EPOCH 翻倍重跑，或换 TMPL（如 isaac_45_7 / 3_Medium_Wrap / fingertip_small）")
EOF
echo "回灌模板: $PY tools/promote_templates.py --exp-dir $EXP --prefix isaac_$OID"
