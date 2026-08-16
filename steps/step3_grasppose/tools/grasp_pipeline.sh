#!/usr/bin/env bash
# One-shot pipeline: object mesh -> Dexonomy synthesis -> Isaac physics validation.
#
#   tools/grasp_pipeline.sh <mesh.obj> [oid]
#
#   oid defaults to "obj_<parent dir name>" of the mesh path.
#   Tunables via env vars (defaults in parens):
#     TMPL (fingertip_mid)  template name, e.g. isaac_45_7 / 3_Medium_Wrap
#     EPOCH (50)            init sampling epochs (~N_FINAL candidates each)
#     INIT_POINTS (1024)    per-epoch surface sample points
#     N_FINAL (10)          per-epoch keep quota
#                           ★提速配方(2026-08-14 实测, 瓶 5_Light_Tool): 默认
#                           50×1024×10 = 198s/500候选/QP 25; 改 10×4096×50 = 77s/500/QP 33
#                           —— 2.6x 且质量更好。原理: 轮次开销是瓶颈(GPU 只占 13%),
#                           加大采样池并**按同比例**放大配额(0.12%->0.15%), 每轮质量不降。
#                           ⚠ 别只放大配额: 5×8192×100(配额 1.2%) 实测 QP 掉到 9。
#     COACD_PRE (auto)      CoACD preprocess mode: auto|on|off. Meshes with degenerate
#                           (zero-area sliver) triangles make CoACD assert & core-dump in
#                           its manifold check -- bottle_cap has 60 such faces out of 3668.
#                           Use "on" to force voxel remeshing, which repairs them.
#     COACD_RES (50)        preprocess voxel resolution; thin walls / fine threads may need 100-200
#     PLANE_THRE (0.012)    init hand-skeleton min table clearance [m]
#     PLANE_MARGIN (0.008)  grasp-stage table repulsion cushion [m]
#     MASS (0.1)            object mass in Isaac [kg]
#     UP ("")               override resting-up dir, e.g. "0 1 0" (default: auto stable pose)
#     HAND (sharpa_wave)    hand name; use sharpa_wave_left for the left hand
#     SKIP_ISAAC (0)        1 = stop after step 5 (export). Dexonomy's Isaac validation has
#                           no predictive power for the RL_Correction platform (fixed-base
#                           palm-welded hand, uniform 3.0 friction) yet costs ~75% of the
#                           runtime; we consume grasp_data AND grasp_data_isaac_failed and
#                           screen with RL_Correction/tasks/pregrasp/screen_prior.py instead.
#                           See RL_Correction/docs/GRASPPOSE_SCREENING.md section 0.
#     KEEP_SERVER (0)       1 = leave the Isaac server running after the run (faster next run);
#                           0 = shut it down at the end to free ~4GB GPU memory
#
# Results: output/<oid>_sharpa_wave/isaac_traj/isaac_summary.json (+ per-grasp video.mp4)

set -euo pipefail

MESH="${1:?usage: tools/grasp_pipeline.sh <mesh.obj> [oid]}"
OID="${2:-obj_$(basename "$(dirname "$MESH")")}"
TMPL="${TMPL:-fingertip_mid}"
EPOCH="${EPOCH:-50}"
INIT_POINTS="${INIT_POINTS:-1024}"
N_FINAL="${N_FINAL:-10}"
PLANE_THRE="${PLANE_THRE:-0.012}"
PLANE_MARGIN="${PLANE_MARGIN:-0.008}"
MASS="${MASS:-0.1}"
UP="${UP:-}"
KEEP_SERVER="${KEEP_SERVER:-0}"
HAND="${HAND:-sharpa_wave}"
SKIP_ISAAC="${SKIP_ISAAC:-0}"
COACD_PRE="${COACD_PRE:-auto}"
COACD_RES="${COACD_RES:-50}"

# 路径自适应: 本机(anaconda3)与 UCB 部署(miniconda3)通用, 也可用环境变量覆写
DEXO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC="$DEXO/isaac"
for _c in "$HOME/anaconda3/envs/dexonomy" "$HOME/miniconda3/envs/dexonomy"; do
  [ -x "$_c/bin/python" ] && DEXO_ENV="$_c" && break
done
PY="${PY:-$DEXO_ENV/bin/python}"
DEXRUN="${DEXRUN:-$DEXO_ENV/bin/dexrun}"
SERVER=http://127.0.0.1:8765
export MUJOCO_GL=egl
cd "$DEXO"

[ -f "$MESH" ] || { echo "mesh not found: $MESH"; exit 1; }
EXP="output/${OID}_${HAND}"
echo "=== [$OID] mesh=$MESH tmpl=$TMPL epoch=$EPOCH ==="

echo "--- step 1/7 import"
if [ -n "$UP" ]; then
  # shellcheck disable=SC2086
  "$PY" tools/import_object.py --mesh "$MESH" --oid "$OID" --up $UP ${REGION_NPZ:+--region "$REGION_NPZ"} --coacd-preprocess "$COACD_PRE" --coacd-resolution "$COACD_RES" 2>&1 | grep -v CoACD | tail -2
else
  "$PY" tools/import_object.py --mesh "$MESH" --oid "$OID" --auto-up ${REGION_NPZ:+--region "$REGION_NPZ"} --coacd-preprocess "$COACD_PRE" --coacd-resolution "$COACD_RES" 2>&1 | grep -v CoACD | tail -3
fi

echo "--- step 2/7 init (epoch=$EPOCH)"
"$DEXRUN" op=init hand="$HAND" exp_name="$OID" tmpl_name="$TMPL" 'init_gpu=[0]' \
  "op.object.cfg_path=assets/object/custom/scene_cfg/$OID/tabletop/*.npy" \
  op.object.n_cfg=1 op.epoch="$EPOCH" op.object.n_init_point="$INIT_POINTS" \
  op.filter.general.n_final="$N_FINAL" op.filter.collision.plane_thre="$PLANE_THRE" 2>&1 | tail -1
N_INIT=$(find "$EXP/init_data" -name '*.npy' 2>/dev/null | wc -l)
echo "    init candidates: $N_INIT"
[ "$N_INIT" -gt 0 ] || { echo "no initializations — object may be too large/small for template $TMPL"; exit 1; }

echo "--- step 3/7 grasp refine"
"$DEXRUN" op=grasp hand="$HAND" exp_name="$OID" "op.grasp.plane_margin=$PLANE_MARGIN" 2>&1 | tail -1
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

if [ -n "${REGION_NPZ:-}" ] || python -c "import numpy,sys;d=numpy.load(sys.argv[1],allow_pickle=True).item();sys.exit(0 if (d.get('scene_cfg',{}).get('task') or {}).get('region') else 1)" "$(find "$EXP/grasp_data" -name '*.npy' | head -1)" 2>/dev/null; then
  echo "--- step 4.6 region filter (contacts must land in expected area)"
  # 默认只打分不淘汰(REGION_HARD=1 恢复硬过滤)
  "$PY" tools/filter_region.py --exp-dir "$EXP" --data grasp_data \
      ${REGION_HARD:+--min-frac "${REGION_MIN_FRAC:-0.25}"} ${REGION_HARD:---rank-only} 2>&1 | tail -1
  N_REG=$(find "$EXP/grasp_data" -name '*.npy' 2>/dev/null | wc -l)
  echo "    in-region scored: $N_REG"
  [ "$N_REG" -gt 0 ] || { echo "no grasps left"; exit 1; }
fi

echo "--- step 5/7 export Isaac trajectories"
"$PY" tools/export_isaac_traj.py --exp-dir "$EXP" --data grasp_data 2>&1 | tail -2

if [ "$SKIP_ISAAC" = "1" ]; then
  echo "--- SKIP_ISAAC=1: 跳过 Isaac 验证 (step 6-8)"
  echo "=== DONE (no Isaac). 候选池: $EXP/grasp_data/  ($N_NAT 个) ==="
  echo "    下一步 (RL_Correction 侧筛选):"
  echo "      \$PY -m tasks.pregrasp.screen_prior --clip <clip> \\"
  echo "          --grasp_dir $DEXO/$EXP --info_json $DEXO/assets/object/custom/processed_data/$OID/info/simplified.json"
  exit 0
fi

echo "--- step 6/7 Isaac server"
if ! curl -s -m 2 "$SERVER/status" >/dev/null 2>&1; then
  echo "    starting Isaac server (takes 1-2 min)..."
  ( cd "$ISAAC" && export OCIR_DATA_ROOT=$PWD/data && \
    nohup scripts/run_isaacsim_conda.sh scripts/start_isaacsim_server.py --mode local \
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
  ( cd "$ISAAC" && scripts/run_isaacsim_conda.sh scripts/start_isaacsim_server.py --shutdown-server >/dev/null 2>&1 ) || true
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
