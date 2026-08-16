#!/usr/bin/env bash
# 从一条重建 take 生成 GraspPose —— 基础 Dexonomy 管线 + 经验先验驱动的模板选择。
#
#   tools/take_grasp_pipeline.sh <take目录> <object_id> <left|right> <oid>
#
# 链路(2026-08-15 定):
#   ① 摆放      upright_from_recon: 开头10帧位姿中位数 -> 吸附稳定静置姿态
#               up -> +Z, front -> -X (front 来自重建世界系 +X="相机前向", 与 RL 机器人侧一致)
#   ② 粗筛      coarse_filter_templates: 只砍**几何上不可能**的(指数±tol, 隐含直径比)
#               ★不用 VLM 的 depth/palm 硬过滤 —— 实测它一刀砍掉 12 个模板, 含最好用的
#                 1_Large_Diameter; 尺寸闸已判"改按环握族"却忘了放开 depth, 逻辑自相矛盾
#   ③ 查先验    template_prior: 按(形状|直径比档|空心) 取历史表现排序, 含 ε-greedy 探索位
#   ④ 低epoch扫 逐个测, 够用就停(达到该档位历史最高 × STOP_FRAC)
#   ⑤ 精生成    选中的模板用高 epoch 重跑
#   ⑥ 评分排序  rank_by_coverage: 0.4×覆盖率@1cm + 0.3×高度匹配 + 0.3×离桌余量
#               硬闸: 虎口朝下 / 腔内 / 离桌余量(用**完整网格**, 现有 filter_plane_clearance
#               只查骨架胶囊会漏 —— 实测瓶子 5_35 过了那道闸但按包围球算已在桌下)
#   ⑦ 回写先验  实测结果写回, 用得越多越准
#
# ⚠ 别再替换 op=init。2026-08-14 曾用"视频姿态种子"换掉它, 理由是"盲搜要 20~40 分钟" ——
#   那是因为当时把 op.epoch 设成了 3000, 默认值是 50。替换它还丢掉了自带的
#   op.filter.collision, 直接导致手指插进物体。
#
# 环境变量:
#   HAND_BASE (sharpa_wave_v2)  手模型。v2 = 官方 menagerie 版, 掌部 32 块凸分解;
#                               旧版只有 8 块粗凸包(体积 213.6 vs 57.0 cm³), 会把大量候选
#                               误判成"手掌撞物体"。实测同条件杯子力封闭 397/500 -> 496/500
#   SCAN_EPOCH (3) / SCAN_PTS (2048)               扫描阶段
#   GEN_EPOCH (10) / GEN_PTS (4096) / GEN_NFINAL (50)  精生成阶段
#                               ★这组比上游默认 50×1024×10 快 2.6 倍且质量更好 —— 瓶颈是
#                                 轮次开销, 加大采样池+按同比例放大配额即可。但配额不能单独
#                                 放大(5×8192×n_final100 实测 QP 从 25 掉到 9: 配额补齐会
#                                 从被拒的候选里随机抓来凑数)
#   STOP_FRAC (0.85) / MIN_TRY (3) / MAX_TRY (5) / TOP_N (5)
#                               MIN_TRY: 早停前至少试满几个模板。只试 1 个就收工会让
#                               排序阶段无从选择 —— 天花板低的档位尤其需要多几批候选
set -euo pipefail
TAKE="${1:?usage: $0 <take> <object_id> <left|right> <oid>}"
OBJ="${2:?}"; SIDE="${3:?}"; OID="${4:?}"

DEXO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$DEXO"
DEXO_ENV="${DEXO_ENV:-}"
[ -n "$DEXO_ENV" ] || for _c in "$HOME/anaconda3/envs/dexonomy" "$HOME/miniconda3/envs/dexonomy"; do
  [ -x "$_c/bin/python" ] && DEXO_ENV="$_c" && break
done
PY="${PY:-$DEXO_ENV/bin/python}"; export MUJOCO_GL=egl
# ★必须把 env 的 bin 加进 PATH: dexrun 是 console script, 只设 PY 不够。
#   踩过的坑: 少了这句 + `|| true` 吞掉 "command not found", 每个模板静默产出 0 个候选,
#   而 output/scan_* 目录压根没被创建过 —— 表现成"扫描目录凭空消失"。
export PATH="$DEXO_ENV/bin:$PATH"
command -v dexrun >/dev/null || { echo "找不到 dexrun ($DEXO_ENV/bin)"; exit 1; }
LOGD="output/_logs"; mkdir -p "$LOGD"
run() {   # run <日志名> <dexrun 参数...>   —— 失败只警告不中断, 但一定说出来
  local tag="$1"; shift
  dexrun "$@" >"$LOGD/$tag.log" 2>&1 || echo "     ⚠ $tag 失败, 见 $LOGD/$tag.log"
}
HAND_BASE="${HAND_BASE:-sharpa_wave_v2}"
SUF=""; XML=right.xml; [ "$SIDE" = left ] && { SUF="_left"; XML=left.xml; }
HAND="${HAND_BASE}${SUF}"; HXML="assets/hand/$HAND/$XML"
SCAN_EPOCH="${SCAN_EPOCH:-3}"; SCAN_PTS="${SCAN_PTS:-2048}"
GEN_EPOCH="${GEN_EPOCH:-10}"; GEN_PTS="${GEN_PTS:-4096}"; GEN_NFINAL="${GEN_NFINAL:-50}"
STOP_FRAC="${STOP_FRAC:-0.85}"; MIN_TRY="${MIN_TRY:-3}"; MAX_TRY="${MAX_TRY:-5}"; TOP_N="${TOP_N:-5}"
PD=assets/object/custom/processed_data
T0=$(date +%s)

echo "══ $OID  ($OBJ × ${SIDE}手, $HAND)"

# ── ① 摆放
# PLACEMENT_MODE=stable_only 由 run_take.py 在 Step2 判 rotation_usable=false 时传入
UPRIGHT_ARGS=""; [ "${PLACEMENT_MODE:-recon}" = stable_only ] && UPRIGHT_ARGS="--stable-only"
J=$("$PY" tools/upright_from_recon.py --recon "$TAKE" --object "$OBJ" $UPRIGHT_ARGS)
UP=$(echo "$J" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['UP_ENV'])")
FRONT=$(echo "$J" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['FRONT_ENV'])")
PLACEMENT_JSON="$J"
echo "  [①摆放] UP=$UP  FRONT=$FRONT  来源=$(echo "$J" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['source'])")"

REG="/tmp/region_${OID}.npz"
"$PY" - "$TAKE" "$OBJ" "$SIDE" "$REG" <<'EOF'
import numpy as np, sys
take, obj, side, out = sys.argv[1:5]
z = np.load(f"{take}/contact/contact_v2_{obj}_{side}.npz", allow_pickle=True)
np.savez_compressed(out, points=np.asarray(z["probe_local"], np.float32),
                    weight=np.asarray(z["weight"], np.float32))
print(f"  [接触区] {int((np.asarray(z['weight'])>=0.5).sum())} 个热点")
EOF

rm -rf "$PD/$OID" "assets/object/custom/scene_cfg/$OID"
"$PY" tools/import_object.py --mesh "$TAKE/objects/$OBJ/object_mesh_scaled_final.obj" --oid "$OID" \
  --up $UP --front $FRONT --region "$REG" --region-radius 0.03 2>&1 \
  | grep -E "front|region|convex" | sed 's/^/  /'

# ── 物体特征: 接触带直径/高度、是否空心 (供粗筛与先验分档)
FEAT=$("$PY" - "$PD/$OID" <<'EOF2'
import numpy as np, sys, trimesh
pd = sys.argv[1]
m = trimesh.load(f"{pd}/mesh/simplified.obj", force="mesh", process=False)
V = np.asarray(m.vertices); lo, hi = V[:, 2].min(), V[:, 2].max()
rz = np.load(f"{pd}/region.npz")
hot = np.asarray(rz["points"], float)[np.asarray(rz["weight"], float) >= float(rz["min_weight"])]
# 接触带直径 = 热点到主轴距离中位数 ×2。不是整体 AABB —— 手抓的是那一圈, 不是最大截面
dia = 2 * np.median(np.linalg.norm(hot[:, :2], axis=1)) * 1000
ref_h = (np.median(hot[:, 2]) - lo) / (hi - lo) * 100
# ★空心判据用**凸分解块的总体积 / 凸包体积**, 不能用 mesh.volume ——
#   SAM3D 出来的网格常常不水密(实测瓶子 is_watertight=False), volume 不可靠,
#   按体积比会把实心瓶子误判成空心(3.45>2.0), 进而查错先验档位。
#   凸分解块是实打实填充实体的: 杯 105/601=0.17(空心), 瓶 1 块 ≈ 凸包(实心)。
import glob as _g
pv = sum(trimesh.load(f, force="mesh", process=False).convex_hull.volume
         for f in _g.glob(f"{pd}/urdf/meshes/*.obj"))
hollow = 1 if pv / max(m.convex_hull.volume, 1e-9) < 0.5 else 0
print(f"{dia:.1f} {hollow} {ref_h:.0f}")
EOF2
)
OBJ_DIA=$(echo $FEAT | cut -d' ' -f1); HOLLOW=$(echo $FEAT | cut -d' ' -f2)
REF_H=$(echo $FEAT | cut -d' ' -f3)
# 下面三个可由 run_take.py 从 Step2 交接件覆写, 覆写后就不用这里现算的:
#   REF_H_OVERRIDE  <- grasp_prompt.json 的 contact_region.height_pct_median
#                      (Step2 在**原始接触点云**上算, 比这里在重采样后的 region.npz 上算更权威;
#                       pour/17 两者都是 61%, 是这条改动的对拍基准)
#   SHAPE           <- vlm_grasp.json 的 answer.<side>.object_shape (先验档位键的形状维)
#   DIGITS          <- vlm_grasp.json 的 answer.<side>.n_contact_fingers (粗筛的指数目标)
REF_H="${REF_H_OVERRIDE:-$REF_H}"; SHAPE="${SHAPE:-cylinder}"; DIGITS="${DIGITS:-4}"
echo "  [物体] 接触带直径 ${OBJ_DIA}mm  接触带高度 ${REF_H}%  形状 ${SHAPE}  $([ "$HOLLOW" = 1 ] && echo 空心 || echo 实心)"

# ── ② 粗筛
CAND=$("$PY" tools/coarse_filter_templates.py --hand "assets/hand/$HAND" \
        --obj-diameter-mm "$OBJ_DIA" --digits "$DIGITS" --tol 1 2>/dev/null | tail -1)
echo "  [②粗筛] 保留 $(echo $CAND | wc -w) 个模板"

# ── ③ 查先验(含 ε-greedy 探索位)
HOLLOW_FLAG=""; [ "$HOLLOW" = 1 ] && HOLLOW_FLAG="--hollow"
ORDER=$("$PY" tools/template_prior.py query --shape "$SHAPE" --obj-dia-mm "$OBJ_DIA" $HOLLOW_FLAG \
        --candidates $CAND 2>/dev/null | sed -n 's/^ *[0-9]*\. \([^ ]*\).*/\1/p') || true
[ -z "${ORDER:-}" ] && ORDER="$CAND"
THR=$("$PY" tools/template_prior.py query --shape "$SHAPE" --obj-dia-mm "$OBJ_DIA" $HOLLOW_FLAG 2>/dev/null \
      | sed -n 's/.*够用阈值 \([0-9.]*\)%.*/\1/p') || true
THR=${THR:-30}
echo "  [③先验] 试序: $(echo $ORDER | cut -d' ' -f1-$MAX_TRY)   够用阈值 ${THR}%"

# ── ④ 低 epoch 扫描 + 早停 + 回写先验
PICKED=""; n=0
for TM in $ORDER; do
  n=$((n+1)); [ $n -gt "$MAX_TRY" ] && break
  E="output/scan_${TM}_${HAND}"; rm -rf "$E"
  run "init_scan_$TM" op=init hand="$HAND" exp_name="scan_${TM}" tmpl_name="$TM" 'init_gpu=[0]' \
    "op.object.cfg_path=assets/object/custom/scene_cfg/$OID/tabletop/*.npy" \
    op.object.n_cfg=1 op.epoch="$SCAN_EPOCH" op.object.n_init_point="$SCAN_PTS" \
    op.filter.general.n_final=50 op.filter.collision.plane_thre=0.012
  run "grasp_scan_$TM" op=grasp hand="$HAND" exp_name="scan_${TM}" op.grasp.plane_margin=0.008
  BEST=$("$PY" tools/rank_by_coverage.py --glob "$E" --hand "$HXML" --region "$PD/$OID/region.npz" \
         --mesh "$PD/$OID/mesh/simplified.obj" --r 0.01 --ref-height "$REF_H" --top 1 2>/dev/null \
         | sed -n 's/^ *1 *\([0-9.]*\).*/\1/p') || true
  BEST=${BEST:-0}
  PCT=$("$PY" -c "print(f\"{float('$BEST')*100:.1f}\")")
  echo "     [$TM] 最高分 ${PCT}%"
  PICKED="$PICKED $TM"
  "$PY" tools/template_prior.py update --shape "$SHAPE" --obj-dia-mm "$OBJ_DIA" $HOLLOW_FLAG \
        --tmpl "$TM" --cov "$BEST" >/dev/null
  # ★早停要等试满 MIN_TRY 个。只试 1 个就收工的坑(pour/17 杯子实测): 先验里
  #   18_Extensior_Type 有 37.1%, 而该档位没有更高的历史, 兜底阈值 30% 一上来就被满足 ——
  #   于是只精生成了这一个模板, 排序阶段无从选择, Top-1 是个从上方钩杯沿的姿势。
  #   天花板低的档位更需要多几批候选给排序挑, 不是更不需要。
  if [ $n -ge "$MIN_TRY" ] && "$PY" -c "import sys;sys.exit(0 if float('$PCT')>=float('$THR') else 1)"; then
    echo "     ★ 已试 $n 个且达到够用阈值, 不再试后面的模板"; break
  fi
done

# ── ⑤ 精生成
echo "  [⑤精生成]$PICKED  (epoch=$GEN_EPOCH pts=$GEN_PTS)"
EXP="output/${OID}_${HAND}"; rm -rf "$EXP"; mkdir -p "$EXP/grasp_data"
for TM in $PICKED; do
  E="output/gen_${TM}_${HAND}"; rm -rf "$E"
  run "init_gen_$TM" op=init hand="$HAND" exp_name="gen_${TM}" tmpl_name="$TM" 'init_gpu=[0]' \
    "op.object.cfg_path=assets/object/custom/scene_cfg/$OID/tabletop/*.npy" \
    op.object.n_cfg=1 op.epoch="$GEN_EPOCH" op.object.n_init_point="$GEN_PTS" \
    op.filter.general.n_final="$GEN_NFINAL" op.filter.collision.plane_thre=0.012
  run "grasp_gen_$TM" op=grasp hand="$HAND" exp_name="gen_${TM}" op.grasp.plane_margin=0.008
  echo "     [$TM] 力封闭 $(find "$E/grasp_data" -name '*_grasp.npy' 2>/dev/null | wc -l)"  # 只数 grasp_data, init_data 下同名文件是初始化候选
  find "$E/grasp_data" -name '*_grasp.npy' -exec sh -c \
    'cp "$1" "$2/'"$TM"'__$(basename "$1")"' _ {} "$EXP/grasp_data" \; 2>/dev/null || true
done

# ── ⑥ 评分排序 + 出图
rm -rf "$EXP/top" "output/${OID}_final"
"$PY" tools/rank_by_coverage.py --glob "$EXP" --hand "$HXML" --region "$PD/$OID/region.npz" \
    --mesh "$PD/$OID/mesh/simplified.obj" --r 0.01 --ref-height "$REF_H" \
    --top "$TOP_N" --copy-top-to "$EXP/top" --json "$EXP/ranking.json" | sed 's/^/  /'
"$PY" tools/render_grasp_views.py --grasp-dir "$EXP/top" --hand "$HXML" \
    --out-dir "output/${OID}_final" --n "$TOP_N" | sed 's/^/  /'

# ── ⑦ 本次运行的机读小结, 给 run_take.py 汇总成 grasp_pose_plan.json
"$PY" - "$EXP" "$OID" "$OBJ" "$SIDE" "$HAND" "$OBJ_DIA" "$REF_H" "$SHAPE" "$HOLLOW" \
      "$PICKED" "$DEXO/output/${OID}_final" <<'EOF3'
import json, os, sys
exp, oid, obj, side, hand, dia, refh, shape, hollow, picked, viz = sys.argv[1:12]
rk = os.path.join(exp, "ranking.json")
rkd = json.load(open(rk)) if os.path.isfile(rk) else {}
json.dump({"oid": oid, "object_id": obj, "hand": side, "hand_model": hand,
           "contact_band_diameter_mm": float(dia), "contact_band_height_pct": float(refh),
           "shape": shape, "hollow": bool(int(hollow)),
           "templates_tried": picked.split(), "viz_dir": viz,
           "ref_height_video": rkd.get("ref_height_video"),
           "ref_height_used": rkd.get("ref_height_used"),
           "ref_lifted": rkd.get("ref_lifted", False),
           "ranking": rkd.get("rows", [])},
          open(os.path.join(exp, "summary.json"), "w"), ensure_ascii=False, indent=1)
EOF3
[ -n "$PLACEMENT_JSON" ] && echo "$PLACEMENT_JSON" > "$EXP/placement.json"
echo "══ 总计 $(( $(date +%s)-T0 ))s   可视化: $DEXO/output/${OID}_final"
