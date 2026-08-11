#!/usr/bin/env bash
# arctic15 批量评测：远端补 bridge+接触区间 → 拉回 → 逐条对真值 → 汇总表。
# 幂等：已处理过的 take 会跳过重复的远端计算，只是重算本地指标（秒级）。
#
# 为什么 postproc 放在远端跑：接触检测要读 take 目录下的 masks/（几千个 png），
# 本地 rsync 时刻意排除了它们（省空间），所以在远端算完只拉回小 json。
set -uo pipefail
RR=/home/lyh/Project/Reconstruct_and_Retarget
S=/tmp/claude-1000/-home-lyh/90060bb3-d0eb-4640-b9d4-aab56bc267bb/scratchpad
H=yanghong@169.229.192.185
REMOTE=~/Reconstruct_and_Retarget/Output/ReconstructOutput/arctic15

echo "=== 1/3 远端补 bridge + 接触区间 ==="
ssh $H "sed 's|/arctic$|/arctic15|' /tmp/remote_postproc.sh > /tmp/remote_postproc15.sh; bash /tmp/remote_postproc15.sh" 2>&1 | tail -14

echo "=== 2/3 拉回（不含 masks）==="
for t in $(ssh $H "find $REMOTE -name world_fused.npz | sed 's|.*/arctic15/||;s|/world_fused.npz||'"); do
  mkdir -p "$RR/Output/ReconstructOutput/arctic15/$t"
  rsync -az --exclude masks "$H:$REMOTE/$t/" "$RR/Output/ReconstructOutput/arctic15/$t/" 2>/dev/null
done
scp -q "$H:~/Reconstruct_and_Retarget/Data/VideoPrior/poseqa/pose_audit.json" "$S/audit15.json" 2>/dev/null

echo "=== 3/3 逐条对真值 ==="
cd "$RR"
for t in $(ls -d Output/ReconstructOutput/arctic15/*/*/ 2>/dev/null); do
  sub=$(basename $(dirname "$t")); seq=$(basename "$t")
  meta="$S/arctic15/${sub}__${seq}.meta.json"
  [ -f "$meta" ] || { echo "  跳过 $sub/$seq（缺 meta）"; continue; }
  python3 tools/arctic_eval/arctic_eval.py --take "$t" --meta "$meta" \
    --audit "$S/audit15.json" --out Output/arctic_eval/rows15 >/dev/null 2>&1 \
    || echo "  X $sub/$seq 评测失败"
done
python3 tools/arctic_eval/shape_table.py 2>/dev/null || true
echo "完成：$(ls Output/arctic_eval/rows15/*.json 2>/dev/null | wc -l) 条已评测"
