#!/usr/bin/env bash
# 重跑前后快照 —— 把"赌一把"变成"可撤销的尝试"。
#
# 为什么必须有 (2026-08-17 实测): fp_pose 是非确定性的, 同配置重跑 conf 能飘 25~36 分。
# 重建链会**覆盖**产物, 而闭环搜帧只在本轮搜出的配置里择优, 不知道重跑前的旧成绩。
# 已经因此弄坏两条: screw/4 (53.5→38.0)、screw/7 (30.0→29.0 掉出合格线), 旧产物找不回。
#
#   take_snapshot.sh save    <take目录>   跑之前存
#   take_snapshot.sh diff    <take目录>   跑之后比(打印分数变化, 不动文件)
#   take_snapshot.sh restore <take目录>   变差了就回滚
set -uo pipefail
ACT="${1:?save|diff|restore}"; TAKE="${2:?take 目录}"
TAKE="${TAKE%/}"
SNAP="$TAKE/.snapshot_before_rerun"
FILES=(confidence_complete.json world_fused.npz)
PY=${PY:-$HOME/miniconda3/envs/hawor/bin/python}

case "$ACT" in
  save)
    mkdir -p "$SNAP"
    for f in "${FILES[@]}"; do [ -f "$TAKE/$f" ] && cp -p "$TAKE/$f" "$SNAP/"; done
    [ -d "$TAKE/objects" ] && { rm -rf "$SNAP/objects"; cp -rp "$TAKE/objects" "$SNAP/"; }
    date -Iseconds > "$SNAP/.saved_at"
    echo "  [snapshot] 已存 $(du -sh "$SNAP" 2>/dev/null|cut -f1)  ($TAKE)"
    ;;
  diff)
    [ -d "$SNAP" ] || { echo "  [snapshot] 没有快照, 跳过对比"; exit 0; }
    $PY - "$TAKE" "$SNAP" <<'PYEOF'
import json, sys, os
take, snap = sys.argv[1], sys.argv[2]
def load(p):
    f = os.path.join(p, "confidence_complete.json")
    if not os.path.exists(f): return {}
    d = json.load(open(f))
    return {k: (v.get("conf_rot_median"), v.get("refuted_frames"), v.get("rotation_usable"))
            for k, v in (d.get("objects") or {}).items()}
old, new = load(snap), load(take)
n_o = len(os.listdir(os.path.join(snap, "objects"))) if os.path.isdir(os.path.join(snap, "objects")) else 0
n_n = len(os.listdir(os.path.join(take, "objects"))) if os.path.isdir(os.path.join(take, "objects")) else 0
print("  [snapshot] 物体数 %d -> %d%s" % (n_o, n_n, "  ★多了" if n_n > n_o else ("  ★少了!" if n_n < n_o else "")))
worse = False
for k in sorted(set(old) | set(new)):
    o, w = old.get(k), new.get(k)
    if o == w: continue
    tag = ""
    if o and w and o[2] and not w[2]: tag = "  ★★从可用掉成不可用"; worse = True
    elif o and w and (w[0] or 0) + 5 < (o[0] or 0): tag = "  ★分数明显下降"; worse = True
    print("  [snapshot] %-10s crot %s -> %s  证伪 %s -> %s%s" % (
        k, o and o[0], w and w[0], o and o[1], w and w[1], tag))
# ★ 物体数变多是**我们要的结果**, 不能因为主物体分数降了就判"建议 restore"。
# 2026-08-17 实测: screw/13 分出瓶盖后 物体数 1->2(目标达成) 但瓶身 crot 77->46,
# 工具当时报"建议 restore" —— 那会让人把刚救回来的盖子又删掉。
# 少物体才是真的坏; 多物体 + 主物体降档是**已知代价**, 交给人判断, 不给 restore 建议。
if n_n > n_o:
    print("  [snapshot] 判定: 物体数增加(目标达成); 主物体降档是已知代价 —— **不建议 restore**, 请人工权衡")
elif n_n < n_o:
    print("  [snapshot] 判定: ★物体数减少 —— 建议 restore")
else:
    print("  [snapshot] 判定: %s" % ("★变差 —— 建议 restore" if worse else "没变差"))
PYEOF
    ;;
  restore)
    [ -d "$SNAP" ] || { echo "  [snapshot] X 没有快照可回滚"; exit 1; }
    for f in "${FILES[@]}"; do [ -f "$SNAP/$f" ] && cp -p "$SNAP/$f" "$TAKE/"; done
    [ -d "$SNAP/objects" ] && { rm -rf "$TAKE/objects"; cp -rp "$SNAP/objects" "$TAKE/"; }
    echo "  [snapshot] 已回滚到 $(cat "$SNAP/.saved_at" 2>/dev/null)"
    ;;
  *) echo "用法: $0 save|diff|restore <take目录>"; exit 2 ;;
esac
