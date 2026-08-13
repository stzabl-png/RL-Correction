"""绕过 v17A 时序质量门: 把所有帧(含 rejected)标成 accepted, 让杜邦自己的门去筛。

用意: v17A 的门砍掉 ketchup 97% 的帧, 只剩 9 帧 -> 选帧器弃权。
但杜邦第一段自己也有碎裂门(frag = 最大连通块/总面积 >= 0.98), **比 v17A 更严**。
本脚本让两道门只剩一道, 看候选池从 9 涨到多少、他最终会挑哪一帧。
被拒帧的 mask 路径在 raw_mask 里(mask 字段为 None), 一并搬过去。
"""
import json, sys, os
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
n_all = n_fixed = 0
for fr in d["frames"]:
    for oid, o in fr.get("objects", {}).items():
        if not isinstance(o, dict):
            continue
        n_all += 1
        if o.get("status") != "accepted":
            raw = o.get("raw_mask")
            if raw and raw not in ("None", None) and os.path.isfile(raw):
                o["mask"] = raw          # 被拒帧的 mask 字段是空的, 用 raw_mask 顶上
                o["status"] = "accepted"
                o["_bypassed_v17a_gate"] = True
                n_fixed += 1
d["schema_version"] = "persistent_video_mask_sequence_v1_synth"
d.setdefault("provenance", {})["gate_bypass"] = f"v17A 时序门已绕过: {n_fixed}/{n_all} 帧由 rejected 提为 accepted"
json.dump(d, open(dst, "w"), ensure_ascii=False)
print(f"  总条目 {n_all}, 提升 {n_fixed} 帧 -> {dst}")
