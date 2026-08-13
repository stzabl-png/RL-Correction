import json, sys, statistics as st
p = sys.argv[1]
d = json.load(open(p))
acc, rej = [], []
for fr in d["frames"]:
    for oid, o in fr.get("objects", {}).items():
        if not isinstance(o, dict):
            continue
        (acc if o.get("status") == "accepted" else rej).append(o.get("metrics") or {})
def med(rows, k):
    v = [r[k] for r in rows if isinstance(r.get(k), (int, float, bool))]
    return f"{st.median(v):.4f}" if v else "-"
print(f"{'指标':30}{f'accepted({len(acc)})':>16}{f'rejected({len(rej)})':>18}")
for k in ("area_pixels", "area_fraction", "largest_component_fraction",
          "component_count", "touches_image_border", "bbox_diagonal_pixels"):
    print(f"  {k:28}{med(acc,k):>16}{med(rej,k):>18}")
