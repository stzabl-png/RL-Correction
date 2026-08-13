import json, sys, os, cv2, numpy as np
d = json.load(open(sys.argv[1]))
miss = empty = tiny = ok = 0
areas = []
for fr in d["frames"]:
    for oid, o in fr.get("objects", {}).items():
        if not isinstance(o, dict): continue
        p = o.get("raw_mask")
        if not p or p in ("None", None) or not os.path.isfile(p):
            miss += 1; continue
        m = cv2.imread(p, 0)
        if m is None: miss += 1; continue
        a = int((m > 127).sum())
        areas.append(a)
        if a == 0: empty += 1
        elif a < 500: tiny += 1
        else: ok += 1
print(f"  raw_mask 文件: 缺失/读不出 {miss} | 全空(0px) {empty} | 极小(<500px) {tiny} | 可用 {ok}")
if areas:
    a = np.array(areas)
    print(f"  面积分位: p10 {np.percentile(a,10):.0f}  中位 {np.median(a):.0f}  p90 {np.percentile(a,90):.0f}")
