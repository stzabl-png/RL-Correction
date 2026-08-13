"""把 v17A 自动认物体的结果画出来: 种子帧 + 若干传播帧的 mask overlay。"""
import cv2, numpy as np, json, os, sys, glob
vid_path, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
cap = cv2.VideoCapture(vid_path)
tiles = []
COLORS = [(0,0,255),(0,220,255),(255,120,0),(0,255,120)]
for ep in sorted(glob.glob(f"{root}/episode_*")):
    seq = sorted(glob.glob(f"{ep}/sequence_*/masks"))
    if not seq: continue
    frames = sorted(os.listdir(seq[0]))
    pick = [frames[0]] + [frames[len(frames)*k//4] for k in (1,2,3)]
    for fd in pick:
        fi = int(fd.split("_")[1])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, im = cap.read()
        if not ok: continue
        objs = sorted(os.listdir(f"{seq[0]}/{fd}"))
        lab = []
        for j, o in enumerate(objs):
            m = cv2.imread(f"{seq[0]}/{fd}/{o}", 0)
            if m is None: continue
            a = m > 127
            if a.sum() < 20: continue
            im[a] = (0.45*im[a] + 0.55*np.array(COLORS[j % 4])).astype(np.uint8)
            lab.append(f"{o[:-4]}:{int(a.sum())}px")
        cv2.rectangle(im, (0,0), (im.shape[1],46), (15,15,15), -1)
        cv2.putText(im, f"{os.path.basename(ep)} f{fi}", (6,18),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
        cv2.putText(im, " | ".join(lab)[:70], (6,38),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (200,255,200), 1)
        tiles.append(cv2.resize(im, (520,390)))
if tiles:
    W = 4
    rows = [np.hstack(tiles[i:i+W] + [np.zeros((390,520,3),np.uint8)]*((W-len(tiles[i:i+W]))%W))
            for i in range(0, len(tiles), W)]
    cv2.imwrite(out, np.vstack(rows)); print("ok", len(tiles), "tiles")
else:
    print("no tiles")
