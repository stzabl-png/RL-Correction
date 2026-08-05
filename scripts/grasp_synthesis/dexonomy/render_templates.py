"""把一只手的全部抓型模板渲成一张对照图 (含标注). 默认 SharpaWave, 也能渲 Shadow.

  PY=/home/lyh/anaconda3/envs/dexonomy/bin/python

  # SharpaWave (9 个)
  MUJOCO_GL=egl $PY tools/render_templates.py

  # Shadow 的 GRASP 分类学 33 型 + 3 自定义
  MUJOCO_GL=egl $PY tools/render_templates.py \
      --tmpl_dir assets/hand/shadow/init_tmpl --hand_xml assets/hand/shadow/right.xml \
      --out output/shadow_taxonomy_montage.png --size 400 --cols 6

产物: <out>.png + <out 去扩展名>_renders/<名>.png (每只手各自一个目录, 防重名覆盖)

标注 "指尖 N/5  必接触 M":
  N = 接触体里涉及了几根手指的**远端节** (作者标注的)
  M = 其中**强制**接触远端节的有几根 —— body_group 把每指 DP/MP/PP 归成一组,
      同组"任一接触即可", 所以松弛组里的指尖可被指节顶替 (10_Power_Disk N=5 但 M=0)

⚠ 对 SharpaWave 而言指尖恰好就是 5 个 elastomer 垫 —— 唯一有高摩擦(9.0)和力传感器的
地方, MP/PP/掌心是 0.6 且不可感知。所以 **M** < 4 的模板生成的抓姿过不了 RL 侧的
`pads* >= 4` (见 RL_Correction/docs/GRASPPOSE_SCREENING.md §2)。上色与默认排序都按 M。
我们自己移植时 contact 只写 5 个 `*_DP`, 每组只剩一个成员, N == M。
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

# 指尖(远端节)的 body 命名: sharpa_wave = right_<指>_DP, shadow = rh_<指>distal.
# ⚠ 对 sharpa_wave 而言指尖恰好就是 5 个 elastomer 垫 (唯一有高摩擦 9.0 + 力传感器
#   的地方), 所以"指尖指数"直接等于 RL 侧候选判据的 pads —— 需要 >=4 才可能过 Gate 2。
DISTAL_RE = re.compile(r"(?:right_(thumb|index|middle|ring|pinky)_DP"
                       r"|rh_(th|ff|mf|rf|lf)distal)$")


def _fing(b):
    m = DISTAL_RE.match(b)
    return (m.group(1) or m.group(2)) if m else None


def distal_fingers(cbody, required=None):
    """-> (标注指尖数, 非指尖接触体数, **必接触**指尖数)

    ⚠ body_group.yaml 把每根手指的 DP/MP/PP 归成一组, 同组"任一接触即可"。
    所以"标注了指尖"不等于"指尖一定接触" —— required_cbody 里长度 >1 的组是松弛的,
    组里那个指尖可以被指节顶替 (10_Power_Disk 标了 5 个指尖但必接触指尖 = 0)。
    我们自己移植时只写 5 个 *_DP, 每组只剩一个成员, 两个数就相等。
    """
    fing, other = set(), set()
    for b in set(cbody):
        f = _fing(b)
        (fing.add(f) if f else other.add(b))
    must = set()
    if required is not None:
        for g in required:
            g = list(g)
            if len(g) == 1 and _fing(g[0]):
                must.add(_fing(g[0]))
    return len(fing), len(other), len(must)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmpl_dir", default="assets/hand/sharpa_wave/init_tmpl")
    ap.add_argument("--hand_xml", default="assets/hand/sharpa_wave/right.xml")
    ap.add_argument("--out", default="output/sharpa_templates_montage.png")
    ap.add_argument("--size", type=int, default=560)
    ap.add_argument("--distance", type=float, default=0.26)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--render_dir", default=None,
                    help="单张落盘处; 默认由 --out 派生 (避免多只手重名互相覆盖)")
    ap.add_argument("--group-meta", default=None,
                    help="json: {模板名: {shadow_fingers: N, ...}}; 给了就按 N 分组出带标题的大图")
    ap.add_argument("--sort", default="fingers", choices=("name", "fingers"),
                    help="fingers = 指尖指数降序 (够格的排前面), name = 字典序")
    ap.add_argument("--bg", default=None,
                    help='背景色 "R,G,B" (如 252,228,236 浅粉). 省略=MuJoCo 默认深色。'
                         "给了就用分割掩膜把手抠出来合成, 边缘精确; 标注区同时转浅色主题")
    args = ap.parse_args()

    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    files = sorted(glob.glob(os.path.join(args.tmpl_dir, "*.npy")))
    if not files:
        print(f"no templates under {args.tmpl_dir}")
        return 1

    m = mujoco.MjModel.from_xml_path(args.hand_xml)
    d = mujoco.MjData(m)
    m.vis.global_.offwidth = args.size
    m.vis.global_.offheight = args.size
    m.vis.headlight.ambient[:] = 0.55
    m.vis.headlight.diffuse[:] = 0.75
    renderer = mujoco.Renderer(m, args.size, args.size)

    BG = None
    if args.bg:
        BG = np.array([int(v) for v in args.bg.split(",")], dtype=np.uint8)
        assert BG.shape == (3,), "--bg 要写成 R,G,B"
    light = BG is not None and int(BG.astype(int).sum()) > 384      # 浅底 -> 深字
    if light:
        # 浅底上手会变成黑剪影, 要显著提亮才看得出指节形态
        m.vis.headlight.ambient[:] = 0.30
        m.vis.headlight.diffuse[:] = 0.95
        m.vis.headlight.specular[:] = 0.20
        for gi in range(m.ngeom):     # 目标: 中灰手 + 明显明暗过渡, 既不黑成剪影也不惨白
            c = m.geom_rgba[gi]
            if c[3] > 0:
                m.geom_rgba[gi][:3] = np.clip(c[:3] + 0.16, 0, 1)

    outdir = args.render_dir or (os.path.splitext(args.out)[0] + "_renders")
    os.makedirs(outdir, exist_ok=True)
    # 标注含中文, 必须用带 CJK 字形的字体 (DejaVu 会渲成方框)
    CJK = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
           "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]
    CJK_R = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]

    def _load(paths, size):
        for p in paths:
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
        return ImageFont.load_default()

    font = _load(CJK, 22)
    font_s = _load(CJK_R, 17)

    if args.sort == "fingers":
        def _key(f):
            r = np.load(f, allow_pickle=True).item()
            nf, no, nm = distal_fingers(list(r["hand_cbody"]), r.get("required_cbody"))
            return (-nm, -nf, -no, os.path.basename(f))
        files = sorted(files, key=_key)

    tiles = []
    for f in files:
        rec = np.load(f, allow_pickle=True).item()
        name = os.path.basename(f)[:-4]
        # grasp_qpos = [pos3, quat4(wxyz), 指22]; 手型只看后 22 维 (基座位姿与模板无关)
        d.qpos[:] = np.asarray(rec["grasp_qpos"][0][7:], np.float64)[:m.nq]
        mujoco.mj_forward(m, d)

        lookat = d.xpos[1:].mean(axis=0)
        views = []
        for az, el in ((120, -25), (240, -25)):
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = lookat
            cam.distance = args.distance
            cam.azimuth, cam.elevation = az, el
            renderer.update_scene(d, cam)
            rgb = renderer.render().copy()
            if BG is not None:
                # 分割渲染给出每像素的 geom id, -1 = 背景 -> 精确掩膜, 不靠颜色阈值
                renderer.enable_segmentation_rendering()
                renderer.update_scene(d, cam)
                seg = renderer.render()
                renderer.disable_segmentation_rendering()
                mask = (seg[:, :, 0] != -1)[..., None]
                rgb = np.where(mask, rgb, BG[None, None, :]).astype(np.uint8)
            views.append(rgb)
        img = np.concatenate(views, axis=0)

        cbody = list(rec["hand_cbody"])
        n_cpn = int(np.asarray(rec["hand_cpn_w"]).shape[0])
        n_fing, n_other, n_must = distal_fingers(cbody, rec.get("required_cbody"))

        # ---- 贴标注 ----
        pil = Image.fromarray(img).convert("RGB")
        bar = 96
        bar_bg = tuple(int(v) for v in (BG * 0.93).astype(int)) if light else (18, 18, 20)
        canvas = Image.new("RGB", (pil.width, pil.height + bar), bar_bg)
        canvas.paste(pil, (0, 0))
        dr = ImageDraw.Draw(canvas)
        y = pil.height + 8
        dr.text((10, y), name, font=font, fill=(25, 20, 25) if light else (255, 255, 255))
        y += 26
        dr.text((10, y), f"接触点 {n_cpn}  |  接触体 {len(set(cbody))}",
                font=font_s, fill=(105, 95, 105) if light else (190, 190, 195))
        y += 21
        # 指尖指数 >=4 才可能过 RL 侧的 pads*>=4
        # 按**必接触**指尖数上色 (松弛组里的指尖可被指节顶替, 不算数)
        if light:      # 浅底上要用深一档的色, 否则绿/黄看不清
            col = (22, 128, 62) if n_must >= 4 else \
                  (168, 110, 10) if n_must == 3 else (190, 40, 55)
        else:
            col = (110, 220, 130) if n_must >= 4 else \
                  (240, 200, 90) if n_must == 3 else (240, 120, 110)
        txt = f"指尖 {n_fing}/5  必接触 {n_must}"
        if n_other:
            txt += f"  +非指尖 {n_other}"
        if n_must < 4:
            txt += "   ✗"
        dr.text((10, y), txt, font=font_s, fill=col)

        arr = np.asarray(canvas)
        imageio.imwrite(f"{outdir}/{name}.png", arr)
        tiles.append(arr)
        print(f"rendered {name:26s} cpn={n_cpn:2d} 指尖={n_fing}/5 "
              f"必接触={n_must} 非指尖={n_other}")

    # ---- 分组 (可选) ----
    cols = args.cols
    if args.group_meta:
        import json
        meta = json.load(open(args.group_meta))
        order_names = [os.path.basename(f)[:-4] for f in files]
        blocks = []
        for n_fing in (5, 4, 3, 2, 1):
            idx = [k for k, nm in enumerate(order_names)
                   if meta.get(nm, {}).get("shadow_fingers") == n_fing]
            if not idx:
                continue
            hdr_h = 76
            hdr = Image.new("RGB", (tiles[0].shape[1] * cols, hdr_h),
                            (86, 96, 120) if n_fing >= 4 else (120, 96, 86))
            fb = _load(CJK, 34)
            ImageDraw.Draw(hdr).text(
                (18, 20), f"  {n_fing} 指抓法   (Shadow 原设定)   —— {len(idx)} 个",
                font=fb, fill=(255, 255, 255))
            sub = [tiles[k] for k in idx]
            rr = [sub[i:i + cols] for i in range(0, len(sub), cols)]
            gg = []
            for r in rr:
                while len(r) < cols:
                    pad = np.empty_like(tiles[0])
                    pad[:] = (BG if BG is not None else np.array([18, 18, 18], np.uint8))
                    r.append(pad)
                gg.append(np.concatenate(r, axis=1))
            blocks.append(np.asarray(hdr))
            blocks.append(np.concatenate(gg, axis=0))
            print(f"  [{n_fing} 指] {len(idx)} 个: "
                  + ", ".join(order_names[k] for k in idx))
        montage = np.concatenate(blocks, axis=0)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        imageio.imwrite(args.out, montage)
        print(f"\nsaved montage {montage.shape[1]}x{montage.shape[0]} -> {args.out}")
        print(f"       单张     -> {outdir}/")
        return 0

    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    h, w = tiles[0].shape[:2]
    grid = []
    for r in rows:
        while len(r) < cols:
            pad = np.empty_like(tiles[0])
            pad[:] = (BG if BG is not None else np.array([18, 18, 18], np.uint8))
            r.append(pad)
        grid.append(np.concatenate(r, axis=1))
    montage = np.concatenate(grid, axis=0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.imwrite(args.out, montage)
    print(f"\nsaved montage {montage.shape[1]}x{montage.shape[0]} -> {args.out}")
    print(f"       单张     -> {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
