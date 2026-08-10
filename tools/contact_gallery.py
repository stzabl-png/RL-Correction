#!/usr/bin/env python
"""One self-contained HTML page to triage every take's contact result at a glance.

Sorted worst-first, because the point is to find the failures, not to admire the wins.
Each take gets a verdict from the numbers that can actually invalidate a result:

  RED     the result contradicts the observation or physics -- do not use it
            pads on provably-untouched surface > 50%   (hand is on the wrong side)
            penetration > 2 cm                          (sunk into the object)
            object projection IoU < 0.35                (the OBJECT is mis-reconstructed,
                                                         so heat painted on it is meaningless)
  AMBER   usable with care: weak evidence or a soft violation
  GREEN   nothing objectionable

Usage:
  python tools/contact_gallery.py <RetargetOutput subtree> [--out gallery.html]
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def verdict(d) -> tuple:
    a, h = d.get("alignment") or {}, d.get("heatmap") or {}
    iou = a.get("bite_iou_after", 0.0)
    neg = a.get("pads_on_free_surface", 0.0)
    pen = a.get("penetration_max_cm", 0.0)
    obj = d.get("object_mask_iou", 0.0)
    bite = d.get("occlusion", {}).get("bite_px", 0)

    red, amber = [], []
    if neg > 0.5:
        red.append(f"{neg*100:.0f}% of touching pads sit on surface proven untouched")
    if pen > 2.0:
        red.append(f"penetrates {pen:.1f} cm into the object")
    if obj < 0.35:
        red.append(f"object itself mis-reconstructed (projection IoU {obj:.2f})")
    if iou < 0.35:
        amber.append(f"low bite IoU {iou:.2f}")
    if obj < 0.5:
        amber.append(f"weak object reconstruction (IoU {obj:.2f})")
    if pen > 1.0:
        amber.append(f"penetration {pen:.1f} cm")
    if neg > 0.1:
        amber.append(f"{neg*100:.0f}% pads on untouched surface")
    if bite < 1000:
        amber.append(f"tiny observed bite ({bite} px) -- IoU is a noisy measurement here")
    if h.get("hot_area_fraction", 0) < 0.005:
        amber.append("almost no contact area")
    return ("RED", red) if red else (("AMBER", amber) if amber else ("GREEN", []))


def thumb(path: Path, width=900) -> str:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return ""
    h = int(img.shape[0] * width / img.shape[1])
    img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(buf).decode() if ok else ""


CSS = """
body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:24px;background:#111;color:#e8e8e8}
h1{font-size:20px;margin:0 0 4px} .sub{color:#999;font-size:13px;margin-bottom:20px}
.take{border:1px solid #333;border-radius:8px;margin-bottom:20px;overflow:hidden;background:#1a1a1a}
.hd{display:flex;align-items:center;gap:14px;padding:10px 14px;font-size:14px;flex-wrap:wrap}
.badge{padding:2px 10px;border-radius:4px;font-weight:700;font-size:12px}
.RED{background:#8b1a1a;color:#fff}.AMBER{background:#8a6100;color:#fff}.GREEN{background:#1e6b34;color:#fff}
.id{font-weight:700;font-size:16px} .m{color:#bbb;font-family:ui-monospace,monospace;font-size:12px}
.why{padding:0 14px 8px;color:#e5a}
.imgs{display:flex;gap:8px;padding:0 14px 14px;flex-wrap:wrap}
.imgs figure{margin:0;flex:1 1 440px} .imgs img{width:100%;border-radius:4px;display:block}
figcaption{font-size:11px;color:#888;padding-top:4px}
.path{color:#6a8fb5;font-family:ui-monospace,monospace;font-size:10px;word-break:break-all}
table{border-collapse:collapse;margin-bottom:24px;font-size:12px;font-family:ui-monospace,monospace}
th,td{padding:3px 9px;border-bottom:1px solid #333;text-align:right}th{text-align:right;color:#999}
td:first-child,th:first-child{text-align:left}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--absolute-paths", action="store_true",
                    help="print absolute source paths under each figure (default: as given, "
                         "so the page does not leak the full local tree when shared)")
    a = ap.parse_args()

    items = []
    for jp in sorted(a.root.rglob("stage4_*.json")):
        d = json.loads(jp.read_text())
        v, why = verdict(d)
        items.append((d, jp, v, why))
    if not items:
        print(f"no stage4_*.json under {a.root}")
        return 1

    order = {"RED": 0, "AMBER": 1, "GREEN": 2}
    items.sort(key=lambda it: (order[it[2]], (it[0].get("alignment") or {}).get("bite_iou_after", 0)))

    rows, blocks = [], []
    for d, jp, v, why in items:
        al, hm = d.get("alignment") or {}, d.get("heatmap") or {}
        tk = Path(d["take"]).name
        variant = jp.parent.name
        rows.append(
            f"<tr><td>{tk}</td><td>{d['side']}</td><td>{d['frame']}</td>"
            f"<td>{al.get('bite_iou_after', float('nan')):.3f}</td>"
            f"<td>{d.get('object_mask_iou', float('nan')):.3f}</td>"
            f"<td>{al.get('pad_min_cm_after', float('nan')):.2f}</td>"
            f"<td>{al.get('penetration_max_cm', float('nan')):.2f}</td>"
            f"<td>{al.get('pads_on_free_surface', float('nan'))*100:.0f}%</td>"
            f"<td>{100*hm.get('hot_area_fraction', float('nan')):.1f}%</td>"
            f"<td><span class='badge {v}'>{v}</span></td></tr>")

        figs = []
        for pat, cap in ((f"stage3_align_frame{d['frame']:04d}_{d['side']}.png",
                          "alignment: red = what the video shows the hand hiding, "
                          "cyan = what SharpaWave hides. They should coincide."),
                         (f"stage4_heatmap_frame{d['frame']:04d}_{d['side']}.png",
                          "heatmap: heat should be one coherent patch, and several "
                          "fingers should own it.")):
            p = jp.parent / pat
            if p.exists():
                b64 = thumb(p, a.width)
                if b64:
                    src = p.resolve() if a.absolute_paths else p
                    figs.append(f"<figure><img src='data:image/jpeg;base64,{b64}'>"
                                f"<figcaption>{cap}<br><span class='path'>{src}</span>"
                                f"</figcaption></figure>")
        blocks.append(
            f"<div class='take'><div class='hd'><span class='badge {v}'>{v}</span>"
            f"<span class='id'>take {tk}</span><span class='m'>{d['side']} hand · frame "
            f"{d['frame']}/{d['Tv']} · {variant}</span>"
            f"<span class='m'>bite IoU {al.get('bite_iou_after', float('nan')):.3f} · "
            f"obj IoU {d.get('object_mask_iou', float('nan')):.3f} · "
            f"pad {al.get('pad_min_cm_after', float('nan')):.2f} cm · "
            f"pen {al.get('penetration_max_cm', float('nan')):.2f} cm · "
            f"neg {al.get('pads_on_free_surface', float('nan'))*100:.0f}% · "
            f"hot {100*hm.get('hot_area_fraction', float('nan')):.1f}%</span></div>"
            + (f"<div class='why'>{' · '.join(why)}</div>" if why else "")
            + f"<div class='imgs'>{''.join(figs)}</div></div>")

    n = {k: sum(1 for it in items if it[2] == k) for k in ("RED", "AMBER", "GREEN")}
    head = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Contact alignment + heatmap review</title>")
    html = (head + f"<style>{CSS}</style></head><body>"
            "<h1>Contact alignment + heatmap review</h1>"
            f"<div class='sub'>{len(items)} takes under {a.root} &mdash; "
            f"<span class='badge RED'>RED {n['RED']}</span> "
            f"<span class='badge AMBER'>AMBER {n['AMBER']}</span> "
            f"<span class='badge GREEN'>GREEN {n['GREEN']}</span> &mdash; worst first</div>"
            "<table><tr><th>take</th><th>side</th><th>frame</th><th>bite IoU</th><th>obj IoU</th>"
            "<th>pad cm</th><th>pen cm</th><th>neg</th><th>hot</th><th></th></tr>"
            + "".join(rows) + "</table>" + "".join(blocks) + "</body></html>")

    out = a.out or (a.root / "contact_gallery.html")
    out.write_text(html)
    print(f"{len(items)} takes | RED {n['RED']} AMBER {n['AMBER']} GREEN {n['GREEN']}")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
