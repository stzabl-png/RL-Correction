"""自动接触检测(recon-only)—— 从逐帧 2D 手/物 mask 生成分左右手的接触区间。

信号:某只手的 mask 与(膨胀后的)物体 mask 的重叠比例。手贴上物体时该比例
明显 > 0,离开时为 0,天然分左右手(用 left_hand_0 / right_hand_0 两套 mask)。
不需要 MANO 前向,也不需要 hoi_detr 权重,已重建的 take 直接可跑。

产物:take_dir/contact_auto.json,结构与 grasp_annotation.json 一致(便于
phase.auto 直接消费),额外带 method/params/per_frame 供调参与对齐 hoi_detr。

用法:
  python -m phase.detect <take_dir> [--dilation-frac 0.012] [--frac-thr 0.02]
                                    [--min-len 3] [--max-gap 4] [--dry-run]
  # 或批量:对多个 take 目录循环调用。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HAND_FILES = {"left": "left_hand_0.png", "right": "right_hand_0.png"}
METHOD = "mask_adjacency_v1"


def _load_mask(path: Path):
    if not path.is_file():
        return None
    try:
        from PIL import Image
        a = np.asarray(Image.open(path).convert("L"))
    except Exception:
        return None
    return a > 127


def _dilate(mask: np.ndarray, iters: int):
    if iters <= 0:
        return mask
    try:
        from scipy import ndimage as ndi
        return ndi.binary_dilation(mask, iterations=int(iters))
    except Exception:
        # 无 scipy 时用纯 numpy 的 4 邻域膨胀兜底
        out = mask.copy()
        for _ in range(int(iters)):
            p = np.zeros_like(out)
            p[1:, :] |= out[:-1, :]; p[:-1, :] |= out[1:, :]
            p[:, 1:] |= out[:, :-1]; p[:, :-1] |= out[:, 1:]
            out |= p
        return out


def _overlap_frac(hand: np.ndarray, obj_dilated: np.ndarray) -> float:
    """手 mask 落在膨胀物体 mask 内的像素占手面积的比例。"""
    if hand is None or obj_dilated is None:
        return 0.0
    ha = int(hand.sum())
    if ha == 0:
        return 0.0
    return float((hand & obj_dilated).sum()) / ha


def _binarize_clean(flags: np.ndarray, min_len: int, max_gap: int) -> list[list[int]]:
    """逐帧布尔 -> 含端点的 [s,e] 区间,先补 <=max_gap 的空洞,再删 <min_len 的碎段。"""
    f = np.asarray(flags, dtype=bool).copy()
    n = len(f)
    # 补空洞
    i = 0
    while i < n:
        if not f[i]:
            j = i
            while j < n and not f[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                f[i:j] = True
            i = j
        else:
            i += 1
    # 收集区间并删碎段
    segs = []
    i = 0
    while i < n:
        if f[i]:
            j = i
            while j < n and f[j]:
                j += 1
            if (j - i) >= min_len:
                segs.append([int(i), int(j - 1)])
            i = j
        else:
            i += 1
    return segs


def _object_masks_dir(take_dir: Path) -> Path:
    return take_dir / "masks" / "objects" / "frames"


def _hand_masks_dir(take_dir: Path) -> Path:
    return take_dir / "masks" / "hands" / "frames"


def _frame_dir(base: Path, i: int) -> Path:
    return base / f"frame_{i:06d}_masks"


def _num_frames(take_dir: Path) -> int:
    wf = take_dir / "world_fused.npz"
    if wf.is_file():
        try:
            return int(np.load(wf, allow_pickle=True)["num_frames"])
        except Exception:
            pass
    # 回退:数手 mask 帧目录
    hd = _hand_masks_dir(take_dir)
    if hd.is_dir():
        return sum(1 for _ in hd.glob("frame_*_masks"))
    raise ValueError(f"无法确定 num_frames: {take_dir}")


def detect_contact(take_dir, *, dilation_frac: float = 0.012, frac_thr: float = 0.02,
                   min_len: int = 3, max_gap: int = 4, num_frames: int | None = None,
                   fps: float | None = None, object_id: str | None = None) -> dict:
    """对一条 take 做 mask 邻接接触检测,返回 contact_auto.json 的内容 dict。

    object_id=None 时对全部 object_*.png 求并集(旧行为, phase.auto 消费用);
    指定 object_id 则只对该物体判接触 —— 多物体 take 里"右手握的是瓶不是杯",
    并集会把两只手都判成"在接触", 逐物体才分得开(2026-08-10 多物体接线)。"""
    take = Path(take_dir)
    hd, od = _hand_masks_dir(take), _object_masks_dir(take)
    if not hd.is_dir() or not od.is_dir():
        raise FileNotFoundError(f"缺少手/物 mask 目录: hands={hd.is_dir()} objects={od.is_dir()}")
    n = int(num_frames or _num_frames(take))

    # 从 world_fused 拿 fps / video 路径(可选)
    video = None
    if fps is None:
        wf = take / "world_fused.npz"
        if wf.is_file():
            try:
                z = np.load(wf, allow_pickle=True)
                video = str(z["video_id"]) if "video_id" in z.files else None
                if "phase_meta" in z.files:
                    pm = json.loads(str(z["phase_meta"]))
                    fps = pm.get("fps")
            except Exception:
                pass

    dil_iters = None  # 按第一帧图像尺寸算膨胀迭代数(分辨率无关)
    per = {"left": [0.0] * n, "right": [0.0] * n}
    flags = {"left": np.zeros(n, bool), "right": np.zeros(n, bool)}

    for i in range(n):
        # 该帧所有物体 mask 求并集(支持多实例 object_*.png)
        ofd = _frame_dir(od, i)
        obj = None
        if ofd.is_dir():
            names = [f"{object_id}.png"] if object_id else sorted(
                q.name for q in ofd.glob("object_*.png"))
            for nm in names:
                m = _load_mask(ofd / nm)
                if m is None:
                    continue
                obj = m if obj is None else (obj | m)
        if obj is None or obj.sum() == 0:
            continue
        if dil_iters is None:
            diag = float(np.hypot(*obj.shape))
            dil_iters = max(1, int(round(dilation_frac * diag)))
        obj_d = _dilate(obj, dil_iters)

        hfd = _frame_dir(hd, i)
        for side, fname in HAND_FILES.items():
            hand = _load_mask(hfd / fname)
            fr = _overlap_frac(hand, obj_d)
            per[side][i] = round(fr, 4)
            if fr >= frac_thr:
                flags[side][i] = True

    segs = {side: _binarize_clean(flags[side], min_len, max_gap) for side in HAND_FILES}

    return {
        "video": video,
        "num_frames": n,
        "fps": fps,
        "source": "auto",
        "method": METHOD,
        "params": {
            "dilation_frac": dilation_frac, "dilation_iters": dil_iters,
            "frac_thr": frac_thr, "min_len": min_len, "max_gap": max_gap,
        },
        "object_id": object_id,
        "annotations": {"left": segs["left"], "right": segs["right"]},
        "per_frame": per,
    }


AUTO_RESULT_NAME = "contact_auto.json"


def write_contact_auto(take_dir, *, dry_run: bool = False, out_name: str | None = None,
                       **kw) -> dict:
    take = Path(take_dir)
    doc = detect_contact(take, **kw)
    out = take / (out_name or AUTO_RESULT_NAME)
    if not dry_run:
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return doc


def _summary(doc: dict) -> str:
    a = doc["annotations"]
    def _dur(segs):
        return sum(e - s + 1 for s, e in segs)
    return (f"L: {len(a['left'])}段/{_dur(a['left'])}帧 {a['left']}  |  "
            f"R: {len(a['right'])}段/{_dur(a['right'])}帧 {a['right']}  "
            f"(n={doc['num_frames']}, method={doc['method']})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="recon-only 自动接触检测 -> contact_auto.json")
    ap.add_argument("take_dir", nargs="+", help="一个或多个 take 目录")
    ap.add_argument("--dilation-frac", type=float, default=0.012)
    ap.add_argument("--frac-thr", type=float, default=0.02)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="只打印不写文件")
    ap.add_argument("--object-id", default=None,
                    help="只对该物体判接触(如 object_1); 缺省=全部物体并集(旧行为)")
    ap.add_argument("--out-name", default=None,
                    help="输出文件名(缺省 contact_auto.json; 多物体建议 contact_auto_<oid>.json)")
    args = ap.parse_args(argv)
    for td in args.take_dir:
        try:
            doc = write_contact_auto(
                td, dry_run=args.dry_run, dilation_frac=args.dilation_frac,
                frac_thr=args.frac_thr, min_len=args.min_len, max_gap=args.max_gap,
                object_id=args.object_id, out_name=args.out_name)
            print(f"[{'dry' if args.dry_run else 'ok '}] {td}\n      {_summary(doc)}")
        except Exception as e:
            print(f"[err] {td}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
