#!/usr/bin/env python3
"""VLM 透明门 —— 重建开始前逐实例判材质, 只过滤"空透明"(规则 v2, 2026-08-10)。

设计要点(与 auto_label 的讨论结论一一对应):
  * 不为 VLM 单独跑 mask —— 复用 v17A 已有的实例 mask;
  * 不抠图 —— 判透明恰恰需要看到物体背后的背景, 用整帧 + 红色轮廓线做"视觉指代",
    答案天然绑定实例 ID, 不经过"名字→实例"的脆弱匹配;
  * 跨时间抽 2~3 帧(空透明 vs 装了液体需要看内容物状态), 拼成 1fps 小视频喂服务
    (GPU7 Qwen3.5 部署只吃 video_url, 见 ~/bin/vlm_client.py);
  * 按已验证的 VLM 纪律: schema+enum+强制说依据+关思维链, temperature 0;
  * 只有 empty_transparent 且 confidence=high 才过滤 —— 拿不准就放进来让 confidence 打分。

校准基线(2026-08-10, 四案例全对才算通过):
  3_scene 空透明瓶 → empty_transparent;  pour/11 茶瓶 → transparent_with_contents;
  2_scene 金属瓶 → opaque;              pour/11 灰杯 → opaque。

CLI(校准/调试):
  python3 vlm_transparency_gate.py --video x.mp4 --sample 10:/path/mask.png --sample 80:/m.png
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("VLM_API_BASE", "http://127.0.0.1:8807/v1")
MODEL = os.environ.get("VLM_MODEL", "vlm")

VERDICTS = ("empty_transparent", "transparent_with_contents", "opaque")
SCHEMA = {
    "type": "object",
    "properties": {
        "material": {"type": "string", "enum": list(VERDICTS),
                     "description": "红色轮廓圈出的物体的材质判定"},
        "confidence": {"type": "string", "enum": ["high", "low"]},
        "evidence": {"type": "string",
                     "description": "判定依据: 能否透过物体看到背景/内容物颜色/标签覆盖情况"},
    },
    "required": ["material", "confidence", "evidence"],
}
PROMPT = (
    "视频每一帧里都有一个用红色粗轮廓线圈出的物体(同一个物体, 不同时刻)。只针对这个物体判定, 三选一。\n"
    "核心判据只有一条: **能否透过物体主体看到它背后的东西**。\n"
    "- empty_transparent: 能看穿 —— 透明容器, 空的、或只装无色透明液体(清水也算此类!\n"
    "  清水不遮挡视线), 透过瓶身能看到背景/桌面\n"
    "- transparent_with_contents: 容器本身透明, 但装有**深色或不透明**的内容物(茶/咖啡/\n"
    "  牛奶等有颜色的液体), 或被大面积不透明标签包裹 —— 物体主体大部分看不穿\n"
    "- opaque: 材质本身不透明(金属/陶瓷/不透明塑料等)\n"
    "注意: 反光高光不等于透明; 液面存在不等于有内容物 —— 无色清水一律算 empty_transparent。\n"
    "把判定依据写进 evidence(能否看穿主体/内容物颜色/标签覆盖)。"
)


class VLMUnavailable(RuntimeError):
    pass


def _ask(video_path: Path, prompt: str, fps: float = 1.0, timeout: int = 600) -> dict:
    b64 = base64.b64encode(video_path.read_bytes()).decode()
    body = {
        "model": MODEL, "temperature": 0.0, "max_tokens": 400,
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64}},
            {"type": "text", "text": prompt}]}],
        "mm_processor_kwargs": {"fps": fps},
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "material", "schema": SCHEMA}},
    }
    req = urllib.request.Request(API_BASE + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise VLMUnavailable(f"VLM 服务不可达({API_BASE}): {e}") from e
    return json.loads(out["choices"][0]["message"]["content"])


def _overlay_clip(video: Path, samples: list[tuple[int, Path]], out_mp4: Path,
                  max_h: int = 720) -> None:
    """整帧 + 红色轮廓 → 1fps 小视频。不抠图: 透明判定需要物体背后的背景。"""
    import cv2

    cap = cv2.VideoCapture(str(video))
    frames = []
    for fi, mp in samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if not ok:
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours((m > 127).astype("uint8"), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (0, 0, 255), 4)
        if img.shape[0] > max_h:
            s = max_h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * s), max_h))
        frames.append(img)
    cap.release()
    if not frames:
        raise RuntimeError(f"没抽到任何叠加帧: {video} {samples}")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def judge_instance(video: Path, samples: list[tuple[int, Path]],
                   workdir: Path | None = None) -> dict:
    """samples: [(frame_idx, mask_path), ...] 2~3 个跨时间样本。返回 schema dict。"""
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        clip = Path(td) / "gate.mp4"
        _overlay_clip(video, samples, clip)
        return _ask(clip, PROMPT)


def should_filter(verdict: dict) -> bool:
    """规则 v2: 只有高置信的空透明才过滤; 拿不准放进来让 confidence 打分。"""
    return (verdict.get("material") == "empty_transparent"
            and verdict.get("confidence") == "high")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--sample", action="append", required=True,
                    help="frame_idx:mask_path, 可重复 2~3 次")
    a = ap.parse_args(argv)
    samples = []
    for s in a.sample:
        fi, mp = s.split(":", 1)
        samples.append((int(fi), Path(mp)))
    v = judge_instance(a.video, samples)
    v["filter"] = should_filter(v)
    print(json.dumps(v, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
