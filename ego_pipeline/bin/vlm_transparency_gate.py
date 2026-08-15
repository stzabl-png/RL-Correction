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

# QWEN_BASE_URL 是 qwen_client 那条线用的名字; 两个名字指同一个服务, 别让人记两套。
API_BASE = (os.environ.get("VLM_API_BASE") or os.environ.get("QWEN_BASE_URL")
            or "http://127.0.0.1:8807/v1")
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


# ───────────────────────── 分件/合件(Retrieval 判据) ─────────────────────────
# 为什么要问这个: 重建出来的是**单个刚体网格**。若被操作的物体在视频里会一分为多
# (瓶子→瓶身+瓶盖)或多合一, 单刚体在原理上就表达不了这个动作 —— 拧盖的本质就是
# 盖相对瓶身转。这类 take 必须换成**分件资产**(Retrieval), 而不是把重建网格凑合用。
# 实测佐证: clip 0 用单件 SAM3D 网格时 conf_pos 50/rot 23(旋转弃用);
# 换成 CAD 瓶身+瓶盖两件后 conf_pos 82/rot 34(旋转可用)。
PART_VERDICTS = ("separates", "combines", "both", "none")
PART_SCHEMA = {
    "type": "object",
    "properties": {
        "part_change": {"type": "string", "enum": list(PART_VERDICTS),
                        "description": "被操作物体在视频中是否发生分件/合件"},
        "parts": {"type": "array", "items": {"type": "string"},
                  "description": "涉及的部件名, 如 ['瓶身','瓶盖']; none 时给空数组"},
        "confidence": {"type": "string", "enum": ["high", "low"]},
        "evidence": {"type": "string", "description": "在第几段发生、看到哪两部分分开或合上"},
    },
    "required": ["part_change", "parts", "confidence", "evidence"],
}
PART_PROMPT = (
    "看完整段视频。判据只有一条: **被手操作的那个物体, 本身是否发生了部件的分离或合并**。\n"
    "- separates: 原本是一个整体, 过程中被拆成两个及以上可分开的部件"
    "(拧下瓶盖、揭开锅盖、拔出笔帽)\n"
    "- combines: 原本分开的部件被装配成一个整体(把盖拧回瓶子、把笔帽套回去)\n"
    "- both: 先分开又合上(或先合上又分开)\n"
    "- none: 物体始终是一个整体, 只是被拿起/移动/倾倒/放下\n"
    "★ 不算分件的情况: 手挡住物体的一部分; 把物体放进/拿出另一个容器;"
    "倒出液体或内容物; 物体只是转动或变形。\n"
    "★ 判据是**部件之间的相对运动**: 两部分能各自独立移动才叫分开。\n"
    "evidence 里写清是在视频的哪一段、看到哪两个部分分离或合上。"
)


def judge_part_identity(video_path: Path, samples: list, part_names: list[str],
                        fps: float = 1.0) -> dict:
    """红轮廓圈出的实例**是哪个部件**。用于实例数 < 部件数时的指认。

    ★ 为什么不能按大小猜: 只有一个实例时"最大实例配最大部件"会无条件装上大件 ——
      实测 clip 2 的唯一实例是**瓶盖**(mask 4696px / 网格 3.8cm), 却被装成 bottle_body。
      两个实例且面积比够大时排序是可靠的; 否则必须看图指认。
    """
    schema = {
        "type": "object",
        "properties": {
            "part": {"type": "string", "enum": list(part_names) + ["none"],
                     "description": "红色轮廓圈出的那个物体是哪个部件; 都不是则 none"},
            "confidence": {"type": "string", "enum": ["high", "low"]},
            "evidence": {"type": "string", "description": "依据: 形状/大小/在整体中的位置"},
        },
        "required": ["part", "confidence", "evidence"],
    }
    prompt = (
        "视频每帧里有一个用红色粗轮廓线圈出的物体(同一个物体, 不同时刻)。\n"
        "判断它是下列哪个部件, 只能选一个: " + " / ".join(part_names) + " (都不是就选 none)。\n"
        "判据: 形状与相对大小 —— 瓶身是细长的主体, 瓶盖是扁的小圆盘/短圆柱。\n"
        "注意: 只看红色轮廓**圈住的那一块**, 不要被画面里其他物体带偏。\n"
        "evidence 里写清你依据的是什么形状特征。"
    )
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "id.mp4"
        _overlay_clip(video_path, samples, clip)
        return _ask(clip, prompt, fps=fps, schema=schema, schema_name="part_identity")


def judge_part_change(video_path: Path, fps: float = 1.0) -> dict:
    """→ {part_change, parts, confidence, evidence}。直接看原视频, 不需要 mask。"""
    v = _ask(video_path, PART_PROMPT, fps=fps, schema=PART_SCHEMA, schema_name="part_change")
    v["needs_retrieval"] = v.get("part_change") in ("separates", "combines", "both")
    return v


def gate_path(dataset: str, video_id: str) -> "Path":
    """vlm_gate.json 的规范位置。RECON_INTERIM_ROOT 与管线其余部分保持一致。"""
    root = os.environ.get("RECON_INTERIM_ROOT")
    base = Path(root) if root else (Path(__file__).resolve().parents[2]
                                    / "Output/ReconstructOutput/interim")
    return base / dataset / video_id / "vlm_gate.json"


def judge_instances(video: Path, samples: dict, *, dataset: str, video_id: str,
                    policy: str = "v2", force: bool = False,
                    max_instances: int = 4) -> dict:
    """逐实例判材质, **结果缓存在 vlm_gate.json**。→ {inst: verdict}

    ============================ 为什么要缓存 ============================

    同一条视频的材质判定原来有**两个调用点**, 各问一遍 VLM:
      * auto_label_v17a 里的透明门 —— 它要据此把透明实例从 label_prompt 里剔掉
      * vlm_gate_step   —— 它还要判分件(needs_retrieval), 顺带也判了材质

    两次调用花两倍 VLM(每次要把整段视频 base64 传过去), 而且**结果可能不一致**
    (温度虽为 0, 但采样帧不同就会不同) —— 于是"标注时判它透明"和"门里判它不透明"
    可以同时存在, 谁也不知道该信哪个。

    现在: 谁先跑谁写 vlm_gate.json, 后来者读缓存。判定只发生一次。
    force=True 或缺的实例才会真去问。
    """
    gp = gate_path(dataset, video_id)
    doc = {}
    if gp.is_file() and not force:
        try:
            doc = json.loads(gp.read_text())
        except json.JSONDecodeError:
            doc = {}
    cached = (doc.get("objects") or {}) if doc.get("status") == "ok" else {}

    out, asked = {}, 0
    for inst, sm in list(samples.items())[:max_instances]:
        hit = cached.get(inst)
        if hit and not force and "error" not in hit:
            out[inst] = hit
            continue
        v = judge_instance(video, sm)
        v["filter"] = should_filter(v, policy)
        v["sample_frames"] = [f for f, _ in sm]
        out[inst] = v
        asked += 1

    if asked:
        doc.update({"status": "ok", "api": API_BASE, "filter_policy": policy,
                    "objects": {**cached, **out},
                    "note": "只记录不删数据; 过滤与否交下游"})
        gp.parent.mkdir(parents=True, exist_ok=True)
        gp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


class VLMUnavailable(RuntimeError):
    pass


def _ask(video_path: Path, prompt: str, fps: float = 1.0, timeout: int = 600,
         schema: dict | None = None, schema_name: str = "material") -> dict:
    b64 = base64.b64encode(video_path.read_bytes()).decode()
    body = {
        "model": MODEL, "temperature": 0.0, "max_tokens": 400,
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64}},
            {"type": "text", "text": prompt}]}],
        "mm_processor_kwargs": {"fps": fps},
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": schema_name, "schema": schema or SCHEMA}},
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


# 过滤策略。默认 **v2** —— 只剔"空透明"(含只装清水), 装深色/不透明内容物的**保留**。
#
# ★ 这条规则翻过两次, 把经过记下来免得再翻:
#     2026-08-10 v2      用户裁定: 只滤空透明(pour/11 茶瓶实测 conf 87/46 可用)
#     2026-08-13 strict  收紧为"透明一律不用"
#     2026-08-14 **改回 v2**  用户重申: "透明瓶子清水要去掉, 但是有深色液体的还是可以留下"
#
#   改回来的实证依据(2026-08-14 pour 试跑 5 条): strict 会把 pour 任务里"透明瓶装深色
#   液体"这一类系统性剔掉 —— 5 条里 2 条(40%)因此只剩杯子, 倒水动作缺了倒水的那只手,
#   整个任务退化成单手扶杯。而 VLM 的三分类本身判得很准(清水那条给出"透过瓶身可以
#   清晰看到后方的绿色床单和粉色枕头"), 分得开这两类, 不该在过滤这一步把它们合并。
#
FILTER_POLICIES = ("strict", "v2")


def should_filter(verdict: dict, policy: str = "v2") -> bool:
    """→ 该实例是否应被剔除。

    strict(默认): 只要判为透明材质就剔除, 不看置信度也不看装没装东西。
    v2(旧):       只有高置信的"空透明"才剔除; 装了深色内容物的放行。
    """
    m = verdict.get("material")
    if policy == "v2":
        return m == "empty_transparent" and verdict.get("confidence") == "high"
    if policy != "strict":
        raise ValueError(f"未知过滤策略 {policy!r}; 可选 {FILTER_POLICIES}")
    return m in ("empty_transparent", "transparent_with_contents")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--sample", action="append", required=True,
                    help="frame_idx:mask_path, 可重复 2~3 次")
    ap.add_argument("--filter-policy", choices=list(FILTER_POLICIES), default="v2")
    a = ap.parse_args(argv)
    samples = []
    for s in a.sample:
        fi, mp = s.split(":", 1)
        samples.append((int(fi), Path(mp)))
    v = judge_instance(a.video, samples)
    v["filter"] = should_filter(v, a.filter_policy)
    v["filter_policy"] = a.filter_policy
    print(json.dumps(v, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
