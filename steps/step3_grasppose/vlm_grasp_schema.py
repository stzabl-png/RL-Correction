#!/usr/bin/env python3
"""VLM 双手抓取描述 —— schema v3.2(定稿), 单 take CLI。

从验证会话的 run_vlm_v32.py 抢救入仓(2026-08-11; 原件只活在 scratchpad)。
实测成绩(100 clip, 人工判定): v3.2 = 92.6%, active_hand 95.8%。设计依据全文见
[[vlm-grasp-prior-validated]] 记忆; 三条铁律别破坏:
  1) 角色枚举值必须叫 `support`(叫 stabilize 模型不认, idle 会炸到 59%);
  2) 手指问"数量"不问"身份"(身份自洽率仅 33.7%, 且模板可分性证明不需要);
  3) 不注入 EgoDex 任务名(用户裁定: 那测的是"给答案填表"不是"看懂视频")。

输入: 重建 take(用其 masks/hands 逐帧手 mask 做绿L/橙R轮廓叠加 —— prompt 依赖此标记
判左右, 不能省) + 原视频。输出: <take>/vlm_grasp.json。
服务: GPU7 vLLM(默认 127.0.0.1:8807, 本地跑要隧道+VLM_API_BASE)。

用法:
  python vlm_grasp_schema.py --take <take目录> --video <原视频.mp4> [--out xx.json]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("VLM_API_BASE", "http://127.0.0.1:8807/v1")

HAND = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "enum": ["manipulate", "support", "idle"],
                 "description": "manipulate=主动操作(拧/插/拿起/移动/改变物体状态); "
                                "support=有接触但不主动改变物体(扶住/托举/固定/按住/垫着); "
                                "idle=完全没有碰到任何物体"},
        "target_object": {"type": "string", "description": "这只手接触的物体，中文；没接触填「无」"},
        "target_part": {"type": "string", "description": "接触在物体的哪个部位，中文；没接触填「无」"},
        "n_contact_fingers": {"type": "integer", "enum": [0, 1, 2, 3, 4, 5],
                              "description": "这只手有几根手指真正碰到物体（不含手掌）"},
        "palm_contact": {"type": "boolean", "description": "手掌是否贴到物体"},
        "contact_depth": {"type": "string", "enum": ["fingertip", "pad", "whole_finger", "none"],
                          "description": "指尖点触 / 指腹 / 整根手指包住"},
        "opposition": {"type": "string",
                       "enum": ["thumb_vs_fingers", "fingers_vs_palm", "lateral", "other", "none"]},
        "object_shape": {"type": "string",
                         "enum": ["cylinder", "sphere", "disk", "stick", "ring", "box", "irregular", "none"]},
        "object_scale": {"type": "string", "enum": ["very_small", "small", "medium", "large", "none"],
                         "description": "相对人手掌的大小"},
        "transparency": {"type": "string", "enum": ["transparent", "translucent", "opaque", "none"]},
    },
    "required": ["role", "target_object", "target_part", "n_contact_fingers", "palm_contact",
                 "contact_depth", "opposition", "object_shape", "object_scale", "transparency"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "task_summary": {"type": "string",
                         "description": "一句中文，说明双手共同在完成什么任务，要说清左右手各自的作用"},
        "left": HAND, "right": HAND,
        "evidence": {"type": "string", "description": "一句中文，说明你依据画面中的什么现象做出以上判断"},
    },
    "required": ["task_summary", "left", "right", "evidence"],
    "additionalProperties": False,
}
PROMPT = (
    "这是一段第一人称双手操作视频，画面上已经标出了手部分割：\n"
    "**绿色轮廓 + 字母 L = 左手，橙色轮廓 + 字母 R = 右手。**以此为准判断左右手，不要自行推断。\n"
    "请先看懂整段任务在做什么，再分别描述左手和右手。\n"
    "要点：\n"
    "1) task_summary 先概括双手共同完成的任务，要说清左右手各自的作用。\n"
    "2) 左右手分别填写：角色（主动操作 / 扶住固定托举 / 没参与）、接触的是哪个物体的哪个部位、"
    "**有几根手指**真正碰到物体、手掌有没有贴上去。\n"
    "3) 只描述画面中实际看到的，不要猜测被遮挡的部分；某只手没参与就填 idle 和「无」。\n"
    "4) 所有中文字段一律用中文回答。"
)


def overlay_hands_clip(take: Path, video: Path, out_mp4: Path, max_h: int = 720,
                       stride: int = 4, out_fps: float = 4.0) -> None:
    """绿L/橙R 3px 轮廓叠加(不实心填充 —— 管线自带的 alpha 填充会把手指涂没)。"""
    import cv2

    hd = take / "masks" / "hands" / "frames"
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = None
    for i in range(0, n, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            break
        for side, color, letter in (("left", (0, 200, 0), "L"), ("right", (0, 140, 255), "R")):
            m = cv2.imread(str(hd / f"frame_{i:06d}_masks" / f"{side}_hand_0.png"),
                           cv2.IMREAD_GRAYSCALE)
            if m is None or not (m > 127).any():
                continue
            if m.shape[:2] != img.shape[:2]:
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours((m > 127).astype("uint8"), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, color, 3)
            ys, xs = (m > 127).nonzero()
            cv2.putText(img, letter, (int(xs.mean()), int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        if img.shape[0] > max_h:
            s = max_h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * s), max_h))
        if vw is None:
            vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                                 out_fps, (img.shape[1], img.shape[0]))
        vw.write(img)
    cap.release()
    if vw is None:
        raise RuntimeError(f"没写出任何叠加帧: {video}")
    vw.release()


def ask(clip: Path, timeout: int = 900) -> dict:
    b64 = base64.b64encode(clip.read_bytes()).decode()
    body = {"model": os.environ.get("VLM_MODEL", "vlm"), "temperature": 0.0, "max_tokens": 900,
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {"fps": 2.0},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "bimanual_v32", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64}},
                {"type": "text", "text": PROMPT}]}]}
    req = urllib.request.Request(API_BASE + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return json.loads(d["choices"][0]["message"]["content"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--take", type=Path, required=True, help="重建 take 目录(用其 masks/hands)")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="缺省 <take>/vlm_grasp.json")
    a = ap.parse_args(argv)
    out = a.out or (a.take / "vlm_grasp.json")
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "hands.mp4"
        overlay_hands_clip(a.take, a.video, clip)
        ans = ask(clip)
    out.write_text(json.dumps({"schema": "bimanual_v32", "answer": ans},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[vlm_grasp] {a.take.name}: {ans['task_summary']}")
    for s in ("left", "right"):
        h = ans[s]
        print(f"  {s}: {h['role']} {h['target_object']}/{h['target_part']} "
              f"指数{h['n_contact_fingers']} 掌{h['palm_contact']} {h['contact_depth']}")
    print(f"[vlm_grasp] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
