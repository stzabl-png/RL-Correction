#!/usr/bin/env python
"""Interactive static-pose recorder for mapping calibration.

2026-07-29 user protocol: operator wears headset + trackers (NO controllers)
and holds static poses. Per take: press Q -> 5 s countdown (get into pose) ->
5 s recording (auto-stop). 3 takes per pose, fresh interaction each take.
The labeled recordings then drive OFFLINE closed-loop mapping tuning - no
more wearing needed.

Frames are saved per-take as raw msgpack concatenation (exact wire bytes,
PicoLogSource-compatible) plus a manifest.json with labels.

Run (producer must be streaming - real SDK, --body-full recommended):
    .venv-pico/bin/python scripts/pose_capture.py --out logs/pose_capture_$(date +%Y%m%d)
"""
import argparse
import json
import os
import pathlib
import select
import sys
import time

import msgpack
import numpy as np
import zmq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from magicdexmate.actions import CAPTURABLE  # noqa: E402
from magicdexmate.sources.pico_source import is_body_joint  # noqa: E402

# the catalogue lives in magicdexmate.actions so the recorder, the joint
# teach bench and the offline analysers can never drift apart on labels
POSES = CAPTURABLE
TRACKER_KEYS = ("LWRIST", "RWRIST", "WAIST")


class Stream:
    """Drains the producer PUB continuously; tracks health + freeze."""

    def __init__(self, sub_addr, want=5):
        self.want = want          # expected raw-tracker count for the health line
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.RCVHWM, 10000)
        self.sock.connect(sub_addr)
        self.last_msg = None
        self.last_recv_wall = 0.0
        self._trk_blob = None
        self._trk_change_wall = 0.0

    def drain(self, sink=None):
        """Pull everything pending. sink: list to append raw bytes to."""
        n = 0
        while True:
            try:
                raw = self.sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            n += 1
            self.last_recv_wall = time.time()
            if sink is not None:
                sink.append(raw)
            try:
                self.last_msg = msgpack.unpackb(raw)
            except Exception:
                continue
            trk = self.last_msg.get("trackers") or {}
            blob = b"".join(np.asarray(trk[k], dtype=np.float64).tobytes()
                            for k in sorted(trk))
            if blob != self._trk_blob:
                self._trk_blob = blob
                self._trk_change_wall = self.last_recv_wall
        return n

    def status(self):
        now = time.time()
        if self.last_msg is None:
            return "等待数据流… (producer 没起或头显没发送)", False
        age_ms = (now - self.last_recv_wall) * 1000.0
        trk = self.last_msg.get("trackers") or {}
        # named keys = body model; serial keys = raw channel; both may
        # coexist (2026-07-29: headset can emit both simultaneously)
        allk = [str(k) for k in trk]
        named = [k for k in allk if is_body_joint(k)]
        serials = [k for k in allk if k not in named]
        if serials and named:
            mode = "raw+body"
        elif serials:
            mode = "raw"
        elif named:
            mode = "body"
        else:
            mode = "-"
        keys = serials if len(serials) >= self.want else named
        frozen_ms = (now - self._trk_change_wall) * 1000.0 if trk else 1e9
        ok = age_ms < 300 and len(keys) >= self.want and frozen_ms < 600
        s = (f"链路 {age_ms:4.0f}ms | trk {len(keys)}/{self.want}[{mode}]"
             f" ({'+'.join(str(k)[-6:] for k in keys[:3]) or '无'})"
             f" | 数值{'活' if frozen_ms < 600 else f'冻结 {frozen_ms/1000:.1f}s'}")
        return s, ok


def wait_key(valid, stream, prompt):
    """Single-key wait with a live health line. Falls back to input() when
    stdin is not a tty (tests/pipes)."""
    if not sys.stdin.isatty():
        while True:
            stream.drain()
            s = input(prompt).strip().lower()
            if s in valid:
                return s
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            stream.drain()
            st, ok = stream.status()
            mark = "✅" if ok else "⚠️ "
            sys.stdout.write(f"\r{mark} {st}   {prompt}   ")
            sys.stdout.flush()
            r, _, _ = select.select([sys.stdin], [], [], 0.15)
            if r:
                ch = sys.stdin.read(1).lower()
                if ch in valid:
                    sys.stdout.write("\n")
                    return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def take_stats(raws):
    """Steadiness check: per-tracker position std over the take (mm).
    Works for both fixed names (Full Body) and raw serials (Object mode)."""
    pos = {}
    for raw in raws:
        try:
            m = msgpack.unpackb(raw)
        except Exception:
            continue
        for k, p in (m.get("trackers") or {}).items():
            pos.setdefault(k, []).append(np.asarray(p[:3], dtype=np.float64))
    out = {}
    for k, v in pos.items():
        if len(v) >= 2:
            out[str(k)[-6:]] = float(
                np.linalg.norm(np.std(np.asarray(v), axis=0)) * 1000)
    return out


def take_verdict(raws, want_trackers, allow_frozen=1):
    """Per-take gate: is the data there, alive, and PLAUSIBLE?

    Three failure modes, and steadiness catches none of them:

    missing   a tracker never arrives (not paired, or truncated upstream).
    frozen    it left the headset's view and holds its last value bit-for-bit,
              which reads as perfectly steady. The 2026-07-29 raw batch died
              this way. One frozen tracker is normal - the waist rides in the
              cameras' blind spot - two means an arm went dark.
    collapsed the body model keeps streaming a pose that is not the operator's.
              It never freezes, so nothing above sees it, and the 2026-07-29
              body batch was only caught days later. It IS visible in the limb
              lengths: PICO holds the upper arm rigid to 0.000 mm but lets the
              elbow->wrist distance range over 128-896 mm, so a forearm far
              from the session norm means the arm is being extrapolated.
    """
    seen, changed, frames = {}, {}, 0
    fore = {"left": [], "right": []}
    for raw in raws:
        try:
            m = msgpack.unpackb(raw)
        except Exception:
            continue
        frames += 1
        trk = m.get("trackers") or {}
        for k, p in trk.items():
            k = str(k)
            p3 = tuple(float(v) for v in p[:3])
            if k in seen and p3 != seen[k]:
                changed[k] = changed.get(k, 0) + 1
            seen[k] = p3
        for side in ("left", "right"):
            e, w = f"{side[0].upper()}_ELBOW", f"{side[0].upper()}_WRIST"
            if e in trk and w in trk:
                fore[side].append(float(np.linalg.norm(
                    np.asarray(trk[e][:3], float) - np.asarray(trk[w][:3], float))))

    if frames == 0:
        return False, ["\u274c 这一段没有收到任何帧"]

    serials = [k for k in seen if not is_body_joint(k)]
    named = [k for k in seen if k not in serials]
    # body mode streams fixed names and no serials; raw mode the opposite
    channel = "raw" if len(serials) >= len(named) else "body"
    have = serials if channel == "raw" else named

    lines, ok, frozen = [], True, []
    if len(have) < want_trackers:
        ok = False
        lines.append(f"\u274c 只看到 {len(have)} 个 [{channel}] 通道条目,"
                     f"期望 {want_trackers} 个")
    for k in sorted(seen):
        n = changed.get(k, 0)
        pct = 100.0 * n / max(frames - 1, 1)
        if pct < 20.0:
            frozen.append(k)
            lines.append(f"   \u2744 {k[-8:]:>8s} 位置只变化 {pct:4.0f}% 的帧 —— 出视野")
        else:
            lines.append(f"   \u2713 {k[-8:]:>8s} 位置变化 {pct:4.0f}% 的帧")
    if len(frozen) > allow_frozen:
        ok = False
        lines.append(f"   \u274c {len(frozen)} 个冻结(最多容忍 {allow_frozen} 个=腰)")
    elif frozen:
        lines.append("   (1 个冻结属正常:腰在摄像头盲区)")

    # collapse check - only the body model has elbows to measure
    for side, v in fore.items():
        if len(v) < 20:
            continue
        a = np.asarray(v)
        lo, hi, sd = a.min() * 1000, a.max() * 1000, a.std() * 1000
        if sd > 30.0:
            ok = False
            lines.append(f"   \u274c {side} 小臂长 {lo:.0f}-{hi:.0f}mm 波动 "
                         f"σ={sd:.0f}mm —— 手臂被模型外推了,这段的手不可信")
        else:
            lines.append(f"   \u2713 {side} 小臂长 {a.mean() * 1000:.0f}mm "
                         f"σ={sd:.0f}mm(骨长守恒,手臂可信)")

    lines.insert(0, ("\u2705 数据完整、在动、且几何合理" if ok
                     else "\u26a0 这一段有问题,建议 R 重录"))
    return ok, lines


def record_segment(stream, countdown, hold, prep="准备"):
    t0 = time.time()
    while True:
        left = countdown - (time.time() - t0)
        if left <= 0:
            break
        stream.drain()
        st, _ = stream.status()
        sys.stdout.write(f"\r⏳ {prep}… {left:3.1f}s   [{st}]      ")
        sys.stdout.flush()
        time.sleep(0.1)
    raws = []
    t0 = time.time()
    while True:
        el = time.time() - t0
        if el >= hold:
            break
        stream.drain(sink=raws)
        st, _ = stream.status()
        sys.stdout.write(f"\r🔴 录制中 {el:3.1f}/{hold:.0f}s 帧数={len(raws)}   [{st}]      ")
        sys.stdout.flush()
        time.sleep(0.05)
    sys.stdout.write("\n")
    return raws


def format_verdict(raws, expect):
    """Gate for the opening test take: right mode (raw serials vs body-model
    names) AND live values (the 2nd session of 2026-07-29 was recorded with a
    frozen estimator - every take bit-identical - this catches both)."""
    keys, moved, body24 = set(), set(), False
    lo, hi = {}, {}
    for raw in raws:
        try:
            m = msgpack.unpackb(raw)
        except Exception:
            continue
        b = m.get("body24") if "body24" in m else m.get(b"body24")
        if b is not None:
            body24 = True
        for k, p in (m.get("trackers") or {}).items():
            k = str(k)
            keys.add(k)
            p3 = np.asarray(p[:3], dtype=np.float64)
            lo[k] = np.minimum(lo.get(k, p3), p3)
            hi[k] = np.maximum(hi.get(k, p3), p3)
    for k in keys:
        if np.linalg.norm(hi[k] - lo[k]) > 0.05:
            moved.add(k)
    named = {k for k in keys
             if "WRIST" in k or "WAIST" in k or "ELBOW" in k}
    serials = keys - named
    if serials and (named or body24):
        mode = "both"          # raw channel AND body model live simultaneously
    elif serials:
        mode = "raw"
    elif named or body24:
        mode = "body"
    else:
        mode = "none"
    probs = []
    if expect == "raw":
        # raw serials present is what matters; body riding along is a bonus
        if len(serials) < 3:
            probs.append(f"裸序列号 tracker 只有 {len(serials)} 个(<3)"
                         f"(头显 Object 模式没生效——系统级 Motion Tracker "
                         f"设置/重启 app)")
        elif len(moved & serials) < 2:
            probs.append(f"裸 tracker 中只有 {len(moved & serials)} 个在动 "
                         f"—— 疑似冻结(挥动追踪器后重测)")
    elif expect == "body":
        if not (named or body24):
            probs.append("无人体模型数据(期望 body)")
        if len(moved) < 2:
            probs.append("数值冻结")
    return (not probs), mode, sorted(keys), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="tcp://127.0.0.1:5581")
    ap.add_argument("--out", default=None, help="output dir")
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--expect", choices=["raw", "body", "any"], default="raw",
                    help="format gate for the opening test take: raw = serial "
                         "trackers (Object mode), body = 24-joint model, "
                         "any = skip the gate")
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--countdown", type=float, default=5.0)
    ap.add_argument("--trackers", type=int, default=5,
                    help="期望的裸追踪器个数;每段录完按此判定是否完整")
    ap.add_argument("--allow-frozen", type=int, default=1,
                    help="容忍几个追踪器全程冻结(默认 1 = 腰,它在摄像头盲区)")
    ap.add_argument("--poses", default=None,
                    help="comma-separated slugs to run (default: all six)")
    args = ap.parse_args()

    out = args.out or time.strftime("logs/pose_capture_%Y%m%d_%H%M%S")
    os.makedirs(out, exist_ok=True)
    manifest_path = os.path.join(out, "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))

    poses = POSES
    if args.poses:
        want = {s.strip() for s in args.poses.split(",")}
        poses = [p for p in POSES if p[1] in want]

    stream = Stream(args.sub, want=args.trackers)
    print(f"[capture] 订阅 {args.sub},输出 {out}/")
    print(f"[capture] 流程:每段 = 按 Q -> 倒数 {args.countdown:.0f}s 摆姿势 -> "
          f"录制 {args.hold:.0f}s 保持 -> 自动停;每个动作 {args.takes} 段。")
    print("[capture] 按键:Q=开始本段  R=重录上一段  X=退出\n")

    if args.expect != "any":
        n_test = 0
        while True:
            print(f"\n=== 测试段:随便挥动双臂 {args.hold:.0f}s,先验数据格式 ===")
            ch = wait_key({"q", "x"}, stream, "就绪后按 Q 开始测试(X 退出)")
            if ch == "x":
                print("[capture] 用户退出。")
                return
            raws = record_segment(stream, args.countdown, args.hold)
            n_test += 1
            with open(os.path.join(out, f"00_format_test_take{n_test}.msgpack"),
                      "wb") as f:
                for raw in raws:
                    f.write(raw)
            ok, mode, keys, probs = format_verdict(raws, args.expect)
            if ok:
                print(f"✅ 格式正确:模式={mode},trackers={keys},数值活。"
                      f"进入正式录制。")
                break
            print("❌ 格式不对:")
            for p in probs:
                print(f"   - {p}")
            ch = wait_key({"q", "f", "x"}, stream,
                          "修好后按 Q 重测;F=无视格式强行继续;X=退出")
            if ch == "x":
                print("[capture] 用户退出。")
                return
            if ch == "f":
                print("⚠️  用户选择强行继续(格式未达标)。")
                break

    for pi, (label, slug, pose_hold) in enumerate(poses, 1):
        hold = pose_hold or args.hold
        ti = 1
        while ti <= args.takes:
            print(f"\n=== 动作 {pi}/{len(poses)}「{label}」 第 {ti}/{args.takes} 段 ===")
            ch = wait_key({"q", "x"}, stream,
                          f"就绪后按 Q 开始(X 退出)")
            if ch == "x":
                print("[capture] 用户退出,已录内容保留。")
                return
            # countdown: operator gets into the pose
            t0 = time.time()
            while True:
                left = args.countdown - (time.time() - t0)
                if left <= 0:
                    break
                stream.drain()
                st, ok = stream.status()
                sys.stdout.write(f"\r⏳ 摆姿势… {left:3.1f}s   [{st}]      ")
                sys.stdout.flush()
                time.sleep(0.1)
            # record
            raws = []
            t0 = time.time()
            while True:
                el = time.time() - t0
                if el >= hold:
                    break
                stream.drain(sink=raws)
                st, _ = stream.status()
                sys.stdout.write(f"\r🔴 录制中 {el:3.1f}/{hold:.0f}s "
                                 f"帧数={len(raws)}   [{st}]      ")
                sys.stdout.flush()
                time.sleep(0.05)
            fname = f"{pi:02d}_{slug}_take{ti}.msgpack"
            with open(os.path.join(out, fname), "wb") as f:
                for raw in raws:
                    f.write(raw)
            stats = take_stats(raws)
            stats_s = "  ".join(f"{k} σ={v:.0f}mm" for k, v in stats.items())
            rate = len(raws) / max(hold, 1e-9)
            print(f"\n💾 已保存 {fname}: {len(raws)} 帧 ({rate:.0f}Hz)  {stats_s}")
            # completeness + liveness, judged NOW so a dead take is re-recorded
            # while the operator is still in the pose
            good, lines = take_verdict(raws, args.trackers, args.allow_frozen)
            for ln in lines:
                print(ln)
            if not stats or min(stats.values(), default=1e9) > 80:
                good = False
                print("⚠️  该段位置抖动异常大——建议重录(R)。")
            manifest.append({"pose": label, "slug": slug, "take": ti,
                             "file": fname, "frames": len(raws),
                             "unix": time.time(), "steady_mm": stats,
                             "verdict_ok": good})
            json.dump(manifest, open(manifest_path, "w"),
                      ensure_ascii=False, indent=1)
            prompt = ("Q=下一段  R=重录本段  X=退出" if good else
                      "⚠ 本段未通过检查:R=重录(建议)  Q=仍然继续  X=退出")
            ch = wait_key({"q", "r", "x"}, stream, prompt)
            if ch == "x":
                print("[capture] 用户退出,已录内容保留。")
                return
            if ch == "r":
                manifest.pop()
                json.dump(manifest, open(manifest_path, "w"),
                          ensure_ascii=False, indent=1)
                continue
            ti += 1
    print(f"\n[capture] 全部完成:{len(manifest)} 段已入 {out}/ (manifest.json)")


if __name__ == "__main__":
    main()
