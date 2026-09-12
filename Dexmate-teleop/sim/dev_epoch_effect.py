#!/usr/bin/env python3
"""「开始新一段」到底降不降顶限位 —— 这是第 5 条的判据,必须量。

回放一段素材两次:一次不开新段(基线),一次每 15 秒开一次新段。两次都导出
关节 CSV,再用 sim/dev_limit_lock.py 量「贴限位的帧占比」与「最长连续贴限位」。

回放模式下没有控制台,所以这里自己起一个开关频道发布者,按固定间隔递增
epoch —— 与页面点按钮是同一条通路。
"""
import pathlib
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from magicdexmate.control_link import ControlPublisher  # noqa: E402

S = pathlib.Path(__file__).resolve().parent
CLIP = "logs/playlist_all13.msgpack"
DUR = 140
EVERY = 15.0

BASE = ("--source replay --replay-file " + CLIP +
        " --mode trackers --tracker-left LWRIST --tracker-right RWRIST"
        " --tracker-waist WAIST --map chest-anchor --ori-mode palm"
        " --control-mode arms+head+waist --waist-mode j3 --pos-scale 1.0"
        " --physics-device cpu --physics-hz 50 --control_hz 50 --no-sym-lock"
        " --headless --duration %d" % DUR)


def run(tag: str, epochs: bool) -> pathlib.Path:
    csv = S / f"epoch_{tag}.csv"
    env = {"OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONUNBUFFERED": "1",
           "VEGA_WELD_WHEELS": "1", "VEGA_TORSO_HOME": "legacy"}
    import os
    stop = threading.Event()

    def pub():
        ctl = ControlPublisher("tcp://*:5584")
        n, t0 = 0, time.time()
        while not stop.is_set():
            if epochs and time.time() - t0 > EVERY * (n + 1):
                n += 1
            ctl.send({"epoch": n})
            time.sleep(0.05)

    th = threading.Thread(target=pub, daemon=True)
    th.start()
    cmd = (f"{ROOT}/.venv-isaac/bin/python -u sim/teleop_vega_pico.py {BASE} "
           f"--joint-csv {csv}")
    p = subprocess.run(cmd, shell=True, cwd=ROOT, env={**os.environ, **env},
                       capture_output=True, text=True, timeout=1800)
    stop.set()
    th.join(timeout=2)
    out = p.stdout + p.stderr
    n_ep = out.count("[epoch]")
    summ = [ln for ln in out.splitlines() if "teleop epochs started" in ln]
    print(f"[{tag}] 求解器归位次数 {n_ep};{summ[0].strip() if summ else '没有 summary 行'}")
    if epochs and n_ep == 0:
        print(f"[{tag}] !!! 广播了新段号但求解器一次都没归位 —— 接线没生效,"
              f"下面的对比无效")
    return csv


if __name__ == "__main__":
    print(f"素材 {CLIP},{DUR} 秒;新段每 {EVERY:.0f} 秒一次")
    a = run("off", False)
    b = run("on", True)
    time.sleep(11)          # dev_limit_lock 拒绝分析 10 秒内被写过的文件
    subprocess.run([f"{ROOT}/.venv/bin/python", "sim/dev_limit_lock.py",
                    str(a), str(b)], cwd=ROOT)
