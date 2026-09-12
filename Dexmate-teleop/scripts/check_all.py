#!/usr/bin/env python3
"""一条命令跑完所有**不需要硬件**的检查。

    cd ~/dexmate/MagicDexMate && .venv/bin/python scripts/check_all.py
    .venv/bin/python scripts/check_all.py --quick     # 只跑秒级的那几项

改完遥操作的任何一环之后跑这个。它把散落的台架脚本收成一个入口 —— 此前它们
分别躺在 sim/ 下,要靠人想起来哪个该跑,而「想起来」不是一种机制。

**不包含的**:任何需要真机、真手套、真相机、或者戴头显的检查。那些在
`workflow_fj.md` 的待办里单列。

每一项都必须是「能对着错误失败」的:只报 exit code,不看输出好不好看。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"
PY_ISAAC = ROOT / ".venv-isaac/bin/python"

# (名字, 命令, 大约多久, 要不要 Isaac)
CHECKS = [
    ("未定义名(全仓)", [PY, "sim/dev_undefined_names.py"], 5, False),
    ("新模块单测", [PY, "-m", "pytest", "tests/test_new_modules.py", "-q"], 10, False),
    ("手部重定向单测", [PY, "-m", "pytest", "tests/test_retarget_pipeline.py", "-q"], 20, False),
    ("PICO 数学单测", [ROOT / ".venv-pico/bin/python", "-m", "pytest",
                    "tests/test_pico_math.py", "-q"], 15, False),
    ("拍手对齐尺子自检", [PY, "scripts/check_clap_sync.py", "--self-check"], 3, False),
    ("限位放松自锁回归", [PY, "sim/dev_relax_latch.py"], 8, False),
    ("腕部微抖尺自检", [PY, "sim/dev_wrist_jitter.py", "--self-check"], 2, False),
    # 须用 .venv-isaac(cuRobo 只装在那里;客户端拿 sys.executable 起求解
    # 子进程)。需要 GPU:没有时模块整体 skip 而非假绿。热 warp 缓存实测
    # 7 秒,冷缓存(重启后首次)约 40 秒到 2 分钟。
    ("cuRobo IK 接口单测", [PY_ISAAC, "-m", "pytest",
                        "tests/test_curobo_ik.py", "-q"], 30, False),
    ("真机姿态回显", [PY, "sim/dev_robot_echo.py"], 10, False),
    ("相机画面进页面", [PY, "sim/dev_cam_view.py"], 20, False),
    ("点云自检(合成)", [PY, "scripts/kinect_pointcloud.py", "--self-check"], 20, False),
    # 相机<->机器人外参标定工具(RobotCamCalib,Vega+Kinect 适配):URDF FK ->
    # 合成 AprilTag 板 -> 检测 -> PnP -> 求解,与已知真值比对;不需要相机/机器人。
    ("相机外参标定自检(合成)", [PY, "RobotCamCalib/extr_calib_vega_kinect.py", "--self-check"], 10, False),
    ("对比台自检", [PY, "sim/viser_verify_selfcheck.py"], 40, False),
    ("桥接台架", [PY, "sim/dev_bridge_sync.py"], 60, False),
    ("单进程双手接线", [PY, "sim/dev_dual_hands.py"], 60, False),
    ("手套分段录制", [PY, "sim/dev_rec_glove.py"], 40, False),
    ("PICO 分段录制", [PY, "sim/dev_rec_pico.py"], 40, False),
    ("点云分段录制", [PY, "sim/dev_rec_cloud.py"], 40, False),
    ("控制台全链路", [PY, "sim/dev_console_smoke.py"], 120, True),
]
QUICK = {"未定义名(全仓)", "新模块单测", "拍手对齐尺子自检", "手部重定向单测",
         "PICO 数学单测", "限位放松自锁回归"}


def live_robot_procs() -> list[str]:
    """已连接真机的进程。在它们面前跑测试套件出过事故:2026-08-10,桥接台架
    把假指令发布在生产端口上,一个订阅着的真机桥接照单执行,机器人被驱动,
    用户按了急停。订阅者不占端口绑定,所以「端口冲突会让测试自动失败」的
    假设不成立 —— 必须显式检查并拒绝。"""
    out = subprocess.run(["pgrep", "-af", "dexmate_bridge|sharpa_real_runner"],
                         capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines()
            if ("--live" in ln and "dexmate_bridge" in ln)
            or "sharpa_real_runner" in ln]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="只跑秒级的那几项")
    ap.add_argument("--only", default="", help="只跑名字里含这个子串的项")
    args = ap.parse_args()

    live = live_robot_procs()
    if live:
        print("⛔ 检测到已连接真机的进程,测试套件拒绝运行(先断开真机再测):")
        for ln in live:
            print("   ", ln)
        return 2

    todo = [c for c in CHECKS
            if (not args.quick or c[0] in QUICK)
            and (not args.only or args.only in c[0])]
    # ROS 的 PYTHONPATH 会把 /opt/ros 的包挤进来,pytest 一 import 就炸(仓库
    # 文档里的命令本来就带 env -u PYTHONPATH)。这里统一清掉,免得每个调用点
    # 各写一遍、漏一个就假失败。
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    print(f"要跑 {len(todo)} 项,预计 {sum(c[2] for c in todo)} 秒左右\n")
    rows = []
    for name, cmd, _, _ in todo:
        t0 = time.time()
        r = subprocess.run([str(x) for x in cmd], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=900)
        dt = time.time() - t0
        ok = r.returncode == 0
        rows.append((name, ok, dt, r))
        print(f"  {'✓' if ok else '✗'} {name:20s} {dt:5.1f}s"
              + ("" if ok else f"   退出码 {r.returncode}"))

    bad = [x for x in rows if not x[1]]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} 项通过")
    for name, _, _, r in bad:
        print(f"\n--- {name} 的最后几行 ---")
        tail = (r.stdout + r.stderr).strip().splitlines()[-12:]
        print("\n".join("    " + x for x in tail))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
