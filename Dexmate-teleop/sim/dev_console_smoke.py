#!/usr/bin/env python3
"""控制台冒烟:改完界面之后,「一条命令 -> 启动 -> 四个进程起来 -> 正常停止」
这条路径还能不能走通。不开浏览器,直接驱动 Console 对象,走的是页面按钮
调用的同一批函数。

判据(每条都能失败):
  1. start_stack 之后,勾选的每个子进程都活着;
  2. 仿真进程的日志里出现链路建立的迹象(不是只起了个进程就死);
  3. 停止走的是广播,consumer 日志里要有「收到停止指令,正常退出」;
  4. 停完 CUDA 还能用 —— 用信号杀长跑的 Isaac 会搞坏 nvidia_uvm,这个仓库
     踩过 6 次,所以每次验停止都要顺带验这一条。
"""
import subprocess
import sys
import time
import pathlib

# ⛔ 安全:本冒烟会在生产端口(:5581/:5583/:5584)上发布模拟数据。已连接
# 真机的进程订阅着这些端口,会把模拟数据当真指令执行(2026-08-10 桥接台架
# 在同样的机制下驱动了真机,用户按了急停)。真机进程在,就拒绝运行。
_live = subprocess.run(["pgrep", "-af", "dexmate_bridge|sharpa_real_runner"],
                       capture_output=True, text=True).stdout
_live = [ln for ln in _live.splitlines()
         if ("--live" in ln and "dexmate_bridge" in ln)
         or "sharpa_real_runner" in ln]
if _live:
    print("[冒烟] ⛔ 检测到已连接真机的进程,拒绝运行:")
    for ln in _live:
        print("    " + ln)
    sys.exit(2)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = ["vega_console"]
import vega_console as V  # noqa: E402

ok = True


def check(cond, msg):
    global ok
    print(("[通过] " if cond else "[失败] ") + msg)
    ok = ok and bool(cond)


c = V.Console()
# 页面上的那几个勾:模拟数据 + 双臂 + 头 + Sharpa 手
c.c_fake.value = True
c.c_left.value = c.c_right.value = True
c.c_head.value = True
c.c_waist.value = False
c.c_hand_l.value = c.c_hand_r.value = True
c._apply_rules()
print(f"[冒烟] 勾选 左{c.c_left.value:d} 右{c.c_right.value:d} 头{c.c_head.value:d} "
      f"腰{c.c_waist.value:d} 左手{c.c_hand_l.value:d} 右手{c.c_hand_r.value:d} "
      f"模拟{c.c_fake.value:d}")

c.start_stack()
print(f"[冒烟] start_stack 返回,进程表 {sorted(c.procs)}")

# Isaac 要几十秒才起得来。期间持续 tick,让镜像的广播照常发。
t0 = time.time()
while time.time() - t0 < 150:
    c.mirror.tick()
    time.sleep(0.02)
    if c.mirror.n_arm > 0:
        break
print(f"[冒烟] 等待 {time.time() - t0:.0f} 秒后,镜像收到手臂帧 {c.mirror.n_arm}")

for name in ("producer", "consumer", "glove_r", "glove_l"):
    p = c.procs.get(name)
    check(p is not None and p.alive(), f"{name} 在运行")

check(c.mirror.n_arm > 0, f"镜像收到了手臂数据({c.mirror.n_arm} 帧)")
check(c.mirror.hand_state["n"] > 0,
      f"镜像收到了手部数据({c.mirror.hand_state['n']} 帧)")

# 停止:走广播,不发信号
c.stop_all()
c._drain_shutdown()
cons_log = (ROOT / "logs/console/consumer.log").read_text(errors="replace")
check("正常退出" in cons_log[-4000:], "仿真是收到停止指令后自己退出的,不是被信号杀的")
for name in ("producer", "consumer", "glove_r", "glove_l"):
    check(name not in c.procs or not c.procs[name].alive(), f"{name} 已停止")

print("\n结论:" + ("全部通过" if ok else "有失败项"))
sys.exit(0 if ok else 1)
