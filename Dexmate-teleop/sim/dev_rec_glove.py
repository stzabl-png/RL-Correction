import pathlib, subprocess, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import shutil
import msgpack
from magicdexmate.control_link import ControlPublisher

SESS = "_dev_glove"
# 录制器是**追加**模式打开文件(生产上是对的,不能覆盖已有数据),
# 所以测试必须先清掉上次留下的目录 —— 否则读到的是新旧两跑拼在
# 一起的流,时间跨度会变成几小时,而断言看起来像产品坏了。
shutil.rmtree(ROOT/'data/sessions'/'_dev_glove', ignore_errors=True)
ctl = ControlPublisher("tcp://*:16597")   # 隔离端口,绝不用生产的 5584
p = subprocess.Popen([str(ROOT/".venv/bin/python"), "-u", "scripts/teleop_retarget.py",
                      "--source", "mock", "--hand", "right", "--no-pub",
                      "--control", "tcp://127.0.0.1:16597", "--duration", "12"],
                     cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                     env={**__import__("os").environ, "PYTHONPATH": ""})
t0 = time.time()
while time.time()-t0 < 12:
    t = time.time()-t0
    ctl.send({"recording": 3.0 < t < 8.0, "session": SESS})
    time.sleep(0.05)
p.wait(timeout=20)
out = p.stdout.read()
print("\n".join("    "+l for l in out.strip().splitlines()[-6:]))

f = ROOT/"data/sessions"/SESS/"glove_right.msgpack"
ok = True
if not f.exists():
    print("[失败] 没有写出手套数据文件"); sys.exit(1)
rows = list(msgpack.Unpacker(open(f,"rb"), raw=False))
print(f"[{'通过' if rows else '失败'}] 写出 {len(rows)} 帧,{f.stat().st_size/1024:.0f} KB")
ok &= bool(rows)
r = rows[0]
need = {"t_us","t_wall_us","hand","kp","valid"}
print(f"[{'通过' if need <= set(r) else '失败'}] 字段齐全:{sorted(r)}")
ok &= need <= set(r)
import numpy as np
kp = np.array(r["kp"])
print(f"[{'通过' if kp.shape[-1]==3 and kp.size>=45 else '失败'}] 原始关键点形状 {kp.shape}")
ok &= kp.shape[-1]==3
# 墙钟必须单调且落在这次运行的时间窗内
tw = np.array([x["t_wall_us"] for x in rows])/1e6
print(f"[{'通过' if (np.diff(tw)>=0).all() and abs(tw[0]-(t0+3))<2 else '失败'}] "
      f"墙钟单调,首帧在开录后 {tw[0]-t0:.1f} 秒(期望约 3)")
ok &= (np.diff(tw)>=0).all()
# 只在开关打开的那 5 秒里录
dur = tw[-1]-tw[0]
print(f"[{'通过' if 3.5 < dur < 6.0 else '失败'}] 录制时长 {dur:.1f} 秒(开关只开了 5 秒)")
ok &= 3.5 < dur < 6.0
print("\n结论:" + ("全部通过" if ok else "有失败项"))
sys.exit(0 if ok else 1)
