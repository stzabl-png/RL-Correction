import os, pathlib, subprocess, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import shutil
import msgpack, numpy as np
from magicdexmate.control_link import ControlPublisher
SESS="_dev_pico"
# 录制器是**追加**模式打开文件(生产上是对的,不能覆盖已有数据),
# 所以测试必须先清掉上次留下的目录 —— 否则读到的是新旧两跑拼在
# 一起的流,时间跨度会变成几小时,而断言看起来像产品坏了。
shutil.rmtree(ROOT/'data/sessions'/'_dev_pico', ignore_errors=True)
ctl = ControlPublisher("tcp://*:16595")   # 隔离端口,绝不用生产的 5584
p = subprocess.Popen([str(ROOT/".venv-pico/bin/python"), "-u", "scripts/teleop_pico_producer.py",
                      "--fake", "--body-full", "--control", "tcp://127.0.0.1:16595",
                      # 数据口也必须隔离:默认绑 5581,控制台在跑时会撞死,
                      # 而且发布数据的测试按事故规则一律不碰生产端口
                      "--pub", "tcp://*:16598",
                      ],
                     cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
t0=time.time()
while time.time()-t0 < 12:
    t=time.time()-t0
    ctl.send({"recording": 3.0 < t < 8.0, "session": SESS})
    time.sleep(0.05)
p.terminate(); p.wait(timeout=10); out=p.stdout.read()
print("\n".join("    "+l for l in out.strip().splitlines()[-4:]))
f = ROOT/"data/sessions"/SESS/"pico.msgpack"
ok = f.exists()
print(f"[{'通过' if ok else '失败'}] 文件落盘 {f.name}")
if not ok: sys.exit(1)
rows=list(msgpack.Unpacker(open(f,'rb'), raw=False))
tw=np.array([r['t_us'] for r in rows])/1e6
dur=tw[-1]-tw[0]
print(f"[{'通过' if 3.5<dur<6.0 else '失败'}] 只录了开关打开的那段:{len(rows)} 帧 / {dur:.1f} 秒(开关开了 5 秒)")
print(f"[{'通过' if 'body24' in rows[0] and 'trackers' in rows[0] else '失败'}] 字段完整(含 body24 与 trackers)")
print("\n结论:" + ("全部通过" if 3.5<dur<6.0 and 'body24' in rows[0] else "有失败项"))
