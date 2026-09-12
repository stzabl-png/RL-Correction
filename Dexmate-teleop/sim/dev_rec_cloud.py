import pathlib, subprocess, sys, time, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import shutil
import msgpack, numpy as np
from magicdexmate.control_link import ControlPublisher
SESS="_dev_cloud"
# 录制器是**追加**模式打开文件(生产上是对的,不能覆盖已有数据),
# 所以测试必须先清掉上次留下的目录 —— 否则读到的是新旧两跑拼在
# 一起的流,时间跨度会变成几小时,而断言看起来像产品坏了。
shutil.rmtree(ROOT/'data/sessions'/'_dev_cloud', ignore_errors=True)
ctl = ControlPublisher("tcp://*:16596")   # 隔离端口,绝不用生产的 5584
p = subprocess.Popen([str(ROOT/".venv/bin/python"), "-u", "scripts/kinect_pointcloud.py",
                      "--source","mock","--mock-cams","2","--record-session","tcp://127.0.0.1:16596"],
                     cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
t0=time.time()
while time.time()-t0 < 14:
    t=time.time()-t0
    ctl.send({"recording": 3.0 < t < 9.0, "session": SESS})
    time.sleep(0.05)
p.terminate(); p.wait(timeout=10); out=p.stdout.read()
print("\n".join("    "+l for l in out.strip().splitlines()[-4:]))
d = ROOT/"data/sessions"/SESS
# 多相机:两台 mock 各有自己的目录与时间戳流
dirs = sorted(d.glob("cloud_MOCK*"))
ok = len(dirs) == 2 and all((d/f"cloud_t_{x.name[6:]}.msgpack").exists() for x in dirs)
print(f"[{'通过' if ok else '失败'}] 两台相机各自落盘:{[x.name for x in dirs]}")
if not ok: sys.exit(1)
files=sorted(dirs[0].glob("*.npz"))
ok = len(files)>0 and len(sorted(dirs[1].glob("*.npz")))>0
print(f"[{'通过' if ok else '失败'}] 每台都有帧(台1 {len(files)},台2 {len(sorted(dirs[1].glob('*.npz')))})")
if not ok: sys.exit(1)
rows=list(msgpack.Unpacker(open(d/f"cloud_t_{dirs[0].name[6:]}.msgpack","rb"), raw=False))
tw=np.array([r["t_wall_us"] for r in rows])/1e6
dur=tw[-1]-tw[0]
hz=(len(rows)-1)/dur if dur>0 else 0
print(f"[{'通过' if 4.0<dur<7.0 else '失败'}] 只录开关打开那段:{dur:.1f} 秒(开了 6 秒)")
print(f"[{'通过' if hz<=11 else '失败'}] 帧率 {hz:.1f} Hz 未超过设定的 10 Hz(限流是上限;取点云本身更慢时达不到设定值是正常的)")
z=np.load(files[0])
mb=sum(f.stat().st_size for f in files)/1e6
print(f"[{'通过' if len(rows)==len(files) else '失败'}] 时间戳与文件数一致 {len(rows)}/{len(files)}")
print(f"      每帧 {z['xyz_mm'].shape[0]} 点,合计 {mb:.1f} MB,码率 {mb/dur:.2f} MB/s")
