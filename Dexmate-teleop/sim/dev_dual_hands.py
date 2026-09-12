"""--hands both 那条路(单进程双手)的接线验证。真手套要硬件,但接线不用。"""
import pathlib, subprocess, sys, time, os, json
M = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(M)); sys.path.insert(0, str(M/"scripts"))
import zmq, viser
from viser_isaac_mirror import Mirror
p = subprocess.Popen([str(M/".venv/bin/python"), "-u", "scripts/teleop_retarget_dual.py",
                      "--source", "mock", "--motion", "cycle"],
                     cwd=M, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                     env={**os.environ, "PYTHONPATH": ""})
time.sleep(6)
srv = viser.ViserServer(port=8098)
m = Mirror(srv, hand_right="tcp://127.0.0.1:5556", hand_left="tcp://127.0.0.1:5557",
           control="tcp://127.0.0.1:5597", pico="")
t0 = time.time()
while time.time()-t0 < 10:
    m.tick(); time.sleep(0.02)
p.terminate(); p.wait(timeout=8); out = p.stdout.read()
ok = True
crcs = m.hand_state["crc"]
for side in ("right", "left"):
    got = crcs.get(side)
    good = got is not None
    ok &= good
    print(f"[{'通过' if good else '失败'}] {side} 手的 hello 收到了,crc={got}")
n = m.hand_state["n"]
print(f"[{'通过' if n > 100 else '失败'}] 双手 qpos 共收到 {n} 帧")
ok &= n > 100
nm = m.hand_state["names"]
both = nm.get("right") and nm.get("left")
print(f"[{'通过' if both else '失败'}] 两只手的关节名都拿到了 "
      f"({len(nm.get('right') or [])} / {len(nm.get('left') or [])} 个)")
ok &= bool(both)
print("\n结论:" + ("单进程双手接线通" if ok else "有失败项"))
print("\n".join("    "+l for l in out.strip().splitlines()[-3:]))
srv.stop(); os._exit(0 if ok else 1)
