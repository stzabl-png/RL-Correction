import pathlib, subprocess, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import viser
from viser_isaac_mirror import Mirror
p=subprocess.Popen([str(ROOT/".venv-pico/bin/python"),"-u","scripts/teleop_pico_producer.py",
                    "--fake","--body-full"],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
time.sleep(2)
srv=viser.ViserServer(port=8099)
m=Mirror(srv, hand_right="", hand_left="", control="tcp://127.0.0.1:5599")
t0=time.time()
while time.time()-t0 < 8:
    m.tick(); time.sleep(0.02)
p.terminate(); p.wait(timeout=5)
print(f"[{chr(0x901A)+chr(0x8FC7) if m.n_pico>50 else chr(0x5931)+chr(0x8D25)}] 镜像收到操作者骨架 {m.n_pico} 帧")
print(f"[{chr(0x901A)+chr(0x8FC7) if m.n_palm>50 else chr(0x5931)+chr(0x8D25)}] 掌心朝向箭头画了 {m.n_palm} 帧")
print(f"[{'通过' if m.g_pico.value.startswith('✅') else '失败'}] 面板读数 = {m.g_pico.value!r}")
srv.stop()
import os; os._exit(0 if m.n_pico>50 else 1)
