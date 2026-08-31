"""squeeze 剂量标定: 4 env 各带 β=1.0/1.5/2.0/3.0, 零动作过提升段, 看谁持住。"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("beta")
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ.pop("POUR_SQUEEZE_FF", None)      # 前馈不带squeeze, 本探针自配剂量
import task_config as TC
import task_env as PE

N = 4
BETA = [1.0, 1.5, 2.0, 3.0]
cfg = PE.build_cfg(num_envs=N)
E = PE.UnscrewEnv(cfg)
E.force_entry = [0] * N
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))
sqr = (np.asarray(np.load(TC.PRIOR_MAIN)["squeeze"], np.float64).reshape(-1)[7:29]
       if TC.PRIOR_MAIN else np.zeros(22))   # [TASK] 盖侧设定B无 squeeze prior
sql = np.asarray(np.load(TC.PRIOR_AUX)["squeeze"], np.float64).reshape(-1)[7:29]
ref = E.ref58.cpu().numpy()
dsq = np.concatenate([sqr - ref[E.IA0, 14:36], sql - ref[E.IA0, 36:58]])
dsq_t = torch.tensor(dsq, dtype=torch.float32, device=dev)

def drive(rows_per_env, sq_scale):
    tgt = torch.zeros(N, 58, device=dev)
    for i in range(N):
        r = min(rows_per_env[i], E.T_ROW - 1)
        tgt[i] = E.ref58[r]
        tgt[i, 14:] += sq_scale[i] * dsq_t
    full = E.hand.data.joint_pos.clone()
    full[:, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    E._update_screw_drive_gain()
    for _ in range(DECI):
        E._SA.apply_screw(E)
        E.scene.write_data_to_sim(); E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())

# 第一幕: approach 全员照谱 (无squeeze); 行号动态取 env (Pour 硬编码已废)
APP, IA0, IA1 = E.APP_END, E.IA0, E.IA1
for r in range(0, APP):
    drive([r]*N, [0.0]*N)
# 缝1: 各自剂量渐入 + 站稳30步
for i2, r in enumerate(range(APP, IA0)):
    a = (i2 + 1) / max(IA0 - APP, 1)
    drive([r]*N, [b * a for b in BETA])
for _ in range(30):
    drive([IA0]*N, BETA)
org = E.scene.env_origins
d0 = (E.hand.data.body_pos_w[:, E.wid["L"]] - E.object.data.root_pos_w).norm(dim=1).clone()
f = E._pads_f().norm(dim=-1)
print("[beta] 站位垫数: " + " ".join(
    f"β{BETA[i]}:L{int((f[i,:5]>0.5).sum())}/R{int((f[i,5:]>0.5).sum())}"
    for i in range(N)), flush=True)
# 交互关键段: 播交互前 60% 行 (覆盖 拿起+转平+拧盖窗; 数据引擎逐 clip 通用)
KEY_END = IA0 + int(0.6 * (IA1 - IA0))
for r in range(IA0, KEY_END):
    drive([r]*N, BETA)
    if r % 10 == 0:
        wz = E.hand.data.body_pos_w[:, E.wid["L"], 2]
        bz_ = E.object.data.root_pos_w[:, 2]
        dd = (E.hand.data.body_pos_w[:, E.wid["L"]] - E.object.data.root_pos_w).norm(dim=1)
        ff = E._pads_f().norm(dim=-1)
        print("[beta] 行%d " % r + " | ".join(
            f"β{BETA[i]}: 腕z={float(wz[i]-org[i,2]):.3f} 瓶z={float(bz_[i]-org[i,2]):.3f} "
            f"距Δ={float((dd[i]-d0[i])*100):+.1f}cm 垫L={int((ff[i,:5]>0.5).sum())}"
            for i in range(N)), flush=True)
d_r = (E.hand.data.body_pos_w[:, E.wid["L"]] - E.object.data.root_pos_w).norm(dim=1)
f = E._pads_f().norm(dim=-1)
held = []
for i in range(N):
    slip = float((d_r[i] - d0[i]) * 100)
    npr = int((f[i, :5] > 0.5).sum())
    bz = float(E.object.data.root_pos_w[i, 2] - org[i, 2])
    # 判读: 关键段末瓶应离桌 (母带该行 z) 且滑移小; 阈值随 clip 从母带取
    ref_z = float(E.PB.ref_obj[0][int(0.6 * (IA1 - IA0))][2])
    ok = slip < 3 and npr >= 2 and bz > ref_z - 0.07
    held.append(ok)
    print(f"[beta] β={BETA[i]}: 滑移={slip:+.1f}cm 左垫={npr} 瓶z={bz:.3f} "
          f"(母带 {ref_z:.3f}) -> {'✅持住了' if ok else '❌脱手'}", flush=True)
ok_any = any(held)
if ok_any and all(held):
    print("[beta] 四档全通过；下次向更低剂量扩展以找到下界", flush=True)
elif not ok_any:
    print("[beta] ❌ 四档全失败；当前剂量窗没有可用值", flush=True)
else:
    print(f"[beta] 可用剂量: {[BETA[i] for i, ok in enumerate(held) if ok]}", flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
sys.stdout.flush()
os._exit(0 if ok_any else 1)
