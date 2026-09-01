"""认证失败分解 (Unscrew/17): 当前策略 (确定性 mu) N env 从 t0 跑, 每次 G2 认证判定步记录四个条件的实测值."""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--num_envs", type=int, default=8); p.add_argument("--steps", type=int, default=420)
AppLauncher.add_app_launcher_args(p); args = p.parse_args()
app = AppLauncher(args).app
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "part4", "C_Wiring"))
import task_env as PE
from progress import CERT_RISE, CERT_SLIP, CERT_HOLD, CERT_RAMP
from rl_rebuild.algo.ppo.ppo import PPO
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "part4", "C_Wiring")
N = args.num_envs
cfg = PE.build_cfg(num_envs=N); raw = PE.UnscrewEnv(cfg); raw.force_entry = [0] * N
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
agent_cfg = yaml.safe_load(open(os.path.join(_HERE, "ppo_task.yaml"))); agent_cfg["algorithm"]["num_actors"] = N
agent = PPO(env, output_dir="/tmp/unscrew_certprobe", full_config=ConfigWrapper(agent_cfg, {}, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint); agent.set_eval()
obs = env.reset()
PB = raw.PB; prev_phase = PB.cert_phase.clone(); wz0 = torch.zeros(N, device=raw.device); rows = []
with torch.no_grad():
    for t in range(args.steps):
        inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
        mu = agent.model.act_inference(inp)
        obs, rew, dones, infos = env.step(torch.clamp(mu, -1.0, 1.0))
        ph = PB.cert_phase
        org = raw.scene.env_origins
        wl = raw.hand.data.body_pos_w[:, raw.wid["L"]] - org
        bot = raw.object.data.root_pos_w - org; cap = raw.aux.data.root_pos_w - org
        f = raw._pads_f().norm(dim=-1); npads = (f[:, :5] > 0.5).sum(1)
        started = (prev_phase == 0) & (ph == 1)
        wz0 = torch.where(started, wl[:, 2], wz0)
        checked = (prev_phase == 2) & (ph == 3)
        for i in checked.nonzero().squeeze(1).tolist():
            dzb = float(bot[i, 2] - PB.cert_z0[i, 0]); dzc = float(cap[i, 2] - PB.cert_z0[i, 1])
            rel = float(((wl[i] - bot[i, :3]) - PB.cert_rel0[i]).norm()); dwz = float(wl[i, 2] - wz0[i])
            ok = dzb >= CERT_RISE and dzc >= CERT_RISE and rel < CERT_SLIP and int(npads[i]) >= 3
            rows.append((t, i, dzb, dzc, rel, int(npads[i]), dwz, ok, float(PB.cert_try[i])))
            print(f"[cert] t{t:3d} env{i} try{int(PB.cert_try[i])}: 瓶Δz {dzb*100:+.2f}cm 盖Δz {dzc*100:+.2f}cm (需≥0.5) | 腕-瓶滑移 {rel*100:.2f}cm (需<0.8) | 左垫 {int(npads[i])} (需≥3) | 左腕实际抬 {dwz*100:+.2f}cm (参考+1.5) -> {'PASS' if ok else 'FAIL'}", flush=True)
        prev_phase = ph.clone()
        if bool(dones.all()): break
if rows:
    a = np.array([[r[2], r[3], r[4], r[5], r[6]] for r in rows])
    print(f"[cert] 共 {len(rows)} 次判定, PASS {sum(r[7] for r in rows)} | 中位: 瓶Δz {np.median(a[:,0])*100:+.2f}cm 盖Δz {np.median(a[:,1])*100:+.2f}cm 滑移 {np.median(a[:,2])*100:.2f}cm 垫 {np.median(a[:,3]):.0f} 腕抬 {np.median(a[:,4])*100:+.2f}cm", flush=True)
    print(f"[cert] 失败原因计数: 瓶不升 {int((a[:,0] < CERT_RISE).sum())} | 盖不升 {int((a[:,1] < CERT_RISE).sum())} | 滑移超 {int((a[:,2] >= CERT_SLIP).sum())} | 垫不足 {int((a[:,3] < 3).sum())}", flush=True)
else:
    print("[cert] 没有发生过认证判定 (G1 未立?)", flush=True)
print("[cert] done", flush=True)
app.close(); os._exit(0)
