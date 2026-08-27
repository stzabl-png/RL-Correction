"""逐步账本探针: 4桶×4env 确定性回放, 每步录全套奖惩与物理量, 死亡验尸自动化。
输出: dp_trace.npz + 终端验尸报告 (死亡前的量-阈值交叉)。"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=700)
p.add_argument("--out", default="tasks/Pour/17/B_SmokeTest/dp_trace.npz")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("pour17_trace")
app = AppLauncher(args).app
import numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pour_env as PE
from rl_rebuild.algo.ppo.ppo import PPO
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

N = 16
cfg = PE.build_cfg(num_envs=N)
raw = PE.PourEnv(cfg)
labels = [e[4] for e in raw.entries]
buckets = [0, labels.index("seam1"), labels.index("green@87"),
           labels.index("seam2_ret")]
raw.force_entry = [buckets[i % 4] for i in range(N)]
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ppo_pour.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["algorithm"]["num_actors"] = N
agent = PPO(env, output_dir="/tmp/p17tr", full_config=ConfigWrapper(acfg, {}, test=True),
            create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()
obs = env.reset()

FIELDS = ["row", "k", "ms1", "ms2", "ms3", "adv", "leash", "ms_r", "fail_pb",
          "fail_env", "d4_r", "d4_l", "npad_r", "npad_l", "relv_r", "relv_l",
          "dev_cup", "dev_bot", "tilt_bot", "done"]
buf = {f: [] for f in FIELDS}
first_done = [None] * N
with torch.no_grad():
    for t in range(args.steps):
        inp = {"obs": agent.running_mean_std(obs["obs"]),
               "priv_info": obs["priv_info"]}
        mu = agent.model.act_inference(inp)
        obs, rew, dones, infos = env.step(torch.clamp(mu, -1.0, 1.0))
        o = raw._tick_out
        cup, bot = o["cup"], o["bot"]
        d_r = (raw.hand.data.body_pos_w[:, raw.wid["R"]]
               - raw.object.data.root_pos_w).norm(dim=1)
        d_l = (raw.hand.data.body_pos_w[:, raw.wid["L"]]
               - raw.aux.data.root_pos_w).norm(dim=1)
        f = raw._pads_f().norm(dim=-1)
        wlr = raw.hand.data.body_lin_vel_w[:, raw.wid["R"]]
        wll = raw.hand.data.body_lin_vel_w[:, raw.wid["L"]]
        ki = (raw.row.clamp(max=raw.T_ROW - 1) - raw.IA0).clamp(0, raw.PB.N_ROW - 1)
        from isaaclab.utils.math import quat_apply
        qb = bot[:, 3:7] / bot[:, 3:7].norm(dim=1, keepdim=True).clamp(min=1e-9)
        upv = quat_apply(qb, raw.PB.up.unsqueeze(0).expand(N, 3))
        vals = {
            "row": raw.row.clone(), "k": raw.PB.k.clone(),
            "ms1": raw.PB.ms1.clone(), "ms2": raw.PB.ms2.clone(),
            "ms3": raw.PB.ms3.clone(),
            "adv": o["out"]["adv"], "leash": o["out"]["leash"],
            "ms_r": o["out"]["ms"], "fail_pb": o["out"]["fail"],
            "fail_env": o["fail_env"],
            "d4_r": (d_r - raw.grasp_d0[:, 0]),
            "d4_l": (d_l - raw.grasp_d0[:, 1]),
            "npad_r": (f[:, :5] > 0.5).sum(1), "npad_l": (f[:, 5:] > 0.5).sum(1),
            "relv_r": (wlr - raw.object.data.root_lin_vel_w).norm(dim=1),
            "relv_l": (wll - raw.aux.data.root_lin_vel_w).norm(dim=1),
            "dev_cup": (cup[:, :3] - raw.PB.ref_obj[0][ki][:, :3]).norm(dim=1),
            "dev_bot": (bot[:, :3] - raw.PB.ref_obj[1][ki][:, :3]).norm(dim=1),
            "tilt_bot": torch.acos((upv[:, 2] / upv.norm(dim=1).clamp(min=1e-9))
                                   .clamp(-1, 1)),
            "done": dones,
        }
        for fkey in FIELDS:
            buf[fkey].append(vals[fkey].float().cpu().numpy())
        d = dones.cpu().numpy().astype(bool)
        for i in range(N):
            if first_done[i] is None and d[i]:
                first_done[i] = t
        if all(v is not None for v in first_done):
            break
T = len(buf["row"])
arr = {k: np.stack(v, axis=0) for k, v in buf.items()}      # (T, N)
np.savez_compressed(args.out, first_done=np.array(
    [v if v is not None else -1 for v in first_done]),
    buckets=np.array([labels[buckets[i % 4]] for i in range(N)]), **arr)

print("\n===== 逐步验尸 (死亡前关键量) =====")
for i in range(N):
    td = first_done[i]
    if td is None:
        print(f"env{i:2d} [{labels[buckets[i%4]]:10s}] 未终止"); continue
    s = slice(max(td - 5, 0), td + 1)
    print(f"env{i:2d} [{labels[buckets[i%4]]:10s}] 死@t{td} "
          f"k={int(arr['k'][td, i])} ms1={int(arr['ms1'][td, i])} "
          f"fail_pb={int(arr['fail_pb'][td, i])} fail_env={int(arr['fail_env'][td, i])}")
    print(f"   末6步 d4_r(cm)={np.round(arr['d4_r'][s, i]*100, 1)} "
          f"d4_l(cm)={np.round(arr['d4_l'][s, i]*100, 1)}")
    print(f"   垫R={arr['npad_r'][s, i].astype(int)} 垫L={arr['npad_l'][s, i].astype(int)} "
          f"dev_bot(cm)={np.round(arr['dev_bot'][s, i]*100, 1)} "
          f"tilt_bot(°)={np.round(np.degrees(arr['tilt_bot'][s, i]), 0)}")
print(f"[trace] ✅ {args.out}")
try:
    _slot.release()
except Exception:
    pass
app.close()
