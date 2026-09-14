"""Clean/3 Stage-2 确定性评测 (正式口径): 载 ckpt, N env 从第 0 行起, mu 确定性, 每 env 一回合, 读 env.last_ep 终局摘要。

  SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Clean/3/C_Wiring/eval_clean.py --checkpoint logs/Clean3_task_s42/stage1_nn/last.pth --num_envs 16 --headless
产物: <ckpt目录>/../eval/eval_<tag>.json + 终端表。课程档位 --release_row (默认 RELEASE_MIN=10, 即最终档; 训练中期评测请传当前档)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--release_row", type=int, default=None, help="默认 RELEASE_MIN (最终档)")
p.add_argument("--jitter", type=int, default=0, help="release 抖动行数 (默认 0 = 固定档位)")
p.add_argument("--tag", default=None)
p.add_argument("--seed", type=int, default=2026,
               help="抽样种子 (照 Pour eval10 协议加的; 换种子 = 另一组独立抽样)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
ckpt = os.path.abspath(args.checkpoint)
wj = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "world.json")
world = json.load(open(wj)) if os.path.isfile(wj) else {}
_s2 = world.get("stage2", {})                   # ★旗必须在 import task_config 之前按 world.json 还原
os.environ["CLEAN_S2_CLOCK_CONTACT"] = "1" if _s2.get("S2_CLOCK_NEEDS_CONTACT") else "0"
os.environ["CLEAN_S2_REANCHOR"] = "1" if _s2.get("S2_REANCHOR") else "0"
os.environ["CLEAN_S2_SOFT_REL"] = "1" if _s2.get("S2_SOFT_REL") else "0"
if _s2.get("S2_SOFT_W") is not None: os.environ["CLEAN_S2_SOFT_W"] = str(_s2["S2_SOFT_W"])
os.environ["CLEAN_S2_GATE_HOLD"] = "1" if _s2.get("S2_GATE_HOLD") else "0"
os.environ["CLEAN_S2_PUSH"] = os.environ.get("CLEAN_S2_PUSH_OVERRIDE", "0")   # 正式评测默认无推力 (鲁棒性评测传 CLEAN_S2_PUSH_OVERRIDE=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_config as TC  # noqa: E402
for k, v in TC.PHYS.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("SHARPA_WANDB", "0")
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("clean3_eval")
app = AppLauncher(args).app
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import clean_task_env as CE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if world:
    bad = [f"{k}: 记录={v} 当前={os.environ.get(k)}" for k, v in world.get("physics", {}).items() if os.environ.get(k) != v]
    io = world.get("policy_io", {})
    if (io.get("obs_dim"), io.get("priv_dim"), io.get("act_dim")) != (CE.OBS_DIM, CE.PRIV_DIM, CE.ACT_DIM):
        bad.append(f"policy_io 记录={io} 当前=({CE.OBS_DIM},{CE.PRIV_DIM},{CE.ACT_DIM})")
    for k, v in world.get("stage2", {}).items():
        if v is None or k == "S2_PUSH":
            continue                                                          # 老 world.json 没记的键 = 未验不拦; S2_PUSH 评测期有意关
        cur = getattr(TC, k, None)
        cur = json.loads(json.dumps(cur)) if cur is not None else None      # 元组/字典经 JSON 归一化后再比
        same = (abs(float(cur) - float(v)) < 1e-9) if isinstance(v, (int, float)) and not isinstance(v, bool) and cur is not None else (cur == v)
        if not same:
            bad.append(f"{k}: 记录={v} 当前={cur}")
    if bad and not os.environ.get("CLEAN_IGNORE_WORLD"):
        print("[world] ✗ 与 ckpt 出生世界不符 (CLEAN_IGNORE_WORLD=1 可跳过):\n  " + "\n  ".join(bad), flush=True); os._exit(2)
    print(f"[world] ✅ 与 world.json 匹配 (task={world.get('task')} recipe={world.get('recipe')} 物理={world.get('physics')})", flush=True)
# 抽样种子 (2026-09-11 加, 照 Pour eval10 协议): 不给种子时 10 个 env 的差异只来自 PhysX 逐 env
# 数值发散 —— 不可控也换不了一组独立抽样。这里把 cfg.seed 与 torch/np 都钉死, 种子相同即可复现。
_cfg = CE.build_cfg(args.num_envs)
_cfg.seed = int(args.seed)
torch.manual_seed(args.seed); np.random.seed(args.seed)
raw = CE.CleanTaskEnv(_cfg)
raw.release_row_cur = int(args.release_row if args.release_row is not None else TC.RELEASE_MIN)
TC.RELEASE_JITTER = int(args.jitter)
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "ppo_clean.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["algorithm"]["num_actors"] = args.num_envs
agent = PPO(env, output_dir=os.path.join(os.environ.get("TMPDIR", "/tmp"), "clean3_eval"),
            full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
agent.restore_test(ckpt); agent.set_eval()
N = args.num_envs
done_at = np.full(N, -1, dtype=int)
obs = env.reset()
with torch.no_grad():
    for t in range(raw.T_EP + 2):
        inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
        act = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, rew, dones, infos = env.step(act)
        d = (dones.reshape(-1) > 0).cpu().numpy() if torch.is_tensor(dones) else np.asarray(dones, bool).reshape(-1)
        done_at = np.where(d & (done_at < 0), t, done_at)
        if (done_at >= 0).all():
            break
le = {k: v.cpu().numpy().tolist() for k, v in raw.last_ep.items()}
DIE = {0: "-", 1: "rel", 2: "tilt", 3: "plate_dev", 4: "drop", 5: "table"}
rows = []
print(f"\n[eval_clean] ckpt={ckpt} N={N} release={raw.release_row_cur} jitter={args.jitter}")
print(" env  cert  succ  clock  cov   trav_cm  relp P/S cm   rot P/S °   die   cross  len")
for i in range(N):
    rows.append({k: le[k][i] for k in le})
    print(f"{i:4d}  {int(le['cert'][i])}     {int(le['success'][i])}     {le['clock_frac'][i]:.2f}   {le['coverage'][i]:.2f}  {le['travel_cm'][i]:6.1f}   "
          f"{le['relp_max_plate_cm'][i]:4.2f}/{le['relp_max_sponge_cm'][i]:4.2f}   {le['relrot_max_plate_deg'][i]:4.1f}/{le['relrot_max_sponge_deg'][i]:4.1f}   "
          f"{DIE[int(le['die_kind'][i])]:9s} {le['cross_frac'][i]:.2f}  {int(le['ep_len'][i])}")
def m(k): return float(np.mean(le[k]))
def med(k): return float(np.median(le[k]))
summ = {"checkpoint": ckpt, "num_envs": N, "release_row": raw.release_row_cur, "jitter": args.jitter,
        "cert": m("cert"), "success": m("success"), "clock_done": float(np.mean(np.array(le["clock_frac"]) >= 0.999)),
        "clock_frac_mean": m("clock_frac"), "coverage_med": med("coverage"), "travel_cm_med": med("travel_cm"),
        "relp_max_plate_cm_med": med("relp_max_plate_cm"), "relp_max_sponge_cm_med": med("relp_max_sponge_cm"),
        "relrot_max_plate_deg_med": med("relrot_max_plate_deg"), "relrot_max_sponge_deg_med": med("relrot_max_sponge_deg"),
        "die_frac": float(np.mean(np.array(le["die_kind"]) > 0)),
        "die_kinds": {DIE[k]: int((np.array(le["die_kind"]) == k).sum()) for k in range(6)},
        "cross_frac_mean": m("cross_frac"), "seed": int(args.seed), "episodes": rows}
print(f"[eval_clean] ★ cert={summ['cert']:.3f} success={summ['success']:.3f} clock_done={summ['clock_done']:.3f} clock_frac={summ['clock_frac_mean']:.3f} "
      f"cov_med={summ['coverage_med']:.3f} travel_med={summ['travel_cm_med']:.1f}cm relp_med P/S={summ['relp_max_plate_cm_med']:.2f}/{summ['relp_max_sponge_cm_med']:.2f}cm "
      f"rot_med P/S={summ['relrot_max_plate_deg_med']:.1f}/{summ['relrot_max_sponge_deg_med']:.1f}° die={summ['die_frac']:.2f} {summ['die_kinds']} cross={summ['cross_frac_mean']:.2f}", flush=True)
out_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "eval"); os.makedirs(out_dir, exist_ok=True)
tag = args.tag or f"{os.path.splitext(os.path.basename(ckpt))[0]}_r{raw.release_row_cur}_j{args.jitter}"
with open(os.path.join(out_dir, f"eval_{tag}.json"), "w") as f:
    json.dump(summ, f, indent=1, ensure_ascii=False)
print(f"[eval_clean] → {os.path.join(out_dir, f'eval_{tag}.json')}", flush=True)
sys.stdout.flush()
try: _slot.release()
except Exception: pass
os._exit(0)
