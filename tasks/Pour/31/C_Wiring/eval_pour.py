"""Pour17 确定性评测 — 关探索噪声 (只用 mu), 全程 t0 口径 (考试分布).

  SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Pour/17/C_Wiring/eval_pour.py \
      --checkpoint logs/Pour17_0/stage1_nn/last.pth --num_envs 256 --headless

成功 = M4 (终局); 逐关率 M1-M4 按"完成回合"精确计, 另报失败谱 (超时/env侧/进度机)。
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512, help="总完成回合数(下限)")
parser.add_argument("--seed", type=int, default=2026,
                    help="扰动抽样种子 (POUR_OBJ_JITTER 的 XY 偏移由 torch RNG 抽); 固定它=九条 run 用同一组偏移, 可配对比较")
parser.add_argument("--entry", type=str, default="0",
                    help="出生点(逗号分隔), 默认 0=全程t0(官方考试口径)。"
                         "给多个则逐个考、出逐出生点分解表(诊断用)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("pour17_eval")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import world_fingerprint as _WF  # noqa: E402
_WF.restore_physics_env(_WF.world_json_of(args.checkpoint))  # L5-34: 先还原再建环境
import pour_env as PE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
cfg = PE.build_cfg(num_envs=args.num_envs)
import numpy as _np  # noqa: E402
import random as _random  # noqa: E402
_random.seed(args.seed); _np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
if hasattr(cfg, "seed"):
    cfg.seed = args.seed
print(f"[eval] 扰动种子 seed={args.seed} | POUR_OBJ_JITTER={os.environ.get('POUR_OBJ_JITTER', '0')}", flush=True)
raw = PE.PourEnv(cfg)
raw.force_entry = [0]                       # 考试分布 = 全程 t0
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "ppo_pour.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs
agent = PPO(env, output_dir="/tmp/pour17_eval", full_config=ConfigWrapper(agent_cfg, {}, test=True),
            create_output_dir=False)
agent.restore_test(args.checkpoint)
import world_fingerprint as WF  # noqa: E402
# ★世界核对硬闸 (L5-12): ckpt 的出生世界 = 它同目录的 world.json。
# 不核对就回放 = 拿旧记忆在新几何上走位, 实测同类事故 0/64 且 obs 维度一字未变。
_wj = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(args.checkpoint))), "world.json")
if not os.environ.get("POUR_IGNORE_WORLD"):
    WF.assert_match(raw, _wj, strict=True)
else:
    print("[world] ⚠ POUR_IGNORE_WORLD=1 已跳过世界核对", flush=True)

agent.set_eval()

N = args.num_envs
ENTRIES = [int(x) for x in args.entry.split(",") if x.strip() != ""]


def _run_one(k):
    """把全部 env 钉在出生点 k 上考一轮, 返回 (干净 rates, 初始复位注入的假回合数)。

    ★ 初始复位会污染分母: env.reset() 走 _reset_idx ⟹ PB.reset_idx 把当时全部 env
    当成"结算的回合"计进 _acc (且四个 Gate 全 False)。旧版评测只 reset 一次就开跑,
    这 N 个零成功假回合一直留在分母里 —— num_envs=256/episodes=512 时把成功率
    压到真值的 512/768 = 0.667 倍。这里 reset 后先 pop 一次把它倒掉, 并把倒掉的
    数量原样报出来, 让"旧口径"可复算、可对照。
    """
    raw.force_entry = [k]
    obs = env.reset()
    inject = float(raw.PB.pop_rates()["n/ep_done"])     # 倒掉初始复位注入的假回合
    done_n = 0
    batches = []            # ★ 每一步结算了几个回合
    with torch.no_grad():
        while done_n < args.episodes:
            inp = {"obs": agent.running_mean_std(obs["obs"]),
                   "priv_info": obs["priv_info"]}
            mu = agent.model.act_inference(inp)
            obs, rew, dones, infos = env.step(torch.clamp(mu, -1.0, 1.0))
            nd = int(dones.sum())
            if nd:
                batches.append(nd)
            done_n += nd
    return raw.PB.pop_rates(), inject, batches


def _report_batches(k, b):
    """★ 有效样本量: `--episodes 512` 未必等于 512 个独立样本。

    评测把 force_entry 钉死、obj_jitter_xy=0、策略取均值 ⟹ 全部 env 的初始状态
    **逐位相同**, 于是它们会在几乎同一步同时结算。这样"512 个回合"其实是少数几
    **批**同一件事的复现, 二项标准误 sqrt(p(1-p)/512) 完全不适用 —— 它会把不确定
    度报小一个数量级, 让人以为 ±2 个点, 实测 run 间却晃 6 个点。

    用逆辛普森指数量"等效批数": n_eff = (Σn)² / Σn²
      · 全部一次结算完 -> n_eff = 1
      · 均分成两批     -> n_eff = 2
      · 一个一个陆续来 -> n_eff = 回合数 (才是真的 512 个独立样本)
    """
    tot = sum(b)
    n_eff = (tot ** 2) / sum(x * x for x in b) if b else float("nan")
    top = sorted(b, reverse=True)[:5]
    print(f"[batch] 出生点 {k}: 共 {tot} 回合, 分 {len(b)} 步结算; "
          f"最大的几批 ={top}; 前3批占 {100*sum(top[:3])/max(tot,1):.1f}%")
    print(f"[batch]   ★等效独立批数 n_eff = {n_eff:.1f}  "
          f"(理想=回合数 {tot}; =1~3 则'512回合'实为少数几批的复现)")
    print(f"[batch]   按 n_eff 折算的成功率标准误 ≈ {50.0/max(n_eff,1)**0.5:.1f} 个点 "
          f"(对比按 {tot} 回合的二项口径 {50.0/max(tot,1)**0.5:.1f} 个点)", flush=True)
    return n_eff


results = {}
for _k in ENTRIES:
    print(f"\n[eval] ===== 出生点 {_k} " +
          ("(t0 = 从头做完整任务, 官方考试口径)" if _k == 0 else "(预置出生, 诊断口径)")
          + " =====", flush=True)
    _rr, _ij, _bt = _run_one(_k)
    results[_k] = (_rr, _ij)
    print("[eval] " + " ".join(f"{a}={b:.3f}" for a, b in _rr.items()), flush=True)
    _report_batches(_k, _bt)

print("\n" + "=" * 78)
print("[eval] 逐出生点分解 (每格 = 该出生点下的通关率)")
print("=" * 78)
print(f"{'出生点':>6} {'回合':>6} {'G1':>7} {'G2':>7} {'G3倒水':>7} {'放回':>7}"
      f" {'★成功':>7} {'G4':>7} {'认证':>7}   {'G3松':>7}")
for _k in ENTRIES:
    _r, _inj = results[_k]
    _n = _r["n/ep_done"]
    print(f"{_k:>6} {_n:>6.0f} {_r['sr/gate1']:>7.3f} {_r['sr/gate2']:>7.3f}"
          f" {_r['sr/gate3']:>7.3f} {_r['sr/placed']:>7.3f}"
          f" {_r['sr/success']:>7.3f} {_r['sr/gate4']:>7.3f}"
          f" {_r['sr/cert_pass']:>7.3f}   {_r['sr/g3_loose']:>7.3f}")
print("★ 主判据 = ★成功 = G3倒水 ∧ 放回 (L5-32, 2026-08-31)。两项都是绝对量,")
print("  不读参考轨迹形状 ⟹ 六条臂用的是同一把尺, 可以直接横向比。")
print("★ G3松 = 旧口径(3D距<=12cm, 不分上下)的影子。**G3松 − G3倒水 = 旧口径里")
print("  被'横杵在杯子旁边'刷出来的那部分** —— 差得越多, 该 ckpt 的历史数字越虚。")
_r0 = results.get(0, (None, None))[0]
if _r0 is not None:
    print(f"\n[eval] ★Success(G3_pour ∧ placed, t0 口径) = {_r0['sr_t0/success']:.4f}"
          f"   (t0 分母 {_r0['n/ep_done_t0']:.0f} 回合)")
    print(f"[eval]   分解: G3倒水 {_r0['sr_t0/gate3']:.4f} · 放回 {_r0['sr_t0/placed']:.4f}"
          f" · G3松 {_r0['sr_t0/g3_loose']:.4f} · 走完时钟 {_r0['sr_t0/clock_done']:.4f}")
    print(f"[eval]   撤离 G4 {_r0['sr_t0/gate4']:.4f} (**不在主判据内**, 用户 2026-08-30 裁定不做撤离)")
print("=" * 78, flush=True)

try:
    _slot.release()
except Exception:
    pass
app.close()
