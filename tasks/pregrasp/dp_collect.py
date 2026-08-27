"""AAG-F 策略确定性 rollout → DP staging (双手贴合闭环数据, 2026-08-28 用户拍板)。

零改动挂钩: 原样 runpy 执行 train.py(AAGF 原始命令原件逐旗喂入, 构造保真100%),
仅猴补 PPO.train 为采集循环。ckpt 显式必填且拒绝 best.pth(训过头指纹, RL_Training
核验 eval_best.pth ≡ 6.6M 首个确定性100%)。

用法:
  SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=. $PY -u tasks/pregrasp/dp_collect.py \
    --ckpt results/AAG_pour17_solved/eval_best.pth --n_target 128 --num_envs 64 --headless
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", required=True)
_ap.add_argument("--out_dir", default="tasks/Pour/17/B_SmokeTest/dp_staging_aagf")
_ap.add_argument("--n_target", type=int, default=128)
_ap.add_argument("--num_envs", type=int, default=64)
_ap.add_argument("--pf_ep", type=int, default=128, help="预检回合数(闸口径, 非1024对等)")
_ap.add_argument("--headless", action="store_true")
_a, _rest = _ap.parse_known_args()
assert os.path.basename(_a.ckpt) != "best.pth", \
    "拒绝 best.pth: 按训练奖励选的, 极可能是13.1M训过头权重 (RL_Training 指纹核验)"
assert os.path.exists(_a.ckpt), _a.ckpt
os.environ["DPC_CKPT"] = os.path.abspath(_a.ckpt)
os.environ["DPC_OUT"] = os.path.abspath(_a.out_dir)
os.environ["DPC_NTARGET"] = str(_a.n_target)
os.environ["DPC_PFEP"] = str(_a.pf_ep)
os.environ.setdefault("RL_HAND_JOINTS", "1")
os.environ.setdefault("RL_FC_PLAY_REF", "1")      # ★语义闸: 设0会静默移FC目标6cm

# ---- AAGF_pour17 原始命令原件 (transcript 挖出, 去训练专属旗) ----
AAGF_ARGV = [
    "train.py", "--headless",
    "--name", "DPCOLLECT_aagf", "--clip", "Pour17_bottle",
    "--num_envs", str(_a.num_envs), "--seed", "42",
    "--prior_npz", "tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz",
    "--prior_yaw", "19.5",
    "--bimanual", "--bi_native",
    "--prior_b", "tasks/pregrasp/priors/Pour17_cup_thumbfix.npz",
    "--prior_b_yaw", "90",
    "--curobo_ref", "tasks/pregrasp/priors/aag_pour17_ref_side05.npz",
    "--ff_freeze_cm", "0", "--dyn_far", "2.0",
    "--arm_abs", "--arm_abs_dev_deg", "2.86",
    "--pregrasp_only", "--phase2", "--fin_cart", "--fc_ref_end",
    "--fin_pot", "5.0", "--fcd", "--aag",
    "--grasp_cent_fix", "--eps_pos_cm", "1.35", "--pad_gate",
    "--fin_gate", "--fin_gate_dpos_cm", "3.0", "--c0_min", "0.85",
    "--fin_fade", "--fc_earn", "--cent_earn", "--seg_gate",
    "--pre_close", "0.03", "--pre_close_dead_cm", "1.0",
    "--pre_close_tilt_deg", "5.0", "--obj_obs",
    "--fin_dev_scale", "2.0", "--qd_soft_arm", "4.0", "--qd_soft_fin", "14.0",
    "--qvel_settle", "--form_pot", "2.0", "--arrive_bonus", "8.0",
    "--w_toppled", "2.0", "--direct_grasp_prob", "0", "--dgp_rsi_hi", "0",
    "--approach", "--minimal",
]


def _collect(self):
    import json

    import numpy as np
    import torch
    from isaaclab.utils.math import quat_apply, quat_conjugate

    from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER
    from tasks.pregrasp import bimanual as BM

    ckpt = os.environ["DPC_CKPT"]
    out_dir = os.environ["DPC_OUT"]
    n_target = int(os.environ["DPC_NTARGET"])
    pf_ep = int(os.environ["DPC_PFEP"])
    # ---- 维度闸 (首道): 旗集漂移在此炸响, 不静默 ----
    assert tuple(self.obs_shape) == (348,), \
        f"obs {self.obs_shape} != (348,) — 旗集与 ckpt 不符, 停"
    self.restore_test(ckpt)
    self.set_eval()
    print(f"[dpc] ckpt={os.path.basename(ckpt)} 维度闸过(348) restore_test OK",
          flush=True)

    raw = self.env.unwrapped
    N = raw.num_envs
    jn = list(raw.hand.joint_names)
    ids = [jn.index(f"{P}_arm_j{i}") for P in ("R", "L") for i in range(1, 8)]
    for s in ("right", "left"):
        ids += [jn.index(n.replace("right_", f"{s}_")) for n in GENERIC_JOINT_ORDER]
    map_t = torch.tensor(ids, dtype=torch.long, device=raw.device)
    bn = list(raw.hand.body_names)
    wid = {"R": bn.index("right_hand_C_MC"), "L": bn.index("left_hand_C_MC")}
    org = raw.scene.env_origins
    SQ = float(raw.cfg.squeeze_f0)
    # 侧名探测
    try:
        with BM.use_side(raw, "right"):
            pass
        SIDES = ("right", "left")
    except Exception:
        SIDES = ("A", "B")
    print(f"[dpc] N={N} 侧名={SIDES}", flush=True)

    def read_state():
        q58 = raw.hand.data.joint_pos[:, map_t]
        parts = [q58]
        for s in ("R", "L"):
            wp = raw.hand.data.body_pos_w[:, wid[s]] - org
            wq = raw._qsign(raw.hand.data.body_quat_w[:, wid[s]])
            parts += [wp, wq]
        objs, tacs = [], []
        for si, s in enumerate(SIDES):
            with BM.use_side(raw, s):
                op = raw.object.data.root_pos_w - org
                oq = raw._qsign(raw.object.data.root_quat_w)
                F = torch.cat([x.data.force_matrix_w.view(N, 1, 3)
                               for x in raw._contact_sensors[:5]],
                              dim=1).nan_to_num(0.0)
            wq_s = raw._qsign(raw.hand.data.body_quat_w[:, wid["R" if si == 0 else "L"]])
            qin = quat_conjugate(wq_s).unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
            tacs.append((quat_apply(qin, F.reshape(-1, 3)).reshape(N, 15) / SQ)
                        .clamp(-3, 3))
            objs.append(torch.cat([op, oq], dim=1))
        # state116 布局: 关节58 + 腕R7+腕L7 + 杯7+瓶7 + 触R15+触L15
        # (SIDES[0]=right侧=瓶, SIDES[1]=left侧=杯 —— 布局按 杯,瓶 序放)
        st = torch.cat(parts + [objs[1], objs[0], tacs[0], tacs[1]], dim=1)
        return st.float().cpu().numpy()

    common = {"schema_state": "state116_bimanual_v1",
              "schema_action": "act58_bimanual_v1", "hz": 20.0,
              "quat": "wxyz,w>=0", "units": "m/rad",
              "source": f"policy_rollout_closedloop_aagf ({os.path.basename(ckpt)}"
                        " ≡ 6.6M 首个确定性100%)",
              "task": "pour17_bimanual_pregrasp_aagf",
              "world": os.environ.get("DEXMATE_FIXED_USD", "current_stance"),
              "success_rule": "env自身双手g2成功事件(newly_success, 腕1.35cm/15°+五指尖FC<1cm 保持5步)",
              "tactile": "双侧真实读数, wrist-frame, /squeeze_f0, clamp±3",
              "note_terminal": "终止步 action 因 auto-reset 污染, 每集裁去末行",
              "scope_note": ("闭环策略数据(含误差修正样本+双侧真实触觉)。"
                            "边界: AAG-F 体制无抬升 —— 测双手贴合流, 不外推抓起/任务能力")}
    os.makedirs(out_dir, exist_ok=True)
    bufS = [[] for _ in range(N)]
    bufA = [[] for _ in range(N)]
    saved, ep_done, ep_succ = 0, 0, 0
    manifest = dict(common, episodes=[])
    obs = self.env.reset()
    pf_reported = False
    with torch.no_grad():
        while saved < n_target:
            st = read_state()
            inp = {"obs": self.running_mean_std(obs["obs"]),
                   "priv_info": obs["priv_info"]}
            mu = self.model.act_inference(inp)
            obs, rew, dones, infos = self.env.step(torch.clamp(mu, -1.0, 1.0))
            act = raw.hand.data.joint_pos_target[:, map_t].float().cpu().numpy()
            sig = getattr(raw, "_sig_merged", {}) or {}
            newly = sig.get("newly_success")
            succ_now = newly.cpu().numpy() if newly is not None else np.zeros(N, bool)
            d = dones.cpu().numpy().astype(bool)
            if d.any() and ep_done < 3:                     # 首波诊断
                _sa = raw._A.data.get("succeeded")
                _sb = raw._B.data.get("succeeded")
                print(f"[dpc-dbg] sig键={sorted(sig.keys())[:12]}")
                print(f"[dpc-dbg] newly存在={newly is not None} "
                      f"succA={int(_sa.sum()) if _sa is not None else 'None'} "
                      f"succB={int(_sb.sum()) if _sb is not None else 'None'} "
                      f"本步done={int(d.sum())}")
            for i in range(N):
                bufS[i].append(st[i])
                bufA[i].append(act[i])
                if d[i]:
                    ep_done += 1
                    if succ_now[i]:
                        ep_succ += 1
                        if saved < n_target and len(bufS[i]) > 2:
                            S = np.stack(bufS[i][:-1]).astype(np.float32)
                            A = np.stack(bufA[i][:-1]).astype(np.float32)
                            ep = {"success": True, "len": int(len(S))}
                            np.savez_compressed(
                                os.path.join(out_dir, f"episode_{saved:04d}.npz"),
                                state=S, action=A, success=np.bool_(True),
                                meta=json.dumps(dict(ep, **common),
                                                ensure_ascii=False))
                            manifest["episodes"].append(ep)
                            saved += 1
                    bufS[i], bufA[i] = [], []
            if not pf_reported and ep_done >= pf_ep:
                rate = ep_succ / max(ep_done, 1)
                print(f"[dpc] ★预检(闸口径 n={ep_done}): 成功率 {rate:.1%} "
                      f"(基准=100%@n1024; 本口径只判>=90%闸)", flush=True)
                assert rate >= 0.90, f"预检未过闸 ({rate:.1%}) — ckpt/旗集/漂移三查"
                pf_reported = True
            if ep_done and ep_done % 64 == 0:
                print(f"[dpc] 集{ep_done} 成功{ep_succ} 已存{saved}/{n_target}",
                      flush=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[dpc] ✅ {saved} 集入库 -> {out_dir} | 总回合{ep_done} 成功率"
          f"{ep_succ / max(ep_done, 1):.1%}", flush=True)


import rl_rebuild.algo.ppo.ppo as _ppo_mod  # noqa: E402

_ppo_mod.PPO.train = _collect
sys.argv = AAGF_ARGV + (["--headless"] if _a.headless else [])
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"),
               run_name="__main__")
