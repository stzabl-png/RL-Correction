#!/usr/bin/env python
"""Pour17 统一评测器 —— 与训练 reward 解耦, baseline 与 ours 共用。

============================ 设计红线 ============================

1. **不 import ours 的任何训练件**: 不碰 eval_pour.py / PPO wrapper / 503D obs /
   58D residual action / GraspPose / CuRobo / confidence reward。本文件只依赖:
     - 冻结的判据 progress.py (G1..G4 / Placed / 死线阈值)
     - 世界指纹 world_manifest.json
     - 母带 npz (evaluator 从中取瓶口/杯口方向与静置位姿)
   ⟹ baseline 用自己的 env + policy 接进来即可, 不会把 ours 的方法带进去。

2. **判据常量一律 import, 不复制**。progress.py 是唯一真源;
   本文件出现的任何阈值数字都来自 `from progress import ...`。

3. **D1–D8 原因分类不读 ours 的 pour_env.tb**(那是 ours 的 env 成员)。
   批量版 PourProgressBatch 的 `fail` 只是 bool 张量、不带原因, 所以本文件按
   progress.py 的**同一组条件**自行判因(见 classify_failure), 阈值仍来自 import。

============================ 评测协议(固定) ============================

  - 全部 episode **从 t0 reset**, **不使用 RSI**(不走 entry_table 的 g1/g2/g3 预置)
  - **不加 exploration noise**, 用 deterministic policy mean
  - **≥512 episodes**(默认 512)
  - 固定 seed list + reset manifest 落盘, 可逐 episode 复现
  - 报告: success rate / 达到指定成功率所需 environment steps / wall-clock / GPU 数
  - 输出完整 failure/timeout 分布与 D1–D8 逐条计数

============================ 接入方式 ============================

baseline 侧只需实现两个回调, 其余全由本文件固定:

    def make_env(num_envs, seed, device) -> EnvLike
        # EnvLike 需提供:
        #   reset() -> obs
        #   step(action) -> (obs, terminated, truncated, info)
        #   以及 evaluator 需要的实测量读取接口(见 ObsAdapter)
    def policy(obs) -> action        # deterministic mean, 不加噪声

    python run_eval.py --entry your_pkg.your_module:make_env,policy \
                       --episodes 512 --out results.json

不提供 --entry 时只做**自检**: 打印协议、判据阈值、世界指纹核对结果并退出 0,
用来确认 bundle 完整性(不需要 Isaac)。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(HERE))          # 冻结的 progress.py 就在同目录

from progress import (  # noqa: E402  —— 判据唯一真源, 不复制任何阈值
    CERT_HOLD, CERT_RAMP, CERT_RET, CERT_RISE, CERT_SLIP, D1_DROP, D2_TILT, D3_DEV,
    G1_HOLD, M2_HOLD, M2_TILT, M3_HOLD, M3_POS, M3_ROT, M4_ARM, M4_DIST_POS,
    M4_DIST_ROT, M4_HOLD, TABLE_Z, PourProgress, _axis_tilt,
)

PAD_FTH_N = 0.5        # 指垫接触力阈(与 world_manifest.sensors.pad_force_threshold_N 核对)
PADS_MIN = 3           # 双手各需 >=3/5 垫
MOUTH_GATE = 0.12      # 瓶口-杯口距离门 (m)
MOUTH_HALF_BOTTLE = 0.087   # 与 ours 的 _mouth_local 同参(pour_env.py:163)
MOUTH_HALF_CUP = 0.066      # 同上 (pour_env.py:164)

# ── 评测协议硬常量(canonical_reset_v1.json 是权威, 这里是它的代码副本) ──
# ★ horizon 由评测器**强制**, 不依赖 baseline env 的 truncated —— 两边 truncated
#   口径不同会让成功率不可比。903 = int(T_ROW*1.2)+120, T_ROW=母带全长 653。
#   ⚠ world_manifest.time.episode_length_s=13.125 对 Pour17 **无效**(ours 从不用
#   env 的 truncated 走超时), 别拿它当 horizon。
MAX_CONTROL_STEPS = 903
CERT_LIFT_M = 0.015          # G2 认证: 双腕沿**世界 +Z** 抬 15mm, 姿态不变
WARMUP_CLAMP_STEPS = 0       # 评测协议不做焊接热身(ours 训练默认 15, 那是脚手架)


# ───────────────────────── 判据侧工具 ─────────────────────────
def mouth_local(z, rows, oi: int, half: float) -> np.ndarray:
    """物体局部系的"口"方向向量。与 ours pour_env._mouth_local 逐行等价。"""
    q = np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows][0]
    w, x, y, zz = q / np.linalg.norm(q)
    R = np.array([[1 - 2 * (y * y + zz * zz), 2 * (x * y - w * zz), 2 * (x * zz + w * y)],
                  [2 * (x * y + w * zz), 1 - 2 * (x * x + zz * zz), 2 * (y * zz - w * x)],
                  [2 * (x * zz - w * y), 2 * (y * zz + w * x), 1 - 2 * (x * x + y * y)]])
    up = np.array([0.0, half, 0.0])
    return up if (R @ up)[2] > (R @ (-up))[2] else -up


def classify_failure(prog: PourProgress, obj0, obj1) -> list[str]:
    """D1–D8 原因分类 —— **返回全部命中的死线**(可能多条同时成立)。

    ★ 为什么要自己判: 批量版 PourProgressBatch 的 fail 只是 bool 张量、不带原因;
      而 ours 的 pour_env.tb 属于 ours 的 env, baseline 不该碰。

    ★ 为什么返回列表而不是挑一条(2026-08-29 与 RL session 定稿):
      progress.py 的 `out["fail"]` 是反复赋值、**最后命中覆盖前面的**, 那个顺序是
      写终止逻辑时的副产物, 对归因没有意义。评测报告要的是分布, 所以这里返回全部命中,
      统计时**每条死线各自计数**(一个 episode 可计入多条), 口径无歧义、也不与
      progress.py 的赋值顺序绑定。对"是否终止"的判定无影响(任一成立即终止)。

    阈值全部来自 progress.py 的 import; 条件与 progress_batch.py:345-360 逐条对应。
    """
    fired: list[str] = []
    up_b, up_c = prog.up_b, np.array([0.0, 1.0, 0.0])
    k = min(prog.k, prog.N - 1)
    for oi, act, up_ in ((0, obj0, up_c), (1, obj1, up_b)):
        tl = _axis_tilt(act[3:7], up_)
        if not prog.g[2] and tl > np.radians(60):
            fired.append(f"D2pre_fallen_obj{oi}")
        if act[2] < TABLE_Z - D1_DROP:
            fired.append(f"D1_drop_obj{oi}")
        if not prog.placed and np.linalg.norm(act[:3] - prog.obj[oi][k][:3]) > D3_DEV:
            fired.append(f"D3_dev_obj{oi}")
        if prog.placed:
            # m3_snap: reset 时为 None, placed 达成/预置时被赋值 ⟹ placed 为真时必非 None
            # (progress.py:130/154/299 —— 已核)。故此处直接取, 不做无谓回退。
            assert prog.m3_snap is not None, "placed=True 但 m3_snap 为 None, 判据侧有变更"
            sn = prog.m3_snap[oi]
            if (np.linalg.norm(act[:3] - sn[:3]) > M4_DIST_POS or tl > M4_DIST_ROT):
                fired.append(f"D8_disturb_obj{oi}")
            if tl > D2_TILT:
                fired.append(f"D2_tilt_obj{oi}")
    return sorted(fired)


# ───────────────────────── 协议与指纹 ─────────────────────────
def criteria_snapshot() -> dict:
    return {
        "G1": {"pads_min_per_hand": PADS_MIN, "pad_force_threshold_N": PAD_FTH_N,
               "hold_steps": G1_HOLD},
        "G2": {"rise_m": CERT_RISE, "slip_max_m": CERT_SLIP},
        "G3": {"tilt_min_deg": round(float(np.degrees(M2_TILT)), 1),
               "mouth_gate_m": MOUTH_GATE, "hold_steps": M2_HOLD},
        "Placed": {"pos_m": M3_POS, "rot_deg": round(float(np.degrees(M3_ROT)), 1),
                   "hold_steps": M3_HOLD},
        "G4_success": {"arm_per_joint_deg": round(float(np.degrees(M4_ARM)), 1),
                       "obj_disturb_pos_m": M4_DIST_POS,
                       "obj_disturb_rot_deg": round(float(np.degrees(M4_DIST_ROT)), 1),
                       "hold_steps": M4_HOLD},
        "deadlines": {"D1_drop_m": D1_DROP,
                      "D2_tilt_deg": round(float(np.degrees(D2_TILT)), 1),
                      "D3_dev_m": D3_DEV, "table_z": TABLE_Z},
        "source": "progress.py (frozen, reward-decoupled)",
    }


def protocol_snapshot(episodes: int) -> dict:
    return {
        "reset": "t0 only", "rsi": False, "exploration_noise": False,
        "policy": "deterministic mean", "episodes": episodes,
        "max_control_steps": MAX_CONTROL_STEPS,
        "max_control_steps_enforced_by": "evaluator (不依赖 baseline env 的 truncated)",
        "warmup_clamp_steps": WARMUP_CLAMP_STEPS,
        "certification": {"lift_m": CERT_LIFT_M, "frame": "world +Z, 姿态保持",
                          "ramp": CERT_RAMP, "hold": CERT_HOLD, "return": CERT_RET,
                          "driven_by": "evaluator via env.apply_certification_offset(alpha)"},
        "canonical_reset": "world/canonical_reset_v1.json",
        "note": "全部 episode 从 t0 reset; 不使用 RSI 预置(不走 entry_table 的 g1/g2/g3)"}


def cert_alpha(phase: int, t: int) -> float:
    """G2 认证的 alpha 时间表 —— 与 ours pour_env.py:373-380 同式。

    phase: PourProgress.cert_phase (0 未启动 / 1 斜升 / 2 保持 / 3 斜降)
    返回 0..1, 由评测器喂给 env.apply_certification_offset(alpha)。
    ★ 认证是**统一评测器发起的外部抓稳测试**, 不是 policy 的动作 ——
      否则 baseline 抓得再稳也永远过不了 G2(它无从知道该抬手)。
    """
    if phase == 1:
        return min((t + 1) / CERT_RAMP, 1.0)
    if phase == 2:
        return 1.0
    if phase == 3:
        return max(1.0 - (t + 1) / CERT_RET, 0.0)
    return 0.0


def gpu_info() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            text=True, timeout=20).strip().splitlines()
        return {"count": len(out), "devices": out}
    except Exception as e:
        return {"count": "unavailable", "reason": f"{type(e).__name__}"}


def verify_world(manifest: Path) -> dict:
    """世界指纹核对: 判据侧常量 vs manifest 运行时值。不一致要显式报出来。"""
    if not manifest.is_file():
        return {"status": "missing", "path": str(manifest)}
    m = json.loads(manifest.read_text(encoding="utf-8"))
    checks = []
    tbl = m.get("table", {})
    tz = tbl.get("table_top_z_m", tbl.get("table_top_z"))
    checks.append({"item": "table_top_z", "manifest": tz, "evaluator_const": TABLE_Z,
                   "match": (tz is not None and abs(float(tz) - TABLE_Z) < 1e-9),
                   "note": "progress.py 的 TABLE_Z 是独立硬编码副本, 与 cfg 无联动 —— "
                           "改桌高必须两处同步改"})
    fth = m.get("sensors", {}).get("pad_force_threshold_N")
    checks.append({"item": "pad_force_threshold_N", "manifest": fth,
                   "evaluator_const": PAD_FTH_N,
                   "match": (fth is not None and abs(float(fth) - PAD_FTH_N) < 1e-9)})
    pm = m.get("sensors", {}).get("pads_min_per_hand")
    checks.append({"item": "pads_min_per_hand", "manifest": pm,
                   "evaluator_const": PADS_MIN,
                   "match": (pm is not None and int(pm) == PADS_MIN)})
    return {"status": "ok", "all_match": all(c["match"] for c in checks),
            "checks": checks,
            "robot_usd": m.get("robot", {}).get("usd"),
            "reference_tape_md5": m.get("reference_tape", {}).get("md5"),
            "control_dt_s": m.get("time", {}).get("control_dt_s"),
            "decimation": m.get("time", {}).get("decimation")}


def make_seeds(n: int, base: int = 20260829) -> list[int]:
    """固定 seed list —— 纯函数, 任何机器上同 n/base 得到同一串。"""
    return [base + i for i in range(n)]


# ───────────────────────── 主流程 ─────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--tape", type=Path,
                    default=BUNDLE / "reference" / "pour17_reference_v2__2ed81358__653rows.npz",
                    help="母带 —— evaluator 从中取瓶口/杯口方向与静置位姿。"
                         "★ 换母带会改变判据, 必须显式声明用了哪一版")
    ap.add_argument("--manifest", type=Path, default=BUNDLE / "world" / "world_manifest.json")
    ap.add_argument("--entry", default=None,
                    help="baseline 接入点 'pkg.mod:make_env,policy'; 不给则只自检")
    ap.add_argument("--out", type=Path, default=HERE / "eval_result.json")
    ap.add_argument("--seed-base", type=int, default=20260829)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    if not a.tape.is_file():
        print(f"X 母带不存在: {a.tape}", file=sys.stderr)
        return 2
    tape_md5 = hashlib.md5(a.tape.read_bytes()).hexdigest()
    z = np.load(a.tape, allow_pickle=True)
    rows = np.where(np.asarray(z["source"]) == 1)[0]
    mb = mouth_local(z, rows, 1, MOUTH_HALF_BOTTLE)
    mc = mouth_local(z, rows, 0, MOUTH_HALF_CUP)

    seeds = make_seeds(a.episodes, a.seed_base)
    world = verify_world(a.manifest)
    head = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(), "python": platform.python_version(),
        "protocol": protocol_snapshot(a.episodes),
        "criteria": criteria_snapshot(),
        "world_fingerprint": world,
        "reference_tape": {"path": str(a.tape), "md5": tape_md5,
                           "interaction_rows": int(len(rows)),
                           "mouth_local_bottle": mb.tolist(),
                           "mouth_local_cup": mc.tolist()},
        "seeds": {"base": a.seed_base, "count": len(seeds), "list": seeds},
        "gpu": gpu_info(),
    }

    print("=" * 72)
    print("Pour17 统一评测器 —— 自检")
    print("=" * 72)
    print(f"母带      : {a.tape.name}\n            md5={tape_md5}  交互行={len(rows)}")
    print(f"瓶口局部  : {np.round(mb, 4).tolist()}   杯口局部: {np.round(mc, 4).tolist()}")
    if world["status"] == "ok":
        print(f"世界指纹  : {'✅ 全部一致' if world['all_match'] else '❌ 有不一致, 见下'}")
        for c in world["checks"]:
            flag = "✅" if c["match"] else "❌"
            print(f"   {flag} {c['item']:24s} manifest={c['manifest']}  "
                  f"evaluator={c['evaluator_const']}")
        print(f"   robot USD  : {world['robot_usd']}")
        print(f"   control_dt : {world['control_dt_s']}  decimation={world['decimation']}")
        print(f"   manifest 母带 md5: {world['reference_tape_md5']}")
        if world["reference_tape_md5"] and not str(world["reference_tape_md5"]).startswith(tape_md5[:8]):
            print("   ⚠ 本次用的母带与 manifest 记录的不是同一版 —— 请确认这是有意为之")
    else:
        print(f"世界指纹  : ⚠ {world['status']} ({world.get('path')})")
    print(f"协议      : t0 reset / 无 RSI / deterministic mean / {a.episodes} episodes")
    print(f"            horizon {MAX_CONTROL_STEPS} 步 (评测器强制, 不看 env 的 truncated)"
          f" | 焊接热身 {WARMUP_CLAMP_STEPS} 步")
    print(f"            G2 认证 双腕 +Z {CERT_LIFT_M*1000:.0f}mm 姿态不变, "
          f"斜升{CERT_RAMP}/保持{CERT_HOLD}/斜降{CERT_RET} 步 (评测器发起)")
    print(f"seeds     : {seeds[0]}..{seeds[-1]} (共 {len(seeds)}, 由 seed-base 纯函数生成)")
    print(f"GPU       : {head['gpu']}")

    if a.entry is None:
        head["status"] = "selftest_only"
        head["note"] = ("未提供 --entry, 只做自检。baseline 接入后本文件会跑满 "
                        f"{a.episodes} episodes 并写 success rate / env steps / "
                        "wall-clock / D1-D8 分布。")
        a.out.write_text(json.dumps(head, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\n自检通过 -> {a.out}")
        print("接入 baseline: --entry your_pkg.your_module:make_env,policy")
        return 0

    # ---------------- 有 --entry: 真跑 ----------------
    mod_name, fns = a.entry.split(":")
    make_env_name, policy_name = [x.strip() for x in fns.split(",")]
    mod = importlib.import_module(mod_name)
    make_env = getattr(mod, make_env_name)
    policy = getattr(mod, policy_name)

    reset_manifest, results = [], []
    t0 = time.time()
    total_steps = 0
    for ep, sd in enumerate(seeds):
        env = make_env(num_envs=1, seed=sd, device=a.device)
        prog = PourProgress(str(a.tape), mouth_local_bot=mb, mouth_local_cup=mc,
                            mouth_gate=MOUTH_GATE)
        obs = env.reset()
        reset_manifest.append({"episode": ep, "seed": sd,
                               "init": env.reset_state() if hasattr(env, "reset_state") else "unavailable"})
        fail_reason, steps = None, 0
        deadlines_fired: set[str] = set()
        while True:
            # ── G2 认证: 由评测器发起, env 用自己的控制器达成同一个笛卡尔目标 ──
            alpha = cert_alpha(int(getattr(prog, "cert_phase", 0)),
                               int(getattr(prog, "cert_t", 0)))
            if alpha > 0.0:
                if not hasattr(env, "apply_certification_offset"):
                    raise RuntimeError(
                        "env 缺少 apply_certification_offset(alpha) —— G2 认证需要 env 配合"
                        "把双腕沿世界 +Z 抬 CERT_LIFT_M(姿态不变), 否则 G2 永远过不了。"
                        "见 INTEGRATION.md 的 G2 一节。")
                env.apply_certification_offset(alpha)
            act = policy(obs)                                  # deterministic mean
            obs, terminated, truncated, info = env.step(act)
            steps += 1
            total_steps += 1
            o0, o1 = info["object_0_pose"], info["object_1_pose"]
            out = prog.step(o0, o1, info["arm_q_right"], info["arm_q_left"],
                            info["pads3"], info["wrist_right"], info["wrist_left"])
            fired = classify_failure(prog, o0, o1)      # 全部命中的死线
            if fired:
                deadlines_fired.update(fired)
            if out["fail"]:                                # progress.py 自己的归因(单条)
                fail_reason = out["fail"]
            # ★ horizon 由评测器强制, 不看 baseline env 的 truncated
            hit_cap = steps >= MAX_CONTROL_STEPS
            if out["done"] or terminated or hit_cap:
                truncated = bool(hit_cap and not (out["done"] or terminated))
                break
        results.append({"episode": ep, "seed": sd, "steps": steps,
                        "success": bool(prog.g[4]),
                        "gates": {f"G{i}": bool(prog.g[i]) for i in (1, 2, 3, 4)},
                        "placed": bool(prog.placed),
                        # progress.py 的单条归因(供与 ours 对拍)
                        "fail_reason_progress": fail_reason
                        or ("timeout" if truncated else None),
                        # 本评测器口径: 全部命中的死线, 分布按每条各自计数
                        "deadlines_fired": sorted(deadlines_fired)})
        env.close() if hasattr(env, "close") else None

    wall = time.time() - t0
    succ = sum(r["success"] for r in results)
    from collections import Counter
    # 死线分布: 每条各自计数(一个 episode 可计入多条) —— 口径见 classify_failure
    reasons = Counter(d for r in results for d in r["deadlines_fired"])
    reasons["_none"] = sum(1 for r in results if not r["deadlines_fired"])
    reasons["_timeout"] = sum(1 for r in results
                              if r["fail_reason_progress"] == "timeout")
    gates = {f"G{i}": sum(r["gates"][f"G{i}"] for r in results) for i in (1, 2, 3, 4)}
    n = len(results)
    head.update({
        "status": "completed",
        # ★ 任何"率"旁边必须有它的分母 —— 分母为 0 时给 null 而不是 0.0。
        #   (2026-08-29 教训: 把"本窗零回合"渲染成"成功率 0%", 差点据此误判两条训练线
        #    "开局即死", 真相是回合太长还没结算。)
        "success_rate": (succ / n) if n else None,
        "success_rate_denominator": n,
        "successes": succ, "episodes_run": n,
        "environment_steps_total": total_steps,
        "environment_steps_per_episode_mean": total_steps / max(len(results), 1),
        "wall_clock_s": round(wall, 1),
        "gate_reached_counts": gates,
        "gate_reached_rates": ({k: v / n for k, v in gates.items()} if n else None),
        "gate_reached_denominator": n,
        "deadline_distribution": dict(reasons),
        "deadline_distribution_note": ("每条死线各自计数, 一个 episode 可计入多条; "
                                       "_none = 无死线命中, _timeout = 超时"),
        "episodes": results,
    })
    a.out.write_text(json.dumps(head, indent=1, ensure_ascii=False), encoding="utf-8")
    (a.out.parent / "reset_manifest.json").write_text(
        json.dumps({"seeds": seeds, "episodes": reset_manifest}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n成功率 {succ}/{n} = "
          + (f"{succ/n*100:.1f}%" if n else "N/A (分母为 0)"))
    print(f"env steps {total_steps} | wall {wall:.0f}s | GPU {head['gpu']['count']}")
    print(f"各关卡到达: {gates}")
    print(f"死线分布  : {dict(reasons)}  (每条各自计数, 可重叠)")
    print(f"-> {a.out}  +  reset_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
