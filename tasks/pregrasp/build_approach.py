"""★标准接近路径规划入口 (2026-09-05 定稿)。

给"物体 clip + 每只手的 GraspPose(prior npz + yaw)", 一键产出
从对称站姿到 GraspPose 的**双手同时接近**参考 Approach_<name>.npz
(= cuRobo 站姿→PreGrasp0 叠加同时 + 成形梯 PreGrasp0→Grasp)。

内置标准配方 (pour25 定稿, 见 memory pour25-approach-curobo-fixes):
  --joint 0            : 先分别单臂规划, 再叠加成"双手同时"(joint=1 联合常几何无解)
  --own_obstacle 1     : 接近段瓶杯**都常开障碍**(默认 V2AP 会排除本手目标 => 臂穿物)
  --sphere_buffer 0.008 --sphere_buffer_links arm
                       : 只把**臂** link 碰撞球充胖 0.8cm, 补 cuRobo 球模型对真实 mesh
                         的欠近似(实测臂/手逐 link 凸出 0.4~0.8cm), 修臂扫物;
                         不动手指(全局充胖会把 PreGrasp0 判死, 手指须贴物)
只用 cuRobo 到 **PreGrasp0**(丢弃它的 grasp-reach —— 那段世界为空会穿物),
最后一段贴合交给成形梯(沿 prior 设计的接近方向逐级下探)。

用法:
  cd /home/lyh/Project/RL_Correction
  SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/pregrasp/build_approach.py \
    --clip Pour25_bottle \
    --prior_a tasks/pregrasp/priors/Pour25_bottle_thumbfix.npz --yaw_a 90 \
    --prior_b tasks/pregrasp/priors/Pour25_cup_thumbfix.npz   --yaw_b 315 \
    --out tasks/Pour/25/A_Design/L1_Data/Motion_Planning/Approach_pour25.npz
  ($PY = /home/lyh/luhr/MagicSim/.venv/bin/python)

约定: prior_a/yaw_a = 主手(clip 的 hand_side, 通常右)抓 primary 物体;
      prior_b/yaw_b = 另一手抓 secondary 物体。thumbfix prior 对两阶段都适用
      (cuRobo 阶段手指锁开, 只用腕位; 成形阶段用手指)。
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = sys.executable

p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                            description=__doc__)
p.add_argument("--clip", required=True, help="物体 clip 名 (含 primary+secondary 摆位)")
p.add_argument("--prior_a", required=True, help="主手 GraspPose prior npz")
p.add_argument("--yaw_a", type=float, required=True, help="主手绕物体轴 yaw (度)")
p.add_argument("--prior_b", required=True, help="另一手 GraspPose prior npz")
p.add_argument("--yaw_b", type=float, required=True, help="另一手绕物体轴 yaw (度)")
p.add_argument("--out", required=True, help="Approach 输出全路径 (.npz)")
p.add_argument("--workdir", default="", help="中间产物目录 (默认 = out 所在目录)")
# —— 标准配方 (一般不用改) ——
p.add_argument("--own_obstacle", type=int, default=1)
p.add_argument("--sphere_buffer", type=float, default=0.008)
p.add_argument("--sphere_buffer_links", default="arm")
p.add_argument("--act_dist", type=float, default=0.015)
p.add_argument("--form_frames", type=int, default=12)
p.add_argument("--timeout", type=int, default=2400, help="每个 Isaac 阶段墙钟上限(s)")
args = p.parse_args()

WORK = args.workdir or os.path.dirname(os.path.abspath(args.out))
os.makedirs(WORK, exist_ok=True)
SEQ = os.path.join(WORK, "curobo_stance2pregrasp_seq.npz")
SIM = os.path.join(WORK, "curobo_stance2pregrasp_sim.npz")
ENV = dict(os.environ, SHARPA_WANDB="0", RL_ISAAC_NO_GUARD="1",
           PYTHONPATH=REPO)


def run_isaac(name, cmd, log, done_markers, out_file, fail_markers=()):
    """跑一个 Isaac 阶段: 轮询到成功标记+产物即杀(这些一次性脚本卡在 Isaac 关闭不退)。"""
    fail_markers = tuple(fail_markers) + ("Traceback (most recent call last)",
                                          "CUDA out of memory")
    print(f"\n[approach] ▶ {name} ...\n  {' '.join(cmd)}", flush=True)
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=REPO, env=ENV)
    t0 = time.time()
    ok = False
    while time.time() - t0 < args.timeout:
        if proc.poll() is not None:
            break                                   # 自己退了
        txt = _read(log)
        if any(m in txt for m in done_markers) and os.path.exists(out_file):
            ok = True
            break
        if any(m in txt for m in fail_markers):
            break
        time.sleep(5)
    else:
        print(f"[approach] ⚠ {name} 超时 {args.timeout}s", flush=True)
    # 收尾: 杀进程 + 等 carb + 清 comm 过滤残留
    try:
        proc.kill()
    except Exception:
        pass
    time.sleep(10)
    ok = ok or (os.path.exists(out_file) and any(m in _read(log) for m in done_markers))
    tail = "\n  ".join(_read(log).splitlines()[-6:])
    print(f"[approach] {'✅' if ok else '❌'} {name} | 产物 {'有' if os.path.exists(out_file) else '无'}"
          f"\n  末尾:\n  {tail}", flush=True)
    if not ok:
        raise SystemExit(f"[approach] {name} 失败, 见 {log}")


def _read(path):
    try:
        with open(path, errors="ignore") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""


def overlap_pregrasp(seq, sim):
    """从 joint=0 顺序 plan 抽两手 pregrasp 段(丢 grasp-reach), 叠加成同一帧窗同时伸手。"""
    S = np.load(seq, allow_pickle=True)
    rq, lq = np.asarray(S["right_q"]), np.asarray(S["left_q"])
    names = [str(x) for x in S["seg_frames"]]
    lens = list(S["seg_lens"].tolist())
    b = np.cumsum([0] + lens)
    ri = next(i for i, n in enumerate(names) if n.startswith("right") and "pregrasp" in n)
    li = next(i for i, n in enumerate(names) if n.startswith("left") and "pregrasp" in n)
    r_pg, l_pg = rq[b[ri]:b[ri + 1]], lq[b[li]:b[li + 1]]
    T = max(len(r_pg), len(l_pg))

    def rs(a):
        if len(a) == T:
            return a
        xp = np.linspace(0, 1, len(a)); x = np.linspace(0, 1, T)
        return np.stack([np.interp(x, xp, a[:, j]) for j in range(a.shape[1])], axis=1)

    r_pg, l_pg = rs(r_pg), rs(l_pg)
    np.savez(sim, right_q=r_pg.astype(np.float32), left_q=l_pg.astype(np.float32),
             right_names=S["right_names"], left_names=S["left_names"],
             seg_frames=np.array(["curobo_sim_pregrasp_only"], dtype=object),
             seg_lens=np.array([T], np.int64))
    print(f"[approach] 叠加: 右pg{r_pg.shape} 左pg{l_pg.shape} -> 同时伸手 {T} 帧", flush=True)
    return T


# ---- 阶段1: cuRobo 双手规划(标准修复旗)-> 顺序 plan ----
run_isaac(
    "cuRobo 双手规划",
    [PY, "-u", os.path.join(HERE, "view_curobo_plan.py"),
     "--clip", args.clip,
     "--prior_a", args.prior_a, "--yaw_a", str(args.yaw_a),
     "--prior_b", args.prior_b, "--yaw_b", str(args.yaw_b),
     "--joint", "0", "--own_obstacle", str(args.own_obstacle),
     "--sphere_buffer", str(args.sphere_buffer),
     "--sphere_buffer_links", args.sphere_buffer_links,
     "--act_dist", str(args.act_dist),
     "--save_plan", SEQ, "--headless"],
    log=os.path.join(WORK, "_stage1_curobo.log"),
    done_markers=("[save] 逐侧臂参考已存",), out_file=SEQ,
    fail_markers=("规划失败",))

# ---- 阶段2: 抽 pregrasp-only + 叠加同时 ----
overlap_pregrasp(SEQ, SIM)

# ---- 阶段3: 成形梯 -> Approach ----
run_isaac(
    "成形梯拼装",
    [PY, "-u", os.path.join(HERE, "build_motion.py"),
     "--clip", args.clip,
     "--prior_a", args.prior_a, "--yaw_a", str(args.yaw_a),
     "--prior_b", args.prior_b, "--yaw_b", str(args.yaw_b),
     "--plan", SIM, "--out", args.out,
     "--form_frames", str(args.form_frames), "--headless"],
    log=os.path.join(WORK, "_stage3_form.log"),
    done_markers=("[build] Approach ->",), out_file=args.out)

# ---- 完成 + GUI 命令 ----
z = np.load(args.out, allow_pickle=True)
T = len(z["right_q"])
print(f"\n[approach] ✅ 完成 -> {args.out} ({T} 帧)")
print(f"[approach] GUI 复看:")
print(f"  SHARPA_WANDB=0 PYTHONPATH=. {PY} \\")
print(f"    tasks/pregrasp/view_approach.py --approach {args.out} \\")
print(f"    --clip {args.clip} --prior_a {args.prior_a} --yaw_a {args.yaw_a}")
