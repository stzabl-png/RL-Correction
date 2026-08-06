"""不跑 op=tmpl, 秒级预览一个 raw_anno YAML 的几何量 —— 设计模板时用来迭代 qpos.

  /home/lyh/anaconda3/envs/dexonomy/bin/python tools/probe_template.py \
      assets/hand/sharpa_wave/raw_anno/thinplate_35.yaml

  # 扫参数找目标张口 (四指屈曲 x 拇指 CMC_FE)
  ... tools/probe_template.py --sweep

报出:
  张口       拇指 DP 接触点 ↔ 四指 DP 接触点 的平均距离 (mm) —— 决定能夹多厚的物体
  四指跨     四指接触点之间的最大距离 (mm)
  法向对置   接触法向两两内积的最小值, -1 = 完美对置 (会随屈曲变化, 值得盯)
  垫覆盖     接触体里有几个落在 elastomer 垫(5 个 DP)上 —— RL 侧只有它们有力传感器
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

HAND = "assets/hand/sharpa_wave"
PADS4 = ["right_index_DP", "right_middle_DP", "right_ring_DP", "right_pinky_DP"]
PAD5 = set(PADS4) | {"right_thumb_DP"}
# 关节序 (与 raw_anno 的 qpos 一致)
ORDER = ("thumb[CMC_FE,CMC_AA,MCP_FE,MCP_AA,IP] index[MCP_FE,AA,PIP,DIP] "
         "middle[...] ring[...] pinky[CMC,MCP_FE,AA,PIP,DIP]")


def build():
    import mujoco
    m = mujoco.MjModel.from_xml_path(f"{HAND}/right.xml")
    return m, mujoco.MjData(m), yaml.safe_load(open(f"{HAND}/keypoint.yaml"))


def contacts_world(m, d, kp, qpos, spec):
    """spec: {body: idx} 或 {body: [idx,...]} -> {body: [(pos,normal), ...]}"""
    import mujoco
    d.qpos[:] = np.asarray(qpos, float)
    mujoco.mj_forward(m, d)
    out = {}
    for b, v in spec.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        R, t = d.xmat[bid].reshape(3, 3), d.xpos[bid]
        idxs = v if isinstance(v, list) else [v]
        pts = []
        for i in idxs:
            k = np.asarray(kp[b][i], float)
            pts.append((R @ k[:3] + t, R @ k[3:6]))
        out[b] = pts
    return out


def metrics(cw):
    th = [p for p, _ in cw.get("right_thumb_DP", [])]
    F = [p for b in PADS4 if b in cw for p, _ in cw[b]]
    ap = (np.linalg.norm(np.array(F) - np.mean(th, 0), axis=1).mean() * 1000
          if th and F else float("nan"))
    sp = (np.linalg.norm(np.array(F)[:, None] - np.array(F)[None], axis=-1).max() * 1000
          if len(F) > 1 else 0.0)
    N = np.array([n / max(np.linalg.norm(n), 1e-9) for b in cw for _, n in cw[b]])
    opp = float((N @ N.T).min()) if len(N) > 1 else 0.0
    pads = len([b for b in cw if b in PAD5])
    return ap, sp, opp, pads, len(N)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("yaml_path", nargs="?")
    ap_.add_argument("--sweep", action="store_true", help="扫四指屈曲 x 拇指 CMC_FE")
    a = ap_.parse_args()
    m, d, kp = build()

    if a.sweep:
        spec = {"right_index_DP": 0, "right_middle_DP": 1, "right_ring_DP": 0,
                "right_pinky_DP": 1, "right_thumb_DP": 1}
        print(f"{'四指(MCP,PIP,DIP)':>20s} {'拇指CMC_FE':>10s} {'张口mm':>8s} "
              f"{'四指跨mm':>9s} {'法向对置':>9s}")
        for f in ([0.6, 0, 0.7, 0.5], [0.75, 0, 0.85, 0.6],
                  [0.9, 0, 1.0, 0.65], [1.05, 0, 1.15, 0.7]):
            for t0 in (1.55, 1.66, 1.80, 1.92):
                q = [t0, 0.15, -0.51, 0.08, 0.11] + f + f + f + [0.05] + f
                r = metrics(contacts_world(m, d, kp, q, spec))
                print(f"  {f[0]:.2f},{f[2]:.2f},{f[3]:.2f}      {t0:10.2f} "
                      f"{r[0]:8.1f} {r[1]:9.1f} {r[2]:+9.3f}")
        print(f"\n关节序: {ORDER}")
        return 0

    if not a.yaml_path:
        ap_.error("给一个 raw_anno yaml, 或用 --sweep")
    doc = yaml.safe_load(open(a.yaml_path))
    r = metrics(contacts_world(m, d, kp, doc["qpos"], doc.get("contact", {})))
    print(f"\n{os.path.basename(a.yaml_path)}")
    print(f"  张口(拇指↔四指) {r[0]:.1f} mm   四指跨 {r[1]:.1f} mm")
    print(f"  法向对置 {r[2]:+.3f}   接触点 {r[4]}   垫覆盖 {r[3]}/5")
    if r[3] < 4:
        print("  ⚠ 垫覆盖 <4 —— RL 侧候选判据要求 pads>=4, 这个模板生成的抓姿必然过不了 Gate 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
