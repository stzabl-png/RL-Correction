"""开训前的依赖预检: 不启动 Isaac, 一次列出**所有**缺失的资产。

    PYTHONPATH=. $PY -m tasks.pregrasp.preflight [clip名 ...]

新机器上线 / 换数据仓时**先跑这个**。

2026-08-16 加的, 起因: 双卡 A6000 上线时连续四轮"起训练->崩->补一个文件->再起",
每轮都要等几分钟建场景才发现下一个缺失。缺的依次是: tasks/ 整个目录、躯干锁死版
机器人 USD(25MB)、机器人 URDF(16MB)、world_fused.npz(相机外参)。这条链依次要: clip 注册表 -> npz/mesh/usd
-> world_fused.npz(相机外参) -> 机器人 URDF(FK) -> affordance -> GraspPose prior。
逐个报, 一次把缺的全列出来, 别再一轮只发现一个。
"""
import os
import sys
import traceback
sys.path.insert(0, ".")

CLIPS = sys.argv[1:] or ["Grasp3", "Pour17_bottle", "Pour17_cup"]
ok_all = True
try:
    from rl_rebuild.correction.kinematics import URDF_PATH
    print(f"[URDF] {'✅' if os.path.exists(URDF_PATH) else '⛔缺'}  {URDF_PATH}")
    ok_all &= os.path.exists(URDF_PATH)
except Exception as e:
    print(f"[URDF] ⛔ import 失败: {e}"); ok_all = False

from rl_rebuild.correction import clips
from rl_rebuild.correction.place_camera import find_cam_src

for name in CLIPS:
    print(f"\n=== {name} ===")
    try:
        e = clips.clip_entry(name)
    except Exception as ex:
        print(f"  ⛔ 注册表: {ex}"); ok_all = False; continue
    for k in ("npz", "mesh", "usd", "affordance", "grasp_prior_npz_default",
              "scene_layout_json", "keyframes_json"):
        p = e.get(k)
        if not isinstance(p, str) or not p:
            continue
        good = os.path.exists(p)
        ok_all &= good
        print(f"  {'✅' if good else '⛔缺'} {k:26s} {p}")
    cam = find_cam_src(e["mesh"], e["npz"])
    print(f"  {'✅' if cam else '⛔缺'} {'world_fused.npz(相机外参)':26s} "
          f"{os.path.dirname(cam) if cam else '三处候选都没有'}")
    ok_all &= bool(cam)

print("\n" + "=" * 60)
print("✅ 预检通过, 可以开训" if ok_all else "⛔ 预检未通过, 先补上面标 ⛔ 的")
print("=" * 60)
sys.exit(0 if ok_all else 1)
