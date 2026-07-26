"""吞吐实测: DexMate 按 env 复制后能跑多少 FPS.

对照组 = 现在的飞手 (22 关节)。实验组 = DexMate 整机 (67 关节 / 110 links)。
不接任何控制, 纯物理步进, 只测 num_envs 对吞吐的影响。

  $PY -m rl_rebuild.correction.bench_throughput --robot dexmate --num_envs 256
  $PY -m rl_rebuild.correction.bench_throughput --robot flying  --num_envs 256
"""
import argparse
import os
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--robot", choices=["dexmate", "flying"], default="dexmate")
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--warmup", type=int, default=30)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True

# GPU 独占槽位 (CLAUDE.md §7: 两个 Isaac 同时满载会把电源打断)
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("bench")

app = AppLauncher(args).app

import math  # noqa: E402

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets")
DEXMATE_USD = os.path.join(
    os.environ.get("MAGICSIM_ASSETS", "/home/lyh/luhr/MagicSim/Assets"),
    "Robots", "vega_1p_sharpa.usd")

# 和训练完全同一套物理参数, 否则测出来没有可比性
SIM = SimulationCfg(
    dt=1 / 240, render_interval=2, gravity=(0.0, 0.0, -9.81),
    physx=PhysxCfg(solver_type=1, max_position_iteration_count=8,
                   max_velocity_iteration_count=0, bounce_threshold_velocity=0.2,
                   gpu_max_rigid_contact_count=2**23,
                   gpu_max_rigid_patch_count=5 * 2**18))

if args.robot == "dexmate":
    ROBOT = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=DEXMATE_USD, activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(-0.5, 0.0, 0.0)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"],
                                              stiffness=None, damping=None)})
else:
    ROBOT = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_ASSETS, "SharpaWave/right_sharpa_wave.usda"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, angular_damping=0.01,
                max_linear_velocity=1000.0, max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0, max_contact_impulse=1e32),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False, enabled_self_collisions=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.3, 1.2)),
        actuators={"fingers": ImplicitActuatorCfg(joint_names_expr=[".*"],
                                                  stiffness=20.0, damping=2.0)})


@configclass
class BenchSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ROBOT


sim = SimulationContext(SIM)
scene_cfg = BenchSceneCfg(num_envs=args.num_envs, env_spacing=2.0, replicate_physics=False)
t0 = time.time()
scene = InteractiveScene(scene_cfg)
# 桌子 (静态碰撞体, 和训练一致)
tbl = sim_utils.CuboidCfg(size=(1.2, 1.2, 0.04),
                          collision_props=sim_utils.CollisionPropertiesCfg())
tbl.func("/World/envs/env_.*/Table", tbl, translation=(0.0, 0.0, 0.85 - 0.02))
spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
sim.reset()
build_s = time.time() - t0

rb = scene["robot"]
print(f"[bench] robot={args.robot}  num_envs={args.num_envs}  "
      f"关节={rb.num_joints}  刚体={rb.num_bodies}  建场景 {build_s:.1f}s", flush=True)

dt = sim.get_physics_dt()
for _ in range(args.warmup):
    scene.write_data_to_sim(); sim.step(render=False); scene.update(dt)
torch.cuda.synchronize()


def _gpu():
    """(功耗 W, 显存 MiB). 这台机器 5090+14700K 供电余量极薄, 功耗是硬约束."""
    try:
        import subprocess
        o = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=3).stdout.split(",")
        return float(o[0]), float(o[1])
    except Exception:
        return float("nan"), float("nan")


pw, mem = [], []
t0 = time.time()
for i in range(args.steps):
    scene.write_data_to_sim(); sim.step(render=False); scene.update(dt)
    if i % 20 == 0:
        a, b = _gpu(); pw.append(a); mem.append(b)
torch.cuda.synchronize()
el = time.time() - t0
pw_max = max(pw) if pw else float("nan")
mem_max = max(mem) if mem else float("nan")

phys_fps = args.steps / el
ctrl_fps = phys_fps / 12                       # decimation=12 -> 20Hz 控制
print(f"[bench] {args.steps} 物理步 用时 {el:.2f}s")
print(f"[bench] 物理 {phys_fps:.0f} 步/s | 控制步 {ctrl_fps:.1f} 步/s | "
      f"env·控制步/s = {ctrl_fps * args.num_envs:.0f}")
print(f"[bench] GPU 峰值 {pw_max:.0f}W / {mem_max:.0f}MiB")
print(f"BENCH_RESULT {args.robot} {args.num_envs} {rb.num_joints} {rb.num_bodies} "
      f"{phys_fps:.1f} {ctrl_fps * args.num_envs:.0f} {build_s:.1f} "
      f"{pw_max:.0f} {mem_max:.0f}", flush=True)
import sys  # noqa: E402
sys.stdout.flush()
_slot.release()
# ⚠ 不能用 sim.stop()+app.close(): 大 num_envs 下 Isaac 关闭流程会**永久挂住**
# (实测 flying/1024 卡了 10 分钟仍不退, 显存不放, 后续实验全部排不上).
# 数已经测完了, 直接硬退.
os._exit(0)
