"""物理参数审计: 打印 SharpaCorrectionEnv 运行时真实生效的物理参数.
  python -m rl_rebuild.correction.audit_physics --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--freeze", action="store_true",
                    help="冻结参考帧: 手悬停在 t0, 隔离测物体-桌子静置稳定性")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from rl_rebuild.correction.env.correction_env import SharpaCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.correction_env_cfg import SharpaCorrectionEnvCfg  # noqa: E402

cfg = SharpaCorrectionEnvCfg()
cfg.scene.num_envs = 2
env = SharpaCorrectionEnv(cfg)
env.reset()

print("\n================ 物理审计 ================")
print(f"[sim] dt={env.physics_dt}  decimation={cfg.decimation}  "
      f"gravity={env.sim.cfg.gravity}  solver_type={cfg.sim.physx.solver_type}(1=TGS)  "
      f"bounce_thresh={cfg.sim.physx.bounce_threshold_velocity}")

# ---- 手: 材质 / 质量 / 关节增益 ----
hmat = env.hand.root_physx_view.get_material_properties()[0]   # (bodies*shapes?,3)
print(f"\n[hand] 材质 shape={tuple(hmat.shape)} "
      f"static_fric 范围 [{hmat[:, 0].min():.2f}, {hmat[:, 0].max():.2f}] "
      f"dyn [{hmat[:, 1].min():.2f}, {hmat[:, 1].max():.2f}] "
      f"restitution max {hmat[:, 2].max():.2f}")
tips = [i for i, n in enumerate(env.hand.body_names) if "elastomer" in n or "fingertip" in n]
stiff = env.hand.data.default_joint_stiffness[0]
damp = env.hand.data.default_joint_damping[0]
print(f"[hand] 关节 stiffness 范围 [{stiff.min():.1f}, {stiff.max():.1f}]  "
      f"damping [{damp.min():.2f}, {damp.max():.2f}]")
print(f"[hand] 总质量 {env.hand.root_physx_view.get_masses()[0].sum():.3f} kg")

# ---- 物体: 质量 / 惯量 / 材质 / USD 物理属性 ----
omass = env.object.root_physx_view.get_masses()[0]
oin = env.object.root_physx_view.get_inertias()[0]
omat = env.object.root_physx_view.get_material_properties()[0]
print(f"\n[object] mass={float(omass):.3f} kg  "
      f"惯量对角 {[f'{oin[i]:.5f}' for i in (0, 4, 8)]} kg·m²")
print(f"[object] 材质 static/dyn/rest = "
      f"{omat.reshape(-1, 3)[0].tolist()}")

import omni.usd  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402
stage = omni.usd.get_context().get_stage()
obj = stage.GetPrimAtPath("/World/envs/env_0/Object")
rb = PhysxSchema.PhysxRigidBodyAPI(obj)
gyro = rb.GetEnableGyroscopicForcesAttr().Get() if rb else None
sleep = rb.GetSleepThresholdAttr().Get() if rb else None
stab = rb.GetStabilizationThresholdAttr().Get() if rb else None
print(f"[object] gyroscopic={gyro}  sleep_thresh={sleep}  stabilization={stab}")
for prim in Usd.PrimRange(obj):
    if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
        co = PhysxSchema.PhysxCollisionAPI(prim)
        c = co.GetContactOffsetAttr().Get() if co else None
        r = co.GetRestOffsetAttr().Get() if co else None
        ap = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        print(f"[object] collider {prim.GetName()}: approx={ap} "
              f"contact_offset={c} rest_offset={r}")

# ---- 手 collider offsets (取一个指尖 elastomer 看) ----
for name in ["right_thumb_elastomer", "right_index_elastomer"]:
    hits = [p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot"))
            if name in str(p.GetPath()) and p.HasAPI(UsdPhysics.CollisionAPI)]
    for p in hits[:1]:
        co = PhysxSchema.PhysxCollisionAPI(p)
        print(f"[hand] collider {name}: contact_offset={co.GetContactOffsetAttr().Get()} "
              f"rest_offset={co.GetRestOffsetAttr().Get()}")

# ---- 桌子 ----
tbl = stage.GetPrimAtPath("/World/envs/env_0/Table")
for p in Usd.PrimRange(tbl):
    if p.HasAPI(UsdPhysics.CollisionAPI):
        mb = UsdPhysics.MaterialAPI(p)
        print(f"[table] collider {p.GetName()} (材质经 binding, 见 cfg: 0.7/0.7)")
        break

# ---- 物体初始是否稳定: 静置 60 步看漂移 ----
if args.freeze:
    env._ref_t = lambda: torch.full_like(env.episode_length_buf, env.t0)
    print("[settle] freeze 模式: 手悬停 t0 参考帧, 不参与运动")
zero = torch.zeros(env.num_envs, 28, device=env.device)
p0 = env.object.data.root_pos_w.clone()
with torch.inference_mode():
    for _ in range(60):
        env.step(zero)
drift = (env.object.data.root_pos_w - p0).norm(dim=1)
print(f"\n[settle] 物体 60 步(3s)漂移: {[f'{d*100:.2f}cm' for d in drift.tolist()]} "
      f"(手在动, 若被碰到会偏大; <2cm 视为稳定)")
# 分解: 滑动(xy) vs 掉落(z) vs 倾倒(旋转角)
from isaaclab.utils.math import quat_conjugate as _qc, quat_mul as _qm, axis_angle_from_quat as _aa  # noqa: E402
d = env.object.data.root_pos_w - p0
q1 = env.object.data.root_quat_w
q_err = _qm(q1, _qc(env.ref_obj_quat[env.t0].expand_as(q1)))
q_err = q_err * torch.sign(q_err[:, 0:1])
ang = _aa(q_err).norm(dim=1) * 57.3
print(f"[settle] xy 滑动: {[f'{v*100:.1f}cm' for v in d[:, :2].norm(dim=1).tolist()]}  "
      f"z 变化: {[f'{v*100:.1f}cm' for v in d[:, 2].tolist()]}  "
      f"旋转: {[f'{a:.0f}deg' for a in ang.tolist()]}")
print("==========================================")
env.close()
app.close()
