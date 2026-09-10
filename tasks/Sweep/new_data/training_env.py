"""Task3 asset adapter; inherits Sweep2's observations, PPO task rewards and dones."""
import json
from pathlib import Path
from tasks.Sweep.new_data.trajectory_env import TrajectoryEnv, SE

class TrainingEnv(SE.SweepEnv):
    # Reuse only accepted physical assets and finger reset, never trajectory-only
    # scene, reward, reset or progression overrides.
    _replace_pan_collision = TrajectoryEnv._replace_pan_collision
    def _setup_scene(self):
        super()._setup_scene()
        task = json.loads(Path(self.cfg.sweep_task_config).read_text())
        half = float(task["training_geometry"]["cube_half"])
        if abs(half - .0125) < 1e-9:
            return
        # Before physics starts, resize the same USD Cube used for both display
        # and collision. The base Sweep2 spawner and its 25 mm default stay intact.
        import omni.usd
        from pxr import Usd, UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        count = 0
        for i in range(self.cfg.scene.num_envs):
            cube_root = stage.GetPrimAtPath(f"/World/envs/env_{i}/SweepCube")
            shapes = [prim for prim in Usd.PrimRange(cube_root) if prim.IsA(UsdGeom.Cube)]
            assert len(shapes) == 1, "Expected the single shared visual/collision cube"
            shape = UsdGeom.Cube(shapes[0])
            shape.GetSizeAttr().Set(2 * half)
            shape.CreateExtentAttr().Set([Gf.Vec3f(-half), Gf.Vec3f(half)])
            count += 1
        self.cube.cfg.spawn.size = (2*half,) * 3
        print(f"[Task3] cube visual/collision side={2*half:.6f}m count={count}", flush=True)

    def _validate_attachment_reset(self):
        import torch
        for i in (0, 1):
            expected = torch.as_tensor(self._z[f"obj_pos_{i}"], device=self.device)
            assert torch.equal(self.ref_pos[i], expected), "Task3 forbids runtime reference shifts"
        return super()._validate_attachment_reset()

    def _settle_attachment_reset(self, physics_steps=24):
        import torch
        values = [float(v) for prior in (self._broom_prior_npz, self._pan_prior_npz)
                  for v in prior["grasp"][7:29]]
        self.fixed_finger_q = torch.tensor(values, dtype=torch.float32,
            device=self.device).expand(self.num_envs, -1).clone()
        return super()._settle_attachment_reset(physics_steps)

    def __init__(self, cfg, **kwargs):
        task = json.loads(Path(cfg.sweep_task_config).read_text())
        assert task.get("cube_status") == "training_geometry_calibrated"
        self._task_lip_y = task.get("training_lip_y", -.0145)
        self._pan_floor_profile = task.get("pan_floor_profile")
        self._pan_floor_tolerance = float(task.get("pan_floor_tolerance_m", .001))
        super().__init__(cfg, **kwargs)
        self.geometry = SE.SweepGeometry(**task["training_geometry"])
        # SweepEnv initialized this height using its frozen 25 mm baseline.
        # Match Task3 reset position to the resized physical cube before reset.
        self.cube_start_ref[2] = float(cfg.table_top_z) + self.geometry.cube_half + .0005
        self.progress = SE.SweepProgressBatch(self.num_envs, self.device, self.geometry)

    def _signals(self):
        import torch
        from isaaclab.utils.math import quat_apply
        sig, ready, near = super()._signals()
        pan_p, pan_q = self._semantic_pan_pose()
        width = self.geometry.pan_half_width
        mouth = self.geometry.pan_mouth_z
        lip = torch.tensor([[-width, self._task_lip_y, mouth], [width, self._task_lip_y, mouth]],
                           device=self.device, dtype=pan_p.dtype)
        world = quat_apply(pan_q[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
                           lip.expand(self.num_envs, -1, -1).reshape(-1, 3))
        world = world.reshape(self.num_envs, 2, 3) + pan_p[:, None, :]
        sig["mouth_clearance"] = world[:, :, 2].amin(dim=1) - self.cfg.table_top_z
        if self._pan_floor_profile:
            # A thin-entry pan has a descending work surface: one constant
            # centre-height bound either rejects valid mouth entry or accepts a
            # cube underneath the raised rear basin.  Interpolate the measured
            # inner floor in pan coordinates and require the cube's bottom face
            # to sit on or above it.  This changes only Task3 geometry signals.
            profile = torch.as_tensor(self._pan_floor_profile, device=pan_p.device,
                                      dtype=pan_p.dtype)
            c = sig["cube_pan"]
            idx = torch.bucketize(c[:, 2], profile[:, 0]).clamp(1, len(profile)-1)
            lo, hi = profile[idx-1], profile[idx]
            alpha = ((c[:, 2] - lo[:, 0]) /
                     (hi[:, 0] - lo[:, 0]).clamp_min(1e-6)).clamp(0, 1)
            floor_y = lo[:, 1] + alpha * (hi[:, 1] - lo[:, 1])
            above_floor = (c[:, 1] - self.geometry.cube_half >=
                           floor_y - self._pan_floor_tolerance)
            sig["above_pan_floor"] = above_floor
            for key in ("entered", "fully_inside", "deep_inside"):
                sig[key] &= above_floor
            for key in ("progress", "full_progress", "deep_progress"):
                sig[key] *= above_floor.float()
        return sig, ready, near
