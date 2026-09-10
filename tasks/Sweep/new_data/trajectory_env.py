"""Task3-only trajectory inspection. No policy, reward or cube-based termination."""
from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tasks/Sweep/2/C_Wiring'))
import sweep_env as SE
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER

class TrajectoryEnv(SE.SweepEnv):
    def _validate_attachment_reset(self):
        # Fail instead of accepting the shared initializer's per-pan calibration.
        for i in (0,1):
            expected=torch.as_tensor(self._z[f'obj_pos_{i}'],device=self.device)
            assert torch.equal(self.ref_pos[i],expected), "Task3 forbids per-tool runtime reference shifts"
        return super()._validate_attachment_reset()

    def _setup_scene(self):
        super()._setup_scene()
        import omni.usd
        from pxr import Usd,UsdGeom,UsdPhysics
        stage=omni.usd.get_context().get_stage()
        for i in range(self.cfg.scene.num_envs):
            root=stage.GetPrimAtPath(f'/World/envs/env_{i}/SweepCube')
            UsdGeom.Imageable(root).MakeInvisible()
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
        print('[Task3] cube invisible, noncolliding, kinematic; no cube reward/done',flush=True)

    def _settle_attachment_reset(self,physics_steps=24):
        # Sweep2's legacy initializer reads PRIOR_BROOM for these finger values.
        # Correct it exclusively in this Task3 subclass, before the first settle.
        values=[]
        for side,prior in [('right',self._broom_prior_npz),('left',self._pan_prior_npz)]:
            values.extend(float(v) for v in prior['grasp'][7:29])
        self.fixed_finger_q=torch.tensor(values,dtype=torch.float32,device=self.device).expand(self.num_envs,-1).clone()
        return super()._settle_attachment_reset(physics_steps)

    def _replace_pan_collision(self):
        import omni.usd
        from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf,Sdf
        stage=omni.usd.get_context().get_stage()
        task=json.loads(Path(self.cfg.sweep_task_config).read_text())
        measured=task.get("trajectory_collision")
        assert measured or task["task_name"]=="SweepP4Take32", "Missing measured task pan collision"
        # take32 measurements, in actual input axes (width=-X, up=-Y, mouth=+Z).
        # No scaled Sweep2 millimetre constants.
        for i in range(self.cfg.scene.num_envs):
            root=stage.GetPrimAtPath(f'/World/envs/env_{i}/Aux')
            if stage.GetPrimAtPath(str(root.GetPath())+'/Task3OpenCollision').IsValid():
                continue  # cloned environments already inherit env_0 colliders
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
            path=str(root.GetPath())+'/Task3OpenCollision'
            f=UsdGeom.Xform.Define(stage,path)
            rotation=measured.get('quat_wxyz',[0,0,0,1]) if measured else [0,0,0,1]
            UsdGeom.Xformable(f).AddOrientOp().Set(Gf.Quatf(float(rotation[0]),*map(float,rotation[1:])))
            # Thin basin slab + descending mouth wedge; walls approximate the
            # visible rim and leave the +Z entry completely open.
            boxes={
                'floor':((0,-.0105,.041),(.145,.005,.062),0),
                'ramp':((0,-.011,.0925),(.140,.003,.029),5.91),
                'left_wall':((-.091,-.002,.058),(.003,.020,.070),0),
                'right_wall':((.091,-.002,.058),(.003,.020,.070),0),
                'back_wall':((0,-.002,.006),(.090,.020,.003),0)}
            if measured:boxes=measured['boxes']
            for name,(center,size,angle) in boxes.items():
                cube=UsdGeom.Cube.Define(stage,path+'/'+name);cube.CreateSizeAttr(1)
                xf=UsdGeom.Xformable(cube);xf.AddTranslateOp().Set(Gf.Vec3d(*center))
                if angle:xf.AddRotateXOp().Set(angle)
                xf.AddScaleOp().Set(Gf.Vec3d(*size))
                cube.CreateVisibilityAttr().Set('invisible')
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
                collision=PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
                collision.CreateContactOffsetAttr(.0002);collision.CreateRestOffsetAttr(0)
        print('[Task3] task-owned measured open pan colliders installed',flush=True)

    def _get_dones(self):
        # Replay every row regardless of provisional cube position or task gates.
        self.row=(self.row+1).clamp(max=self.T-1)
        zeros=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        return zeros,zeros

    def _get_rewards(self):
        return torch.zeros(self.num_envs,device=self.device)

    def _reset_idx(self,env_ids):
        self._tick_out=None
        return super()._reset_idx(env_ids)
