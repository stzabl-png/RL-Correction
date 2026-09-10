"""Task3 PowerDisk control: identical arm PPO task, optional free physical tools."""
import json
from pathlib import Path
import numpy as np
import torch
from tasks.Sweep.new_data.training_env import TrainingEnv
from tasks.Sweep.new_data.trajectory_env import SE

class PowerDiskEnv(TrainingEnv):
    def __init__(self,cfg,**kw):
        self.powerdisk=json.loads(Path(cfg.sweep_task_config).read_text())
        self.free_tools=self.powerdisk['grip_mode']=='contact'
        self.prelude=int(self.powerdisk['scripted_prelude_steps'])
        cfg.robot_cfg.actuators['hands'].stiffness=float(self.powerdisk.get('hand_stiffness',20.0))
        cfg.robot_cfg.actuators['hands'].damping=float(self.powerdisk.get('hand_damping',2.0))
        # fixed_attached_tools in the legacy prior loader means "preserve provided
        # initial pose"; actual constraint creation is controlled below.
        super().__init__(cfg,**kw)

    def _replace_pan_collision(self):
        self.pan_scale=np.ones(3)
        cfg=json.loads(Path(self.cfg.sweep_task_config).read_text())
        if cfg.get('thin_entry_collision'):
            import omni.usd
            from pxr import PhysxSchema,Usd,UsdGeom,UsdPhysics,Vt
            import trimesh
            stage=omni.usd.get_context().get_stage()
            mesh=trimesh.load(cfg['assets']['dustpan_mesh'],force='mesh',process=False)
            parts=json.loads(Path(cfg['thin_entry_collision']).read_text())
            handle=trimesh.convex.convex_hull(np.asarray(mesh.vertices)[np.asarray(mesh.vertices)[:,2]<-.012])
            parts=parts+[dict(name='handle',vertices=handle.vertices.tolist(),faces=handle.faces.tolist())]
            for i in range(self.cfg.scene.num_envs):
                root=stage.GetPrimAtPath(f'/World/envs/env_{i}/Aux')
                for prim in Usd.PrimRange(root):
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                for part in parts:
                    m=UsdGeom.Mesh.Define(stage,str(root.GetPath())+'/ThinEntryCollision/'+part['name'])
                    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(part['vertices'],np.float32)))
                    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(part['faces']),3,np.int32)))
                    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(part['faces'],np.int32).ravel()))
                    m.CreateVisibilityAttr('invisible')
                    UsdPhysics.CollisionAPI.Apply(m.GetPrim()).CreateCollisionEnabledAttr(True)
                    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
                    pc=PhysxSchema.PhysxCollisionAPI.Apply(m.GetPrim())
                    pc.CreateContactOffsetAttr(.0001);pc.CreateRestOffsetAttr(0)
            print('[PowerDisk] final thin-entry collision pieces',len(parts),flush=True)
            return
        SE.SweepEnv._replace_pan_collision(self)
        import omni.usd
        from pxr import UsdGeom,UsdPhysics,Vt
        import trimesh
        stage=omni.usd.get_context().get_stage()
        cfg=json.loads(Path(self.cfg.sweep_task_config).read_text())
        mesh=trimesh.load(cfg['assets']['dustpan_mesh'],force='mesh',process=False)
        # Original Sweep2 pan handle: preserve its true vertices, do not invent a
        # large invisible grasp support. Convex hull is limited to the handle.
        points=np.asarray(mesh.vertices);points=points[points[:,2]<-.012]
        hull=trimesh.convex.convex_hull(points)
        for i in range(self.cfg.scene.num_envs):
            p=f'/World/envs/env_{i}/Aux/PowerDiskHandle'
            if stage.GetPrimAtPath(p).IsValid():continue
            m=UsdGeom.Mesh.Define(stage,p)
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(hull.vertices,np.float32)))
            m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(hull.faces),3,np.int32)))
            m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(hull.faces,np.int32).ravel()))
            m.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(m.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
        print('[PowerDisk] original pan open basin + measured handle collider',flush=True)

    def _create_tool_joints(self):
        import omni.usd
        import isaaclab.sim as sim
        from pxr import Sdf,Usd,UsdPhysics,PhysxSchema
        stage=omni.usd.get_context().get_stage()
        handmat=sim.RigidBodyMaterialCfg(static_friction=self.powerdisk['hand_friction'],dynamic_friction=self.powerdisk['hand_friction'],restitution=0.,friction_combine_mode='multiply')
        handmat.func('/World/Materials/PowerDiskHand',handmat)
        toolmat=sim.RigidBodyMaterialCfg(static_friction=3.,dynamic_friction=3.,restitution=0.,friction_combine_mode='multiply')
        toolmat.func('/World/Materials/PowerDiskTool',toolmat)
        for i in range(self.cfg.scene.num_envs):
            env=f'/World/envs/env_{i}'
            for role in ('Object','Aux'):
                sim.bind_physics_material(env+'/'+role,'/World/Materials/PowerDiskTool')
                root=stage.GetPrimAtPath(env+'/'+role)
                UsdPhysics.RigidBodyAPI(root).CreateKinematicEnabledAttr(False)
                PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateDisableGravityAttr(False)
            for prim in Usd.PrimRange(stage.GetPrimAtPath(env+'/Robot')):
                if prim.HasAPI(UsdPhysics.CollisionAPI) and any('/'+s+'_' in str(prim.GetPath()) for s in ('right','left')):
                    sim.bind_physics_material(str(prim.GetPath()),'/World/Materials/PowerDiskHand')
            # The reconstructed frame zero places the brush head inside the open
            # pan.  Keep both tools dynamic, but exclude their mutual collision so
            # that this valid sweep contact does not inject an initialization
            # impulse.  Hand/tool and cube/tool contacts remain physical.
            UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(env+'/Object')) \
                .CreateFilteredPairsRel().AddTarget(Sdf.Path(env+'/Aux'))
        if not self.free_tools:
            SE.SweepEnv._create_tool_joints(self)
        print('[PowerDisk] grip_mode=',self.powerdisk['grip_mode'],'hand_mu=',self.powerdisk['hand_friction'],'tool_mu=3; fingers position-controlled',flush=True)

    def _settle_attachment_reset(self,physics_steps=24):
        # Initialize directly in delivered grasp. No hidden temporary weld, pose
        # reattachment, tool stabilization force, or tool writes after reset.
        assert self.powerdisk.get('finger_pose','grasp') == 'grasp'
        values=np.concatenate([p['grasp'][7:29] for p in (self._broom_prior_npz,self._pan_prior_npz)])
        self.fixed_finger_q=torch.as_tensor(values,dtype=torch.float32,device=self.device).expand(self.num_envs,-1).clone()
        tighten=float(self.powerdisk.get('grip_tighten_rad',0.0))
        if tighten:
            flex_names={'thumb_IP','thumb_MCP_FE','index_MCP_FE','index_PIP','index_DIP','middle_MCP_FE','middle_PIP','middle_DIP','ring_MCP_FE','ring_PIP','ring_DIP','pinky_MCP_FE','pinky_PIP','pinky_DIP'}
            generic=('thumb_CMC_FE','thumb_CMC_AA','thumb_MCP_FE','thumb_MCP_AA','thumb_IP','index_MCP_FE','index_MCP_AA','index_PIP','index_DIP','middle_MCP_FE','middle_MCP_AA','middle_PIP','middle_DIP','ring_MCP_FE','ring_MCP_AA','ring_PIP','ring_DIP','pinky_CMC','pinky_MCP_FE','pinky_MCP_AA','pinky_PIP','pinky_DIP')
            lim=self.hand.data.joint_pos_limits[0]
            for side in ('right','left'):
                for name in flex_names:
                    gi=(0 if side=='right' else 22)+generic.index(name)
                    jid=self.hand.joint_names.index(f'{side}_{name}')
                    self.fixed_finger_q[:,gi]=torch.clamp(self.fixed_finger_q[:,gi]+tighten,lim[jid,0],lim[jid,1])
            print('[PowerDisk] grasp hold tightening rad=',tighten,flush=True)
        q=self.hand.data.default_joint_pos.clone();q[:,self.map_ids_t]=self.ref_arm[0];q[:,self.fixed_finger_ids]=self.fixed_finger_q
        self.hand.write_joint_state_to_sim(q,torch.zeros_like(q));self.hand.set_joint_position_target(q)
        for oi,art in ((0,self.aux),(1,self.object)):
            pose=torch.cat([self.ref_pos[oi][0].expand(self.num_envs,3),self.ref_quat[oi][0].expand(self.num_envs,4)],1).clone();pose[:,:3]+=self.scene.env_origins
            art.write_root_pose_to_sim(pose);art.write_root_velocity_to_sim(torch.zeros(self.num_envs,6,device=self.device))

    def _validate_attachment_reset(self):
        import omni.usd
        from pxr import UsdPhysics
        stage=omni.usd.get_context().get_stage()
        for i in (0,1):
            assert torch.equal(self.ref_pos[i],torch.as_tensor(self._z[f'obj_pos_{i}'],device=self.device))
        joints=[p for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint) and 'tool_fixed' in str(p.GetPath())]
        assert len(joints)==(0 if self.free_tools else 2*self.num_envs)
        print('[PowerDisk] verified tool FixedJoint count',len(joints),flush=True)

    def _load_broom_working_points(self):
        import trimesh
        mesh=trimesh.load(self.powerdisk['assets']['broom_mesh'],force='mesh',process=False)
        v=np.asarray(mesh.vertices)
        points=v[(v[:,1]<-.050)&(v[:,2]>.020)&(v[:,2]<.090)]
        assert len(points)>=2048
        points=points[np.random.RandomState(0).choice(len(points),2048,replace=False)]
        # Task-owned USD is explicitly baked with mesh_to_root identity.
        return points.astype(np.float32),np.asarray(self._z['brush_contact_local'],np.float32)

    def _pre_physics_step(self,actions):
        self.prev_act=self.last_act.clone()
        active=self.episode_length_buf>=self.prelude
        self.last_act=actions.clamp(-1,1)*active.unsqueeze(1)
        r=self.row.clamp(max=self.T-1);confidence=self.conf[r]
        c14=torch.cat([confidence[:,:1].expand(-1,7),confidence[:,1:].expand(-1,7)],1)
        step=self.step_hi+c14*(self.step_lo-self.step_hi);dev=self.dev_hi+c14*(self.dev_lo-self.dev_hi)
        self.cum_res=torch.maximum(torch.minimum(self.cum_res+self.last_act*step,dev),-dev)
        self.q_tgt=self.ref_arm[r]+self.cum_res

    def _get_observations(self):
        obs=super()._get_observations()
        active=(self.episode_length_buf>=self.prelude).float().unsqueeze(1)
        obs['policy'][:,-1:]=active;obs['actor_mask']=active
        return obs
