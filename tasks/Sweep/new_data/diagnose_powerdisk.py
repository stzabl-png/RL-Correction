"""Bounded real-physics grip and trajectory audit; no optimizer or policy training."""
from pathlib import Path
import argparse,json,os,sys
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out_dir',required=True);p.add_argument('--steps',type=int,default=0);p.add_argument('--hold_steps',type=int,default=40);p.add_argument('--record',action='store_true');p.add_argument('--num_envs',type=int,default=1)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
slot=isaac_slot('task3_powerdisk_diagnostic');app=AppLauncher(a).app
import numpy as np,torch
from tasks.Sweep.new_data.powerdisk_env import PowerDiskEnv,SE
import omni.usd
from pxr import UsdGeom,UsdPhysics,Usd,Gf
class Replay(PowerDiskEnv):
    def _setup_scene(self):
        super()._setup_scene();stage=omni.usd.get_context().get_stage()
        for i in range(self.cfg.scene.num_envs):
            root=stage.GetPrimAtPath(f'/World/envs/env_{i}/SweepCube');UsdGeom.Imageable(root).MakeInvisible()
            for p in Usd.PrimRange(root):
                if p.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
            UsdPhysics.RigidBodyAPI(root).CreateKinematicEnabledAttr(True)
    def _get_dones(self):
        if not getattr(self,'hold_reference',False):self.row=(self.row+1).clamp(max=self.T-1)
        z=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device);return z,z
    def _get_rewards(self):return torch.zeros(self.num_envs,device=self.device)
raw=Replay(SE.build_cfg(a.num_envs,task_config=a.config));out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
raw.reset();stage=omni.usd.get_context().get_stage()
writers=[];annots=[]
if a.record:
 import imageio,omni.replicator.core as rep
 for name,eye,target,focal in [('overview',(1.25,-1.65,1.55),(.02,0,1.02),18),('closeup',(.22,-.62,1.38),(0,-.01,.9),24)]:
  cam=UsdGeom.Camera.Define(stage,'/World/PowerDisk_'+name);cam.CreateFocalLengthAttr(focal);cam.CreateClippingRangeAttr(Gf.Vec2f(.01,100));m=Gf.Matrix4d();m.SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1));UsdGeom.Xformable(cam).AddTransformOp().Set(m.GetInverse())
  rp=rep.create.render_product(str(cam.GetPath()),(960,540));an=rep.AnnotatorRegistry.get_annotator('rgb');an.attach(rp);annots.append(an);writers.append(imageio.get_writer(str(out/(name+'.mp4')),fps=20))
 for _ in range(8):raw.sim.render()
trace=[];n=a.steps or raw.T
from isaaclab.utils.math import quat_apply,quat_conjugate,quat_mul
with torch.no_grad():
 for t in range(a.hold_steps+n):
  raw.hold_reference=t<a.hold_steps
  raw.step(torch.zeros(raw.num_envs,14,device=raw.device))
  item={'t':t,'row':int(raw.row[0]),'q':raw.hand.data.joint_pos[0,raw.map_ids_t].cpu().tolist(),'finger_target':raw.fixed_finger_q[0].cpu().tolist(),'finger_actual':raw.hand.data.joint_pos[0,raw.fixed_finger_ids].cpu().tolist()}
  for side,role,art,prior in [('right','broom',raw.object,raw._broom_prior_npz),('left','pan',raw.aux,raw._pan_prior_npz)]:
   bid=raw.hand_bid[side];hp=raw.hand.data.body_pos_w[:,bid];hq=raw.hand.data.body_quat_w[:,bid];op=art.data.root_pos_w;oq=art.data.root_quat_w
   g=torch.as_tensor(prior['grasp'][:7],device=raw.device,dtype=op.dtype)
   p=op+quat_apply(oq,g[:3].expand(raw.num_envs,3));q=quat_mul(oq,g[3:].expand(raw.num_envs,4))
   item[role+'_pose']=torch.cat([op[0],oq[0]]).cpu().tolist();item[side+'_hand_pose']=torch.cat([hp[0],hq[0]]).cpu().tolist()
   item[role+'_slip_mm']=float((hp-p).norm(dim=1).max()*1000);item[role+'_slip_deg']=float(torch.rad2deg(SE._qangle(hq,q)).max())
  trace.append(item)
  if a.record:
   raw.sim.render()
   for j,(an,w) in enumerate(zip(annots,writers)):
    f=np.asarray(an.get_data())[...,:3];w.append_data(f)
    if t in (0,a.hold_steps-1,a.hold_steps+n//2,a.hold_steps+n-1):imageio.imwrite(out/f'{j}_frame_{t:04d}.png',f)
  if t%50==0:print('tick',t,{k:v for k,v in item.items() if 'slip' in k},flush=True)
for w in writers:w.close()
(out/'trace.json').write_text(json.dumps(trace))
summary={'config':a.config,'mode':raw.powerdisk['grip_mode'],'hold_steps':a.hold_steps,'motion_steps':n,'reference_rows':raw.T,'max_slip':{k:max(i[k] for i in trace) for k in ('broom_slip_mm','broom_slip_deg','pan_slip_mm','pan_slip_deg')},'after_hold':{k:v for k,v in trace[a.hold_steps-1].items() if 'slip' in k},'last':{k:v for k,v in trace[-1].items() if 'slip' in k},'hand_friction':raw.powerdisk['hand_friction'],'tool_fixed_joints':0 if raw.free_tools else 2*raw.num_envs}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
slot.release();sys.stdout.flush();os._exit(0)
