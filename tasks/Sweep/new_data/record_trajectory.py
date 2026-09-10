"""Record the Task3 trajectory-only world. Training/checkpoint loading is absent."""
from pathlib import Path
import argparse,os,sys,json
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--config',required=True)
p.add_argument('--out_dir',required=True)
p.add_argument('--steps',type=int,default=0)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
slot=isaac_slot('task3_trajectory_record')
app=AppLauncher(args).app
import imageio,numpy as np,torch
import omni.usd,omni.replicator.core as rep
from pxr import Usd,UsdGeom,Gf,UsdPhysics
from tasks.Sweep.new_data.trajectory_env import TrajectoryEnv,SE
ROOT=Path(__file__).resolve().parents[3]
out=(ROOT/args.out_dir).resolve()
assert out.is_relative_to(ROOT/'outputs_video')
out.mkdir(parents=True,exist_ok=True)
raw=TrajectoryEnv(SE.build_cfg(1,task_config=args.config))
stage=omni.usd.get_context().get_stage()
cam=UsdGeom.Camera.Define(stage,'/World/Task3Camera');cam.CreateFocalLengthAttr().Set(18);cam.CreateClippingRangeAttr().Set(Gf.Vec2f(.01,100))
m=Gf.Matrix4d();m.SetLookAt(Gf.Vec3d(1.25,-1.65,1.55),Gf.Vec3d(.02,0,1.02),Gf.Vec3d(0,0,1))
UsdGeom.Xformable(cam).AddTransformOp().Set(m.GetInverse())
rp=rep.create.render_product('/World/Task3Camera',(1280,720))
annot=rep.AnnotatorRegistry.get_annotator('rgb');annot.attach(rp)
close=UsdGeom.Camera.Define(stage,'/World/Task3CloseCamera');close.CreateFocalLengthAttr().Set(24);close.CreateClippingRangeAttr().Set(Gf.Vec2f(.01,100))
cm=Gf.Matrix4d();cm.SetLookAt(Gf.Vec3d(.22,-.62,1.38),Gf.Vec3d(0,-.01,.9),Gf.Vec3d(0,0,1))
UsdGeom.Xformable(close).AddTransformOp().Set(cm.GetInverse())
crp=rep.create.render_product('/World/Task3CloseCamera',(1280,720))
ca=rep.AnnotatorRegistry.get_annotator('rgb');ca.attach(crp)
raw.reset()
for _ in range(8):raw.sim.render()
cache=UsdGeom.XformCache()
chain={}
for role,path in [('dustpan','Aux'),('broom','Object')]:
    root=stage.GetPrimAtPath('/World/envs/env_0/'+path)
    root_world=cache.GetLocalToWorldTransform(root);meshes=[]
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            world=cache.GetLocalToWorldTransform(prim)
            meshes.append(dict(path=str(prim.GetPath()),mesh_to_root=np.array(world*root_world.GetInverse()).tolist()))
    chain[role]=dict(root_to_world=np.array(root_world).tolist(),meshes=meshes)
(out/'runtime_coordinate_chain.json').write_text(json.dumps(chain,indent=2)+'\n')
cube=stage.GetPrimAtPath('/World/envs/env_0/SweepCube')
assert UsdGeom.Imageable(cube).ComputeVisibility()=='invisible'
assert all(not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() for p in Usd.PrimRange(cube) if p.HasAPI(UsdPhysics.CollisionAPI))
n=args.steps or raw.T
trace={k:[] for k in ['row','q','q_target','pan_pose','broom_pose','hand_right_pose','hand_left_pose','cum_res']}
writer=imageio.get_writer(str(out/'trajectory.mp4'),fps=20)
cw=imageio.get_writer(str(out/'closeup.mp4'),fps=20)
try:
    with torch.no_grad():
        for t in range(n):
            row=int(raw.row[0])
            raw.step(torch.zeros((1,14),device=raw.device))
            raw.sim.render()
            frame=np.asarray(annot.get_data())[...,:3].copy()
            detail=np.asarray(ca.get_data())[...,:3].copy()
            writer.append_data(frame);cw.append_data(detail)
            if t in [0,n//2,n-1]:
                imageio.imwrite(out/f'frame_{t:04d}.png',frame)
                imageio.imwrite(out/f'closeup_{t:04d}.png',detail)
            trace['row'].append(row)
            trace['q'].append(raw.hand.data.joint_pos[0,raw.map_ids_t].cpu().numpy().copy())
            trace['q_target'].append(raw.q_tgt[0].cpu().numpy().copy())
            trace['cum_res'].append(raw.cum_res[0].cpu().numpy().copy())
            for role,art in [('pan',raw.aux),('broom',raw.object)]:
                trace[role+'_pose'].append(torch.cat([art.data.root_pos_w[0],art.data.root_quat_w[0]]).cpu().numpy().copy())
            for side in ['right','left']:
                bid=raw.hand_bid[side]
                trace['hand_'+side+'_pose'].append(torch.cat([raw.hand.data.body_pos_w[0,bid],raw.hand.data.body_quat_w[0,bid]]).cpu().numpy().copy())
            if t%100==0:print('[Task3] frame',t,'row',row,flush=True)
finally:
    writer.close();cw.close()
np.savez_compressed(out/'trajectory_trace.npz',**{k:np.asarray(v) for k,v in trace.items()})
assert np.max(np.abs(trace['cum_res']))==0
(out/'record_summary.json').write_text(json.dumps(dict(frames=n,reference_rows=raw.T,last_row=trace['row'][-1],
    cube_visible=False,cube_collisions=False,cube_termination=False,trained=False,
    acceptance='pending_visual_and_numeric_review'),indent=2)+'\n')
print('[Task3] complete',out,'frames',n,flush=True)
slot.release();sys.stdout.flush();os._exit(0)
