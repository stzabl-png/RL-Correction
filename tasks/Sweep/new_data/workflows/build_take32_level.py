from pathlib import Path
import json,numpy as np,trimesh
from scipy.spatial.transform import Rotation as R
from rl_rebuild.correction.kinematics import ArmIK
root=Path('/home/msc-auto/RL_sweep');run=root/'logs/task1_take32_level_20260910'
old=json.loads((root/'tasks/Sweep/new_data/configs/take_32_universal_fixed_semantic_v1.json').read_text())
donor=json.loads((root/'tasks/Sweep/new_data/configs/take9_powerdisk_fixed_cube15_v1.json').read_text())
z=dict(np.load(root/old['reference'],allow_pickle=True)); n=len(z['left_q'])
# Undo mesh-local semantic baking by composing it into the world pose. This is
# exactly geometry-preserving, and permits canonical donor collision/grasp reuse.
for role,oi in [('dustpan',0),('broom',1)]:
 pr=np.load(root/old['grasppose'][role]['prior']);rr=R.from_quat(z[f'obj_quat_{oi}'],scalar_first=True)
 tr=pr['semantic_take9_to_take32_t']; rot=pr['semantic_take9_to_take32_R']
 z[f'obj_pos_{oi}']=(z[f'obj_pos_{oi}']+rr.apply(tr)).astype(np.float32)
 z[f'obj_quat_{oi}']=(rr*R.from_matrix(rot)).as_quat(scalar_first=True).astype(np.float32)
z['pan_semantic_R_input']=np.eye(3,dtype=np.float32);z['pan_semantic_quat_offset']=np.array([1,0,0,0],np.float32)
v=np.asarray(trimesh.load(root/donor['assets']['dustpan_mesh'],process=False).vertices)
g=np.load(root/donor['grasppose']['dustpan']['prior'])['grasp'];gr=R.from_quat(g[3:7],scalar_first=True).as_matrix()
anchor=np.array(json.loads((root/'tasks/Sweep/2/A_Design/L2_Reference/anchor_T_sweep2.json').read_text())['anchor_T']);ik=ArmIK('left',anchor_link='arm_center',anchor_T=anchor)
oldq=z['left_q'].copy();oldp=z['obj_pos_0'].copy();sol=[];err=[];ang=[];tilts=[]
for i in range(n):
 rr=R.from_quat(z['obj_quat_0'][i],scalar_first=True).as_matrix();tilts.append(float(np.degrees(np.arccos(np.clip(rr[2,1],-1,1)))))
 x=rr[:,0].copy();x[2]=0;x/=np.linalg.norm(x);up=np.array([0.,0.,1.]);rr=np.stack([x,up,np.cross(x,up)],axis=1)
 p=oldp[i].copy();p[2]=.870+.0005-(v@rr.T)[:,2].min()
 target=p+rr@g[:3];rot=rr@gr;seeds=([sol[-1]] if sol else [])+[oldq[i],ik.q_default]
 trials=[]
 for seed in seeds:
  a=ik.solve(target,rot,q0=seed,iters=160,pos_tol=.02,rot_tol=np.radians(5))
  trials.append(a)
  if a['ok']:break
 good=[a for a in trials if a['pos_err']<=.025 and a['rot_err']<=np.radians(5)]
 if not good:
  rng=np.random.default_rng(32000+i)
  extras=[a['q'] for a in trials]+[rng.uniform(ik.lower,ik.upper) for _ in range(8)]
  for seed in extras:
   a=ik.solve(target,rot,q0=seed,iters=500,pos_tol=.02,rot_tol=np.radians(5))
   trials.append(a)
   if a['pos_err']<=.025 and a['rot_err']<=np.radians(5):
    good.append(a);break
 assert good,('IK failed',i,[(a['pos_err'],a['rot_err']) for a in trials])
 a=min(good,key=lambda a:np.linalg.norm(a['q']-seeds[0]));sol.append(a['q']);err.append(a['pos_err']);ang.append(a['rot_err'])
 z['obj_pos_0'][i]=p;z['obj_quat_0'][i]=R.from_matrix(rr).as_quat(scalar_first=True)
 if i%50==0:print('left IK',i,n,'error_mm',1000*a['pos_err'],flush=True)
z['left_q']=np.asarray(sol,np.float32);z['human_left_q']=z['left_q'].copy()
for oi in (0,1):z[f'scene_pose_{oi}']=np.r_[z[f'obj_pos_{oi}'][0],z[f'obj_quat_{oi}'][0]]
steps=np.degrees(np.abs(np.diff(z['left_q'],axis=0))).max(1)
report=dict(rows=n,left_max_error_mm=1000*max(err),left_max_rotation_error_deg=float(np.degrees(max(ang))),left_max_joint_step_deg=float(max(steps)),old_up_axis_tilt_deg=[min(tilts),max(tilts)],new_up_axis_tilt_deg=0,minimum_nominal_pan_table_gap_mm=.5,max_pan_translation_mm=float(1000*np.linalg.norm(z['obj_pos_0']-oldp,axis=1).max()),right_joints_unchanged=True,canonical_donor_asset_and_collision=True)
(run/'level_report.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
assert max(steps)<30,report
ref=root/'tasks/Sweep/new_data/references/take32_fixed_level_cube15_base.npz';np.savez(ref,**z)
cfg=old.copy()
for key in ['assets','grasppose','training_geometry','training_lip_y','pan_floor_profile','pan_floor_tolerance_m','thin_entry_collision','hand_stiffness','hand_damping','finger_pose']:
 cfg[key]=donor[key]
cfg.update(reference=str(ref.relative_to(root)),task_name='Task3Take32FixedLevelCube15',cube_status='training_geometry_calibrated',scripted_prelude_steps=0,training_input={'cube_size_m':.015,'residual_from_step':1},cube_half_m=.0075,acceptance='level_ik_pass_cube_pending')
cfg['universal_asset_substitution']={'method':'donor canonical pose composition; world pan up leveled by left-arm IK','report':str((run/'level_report.json').relative_to(root))}
(root/'tasks/Sweep/new_data/configs/take32_fixed_level_cube15_base.json').write_text(json.dumps(cfg,indent=2)+'\n')
