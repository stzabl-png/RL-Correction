from pathlib import Path
import numpy as np,json,trimesh
from scipy.spatial.transform import Rotation as R
from rl_rebuild.correction.kinematics import ArmIK
base=Path('tasks/Sweep/new_data/assets/take9_powerdisk/dustpan_thin_entry_20260909');v=np.asarray(trimesh.load(base/'object_mesh_scaled_final.obj',process=False).vertices)
z=dict(np.load('tasks/Sweep/new_data/references/take9_powerdisk_v7_pan_pitch5.npz',allow_pickle=True));cfg=json.loads(Path('tasks/Sweep/new_data/configs/take9_powerdisk_fixed.json').read_text())
anchor=np.array(json.loads(Path('tasks/Sweep/2/A_Design/L2_Reference/anchor_T_sweep2.json').read_text())['anchor_T']);ik=ArmIK('left',anchor_link='arm_center',anchor_T=anchor)
g=np.load(cfg['grasppose']['dustpan']['prior'])['grasp'];gr=R.from_quat(g[[4,5,6,3]]).as_matrix();pold=z['obj_pos_0'].copy();qold=z['obj_quat_0'].copy();oldarm=z['left_q'].copy();ans=[];clear=[]
for i,(p,q) in enumerate(zip(pold,qold)):
 rr=R.from_quat(q[[1,2,3,0]]).as_matrix();x=rr[:,0].copy();x[2]=0;x/=np.linalg.norm(x);forward=np.cross(x,[0,0,1]);pitch=np.deg2rad(3)
 y=np.cos(pitch)*np.array([0,0,1])+np.sin(pitch)*forward;front=np.cos(pitch)*forward-np.sin(pitch)*np.array([0,0,1]);rr=np.stack([x,y,front],1)
 p=p.copy();p[0]-=.02;p[2]=.870-(v@rr.T)[:,2].min();t=p+rr@g[:3];rot=rr@gr
 seeds=[ans[-1]['q']] if ans else [oldarm[i]]
 seeds.append(oldarm[i]);trials=[ik.solve(t,rot,q0=s,iters=200,pos_tol=.0001,rot_tol=.001) for s in seeds];valid=[r for r in trials if r['ok']]
 r=min(valid,key=lambda r:np.linalg.norm(r['q']-seeds[0])) if valid else min(trials,key=lambda r:r['pos_err']+r['rot_err']*.1)
 ans.append(r);z['obj_pos_0'][i]=p;z['obj_quat_0'][i]=R.from_matrix(rr).as_quat()[[3,0,1,2]]
 lip=v[(v[:,2]>.104)&(abs(v[:,0])<.06)]@rr.T+p;clear.append([lip[:,2].min()-.87,lip[:,2].max()-.87])
 if i%100==0:print(i,r['ok'],r['pos_err'],flush=True)
rep={'ok_ratio':float(np.mean([r['ok'] for r in ans])),'pos_max_mm':max(r['pos_err'] for r in ans)*1000,'rot_max_deg':np.rad2deg(max(r['rot_err'] for r in ans)),'joint_step_max_deg':float(np.rad2deg(abs(np.diff(np.array([r['q'] for r in ans]),axis=0))).max()),'lip_band_world_gap_mm':(np.array(clear).max(0)*1000).tolist(),'pan_translation_max_mm':float(np.linalg.norm(z['obj_pos_0']-pold,axis=1).max()*1000)}
print(rep,flush=True);(base/'reference_report.json').write_text(json.dumps(rep,indent=2))
assert rep['ok_ratio']==1 and rep['joint_step_max_deg']<8
z['left_q']=np.array([r['q'] for r in ans],np.float32);z['scene_pose_0']=np.r_[z['obj_pos_0'][0],z['obj_quat_0'][0]];z['acceptance']=np.array('asset_contact_preview_only_not_training');np.savez(base/'contact_reference.npz',**z)
cfg['reference']=str(base/'contact_reference.npz');cfg['assets']['dustpan_mesh']=str(base/'object_mesh_scaled_final.obj');cfg['assets']['dustpan_usd']=str(base/'dustpan.usd');cfg['thin_entry_collision']=str(base/'collision.json');cfg['asset_provenance']='20260909 thin mouth, level lateral entry, 3deg pitch; diagnostic only';cfg['cube_status']='disabled_for_asset_review';(base/'preview_config.json').write_text(json.dumps(cfg,indent=2))
