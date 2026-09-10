"""Apply ONE shared initial Z registration and re-solve all arm targets."""
import argparse,ast,json
from pathlib import Path
import numpy as np
from rl_rebuild.correction.kinematics import ArmIK
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--base',required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[3];cfg=json.loads((ROOT/a.config).read_text());z=dict(np.load(ROOT/a.base,allow_pickle=True))
assert Path(str(z['data_dir'])).resolve()==(ROOT/cfg['data_dir']).resolve()
tree=ast.parse((ROOT/'tasks/Sweep/new_data/build_reference.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'builder helpers','exec'),globals())
dz=float(z['source_table_z'])-cfg['scene_table_z'];assert abs(dz)<.02
shift=np.array([0,0,dz]);n=len(z['source_frame']);anchor=np.array(json.loads((ROOT/'tasks/Sweep/2/A_Design/L2_Reference/anchor_T_sweep2.json').read_text())['anchor_T'])
for i in (0,1):
 z[f'obj_pos_{i}']=(z[f'obj_pos_{i}']+shift).astype(np.float32);z[f'scene_pose_{i}']=z[f'scene_pose_{i}'].copy();z[f'scene_pose_{i}'][:3]+=shift
report={}
for side,role,i in [('right','broom',1),('left','dustpan',0)]:
 ik=ArmIK(side,anchor_link='arm_center',anchor_T=anchor)
 g=np.load(ROOT/cfg['grasppose'][role]['prior'])['grasp']
 tool=np.array([pose_T(p,q) for p,q in zip(z[f'obj_pos_{i}'],z[f'obj_quat_{i}'])]);target=tool@pose_T(g[:3],g[3:7])
 data=ROOT/cfg['data_dir'];hp=data/f'retarget/ref_qpos_{side}.npz'
 if not hp.exists():hp=data/f'ref_qpos_{side}.npz'
 h=np.load(hp,allow_pickle=True);w=interp_T(np.array([pose_T(p,q) for p,q in zip(h['wrist_pos'],h['wrist_quat_wxyz'])]),z['source_frame'].astype(float))
 yaw=np.radians(float(z['scene_yaw_deg']));rw=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
 human=w.copy()
 for j,t in enumerate(w):
  human[j,:3,3]=target[0,:3,3]+rw@(t[:3,3]-w[0,:3,3])
  human[j,:3,:3]=rw@(t[:3,:3]@w[0,:3,:3].T)@rw.T@target[0,:3,:3]
 for key,targets in [(side+'_q',target),('human_'+side+'_q',human)]:
  ans=[]
  for j,(old,t) in enumerate(zip(z[key],targets)):
   r=ik.solve(t[:3,3],t[:3,:3],q0=old,iters=300,pos_tol=.002,rot_tol=.02)
   if not r['ok'] and ans:r=ik.solve(t[:3,3],t[:3,:3],q0=ans[-1]['q'],iters=400,pos_tol=.002,rot_tol=.02)
   ans.append(r)
  z[key]=np.array([r['q'] for r in ans],np.float32)
  report[key]=dict(ok_ratio=float(np.mean([r['ok'] for r in ans])),pos_max_mm=float(max(r['pos_err'] for r in ans)*1000),rot_max_deg=float(np.degrees(max(r['rot_err'] for r in ans))),joint_step_max_deg=float(np.degrees(abs(np.diff(z[key],axis=0))).max()))
  print(key,report[key],flush=True)
z['source_table_z']=np.float32(cfg['scene_table_z']);z['shared_registration_delta_z_m']=np.float64(dz);z['registration_report']=np.array(json.dumps(report));z['acceptance']=np.array('numeric_candidate_pending_physics')
out=ROOT/cfg['reference'];np.savez(str(out)+'.candidate.npz',**z)
for key,r in report.items():
 assert r['ok_ratio']>=.99 and r['pos_max_mm']<5 and r['rot_max_deg']<2,report
 assert r['joint_step_max_deg']<=(12 if key.startswith('human') else 8),report
np.savez(out,**z);print('wrote',out,'shared dz',dz,flush=True)
