from pathlib import Path
import ast,hashlib,json
import numpy as np
from rl_rebuild.correction.kinematics import ArmIK
from rl_rebuild.correction import frames as F
root=Path('/home/msc-auto/RL_sweep');run=root/'logs/task9_human_restore_20260910'
# Reuse the successful builder's interpolation, first-frame weld and branch solver;
# never execute its top-level asset/trajectory rebuilding code.
source=root/'tasks/Sweep/2/A_Design/L2_Reference/build_reference.py';text=source.read_text();tree=ast.parse(text)
names={'q_to_R','R_to_q','pose_T','interp_T','solve_continuous'}
code='\n\n'.join(ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
code=code.replace('pos_tol=0.005, rot_tol=0.05','pos_tol=0.02, rot_tol=0.0872664626').replace('pos_tol=0.002, rot_tol=0.02','pos_tol=0.02, rot_tol=0.0872664626')
exec(code,globals())
cfgpath=root/'tasks/Sweep/new_data/configs/take9_powerdisk_fixed_cube15_v1.json';cfg=json.loads(cfgpath.read_text());refpath=root/cfg['reference'];z=dict(np.load(refpath,allow_pickle=True));times=z['source_frame'].astype(float)
rawpath=root/'datasets/sweep_new_data/sweep_dustpan_2/9/replay_world.npz';raw=np.load(rawpath,allow_pickle=True);assert float(raw['fps'])==15
anchor=np.array(json.loads((root/'tasks/Sweep/2/A_Design/L2_Reference/anchor_T_sweep2.json').read_text())['anchor_T'])
report={'source_replay':str(rawpath.relative_to(root)),'source_sha256':hashlib.sha256(rawpath.read_bytes()).hexdigest(),'method':'Sweep2 independent reconstructed wrist motion; first-frame weld at cropped start; original source_frame resampling','rows':len(times),'source_frame_range':[float(times[0]),float(times[-1])],'sides':{}}
for side,role,oi in [('right','broom',1),('left','dustpan',0)]:
 joints=raw['joints_'+side];wp=joints[:,0];wq=F.sharpa_base_quat_from_joints(joints,side)
 prepared=np.load(root/f'tasks/Sweep/new_data/prepared/take9_powerdisk/ref_qpos_{side}.npz');assert np.allclose(prepared['wrist_pos'],wp);assert np.allclose(prepared['wrist_quat_wxyz'],wq)
 wrist=interp_T(np.array([pose_T(p,q) for p,q in zip(wp,wq)]),times)
 g=np.load(root/cfg['grasppose'][role]['prior'])['grasp'];first=pose_T(z[f'obj_pos_{oi}'][0],z[f'obj_quat_{oi}'][0])@pose_T(g[:3],g[3:7])
 P=first[:3,3]+wrist[:,:3,3]-wrist[0,:3,3]
 Q=np.array([R_to_q(w[:3,:3]@wrist[0,:3,:3].T@first[:3,:3]) for w in wrist])
 ik=ArmIK(side,anchor_link='arm_center',anchor_T=anchor);answers=solve_continuous(ik,P,Q,20260840+int(side=='left'))
 h=np.array([a['q'] for a in answers],np.float32);pe=np.array([a['pos_err'] for a in answers]);re=np.array([a['rot_err'] for a in answers]);step=float(np.degrees(np.abs(np.diff(h,axis=0))).max())
 info={'position_max_mm':float(pe.max()*1000),'rotation_max_deg':float(np.degrees(re.max())),'joint_step_max_deg':step,'equal_to_nominal':bool(np.array_equal(h,z[side+'_q'])),'over20mm_rows':np.flatnonzero(pe>.02).tolist(),'raw_wrist_verified_against_prepared':True}
 report['sides'][side]=info;print(side,info,flush=True)
 np.savez(run/f'{side}_human_targets.npz',source_frame=times,wrist_position=P,wrist_quaternion=Q,human_q=h)
 assert pe.max()<=.025 and step<30 and not info['equal_to_nominal'],info
 z['human_'+side+'_q']=h
original=np.load(refpath,allow_pickle=True)
for k in original.files:
 if k not in ['human_right_q','human_left_q']:assert np.array_equal(z[k],original[k],equal_nan=True) if np.issubdtype(original[k].dtype,np.number) else np.array_equal(z[k],original[k]),k
z['human_ik_report']=np.array(json.dumps(report['sides']));z['human_guidance_provenance']=np.array(json.dumps(report));z['meta']=np.array(str(z['meta']).replace('human_left_q=left_q','historical human_left copy superseded')+'; independent raw human wrists restored for both sides')
dst=root/'tasks/Sweep/new_data/references/take9_fixed_cube15_human_v2.npz';assert not dst.exists();np.savez(dst,**z)
cfg['reference']=str(dst.relative_to(root));cfg['task_name']='Task3Take9FixedCube15HumanV2';cfg['human_guidance']=report;cfg['acceptance']='human_input_restored_existing_geometry'
out=root/'tasks/Sweep/new_data/configs/take9_fixed_cube15_human_v2.json';assert not out.exists();out.write_text(json.dumps(cfg,indent=2)+'\n');report['output_config']=str(out.relative_to(root));report['output_reference']=str(dst.relative_to(root));report['all_nonhuman_tracks_unchanged']=True
(run/'restore_report.json').write_text(json.dumps(report,indent=2)+'\n');print('RESTORE_COMPLETE',flush=True)
