"""Recompute one changed grasp side using the same builder solver and full path."""
import argparse,ast,json,os
from pathlib import Path
import numpy as np
from rl_rebuild.correction.kinematics import ArmIK
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--base',required=True);p.add_argument('--side',choices=['left','right'],required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[3]
cfg=json.loads((ROOT/a.config).read_text());z=dict(np.load(ROOT/a.base,allow_pickle=True));side=a.side
assert Path(str(z['data_dir'])).resolve()==(ROOT/cfg['data_dir']).resolve()
source=ast.parse((ROOT/'tasks/Sweep/new_data/build_reference.py').read_text())
functions=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
exec(compile(functions,'build_reference.py helper definitions','exec'),globals())
anchor=np.array(json.loads((ROOT/'tasks/Sweep/2/A_Design/L2_Reference/anchor_T_sweep2.json').read_text())['anchor_T'])
oi=0 if side=='left' else 1;role='dustpan' if oi==0 else 'broom'
g=np.load(ROOT/cfg['grasppose'][role]['prior'])['grasp']
tools=np.array([pose_T(p,q) for p,q in zip(z[f'obj_pos_{oi}'],z[f'obj_quat_{oi}'])])
grasp=pose_T(g[:3],g[3:7]);target=tools@grasp
ik=ArmIK(side,anchor_link='arm_center',anchor_T=anchor)
P=target[:,:3,3];Q=np.array([R_to_q(t[:3,:3]) for t in target])
report={}
def solve_store(key,P,Q,seed):
 ans=solve_continuous(ik,P,Q,seed)
 qs=np.array([r['q'] for r in ans],np.float32)
 report[key]=dict(ok_ratio=float(np.mean([r['ok'] for r in ans])),pos_max_cm=float(max(r['pos_err'] for r in ans)*100),
 rot_max_deg=float(np.degrees(max(r['rot_err'] for r in ans))),joint_step_max_deg=float(np.degrees(abs(np.diff(qs,axis=0))).max()))
 z[key]=qs;print(key,report[key],flush=True)
solve_store(side+'_q',P,Q,20260830+int(side=='left'))
# Do not spend a second solve on human motion after a clearly discontinuous robot path.
if report[side+'_q']['joint_step_max_deg']>30:
    failure=ROOT/(cfg['reference']+'.robot_failure.json')
    failure.write_text(json.dumps(dict(config=a.config,base=a.base,report=report,acceptance='robot_discontinuous_no_reference_written'),indent=2)+'\n')
    raise RuntimeError('Robot joint discontinuity exceeds 30deg; human solve omitted; see '+str(failure))

data=ROOT/cfg['data_dir'];path=data/f'retarget/ref_qpos_{side}.npz'
if not path.exists():path=data/f'ref_qpos_{side}.npz'
h=np.load(path,allow_pickle=True);assert float(h['fps'])==float(z['source_fps'])
w=interp_T(np.array([pose_T(p,q) for p,q in zip(h['wrist_pos'],h['wrist_quat_wxyz'])]),z['source_frame'].astype(float))
yaw=np.radians(float(z['scene_yaw_deg']));rw=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
P=np.array([target[0,:3,3]+rw@(t[:3,3]-w[0,:3,3]) for t in w])
Q=np.array([R_to_q(rw@(t[:3,:3]@w[0,:3,:3].T)@rw.T@target[0,:3,:3]) for t in w])
solve_store('human_'+side+'_q',P,Q,20260840+int(side=='left'))
z[side+'_f']=np.repeat(g[None,7:29],len(tools),axis=0).astype(np.float32)
r=ast.literal_eval(str(z['ik_report']));r[side]=report[side+'_q'];z['ik_report']=np.array(str(r))
h=ast.literal_eval(str(z['human_ik_report']));h[side]=report['human_'+side+'_q'];z['human_ik_report']=np.array(str(h))
z['changed_side_report']=np.array(json.dumps(report));z['acceptance']=np.array('numeric_candidate_pending_review')
out=ROOT/cfg['reference'];np.savez(str(out)+'.candidate.npz',**z)
for key,r in report.items():
 assert r['ok_ratio']>=.99 and r['pos_max_cm']<(.5 if not key.startswith('human') else 1),report
 assert r['joint_step_max_deg']<=(12 if key.startswith('human') else 8),report
np.savez(out,**z);print('wrote',out,flush=True)
