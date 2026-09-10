"""Bounded CPU audit of a common XY translation; no motion/algorithm changes."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rl_rebuild.correction.kinematics import ArmIK
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--yaw_offsets',nargs='*',type=float);p.add_argument('--source_frames',nargs='+',type=float,default=[0,60,79,80,100,119,160,220,299]);a=p.parse_args()
root=Path(__file__).resolve().parents[3];c=json.loads((root/a.config).read_text());z=np.load(root/a.base)
anchor=np.array(json.loads((root/'tasks/Sweep/2/A_Design/L2_Reference/anchor_T_sweep2.json').read_text())['anchor_T'])
source=a.source_frames;rows=sorted(set(int(np.argmin(abs(z['source_frame']-v))) for v in source))
targets={}
for side,oi,role in [('right',1,'broom'),('left',0,'dustpan')]:
 g=np.load(root/c['grasppose'][role]['prior'])['grasp'];ro=Rotation.from_quat(z[f'obj_quat_{oi}'][rows],scalar_first=True)
 targets[side]=(z[f'obj_pos_{oi}'][rows]+ro.apply(g[:3]),(ro*Rotation.from_quat(g[3:7],scalar_first=True)).as_matrix())
result=[]
placements=[(dx,dy,0) for dx,dy in [(0,0),(.08,0),(.16,0),(.08,-.06),(.16,-.06),(.08,.06),(.16,.06)]] if a.yaw_offsets is None else [(dx,dy,yaw) for yaw in a.yaw_offsets for dx,dy in [(0,0),(.08,-.06)]]
for dx,dy,yaw in placements:
 common=Rotation.from_euler('z',yaw,degrees=True);origin=z['obj_pos_1'][0]
 entry=dict(shared_delta_xy_m=[dx,dy],shared_yaw_offset_deg=yaw,poses=[])
 for side,(P,Q) in targets.items():
  ik=ArmIK(side,anchor_link='arm_center',anchor_T=anchor);rng=np.random.default_rng(20260905);previous=ik.q_default
  bank=[ik.q_default]+[rng.uniform(ik.lower,ik.upper) for _ in range(4)]
  for row,pos,rot in zip(rows,P,Q):
   pos=origin+common.apply(pos-origin);rot=common.as_matrix()@rot
   trials=[ik.solve(pos+[dx,dy,0],rot,q0=seed,iters=300,pos_tol=.002,rot_tol=.02) for seed in [previous]+bank]
   good=[r for r in trials if r['ok']]
   chosen=min(good,key=lambda r:np.linalg.norm(r['q']-previous)) if good else min(trials,key=lambda r:r['pos_err']**2+.1225*r['rot_err']**2)
   q=chosen['q'];margin=np.degrees(np.minimum(q-ik.lower,ik.upper-q)).min()
   entry['poses'].append(dict(side=side,row=row,source_frame=float(z['source_frame'][row]),ok=bool(chosen['ok']),pos_mm=float(chosen['pos_err']*1000),rot_deg=float(np.degrees(chosen['rot_err'])),joint_margin_deg=float(margin),q_deg=np.degrees(q).tolist(),distance_default=float(np.linalg.norm(q-ik.q_default))))
   previous=q
 entry['ok_count']=sum(r['ok'] for r in entry['poses'])
 entry['minimum_joint_margin_deg']=min(r['joint_margin_deg'] for r in entry['poses'])
 entry['mean_default_distance']=float(np.mean([r['distance_default'] for r in entry['poses']]))
 result.append(entry);print({k:v for k,v in entry.items() if k!='poses'},flush=True)
(root/a.out).write_text(json.dumps(dict(config=a.config,base=a.base,source_frames=source,candidates=result),indent=2)+'\n')
