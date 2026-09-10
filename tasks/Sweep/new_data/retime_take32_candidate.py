"""Densify only large joint edges of the NEW failed v2 candidate; audit full FK."""
from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
from rl_rebuild.correction.kinematics import ArmIK
ROOT=Path(__file__).resolve().parents[3]
src=ROOT/'tasks/Sweep/new_data/references/take_32_reference_v2.npz.candidate.npz'
z=dict(np.load(src,allow_pickle=True));n=len(z['source_frame'])
q=np.concatenate([z['right_q'],z['left_q']],1)
d=np.degrees(abs(np.diff(q,axis=0))).max(1)
times=[]
for i in range(n-1):
    pieces=max(1,int(np.ceil(d[i]/6.0)))
    times.extend(i+np.arange(pieces)/pieces)
times=np.r_[times,n-1]
out={}
rowkeys=['right_q','left_q','human_right_q','human_left_q','right_f','left_f','source_frame','obj_pos_0','obj_quat_0','confidence_0','obj_pos_1','obj_quat_1','confidence_1']
for k,v in z.items():
    if k not in rowkeys:out[k]=v;continue
    if 'quat' in k:
        out[k]=Slerp(np.arange(n),R.from_quat(v,scalar_first=True))(times).as_quat(scalar_first=True).astype(v.dtype)
    else:
        flat=v.reshape(n,-1)
        out[k]=np.stack([np.interp(times,np.arange(n),flat[:,j]) for j in range(flat.shape[1])],1).reshape((len(times),)+v.shape[1:]).astype(v.dtype)
old=np.searchsorted(times,np.arange(n))
for k in rowkeys:
    if 'quat' in k:
        assert np.max((R.from_quat(out[k][old],scalar_first=True).inv()*R.from_quat(z[k],scalar_first=True)).magnitude())<1e-6
        out[k][old]=z[k]  # quaternion sign is gauge; preserve original bytes
    else:assert np.allclose(out[k][old],z[k],atol=1e-7),k
anchor=np.array(json.loads((ROOT/'tasks/Sweep/2/A_Design/L2_Reference/anchor_T_sweep2.json').read_text())['anchor_T'])
report={}
for side,oi,role in [('right',1,'broom'),('left',0,'dustpan')]:
    ik=ArmIK(side,anchor_link='arm_center',anchor_T=anchor)
    g=np.load(ROOT/f'tasks/Sweep/new_data/priors/take_32_{role}_v2.npz')['grasp']
    ro=R.from_quat(out[f'obj_quat_{oi}'],scalar_first=True)
    targets=out[f'obj_pos_{oi}']+ro.apply(g[:3])
    rotations=ro*R.from_quat(g[3:7],scalar_first=True)
    pe=[];re=[]
    for j,arm in enumerate(out[side+'_q']):
        p,r=ik.fk(arm);pe.append(np.linalg.norm(p-targets[j]))
        re.append((R.from_matrix(r).inv()*rotations[j]).magnitude())
    step=np.degrees(abs(np.diff(out[side+'_q'],axis=0))).max()
    report[side]=dict(pos_max_mm=float(1000*max(pe)),rot_max_deg=float(np.degrees(max(re))),joint_step_max_deg=float(step))
    assert max(pe)<.005 and max(re)<np.radians(2) and step<=8,report
out['contact_row']=np.int32(np.searchsorted(times,int(z['contact_row'])))
out['ik_report']=np.array(str(report))
out['meta']=np.array(str(z['meta'])+'; new candidate joint-edge shared densification, all original samples retained')
out['retime_parent_row']=times
out['acceptance']=np.array('numeric_pass_pending_physical_visual_review')
target=ROOT/'tasks/Sweep/new_data/references/take_32_reference_v3.npz'
np.savez(target,**out)
rep=dict(input=str(src.relative_to(ROOT)),output=str(target.relative_to(ROOT)),old_rows=n,new_rows=len(times),added_rows=len(times)-n,full_original_samples_retained=True,report=report)
(ROOT/'logs/task3_take32_v2_20260905/retime_audit.json').write_text(json.dumps(rep,indent=2)+'\n')
print(json.dumps(rep,indent=2))
cfg=json.loads((ROOT/'tasks/Sweep/new_data/configs/take_32_v2.json').read_text())
cfg['reference']=str(target.relative_to(ROOT))
for k in ['dustpan_mesh','dustpan_usd']:cfg['assets'][k]=cfg['assets'][k].replace('dustpan_v2','dustpan_v3')
(ROOT/'tasks/Sweep/new_data/configs/take_32_v3.json').write_text(json.dumps(cfg,indent=2)+'\n')
