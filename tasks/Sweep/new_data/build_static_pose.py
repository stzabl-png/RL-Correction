"""Build a one-pose diagnostic from source row zero, never from rejected references."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from rl_rebuild.correction.kinematics import ArmIK,quat_to_R,R_to_quat
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER
c=json.loads((ROOT/'tasks/Sweep/new_data/configs/take_32_v2.json').read_text())
data=ROOT/c['data_dir']
raw={i:np.load(data/f'poseqa/rts_sweep_dustpan_32_object_{i}.npz')['object_ob_in_world_smooth'][0] for i in (0,1)}
import trimesh
v=trimesh.load(data/'objects/object_0/object_mesh_scaled_final.obj',force='mesh',process=False).vertices
table=(v@raw[0][:3,:3].T+raw[0][:3,3])[:,2].min()
yaw=np.radians(-14);Rw=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
origin=raw[1][:3,3];target=np.array([-.116,-.177,origin[2]+.87-table])
tools={}
for i,T in raw.items():
    tools[i]=np.eye(4);tools[i][:3,:3]=Rw@T[:3,:3];tools[i][:3,3]=target+Rw@(T[:3,3]-origin)
anchor=np.array(json.loads((ROOT/'tasks/Sweep/2/A_Design/L2_Reference/anchor_T_sweep2.json').read_text())['anchor_T'])
out=dict(control_hz=np.float32(20),contact_row=np.int64(0),cube_start_w=np.array([0,0,.883]),
    brush_contact_local=np.array([0,0,.06]),pan_semantic_R_input=np.diag([-1,-1,1]),
    pan_semantic_quat_offset=np.array([0,0,0,1]),cube_status=np.array('provisional_disabled'),
    acceptance=np.array('static_diagnostic_not_a_trajectory'))
for role,side,i in [('broom','right',1),('dustpan','left',0)]:
    prior=np.load(ROOT/c['grasppose'][role]['prior'])
    h=np.eye(4);h[:3,:3]=quat_to_R(prior['grasp'][3:7]);h[:3,3]=prior['grasp'][:3]
    t=tools[i]@h;ik=ArmIK(side,anchor_link='arm_center',anchor_T=anchor)
    rng=np.random.default_rng(23)
    trials=[ik.solve(t[:3,3],t[:3,:3],q0=q,iters=500,pos_tol=.002,rot_tol=.02)
       for q in [ik.q_default]+[rng.uniform(ik.lower,ik.upper) for _ in range(32)]]
    good=[r for r in trials if r['ok']];assert good
    q=min(good,key=lambda r:np.linalg.norm(r['q']-ik.q_default))['q']
    out[side+'_q']=np.array([q],np.float32);out['human_'+side+'_q']=out[side+'_q'].copy()
    out['obj_pos_'+str(i)]=np.array([tools[i][:3,3]],np.float32)
    out['obj_quat_'+str(i)]=np.array([R_to_quat(tools[i][:3,:3])],np.float32)
    out['confidence_'+str(i)]=np.array([80],np.float32)
path=ROOT/'tasks/Sweep/new_data/references/take_32_static_v2.npz';np.savez(path,**out)
c['reference']=str(path.relative_to(ROOT));(ROOT/'tasks/Sweep/new_data/configs/take_32_static_v2.json').write_text(json.dumps(c,indent=2))
print(path)
