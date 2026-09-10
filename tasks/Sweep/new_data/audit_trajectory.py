"""Post-record full trajectory and attachment audit; does not accept visually."""
from pathlib import Path
import json,numpy as np,trimesh
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[3]
out=ROOT/'outputs_video/Task3_take32_v3/full_zero_retry1'
t=np.load(out/'trajectory_trace.npz');ref=np.load(ROOT/'tasks/Sweep/new_data/references/take_32_reference_v3.npz')
cfg=json.loads((ROOT/'tasks/Sweep/new_data/configs/take_32_v3.json').read_text())
n=len(t['row']);assert n==len(ref['source_frame']) and np.array_equal(t['row'],np.arange(n))
assert np.max(abs(t['cum_res']))==0
report=dict(rows=n,source_first=float(ref['source_frame'][0]),source_last=float(ref['source_frame'][-1]),all_rows_played=True,zero_residual=True,roles={})
for side,role,key,oi in [('right','broom','broom',1),('left','dustpan','pan',0)]:
    prior=np.load(ROOT/cfg['grasppose'][role]['prior'])['grasp']
    pose=t[key+'_pose'];rot=R.from_quat(pose[:,3:7],scalar_first=True)
    hand=t['hand_'+side+'_pose']
    pred=pose[:,:3]+rot.apply(prior[:3])
    jp=np.linalg.norm(hand[:,:3]-pred,axis=1)
    jr=((rot*R.from_quat(prior[3:7],scalar_first=True)).inv()*R.from_quat(hand[:,3:7],scalar_first=True)).magnitude()
    dp=np.linalg.norm(pose[:,:3]-ref[f'obj_pos_{oi}'],axis=1)
    dr=(R.from_quat(ref[f'obj_quat_{oi}'],scalar_first=True).inv()*rot).magnitude()
    verts=np.asarray(trimesh.load(ROOT/cfg['assets'][role+'_mesh'],force='mesh',process=False).vertices)
    clearance=np.array([np.min(verts@mat[2]+p[2])-.87 for mat,p in zip(rot.as_matrix(),pose)])
    report['roles'][role]=dict(joint_pos_max_mm=float(jp.max()*1000),joint_rot_max_deg=float(np.degrees(jr.max())),
      tracking_pos_median_mm=float(np.median(dp)*1000),tracking_pos_max_mm=float(dp.max()*1000),
      tracking_rot_max_deg=float(np.degrees(dr.max())),mesh_table_min_mm=float(clearance.min()*1000),
      worst_clearance_row=int(np.argmin(clearance)))
report['arm_target_error_max_deg']=float(np.degrees(abs(t['q']-t['q_target'])).max())
report['acceptance']='pending_visual_review'
(out/'numeric_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
