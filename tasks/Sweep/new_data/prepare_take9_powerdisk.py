"""Take9 poses with user-authorized original Sweep2 tools; immutable source inputs."""
from pathlib import Path
import json,shutil
import numpy as np
import trimesh
from rl_rebuild.correction import frames as F
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER
ROOT=Path(__file__).resolve().parents[3]
src=ROOT/'datasets/sweep_new_data/sweep_dustpan_2/9'
dst=ROOT/'tasks/Sweep/new_data/prepared/take9_powerdisk'
dst.mkdir(parents=True,exist_ok=True)
for name in ['poseqa','objects/object_0','objects/object_1']:(dst/name).mkdir(parents=True,exist_ok=True)
for i in (0,1):
 shutil.copy2(src/f'poseqa/rts_sweep_dustpan_9_object_{i}.npz',dst/f'poseqa/rts_sweep_dustpan_9_object_{i}.npz')
 donor=ROOT/('tasks/Sweep/2/assets/dustpan_smooth_entry/object_mesh_scaled_final.obj' if i==0 else 'datasets/sweep_2_better/objects/object_1/object_mesh_scaled_final.obj')
 shutil.copy2(donor,dst/f'objects/object_{i}/object_mesh_scaled_final.obj')
z=dict(np.load(src/'replay_world.npz',allow_pickle=True));n=len(z['frames'])
for s in ('left','right'):z['phase_'+s]=np.ones(n,np.int8)
z['phase_obj']=np.ones(n,np.int8)
# The legacy loader uses this only as initialization scaffolding, then Sweep replaces its reference.
np.savez_compressed(dst/'replay_world.npz',**z)
for s in ('left','right'):
 joints=z['joints_'+s]
 np.savez_compressed(dst/f'ref_qpos_{s}.npz',wrist_pos=joints[:,0],wrist_quat_wxyz=F.sharpa_base_quat_from_joints(joints,s),fps=z['fps'],finger_qpos=np.zeros((n,22)),joint_names=np.array(GENERIC_JOINT_ORDER),provenance=np.array('wrist from source joints via existing sharpa_base_quat_from_joints; fingers unused by Sweep'))
shutil.copy2(src/'world_fused.npz',dst/'world_fused.npz')
raw=np.load(dst/'poseqa/rts_sweep_dustpan_9_object_0.npz')['object_ob_in_world_smooth'][0]
v=trimesh.load(dst/'objects/object_0/object_mesh_scaled_final.obj',force='mesh',process=False).vertices
table=float((v@raw[:3,:3].T+raw[:3,3])[:,2].min())
layout={'scene_table_z':table,'objects':{}}
for i,role,side in [(0,'dustpan','left'),(1,'broom','right')]:
 pose=z['obj_pose_all'][i,0];layout['objects'][f'object_{i}']={'identity':role,'anchor_hand':side,'pos':pose[:3].tolist(),'quat_wxyz':pose[3:].tolist()}
(dst/'scene_layout.json').write_text(json.dumps(layout,indent=2))
assets=ROOT/'tasks/Sweep/new_data/assets/take9_powerdisk';assets.mkdir(parents=True,exist_ok=True)
# Independent copies avoid runtime mesh conversion changing the successful baseline asset.
for role,i,source in [('broom',1,ROOT/'datasets/sweep_2_better/retarget/object_1.usd'),('dustpan',0,ROOT/'tasks/Sweep/2/assets/dustpan_smooth_entry/dustpan.usd')]:
 print(role,'USD candidate',source,source.exists())
config={'task_name':'Task3Take9PowerDisk','data_dir':str(dst.relative_to(ROOT)),'source_data_dir':str(src.relative_to(ROOT)),'rts_stem':'rts_sweep_dustpan_9','training_replay':str((dst/'replay_world.npz').relative_to(ROOT)),'scene_layout':str((dst/'scene_layout.json').relative_to(ROOT)),'reference':'tasks/Sweep/new_data/references/take9_powerdisk_v1.npz','cube_status':'training_geometry_calibrated','assets':{},'grasppose':{},'grip_mode':'fixed','scripted_prelude_steps':0,'hand_friction':4.0,'training_geometry':{'pan_half_width':.060,'pan_inside_z_min':.015,'pan_mouth_z':.095,'pan_center_y_min':.018,'pan_center_y_max':.030,'start_outside':.065,'deep_inside_margin':.020},'training_lip_y':.006,'asset_provenance':'original Sweep2 tools, user-authorized 2026-09-08; take9 absolute source poses then shared world registration; no per-tool rotations'}
for role,i,side,candidate in [('broom',1,'right','98_45'),('dustpan',0,'left','45_20')]:
 config['assets'][role+'_mesh']=str((dst/f'objects/object_{i}/object_mesh_scaled_final.obj').relative_to(ROOT))
 config['assets'][role+'_usd']=str((assets/f'{role}.usd').relative_to(ROOT))
 config['grasppose'][role]={'hand':side,'object_id':f'object_{i}','prior':f'tasks/Sweep/new_data/priors/powerdisk_20260908/{role}_10_Power_Disk__{candidate}_grasp.npz'}
(ROOT/'tasks/Sweep/new_data/configs/take9_powerdisk_fixed.json').write_text(json.dumps(config,indent=2))
print('Prepared source-faithful adapter',dst,'table',table)
