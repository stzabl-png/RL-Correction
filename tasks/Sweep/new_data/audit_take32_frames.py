"""Audit the delivered canonical, input mesh and converted USD coordinate chain."""
from pathlib import Path
import json,sys,glob,hashlib
import numpy as np,trimesh
from scipy.spatial import cKDTree
from tasks.Sweep.new_data.prior_frame import load_candidate,convert,matrix
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,glob.glob('/home/msc-auto/rlcorr-venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*')[0])
from pxr import Usd,UsdGeom
report={}
for role,oi,filename in [('dustpan',0,'fingertip_mid__0_1_grasp.npy'),('broom',1,'fingertip_middle__3_23_grasp.npy')]:
    delivery=ROOT/('datasets/sweep_p4_grasppose_16takes_20260905/p4s32_'+role+('_left' if oi==0 else '_right'))

    canonical=json.loads((delivery/'region_rank.json').read_text())['canonical_frame']
    candidate=load_candidate(delivery/'grasp_data'/filename)
    fixed=convert(candidate,canonical)
    meshpath=ROOT/f'datasets/sweep_new_data/sweep_dustpan/32/objects/object_{oi}/object_mesh_scaled_final.obj'
    mesh=trimesh.load(meshpath,force='mesh',process=False)
    nearest=cKDTree(mesh.vertices)
    old=np.load(ROOT/f'tasks/pregrasp/priors/SweepP4_32_{role}.npz')
    st=Usd.Stage.Open(str(ROOT/f'datasets/sweep_new_data/sweep_dustpan/32/cache/task3_{role}.usd'))
    xf=UsdGeom.XformCache()
    rigid=st.GetDefaultPrim()
    usd_meshes=[p for p in st.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(usd_meshes)==1
    usd_mesh=usd_meshes[0]
    usdpoints=np.asarray(UsdGeom.Mesh(usd_mesh).GetPointsAttr().Get())
    transform=np.array(xf.GetLocalToWorldTransform(usd_mesh)*xf.GetLocalToWorldTransform(rigid).GetInverse())
    assert np.allclose(transform,np.eye(4),atol=1e-7)
    assert usdpoints.shape==mesh.vertices.shape
    source=np.load(ROOT/f'datasets/sweep_new_data/sweep_dustpan/32/poseqa/rts_sweep_dustpan_32_object_{oi}.npz')['object_ob_in_world_smooth'][0]
    Rci=matrix(canonical['canonical_from_input_rot_wxyz'])
    p_rt=Rci@(fixed['grasp'][:3]-np.asarray(canonical['com_offset']))
    angle_error=np.linalg.norm(Rci@matrix(fixed['grasp'][3:7])-matrix(candidate['grasp_qpos'][0,3:7]))
    report[role]=dict(candidate=filename,canonical=canonical,
       old_contact_vertex_distance_mm=np.percentile(nearest.query(old['contact_pos'])[0],[0,50,95,100]).tolist(),
       corrected_contact_vertex_distance_mm=np.percentile(nearest.query(fixed['contact_pos'])[0],[0,50,95,100]).tolist(),
       wrist_roundtrip_error_m=float(np.linalg.norm(p_rt-candidate['grasp_qpos'][0,:3])),
       rotation_roundtrip_matrix_error=float(angle_error),
       usd_mesh_to_root=transform.tolist(),obj_usd_point_error_m=float(np.max(np.abs(usdpoints-mesh.vertices))),
       source_root_to_world=source.tolist(),
       canonical_up_in_world=(source[:3,:3]@Rci.T[:,2]).tolist(),
       input_mesh_sha256=hashlib.sha256(meshpath.read_bytes()).hexdigest())
    for key in ['old_contact_vertex_distance_mm','corrected_contact_vertex_distance_mm']:
        report[role][key]=(np.array(report[role][key])*1000).tolist()
out=ROOT/'logs/task3_take32_v2_20260905/coordinate_audit.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
