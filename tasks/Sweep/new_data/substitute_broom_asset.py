"""Task3 donor mesh + validated grasp, rigidly registered in recipient input frame."""
import argparse,json,sys,glob
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tasks.Sweep.new_data.prior_frame import matrix,convert,load_candidate
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--donor',required=True);p.add_argument('--out_config',required=True)
p.add_argument('--rotation',type=float,nargs=9,default=[-1,0,0,0,-1,0,0,0,1])
a=p.parse_args();root=Path(__file__).resolve().parents[3]
c=json.loads((root/a.config).read_text());d=json.loads((root/a.donor).read_text())
take=c['task_name'].removeprefix('SweepP4Take');version=Path(a.out_config).stem.rsplit('_',1)[1]
assert take in ('80','128','180') and d['task_name']=='SweepP4Take32'
target=np.load(root/c['grasppose']['broom']['prior']);donor=np.load(root/d['grasppose']['broom']['prior'])
spec=d['grasppose']['broom'];delivery=root/spec['delivery_dir']
candidate_path=delivery/'all_candidates'/spec['candidate']
if not candidate_path.exists():candidate_path=delivery/'grasp_data'/spec['candidate']
canonical=json.loads((delivery/'region_rank.json').read_text())['canonical_frame']
expected=convert(load_candidate(candidate_path),canonical)
for key in ('grasp','squeeze','pregrasp','contact_pos','contact_normal'):
 assert np.allclose(expected[key],donor[key],atol=1e-12),key
# Registration is measured in input mesh coordinates, never applied to source motion.
R=np.array(a.rotation).reshape(3,3);assert np.allclose(R.T@R,np.eye(3)) and np.isclose(np.linalg.det(R),1)
t=target['contact_centroid']-R@donor['contact_centroid']
out=root/f'tasks/Sweep/new_data/assets/take_{take}/broom_donor32_{version}';out.mkdir(parents=True,exist_ok=True)
mesh=trimesh.load(root/d['assets']['broom_mesh'],force='mesh',process=False)
original=np.asarray(mesh.vertices).copy();mesh.vertices=original@R.T+t
mesh.export(out/'object_mesh_scaled_final.obj')
prior={k:donor[k].copy() for k in donor.files}
for key in ('grasp','squeeze','pregrasp'):
 q=np.atleast_2d(prior[key]).copy();q[:,:3]=q[:,:3]@R.T+t
 q[:,3:7]=Rotation.from_matrix(R@Rotation.from_quat(q[:,[4,5,6,3]]).as_matrix()).as_quat()[:,[3,0,1,2]]
 prior[key]=q[0] if donor[key].ndim==1 else q
prior['contact_pos']=donor['contact_pos']@R.T+t
prior['contact_normal']=donor['contact_normal']@R.T
prior['contact_centroid']=prior['contact_pos'].mean(0)
prior['donor_input_to_recipient_R']=R;prior['donor_input_to_recipient_t']=t
prior['frame_schema']=np.array('donor canonical -> donor input -> recipient input (rigid mesh registration)')
priorpath=root/f'tasks/Sweep/new_data/priors/take_{take}_broom_donor32_{version}.npz';np.savez(priorpath,**prior)
sys.path.insert(0,glob.glob('/home/msc-auto/rlcorr-venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*')[0])
from pxr import Usd,UsdGeom,Vt
stage=Usd.Stage.Open(str(root/d['assets']['broom_usd']));stage=Usd.Stage.Open(stage.Flatten());rigid=stage.GetDefaultPrim();xf=UsdGeom.XformCache()
usd_errors=[]
for prim in stage.Traverse():
 if prim.IsA(UsdGeom.Mesh):
  m=UsdGeom.Mesh(prim);points=np.asarray(m.GetPointsAttr().Get())
  transform=np.asarray(xf.GetLocalToWorldTransform(prim)*xf.GetLocalToWorldTransform(rigid).GetInverse())
  assert np.allclose(transform,np.eye(4),atol=1e-7),str(prim.GetPath())
  error=cKDTree(original).query(points)[0];assert error.max()<1e-6
  new=points@R.T+t;m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(new.astype(np.float32)));m.GetNormalsAttr().Clear()
  m.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([new.min(0),new.max(0)]).astype(np.float32)))
  usd_errors.append(float(cKDTree(mesh.vertices).query(new)[0].max()))
stage.GetRootLayer().Export(str(out/'broom.usd'))
c['assets']['broom_mesh']=str((out/'object_mesh_scaled_final.obj').relative_to(root));c['assets']['broom_usd']=str((out/'broom.usd').relative_to(root))
c['grasppose']['broom']=dict(d['grasppose']['broom']);c['grasppose']['broom']['prior']=str(priorpath.relative_to(root))
c['reference']=f'tasks/Sweep/new_data/references/take_{take}_reference_{version}.npz'
c.pop('broom_head_repair',None)
c['acceptance']='donor_asset_diagnostic_pending'
c['asset_substitution']={'donor_config':a.donor,'recipient_before':a.config,'role':'broom','R':R.tolist(),'t_m':t.tolist(),'scale':1,'anchor':'contact centroid','long_axis_recipient':(R@np.array([0,0,1])).tolist(),'bristles_recipient':(R@np.array([0,-1,0])).tolist(),'trajectory_changed':False}
(root/a.out_config).write_text(json.dumps(c,indent=2)+'\n')
assert np.array_equal(mesh.faces,trimesh.load(root/d['assets']['broom_mesh'],force='mesh',process=False).faces)
assert np.allclose(prior['grasp'][7:],donor['grasp'][7:],atol=0)
# Relative wrist-to-tool and finger-contact geometry must be identical after a rigid change of frame.
relative_before=(donor['contact_pos']-donor['grasp'][:3])@matrix(donor['grasp'][3:7])
relative_after=(prior['contact_pos']-prior['grasp'][:3])@matrix(prior['grasp'][3:7])
report=dict(c['asset_substitution'],canonical_to_donor_prior_verified=True,vertices=len(mesh.vertices),faces=len(mesh.faces),watertight=mesh.is_watertight,usd_vertex_max_error_m=max(usd_errors),grasp_relative_contact_error_m=float(abs(relative_before-relative_after).max()),bounds_m=mesh.bounds.tolist())
assert report['grasp_relative_contact_error_m']<1e-10
(out/'substitution_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
