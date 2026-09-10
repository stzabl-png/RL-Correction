"""Audit canonical->input OBJ->USD root; actual world closure is in replay audit."""
import argparse,json,sys,glob
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial import cKDTree
from tasks.Sweep.new_data.prior_frame import load_candidate,convert,matrix
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[3];cfg=json.loads((ROOT/a.config).read_text())
sys.path.insert(0,glob.glob('/home/msc-auto/rlcorr-venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*')[0])
from pxr import Usd,UsdGeom
report={}
for role,i in [('dustpan',0),('broom',1)]:
 spec=cfg['grasppose'][role];d=ROOT/spec['delivery_dir'];rank=json.loads((d/'region_rank.json').read_text())
 cp=d/'all_candidates'/spec['candidate']
 if not cp.exists():cp=d/'grasp_data'/spec['candidate']
 candidate=load_candidate(cp);fixed=convert(candidate,rank['canonical_frame']);prior=np.load(ROOT/spec['prior'])
 substitution=cfg.get('asset_substitution') if role=='broom' else None
 if substitution:
  rr=np.array(substitution['R']);tt=np.array(substitution['t_m'])
  inverse_position=rr.T@(prior['grasp'][:3]-tt)
  assert np.allclose(inverse_position,fixed['grasp'][:3],atol=1e-12)
  assert np.allclose(rr.T@matrix(prior['grasp'][3:7]),matrix(fixed['grasp'][3:7]),atol=1e-12)
  assert np.array_equal(prior['grasp'][7:],fixed['grasp'][7:])
  assert np.allclose((prior['contact_pos']-tt)@rr,fixed['contact_pos'],atol=1e-12)
  donor_cfg=json.loads((ROOT/substitution['donor_config']).read_text())
  source_path=ROOT/donor_cfg['assets']['broom_mesh']
  source_contact=(prior['contact_pos']-tt)@rr
 else:
  assert np.allclose(fixed['grasp'],prior['grasp'],atol=1e-12)
  inverse_position=prior['grasp'][:3];source_contact=prior['contact_pos']
  source_path=ROOT/cfg['data_dir']/f'objects/object_{i}/object_mesh_scaled_final.obj'
 source=trimesh.load(source_path,process=False)
 repaired=trimesh.load(ROOT/cfg['assets'][role+'_mesh'],process=False)
 st=Usd.Stage.Open(str(ROOT/cfg['assets'][role+'_usd']));xf=UsdGeom.XformCache();root=st.GetDefaultPrim();usd=[]
 for p in st.Traverse():
  if p.IsA(UsdGeom.Mesh):
   transform=np.array(xf.GetLocalToWorldTransform(p)*xf.GetLocalToWorldTransform(root).GetInverse())
   assert np.allclose(transform,np.eye(4),atol=1e-7)
   pts=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get());dist=cKDTree(repaired.vertices).query(pts)[0]
   assert dist.max()<1e-6
   usd.append(dict(mesh=str(p.GetPath()),mesh_to_root=transform.tolist(),point_error_m=float(dist.max())))
 rci=matrix(rank['canonical_frame']['canonical_from_input_rot_wxyz']);offset=np.array(rank['canonical_frame']['com_offset'])
 roundtrip=np.linalg.norm(rci@(inverse_position-offset)-candidate['grasp_qpos'][0,:3]);assert roundtrip<1e-10
 report[role]=dict(candidate=spec['candidate'],canonical=rank['canonical_frame'],roundtrip_m=float(roundtrip),
 source_mesh=str(source_path),asset_substitution=substitution,
 contact_original_mm=(cKDTree(source.vertices).query(source_contact)[0]*1000).tolist(),
 contact_repaired_mm=(cKDTree(repaired.vertices).query(prior['contact_pos'])[0]*1000).tolist(),usd=usd)
out=ROOT/'tasks/Sweep/new_data/prepared'/(Path(a.config).stem+'_frames.json');out.write_text(json.dumps(report,indent=2)+'\n')
print(out)
