"""Read-only source review sheets and candidate contact locations for one take."""
import argparse,json
from pathlib import Path
import numpy as np,cv2,trimesh
from tasks.Sweep.new_data.prior_frame import load_candidate,convert
p=argparse.ArgumentParser();p.add_argument('--take',type=int,required=True);a=p.parse_args();take=a.take
data=Path(f'datasets/sweep_new_data/sweep_dustpan/{take}')
out=Path(f'outputs_video/Task3_take{take}_v2');out.mkdir(parents=True,exist_ok=True)
for stem in ['conf_sweep_dustpan_'+str(take),'recon_render_sweep_dustpan_'+str(take),'textured_viz/turntable_object_0']:
 cap=cv2.VideoCapture(str(data/(stem+'.mp4')));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);imgs=[]
 if count==0:
  print('optional media absent',stem,flush=True);cap.release();continue
 for frame in np.linspace(0,count-1,6).astype(int):
  cap.set(1,int(frame));ok,img=cap.read();assert ok
  if stem.startswith('conf'):img=img[:img.shape[0]*3//5]
  img=cv2.resize(img,(480,300));cv2.putText(img,f'frame {frame}',(10,25),0,.65,(0,0,255),2);imgs.append(img)
 cap.release();cv2.imwrite(str(out/(Path(stem).name+'_sheet.jpg')),np.concatenate([np.concatenate(imgs[:3],1),np.concatenate(imgs[3:],1)],0))
for i,role in [(0,'dustpan'),(1,'broom')]:
 v=np.asarray(trimesh.load(data/f'objects/object_{i}/object_mesh_scaled_final.obj',process=False).vertices)
 r=np.load(data/f'poseqa/rts_sweep_dustpan_{take}_object_{i}.npz')['object_ob_in_world_smooth'][0]
 d=Path(f'datasets/sweep_p4_grasppose_16takes_20260905/p4s{take}_{role}_'+('left' if i==0 else 'right'))
 rank=json.loads((d/'region_rank.json').read_text())
 print(role,'bounds',v.min(0),v.max(0),'worldupInput',r[2,:3],flush=True)
 for row in rank['rows'][:8]:
  p=d/'all_candidates'/row['file']
  if not p.exists():p=d/'grasp_data'/row['file']
  prior=convert(load_candidate(p),rank['canonical_frame'])
  print(row['rank'],row['file'],'angle',row['demo_angle_deg'],'contact bounds',prior['contact_pos'].min(0),prior['contact_pos'].max(0),flush=True)
