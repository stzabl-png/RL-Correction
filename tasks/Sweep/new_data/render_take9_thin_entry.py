import numpy as np,trimesh,cv2
from pathlib import Path
base=Path('tasks/Sweep/new_data/assets/take9_powerdisk/dustpan_thin_entry_20260909')
a=trimesh.load('tasks/Sweep/new_data/prepared/take9_powerdisk/objects/object_0/object_mesh_scaled_final.obj',process=False)
b=trimesh.load(base/'object_mesh_scaled_final.obj',process=False)
def render(mesh,eye,label):
 v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces);center=(v.min(0)+v.max(0))/2
 fw=np.array(eye,float);fw/=np.linalg.norm(fw);right=np.cross([0,1,0],fw);right/=np.linalg.norm(right);up=np.cross(fw,right);p=(v-center)@np.stack([right,up,fw],axis=1);t=p[f];n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-15)
 shade=.35+.65*abs(n@np.array([.25,.45,.857]));scale=min(1150/np.ptp(p[:,0]),580/np.ptp(p[:,1]));xy=(t[:,:,:2]*[scale,-scale]+[650,380]).astype(np.int32)
 im=np.full((760,1300,3),245,np.uint8)
 colors=shade[:,None]*np.array([175,185,200])
 for i in np.argsort(t[:,:,2].mean(1)): cv2.fillConvexPoly(im,xy[i],tuple(int(c) for c in colors[i]))
 cv2.putText(im,label,(40,50),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,50),2)
 return im
cv2.imwrite(str(base/'comparison.png'),np.concatenate([render(a,[-1,0,0],'Original - side profile / mouth right'),render(b,[-1,0,0],'Thin entry - side profile / mouth right')],axis=0))
cv2.imwrite(str(base/'asset.png'),render(b,[-1,1.7,1.7],'Thin entry - top/front view'))
# True mesh central cross section for topology-independent geometric evidence.
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(12,4))
for mesh,color,label in [(a,'#888888','Original'),(b,'#007f9e','Repaired')]:
 sec=mesh.section(plane_origin=[0,0,0],plane_normal=[1,0,0])
 for j,line in enumerate(sec.discrete):ax.plot(line[:,2]*1000,line[:,1]*1000,color=color,lw=2,label=label if j==0 else None)
ax.set_xlim(35,110);ax.set_ylim(-18,20);ax.set_aspect('equal');ax.legend();ax.set_xlabel('Toward entrance (mm)');ax.set_ylabel('Height in asset (mm)');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(base/'section.png',dpi=180)
