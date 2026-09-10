"""CPU orthographic mesh/URDF grasp inspection, no physics or training."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,trimesh,cv2
from rl_rebuild.correction.kinematics import Urdf,_rpy
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER
from tasks.Sweep.new_data.prior_frame import matrix
ROOT=Path(__file__).resolve().parents[3]
def geometry(prior_path,mesh_path,side):
    prior=np.load(prior_path);q=prior['grasp']
    tool=trimesh.load(mesh_path,force='mesh',process=False)
    pieces=[(np.asarray(tool.vertices),np.asarray(tool.faces),np.array([160,160,165]))]
    u=Urdf();xml=ET.parse(u.path).getroot();base=Path(u.path).parent
    angles={n.replace('right_',side+'_'):float(v) for n,v in zip(GENERIC_JOINT_ORDER,q[7:])}
    wrist=np.eye(4);wrist[:3,:3]=matrix(q[3:7]);wrist[:3,3]=q[:3]
    handroot=side+'_hand_C_MC'
    for link in xml.findall('link'):
        name=link.get('name')
        if name!=handroot and handroot not in [u.joints[n]['child'] for n in u.chain_to(name)]:continue
        pose=wrist if name==handroot else u.link_pose(name,angles,base_T=wrist,start_link=handroot)
        for vis in link.findall('visual'):
            mesh=vis.find('geometry/mesh')
            if mesh is None:continue
            fn=mesh.get('filename')
            path=base/fn.replace('package://vega_1p_sharpa/','')
            if not path.exists():raise FileNotFoundError(path)
            m=trimesh.load(path,force='mesh',process=False)
            local=np.eye(4);org=vis.find('origin')
            if org is not None:
                local[:3,3]=np.fromstring(org.get('xyz','0 0 0'),sep=' ')
                local[:3,:3]=_rpy(*np.fromstring(org.get('rpy','0 0 0'),sep=' '))
            transform=pose@local
            v=np.asarray(m.vertices)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            v=v@transform[:3,:3].T+transform[:3,3]
            color=np.array([220,230,240]) if 'pad' not in name.lower() else np.array([80,220,80])
            pieces.append((v,np.asarray(m.faces),color))
    return pieces
def render(pieces,eye,label):
    v=np.concatenate([p[0] for p in pieces]);center=(v.min(0)+v.max(0))/2
    forward=np.asarray(eye,dtype=float);forward/=np.linalg.norm(forward)
    right=np.cross([0,0,1],forward);right/=np.linalg.norm(right);up=np.cross(forward,right)
    rot=np.stack([right,up,forward],axis=1)
    allp=(v-center)@rot;scale=650/max(np.ptp(allp[:,0]),np.ptp(allp[:,1]))
    polys=[];cols=[];depth=[]
    for pts,faces,color in pieces:
        p=(pts-center)@rot;t=p[faces]
        normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
        shade=.40+.60*np.abs(normal@np.array([.2,.3,.9327]))
        xy=t[:,:,:2]*[scale,-scale]+[400,420]
        polys.extend(xy.astype(np.int32));cols.extend((shade[:,None]*color).astype(np.uint8));depth.extend(t[:,:,2].mean(1))
    img=np.full((800,800,3),45,np.uint8)
    for i in np.argsort(depth):cv2.fillConvexPoly(img,polys[i],tuple(int(x) for x in cols[i]))
    cv2.putText(img,label,(20,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
    return img
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior',required=True);p.add_argument('--mesh',required=True);p.add_argument('--side',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    pieces=geometry(a.prior,a.mesh,a.side)
    views=[render(pieces,eye,label) for eye,label in [([1,-2,1],'input view -Y (source upper side)'),([-1,2,1],'input view +Y (opposite side)')]]
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(a.out,np.concatenate(views,axis=1))
