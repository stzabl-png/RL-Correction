import numpy as np
exec(open('screen_screw17.py').read().split('# ================= 左手 =================')[0])
rng=np.random.default_rng(1)
def best(ik,M,w_rot=0.35):
    seeds=[ik.q_default]+[np.clip(ik.q_default+rng.uniform(-0.7,0.7,ik.n),ik.lower,ik.upper) for _ in range(8)]
    return ik.solve_best(M[:3,3],M[:3,:3],seeds,iters=150,pos_tol=0.008,rot_tol=0.12,w_rot=w_rot)
def marg(r,ik):
    m=np.minimum(r['q']-ik.lower, ik.upper-r['q']); j=int(np.argmin(m)); return np.degrees(m[j]), ik.arm_joints[j][-2:], np.round(np.degrees(r['q']))
psi=np.radians(312)
print("右臂 j2 = 肩第2关节 (范围 -89..+26°). 先看: 是位置还是姿态把 j2 顶到限位?")
for f in ['33_Inferior_Pincer__1_47','33_Inferior_Pincer__4_34','33_Inferior_Pincer__2_44']:
    cR=[c for c in C['cap_right']['cands'] if c['file']==f+'_grasp.npy'][0]; Tcw=cand_T(cR,comC)
    for dx,dy in [(0,0),(0.05,0),(0.10,0),(0,0.05),(0,0.10),(0.05,0.05)]:
        Tb=T_env_bottle(40,psi); Tb[:3,3]+= [dx,dy,0]; Mg=Tb@T_bottle_cap@Tcw
        r=best(ikR,Mg); m,j,q=marg(r,ikR); r2=best(ikR,Mg,w_rot=0.03); m2,j2,_=marg(r2,ikR)
        print(f"{f:28s} 场景平移 dx{dx*100:+3.0f} dy{dy*100:+3.0f}cm | 全约束: {r['pos_err']*100:4.1f}cm/{np.degrees(r['rot_err']):3.0f}° 余量 {m:3.0f}°({j}) q={q} | 只管位置: {r2['pos_err']*100:4.1f}cm 余量 {m2:3.0f}°({j2})")
