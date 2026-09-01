import numpy as np
exec(open('screen_screw17.py').read().split('# ================= 左手 =================')[0])
rng=np.random.default_rng(2)
def best(ik,M,w_rot=0.35):
    seeds=[ik.q_default]+[np.clip(ik.q_default+rng.uniform(-0.7,0.7,ik.n),ik.lower,ik.upper) for _ in range(10)]
    return ik.solve_best(M[:3,3],M[:3,:3],seeds,iters=150,pos_tol=0.008,rot_tol=0.12,w_rot=w_rot)
def marg(r,ik):
    m=np.minimum(r['q']-ik.lower, ik.upper-r['q']); j=int(np.argmin(m)); return np.degrees(m[j]), ik.arm_joints[j][-2:]
psi=np.radians(312)
print("右肩位置(默认站姿):", np.round(ikR.link_pose_world('vega_1p_R_arm_l1', ikR.q_default)[:3,3],3), " 左肩:", np.round(ikL.link_pose_world('vega_1p_L_arm_l1', ikL.q_default)[:3,3],3))
print("盖心 env (psi312):", np.round((T_env_bottle(40,psi)@T_bottle_cap)[:3,3],3))
for f in ['33_Inferior_Pincer__1_47','33_Inferior_Pincer__4_34']:
    cR=[c for c in C['cap_right']['cands'] if c['file']==f+'_grasp.npy'][0]; Tcw=cand_T(cR,comC)
    for dx,dy,dz in [(0,0,0),(0,-0.05,0),(0,-0.10,0),(0,-0.15,0),(0,0,0.05),(0,0,0.10),(0,-0.10,0.05),(0.05,-0.10,0.05),(-0.05,0,0),(-0.10,0,0)]:
        Tb=T_env_bottle(40,psi); Tb[:3,3]+=[dx,dy,dz]; Mg=Tb@T_bottle_cap@Tcw
        r=best(ikR,Mg); m,j=marg(r,ikR)
        print(f"{f:26s} 场景平移 dx{dx*100:+3.0f} dy{dy*100:+3.0f} dz{dz*100:+3.0f}cm | {r['pos_err']*100:4.1f}cm/{np.degrees(r['rot_err']):3.0f}° 最小余量 {m:3.0f}°({j}) q={np.round(np.degrees(r['q']))}")
