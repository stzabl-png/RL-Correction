import numpy as np
exec(open('screen_screw17.py').read().split('# ================= 左手 =================')[0])
rng=np.random.default_rng(3)
def allsol(ik,M,n=40,w_rot=0.35):
    sols=[]
    for i in range(n):
        q0=ik.q_default if i==0 else rng.uniform(ik.lower,ik.upper)
        r=ik.solve(M[:3,3],M[:3,:3],q0=q0,iters=200,pos_tol=0.008,rot_tol=0.12,w_rot=w_rot)
        if r['ok']:
            m=np.minimum(r['q']-ik.lower, ik.upper-r['q']); sols.append((np.degrees(m.min()), ik.arm_joints[int(np.argmin(m))][-2:], np.round(np.degrees(r['q']))))
    return sorted(sols,key=lambda s:-s[0])
FIN=['33_Inferior_Pincer__1_47','33_Inferior_Pincer__4_34','33_Inferior_Pincer__6_46','33_Inferior_Pincer__2_44','33_Inferior_Pincer__4_42','33_Inferior_Pincer__8_49','33_Inferior_Pincer__6_44','33_Inferior_Pincer__0_49','8_Prismatic_2_Finger__9_5','33_Inferior_Pincer__7_48']
print("右手: 40 随机种子全约束 IK, 报收敛解数 / 最大限位余量解 (余量°, 顶限关节, q°). psi40 = 312 (人手滚转) 与 282 (不滚转)")
for f in FIN:
    cR=[c for c in C['cap_right']['cands'] if c['file']==f+'_grasp.npy'][0]; Tcw=cand_T(cR,comC)
    line=f"{f:26s}"
    for psi in (312,282):
        Mg=T_env_bottle(40,np.radians(psi))@T_bottle_cap@Tcw; s=allsol(ikR,Mg)
        line+=f" | psi{psi}: {len(s):2d} 解, 最优余量 {s[0][0]:4.0f}°({s[0][1]}) q={s[0][2]}" if s else f" | psi{psi}: 0 解"
    print(line)
# 位置-only 上限: 盖心附近的腕位置本身能有多大余量
cR=[c for c in C['cap_right']['cands'] if c['file']=='33_Inferior_Pincer__1_47_grasp.npy'][0]
Mg=T_env_bottle(40,np.radians(312))@T_bottle_cap@cand_T(cR,comC); s=allsol(ikR,Mg,n=60,w_rot=0.0)
print(f"\n只管位置 (1_47 腕位 {np.round(Mg[:3,3],3)}): {len(s)} 解, 最大余量 {s[0][0]:.0f}°({s[0][1]}) —— 位置本身可达的舒适上限")
cL=[c for c in C['bottle_left']['cands'] if c['file']=='1_Large_Diameter__v4_7_14_grasp.npy'][0]
for tag,M in [('立瓶 yaw280',T_env_bottle(0,np.radians(280))@cand_T(cL,comB)),('放倒 psi312',T_env_bottle(40,np.radians(312))@cand_T(cL,comB))]:
    s=allsol(ikL,M); print(f"左 v4_7_14 {tag}: {len(s)} 解, 最优余量 {s[0][0]:.0f}°({s[0][1]}) q={s[0][2]}")
