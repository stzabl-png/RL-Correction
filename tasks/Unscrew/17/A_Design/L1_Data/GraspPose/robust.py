import json, numpy as np
exec(open('screen_screw17.py').read().split('# ================= 左手 =================')[0])
print("右臂关节及限位(°):", [(n, round(np.degrees(l)), round(np.degrees(u))) for n,l,u in zip(ikR.arm_joints, ikR.lower, ikR.upper)])
print("右臂默认站姿(°):", np.round(np.degrees(ikR.q_default)))
rng=np.random.default_rng(0)
def best(ik, M, q0=None):
    seeds=[ik.q_default]+([q0] if q0 is not None else [])+[np.clip(ik.q_default+rng.uniform(-0.6,0.6,ik.n),ik.lower,ik.upper) for _ in range(6)]
    return ik.solve_best(M[:3,3], M[:3,:3], seeds, iters=150, pos_tol=0.008, rot_tol=0.12)
cL=[c for c in C['bottle_left']['cands'] if c['file']=='1_Large_Diameter__v4_7_14_grasp.npy'][0]
FIN=['33_Inferior_Pincer__1_47','33_Inferior_Pincer__4_34','33_Inferior_Pincer__6_46','33_Inferior_Pincer__7_48','33_Inferior_Pincer__2_44','33_Inferior_Pincer__4_42','33_Inferior_Pincer__8_49','33_Inferior_Pincer__6_44','33_Inferior_Pincer__0_49','8_Prismatic_2_Finger__9_5']
psis=list(range(272,353,10))   # 312±40
print(f"\n右手鲁棒性: psi40 扫 {psis} (中心 312 = 左 yaw280 + 人手滚转 32); 每格: 抓 IK 位置误差cm / 限位余量° (哪个关节顶限)")
for f in FIN:
    cR=[c for c in C['cap_right']['cands'] if c['file']==f+'_grasp.npy'][0]; Tcw=cand_T(cR,comC); Tpw,_=cand_pre_T(cR,comC)
    cells=[]; q=None; nok=0
    for psi in psis:
        Tcap=T_env_bottle(40,np.radians(psi))@T_bottle_cap; Mg=Tcap@Tcw; Mp=Tcap@Tpw
        r=best(ikR,Mg,q); rp=best(ikR,Mp,r['q']); q=r['q'] if r['ok'] else q
        m=np.minimum(r['q']-ikR.lower, ikR.upper-r['q']); j=int(np.argmin(m))
        ok=r['ok'] and rp['ok'] and Mg[2,3]>TABLE+0.03; nok+=int(ok)
        cells.append(f"{'✓' if ok else '✗'}{r['pos_err']*100:4.1f}/{np.degrees(m[j]):3.0f}°{ikR.arm_joints[j][-2:]}")
    appr=(T_env_bottle(40,np.radians(312))@T_bottle_cap@Tcw)[:3,3]-(T_env_bottle(40,np.radians(312))@T_bottle_cap)[:3,3]
    print(f"{f:28s} demo{cR['demo_angle']:4.1f}° obst{cR['obst_touch']} | 可用 {nok}/{len(psis)} | 312 处 盖→腕 {np.round(appr,2)} | " + " ".join(cells))
# 左手: 立瓶 GraspPose + PreGrasp 梯顶 + 放倒后 (Δroll 0/32) 多种子复核
Tcw=cand_T(cL,comB); Tpw,dpre=cand_pre_T(cL,comB)
for yaw in (270,280,290):
    M0=T_env_bottle(0,np.radians(yaw))@Tcw; Mp=T_env_bottle(0,np.radians(yaw))@Tpw
    r0=best(ikL,M0); rp=best(ikL,Mp,r0['q'])
    m=np.minimum(r0['q']-ikL.lower, ikL.upper-r0['q'])
    line=f"左 v4_7_14 yaw{yaw}: 立瓶抓 {r0['pos_err']*100:.2f}cm 余量 {np.degrees(m.min()):.0f}°({ikL.arm_joints[int(np.argmin(m))][-2:]}) PreGrasp(退 {dpre*100:.1f}cm) {rp['pos_err']*100:.2f}cm | 放倒后:"
    for dr in (0,32):
        M40=T_env_bottle(40,np.radians(yaw+dr))@Tcw; r=best(ikL,M40,r0['q']); m=np.minimum(r['q']-ikL.lower, ikL.upper-r['q'])
        line+=f" Δroll{dr:+d} {r['pos_err']*100:.2f}cm 余量 {np.degrees(m.min()):.0f}°({ikL.arm_joints[int(np.argmin(m))][-2:]}) 腕高 {100*(M40[2,3]-TABLE):.1f}cm;"
    print(line)
