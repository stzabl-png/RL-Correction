import json, numpy as np, sys
exec(open('screen_screw17.py').read().split('# ================= 左手 =================')[0])   # 复用场景/IK/候选定义
R = json.load(open('screen_result.json')); LEFT={l['file']:l for l in R['left']}; RIGHT=R['right']
def show_left(name, yaws):
    L=LEFT[name]; print(f"\n### 左 {name} demo{L['demo']}° | 立瓶可达带 {L['band']}")
    for y in yaws:
        ok=[d for (yy,d) in L['lay_ok'] if yy==y]; print(f"   yaw {y}: 腕方位 {L['az_at_band'].get(str(y), L['az_at_band'].get(y))}° | 放倒全程可达的 Δroll: {ok}")
show_left('1_Large_Diameter__v4_7_14_grasp.npy',[270,280,290]); show_left('1_Large_Diameter__v4_4_14_grasp.npy',[270,280,290]); show_left('11_Power_Sphere__v2_41_2_grasp.npy',[240,250,260])
def right_at(psi):
    b=int(round(psi/10)*10)%360
    return sorted([(RIGHT[f]['demo'],f,len(RIGHT[f]['ok_psi'])) for f in RIGHT if b in RIGHT[f]['ok_psi']])
for name,y in [('1_Large_Diameter__v4_7_14_grasp.npy',280),('11_Power_Sphere__v2_41_2_grasp.npy',250)]:
    for dr in (0,32):
        psi=(y+dr)%360; rr=right_at(psi)
        print(f"\n左 {name.split('__')[0]} yaw{y} Δroll{dr:+d} → psi40 {psi}: 可用右候选 {len(rr)} 个, 前 8 (demo°, 名, 带宽/36): {[(d,f.replace('_grasp.npy',''),n) for d,f,n in rr[:8]]}")
# ---- 决赛几何: 左 v4_7_14 yaw280 Δroll32 → psi 312 (bin 310); 右候选前 6 ----
def geom(lname, yaw, droll, rname):
    cL=[c for c in C['bottle_left']['cands'] if c['file']==lname][0]; cR=[c for c in C['cap_right']['cands'] if c['file']==rname][0]
    psi=np.radians(yaw+droll)
    Tb40=T_env_bottle(40,psi); Tcap=Tb40@T_bottle_cap
    Mg=Tcap@cand_T(cR,comC); Mp=Tcap@cand_pre_T(cR,comC)[0]
    rg=reach(ikR,Mg); rp=reach(ikR,Mp,q0=rg['q'])
    ML0=T_env_bottle(0,np.radians(yaw))@cand_T(cL,comB); ML40=Tb40@cand_T(cL,comB)
    rl0=reach(ikL,ML0); rl40=reach(ikL,ML40,q0=rl0['q'])
    capc=Tcap[:3,3]; appr=Mg[:3,3]-capc; pre_dir=Mp[:3,3]-Mg[:3,3]
    marg=lambda r,ik: np.round(np.minimum(r['q']-ik.lower, ik.upper-r['q']).min()*57.3,0)
    print(f"  R {rname.replace('_grasp.npy',''):28s} demo{cR['demo_angle']:4.1f}° | 盖心 env {np.round(capc,3)} | 腕 {np.round(Mg[:3,3],3)} 腕高桌上 {100*(Mg[2,3]-TABLE):.1f}cm | 盖→腕 {np.round(appr,3)} (|{np.linalg.norm(appr)*100:.1f}cm|) | PreGrasp 退让方向 {np.round(pre_dir/np.linalg.norm(pre_dir),2)} | IK 抓 {rg['pos_err']*100:.2f}cm/{np.degrees(rg['rot_err']):.0f}° pre {rp['pos_err']*100:.2f}cm | 右臂最小限位余量 {marg(rg,ikR)}° | in_region {cR['in_region']} obst_touch {cR['obst_touch']}")
    return rl0, rl40, ML0, ML40
print("\n### 决赛几何 (瓶轴 f40 指向 -y = 机器人右侧; 盖在瓶顶)")
for lname,yaw,dr in [('1_Large_Diameter__v4_7_14_grasp.npy',280,32),('1_Large_Diameter__v4_7_14_grasp.npy',280,0)]:
    psi=(yaw+dr)%360; rr=right_at(psi)[:6]
    print(f"\n左 {lname.replace('_grasp.npy','')} yaw{yaw} Δroll{dr:+d} (psi40={psi}):")
    for d,f,n in rr:
        rl0,rl40,ML0,ML40=geom(lname,yaw,dr,f)
    print(f"  L 立瓶腕 env {np.round(ML0[:3,3],3)} (IK {rl0['pos_err']*100:.2f}cm, 余量 {np.round(np.minimum(rl0['q']-ikL.lower, ikL.upper-rl0['q']).min()*57.3,0)}°) | 放倒后腕 {np.round(ML40[:3,3],3)} 腕高桌上 {100*(ML40[2,3]-TABLE):.1f}cm (IK {rl40['pos_err']*100:.2f}cm, 余量 {np.round(np.minimum(rl40['q']-ikL.lower, ikL.upper-rl40['q']).min()*57.3,0)}°)")
