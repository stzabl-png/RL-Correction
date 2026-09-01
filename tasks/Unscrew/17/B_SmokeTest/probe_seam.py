"""零动作从 t0 播机器段+缝1 到站位: 逐 5 步记录左腕 目标(FK of ff) vs 实际, 垫数, 瓶位, 看缝1 进刀有没有到位."""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); args = p.parse_args()
app = AppLauncher(args).app
import numpy as np, torch, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "part4", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
import task_env as PE, task_config as TC
from rl_rebuild.correction.kinematics import ArmIK
cfg = PE.build_cfg(num_envs=1); E = PE.UnscrewEnv(cfg); E.force_entry = [0]; E.reset()
rest = json.load(open(TC.REST_JSON)); ik = ArmIK("left", anchor_link="arm_center", anchor_T=np.asarray(rest["anchor_T_left"]))
org = E.scene.env_origins[0]
z = np.load(E._master_path, allow_pickle=True); rows = np.where(z["source"] == 1)[0]; IA0 = int(rows[0]); APP = E.APP_END
print(f"[seam] APP_END={APP} IA0={IA0} T_ROW={E.T_ROW}", flush=True)
act = torch.zeros(1, PE.ACT_DIM, device=E.device)
for t in range(IA0 + 60):
    E.step(act)
    r = int(E.row[0]); ff = E._ff[0, 7:14].cpu().numpy(); q = E.hand.data.joint_pos[0, E.map_ids_t[7:14]].cpu().numpy()
    pt, _ = ik.fk(ff); pa = (E.hand.data.body_pos_w[0, E.wid["L"]] - org).cpu().numpy()
    f = E._pads_f().norm(dim=-1)[0]; nl = int((f[:5] > 0.5).sum())
    bot = (E.object.data.root_pos_w[0] - org).cpu().numpy()
    if t % 5 == 0 or (APP - 3 <= r <= IA0 + 2):
        print(f"[seam] t{t:3d} row{r:3d} 左腕 目标 {np.round(pt,3)} 实际 {np.round(pa,3)} |Δ| {np.linalg.norm(pt-pa)*100:.1f}cm (z差 {(pa[2]-pt[2])*100:+.1f}) | 关节差max {np.degrees(np.abs(q-ff).max()):.1f}° | 左垫 {nl} | 瓶 {np.round(bot,3)} | cert_phase {int(E.PB.cert_phase[0])} g1 {bool(E.PB.g1[0])} g2 {bool(E.PB.g2[0])}", flush=True)
    if bool(E.PB.g2[0]): print("[seam] G2 达成"); break
print("[seam] done", flush=True); app.close(); os._exit(0)
