import sys, numpy as np
from scipy.spatial.transform import Rotation as Rt
P = np.load("tasks/pregrasp/priors/Screw17_bottle_left.npz", allow_pickle=True)
def qR(q): return Rt.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
def Rq(R): x, y, z, w = Rt.from_matrix(R).as_quat(); return np.array([w, x, y, z])
def variant(theta_deg, dz):
    g = P["grasp"].copy(); c = P["contact_centroid"].copy()
    w = g[:3]; r = np.array([w[0], w[1], 0.0]); r /= np.linalg.norm(r); t = np.cross([0, 0, 1.0], r)
    R = Rt.from_rotvec(t * np.radians(theta_deg)).as_matrix(); off = np.array([0, 0, dz])
    def xf(pose): p = c + R @ (pose[:3] - c) + off; return np.concatenate([p, Rq(R @ qR(pose[3:7])), pose[7:]])
    out = {k: P[k] for k in P.files}
    out.update(grasp=xf(g), squeeze=xf(P["squeeze"]), pregrasp=np.stack([xf(pp) for pp in P["pregrasp"]]),
               contact_pos=(c + (P["contact_pos"] - c) @ R.T) + off, contact_normal=P["contact_normal"] @ R.T,
               contact_centroid=c + off, source=np.array(f"{P['source']} | pitch {theta_deg}deg about contact centroid (wrist-up) + dz {dz}"))
    return out
for theta, dz, name in ((-15, 0.03, "P15u3"), (-20, 0.02, "P20u2")):
    v = variant(theta, dz); np.savez(f"tasks/pregrasp/priors/Screw17_bottle_left_{name}.npz", **v)
    print(name, "wrist", np.round(v["grasp"][:3], 3), "quat", np.round(v["grasp"][3:7], 3), "contacts z", np.round(v["contact_pos"][:, 2].min(), 3), "~", np.round(v["contact_pos"][:, 2].max(), 3), "pregrasp[0]", np.round(v["pregrasp"][0, :3], 3))
