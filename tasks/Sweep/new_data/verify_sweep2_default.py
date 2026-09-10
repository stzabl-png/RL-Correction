"""Compare Sweep2's default world before Task3 wiring and current wiring."""
from pathlib import Path
import argparse,sys,os,types,subprocess
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--version',choices=['head','working'],required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
slot=isaac_slot('task3_sweep2_default_regression');app=AppLauncher(a).app
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[3]
path=ROOT/'tasks/Sweep/2/C_Wiring/sweep_env.py'
sys.path.insert(0,str(path.parent))
if a.version=='head':
    module=types.ModuleType('sweep_default_baseline');module.__file__=str(path)
    source=subprocess.check_output(['git','show','HEAD:tasks/Sweep/2/C_Wiring/sweep_env.py'],cwd=ROOT,text=True)
    exec(compile(source,str(path),'exec'),module.__dict__)
else:
    import sweep_env as module
raw=module.SweepEnv(module.build_cfg(1))
raw.suppress_terminal_reset=True
raw.reset()
trace={k:[] for k in ['q','tools','row','gates','reward','obs','priv']}
with torch.no_grad():
    for t in range(160):
        raw.step(torch.zeros(1,14,device=raw.device))
        obs=raw._get_observations()
        trace['q'].append(raw.hand.data.joint_pos[0,raw.map_ids_t].cpu().numpy().copy())
        trace['tools'].append(torch.cat([raw.object.data.root_state_w[0],raw.aux.data.root_state_w[0],raw.cube.data.root_state_w[0]]).cpu().numpy().copy())
        trace['row'].append(int(raw.row[0]));trace['gates'].append(raw._tick_out['gates'][0].cpu().numpy().copy())
        trace['reward'].append(float(raw._tick_out['reward'][0]))
        trace['obs'].append(obs['policy'][0].cpu().numpy().copy());trace['priv'].append(obs['priv_info'][0].cpu().numpy().copy())
out=ROOT/'logs/task3_take32_v2_20260905'/('sweep2_default_'+a.version+'.npz')
np.savez_compressed(out,**{k:np.array(v) for k,v in trace.items()},
    ref_arm=raw.ref_arm.cpu().numpy(),finger_q=raw.fixed_finger_q.cpu().numpy())
print('[default regression]',a.version,out,flush=True)
slot.release();sys.stdout.flush();os._exit(0)
