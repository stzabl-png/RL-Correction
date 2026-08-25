# PourS-v9S (2026-08-24 用户裁定): carry4 手锚参考 **从头训** (不分叉, 检验"新参考让学习更容易吗")
#   配方 = v8D 全套 (奖励/判据/obs/action 同) + carry4 + 显式 --ff_pull 0
#   自由段行号/预算由 train.py 从 npz free_lo/hi 自动接管 ([85,165), budget 256)
#   账本: ~/poursv9s_steprew (15项逐步 + 8探针env, 1000行/片)
cd ~/RL_Correction && source env_a6000.sh && export RL_ISAAC_NO_GUARD=1 TMPDIR=$HOME/tmp
CUDA_VISIBLE_DEVICES=0 RL_HAND_JOINTS=1 PYTHONPATH=. $PY -u -m tasks.pregrasp.train --headless \
  --name POURSV9S_pour17 --clip Pour17_bottle --num_envs 1024 --seed 42 \
  --prior_npz tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz --prior_yaw 19.5 \
  --bimanual --bi_native --prior_b tasks/pregrasp/priors/Pour17_cup_thumbfix.npz --prior_b_yaw 90 \
  --curobo_ref tasks/pour/carry4a_pour17.npz --carry_npz tasks/pour/carry4a_pour17.npz \
  --ff_freeze_cm 0 --ff_pull 0 --dyn_far 2.0 --pregrasp_only --phase2 --pour_carry --carry_progress \
  --carry_tilt_ms 30,60,90 --carry_squeeze --carry_pad_reward --carry_stable \
  --table_fail --pour_succ --pour_free --grasp_cent_fix --pours_v5 --pours_v6 \
  --grip_prog_gate 20 --tilt_pot 5.0 --prog_pot 40.0 --ff_play --prog_align --prog_joint --mouth_w 60.0 \
  --step_rew_log ~/poursv9s_steprew --load_path logs/POURSV9S_pour17/2026-08-24_03-23-28/stage1_nn/last.pth --resume \
  --approach --minimal --kl_threshold 0.02 --auto_stop dry --max_agent_steps 40000000 \
  > ~/poursv9s.log 2>&1
echo "EXIT=$?" >> ~/poursv9s.log
