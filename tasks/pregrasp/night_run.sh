#!/bin/bash
# 夜班流水线 (2026-07-30): A/B 实验 -> 双评测 -> 双录像 -> 追加 Grasp2 6mm 课程.
# 全程串行, 同一时刻只有一个 Isaac (电源红线). 阶段标记行 "=== " 供监视器播报.
set -e
cd /home/lyh/Project/RL_Correction
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
PRIOR=tasks/pregrasp/priors/Grasp5.npz
STEPS=20000000

run() { SHARPA_WANDB=0 PYTHONPATH=. $PY "$@"; }

echo "=== [1/6] A 组基线训练开始 (Grasp5 无 prior, 20M 步) $(date +%H:%M) ==="
run -m tasks.pregrasp.train --headless --clip Grasp5 --name AB_Grasp5_base \
    --num_envs 1024 --max_agent_steps $STEPS
sleep 15
echo "=== [2/6] B 组 prior 训练开始 (Grasp5 + GraspPose, 20M 步) $(date +%H:%M) ==="
run -m tasks.pregrasp.train --headless --clip Grasp5 --name AB_Grasp5_prior \
    --num_envs 1024 --max_agent_steps $STEPS --grasp_prior
sleep 15

CKA=$(ls -td logs/AB_Grasp5_base/*/ | head -1)stage1_nn/last.pth
CKB=$(ls -td logs/AB_Grasp5_prior/*/ | head -1)stage1_nn/last.pth
echo "=== [3/6] A 组确定性评测 $(date +%H:%M) ==="
run -m tasks.pregrasp.eval --headless --clip Grasp5 --checkpoint "$CKA" \
    --steps 300 > logs/AB_eval_A.log 2>&1 || echo "=== ⚠ A 评测失败 ==="
sleep 15
echo "=== [4/6] B 组确定性评测 $(date +%H:%M) ==="
run -m tasks.pregrasp.eval --headless --clip Grasp5 --checkpoint "$CKB" \
    --grasp_prior $PRIOR --steps 300 > logs/AB_eval_B.log 2>&1 || echo "=== ⚠ B 评测失败 ==="
sleep 15
echo "=== [5/6] 录像 (A 后 B) $(date +%H:%M) ==="
run -m tasks.pregrasp.record --headless --clip Grasp5 --checkpoint "$CKA" \
    --fps 10 || echo "=== ⚠ A 录像失败 ==="
sleep 15
run -m tasks.pregrasp.record --headless --clip Grasp5 --checkpoint "$CKB" \
    --grasp_prior $PRIOR --fps 10 || echo "=== ⚠ B 录像失败 ==="
sleep 15
echo "=== [AB-DONE] A/B 全部完成, 结果就绪 $(date +%H:%M) ==="

echo "=== [6/6] 追加: Grasp2 6mm 课程 (热启动, 跑到被叫停为止) $(date +%H:%M) ==="
run -m tasks.pregrasp.train --headless --clip Grasp2 --name PreGrasp_0_j6 \
    --num_envs 1024 --obj_jitter 0.006 \
    --load_path logs/PreGrasp_0/2026-07-29_17-54-22/stage1_nn/last.pth --resume
echo "=== [NIGHT-END] 夜班全部结束 $(date +%H:%M) ==="
