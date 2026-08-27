# AAG-F → DP 数据采集 RUNBOOK(双卡机/旧站姿场景)

给在双卡机(128.32.164.89)上执行 rollout 采集的同事。一页,五步。

## 背景一句话
`eval_best.pth` = AAG-F 双手贴合确定性 100% 权重(6.6M),**它出生在旧站姿世界**
(躯干锁死 USD md5 `f77f235d…`)。你机器上就是这个旧世界,所以在你那边采集口径天然正确。
⚠ **绝对不要用 `best.pth`**——那是按训练奖励挑的,极可能是 13.1M 训过头(eval 56%)的权重,
采集器已内建拒收。

## 第 0 步(硬闸,先于一切):世界指纹对账

维度闸对"世界换版"是瞎的(形状不变语义变)。**md5 不对不许跑**:
```bash
md5sum assets/vega_1p_sharpa_fixedtorso.usd
# 期望: f77f235df53494508fefdf1d1518af32  (旧站姿 = AAG-F 出生世界)
# 若是: 143385ad217f10f3fe730338d748246d  (新站姿) → 必须 export
#   DEXMATE_FIXED_USD=$PWD/assets/vega_1p_sharpa_fixedtorso_stance0803.usd
```

## 步骤

1. **拉代码**:`git pull`(分支 Step3_Dexonomy),然后**重做第 0 步**(pull 会换 USD)。
2. (已并入第 0 步)
   ```bash
   md5sum assets/vega_1p_sharpa_fixedtorso.usd
   # f77f235d... = 旧世界, 直接跳到第3步
   # 143385...   = 被换成新站姿了 → 后续命令加:
   #   export DEXMATE_FIXED_USD=$PWD/assets/vega_1p_sharpa_fixedtorso_stance0803.usd
   ```
3. **挑卡**:`nvidia-smi`,选空卡(写卡号进 CUDA_VISIBLE_DEVICES)。
4. **冒烟 128 集**:
   ```bash
   cd RL_Correction
   export SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=$PWD \
     RL_HAND_JOINTS=1 RL_FC_PLAY_REF=1 OMNI_KIT_ACCEPT_EULA=YES \
     CUDA_VISIBLE_DEVICES=<卡号>
   python -u tasks/pregrasp/dp_collect.py \
     --ckpt results/AAG_pour17_solved/eval_best.pth \
     --n_target 128 --num_envs 64 --headless
   ```
   采集器自带三道闸,**任何一道不过会自己停**(不会静默产坏数据):
   - 维度闸: obs 必须 348(旗集/代码漂移在此炸响)
   - 预检闸: 前 128 回合确定性成功率 ≥90%(基准 100%@n1024)
   - best.pth 拒收
5. **回传**:`tasks/Pour/17/B_SmokeTest/dp_staging_aagf/`(episode_*.npz + manifest.json)
   整目录打包发回。审计通过后再跑扩量(`--n_target 500`)。

## 产出口径(已冻结, 勿改)
state(T,116) f32 = 臂R7+臂L7+指R22+指L22 + 腕R7+腕L7 + 杯7+瓶7 + 触R15+触L15
action(T,58) f32 = 绝对关节目标(非残差) | 20Hz | wxyz w≥0 | 只导成功回合 |
meta 含 world 戳(stance0803)——此批与新站姿数据**不同世界**,混训必须声明。
边界:AAG-F 体制无抬升,数据测"双手贴合闭环流",不外推抓起/任务能力。
