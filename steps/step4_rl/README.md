# step4_rl —— RL 残差修正

## framework/ —— 任务基础框架(可实例化)

新任务从这里开始:拷贝 `framework/` → 填 `C_Wiring/task_config.py` → 按
`framework/CHECKLIST.md` 逐格打勾。框架身 = Pour17 v5.1 已验收版
(判据自检家族 5 件全绿 + 双变体训练冒烟 + 零动作硬闸链)。

设计总览:`../../docs/V5_TRAINING_FRAMEWORK.md`
(三 Prior + Success Tracker 架构、场景/训练设定、两种参考驱动体制)

## 现役任务实例

- **Pour17**(倒水,双手):`../../tasks/Pour/17/` —— 六线 v5.1 训练中
  (P-HYB×3 / P-OBJ×3);全史权威 `tasks/Pour/17/A_Design/DECISIONS.md`。
- RL_Pour(旧倒水线,v9S):`RL_Pour/`

## 迁移欠账

`tasks/Pour/17` 与 `rl_rebuild/` 仍在上游位置(训练在跑,路径被三机发射脚本引用)。
待训练轮次之间再按 `../README.md` 的迁移约定迁入,迁入时需同步三机脚本与
autorecord 路径。
