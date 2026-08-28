# Task Framework(Noisy-World RL 任务基础框架)

> 框架身 = Pour17 v5 **已验收版**(2026-08-28:判据自检家族全绿 + 双变体训练冒烟全绿
> + 零动作硬闸链)。新任务 = 拷贝本目录 → 填参数 → 按 `CHECKLIST.md` 逐格打勾。
> 总体设计与两种参考体制见 `../../../docs/V5_TRAINING_FRAMEWORK.md`(仓库根 docs/)。

## 目录(与 Pour/17 同构)

```
your_task/
  A_Design/
    DECISIONS.md          ← 全史唯一权威: 每个拍板/标定/事故都落账 (模板已给)
    L2_Reference/         ← reference_v1.npz(人手行版, 流水线产物)
                            reference_v2.npz(build_reference.py 产物)
    L3_Learning/          ← Success Tracker 判据双版 + 自检家族(5件) + REWARD_DOC
  B_SmokeTest/            ← 剂量标定/验收硬闸/认证通过率/IK核查 探针
  C_Wiring/               ← task_config.py(参数单一来源) + env + 训练循环 + 冒烟/评测/录像
```

## 实例化流程(详见 CHECKLIST.md,顺序不可乱)

1. **填 `C_Wiring/task_config.py`**(资产/Prior 路径/物体几何);
2. 把重建流水线的母带放到 `A_Design/L2_Reference/reference_v1.npz`;
3. **β 剂量标定**:`B_SmokeTest/probe_beta.py`(零动作四档同场,关键动作段能扛才算数;
   记住:深捏垫数变少,垫数≠握力);
4. **重铸母带**:`build_reference.py`(物体轨迹反解 IK + 认证行;高速窗按需时间扩张)→
   `probe_ikcheck.py` 验运动学(坏行必须不在关键窗);
5. **改 Success Tracker**:`progress.py` 里所有 `[TASK]` 块(G3 核心判据整块可换)→
   **双版同步** `progress_batch.py` → 自检家族四件全绿
   (`selftest_progress / _batch / _rsi / _variants / _regime`);
6. **验收硬闸**:`probe_acceptance.py` 零动作全链——过不了的段要么修参考,要么在
   DECISIONS.md 明确记档"此段留给 RL"并得到裁定;
7. **训练冒烟**:短发真训练(双变体各一),TB 针全在线 + world.json 指纹正确;
8. 发射(`launch_remote.sh`),判读针**预登记**。

## 硬纪律(全部有过事故学费)

- 判据改动 = 双版同步 + 自检家族重跑,文档随改;
- ckpt 绑定世界版本(world.json:母带 md5/USD/变体/β);回放必回出生世界;
- 成功率只认独立 eval 确定性口径;TB 只认挣的(RSI 预置不计);
- 环境旗保留 `POUR_*` 前缀(框架沿革,勿改名——三机脚本/文档全部按此对齐)。
