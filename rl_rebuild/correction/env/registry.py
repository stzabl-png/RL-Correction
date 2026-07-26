"""按 --robot 选环境实现. 所有入口(train / eval / record / play)统一走这里,
避免每个脚本各自 import 一套, 加新机器人时漏改。

  flying   飞手 SharpaWave: 腕=浮动根刚体, wrench-PD 推, 关重力. 动作 28 维(笛卡尔残差)
  dexmate  DexMate(Vega)+Sharpa: 腕=7 关节链末端, 电机 PD 驱动, 带重力. 动作 29 维(关节残差)

两者共用 reward / RSI / 冻结窗 / 成功判定 / 日志 (DexmateCorrectionEnv 继承飞手环境),
所以同一条 clip 上的数可以直接对照。详见 docs/DEXMATE_TRAINING_FEASIBILITY.md
"""
from __future__ import annotations

ROBOTS = ("flying", "dexmate")


def make_env(robot: str):
    """-> (EnvCls, CfgCls). 必须在 AppLauncher 之后调用 (import 链会拉起 isaaclab)."""
    if robot == "dexmate":
        from rl_rebuild.correction.env.dexmate_env import DexmateCorrectionEnv
        from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg
        return DexmateCorrectionEnv, DexmateCorrectionEnvCfg
    if robot == "flying":
        from rl_rebuild.correction.env.correction_env import SharpaCorrectionEnv
        from rl_rebuild.correction.env.correction_env_cfg import SharpaCorrectionEnvCfg
        return SharpaCorrectionEnv, SharpaCorrectionEnvCfg
    raise ValueError(f"未知 --robot {robot!r}, 可选 {ROBOTS}")
