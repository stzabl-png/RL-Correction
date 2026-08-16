"""自动停止的判定状态机.

**纯 Python, 不 import torch/isaac** —— 所以能脱离 GPU 直接单测
(`tasks/pregrasp/test_auto_stop.py`)。train.py 里只留"什么时候调它"和"怎么执行
停止", 判据本身全在这里, 两边共用同一份代码 (判据逻辑复制一份到测试里必然漂移)。

判据设计见会话记录 2026-08-09; 三条线:
  ① solved  : 确定性成功率 ≥ target_sr 连续 target_hits 次   (早退, 好任务用)
  ② plateau : 最佳成功率连续 patience 次没涨够 delta        (主判据, 兜底所有任务)
  ③ 硬顶    : PPO 自己的 max_agent_steps (不在这里)

两个把门的前提, 每个都对应一种误停:
  · min_steps  —— 之前一律不判. 难物体爬坡期很长, 早期平坦不等于学不会.
  · curriculum —— 课程在推进时 (变难) 平坦是设计内现象, 不能判平台期;
    但课程**停摆**时这个理由不成立 (见 train.py:_curriculum_signature),
    此时放行平台期判据, 否则 sr 卡低位 → 课程冻住 → 永远停不下来。
    solved 分支任何时候都要求课程完全退火完 —— 没退火完的高成功率
    是在更容易的任务上刷的, 不是对外口径。
"""


class StopDecider:
    def __init__(self, target_sr=0.99, target_hits=2, patience=5, delta=0.01,
                 min_steps=0):
        self.target_sr = target_sr
        self.target_hits = target_hits
        self.patience = patience
        self.delta = delta
        self.min_steps = min_steps
        self.best_sr = -1.0
        self.hits = 0          # 连续达标次数
        self.stale = 0         # 连续"没涨"次数
        self.fired = False     # 是否已经触发过 (只记第一次)

    def update(self, sr, curriculum_done, curriculum_stalled, agent_steps):
        """喂一次评测结果, 返回停止原因 (str) 或 None. 会更新 best_sr/hits/stale."""
        self.hits = (self.hits + 1) if (curriculum_done and sr >= self.target_sr) else 0
        if curriculum_done or curriculum_stalled:
            # ⚠ 先用**旧的** best_sr 判"涨没涨", 再更新 best_sr.
            #   顺序反了 sr 永远不可能 > 刚被自己推高的 best, stale 会一直累加 (或恒 0),
            #   判据失效. 所以 best_sr 的更新放在函数末尾。
            self.stale = 0 if sr > self.best_sr + self.delta else self.stale + 1
        else:
            self.stale = 0     # 课程还在推进, 平坦是设计内现象

        reason = None
        if agent_steps < self.min_steps:
            pass               # 步数下限内只记账
        elif self.hits >= self.target_hits:
            reason = (f"solved: 确定性成功率 {sr*100:.2f}% ≥ {self.target_sr*100:.0f}% "
                      f"连续 {self.hits} 次")
        elif self.stale >= self.patience:
            reason = (f"plateau: 最佳 {self.best_sr*100:.2f}%, 连续 {self.stale} 次评测未提升 "
                      f">{self.delta*100:.0f}pt"
                      f"{' (课程停摆)' if curriculum_stalled and not curriculum_done else ''}")

        if sr > self.best_sr:
            self.best_sr = sr
        if reason and not self.fired:
            self.fired = True
            return reason
        return None            # 已经触发过就不重复报
