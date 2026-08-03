"""训练期诊断仪表 —— "设计→训练→纠错→优化" 闭环里的**仪表层**.

## 为什么要单独做这一层

env 现在只在 `_reset_idx` 里往 `self.extras` 写统计, 而 PPO 每步都
`self.extra_info = {}` 重建, 只有 horizon **最后一步**的那一份进 TensorBoard
(ppo.py:439). 于是曲线的分母是"这一步恰好 reset 的那几个 env", 单点非 0 即 1
—— CLAUDE.md 第 2 条坑说的就是它.

拿这种数去推翻设计假设是危险的: 你分不清"曲线跳"是设计错了还是仪表在抖.
**仪表不可信, 纠错闭环就是空转.** 所以这一层的职责只有一个: 让每条诊断量的
分母是**最近 window 个完整回合**, 而不是某一步的采样.

## 做法

逐 env 累计 -> 回合结束推进环形缓冲 -> 每步发布最近 window 个回合的均值.
发布的是 **0 维 tensor**: PPO 的 extras 过滤器认它 (ppo.py:442), 而且不用
`.item()`, 每步不触发 GPU 同步 (1024 env 下每步十几次同步是实打实的吞吐损失).

## 用法

    diag = EpisodeDiagnostics(num_envs, device)
    # 每个控制步:
    diag.tick(active)                                   # 谁在计分
    diag.add("contact2_frac", (n_contact >= 2).float(), active)
    diag.add("lift_peak_cm", lift_cm, active, mode="max")
    extras.update(diag.publish())
    # reset 时:
    diag.finish(env_ids)
"""
from __future__ import annotations

import torch


class EpisodeDiagnostics:
    """回合级滚动统计. 见模块 docstring."""

    def __init__(self, num_envs: int, device, window: int = 256, prefix: str = "diag/"):
        self.n = num_envs
        self.dev = device
        self.win = int(window)
        self.prefix = prefix
        self._acc: dict[str, torch.Tensor] = {}      # (N,) 本回合累计 (sum 或 max)
        self._cnt: dict[str, torch.Tensor] = {}      # (N,) 本回合有效步数 (mean 用)
        self._mode: dict[str, str] = {}
        self._any: dict[str, bool] = {}
        self._ring: dict[str, torch.Tensor] = {}     # (win,) 最近若干回合的值
        self._steps = torch.zeros(num_envs, device=device)   # 本回合走了几步 (滤空回合)
        self._ptr = 0
        self._filled = 0

    # ---- 采集 ----------------------------------------------------------
    def tick(self, active: torch.Tensor) -> None:
        """每个控制步调一次. active=(N,) bool, 只有计分中的 env 算步数."""
        self._steps += active.float()

    def add(self, key: str, value: torch.Tensor, mask: torch.Tensor | None = None,
            mode: str = "mean") -> None:
        """累计一个逐 env 的量.

        mode="mean": 对 mask 内的步求均值 (0/1 量就是"占比")
        mode="max" : 取 mask 内的峰值. ⚠ 只对**非负**量正确 (mask 外按 0 处理).
        """
        if key not in self._acc:
            self._acc[key] = torch.zeros(self.n, device=self.dev)
            self._cnt[key] = torch.zeros(self.n, device=self.dev)
            self._mode[key] = mode
            self._ring[key] = torch.full((self.win,), float("nan"), device=self.dev)
            self._any[key] = False
        v = value.float()
        m = torch.ones_like(v) if mask is None else mask.float()
        if mode == "max":
            self._acc[key] = torch.maximum(self._acc[key], v * m)
        else:
            self._acc[key] += v * m
            self._cnt[key] += m
        self._any[key] = True

    # ---- 结算 / 发布 ---------------------------------------------------
    def finish(self, env_ids: torch.Tensor) -> None:
        """回合结束: 把这些 env 的本回合值推进环形缓冲, 并清零它们的累计器."""
        if not self._acc:
            return
        # 空回合 (init 时的那次 reset, 一步都没走过) 不能进窗口 —— 会掺一堆假 0
        ids = env_ids[self._steps[env_ids] > 0]
        if len(ids) == 0:
            self._reset_envs(env_ids)
            return
        k = len(ids)
        slots = (self._ptr + torch.arange(k, device=self.dev)) % self.win
        for key, acc in self._acc.items():
            if self._mode[key] == "max":
                val = acc[ids]
            else:
                cnt = self._cnt[key][ids]
                # ⚠ 掩码一次都没命中的回合必须记 NaN, **不能记 0**.
                # 踩过: afford_hit 的掩码是"有接触", 整个回合没碰到物体的 env 会被记成
                # 0.0 拉低均值 —— 读出来像"接触点很差", 其实是"根本没接触".
                # place_err 更糟: 没走到 hold 段的回合记 0cm, 看起来像**完美放置**.
                val = torch.where(cnt > 0, acc[ids] / cnt.clamp(min=1.0),
                                  torch.full_like(cnt, float("nan")))
            self._ring[key][slots] = val
        self._ptr = int((self._ptr + k) % self.win)
        self._filled = int(min(self._filled + k, self.win))
        self._reset_envs(env_ids)

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        for key in self._acc:
            self._acc[key][env_ids] = 0.0
            self._cnt[key][env_ids] = 0.0
        self._steps[env_ids] = 0.0

    def publish(self) -> dict[str, torch.Tensor]:
        """最近 window 个回合的均值 (跳过 NaN 回合). 0 维 tensor, 不触发 GPU 同步."""
        if self._filled == 0:
            return {}
        out = {}
        for k, v in self._ring.items():
            w = v[:self._filled]
            out[f"{self.prefix}{k}"] = w.nanmean()          # 全 NaN -> NaN, 判读端会跳过
            if self._mode[k] != "max":
                # 这条量有多少回合真的有样本 —— 覆盖率太低的均值不可信
                out[f"{self.prefix}{k}__cov"] = (~w.isnan()).float().mean()
        out[f"{self.prefix}_window_n"] = torch.tensor(float(self._filled), device=self.dev)
        return out
