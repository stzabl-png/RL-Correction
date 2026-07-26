"""Isaac Sim GPU 独占槽位 + 录像期让出 —— 2026-07-24 整机断电事故后加的防护.

## 为什么需要这个

这台机器 (RTX 5090 / 600W 上限 + i7-14700K / PL1 已被主板解到 253W) 的供电余量极薄.
两次整机瞬断 (2026-07-22 14:46、2026-07-24 16:15) 在 journal 里都**没有任何** shutdown /
panic / OOM / thermal 记录 —— 日志直接断在一条无关的 WiFi 信号行上, 这是电源 OCP
(过流保护) 瞬间切断的特征, 而不是软件崩溃.

诱因是**功耗尖峰的 di/dt**, 不是持续功耗. 最容易制造尖峰的两种情况:
  1. 两个 Isaac Sim 进程同时在 GPU 上满载 (典型: 训练还在跑, autorecord 又拉起一个
     带渲染的录像进程; 或多个 Agent 各自起训练). 2026-07-24 15:39:33~15:40:12 实际
     发生过一次重叠, 内存冲到 82%, 那次侥幸没崩.
  2. PPO 的 rollout <-> update 交替本身就是周期性方波负载, 单进程已经够呛.

本模块管住第 1 类.

## 两条机制

**A. 全局互斥 (flock)** —— 任何时刻只允许一个 Isaac 进程占用 GPU 槽位. 所有 Isaac
入口 (train / record / play / eval_policy / ...) 启动 `AppLauncher` **之前**先拿锁,
拿不到就阻塞等待, 而不是并发挤上去.

**B. 录像让出 (pause 协议)** —— 训练是长跑, 会一直握着锁, 录像进程等不到.
所以 autorecord 先落一个 PAUSE 标记文件, 训练在 **epoch 边界** (刚存完 checkpoint 的
干净点) 检查到就释放锁并空转挂起; 录像进程拿到锁 -> 录完 -> **进程完全退出** ->
autorecord 静置 SETTLE 秒让显存和 carb 释放干净 -> 删掉标记; 训练再重新拿锁继续.

全程 GPU 上只有一个进程在算, 训练不丢进度 (只是墙钟变长).

## 环境变量

  RL_ISAAC_LOCK_DIR   锁/标记文件目录 (默认 ~/.cache/rl_correction)
  RL_ISAAC_SETTLE     录像退出后静置秒数 (默认 10; 见 CLAUDE.md 第 5 条 carb mutex)
  RL_ISAAC_MAX_PAUSE  训练最长挂起秒数, 超时强制恢复 (默认 900), 防录像挂死拖死训练
  RL_ISAAC_NO_GUARD=1 整个机制关掉 (只在确认电源问题已解决后用)
"""
import atexit
import errno
import fcntl
import os
import time
from pathlib import Path

LOCK_DIR = Path(os.environ.get("RL_ISAAC_LOCK_DIR",
                               Path.home() / ".cache" / "rl_correction"))
LOCK_PATH = LOCK_DIR / "isaac.lock"
PAUSE_PATH = LOCK_DIR / "pause.request"

SETTLE_SEC = float(os.environ.get("RL_ISAAC_SETTLE", "10"))
MAX_PAUSE_SEC = float(os.environ.get("RL_ISAAC_MAX_PAUSE", "900"))
DISABLED = os.environ.get("RL_ISAAC_NO_GUARD") == "1"

_POLL = 2.0


def _log(msg):
    print(f"[isaac-slot] {msg}", flush=True)


def _pause_is_stale():
    """PAUSE 标记的属主进程是否已经死了.

    autorecord.sh 把自己的 PID 写进标记文件. 它的 `trap ... EXIT` 只是尽力而为 ——
    bash 在前台子进程 (录像) 运行期间会**推迟**信号处理, `kill -9` 更是完全绕过 trap.
    所以不能只靠 trap 清标记, 否则 autorecord 被强杀后训练要一直挂到 MAX_PAUSE 超时
    (默认 900s) 才自愈. 这里主动查属主活没活, 死了就立刻放行.

    标记为空或内容不是 PID 时一律**当作有效** —— 刚 touch 出来还没写入的瞬间不能误判.
    """
    try:
        raw = PAUSE_PATH.read_text().strip()
    except OSError:
        return False
    if not raw.isdigit():
        return False
    try:
        os.kill(int(raw), 0)          # 只探活, 不发信号
    except ProcessLookupError:
        return True
    except PermissionError:
        return False                  # 进程还在 (只是不属于我们)
    return False


class IsaacSlot:
    """GPU 独占槽位. 用 flock 实现, 进程意外死掉时内核自动释放 (无残留死锁)."""

    def __init__(self, role):
        self.role = role
        self._fh = None
        self._held = False

    # ---- 内部 ----
    def _open(self):
        if self._fh is None:
            LOCK_DIR.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(LOCK_PATH), os.O_RDWR | os.O_CREAT, 0o644)
            self._fh = os.fdopen(fd, "r+")
        return self._fh

    def _holder(self):
        try:
            with open(LOCK_PATH) as f:
                return f.read().strip() or "未知"
        except OSError:
            return "未知"

    def _stamp(self):
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(f"pid={os.getpid()} role={self.role}\n")
        self._fh.flush()

    # ---- 对外 ----
    def acquire(self, timeout=None):
        """阻塞直到拿到槽位. timeout=None 表示无限等 (训练/录像都该无限等)."""
        if DISABLED or self._held:
            return self
        fh = self._open()
        t0 = time.time()
        announced = False
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if not announced:
                    _log(f"[{self.role}] GPU 槽位被占用 (持有者: {self._holder()}), 排队等待中 ...")
                    announced = True
                if timeout is not None and time.time() - t0 > timeout:
                    raise TimeoutError(
                        f"等待 GPU 槽位超过 {timeout}s (持有者: {self._holder()})")
                time.sleep(_POLL)
        self._held = True
        self._stamp()
        if announced:
            _log(f"[{self.role}] 已获得槽位 (等了 {time.time() - t0:.0f}s)")
        return self

    def release(self):
        if DISABLED or not self._held:
            return
        try:
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.flush()
        except OSError:
            pass
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._held = False

    def yield_if_paused(self):
        """训练侧的让出点 —— 在 epoch 边界调用.

        看到 PAUSE 标记就释放槽位并挂起, 直到标记消失 (= 录像进程已完全退出并静置完)
        再重新拿回槽位. 没有标记时开销就是一次 stat, 可以每 epoch 调.
        """
        if DISABLED or not PAUSE_PATH.exists():
            return False
        if _pause_is_stale():
            _log(f"[{self.role}] 发现僵尸让出标记 (autorecord 已死), 直接清掉不挂起")
            try:
                PAUSE_PATH.unlink()
            except FileNotFoundError:
                pass
            return False
        _log(f"[{self.role}] 收到录像让出请求 -> 释放 GPU 槽位, 训练挂起")
        self.release()
        t0 = time.time()
        while PAUSE_PATH.exists():
            if _pause_is_stale():
                _log(f"[{self.role}] 让出标记的属主进程已消失 (autorecord 被强杀?), 立即恢复")
                try:
                    PAUSE_PATH.unlink()
                except FileNotFoundError:
                    pass
                break
            if time.time() - t0 > MAX_PAUSE_SEC:
                _log(f"[{self.role}] ⚠ 挂起已超 {MAX_PAUSE_SEC:.0f}s 仍未解除 "
                     f"(录像多半失败/卡死), 清掉标记强制恢复")
                try:
                    PAUSE_PATH.unlink()
                except FileNotFoundError:
                    pass
                break
            time.sleep(_POLL)
        self.acquire()
        _log(f"[{self.role}] 录像结束, 训练恢复 (共挂起 {time.time() - t0:.0f}s)")
        return True

    # 兼容 with 语句
    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def isaac_slot(role, timeout=None):
    """便捷入口: 拿到槽位并注册 atexit 释放. 在 AppLauncher 之前调用.

        from rl_rebuild.utils.gpu_guard import isaac_slot
        slot = isaac_slot("train")
        app = AppLauncher(args).app
    """
    slot = IsaacSlot(role).acquire(timeout=timeout)
    atexit.register(slot.release)
    return slot
