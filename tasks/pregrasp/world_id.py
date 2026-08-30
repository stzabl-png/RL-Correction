"""世界指纹 —— ckpt 只能在它出生的世界里回放。

★ 由来 (2026-08-28, AAG-F 回放 0/64 事故):
站姿 USD 是**世界版本**, 不是普通资产。换了它, 旧 ckpt 在新世界回放全员超时,
而 **obs 维度一个字节都不变** (348 维不变、语义变) ⟹ 维度闸拦不住。
同族还有: `RL_FC_PLAY_REF=0` (FC 目标点位移 6cm, 维度不变)。
共同点是**所有自动检查都给绿灯**, 因为形状/类型/存在性全都没问题。

用法:
    训练侧  write(log_dir)                 —— 开训时把世界指纹落进 run 目录
    回放侧  check(ckpt_path, allow=False)  —— 回放前对账, 不符则拒跑
"""
import hashlib
import json
import os
from datetime import datetime

_KEYS = ("md5", "bytes")          # 关键项: 不符即拒跑
_INFO = ("usd", "mtime", "env_override")   # 参考项: 只打印


def fingerprint():
    """采当前世界的指纹。★ 读不到时记 None (明示未知), 绝不填默认值 ——
    "读不到"被写成一个看起来正常的值, 是今天刚踩过的"假值"一族。"""
    try:
        from rl_rebuild.correction.env.dexmate_env_cfg import _DEXMATE_USD as u
    except Exception as e:
        return {"usd": None, "md5": None, "bytes": None,
                "error": f"{type(e).__name__}: {e}"}
    d = {"usd": u, "env_override": os.environ.get("DEXMATE_FIXED_USD") or None}
    try:
        with open(u, "rb") as f:
            d["md5"] = hashlib.md5(f.read()).hexdigest()
        d["bytes"] = os.path.getsize(u)
        d["mtime"] = datetime.fromtimestamp(os.path.getmtime(u)).isoformat(timespec="seconds")
    except Exception as e:
        d["md5"] = d["bytes"] = None
        d["error"] = f"{type(e).__name__}: {e}"
    return d


def write(run_dir):
    """开训时落盘。返回指纹 dict。"""
    w = fingerprint()
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "world.json"), "w") as f:
        json.dump(w, f, indent=2, ensure_ascii=False)
    _banner("本次训练的世界", w)
    return w


def _banner(title, w, extra=()):
    print("=" * 78, flush=True)
    print(f"[world] {title}", flush=True)
    print(f"  站姿USD  {w.get('usd')}", flush=True)
    print(f"  md5      {w.get('md5')}   ({w.get('bytes')} bytes, mtime {w.get('mtime')})", flush=True)
    if w.get("env_override"):
        print(f"  ⚠ 被 DEXMATE_FIXED_USD 覆写", flush=True)
    if w.get("error"):
        print(f"  ⚠ 采集失败: {w['error']}  —— 记为 None, 不是 '正常'", flush=True)
    for line in extra:
        print(f"  {line}", flush=True)
    print("=" * 78, flush=True)


def _find_saved(ckpt_path):
    """从 ckpt 位置向上找同目录/父目录的 world.json (ckpt 与其世界同生共死)。"""
    d = os.path.dirname(os.path.abspath(ckpt_path))
    for _ in range(4):
        p = os.path.join(d, "world.json")
        if os.path.exists(p):
            return p
        d = os.path.dirname(d)
    return None


def check(ckpt_path, allow_mismatch=False):
    """回放前对账。关键项不符 ⟹ os._exit(11)。

    ★ 用 os._exit 而非 raise: AppLauncher 之后 `raise SystemExit` 会被 Isaac 吞掉,
    退出码变 0 且照常跑完 —— "报了警却照样跑" 比不设闸更危险 (2026-08-29 教训)。
    """
    cur = fingerprint()
    saved_path = _find_saved(ckpt_path)
    if saved_path is None:
        _banner("当前世界 (★ 该 ckpt 没有 world.json, 无从对账)", cur,
                ["这不是'通过', 是'未验' —— 8/28 之前训的 ckpt 都没有指纹,",
                 "若回放结果异常(如全员超时), 世界错版是第一嫌疑。"])
        return None
    with open(saved_path) as f:
        old = json.load(f)
    bad = [k for k in _KEYS if cur.get(k) != old.get(k)]
    _banner("世界对账" + (" ❌ 不一致" if bad else " ✅ 一致"), cur,
            [f"ckpt 出生世界 {saved_path}",
             f"  md5 {old.get('md5')}  ({old.get('bytes')} bytes)"])
    if not bad:
        return True
    print("!" * 78, flush=True)
    print(f"[FATAL] ckpt 与当前世界不符 (关键项: {', '.join(bad)})", flush=True)
    print("  旧 ckpt 在新世界回放会**全员超时且无任何报错** —— obs 维度一字不变,", flush=True)
    print("  维度闸拦不住。实测前科: AAG-F 0/64。", flush=True)
    print("  解法: 用 DEXMATE_FIXED_USD 指向出生世界的 USD, 或换到那台机器。", flush=True)
    print("  确认要在错版世界里跑, 加 --allow_world_mismatch。", flush=True)
    print("!" * 78, flush=True)
    if allow_mismatch:
        print("[world] --allow_world_mismatch 已给, 继续 (结果不可信)", flush=True)
        return False
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(11)
