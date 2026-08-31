"""★闸门自审 (判据家族第7件): 世界指纹比对必须可红可绿, 且不得"缺数据即通过"。

判别法: 造一个必然该红的输入喂进去, 看它红不红。
本次自审抓到的真实缺陷 (L5-16): compare() 原写法
    if key in a and key in b and _ne(key)
键缺失或值为 None 时**直接跳过 = 静默通过** —— 与 ⑥c 空表同族。
"""
import copy
import os
import sys

_CW = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "C_Wiring"))
sys.path.insert(0, _CW)
import world_fingerprint as WF  # noqa: E402

BASE = {"schema": "world_fingerprint_v1",
        "robot": {"usd_md5": "a" * 32,
                  "controlled_joint_names_in_order": ["j1", "j2"]},
        "reference": {"md5": "b" * 32},
        "policy_io": {"obs_dim": 503, "act_dim": 58},
        "time": {"control_dt_s": 0.05, "decimation": 12},
        "table": {"table_top_z_m": 0.87},
        "objects": {"object_1": {"mass_kg": 0.53, "static_friction": 3.0},
                    "object_0": {"mass_kg": 0.15, "static_friction": 0.5}},
        # ★L5-26: 判据摘要也是关键项 —— 改阈值=改成败判定。
        # 它进 CRITICAL 之后, **没有这两个键的旧指纹会被判为"无法核对"**,
        # 这是设计意图(老 ckpt 拿不出判据摘要, 就不能声称核对过), 不是 bug。
        "criteria": {"schema": 1, "digest": "0" * 16},
        # ★L5-31: conf_flat 消融旗也是关键项 —— 它把置信度整条链路拍平, 时钟门
        # 从红档 8cm 变 5cm、朝向从禁判变照判, **直接改成败判定**(主判据
        # clock_done 就靠时钟门)。而 criteria.digest 抓不到它: 常量一个没变,
        # 变的是"每一行用哪一档"。所以必须单列。
        # 与 L5-27 加 criteria.digest 时同理: 它进 CRITICAL 后, 没有这个键的
        # 旧指纹会被判"无法核对" —— 那是设计意图, 不是 bug。
        "switches": {"conf_flat": False}}


def run(tag, mut):
    cur = copy.deepcopy(BASE)
    mut(cur)
    c, w, u = WF.compare(BASE, cur)
    print(f"  {tag:32s} 不符={len(c)} 无法核对={len(u)}")
    return len(c), len(u)


print("对照(必须全绿):")
a = run("完全一致", lambda d: None)
print("必然该红的输入(每种破坏一个关键项):")
reds = [
    run("① 母带换版", lambda d: d["reference"].__setitem__("md5", "c" * 32)),
    run("② 站姿USD换版", lambda d: d["robot"].__setitem__("usd_md5", "d" * 32)),
    run("③ 桌高改了", lambda d: d["table"].__setitem__("table_top_z_m", 0.85)),
    run("④ 物体质量改了", lambda d: d["objects"]["object_1"].__setitem__("mass_kg", 0.1)),
    run("⑥ 判据阈值变了(digest)",
        lambda d: d["criteria"].__setitem__("digest", "1" * 16)),
    run("⑦ 判据清单扩了(schema)",
        lambda d: d["criteria"].__setitem__("schema", 2)),
    run("⑤ 关节表顺序变了",
        lambda d: d["robot"].__setitem__("controlled_joint_names_in_order",
                                         ["j2", "j1"])),
    # ★L5-31 新增: 拿 flat 训的 ckpt 在 base 环境评测必须被拒
    run("⑧ conf_flat 旗变了",
        lambda d: d["switches"].__setitem__("conf_flat", True)),
]
print("缺数据的输入(必须标为'无法核对', 不得静默通过):")
u1 = run("⑥ 关键项键缺失", lambda d: d["robot"].pop("usd_md5"))
u2 = run("⑦ 关键项值为None", lambda d: d["robot"].__setitem__("usd_md5", None))

ok = (a == (0, 0) and all(x[0] == 1 for x in reds)
      and u1 == (0, 1) and u2 == (0, 1))
print("✅ 世界闸门可红可绿, 且缺数据不冒充通过" if ok else "★世界闸门有问题 —— 见上")
sys.exit(0 if ok else 1)
