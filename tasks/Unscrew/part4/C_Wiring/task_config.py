"""任务参数单一来源 —— Unscrew(拧瓶盖) 实例 (对应 CHECKLIST 第 1 步)。

框架身: Pour17 v5 已验收版 (steps/step4_rl/framework, 2026-08-28)。
本实例 2026-08-29 落地, 数据 = datasets/unscrew_bottle (egodex_part4, 18 条重建,
物体逐帧 conf_pos/conf_rot + 人手逐帧置信度)。

★ 数据引擎口径: 一个实例吃全部 17 条可用 clip —— `UNSCREW_CLIP=<n>` 选条
  (默认 32 = README 榜首, 双物体最低分 85), 所有路径/母带/场景由它派生。
  逐条生成演示数据的循环见 C_Wiring/data_engine.sh。

★ 物体编号约定 (与 PourEnv 同构, 但与数据集相反, 别搞混):
  母带/进度机 obj_0 = 瓶身 (env.object, 左手持) | obj_1 = 盖 (env.aux, 右手拧)
  数据集 object_0=瓶 object_1=盖 (17/18 条; 构建器按尺寸认件, 不信编号)
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
TASK_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
REPO = os.path.abspath(os.path.join(TASK_ROOT, "..", "..", ".."))

# ---- 任务身份 ----
CLIP_ID = os.environ.get("UNSCREW_CLIP", "32")   # 数据引擎逐条覆写
TASK = f"Unscrew{CLIP_ID}"
CLIP = f"unscrew{CLIP_ID}_task"                  # clips.configure_cfg 场景名
TAKE_DIR = os.path.join(REPO, "datasets", "unscrew_bottle", CLIP_ID)

# ---- 三 Prior 路径 ----
# 母带: v1=离线构建 (make_reference.py; 交互段腕参考=物体轨迹推导, 上游腕平移
#       是静态填充死数据 —— clip32 实测全程位移 <0.6cm 而盖走 66cm),
#       v2=Isaac 内物体轨迹反解 IK 重铸 (build_reference.py 产物)
_L2 = os.path.join(TASK_ROOT, "A_Design", "L2_Reference", CLIP_ID)
REF_V1 = os.path.join(_L2, "reference_v1.npz")
REF_V2 = os.path.join(_L2, "reference_v2.npz")
REST_JSON = os.path.join(_L2, "env_rest.json")   # probe_rest.py 产物 (env 实测静置)
ACCEPTANCE_JSON = os.path.join(_L2, "acceptance_v2.json")
# GraspPose 模板: 左手瓶 = Screw27_body (CAD 与本批字节相同, md5 已核对)。
# ⚠ 基类 prior 脚手架**不接** (2026-08-30 拍板, 三轮实测):
#   ① _load_grasp_prior 会把物体重摆成 prior 的 canon 姿态 (日志实锤"物体仍
#      平放于桌面") —— 与本任务"数据驱动直立摆放+盖预合拢"语义直接冲突;
#   ② 它不做侧别镜像 (右手抓姿给左臂解 = 前臂压桌病态构型);
#   ③ 退避族外壳判据只看 z 不看桌面 xy 覆盖, 对本站位结构性误判。
#   V5 接线下基类 approach/verify 机制本来就被全覆写 —— PRIOR_MAIN 留空；
#   task_env 以显式 bypass_lift_scaffold 生成同设备零占位。
PRIOR_MAIN = ""
# squeeze 层 / 离线 v1 左抓锚使用原始右手约定 prior；make_reference 对位姿
# (x,-y,z)/(w,-x,y,-z) 只镜像一次，指值按名映射到左手关节。
PRIOR_AUX = os.path.join(
    REPO, "tasks", "pregrasp", "priors", "Screw27_body.npz")
PRIOR_APPROACH_DEG = -1.0        # 不覆写 yaw (canon_rot 原样)
# ★ 右手盖侧 GraspPose 候选 (2026-09-01 用户裁定"右手没做重定向? 必须马上做")。
# 原设计 βR=0、右指直接用人手指流 —— 而人手在拧盖窗内合拢中位仅 +1.0°, 配人手
# 腕位/指长够用, 搬到 SharpaWave 上就成了"手张在盖外围一圈": probe_pinch 实测
# 三指落在盖系径向 6.4~11.2cm (盖半径仅 1.75cm), 合到 45° 仍零接触, 且食/中指
# 越合越往外 —— 说明缺的是抓取先验, 不是合拢量。
# 这批候选与左手 Screw27_body 同格式 (grasp/squeeze/pregrasp/contact_pos),
# 且本来就是**右手**约定, 不需镜像; 接触点半径 1.9cm 正落在盖面上。
PRIOR_CAP_DIR = os.path.join(REPO, "tasks", "pregrasp", "priors",
                             "Screw27_cap_candidates")
# 逐 clip 定档 (make_reference 按可达率扫; 空=扫描选优, 非空=钉死某个候选)
# 2026-09-01 实测定档: 8_0_calib 是唯一"IK 0.01cm 可达 + 径向对准盖轴"的候选
CAP_GRASP_PICK = os.environ.get("UNSCREW_CAP_GRASP", "8_0_calib")
# 沿盖轴的下压微调 (与左手的 PRIOR_RADIAL_TRIM 同性质): prior 说接触点在盖系
# z≈+0.5cm, 而镜像套到这只手上 probe_pinch 实测指垫在 z+2.8~+6.3cm —— 指长/
# 掌形差异, 手停高了。用 probe_pinch 量出来回填 (换 clip/换手都要复测)。
# 抓握对准量 (盖坐标系, m): probe_capgrasp --fit 闭环实测 —— 量到指垫实际位置后
# 把腕平移到"拇/食指垫中点落在盖心、盖半高", 两轮收敛的累计平移。先验的腕高对
# 这只手偏高 ~6cm, 这个量就是补它的。换 clip/换手用 --fit 复测。
# ⚠ 必须是**盖坐标系**的量: 盖带 yaw, 把闭环的世界系累计平移当局部量用会转错
# (2026-09-01 实测踩到: 母带回放的指垫落到 r5.5~9.5cm, 而对准好的是 r2.8~4.2)。
# 下面这组 = probe_capgrasp --fit 2 打印的"对准后腕位(盖系) − 候选标称"。
CAP_GRASP_TRIM = (0.0046, 0.0171, -0.0417)
CAP_GRASP_Z_TRIM = 0.0        # 已并入 CAP_GRASP_TRIM, 保留键名兼容
# 站位行的额外捏合量 (度): 对准只把拇/食指放到盖轴两侧, 跨距仍比盖径大 ~1.8cm,
# 靠这一档补上。probe_capgrasp --curl 实测 25° 首次接触 (三指1/3, 5.8N)。
# 剩下的收拢交给 RL 的手指残差 (±68.8° 总量, 绰绰有余) —— 参考给到位, RL 修完。
CAP_PINCH_DEG = 25.0
# 数据集自带的盖侧 affordance (60k 点接触频率热区, 逐 clip)
AFFORDANCE_CAP = os.path.join(TAKE_DIR, "contact", "expected_area_object_1_right.npz")

# ---- squeeze 剂量 ---------------------------------------------------------
# reference 是 RL correction 的先验，不要求零动作抓稳。probe_beta 在 clip32 上
# β=1/1.5/2/3 均未通过零动作持握；这里取 β=1，恰好回放 prior 自身的 squeeze，
# 不使用 >1 的关节外推。接触/穿模由策略在有界残差内修正。
BETA_R = 1.0     # [TASK] 右手盖: 2026-09-01 起接上 Screw27_cap 候选的 squeeze
                 #        (原为 0 = 没有重定向, 右手全程碰不到盖 -> 拧转收入恒 0)
BETA_L = 1.0     # [TASK] 左手瓶: 2026-08-31 实测定档 —— 锚点/进刀/预张开修好后
                 #        β=1.0 站位行 **4 垫接触** (G1 只要 3), 瓶全程不倒;
                 #        β=0.6 反而掉到 2 垫 (拇指 30N、其余脱开)。拇指偏硬
                 #        (45N, 右手 prior 按名镜像到左手时拇指最不对称) 是**已知
                 #        瑕疵**, 交给 RL 残差修 —— 这正是 correction 的职责。

# ---- 交互段持瓶朝向重定向 (U35c 逐 clip 标定; make_reference 消费) ----
# 人举瓶的朝向对人顺手, 对机器人肘几何常常不可达 —— 绕世界 z 转一个角度,
# 位置与倾角全不动 (瓶是旋转体, 判据只看轴倾角), 但双臂可达性天差地别。
# clip32 扫描实测 (2026-08-31, 见 DECISIONS T2-6):
#   0° -> 右臂 43% / 左 100%;  -35° -> 右 77% / 左 100%;  -70° -> 右 69% / 左 98%
# 逐 clip 值缺省 0; 新 clip 上线前用同一扫描定一次 (UNSCREW_HOLD_YAW 可覆写)。
HOLD_YAW_BY_CLIP = {"32": -35.0}
# 机器段 pregrasp 净空: 从站位抓握位姿沿"离开物体"的方向让开多少 (cuRobo 的
# cspace 目标就是这个构型的限位内点解)。2026-08-31 提高左手径向净空: 左抓锚
# 修准之后 pregrasp 落在瓶壁上, 充气 10mm 的障碍直接把**目标构型**判碰,
# Approach 全灭 (诊断: "目标原地微动 ❌被判碰")。右手抬升保持 4cm ——
# T2-2 实测 8cm 已超可达域, ≤5cm 才规划得通。
PRE_L_RADIAL = 0.12
PRE_R_LIFT = 0.04
HOLD_YAW_DEG = float(os.environ.get(
    "UNSCREW_HOLD_YAW", HOLD_YAW_BY_CLIP.get(CLIP_ID, 0.0)))

# ---- 物体几何 (Success Tracker 判据原料; CAD 全批统一, md5 已核对) ----
BOTTLE_HALF_H = 0.0985           # 瓶身长轴半长 (mesh 实测 19.7cm/2)
CAP_HALF_H = 0.0085              # 盖半高 (1.7cm/2)
UP_LOCAL_BOTTLE = (0.0, 0.0, 1.0)   # 瓶局部竖直轴 = z (螺轴, 与 screw_assembly 一致)
UP_LOCAL_CAP = (0.0, 0.0, 1.0)
CAP_RADIUS = 0.0175              # 盖半径 (贴近奖/判据几何)
# 右腕锚的"腕→指尖垂距": 决定手停在盖上方多高。演进: 0.203(旧, 张开平手的值)
# -> 0.157(2026-08-31 probe_grasp 实测捏握手型) -> 0.100(2026-09-01)。
# 为什么还要再降: probe_pinch 实测**合到 40° 仍零接触** —— 掌心朝下时手指屈曲是
# 把指尖往掌心收(向上), 而盖在下方, 越合越够不着。必须把腕降到"盖落在指间"的高度,
# 合拢才有意义。改动后必须用 probe_pinch 复测 (它会报三指接触数与指力)。
HAND_DROP = 0.100
# 缝1 的"先到位再合手"分点: 前 SEAM_MOVE_FRAC 手臂进刀 + 手保持张开, 之后手臂
# 停住只合手指并加 squeeze。母带 (make_reference) 与 env/探针的加压曲线必须用
# 同一个分点, 否则手还在路上就被压上去 —— 实测那样会把 0.53kg 的瓶推倒。
SEAM_MOVE_FRAC = 0.7
# 左抓锚的径向微调 (向瓶轴收): prior 记录的接触点在半径 3.2~3.6cm 的瓶面上,
# 而镜像锚照搬过来后指垫实测落在 4.2~5.6cm —— 差这 1~2cm, 手指就不是"环抱"
# 而是"斜推", 且推点在瓶身上部 (+15cm) 力臂长, 0.53kg 的自由瓶一推就倒。
# 用 probe_grasp --pin_bottle 量出来的差值回填 (换 clip/换手都要复测)。
PRIOR_RADIAL_TRIM = 0.007
# squeeze 的**逐关节增量封顶**: prior 的 squeeze 相对 grasp 中位只多合 4.6°,
# 但拇指要多合 64.6° —— 右手 prior 按名镜像到左手时拇指最不对称, 结果拇指
# 顶进瓶身 45N 把瓶推过 D2 死线, 而其他手指才刚碰上。封到 20°: 其他手指的
# 合拢原样保留, 拇指不再顶死。(母带不变, 这是运行时前馈的量)
import numpy as _np_cfg
# 2026-08-31 实测: 封顶 20° 把抓握一起掐掉 (站位垫 4->0), 45° 仍掉到 0 ——
# 拇指那 64.6° 的合拢量正是"把手真正收到瓶面上"的主力, 削不得。设成 180°
# (=不生效), 拇指偏硬 45N 留给 RL 残差修 (见 DECISIONS T2-6d)。
SQUEEZE_DELTA_CAP = float(_np_cfg.radians(180.0))


def seam_squeeze_profile(u):
    """u∈[0,1] 沿缝1 的进度 -> squeeze 系数 (前段 0, 后段线性到 1)。"""
    import numpy as _np
    return _np.clip((_np.asarray(u, float) - SEAM_MOVE_FRAC)
                    / max(1.0 - SEAM_MOVE_FRAC, 1e-6), 0.0, 1.0)

# ---- 螺纹口径 (数据引擎逐条可覆写; 默认承旧台账用户裁定) ----
# turns=0.75 (U30b: 演示实测 ~266° 分离; 2.0 圈是标准件假设, 难 2.7×)
# 释放判据 = screw_assembly 的 detach (拧满 turns 即脱开, screw_detach_at_full)
# 2026-09-01 用户裁定: 瓶盖转 **30°** 即可拧下 (已破封的松盖). 原 0.75 圈
# (270°, U30b 演示实测分离角) 对机器手是三倍难度且需换把。
SCREW_TURNS = float(os.environ.get("UNSCREW_TURNS", str(30.0 / 360.0)))

# ---- 机器段 (cuRobo 规划产物; plan_machine_segs.py 逐 clip 产出) ----
# 没有规划产物时 make_reference 退回关节 smoothstep 占位 (无碰撞背书, 只够冒烟)。
# 本机使用 NVlabs/curobo 新版主线；worker 依次查 CUROBO_ROBOT_YML、MagicSim
# 兼容树和仓库自带的 vega_1p_sharpa_curobo.yml。
MOTION_DIR = os.path.join(TASK_ROOT, "A_Design", "L1_Data", "Motion_Planning",
                          CLIP_ID)
APPROACH_NPZ = os.path.join(MOTION_DIR, "Approach.npz")
RETREAT_NPZ = os.path.join(MOTION_DIR, "Retreat.npz")


def file_md5(path):
    import hashlib

    try:
        with open(path, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()
    except OSError:
        return None


def rest_anchor_T(side, path=None):
    """Load the measured shared arm-center anchor for one side."""
    import json

    import numpy as np

    if side not in ("right", "left"):
        raise ValueError(f"invalid arm side: {side}")
    rest_path = path or REST_JSON
    with open(rest_path, encoding="utf-8") as fh:
        rest = json.load(fh)
    key = f"anchor_T_{side}"
    value = np.asarray(rest.get(key), dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{rest_path}: {key} must be a finite 4x4 matrix")
    return value


def reference_planning_digest(path):
    """Hash exactly what Approach/Retreat planning consumes —— 不多不少。

    plan_machine_segs 实际读的是: 两个 cspace 目标构型 machine_pre_q_*、
    它们所锚的 station_w*、以及 Retreat 世界里的两个障碍位置 (交互**末行**
    的瓶/盖位置)。机器行本身排除在外, 于是规划剪回母带后摘要不变。

    2026-08-30 收窄: 原实现把**整段交互行** (含 right_q/left_q 与逐行物体
    位姿) 一起哈希 —— 与自己的 docstring 矛盾, 且把"改交互段姿态参考"这种
    与规划无关的改动也判成规划失效, 逼出无谓的重规划。反过来说, 只要
    machine_pre/station/末行障碍任一变了, 摘要照样变 —— 该重规划的一次不漏。
    """
    import hashlib

    import numpy as np

    if isinstance(path, dict):          # 就地校验: 直接吃内存里的数组
        import contextlib
        ctx = contextlib.nullcontext(path)
    else:
        ctx = np.load(path, allow_pickle=True)
    with ctx as z:
        src = np.asarray(z["source"], dtype=np.int8)
        rows = np.flatnonzero(src == 1)
        if not len(rows):
            raise ValueError(f"{path}: no interaction rows")
        end = rows[-1:]
        h = hashlib.sha256()
        for key, arr in (
                ("station_wr", z["station_wr"]), ("station_wl", z["station_wl"]),
                # cspace 机器段的真实规划目标 (2026-08-30 起); 旧母带无此键
                *((("machine_pre_q_r", z["machine_pre_q_r"]),
                   ("machine_pre_q_l", z["machine_pre_q_l"]))
                  if "machine_pre_q_r" in (z.files if hasattr(z, "files") else z)
                  else ()),
                # Retreat 世界的障碍: 交互末行的瓶/盖落点
                ("obj_pos_0_end", z["obj_pos_0"][end]),
                ("obj_pos_1_end", z["obj_pos_1"][end])):
            data = np.ascontiguousarray(arr, dtype=np.float64)
            h.update(key.encode())
            h.update(str(data.shape).encode())
            h.update(data.tobytes())
    return h.hexdigest()[:32]


def acceptance_receipt_issues(reference_path, *, acceptance_path=None):
    """Validate the durable trainability receipt without launching Isaac."""
    import json

    receipt_path = acceptance_path or ACCEPTANCE_JSON
    if not os.path.isfile(receipt_path):
        return [f"缺少 v2 训练稳定性凭据: {receipt_path}"]
    try:
        with open(receipt_path, encoding="utf-8") as fh:
            receipt = json.load(fh)
    except Exception as exc:
        return [f"验收凭据不可读: {type(exc).__name__}: {exc}"]
    issues = []
    if receipt.get("schema") != "unscrew_trainability_v1":
        issues.append(f"验收 schema={receipt.get('schema')} 不受支持")
    if str(receipt.get("clip")) != CLIP_ID:
        issues.append(f"验收 clip={receipt.get('clip')} != {CLIP_ID}")
    current_md5 = file_md5(reference_path)
    if not current_md5 or receipt.get("reference_v2_md5") != current_md5:
        issues.append("验收凭据未绑定当前 reference_v2.npz")
    try:
        if (int(receipt.get("stable_envs", -1)) < 3
                or int(receipt.get("num_envs", -1)) != 4):
            issues.append("训练稳定环境数不足 3/4")
    except (TypeError, ValueError):
        issues.append("训练稳定环境数不可读")
    if not isinstance(receipt.get("world"), dict):
        issues.append("验收凭据缺完整 world 指纹")
    return issues


def training_reference_issues(path, *, expected_path=None, parent_v1_path=None,
                              rest_path=None, acceptance_path=None):
    """Return reasons why *path* is not a formal, trainable mother tape."""
    import json

    import numpy as np

    expected = os.path.abspath(expected_path or REF_V2)
    issues = []
    if os.path.abspath(path) != expected:
        issues.append(f"正式训练必须使用 reference_v2.npz: got {path}")
    if not os.path.isfile(path):
        return issues + [f"母带不存在: {path}"]
    try:
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(np.asarray(z["meta"]).item()))
    except Exception as exc:
        return issues + [f"母带或 meta 不可读: {type(exc).__name__}: {exc}"]

    if str(meta.get("clip")) != CLIP_ID:
        issues.append(f"v1 meta clip={meta.get('clip')} != {CLIP_ID}")
    if meta.get("rest_source") != "probe":
        issues.append(f"rest_source={meta.get('rest_source')}，必须来自 probe_rest")
    for seg in ("approach", "retreat"):
        source = meta.get(seg)
        if not isinstance(source, str) or not source.startswith("curobo:"):
            issues.append(f"{seg}={source}，必须是 cuRobo 规划段")
    screw = meta.get("screw", {})
    try:
        if abs(float(screw.get("turns")) - SCREW_TURNS) > 1e-9:
            issues.append(f"母带 turns={screw.get('turns')} != runtime {SCREW_TURNS}")
    except Exception:
        issues.append("母带 screw.turns 缺失或不可读")

    try:
        parent_v1 = parent_v1_path or REF_V1
        actual_basis = reference_planning_digest(parent_v1)
        if meta.get("planning_basis_digest") != actual_basis:
            issues.append("母带 planning_basis_digest 缺失或与交互几何不符")
    except Exception as exc:
        issues.append(f"规划基底不可核对: {type(exc).__name__}: {exc}")
    current_rest_md5 = file_md5(rest_path or REST_JSON)
    if not current_rest_md5 or meta.get("rest_md5") != current_rest_md5:
        issues.append("母带 rest_md5 与当前 env_rest.json 不符")

    if "meta_v2" not in z.files:
        issues.append("缺 meta_v2（尚未经过 Isaac 内 v2 重铸）")
    else:
        tokens = {}
        for item in str(np.asarray(z["meta_v2"]).item()).split(";"):
            if "=" in item:
                key, value = item.split("=", 1)
                tokens[key] = value
        if tokens.get("gen") != "unscrew_v2":
            issues.append(f"meta_v2.gen={tokens.get('gen')} != unscrew_v2")
        if tokens.get("clip") != CLIP_ID:
            issues.append(f"meta_v2.clip={tokens.get('clip')} != {CLIP_ID}")
        parent_md5 = file_md5(parent_v1_path or REF_V1)
        recorded_parent = tokens.get("parent_v1_md5")
        if (not parent_md5 or not recorded_parent
                or not parent_md5.startswith(recorded_parent)):
            issues.append("meta_v2.parent_v1_md5 与当前 reference_v1.npz 不符")

        # IK 误差/穿模属于 correction 的学习对象，只要求诊断字段完整可读，
        # 不再要求 reference 零动作物理完美。
        try:
            int(tokens["critical_bad"])
        except Exception:
            issues.append("meta_v2.critical_bad 缺失或不可读")
        for key, expected_beta in (
                ("betaL", float(os.environ.get("POUR_BETA_L", str(BETA_L)))),
                ("betaR", float(os.environ.get("POUR_BETA_R", str(BETA_R))))):
            try:
                if abs(float(tokens[key]) - expected_beta) > 1e-9:
                    issues.append(f"meta_v2.{key}={tokens[key]} != runtime {expected_beta}")
            except Exception:
                issues.append(f"meta_v2.{key} 缺失或不可读")

    required = ("right_q", "left_q", "obj_pos_0", "obj_quat_0",
                "obj_pos_1", "obj_quat_1", "source", "seg_lens")
    for key in required:
        if key not in z.files:
            issues.append(f"缺核心数组 {key}")
        elif not np.isfinite(np.asarray(z[key], dtype=np.float64)).all():
            issues.append(f"核心数组 {key} 含 NaN/Inf")
    issues.extend(acceptance_receipt_issues(
        path, acceptance_path=acceptance_path))
    z.close()
    return issues


def require_training_reference(path):
    """Reject bootstrap/place-holder references before Isaac is launched."""
    issues = training_reference_issues(path)
    if not issues:
        print(f"[reference] ✅ 正式母带预检通过: {path}", flush=True)
        return
    detail = "\n  - ".join(issues)
    if os.environ.get("UNSCREW_ALLOW_UNVERIFIED_REF") == "1":
        print("[reference] ⚠ UNSCREW_ALLOW_UNVERIFIED_REF=1，显式绕过正式母带预检:"
              f"\n  - {detail}", flush=True)
        return
    raise RuntimeError("正式训练母带未通过预检:\n  - " + detail)

# ---- 时钟/重采样 ----
FPS_RECON = 15.0                 # 重建帧率 (replay_world.fps; 单时钟=母带行轴)

# ============ Unscrew/17 拔盖变体 (2026-09-01, 台账 tasks/Unscrew/17/A_Design/DECISIONS.md) ============
# 用户裁定: 不拧只拔 (U2) / 盖 3g·μ5 (U4) / cuRobo 只给左手·右手参考由数据定 (U6) /
# 右手接触前跟人手活通道、接触后盖定位置 (U8) / 右手闭环 (U9) / GraspPose 选型 §8。
# 开关全部 env 驱动, 默认 = 同事 clip32 原口径不变。
DETACH_MODE = os.environ.get("UNSCREW_DETACH", "twist")     # twist | pull
PULL_N = float(os.environ.get("UNSCREW_PULL_N", "3.0"))      # 拔出阈值 (N), 探针标定量
PULL_MEFF = float(os.environ.get("UNSCREW_PULL_MEFF", "0.2"))
# 原生先验 (Dexonomy 直接为这只手/这套 CAD 生成, 物体系=CAD 系, 不镜像、不做原点修正):
#   左 Screw17_bottle_left = 1_Large_Diameter__v4_7_14 (立瓶 yaw 280 = 人手方位 140°)
#   右 Screw17_cap_candidates/33_Inferior_Pincer__1_47 (装配态盖系, 近侧水平捏盖沿)
# 选型依据与离线 IK 数据: tasks/Unscrew/17/A_Design/L1_Data/GraspPose/ (台账 §8)。
NATIVE_PRIORS = os.environ.get("UNSCREW_NATIVE_PRIORS", "1" if CLIP_ID == "17" else "0") == "1"
LEFT_YAW_PREF_DEG = None                 # 左抓方位扫描的偏好 (None = 纯可达率择优)
LEFT_YAW_PREF_TOL = 20.0
RIGHT_CL = os.environ.get("UNSCREW_RIGHT_CL", "0") == "1"    # U9 右手闭环 (env 内差分 IK 伺服)
if NATIVE_PRIORS:
    PRIOR_AUX = os.path.join(REPO, "tasks", "pregrasp", "priors", "Screw17_bottle_left.npz")
    PRIOR_CAP_DIR = os.path.join(REPO, "tasks", "pregrasp", "priors", "Screw17_cap_candidates")
    CAP_GRASP_PICK = os.environ.get("UNSCREW_CAP_GRASP", "")   # 目录里只有选定的那一个
    CAP_GRASP_TRIM = (0.0, 0.0, 0.0)      # 同事的量是给镜像 Screw27 候选的; 原生先验从零起, 探针复测再填
    CAP_PINCH_DEG = 0.0                   # 原生候选 grasp 本身就合在盖沿上 (3 接触点)
    PRIOR_RADIAL_TRIM = float(os.environ.get("UNSCREW_RADIAL_TRIM", "0.0"))   # 站位探针标定 (正=向瓶轴收)
    LEFT_YAW_PREF_DEG = 280.0             # 台账 §8: 人手接近方位 (144°) 对应的候选 yaw (旧口径, 保留)
    LEFT_AZ_PREF_DEG = 140.0              # 左腕相对瓶心的**世界**方位角偏好 (人手 144°, 候选 yaw280 → 140°)
    # 接触行 盖→腕 目标方向 (世界系, 台账 §8 1_47 @ psi312: 近侧(-x)、水平略高): 定瓶滚转规范
    CAP_WRIST_DIR_WORLD = (-0.14, -0.02, 0.03)
    # 机器段净空梯 (左径向, 左抬升, 右抬升): 右腕站位离桌仅 3cm, cuRobo 保守碰撞球判桌碰,
    # 右抬升给到 10~20cm (离线 IK 余量 20~27°, 台账 §8 复核)
    PRE_LADDER_OVERRIDE = [(0.12, 0.00, 0.10), (0.12, 0.06, 0.12), (0.16, 0.06, 0.15),
                           (0.20, 0.10, 0.15), (0.24, 0.14, 0.18), (0.28, 0.18, 0.20)]
    HOLD_YAW_BY_CLIP.setdefault(CLIP_ID, 0.0)
    HOLD_YAW_DEG = float(os.environ.get("UNSCREW_HOLD_YAW", HOLD_YAW_BY_CLIP.get(CLIP_ID, 0.0)))
