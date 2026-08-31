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
# 数据集自带的盖侧 affordance (60k 点接触频率热区, 逐 clip)
AFFORDANCE_CAP = os.path.join(TAKE_DIR, "contact", "expected_area_object_1_right.npz")

# ---- squeeze 剂量 (CHECKLIST 第 3 步: probe_beta 标定后回填, 现值=初值) ----
BETA_R = 0.0     # [TASK] 右手盖: 无 squeeze prior (三指精捏交给人手指流+残差)
BETA_L = 2.0     # [TASK] 左手瓶: Screw27_body squeeze; 瓶 0.53kg 与 Pour17 同级,
                 #        初值抄 Pour17 重物侧实证值, 必须 probe_beta 复标

# ---- 物体几何 (Success Tracker 判据原料; CAD 全批统一, md5 已核对) ----
BOTTLE_HALF_H = 0.0985           # 瓶身长轴半长 (mesh 实测 19.7cm/2)
CAP_HALF_H = 0.0085              # 盖半高 (1.7cm/2)
UP_LOCAL_BOTTLE = (0.0, 0.0, 1.0)   # 瓶局部竖直轴 = z (螺轴, 与 screw_assembly 一致)
UP_LOCAL_CAP = (0.0, 0.0, 1.0)
CAP_RADIUS = 0.0175              # 盖半径 (贴近奖/判据几何)

# ---- 螺纹口径 (数据引擎逐条可覆写; 默认承旧台账用户裁定) ----
# turns=0.75 (U30b: 演示实测 ~266° 分离; 2.0 圈是标准件假设, 难 2.7×)
# 释放判据 = screw_assembly 的 detach (拧满 turns 即脱开, screw_detach_at_full)
SCREW_TURNS = float(os.environ.get("UNSCREW_TURNS", "0.75"))

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


def reference_planning_digest(path):
    """Hash only geometry consumed by Approach/Retreat planning.

    Machine rows are deliberately excluded, so the digest stays stable after
    the two plans are spliced back into v1/v2.
    """
    import hashlib

    import numpy as np

    with np.load(path, allow_pickle=True) as z:
        src = np.asarray(z["source"], dtype=np.int8)
        rows = np.flatnonzero(src == 1)
        if not len(rows):
            raise ValueError(f"{path}: no interaction rows")
        h = hashlib.sha256()
        for key, arr in (
                ("station_wr", z["station_wr"]), ("station_wl", z["station_wl"]),
                # cspace 机器段的真实规划目标 (2026-08-30 起); 旧母带无此键
                *((("machine_pre_q_r", z["machine_pre_q_r"]),
                   ("machine_pre_q_l", z["machine_pre_q_l"]))
                  if "machine_pre_q_r" in z.files else ()),
                ("right_q", z["right_q"][rows]), ("left_q", z["left_q"][rows]),
                ("obj_pos_0", z["obj_pos_0"][rows]),
                ("obj_quat_0", z["obj_quat_0"][rows]),
                ("obj_pos_1", z["obj_pos_1"][rows]),
                ("obj_quat_1", z["obj_quat_1"][rows])):
            data = np.ascontiguousarray(arr, dtype=np.float64)
            h.update(key.encode())
            h.update(str(data.shape).encode())
            h.update(data.tobytes())
    return h.hexdigest()[:32]


def acceptance_receipt_issues(reference_path, *, acceptance_path=None):
    """Validate the durable physical-acceptance receipt without launching Isaac."""
    import json

    receipt_path = acceptance_path or ACCEPTANCE_JSON
    if not os.path.isfile(receipt_path):
        return [f"缺少 v2 物理验收凭据: {receipt_path}"]
    try:
        with open(receipt_path, encoding="utf-8") as fh:
            receipt = json.load(fh)
    except Exception as exc:
        return [f"验收凭据不可读: {type(exc).__name__}: {exc}"]
    issues = []
    if receipt.get("schema") != "unscrew_acceptance_v1":
        issues.append(f"验收 schema={receipt.get('schema')} 不受支持")
    if str(receipt.get("clip")) != CLIP_ID:
        issues.append(f"验收 clip={receipt.get('clip')} != {CLIP_ID}")
    current_md5 = file_md5(reference_path)
    if not current_md5 or receipt.get("reference_v2_md5") != current_md5:
        issues.append("验收凭据未绑定当前 reference_v2.npz")
    try:
        if int(receipt.get("passes", -1)) < 3 or int(receipt.get("num_envs", -1)) != 4:
            issues.append("验收通过数不足 3/4")
    except (TypeError, ValueError):
        issues.append("验收通过数不可读")
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

        try:
            if int(tokens["critical_bad"]) != 0:
                issues.append(f"meta_v2.critical_bad={tokens['critical_bad']}，关键 IK 窗未过")
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
