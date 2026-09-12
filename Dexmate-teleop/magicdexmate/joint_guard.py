"""Final-command guard: per-joint velocity clamp + self-collision hold.

Ported from V2AP-demo's SmoothingAndSafetyManager (arm_hand_control.py), the
layer that lets their rig move fast without ever visiting a broken pose. Their
recipe, verified by reading the code on 2026-08-06, is not a smarter solver -
it is a slower-converging solver plus TWO guards at the command exit:

  1. per-joint velocity clamp:  dq in +-scale * v_max / hz   (their scale 0.4)
  2. boolean self-collision check on the commanded pose; on hit, HOLD the
     previous safe command instead (never send the colliding one)

This module is that exit layer for our stack. It sits AFTER the IK and the
final one-euro smoothing, immediately before the command is used - so Isaac,
the state publisher, the recorder and the hardware bridge all see guarded
commands, and sim equals real by construction.

Why clamping to the ROBOT'S OWN v_max is not the "one-size rate limit" that
was rejected on 2026-08-05: that objection was about arbitrary limits eating
the operator's real fast wrist twists. v_max here is the hardware spec
([2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7] rad/s - identical numbers in the URDF,
in V2AP's constants, and in the real arm). Motion beyond it does not exist on
hardware whatever we command; clamping merely makes the simulation stop
pretending otherwise.

Collision pairs are curated HERE, not taken from the official SRDF: the SRDF's
link names (`base`, ...) do not match what pinocchio derives from this URDF
(`vega_1_*`), so removeCollisionPairs filters only 33 of 771 pairs and leaves
a base-vs-wheel false positive standing at the working pose. We keep only the
pairs that can actually happen in teleop: cross-arm, and each arm's distal
half against torso/head/base.

KNOWN GAP: the Sharpa hands are not in vega_1.urdf, so their volume is not
checked. `margin` buys some slack; the bridge's 80 mm wrist-gap door stays as
the last line. Deliberately NO aggressive virtual hand spheres - the operator's
standing complaint is that the hands cannot come close ENOUGH; a guard must
not forbid intentional close-hands work.

Self-check (no hardware):  .venv/bin/python -m magicdexmate.joint_guard
"""
from __future__ import annotations

import pathlib

import numpy as np
import pinocchio as pin

DEFAULT_URDF = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()

# Real-arm velocity limits [rad/s]. URDF == V2AP constants == hardware spec.
ARM_VMAX = np.array([2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7])
ARM_JOINTS = {s: [f"{s}_arm_j{i}" for i in range(1, 8)] for s in ("L", "R")}

# Link-name cores (the geometry parent frames carry an inconsistent `vega_1_`
# prefix, so membership is tested on the stripped name).
_CROSS_ARM = [f"arm_l{i}" for i in range(3, 9)]        # elbow-down vs elbow-down
_DISTAL = [f"arm_l{i}" for i in range(5, 9)]           # forearm/wrist vs body
_BODY = ["torso_l1", "torso_l2", "torso_l3",
         "head_l1", "head_l2", "head_l3", "vega_1"]    # vega_1 = base pillar


def geoms_of(by_link: dict, prefix: str, cores) -> list:
    """按连杆名取几何体下标(与 _build_collision 里那个同名闭包同一套规则)。"""
    out = []
    for c in cores:
        name = c if c == "vega_1" else f"{prefix}_{c}" if prefix else c
        out += by_link.get(name, [])
    return out


def _core(frame_name: str) -> str:
    return frame_name[7:] if frame_name.startswith("vega_1_") else frame_name


class JointGuard:
    def __init__(self, control_hz: float, vel_scale: float = 0.4,
                 margin_m: float = 0.015, collision: bool = True,
                 urdf: str | pathlib.Path = DEFAULT_URDF):
        self.hz = float(control_hz)
        self.vel_scale = float(vel_scale)
        self.margin = float(margin_m)
        self.step_max = {}          # joint name -> max |dq| per control step
        if vel_scale > 0.0:
            for s in ("L", "R"):
                for j, vmax in zip(ARM_JOINTS[s], ARM_VMAX):
                    self.step_max[j] = vel_scale * vmax / self.hz

        self.model = self.geom = None
        self.pairs = []
        if collision:
            self._build_collision(pathlib.Path(urdf).expanduser())

        self.prev: dict[str, float] | None = None   # last SAFE command sent
        self.stat = {"clamped_steps": 0, "collision_holds": 0, "checked": 0}

    # -- collision model ------------------------------------------------------
    def _build_collision(self, urdf: pathlib.Path):
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.geom = pin.buildGeomFromUrdf(
            self.model, str(urdf), pin.GeometryType.COLLISION,
            package_dirs=[str(urdf.parent)])
        by_link: dict[str, list[int]] = {}
        for i, g in enumerate(self.geom.geometryObjects):
            by_link.setdefault(_core(self.model.frames[g.parentFrame].name),
                               []).append(i)

        def geoms(prefix, cores):
            out = []
            for c in cores:
                name = c if c == "vega_1" else f"{prefix}_{c}" if prefix else c
                out += by_link.get(name, [])
            return out

        left_x = geoms("L", _CROSS_ARM)
        right_x = geoms("R", _CROSS_ARM)
        body = geoms("", _BODY)
        seen = set()
        for a in left_x:
            for b in right_x:
                seen.add((min(a, b), max(a, b)))
        for side in ("L", "R"):
            for a in geoms(side, _DISTAL):
                for b in body:
                    seen.add((min(a, b), max(a, b)))
        self.geom.removeAllCollisionPairs()
        for a, b in sorted(seen):
            self.geom.addCollisionPair(pin.CollisionPair(a, b))
        self._add_env(by_link)
        self.gdata = self.geom.createData()
        # margin lives in the boolean query itself (hppfcl security_margin):
        # 0.026 ms/check. A true distance query over these mesh pairs costs
        # 250 ms - usable offline for diagnosis, three orders too slow at 50 Hz.
        for req in self.gdata.collisionRequests:
            req.security_margin = self.margin
        self.pairs = list(self.geom.collisionPairs)
        self._qidx = {}
        for jid in range(1, self.model.njoints):
            self._qidx[self.model.names[jid]] = self.model.joints[jid].idx_q

    def _add_env(self, by_link: dict[str, list[int]]):
        """把桌面和墙面当固定几何体加进碰撞模型。

        此前只检查两条手臂互相穿插,而**机器人撞桌子完全没人管** —— 映射把手
        指到桌面以下时,仿真会照常解出来,真机会直接压上去。参考实现(T-Rex
        `robot_descriptions.py:483` 的 add_env_obstacles)把桌子和三面墙当固定
        几何体加进同一个碰撞模型,这里是同一件事。

        尺寸从环境变量读,因为它是**场地相关**的:换一张桌子就得改,写死在
        代码里迟早对不上。桌高填实测值,单位米;设为 0 或负数即关闭该障碍。
        """
        import os
        self.env_names: list[str] = []
        try:
            import hppfcl as fcl
        except ImportError:                                # noqa: BLE001
            try:
                import pinocchio.hppfcl as fcl             # 老版本 pinocchio
            except ImportError:
                print("[guard] 没有 hppfcl,环境障碍物跳过")
                return

        table_h = float(os.environ.get("VEGA_TABLE_HEIGHT", "0") or 0)
        back = float(os.environ.get("VEGA_BACK_WALL", "0") or 0)
        left = float(os.environ.get("VEGA_LEFT_WALL", "0") or 0)
        right = float(os.environ.get("VEGA_RIGHT_WALL", "0") or 0)
        if not any(v > 0 for v in (table_h, back, left, right)):
            return

        # 只和**前臂与手腕**配对(_DISTAL = arm_l5..l8),不和上臂、躯干、基座
        # 配对。两个理由:①要拦的是「把手指进桌子里」,不是上臂;②默认姿势下
        # 上臂 arm_l3 在 z=0.97,比手低 34cm,拿它当判据会让桌子刚放到手能到的
        # 高度时就恒为真(实测过,就是这么发现的)。
        robot_geoms = sorted(set(
            [i for s in ("L", "R") for i in geoms_of(by_link, s, _DISTAL)]))
        if not robot_geoms:
            print("[guard] 找不到前臂几何体,环境障碍物跳过")
            return

        # 桌子在机器人**前方**。参考实现放在 x=+1.1,那是他们坐标系的前方;
        # 这台机器人的手臂朝 -x 伸(实测:胸在 x=-0.08,手在 x=-0.45),所以
        # 默认偏置取 -1.1。照抄 +1.1 的话桌子永远碰不到,而且不会有任何提示 ——
        # 「抄了数值没抄坐标系」正是这一类改动最容易静默失效的地方。
        table_x = float(os.environ.get("VEGA_TABLE_X", "-1.1"))
        specs = []
        if table_h > 0:
            specs.append(("table", fcl.Box(2.0, 4.0, 0.08),
                          np.array([table_x, 0.0, table_h - 0.04])))
        # 墙按距离放,方向由符号决定;不确定朝向时先别开,它们不像桌子那样
        # 有一个朝向无关的表达。
        if back > 0:
            specs.append(("back_wall", fcl.Box(0.02, 5.0, 3.0),
                          np.array([back, 0.0, 1.5])))
        if left > 0:
            specs.append(("left_wall", fcl.Box(5.0, 0.02, 3.0),
                          np.array([0.0, left, 1.5])))
        if right > 0:
            specs.append(("right_wall", fcl.Box(5.0, 0.02, 3.0),
                          np.array([0.0, -right, 1.5])))

        for name, shape, xyz in specs:
            # GeometryObject 的重载在不同 pinocchio 版本里顺序不同:.venv-isaac
            # 上是 (名字, 父frame, 父joint, 几何, 位姿),.venv 上是
            # (名字, 父joint, 位姿, 几何)。只按其中一种写,换个环境就 ArgumentError
            # ——单测就是这么抓到的。两种都试,都不行才报错。
            pose = pin.SE3(np.eye(3), xyz)
            obj = None
            for args in ((name, 0, 0, shape, pose), (name, 0, pose, shape)):
                try:
                    obj = pin.GeometryObject(*args)
                    break
                except Exception:                          # noqa: BLE001
                    continue
            if obj is None:
                print(f"[guard] 这个 pinocchio 版本建不出障碍物几何体,环境障碍物跳过")
                self.env_names = []
                return
            gid = self.geom.addGeometryObject(obj)
            self.env_names.append(name)
            for r in robot_geoms:
                self.geom.addCollisionPair(pin.CollisionPair(r, gid))
        print(f"[guard] 环境障碍物:{'、'.join(self.env_names)}"
              + (f"(桌高 {table_h:.2f}m)" if table_h > 0 else ""))

    def env_self_check(self, home: dict[str, float]) -> None:
        """默认姿势下就撞上障碍物,说明尺寸填错了 —— 直接报错,不要带着一个
        永远为真的碰撞判据跑起来(那会让手臂从第一帧就被冻住,而且不说为什么)。
        参考实现在同一位置也做这件事。"""
        if not getattr(self, "env_names", None):
            return
        if self.in_collision(home):
            raise SystemExit(
                f"[guard] 默认姿势下就与环境障碍物相撞({'、'.join(self.env_names)})。"
                "检查 VEGA_TABLE_HEIGHT 等尺寸是不是填错了 —— 桌高填的是实测的"
                "桌面高度,单位米,机器人基座为原点。")

    def _q_of(self, cmd: dict[str, float]):
        q = pin.neutral(self.model)
        for n, v in cmd.items():
            i = self._qidx.get(n)
            if i is not None:
                q[i] = v
        return q

    def in_collision(self, cmd: dict[str, float]) -> bool:
        """Boolean check with the margin baked into the query. 0.026 ms."""
        pin.computeCollisions(self.model, self.data, self.geom, self.gdata,
                              self._q_of(cmd), True)
        for i in range(len(self.pairs)):
            if self.gdata.collisionResults[i].isCollision():
                pr = self.geom.collisionPairs[i]
                a = self.geom.geometryObjects[pr.first].name
                b = self.geom.geometryObjects[pr.second].name
                key = f"{a}~{b}"
                self.stat.setdefault("hold_pairs", {})
                self.stat["hold_pairs"][key] = \
                    self.stat["hold_pairs"].get(key, 0) + 1
                return True
        return False

    def _min_distance(self, cmd: dict[str, float]) -> float:
        """True distance query - OFFLINE DIAGNOSIS ONLY (~250 ms)."""
        pin.computeDistances(self.model, self.data, self.geom, self.gdata,
                             self._q_of(cmd))
        return min((r.min_distance for r in self.gdata.distanceResults),
                   default=float("inf"))

    # -- the guard ------------------------------------------------------------
    def apply(self, cmd: dict[str, float]) -> dict[str, float]:
        """Guard one control step's final command. Returns the command to send.

        Velocity clamp touches ARM joints only (head/torso have their own
        ramps). The collision check sees the FULL commanded pose, torso and
        head included; on a hit the arms hold at the previous safe command.
        """
        if self.prev is None:
            self.prev = dict(cmd)
            return cmd

        out = dict(cmd)
        clamped = False
        for j, mx in self.step_max.items():
            if j in out and j in self.prev:
                d = out[j] - self.prev[j]
                if abs(d) > mx:
                    out[j] = self.prev[j] + (mx if d > 0 else -mx)
                    clamped = True
        self.stat["clamped_steps"] += clamped

        if self.geom is not None:
            self.stat["checked"] += 1
            if self.in_collision(out):
                self.stat["collision_holds"] += 1
                for s in ("L", "R"):
                    for j in ARM_JOINTS[s]:
                        if j in out and j in self.prev:
                            out[j] = self.prev[j]
                return out          # prev stays: out's arms == prev's arms

        self.prev = dict(out)
        return out

    def reset(self):
        self.prev = None

    def summary(self) -> str:
        s = self.stat
        pairs = s.get("hold_pairs", {})
        top = "; ".join(f"{k}x{v}" for k, v in
                        sorted(pairs.items(), key=lambda kv: -kv[1])[:3])
        return (f"[guard] velocity clamp on {s['clamped_steps']} steps; "
                f"self-collision holds {s['collision_holds']} "
                f"of {s['checked']} checked" + (f"  [{top}]" if top else ""))


# ---------------------------------------------------------------------------
def _self_check() -> int:
    """Known-answer checks. Every one of these can fail on a real bug."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from magicdexmate.head_waist_map import _LEGACY_TORSO
    fails = []

    g = JointGuard(control_hz=50.0, vel_scale=0.4, margin_m=0.015)
    print(f"[self-check] 碰撞对 {len(g.pairs)} 对")

    home = {**{j: v for j, v in zip(ARM_JOINTS["L"], [-0.5, 0.30, 0, -2.2, -0.4, 0, 0])},
            **{j: v for j, v in zip(ARM_JOINTS["R"], [0.5, -0.30, 0, -2.2, 0.4, 0, 0])},
            **{k: float(v) for k, v in _LEGACY_TORSO.items()},
            "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0}

    # 1) the working pose must be clean, with real clearance
    d = g._min_distance(home)         # slow true-distance query, offline only
    print(f"[self-check] 工作位最近距离 {d*1000:.0f}mm")
    if d < 0.05:
        fails.append(f"工作位距离仅 {d*1000:.0f}mm,余量不足,误拦会常态化")

    # 2) arms adducted across the midline (j2) must trigger; j1 swings do not
    #    cross at this home - measured, not assumed (j2 by 1.0 rad = contact)
    cross = dict(home)
    cross["L_arm_j2"] = home["L_arm_j2"] - 1.0
    cross["R_arm_j2"] = home["R_arm_j2"] + 1.0
    if not g.in_collision(cross):
        fails.append("构造的交叉臂姿势没有触发(检测是摆设)")
    else:
        print("[self-check] 交叉臂(j2 内收 1.0 rad)已触发")
    if g.in_collision(home):
        fails.append("工作位被误报碰撞(margin 太大或对表有误)")
    # margin must actually matter: near-contact pose triggers only with margin
    near = dict(home)
    near["L_arm_j2"] = home["L_arm_j2"] - 0.90
    near["R_arm_j2"] = home["R_arm_j2"] + 0.90
    gm0 = JointGuard(50.0, vel_scale=0.0, margin_m=0.0)
    gm5 = JointGuard(50.0, vel_scale=0.0, margin_m=0.05)
    if gm0.in_collision(near) or not gm5.in_collision(near):
        d_near = g._min_distance(near)
        fails.append(f"margin 未按预期生效(近接触位实测距离 {d_near*1000:.0f}mm)")
    else:
        print("[self-check] margin 生效: 同一姿势 margin=0 不触发、margin=50mm 触发")

    # 3) velocity clamp: exact known answer
    g2 = JointGuard(50.0, vel_scale=0.4, collision=False)
    g2.apply(home)
    step = dict(home); step["R_arm_j1"] = home["R_arm_j1"] + 1.0
    out = g2.apply(step)
    want = home["R_arm_j1"] + 0.4 * 2.4 / 50.0
    if abs(out["R_arm_j1"] - want) > 1e-9:
        fails.append(f"限幅算错: {out['R_arm_j1']:.6f} != {want:.6f}")
    else:
        print(f"[self-check] 限幅精确: 1 rad 阶跃被限为 {np.degrees(out['R_arm_j1']-home['R_arm_j1']):.3f}°/步")

    # 4) hold-on-collision: feed a colliding command, arms must NOT move
    g3 = JointGuard(50.0, vel_scale=0.0, margin_m=0.015)
    g3.apply(home)
    held = g3.apply(cross)
    moved = max(abs(held[j] - home[j]) for s in ("L", "R") for j in ARM_JOINTS[s])
    if moved > 1e-12:
        fails.append(f"碰撞时手臂仍移动了 {np.degrees(moved):.2f}°(回退没生效)")
    else:
        print(f"[self-check] 碰撞回退生效: 手臂保持,holds={g3.stat['collision_holds']}")

    # 5) timing
    import time
    t0 = time.perf_counter()
    for _ in range(500):
        g.in_collision(home)
    print(f"[self-check] 布尔检测 {1000*(time.perf_counter()-t0)/500:.3f} ms/次")

    if fails:
        print("[self-check] FAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("[self-check] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
