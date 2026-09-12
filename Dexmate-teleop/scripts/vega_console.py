#!/usr/bin/env python3
"""操作台:打开这一个脚本,其余全部在浏览器里操作。

    cd <仓库根目录> && .venv/bin/python scripts/vega_console.py

浏览器打开 http://localhost:8086。勾选要控制的部位,点「启动」,页面里
就是机器人。连接真机、录制数据、停止,都是页面上的按钮。

本脚本只做两件事:托管各个子进程(它们仍是原来那些程序,一行未改),
以及把 viser_isaac_mirror 的显示嵌进同一个页面。显示逻辑只有那一份。

进程之间只有广播、没有等待:任何一个子进程死掉,其余照常,页面上会显示。
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time

import viser

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from viser_isaac_mirror import Mirror  # noqa: E402  (same directory)

LOGDIR = ROOT / "logs" / "console"
STATE_FILE = ROOT / "logs" / "console_state.json"
# 统一的数据根目录。每次录制一个子目录,六路数据全部写进去 —— 此前手臂写
# logs/episodes/、手写 data/teleop/,两套编号各自增长,已经错开了一位。
DATA_ROOT = ROOT / "data" / "sessions"
PY = {"pico": ROOT / ".venv-pico/bin/python",
      "isaac": ROOT / ".venv-isaac/bin/python",
      "venv": ROOT / ".venv/bin/python"}
DEXCONTROL_PY = pathlib.Path("~/Dexmate/dexcontrol/.venv/bin/python").expanduser()
PC_SERVICE = pathlib.Path("/opt/apps/roboticsservice/runService.sh")
ROBOT_NAME = "dm/vgd1262ab823-1p"
# 机器人用 mDNS 广播自己(服务类型写死在 dexcomm 里),实例名是 ROBOT_NAME
# 去掉 dm/ 前缀。查广播比连一次 dexcontrol 便宜得多,也顺带解决了 IP 会变的
# 问题 —— 广播里带的才是它当前的地址。
ROBOT_HINT = ROBOT_NAME.split("/")[-1]
WIRED_IF = "enp3s0"                        # 机器人与 Sharpa 手都挂在这个网口上
SHARPA_IP = {"left": "192.168.10.10", "right": "192.168.10.20"}

# 手臂栈的已签收配置(KNOWN-GOOD,来源见 workflow_fj.md 续二十一)
CONSUMER_ARGS = ("--source zmq --mode trackers --tracker-left LWRIST "
                 "--tracker-right RWRIST --tracker-waist WAIST "
                 "--map chest-anchor --ori-mode palm --waist-mode j3 "
                 "--pos-scale 1.0 --physics-device cpu --physics-hz 50 "
                 "--control_hz 50 --no-sym-lock --headless")


class Proc:
    """一个被托管的子进程:日志落盘,状态可查,停止走 SIGINT。"""

    def __init__(self, name: str, cmd: str, env: dict | None = None):
        LOGDIR.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.log = LOGDIR / f"{name}.log"
        self._fh = open(self.log, "ab", buffering=0)
        self._fh.write(f"\n===== {time.strftime('%F %T')} {cmd}\n".encode())
        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        self.p = subprocess.Popen(
            cmd, shell=True, cwd=ROOT, env=full_env,
            stdout=self._fh, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid)       # own group: SIGINT reaches children

    def alive(self) -> bool:
        return self.p.poll() is None

    def stop(self, timeout: float = 8.0):
        if self.alive():
            try:
                os.killpg(self.p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            t0 = time.time()
            while self.alive() and time.time() - t0 < timeout:
                time.sleep(0.2)
            if self.alive():
                try:
                    os.killpg(self.p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        self._fh.close()

    def tail(self, pattern: str = "", n: int = 40) -> str:
        try:
            lines = self.log.read_text(errors="replace").splitlines()[-n:]
        except OSError:
            return ""
        if pattern:
            lines = [ln for ln in lines if pattern in ln]
        return lines[-1].strip() if lines else ""


class Console:
    def __init__(self):
        self.procs: dict[str, Proc] = {}
        self.stop_pending = 0.0     # 截止时刻;非零 = 已广播「正常退出」
        self.recording = False
        self.episode = self._next_episode()
        self.session = ""                   # 本次录制的子目录名,空 = 没在录
        self.session_auto = True            # 名字是自动生成的(停止时要补结束时间)
        self.rec_t0 = 0.0
        # 遥操作分段。参考实现每个回合开始都把机器人送回默认姿势、同时重置
        # 求解器构型,坏姿态因此活不过一段。这个计数器一变,仿真侧和桥接侧
        # 各自去做自己那一半。
        self.epoch = 0
        self.dev: dict = {}                 # 设备探测结果,由后台线程写
        self._probe_stop = False
        st = {}
        if STATE_FILE.exists():
            try:
                st = json.loads(STATE_FILE.read_text())
            except Exception:                              # noqa: BLE001
                st = {}

        self.server = viser.ViserServer(port=8086)
        g = self.server.gui

        with g.add_folder("控制哪些部位"):
            self.c_left = g.add_checkbox("左臂", st.get("left", True))
            self.c_right = g.add_checkbox("右臂", st.get("right", True))
            self.c_head = g.add_checkbox("控制头部", st.get("head", True))
            self.c_waist = g.add_checkbox("控制腰部", st.get("waist", False))
            # 手的左右独立勾选(2026-08-10 用户要求)。旧配置只有一个 "hand"
            # 键 —— 迁移:老值同时作为两侧的默认,行为不变。
            _hand_old = st.get("hand", False)
            self.c_hand_l = g.add_checkbox("控制左手",
                                           st.get("hand_l", _hand_old))
            self.c_hand_r = g.add_checkbox("控制右手",
                                           st.get("hand_r", _hand_old))
            g.add_markdown("头部需先勾手臂,腰部需先勾头部;"
                           "左手/右手(Sharpa)独立勾选,与手臂无关。"
                           "更改后需重新启动。")

        with g.add_folder("设备"):
            self.g_dev_arm = g.add_text("Dexmate", "检测中 …")
            self.g_dev_hand = g.add_text("Sharpa", "检测中 …")
            self.g_dev_cam = g.add_text("相机", "检测中 …")
            # 相机独立开关(2026-08-11 用户要求):看画面不必启动整套仿真栈。
            # 打开后点云出现在 3D 场景机器人旁,「相机画面」行显示帧数。
            self.b_cam_on = g.add_button("打开相机画面")
            self.b_cam_off = g.add_button("关闭相机画面")

        with g.add_folder("启动"):
            # 场地相关的尺寸放在界面上,不写死在脚本里 —— 换一张桌子就得改。
            # 0 = 不启用桌面防撞。
            self.n_table = g.add_number("桌面高度(米,0=不用)",
                                        float(st.get("table", 0.0)),
                                        min=0.0, max=2.0, step=0.01)
            # 腕部朝向映射。两种各有代价,只能由戴头显的人判,所以放在页面上。
            self.d_wrist = g.add_dropdown(
                "腕部朝向", ("绝对(掌心与你一致)", "增量(不易超量程)"),
                initial_value=st.get("wrist", "绝对(掌心与你一致)"))
            self.c_fake = g.add_checkbox("模拟数据(无硬件)", False)
            self.b_start = g.add_button("启动")
            self.b_stop = g.add_button("停止", color="red")
            self.g_stack = g.add_text("进程", "未启动")

        # 真机按钮由 Mirror 放进它自己的「真机」折叠组(同一个页面区块)
        def real_buttons(gui):
            self.b_dex = gui.add_button("连接手臂")
            self.b_dex_off = gui.add_button("断开手臂")
            self.b_sharpa = gui.add_button("连接真手")
            self.b_sharpa_off = gui.add_button("断开真手")

        self.mirror = Mirror(
            self.server,
            hand_right="tcp://127.0.0.1:5556",
            hand_left="tcp://127.0.0.1:5557",
            # session 是本次录制的子目录名。手那边据此把自己的数据写进同一个
            # 目录 —— 此前两边各自编号,已经错开了一位,而广播里那个 episode
            # 字段发了三天没有任何人订阅。
            extra_state=lambda: {"recording": self.recording,
                                 "session": self.session,
                                 "epoch": self.epoch,
                                 "episode": self.episode,
                                 "shutdown": bool(self.stop_pending)},
            real_buttons=real_buttons)

        with g.add_folder("录制"):
            self.b_epoch = g.add_button("开始新一段")
            self.g_epoch = g.add_text("段落", "尚未开始")
            self.t_name = g.add_text("本次名称", "")
            self.b_rec = g.add_button("开始录制")
            self.g_rec = g.add_text("录制状态", "未在录制")
            g.add_markdown("**开始新一段**:机器人先慢速回到默认姿势,"
                           "求解器同时归位,然后从那里重新跟随。"
                           "每次开录之前点一下。\n\n"
                           "名称留空则按录制起止时间自动命名。全部数据写进 "
                           "`data/sessions/` 下的同一个子目录。"
                           "手部录制的第一秒在做触觉标零,指尖需悬空。")

        self.b_all_stop = g.add_button("全部停止", color="red")

        # -- wiring -----------------------------------------------------------
        for c in (self.c_hand_l, self.c_hand_r, self.c_left, self.c_right,
                  self.c_head, self.c_waist):
            c.on_update(lambda _: self._on_check())
        self._apply_rules()
        self.b_start.on_click(lambda _: self.start_stack())
        self.b_stop.on_click(lambda _: self.stop_stack())
        self.b_cam_on.on_click(lambda _: self.open_camera())
        self.b_cam_off.on_click(lambda _: self._stop("camera"))
        self.b_dex.on_click(lambda _: self.connect_dexmate())
        self.b_dex_off.on_click(lambda _: self._stop("bridge"))
        self.b_sharpa.on_click(lambda _: self.connect_sharpa())
        self.b_sharpa_off.on_click(lambda _: self._stop("sharpa"))
        self.n_table.on_update(lambda _: self._save())
        self.d_wrist.on_update(lambda _: self._save())
        self.b_epoch.on_click(lambda _: self.new_epoch())
        self.b_rec.on_click(lambda _: self.toggle_record())
        self.b_all_stop.on_click(lambda _: self.stop_all())

    # -- helpers --------------------------------------------------------------
    def _save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({
            "hand_l": self.c_hand_l.value, "hand_r": self.c_hand_r.value,
            "left": self.c_left.value,
            "right": self.c_right.value, "head": self.c_head.value,
            "waist": self.c_waist.value,
            "table": float(self.n_table.value),
            "wrist": self.d_wrist.value}))

    def _on_check(self):
        self._apply_rules()
        self._save()

    def _apply_rules(self):
        """勾选规则:手臂 -> 头部 -> 腰部 逐级依赖;Sharpa 手完全独立。

        依赖用「置灰」表达而不是「自动补勾」。自动补勾会替操作者做他没做的
        决定 —— 此前勾腰会连带把手也勾上,而手和腰之间并没有任何关系。
        """
        if getattr(self, "_in_rules", False):
            return                       # 下面写 .value 会再次触发回调
        self._in_rules = True
        try:
            arms = self.c_left.value or self.c_right.value
            if not arms:
                self.c_head.value = False
            self.c_head.disabled = not arms
            if not self.c_head.value:
                self.c_waist.value = False
            self.c_waist.disabled = not self.c_head.value
        finally:
            self._in_rules = False

    # -- 设备探测 --------------------------------------------------------------
    # 三样都不占用设备:Dexmate 查它自己的 mDNS 广播,Sharpa 走网络 ping(它是
    # 网口设备,不在 lsusb 里),相机只数个数不 open —— open 会独占 USB,面板一
    # 旦持有,采集进程就再也拿不到相机了。
    # 前两样各要一秒,所以整套放在后台线程里,页面只读结果。
    def _probe_loop(self):
        while not self._probe_stop:
            try:
                self.dev = self._probe_once()
            except Exception as e:                         # noqa: BLE001
                self.dev = {"err": repr(e)}
            self._manage_observer()
            time.sleep(3.0)

    def _manage_observer(self):
        """真机姿态只读回显(scripts/dexmate_observer.py)的生命周期。

        检测到机器人广播就自动启动,页面一打开 Viser 里就是真机的实际姿势
        (2026-08-10 现场需求);桥接在跑时不启动 —— 桥接自己读状态,不需要
        第二个 zenoh 客户端;广播消失就停掉。「全部停止」会把它一起杀掉,
        下一轮探测(3 秒后)发现机器人还在会自动拉起来,不需要人管。
        """
        bridge = self.procs.get("bridge")
        obs = self.procs.get("observer")
        want = bool((self.dev or {}).get("vega")) \
            and not (self.dev or {}).get("vega_no_route") \
            and not (bridge and bridge.alive())
        if want and not (obs and obs.alive()):
            print("[console] 检测到机器人,启动只读姿态回显")
            self._start("observer",
                        f"{DEXCONTROL_PY} -u scripts/dexmate_observer.py",
                        env={"ROBOT_NAME": ROBOT_NAME})
        elif not want and obs and obs.alive():
            print("[console] 停止只读姿态回显"
                  + ("(桥接接管)" if bridge and bridge.alive() else "(机器人离线)"))
            self._stop("observer")

    def _probe_once(self) -> dict:
        d: dict = {}
        # 有线口没插时直接短路:两台网络设备都不可能在
        try:
            link = (pathlib.Path("/sys/class/net") / WIRED_IF / "carrier").read_text()
            d["wired"] = link.strip() == "1"
        except OSError:
            d["wired"] = False

        d["vega"] = []
        if d["wired"]:
            try:
                r = subprocess.run(["avahi-browse", "-prt", "_dexmate-zenoh._tcp"],
                                   capture_output=True, text=True, timeout=4)
                for ln in r.stdout.splitlines():
                    f = ln.split(";")
                    if f and f[0] == "=" and len(f) >= 9 and ROBOT_HINT in f[3]:
                        d["vega"].append(f"{f[7]}:{f[8]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        # mDNS 是组播、ping 靠路由,**能收到广播/网口通不代表能连**:机器人在
        # 192.168.50.x、Sharpa 手在 192.168.10.x,本机网口重启/拔插后会丢掉
        # 这些网段的第二地址 —— 2026-08-10 真机现场两个都栽了,面板只说
        # 「已检测到」/「ping 不通」,分不出是设备没上电还是本机没网段。
        # 这里比对 /24 前缀,缺网段就写明和给出那条命令。
        host24: set = set()
        try:
            r = subprocess.run(["ip", "-4", "-o", "addr", "show"],
                               capture_output=True, text=True, timeout=2)
            host24 = {m.rsplit(".", 1)[0] for m in
                      re.findall(r"inet (\d+\.\d+\.\d+\.\d+)/", r.stdout)}
        except Exception:                                  # noqa: BLE001
            pass
        d["vega_no_route"] = False
        if d["vega"] and host24:
            bot24 = {a.split(":")[0].rsplit(".", 1)[0] for a in d["vega"]}
            d["vega_no_route"] = not (host24 & bot24)
        hand24 = {ip.rsplit(".", 1)[0] for ip in SHARPA_IP.values()}
        d["hand_no_route"] = bool(host24) and not (host24 & hand24)

        def _ping(ip):
            return subprocess.run(["ping", "-c1", "-W", "1", ip],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL).returncode == 0

        # 两只手并行 ping。不通时每个要等满一秒,串行就是两秒。
        if d["wired"]:
            with cf.ThreadPoolExecutor(2) as ex:
                d["hand"] = dict(zip(SHARPA_IP,
                                     ex.map(_ping, SHARPA_IP.values())))
        else:
            d["hand"] = {s: False for s in SHARPA_IP}

        d["cam"], d["cam_sn"] = 0, []
        try:
            import pyk4a
            d["cam"] = int(pyk4a.connected_device_count())
            # 序列号必须 open 才拿得到,而 open = 独占。只在没有任何进程用相机
            # 时取一次,取完立刻关掉。
            if d["cam"] and not self.procs.get("camera"):
                for i in range(d["cam"]):
                    k = pyk4a.PyK4A(device_id=i)
                    try:
                        k.open()
                        d["cam_sn"].append(k.serial)
                    except Exception:                      # noqa: BLE001
                        d["cam_sn"].append("序列号读取失败")
                    finally:
                        try:
                            k.close()
                        except Exception:                  # noqa: BLE001
                            pass
        except Exception:                                  # noqa: BLE001
            d["cam"] = -1                                  # pyk4a 装坏了
        return d

    def _dev_text(self) -> tuple[str, str, str]:
        d = self.dev
        if not d:
            return ("检测中 …",) * 3
        conn = self.procs.get("bridge")
        arm = ("已检测到 " + "、".join(d.get("vega") or [])) if d.get("vega") else (
            "未检测到" + ("" if d.get("wired") else "(有线网口未连接)"))
        if d.get("vega") and d.get("vega_no_route"):
            ip24 = (d["vega"][0].split(":")[0].rsplit(".", 1)[0])
            arm += (f",但本机不在 {ip24}.x 网段,连不上 —— 运行 "
                    f"sudo ip addr add {ip24}.100/24 dev {WIRED_IF}")
        elif d.get("vega"):
            arm += ",已连接" if conn and conn.alive() else ",未连接"

        s = self.procs.get("sharpa")
        found = [k for k, v in (d.get("hand") or {}).items() if v]
        hand = ("已检测到 " + "、".join("左手" if k == "left" else "右手"
                                        for k in found)) if found else (
            "未检测到" + ("" if d.get("wired") else "(有线网口未连接)"))
        if not found and d.get("wired") and d.get("hand_no_route"):
            ip24 = next(iter(SHARPA_IP.values())).rsplit(".", 1)[0]
            hand = (f"未检测到:本机不在 {ip24}.x 网段 —— 运行 "
                    f"sudo ip addr add {ip24}.240/24 dev {WIRED_IF}")
        if found:
            hand += ",已连接" if s and s.alive() else ",未连接"

        n = d.get("cam", 0)
        if n < 0:
            cam = "pyk4a 不可用"
        elif n == 0:
            cam = "检测到 0 个"
        else:
            cam = f"检测到 {n} 个:" + "、".join(d.get("cam_sn") or ["序列号未读"])
            cam += "(尚未校准,暂不连接)"
        return arm, hand, cam

    def _next_episode(self) -> int:
        ep = 0
        d = ROOT / "logs" / "episodes"
        for f in d.glob("episode_*_arm.msgpack") if d.exists() else []:
            try:
                ep = max(ep, int(f.name.split("_")[1]) + 1)
            except ValueError:
                pass
        return ep

    def _start(self, name: str, cmd: str, env: dict | None = None):
        self._stop(name)
        self.procs[name] = Proc(name, cmd, env)

    def _stop(self, name: str):
        p = self.procs.pop(name, None)
        if p:
            p.stop()

    def _arm_sides(self) -> str:
        if self.c_left.value and self.c_right.value:
            return "both"
        return "left" if self.c_left.value else "right"

    def _hand_sides(self) -> str | None:
        """手的左右由自己的勾选决定,与手臂无关(2026-08-10 之前跟着手臂
        勾选走,真机现场「只勾右臂 + 控制手」导致左手整条链没起,人在
        排查手套)。None = 一只都没勾。"""
        left, right = self.c_hand_l.value, self.c_hand_r.value
        if left and right:
            return "both"
        if left or right:
            return "left" if left else "right"
        return None

    # -- the stack ------------------------------------------------------------
    def _reap_strays(self):
        """清理不归本控制台管的游离仿真/采集进程。

        2026-08-10 实测的事故形态:控制台 Ctrl-C 的收尾要等仿真进程正常退出
        (约 25 秒),中断它或直接关终端,消费端就成了孤儿继续满载烧 CPU;
        新起的消费端发现状态端口被占**并不报死**,于是多个消费端并存,整机
        被拖到 0.78 倍实时(当晚同时活着 3 个)。「启动」前先清理:凡是
        teleop_vega_pico / teleop_pico_producer 进程而不在本控制台的进程表里
        的,一律 SIGINT 并等待,清不掉就 SIGTERM。"""
        mine = {p.p.pid for p in self.procs.values() if p.alive()}
        out = subprocess.run(
            ["pgrep", "-f", "teleop_vega_pico|teleop_pico_producer"],
            capture_output=True, text=True).stdout
        strays = [int(x) for x in out.split() if x.isdigit()
                  and int(x) not in mine and int(x) != os.getpid()]
        if not strays:
            return
        print(f"[console] 发现 {len(strays)} 个游离的仿真/采集进程(上次未收干净),"
              f"先行清理:{strays}")
        for pid in strays:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        t0 = time.time()
        while time.time() - t0 < 20.0:
            alive = [p for p in strays
                     if subprocess.run(["kill", "-0", str(p)],
                                       capture_output=True).returncode == 0]
            if not alive:
                break
            time.sleep(0.5)
        for pid in strays:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def start_stack(self):
        fake = self.c_fake.value
        arms = self.c_left.value or self.c_right.value
        if not arms and self._hand_sides() is None:
            self.g_stack.value = "没有勾选任何部位"
            return
        self._reap_strays()

        if arms:
            if PC_SERVICE.exists() and not fake and \
                    subprocess.run("pgrep -f roboticsservice", shell=True,
                                   capture_output=True).returncode != 0:
                self._start("service", f"bash {PC_SERVICE}")
            rec = ROOT / "logs" / f"pico_{time.strftime('%Y%m%d_%H%M%S')}.msgpack"
            self._start("producer",
                        f"{PY['pico']} scripts/teleop_pico_producer.py "
                        f"--body-full --record {rec} "
                        f"--control tcp://127.0.0.1:5584"
                        + (" --fake" if fake else ""))
            mode = ("arms+head+waist" if self.c_waist.value else
                    "arms+head" if self.c_head.value else "arms")
            # 默认开两样防护:
            # ① 自碰撞 hold —— 抓到过真问题(纯网格互穿 44 次),代价约 1mm,
            #    实测对机器负载不敏感;
            # ② 腕部限位放松 —— 2026-08-09 曾因「负载下 81.25%」撤回过一次;
            #    2026-08-10 查清那不是负载问题,是触发条件读全部 7 关节造成的
            #    自锁(肩顶限位也维持放松 -> 手臂游荡 -> 肩继续顶死),已改为
            #    只看 j6/j7 + 2° 地板 + 饱和时目标插值。修后空载 7.00%、4 核
            #    负载 7.23%/6.92%(旧码同负载 81.25%),腕关节从顶限位名单
            #    里消失,腕误差与无防护基线持平(17.7/12.8mm)。机理回归见
            #    sim/dev_relax_latch.py(在 check_all 里)。
            # 腕部限位放松的默认史:08-09 开(空载数据)→ 当天撤(误判负载);
            # 08-10 修好自锁后再开 → 08-11 用户实时实测「感觉不如之前」再撤。
            # 嫌疑:实时快动作的朝向滞后常态性超过它的保持阈值,放松粘住不放,
            # 掌心跟随发钝 —— 离线回放的指标看不出这种手感。教训第三次重复:
            # 影响手感的默认值只能由穿戴实测拍板。要试新机制:VEGA_RELAX=on。
            guard = " --self-collision hold"
            if os.environ.get("VEGA_RELAX", "").strip() == "on":
                guard += " --relax-at-limit 0.01 --relax-margin 3.0"
                print("[console] 腕部限位放松:开(VEGA_RELAX=on)")
            # IK 后端切换:环境变量 VEGA_IK=curobo 时试用 curobo 后端(须代码里
            # 已有该选项),默认不设 = pink,行为与从前逐位相同。做成环境变量
            # 而不是页面控件:这是试验期开关,页面只留操作者日常要用的东西。
            _ik = os.environ.get("VEGA_IK", "").strip()
            if _ik:
                guard += f" --ik {_ik}"
                print(f"[console] IK 后端:{_ik}(VEGA_IK 环境变量指定)")
            # 一个下拉顶两个开关:增量朝向必须配默认姿势锚点,配错实测 96.98%
            guard += (" --wrist-map incremental" if "增量" in self.d_wrist.value
                      else " --wrist-map absolute")
            self._start("consumer",
                        f"{PY['isaac']} -u sim/teleop_vega_pico.py "
                        f"{CONSUMER_ARGS}{guard} --control-mode {mode} "
                        f"--hands {self._arm_sides()}",
                        env={"OMNI_KIT_ACCEPT_EULA": "YES",
                             "PYTHONUNBUFFERED": "1",
                             "VEGA_WELD_WHEELS": "1",
                             "VEGA_TORSO_HOME": "legacy",
                             # 桌面防撞。填 0 就不加这个障碍物;consumer 启动时
                             # 会用默认姿势自检一次,尺寸填错会当场报错而不是
                             # 让手臂莫名其妙冻住。
                             "VEGA_TABLE_HEIGHT": f"{self.n_table.value:.3f}"})
            self.mirror.g_mode.value = mode

        if self._hand_sides() is not None:
            src = "mock" if fake else "wuji"
            glove = (f"PYTHONPATH= {PY['venv']} scripts/teleop_retarget.py "
                     f"--source {src} --pinch-weight 20 --relax-distal "
                     f"--control tcp://127.0.0.1:5584")
            if self.c_hand_r.value:
                self._start("glove_r", f"{glove} --hand right")
            if self.c_hand_l.value:
                self._start("glove_l", f"{glove} --hand left --pub 'tcp://*:5557'")

        # 相机:检测到就起点云录制。它自己订开关频道,录制时才写盘;没检测到
        # 就不起 —— 相机不是必需的(还没标外参),不能因为它缺席就挡住别的。
        if not fake and (self.dev.get("cam") or 0) > 0:
            self.open_camera()

    def stop_stack(self):
        # 仿真进程必须走正常退出(广播 shutdown,它自己收尾)。用信号杀
        # 长跑的 Isaac 会把 nvidia_uvm 搞坏 —— 2026-08-06 在本脚本第一版上
        # 复现过一次(CUDA 直接 False)。其余进程 SIGINT 即可。
        c = self.procs.get("consumer")
        if c is not None and c.alive():
            self.stop_pending = time.time() + 25.0
        else:
            self._finish_stop()

    def _finish_stop(self):
        self.stop_pending = 0.0
        # ⚠ 不停 "service"(roboticsservice):它是**头显主动连接的对象**,随
        # 控制台停止一起杀掉的话,头显那边直接 "TCP connection failed",而且
        # 操作者看不出和控制台有什么关系 —— 2026-08-10 真机现场就是这么栽的。
        # 它像系统服务一样常驻即可;控制台启动时 pgrep 查到就不会重复起。
        for n in ("consumer", "producer", "glove_r", "glove_l", "camera"):
            self._stop(n)

    def open_camera(self):
        """独立打开相机画面(不启动仿真栈)。与 start_stack 用同一个进程名,
        栈随后启动时不会重复开;「全部停止」会把它一并停掉。"""
        p = self.procs.get("camera")
        if p and p.alive():
            self.mirror.g_cam.value = "已经开着(见帧数)"
            return
        if self.dev and (self.dev.get("cam") or 0) <= 0:
            why = "pyk4a 不可用" if self.dev.get("cam", 0) < 0 else "未检测到相机"
            self.mirror.g_cam.value = f"{why},没有启动"
            print(f"[console] {why},不启动相机")
            return
        self._start("camera",
                    f"{PY['venv']} -u scripts/kinect_pointcloud.py "
                    f"--record-session tcp://127.0.0.1:5584 "
                    f"--pub-view 'tcp://*:5591'")
        self.mirror.g_cam.value = "启动中 …(首帧要几秒)"

    # -- real hardware --------------------------------------------------------
    def connect_dexmate(self):
        comps = [c for c, on in (("left_arm", self.c_left.value),
                                 ("right_arm", self.c_right.value),
                                 ("head", self.c_head.value),
                                 ("torso", self.c_waist.value)) if on]
        if not comps:
            self.mirror.g_bridge.value = "没有勾选任何部位"
            return
        # 检测不到就别起一个注定失败的进程。探测查的是机器人自己的 mDNS 广播,
        # 也就是 dexcontrol 启动时查的同一个东西 —— 这里看不见,它也连不上,
        # 只是会多花三秒再抛异常,而操作者只看到进程「已退出」。
        if self.dev and not self.dev.get("vega"):
            why = ("有线网口未连接" if not self.dev.get("wired")
                   else "网口通,但没有收到机器人的广播")
            self.mirror.g_bridge.value = f"未检测到机器人({why}),没有启动"
            print(f"[console] 未检测到机器人({why}),不启动桥接")
            return
        if self.dev and self.dev.get("vega_no_route"):
            ip24 = self.dev["vega"][0].split(":")[0].rsplit(".", 1)[0]
            msg = (f"本机不在 {ip24}.x 网段,连不上 —— 先运行 "
                   f"sudo ip addr add {ip24}.100/24 dev {WIRED_IF}")
            self.mirror.g_bridge.value = msg
            print(f"[console] {msg}")
            return
        # 桥接自己读状态。先停回显,别留两个 zenoh 客户端订同一批话题。
        self._stop("observer")
        self._start("bridge",
                    f"{DEXCONTROL_PY} -u scripts/dexmate_bridge.py --live "
                    f"--components {','.join(comps)}",
                    env={"ROBOT_NAME": ROBOT_NAME})

    def connect_sharpa(self):
        sides = self._hand_sides()
        if sides is None:
            self.mirror.g_sharpa.value = "没有勾选左手或右手"
            return
        # 同 connect_dexmate:探测查的是手自己的网络地址,这里 ping 不通,
        # SDK 也发现不了它,只是会多等 1.5 秒再报「设备未找到」。
        # 按**勾选的那几只**判,而不是「任何一只」:此前勾了左手、只有右手
        # 在线也会放行,然后 SDK 在左手上报错。逐侧列出谁不在线。
        if self.dev and (self.dev.get("hand") is not None):
            ping = self.dev["hand"]
            want = ["left", "right"] if sides == "both" else [sides]
            miss = [s for s in want if not ping.get(s)]
            if miss:
                names = "、".join("左手" if s == "left" else "右手" for s in miss)
                why = ("有线网口未连接" if not self.dev.get("wired")
                       else (f"本机不在手的网段,运行 setup_robot_net.sh"
                             if self.dev.get("hand_no_route")
                             else f"{names} ping 不通(未上电?)"))
                self.mirror.g_sharpa.value = f"未检测到 {names}({why}),没有启动"
                print(f"[console] 未检测到 {names}({why}),不启动")
                return
        extra = "--sub-left tcp://127.0.0.1:5557" if sides == "both" else ""
        if sides == "left":
            sides_args = "--hand left --sub tcp://127.0.0.1:5557"
        else:
            sides_args = f"--hand {sides} {extra}"
        self._start("sharpa",
                    f"PYTHONPATH= {PY['venv']} -u scripts/sharpa_real_runner.py "
                    f"{sides_args} --control tcp://127.0.0.1:5584 --tactile-delta "
                    f"--record {DATA_ROOT.relative_to(ROOT)}",
                    env={"LD_LIBRARY_PATH": "/opt/sharpa-wave-sdk/lib"})

    # -- recording ------------------------------------------------------------
    @staticmethod
    def _safe_name(s: str) -> str:
        """用户起的名字要能当目录名。空白转下划线,去掉路径分隔符与控制字符。"""
        s = "_".join(s.split())
        return "".join(ch for ch in s if ch.isprintable()
                       and ch not in '/\\:*?"<>|')[:60]

    def new_epoch(self):
        """开始新的一段:广播段号,仿真侧与桥接侧各自归位。

        故意不在这里等它们做完 —— 两条管线之间只许有广播、不许有等待(这条
        约束是踩过坑之后定的)。做没做完从各自的读数上看。
        """
        self.epoch += 1
        self.g_epoch.value = f"第 {self.epoch} 段(机器人正在回默认姿势)"
        print(f"[console] 开始第 {self.epoch} 段")

    def toggle_record(self):
        if not self.recording:
            self.rec_t0 = time.time()
            given = self._safe_name(self.t_name.value)
            # 留空就按起止时间命名。开始的时候还不知道结束时间,所以先用开始
            # 时间建目录,停止时再补上结束时间改名 —— 名字里带两个时间是用户
            # 要的,而目录必须在开录之前就存在。
            self.session = given or time.strftime("%Y%m%d_%H%M%S",
                                                  time.localtime(self.rec_t0))
            self.session_auto = not given
            d = DATA_ROOT / self.session
            d.mkdir(parents=True, exist_ok=True)
            self._start("rec_arm",
                        f"{PY['venv']} -u scripts/replay_check.py "
                        f"--record {d / 'arm.msgpack'}")
            self.recording = True
            self.b_rec.label = "停止录制"
        else:
            self._stop("rec_arm")
            self.recording = False
            self.b_rec.label = "开始录制"
            self._finish_session()

    def _finish_session(self):
        """自动命名的目录在这里补上结束时间。"""
        if not (self.session and self.session_auto):
            self.session = ""
            return
        d = DATA_ROOT / self.session
        end = time.strftime("%H%M%S")
        new = d.with_name(f"{self.session}__{end}")
        try:
            # 手那边收到 recording=False 之后才收尾,广播的过期窗口是 1.5 秒,
            # 所以先等一下再改名,免得它还在往里写。
            time.sleep(1.8)
            d.rename(new)
            print(f"[console] 本次录制 -> {new}")
        except OSError as e:
            # 改名失败不算错:数据都在,只是名字里少了结束时间。
            print(f"[console] 目录改名失败({e}),数据仍在 {d}")
        self.session = ""

    def stop_all(self):
        self.mirror.g_enable.value = False
        self.mirror.g_hand_en.value = False
        if self.recording:
            self.toggle_record()
        # 真机与录制进程用 SIGINT 没有问题;仿真进程必须走正常退出,
        # 与 stop_stack 相同的原因(nvidia_uvm)。
        for n in ("bridge", "sharpa", "rec_arm", "observer"):
            self._stop(n)
        self.stop_stack()

    def _drain_shutdown(self):
        """同步等仿真进程正常退出(Ctrl-C 收尾时用;期间继续广播)。"""
        c = self.procs.get("consumer")
        if c is None or not c.alive():
            self._finish_stop()
            return
        self.stop_pending = time.time() + 25.0
        while self.stop_pending:
            self.mirror.tick()          # 广播 shutdown 就在 tick 里
            if not c.alive() or time.time() > self.stop_pending:
                self._finish_stop()
                break
            time.sleep(0.05)

    # -- status refresh -------------------------------------------------------
    def refresh(self):
        if self.stop_pending:
            c = self.procs.get("consumer")
            if c is None or not c.alive() or time.time() > self.stop_pending:
                self._finish_stop()
        rows = []
        for name, label in (("producer", "PICO"), ("consumer", "仿真"),
                            ("glove_r", "右手套"), ("glove_l", "左手套"),
                            ("camera", "相机")):
            p = self.procs.get(name)
            if p:
                rows.append(f"{label}{'运行中' if p.alive() else '已退出'}")
        self.g_stack.value = ",".join(rows) if rows else "未启动"

        (self.g_dev_arm.value, self.g_dev_hand.value,
         self.g_dev_cam.value) = self._dev_text()

        b = self.procs.get("bridge")
        if b is None:
            self.mirror.g_bridge.value = "未连接"
        elif not b.alive():
            self.mirror.g_bridge.value = "已退出,详见 logs/console/bridge.log"
        else:
            last = b.tail("[bridge]")
            self.mirror.g_bridge.value = (last.split("[bridge]")[-1].strip()
                                          or "连接中 …")[:60]

        s = self.procs.get("sharpa")
        if s is None:
            self.mirror.g_sharpa.value = "未连接"
        elif not s.alive():
            self.mirror.g_sharpa.value = "已退出,详见 logs/console/sharpa.log"
        else:
            last = s.tail("connected") or s.tail("[runner]") or s.tail("[sharpa")
            self.mirror.g_sharpa.value = (last or "连接中 …")[-60:]

        if self.recording:
            p = self.procs.get("rec_arm")
            n = p.tail("[录制]") if p and p.alive() else ""
            self.g_rec.value = f"episode {self.episode:04d} " + \
                (n.split("]")[-1].strip() if n else "录制中")
        else:
            self.g_rec.value = "未在录制"

    def run(self):
        print("[console] open http://localhost:8086")
        threading.Thread(target=self._probe_loop, daemon=True).start()
        last_refresh = 0.0
        try:
            while True:
                self.mirror.tick()
                if time.time() - last_refresh > 0.5:
                    self.refresh()
                    last_refresh = time.time()
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            pass
        finally:
            print("\n[console] 停止全部子进程 …")
            self.stop_all()
            self._drain_shutdown()
            # viser 的服务线程不是 daemon,不主动收就会让进程挂着不退
            try:
                self.server.stop()
            except Exception:                              # noqa: BLE001
                pass
            os._exit(0)


if __name__ == "__main__":
    Console().run()
