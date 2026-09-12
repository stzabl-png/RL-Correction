# SOP：Dexmate Vega 连接 / 手臂摆位 / Sharpa 手遥操 + 数据录制

> 适用对象：第一次上手的新人。照着从上往下做即可。
> 本机环境：`luhr-Legion-Y9000P-IAH7H`（实验室笔记本），机器人：Dexmate Vega-1P（`dm/vgd1262ab823-1p`），手：Sharpa Wave（左右），手套：Wuji。
> 2026-08-15 起迁到实验室台式机 `msc-Alienware-Aurora-R13`（本文档后续以它为准）：**有线网口 `enp3s0`**（Realtek r8169，笔记本原为 `enp49s0`），Wi-Fi `wlp4s0`；机器人/手/手套仍同上。
> 最后更新：2026-07-27（全流程实测通过：抬臂 → 遥操 → 触觉录制 → viser 回放）。
>
> 完整流程 = **第一部分**（连机器人、键盘抬臂摆位）→ **第二部分**（手套遥操 Sharpa 手 + 录制触觉/轨迹数据）。只做其中一半也可以独立照做。

---

# 第零部分：新机器部署（git clone 之后还差什么）

仓库里已含**全部自研代码** + retarget 内核 `third_party/dex-retargeting`（含我们的修改，完整源码）+ Sharpa 手 URDF 资产。
在一台新电脑上让整套跑起来，还需要装以下**不入 git** 的外部依赖：

1. **Python 环境 ×2**（都在仓库根目录建）：
   - `.venv`（producer/runner/回放用，Python 3.11）：按 `pyproject.toml` 装依赖
     （本机 `.venv` 没有 pip，装包用 `uv pip install --python .venv/bin/python <pkg>`）。
   - `.venv-isaac`（只有跑 sim 才需要）：按 Isaac Sim/Lab 官方文档安装，体量几十 GB。
2. **Sharpa Wave SDK**（真机手 + 触觉必需）：厂商**闭源**发布，找 Sharpa/实验室拷贝安装到
   `/opt/sharpa-wave-sdk`（本机版本 5.0.3；含 SharpaPilot 标定工具）。**不要把它传到公开 GitHub（侵权）**。
   运行真机 runner 时需 `LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib`。
3. **wuji_sdk**（Wuji 手套 SDK）：闭源 pip wheel（本机版本 2026.6.18），装进 `.venv`。同样不入 git。
4. **dexcontrol**（只在需要控制 Vega 手臂时）：Dexmate 官方库（开源），装在 `~/Dexmate/dexcontrol/`，**用它自己的 venv**
   `~/Dexmate/dexcontrol/.venv/bin/python`（`vega_console.py` 与本文所有手臂命令都指向这个路径）。台式机上的做法（2026-08-15）：
   ```bash
   mkdir -p ~/Dexmate && ln -s <仓库根目录>/dexcontrol ~/Dexmate/dexcontrol && ln -s <仓库根目录>/dexmate-urdf ~/Dexmate/dexmate-urdf
   uv venv --python 3.11 ~/Dexmate/dexcontrol/.venv
   uv pip install --python ~/Dexmate/dexcontrol/.venv/bin/python -e ~/Dexmate/dexcontrol tyro msgpack pyzmq
   ```
   （`tyro` 给 keyboard 示例用；`msgpack pyzmq` 给 `scripts/dexmate_bridge.py` / `dexmate_observer.py` 用——它们由控制台用这个 venv 启动。）
   **每台机器人的 zenoh 接入配置 `~/.dexmate/comm/zenoh/<robot>.dzcfg` 是凭证，随机器人发放，绝不能提交到 git**。
5. **硬件网络**：IP 规划见第一部分 §1.2 与第二部分 §5.1（网口加地址每次重启需重做，或固化进 NetworkManager）。

---

# 第一部分：连接机器人 + 键盘控制手臂摆位

## 0. 安全须知（先读这一节）

- **e-stop（急停）放在手边**，脚本运行期间机器人会真的动。
- 确认手臂**运动范围内没有人、没有障碍物**（包括桌面、线缆、Sharpa 手）。
- 第一次操作用默认步长（2°）即可，不要上来就加大 `--step-size`。
- 出现任何异常：按 `q` 退出脚本，或直接拍 e-stop。

---

## 1. 前置条件检查

### 1.1 硬件

- 机器人已上电，e-stop 已释放（旋开）。
- 电脑与机器人之间**有线网线已接好**（本台式机板载网口 `enp3s0`；实验室笔记本为 `enp49s0`）。

### 1.2 网络（最常出问题的一步）

机器人本体的通信服务（Zenoh）在 **`192.168.50.20:7447`**，笔记本必须在 `192.168.50.x` 网段才能连上。

检查：

```bash
ip -br addr show enp3s0
```

输出里应包含 `192.168.50.100/24`（可能同时还有 `192.168.10.240`、`192.168.5.100`，那是 Sharpa 手等其它设备的网段，不冲突、别删）。

**如果没有** `192.168.50.x` 地址（比如刚重启过电脑/重新插过网线），补上：

```bash
sudo ip addr add 192.168.50.100/24 dev enp3s0
```

验证连通：

```bash
ping -c 2 192.168.50.20        # 应通，延迟 <1ms
```

> 注：`sudo ip addr add` 是临时的，**电脑重启或拔插网线后会失效**，到时重新执行一遍即可。

---

## 2. 启动键盘关节控制

一条命令（可整段复制）：

```bash
ROBOT_NAME=dm/vgd1262ab823-1p ~/Dexmate/dexcontrol/.venv/bin/python \
    ~/Dexmate/dexcontrol/examples/advanced_examples/keyboard_joint_control.py \
    --component right_arm
```

要点：

- `ROBOT_NAME=dm/vgd1262ab823-1p` 这个环境变量**必须带**（告诉 SDK 连哪台机器人），少了会直接报错。
- `--component` 可选：`right_arm` / `left_arm` / `torso` / `head`。手臂各 7 个关节（编号 0–6），torso/head 各 3 个（0–2）。
- 可选参数：`--step-size 2.0`（每条指令的度数，默认 2°，手臂安全上限 10°）、`--control-rate 50`（指令频率 Hz）。
- 必须用上面这个 venv 里的 python（`~/Dexmate/dexcontrol/.venv/bin/python`），系统 python 缺依赖跑不了。

### 2.1 按键操作

| 按键 | 作用 |
|---|---|
| `0`–`6` | 选择要控制的关节（按编号） |
| 按住 `w` | 选中关节向 **正** 方向连续运动，**松开即停** |
| 按住 `s` | 选中关节向 **负** 方向连续运动，**松开即停** |
| `q` | 正常退出 |
| `Ctrl+C` | 紧急退出 |

屏幕上会实时显示当前关节编号和角度（度）。摆好位置后按 `q` 退出即可，手臂会保持在当前位置。

---

## 3. 常见问题与解决方法

### 3.1 报错 `ValueError: Variant not specified and neither ROBOT_CONFIG nor ROBOT_NAME environment variables are set`

**原因**：没带 `ROBOT_NAME` 环境变量。
**解决**：命令前加 `ROBOT_NAME=dm/vgd1262ab823-1p`（见 §2），或一次性导出：

```bash
export ROBOT_NAME=dm/vgd1262ab823-1p
```

### 3.2 报错 `Failed to open Zenoh session ... Unable to connect to any of [tls/192.168.50.20:7447, ...]`

（或卡在 `Loading Zenoh config from env var: ...` 很久后报这个错）

**原因**：笔记本到机器人的网络不通。按顺序排查：

1. **最常见：缺 192.168.50.x 地址**（重启/拔插网线后丢了）→ 按 §1.2 重新 `sudo ip addr add 192.168.50.100/24 dev enp3s0`（或直接 `sudo bash scripts/setup_robot_net.sh` 一次补齐三个网段）。
2. 网线没插好 / 机器人没上电 → `ping 192.168.50.20` 不通时，检查物理连接和电源。
3. 想确认机器人到底在不在线，可用 mDNS 搜一下（能看到它宣告的 IP 和端口）：
   ```bash
   timeout 6 avahi-browse -art | grep -B1 -A6 dexmate-zenoh
   ```
   正常应看到 `vgd1262ab823-1p-standalone`，`address = [192.168.50.20]`，`port = [7447]`。
4. 端口级确认：
   ```bash
   timeout 3 bash -c 'echo > /dev/tcp/192.168.50.20/7447' && echo OK || echo FAIL
   ```

### 3.3 报错 `ModuleNotFoundError: No module named 'tyro'`（或 `dexcontrol`）

**原因**：用错了 python（系统 python 或别的 venv）。
**解决**：用完整路径 `~/Dexmate/dexcontrol/.venv/bin/python` 启动（见 §2）。

### 3.4 脚本启动了，但按 `w`/`s` 机器人不动

按顺序检查：

1. e-stop 是否按下了？释放（旋开）后再试。
2. 是否先按数字键选中了关节？看屏幕提示行确认当前关节编号。
3. 终端窗口是否在前台获得焦点？按键是发给终端的，点一下终端窗口再按。
4. 机器人组件未激活/超时报错 → 机器人可能刚上电还没就绪，等十几秒重跑；仍不行就重启机器人。

### 3.5 只动了一小下就停 / 运动断断续续

这是**按住才动、松手即停**的设计（靠键盘重复触发，约 150ms 无按键事件即自动停）。持续运动请**按住**别松；某些远程终端/输入法可能干扰按键重复，换本机终端试。

### 3.6 想控制别的部位

换 `--component`：`left_arm`（左臂）、`torso`（腰）、`head`（头）。head 会自动进入 enable 模式。每个部位的单步安全上限不同（臂 10°、腰 5°、头 10°），脚本会自动限幅。

---

## 3.5+ 保存 / 一键恢复手臂位姿（免每次手动摆）

摆好的臂位姿可以存名字，下次开机一条命令回位。脚本 `goto_arm_pose.py` 和存档 `arm_poses.json` 都在**本仓库根目录**
（用 dexcontrol 的 venv 运行，不是仓库的 `.venv`）。
**已存位姿**：`right_arm_teleop` = 遥操+录制用右臂位姿（2026-07-27 存，deg ≈ [-48.4, -21.1, -11.3, -91.9, -51.7, -4.7, 17.0]）。

```bash
cd <仓库根目录>

# 列出已存位姿（不连机器人）
~/Dexmate/dexcontrol/.venv/bin/python goto_arm_pose.py --list

# 一键回位（会先打印当前/目标角度，回车确认后才动；默认低速 scale 0.2，Ctrl+C 可中断）
ROBOT_NAME=dm/vgd1262ab823-1p ~/Dexmate/dexcontrol/.venv/bin/python goto_arm_pose.py right_arm_teleop

# 把当前位置存成新位姿（只读、不会动机器人）
ROBOT_NAME=dm/vgd1262ab823-1p ~/Dexmate/dexcontrol/.venv/bin/python goto_arm_pose.py --save 新位姿名 --component right_arm --note "备注"
```

运动由机器人端 motion plugin 执行（自带轨迹平滑/重力补偿/碰撞规避）；回位前同样确认范围无障碍、e-stop 在手边。

---

---

# 第二部分：Sharpa 手遥操 + 数据录制

## 5. 前置条件检查（遥操/录制）

### 5.1 硬件与网络

| 设备 | 地址 | 接在哪 |
|---|---|---|
| Sharpa 左手 | `192.168.10.10` | 板载网口 `enp3s0`（`192.168.10.240`） |
| Sharpa 右手 | `192.168.10.20` | 同上 |
| Wuji 左手套 | `192.168.1.100` | USB 转网口 `enx*`（`192.168.1.10`） |
| Wuji 右手套 | `192.168.1.101` | 同上 |

一键自检（全部应显示"在线"）：

```bash
for ip in 192.168.10.10 192.168.10.20 192.168.1.100 192.168.1.101; do
  ping -c1 -W1 $ip >/dev/null && echo "$ip 在线" || echo "$ip 不通"
done
```

### 5.2 规则与注意

- **手套侧必须和手同侧**（右手套驱动右手），且下面两个终端的 `--hand` 必须一致。
- **录制期间不要开 SharpaPilot**（会抢触觉端口，§P.4 踩过坑）。
- 手套需已标定（标定用 SharpaPilot/手套配套工具；拇指不跟手多半是手套标定问题，可用
  `PYTHONPATH= .venv/bin/python scripts/diag_glove.py --hand left` 自检——过程中 wiggle 拇指和四指，
  末尾拇指 range≈0 而四指几十 mm 即为该手套拇指没标好）。e
- 真机安全：runner 启动后默认 **DISENGAGED（不动）**，按 `e` 才开始跟随；手边随时可按 `w` 冻结、`x` 停止。

## 6. 启动遥操（两个终端）

以**右手**为例；左手把两条命令里的 `right` 都换成 `left`。两个终端都先：

```bash
cd <仓库根目录>
```

### 终端 1 — producer（手套 → retarget → ZMQ 发布）

```bash
PYTHONPATH= .venv/bin/python scripts/teleop_retarget.py --source wuji --hand right --pinch-weight 20 --relax-distal
```

（`--pinch-weight 20 --relax-distal` 是定稿参数，必带；`--thumb-ip-margin` 默认 0.16 不用写。）

### 终端 2 — 真机 runner（ZMQ → 真手）+ 录制

```bash
LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib PYTHONPATH= .venv/bin/python scripts/sharpa_real_runner.py \
    --hand right --sub tcp://127.0.0.1:5556 --record data/teleop
```

不需要录制、只遥操时去掉 `--record data/teleop` 即可。
可选：`--tactile f6,deform,raw|none`（默认全录）、`--tactile-hz 30`、`--rate`（控制频率，默认 20Hz）。

### 6.1 runner 按键

| 键 | 作用 |
|---|---|
| `e` | engage — 手开始跟随手套（从当前位姿平滑跟上；重复按无害） |
| `r` | 开始 / 结束一个 episode 录制（编号自增） |
| `w` | freeze（冻结在当前位置） |
| `q` | 回 home 位 + freeze |
| `x` | 停止并退出 |

## 7. 录制一个 episode 的标准流程

1. 两个终端起好，runner 状态行有数据在刷新（此时还是 DISENGAGED）。
2. 按 `e` engage，确认真手正常跟随手套。
3. 按 `r` 开始 episode。**首个 episode 开头约 1 秒在做触觉标零：指尖必须悬空、不接触任何东西**；标零期间手暂停原位属正常。
4. 录制中盯状态行：`rec:ON` + 每指 `max|F|` —— **指尖按压物体时数值应跳动**；一直为 0 = 触觉没出数，停下排查（见 §9）。
5. 再按 `r` 结束。文件在 `data/teleop/episode_0000/episode_0000_right.h5`（编号自增）。
6. 磁盘预算：全录（f6+deform+raw）约 **12–20MB/s/手**，长时间录制先确认剩余空间；只录 f6 几乎不占空间。

## 8. 回放检查（每录一段就看一次）

```bash
cd <仓库根目录>
PYTHONPATH= .venv/bin/python scripts/replay_hand_viser.py data/teleop/episode_0000/episode_0000_right.h5
# 浏览器打开 http://localhost:8080
```

看三样：① 关节回放动作对不对；② 指尖按压时**对应指**的指尖套（elastomer）变色 + 力矢量箭头出现；③ DEFORM 五图面板有图案。
界面上可切换 measured/target 位形源、调拖尾长度、播放倍速/循环。

> 首次在一只新手上录制时，建议专门录一段"**单指逐个按压**"（拇指→食指→中指→无名指→小指），回放确认高亮指头与实际按压一一对应（验证触觉通道映射；左右手映射已于 2026-07-27 在右手真机验证正确）。

## 9. 常见问题与解决方法（遥操/录制）

### 9.1 runner 连不上手 / 手不动

1. `ping 192.168.10.20`（右）/`192.168.10.10`（左）不通 → 手没上电或网线问题（§5.1 自检）。
2. 连上了但手不动 → 是否按了 `e`？默认 DISENGAGED 不动。
3. producer 终端是否在跑、`--hand` 两边是否一致？
4. 报"设备被占用"类错误 → 关掉 SharpaPilot 和其它占用 SDK 的进程后重试。

### 9.2 启动时打印 `Firmware version mismatch`（如 firmware 3.0.4 vs SDK 5.0.x）

**只是警告，不阻塞**。手固件比 SDK 适配表新（2026-07-27 右手 fw 3.0.4 实测：连接、时间同步、触觉、录制全部正常）。遇到怪行为时再想起这茬即可。

### 9.3 `Collision detected ... (code 48)` 或手指互相卡住

自碰撞保护默认**关**（`collision_protection=False`，手完全照手套走）。别用手套摆出手指互挤的姿势即可。
若开了保护（构造参数传 True），固件检测到碰撞会自动调整角度并返回 code 48——代码已容忍该码，不会崩。

### 9.4 触觉 max|F| 一直是 0 / 录出来全是零

1. 标零（calib）时指尖是否碰到了东西？→ 结束 episode，指尖悬空重新开一个 episode（会重新标零）。
2. SharpaPilot 是否在后台占着触觉端口？→ 关掉后重启 runner。
3. runner 启动日志里找 `Touch started: result=0` —— 没有此行说明触觉流没起来，重启 runner。

### 9.5 手套数据不动 / 某根手指不跟

- 手套 ping 通但数据僵死 → 重新插拔手套 USB 网卡（`enx*` 网口，`192.168.1.10`）。
- 单指（尤其拇指）不跟 → 手套标定问题，用 `diag_glove.py` 自检（§5.2），重标手套。**左手套拇指历史上标定不好，用左手前先自检。**

### 9.6 episode 开始时手停了约 1 秒

正常：那是触觉标零，POSITION 模式下固件保持原位。标零完自动恢复跟随。

---

## 10. 附：关键路径/参数速查

| 项 | 值 |
|---|---|
| 手臂键盘控制脚本 | `~/Dexmate/dexcontrol/examples/advanced_examples/keyboard_joint_control.py` |
| 手臂控制 Python 环境 | `~/Dexmate/dexcontrol/.venv/bin/python` |
| 机器人名（环境变量） | `ROBOT_NAME=dm/vgd1262ab823-1p` |
| Zenoh 配置（自动加载） | `~/.dexmate/comm/zenoh/dm_vgd1262ab823-1p.dzcfg` |
| 机器人本体 IP（有线） | `192.168.50.20`（Zenoh 端口 7447）；SoC `192.168.50.21` |
| 本机有线口 | `enp3s0`（台式机；笔记本 `enp49s0`），需有 `192.168.50.100/24`（+ `192.168.10.240/24` 给 Sharpa 手） |
| ssh 上机器人（排查用） | `ssh dexmate`（= `dexmate@192.168.50.20`） |
| 遥操/录制仓库 | `<仓库根目录>`（分支 `fj-retarget-rework`） |
| 遥操 Python 环境 | 仓库内 `.venv`（命令前必须加 `PYTHONPATH=` 清空） |
| Sharpa SDK | `/opt/sharpa-wave-sdk`（runner 需 `LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib`） |
| Sharpa 手网段（同网口，勿删） | `192.168.10.240/24`（左手 `.10`、右手 `.20`） |
| Wuji 手套网段（USB 网卡） | `192.168.1.10/24`（左套 `.100`、右套 `.101`） |
| 录制输出 | `data/teleop/episode_XXXX/episode_XXXX_{left,right}.h5` |
| 回放 | `scripts/replay_hand_viser.py <h5>` → 浏览器 `http://localhost:8080` |
| 手臂位姿存档 + 回位脚本 | 仓库根 `arm_poses.json` + `goto_arm_pose.py`（已存 `right_arm_teleop`；用 dexcontrol venv 运行） |
| 项目详细工作记录 | `/home/luhr/dexmate/fj_work_claude.md`（在实验室笔记本 `luhr-Legion` 上，本台式机没有；原理、调参历史、踩坑全录） |
