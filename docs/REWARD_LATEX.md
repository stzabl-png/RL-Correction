# AAG-F 训练奖励函数(LaTeX)

**口径**:`tasks/pregrasp` 的 `--pregrasp_only --phase2 --aag --fcd --fin_cart --fc_ref_end` 配方,
即跑出**确定性成功率 100.00% (n=1024)** 的 AAG-F 线。权重取自 `cfg.py` 默认值 + F 线命令行覆盖。
双臂:两侧各自独立计算,总回报为两侧之和。

---

## 0. 总式

$$
R_t \;=\; \alpha_t \sum_{k \in \mathcal{K}} w_k\, r^{(k)}_t ,
\qquad
\alpha_t = \mathbb{1}[\text{回合活跃}]
$$

$\mathcal{K}$ = 该配方下生效的 26 项(见下)。`approach_only` 会**移除**一组抓取/搬运段的遗留项:

$$
\mathcal{K} = \mathcal{K}_{\text{all}} \setminus \{\texttt{fail, pad\_approach, pad\_touch, cent\_income,}\\
\texttt{quality\_prog, hold, over\_force, obj\_move, obj\_rot, tilt, push, finger\_cross, carry}\}
$$

**记号**

| 符号 | 含义 |
|---|---|
| $d_p,\ d_r$ | 腕到 GraspPose 的位置差(m)/朝向差(rad) |
| $\phi$ | 接近进度 $\in[0,1]$ |
| $\mathbb{1}_{app}$ | 处于 PREGRASP 相位 |
| $A$ | 到位闩 `arrived` |
| $\mathbf{d}^{fc}\in\mathbb{R}^5$ | 五指尖到各自 FC 目标点的距离(m) |
| $n_{pad}$ | 当前接触指垫数 |
| $\mathbf{q}^{f}$ | 22 维手指关节角 |
| $\mathbf{q}^{f}_{ref}(t)$ | 编舞参考第 $t$ 行的手指位形 |

---

## 1. 接近段(逐步密集)

### 1.1 对齐势差分 `align`

$$
\varphi_t = -\big(d_p + \lambda_{rot}\, d_r\big),\qquad \lambda_{rot}=0.174
$$
$$
r^{align}_t = \mathbb{1}_{app}\cdot \mathrm{clip}\big(\varphi_t-\varphi_{t-1},\,-c,\,c\big),
\qquad w=6.0,\ c=0.02
$$

> 用**势差分**而非绝对距离:绝对量会让"早退晚进"无限刷分。

### 1.2 模仿罚 `imit`

$$
w^{imit}_t = w_0\cdot \rho\cdot\big(1-\phi\big)_+^{p},\qquad w_0=6.0,\ p=2
$$
$$
r^{imit}_t = -\,w^{imit}_t\cdot \mathbb{1}_{app}\cdot
\mathrm{clip}\Big(\big\|(\mathbf{x}^w_t-\mathbf{x}^w_{t-1}) - \Delta\mathbf{x}^{ref}_t\big\|,\ \le 0.02\Big)
$$

$\rho=$ `w_imit_ramp`(训练期系数,由 `arrive_rate` 驱动;**F 线全程 $\rho=0$**,故此项实际未生效)。
罚的是**本步残差用量**(实际走的 − 参考走的),与人手轨迹的常数平移无关。

### 1.3 一次性到位奖 `arrive` / 撞击罚 `hit`

$$
r^{arrive}_t = \mathbb{1}[\text{首次 } A],\quad w=8.0 \qquad
r^{hit}_t = \mathbb{1}[\text{臂撞物}],\quad w=-10.0
$$

---

## 2. 手指编舞跟踪

### 2.1 参考跟踪罚 `fin_track`

$$
r^{track}_t = -\,\eta\cdot\kappa\cdot\underbrace{\Big(1-\tfrac{n_{pad}}{n_{act}}\Big)_{[0,1]}}_{\text{接触淡出}}
\cdot \big\|\mathbf{q}^{f}_t-\mathbf{q}^{f}_{ref}(t)\big\|_1/22,\qquad w=0.05
$$

$\kappa$ = 近场放松系数($d_p<3\text{cm}$ 时放松);接触淡出:每多一个触垫,参考拉力降 $1/n_{act}$ ——
否则"抓住东西"本身会被罚。

### 2.2 形态正向势差分 `form_pot`(只赚不罚)

$$
r^{form}_t = \mathrm{clip}\Big(\big\|\mathbf{q}^{f}_{t-1}-\mathbf{q}^{f}_{ref}(t)\big\|_1
-\big\|\mathbf{q}^{f}_{t}-\mathbf{q}^{f}_{ref}(t)\big\|_1,\ 0,\ 0.02\Big)\cdot\mathbb{1}[\neg\text{cand}],
\quad w=2.0
$$

> **前后两步必须对同一行参考做差** —— 参考行自己在动,拿"距离标量"直接差分会把参考的移动
> 记成手指的倒退(实测左手 $-9.7$ 全是伪信号)。

### 2.3 构型糖 `shape_ms`

$$
r^{shape}_t=\mathbb{1}\big[\text{首次}: t\ge t_{settle}+t_{shape}\ \wedge\ \|\mathbf{q}^f_t-\mathbf{q}^f_{Pose1}\|_1/22<\epsilon_{shape}\big],\ w=1.0
$$

---

## 3. 指尖笛卡尔目标(FC)与 g2 里程碑

$$
\mathbf{d}^{fc}_i = \big\|\mathbf{p}^{tip}_i - \big(R(\mathbf{q}^{obj})\,\mathbf{c}^{local}_i+\mathbf{p}^{obj}\big)\big\|
$$

$\mathbf{c}^{local}$ = **逐行播一遍参考**后在物体系下拍到的五个接触点(2026-08-24 修;
瞬移拍照物理不可达,曾致 g2 结构性恒 0)。

### 3.1 面包屑 `fc_crumb`

$$
r^{crumb}_t=\sum_{i\in\mathcal{A}}\mathbb{1}\big[\text{首次}: A\wedge \mathbf{d}^{fc}_i<\tau\big],\quad \tau=1\text{cm},\ w=1.0
$$

### 3.2 势差分 `fc_pot`(只赚不罚)

$$
r^{pot}_t=\Lambda\cdot A\cdot\sum_{i\in\mathcal{A}}\mathrm{clip}\big(\mathbf{d}^{fc}_{i,t-1}-\mathbf{d}^{fc}_{i,t},\ 0,\ 0.02\big),\quad w=5.0
$$

> earn-only 是必须的:参考在 squeeze 段**故意压过** grasp 点,势差分天然为负,
> 于是"严格复现参考"反而挨罚(实测左手 $-0.046$/步)。

### 3.3 里程碑 `g2_m1` / `g2_m2`

$$
m_1:\ A\wedge d_p<\varepsilon_p\wedge d_r<\varepsilon_r
\qquad
m_2:\ \Big(\bigwedge_{i\in\mathcal{A}}\mathbf{d}^{fc}_i<\tau\Big)\wedge v_{obj}<0.05
$$
$$
r^{m}_t = B\cdot\mathbb{1}[\text{首次达成}],\qquad B=2.0,\ \varepsilon_p=1.35\text{cm},\ \varepsilon_r=15^\circ
$$

**成功判据**(双臂):
$$
\text{success}=\bigwedge_{s\in\{A,B\}}\Big(A^{(s)}\wedge \text{g2\_done}^{(s)}\Big),\qquad
\text{g2\_done}=\Big[\textstyle\sum_{k=0}^{H-1}\mathbb{1}[m_1\wedge m_2]_{t-k}=H\Big],\ H=5
$$

---

## 4. 接触与稳抓

$$
r^{pad\_first}_t=\sum_i \mathbb{1}[\text{首次}: A\wedge \text{垫}_i\text{触}],\quad w=0.5
$$
$$
r^{pad\_hold}_t=\frac{n_{pad}}{n_{act}}\cdot\mathbb{1}[A\wedge\neg\text{cand}]\cdot\mathbb{1}[\text{手静}],\quad w=0.02
$$
$$
r^{cent}_t=\big(\text{向心度}\big)_+\cdot\mathbb{1}[A\wedge\neg\text{cand}],\quad w=0.05
$$
$$
r^{succ\_hold}_t=\frac{n_{pad}}{n_{act}}\cdot\mathbb{1}[\text{本侧已成功}]\cdot\mathbb{1}[\text{手静}],\quad w=0.05
$$

---

## 5. 合拢前禁触 `pre_close`

窗口 $W=\{0<t<t_{close}\}$($t_{close}$ = close 段起始行,由 npz 段表现算):

$$
r^{pre}_t=-\,\mathbb{1}_W\Big[
n_{pad}
+100\,(\delta_{obj}-\delta_0)_+
+\tfrac{1}{5}(\theta_{tilt}-\theta_0)_+
+100\,(g_0-g_{shell})_+
\Big],\quad w=0.03
$$

$\delta_0=1\text{cm}$,$\theta_0=5^\circ$,$g_{shell}$ = 机器人外壳到物体表面最近距离。
量纲对齐:$5^\circ$ 超死区 $\equiv$ 1 个指垫接触 $\equiv$ 1cm 超死区。

---

## 6. 平滑与安全(逐步密集)

$$
r^{act}_t=-\|\mathbf{a}_t-\mathbf{a}_{t-1}\|^2/n,\ w=0.02
\qquad
r^{torque}_t=-\|\boldsymbol{\tau}\|^2/n,\ w=0.005
$$
$$
\tilde{\mathbf{q}}=\Big[\tfrac{\dot q^{arm}}{4.0},\ \tfrac{\dot q^{fin}}{14.0}\Big]:\quad
r^{qvel}_t=-\mathrm{clip}\big(\|\tilde{\mathbf{q}}\|^2/n,\le 10\big),\ w=0.01
$$
$$
r^{qvel\_hard}_t=-\,\nu_t\cdot\mathrm{clip}\Big(\textstyle\sum_j\big(|\dot q_j|-\bar q_j\big)_+^2\Big),\ w=0.1
$$
$\nu_t=\mathbb{1}[\text{不在复位冻结窗}]$ —— 复位传送产生的伪速度不该罚(`qvel_settle`)。

$$
r^{table}=-s^{table},\ w{=}20 \qquad r^{arm\_table}=-s^{arm}, w{=}20 \qquad r^{self}=-s^{self},\ w{=}5
$$
$$
r^{disturb}_t=-\mathrm{clip}(v_{obj},\le 2),\ w=0.05
\qquad
r^{topple}_t=-\mathbb{1}[\theta_{tilt}>60^\circ],\ w=2.0
$$
$$
r^{quiet}_t=-\,\|\mathbf{a}^{fin}_t\|\cdot(1-A)\cdot\kappa,\ w=0.02
\qquad
r^{time}_t=-1,\ w=0.002
$$

---

## 7. 里程碑总项 `milestone`

$$
r^{ms}_t = R_{cand}\,\mathbb{1}[\text{新 candidate}] + R_{succ}\,\mathbb{1}[\text{新 success}]
$$
接近段口径下另叠时间整形 $\big(1-\lambda+\lambda\,\gamma^{(T-T_{ref})_+}\big)$。

---

## 8. ★ 握力信任标量 $g$(2026-08-25 新增,用户裁定)

**不发任何握力奖励** —— 捏紧由参考自身 squeeze 段 + g2 目标点完成。
$g$ 的唯一职责:驱动前段密集项衰减。

$$
\text{slip}_t=\Big\|\underbrace{R(\mathbf{q}^{w})^{\!\top}(\mathbf{p}^{obj}_t-\mathbf{p}^{w}_t)}_{\text{腕系下物体位置}}-\mathbf{c}^{grip}\Big\|
$$

$\mathbf{c}^{grip}$ = 首次进 squeeze 段时各 env 自拍的基线。

$$
g_{t+1}=\mathrm{clip}\Big(g_t+
\begin{cases}
+\tfrac{1}{4}\delta, & t\in\text{squeeze}\ \wedge\ \text{slip}<0.5\text{cm} \quad(\text{弱证据})\\[2pt]
+\delta, & \text{交互段}\ \wedge\ \text{slip}<0.5\text{cm} \quad(\text{强证据})\\[2pt]
-10\,\delta, & \text{slip}>1\text{cm}
\end{cases}
,\ 0,\ 1\Big),\qquad \delta=\tfrac{1}{200}
$$

**C 方案衰减**(用户裁定:一个标量管两件事):
$$
w_k \leftarrow w_k\,(1-g_t),\qquad k\in\{\texttt{align},\ \texttt{imit}\}
$$

> 只衰减**逐步密集**项。one-shot 项(`arrive`/里程碑)不碰 —— 衰减它们等于改判据而非改塑形。
> 依据:F 线退化验尸中,任务饱和后仍在涨且与抓握质量对冲的正是这两类
> (`align` $+33\%$,而 `pad_hold` $-53\%$、`succ_hold` $-68\%$,确定性 eval $100\%\to56\%$)。

---

## 附:零动作基线(F 线实测,400 步均值)

| 正 | 值 | 负 | 值 |
|---|---|---|---|
| `milestone` | $+0.4805$ | `time` | $-0.0020$ |
| `arrive` | $+0.0400$ | `pre_close` | $-0.0009$ |
| `fc_crumb` | $+0.0240$ | `toppled_pen` | $-0.0004$ |
| `form_pot` | $+0.0193$ | `fin_track` | $-0.0004$ |
| `align` | $+0.0179$ | `obj_disturb` | $-0.0003$ |
| `g2_m1` | $+0.0100$ | `qvel_hard` | $-0.0001$ |

**合计 $+0.6167$/步** —— 即"光复现参考"能拿到的分。
**硬约束:任何奖励项都不得让零动作基线为负**(否则策略学到的第一件事是别动)。
