# 重建管线交接 —— ARCTIC 真值校准实验（2026-08-11）

**给谁**：负责重建管线的 agent。
**背景**：ARCTIC 带物体 6DoF / 双手 MANO / 逐帧相机真值，第一次能把我们的重建误差量化。
跑了 2 条 take 端到端（`s05/laptop_grab_01` 828 帧、`s07/ketchup_grab_01` 703 帧），
并把杜邦分支的选帧器接进来试跑。
**证据规则**：每条都给文件行号或实测数字；我的推测会显式标注，没标的都是实测。
**你已经解决的**：见第 5 节，我确认过了，不用重复做。

---

## 0. 先看这条 —— 它让一整夜作业作废

### 0.1 ★★ 本机 CUDA 编号与 nvidia-smi **错位一位**

物理 GPU 0 处于 `ERR!`（nvidia-smi 利用率 `[N/A]`，compute-apps 那行是 `[N/A],...,[N/A]`），
**CUDA 完全枚举不到它** → 只看到 7 张、编号 0–6、整体偏移。
加 `CUDA_DEVICE_ORDER=PCI_BUS_ID` **也无效**（卡对 CUDA 不可见）。

```
nvidia-smi 已用:  smi0=0.20 smi1=20.91 smi2=36.99 smi3=21.16 smi4=18.48 smi5=0.02 smi6=19.77 smi7=18.48
torch  CVD=n:     0→21.17   1→37.25   2→21.42   3→18.74   4→3.04   5→20.03   6→18.74   7→FAIL
```
逐个对上 **`CVD=n ≈ smi(n+1)`**。

后果（都是实际发生的）：以为 `--gpu 5` 是空闲卡、实际是物理 GPU6（别人占 92–100%）→
`CUDA unspecified launch failure`，**一夜 9 条视频报废、机器空转 6 小时**；
vipe `--gpu 0` 实际去了物理 GPU1（已用 21GB）装不下 33GB。

**要做**：找运维重置物理 GPU 0。在那之前所有 `--gpu N` 都要理解成物理 N+1。
自检脚本 `tools/map_gpu.sh`。⚠ GPU0 修好后偏移消失，"减一"反而变错 —— **用前必须自检**。

---

## 1. 实例发现层 —— 最早、最严重、也最没被覆盖

### 1.1 ★★★ ketchup：三个 episode 的 instance_0001 **没有一个是正确物体**

各 episode 取中间的 accepted 帧渲染（图 `Output/arctic_vis/`）：
```
episode_00  f47   1461px  → 只圈住瓶子的**翻盖**
episode_01  f300  3331px  → 还是**翻盖**
episode_02  f347 14522px  → **瓶身下半 + 桌上纸盒 粘成一个连通块**
```
ARCTIC 的 GT 物体是**整个番茄酱瓶**。我们三个候选：两个只有小盖子，一个是瓶身与纸盒的混合体。
**不是选错 episode，是候选里根本没有正确物体** —— 选帧再准也救不了。

降级路径取了 ep02 → SAM3D 重建出一个方块。形状比对（PCA 对齐 + 按最长轴归一化，
**与尺度、朝向无关**）：
```
真值瓶子  轴比 1.00 : 0.42 : 0.25    剪影能看见瓶颈
我们      轴比 1.00 : 0.98 : 0.58    三个正交视图全是矩形
```

### 1.2 ★★ instance 编号**跨 episode 不稳定**，身份链接失败后无人兜底

同一个 `instance_0001`：ep00/ep01 是翻盖，ep02 是混合块。这正是"跨周期视觉身份链接"
该解决的，而它经常失败：
- laptop `failed_global_identity_linking`（第 3 周期匹配分 0.584 vs 次高 0.308，不够决定性）
- ketchup 同样失败，`video_mask_sequence: null`

失败后逐 episode 的 mask 仍可用，**但编号语义已不可跨 episode 使用，而下游无人知道这件事**。

⚠ **我踩过这个坑，别重复**：我写了个"合并各 ready episode 成视频级 manifest"的工具
（`scale_develop/adapt_run_dir.py::synth_video_manifest`），假设跨 episode 同名 instance
是同一物体。**ketchup 上这个假设被证伪**，合并把翻盖和混合块混成一个 id。
**该工具在身份链接失败时是有害的**，需要先加身份一致性检查。

### 1.3 ★★★ 重建错物体时，**下游所有指标都是绿的**

ketchup 重建的是"瓶身+纸盒"混合体，而：
```
confidence   conf_pos 83 / conf_rot 48 / position_grade "good" / rotation_usable True
fp_pose      703/703 帧跟满, 重配准 1, fallback 0
sam2_object  703/703 帧非空, 面积稳定
```
**全绿** —— 它确实在稳定跟踪一个东西，只是跟错了。
我们此前所有"重建质量"评估都默认"重建的就是任务物体"，**这个前提会静默失效**。

**建议（我认为是当前第一优先）**：加一道"重建的是不是任务物体"的检查。
杜邦的 Qwen 终审层已有 `is_discrete_object`，prompt 第一句就要求"先描述 mask 圈住的东西"——
把 `object_description` 和任务名（`ketchup_grab_01`）对一下即可自动发现这类错误。
成本极低，收益是**堵住一整类不可见的失败**。

---

## 2. 选帧层

### 2.1 ★ "遮挡最小"这个目标会**系统性选中退化视角**（选帧器本身的目标函数问题）

杜邦选帧器自己输出的对比图（`Output/arctic_vis/sel_laptop_i1.jpg`）把这点显示得很清楚：
```
#1 f768 occ=0.00 ← 选中：笔记本平放、正上方俯视
#2 f602 occ=0.02        立起来些，能看见侧面
#4 f672 occ=0.06        明显倾斜，厚度清晰可见
```
**物体要展示三维形状就得被拿起来/转动，而那必然意味着手在接触它** → 遮挡分反而更高、排到后面。
以遮挡最小为唯一目标，就会系统性挑到退化视角。

真值后果（laptop，形状口径与尺度朝向无关）：
```
真值  轴比 1.00 : 0.62 : 0.094
我们  轴比 1.00 : 0.61 : 0.235     长宽 -2.2%（很好）  厚度 +150%
```
**误差是各向异性的**：长宽几乎完美，只有最短轴错 2.5 倍 →
`sam3d_scale` 的**单标量尺度原理上修不了它**（无论标量取多少，0.235 变不成 0.094）。

**好消息**：Qwen 的 `view_informative` 判据**已经能准确识别**（本地 vLLM，3/3 稳定）：
```
f642(现选中) view_informative=False "完全正对的俯视视角, 物体呈现为二维平面, 无法体现其三维形状"
f340(立起来) view_informative=True  "能同时观察到顶面、侧面, 提供丰富的深度和三维结构信息"
```
但杜邦的终审里 `HARD_CRITERIA = ("occlusion","completeness","mask_quality")`，
**`view_informative` 只记录不拦截**。真值支持把它提为硬性项 —— 这条建议转给杜邦。

### 2.2 降级选帧取"**最后一个** ready episode 的最早 accepted 帧"，与文档声称相反

`auto_label_v17a.py` docstring 写"该实例**最早的** accepted 帧 —— 通常手尚未接触、遮挡最小"。
实际 `find_episode_manifest()` 里 `best = p` 被后来者覆盖 → 返回**最后一个** ready episode。
```
laptop  4 个 episode 全 ready → ep03(642-770) 首帧 = f642（全片 828 帧，谈不上"最早"）
ketchup 3 个 ready           → ep02(347-352) 首帧 = f347 ← 该 episode 只有 6 帧
```
"最后一个 episode"通常在交互**之后**，与"手尚未接触"的初衷相反。
（你新加的 `frame_plan.json` 让 FP 用 onset+10 绕开了这点，但 **prompt 帧本身和 sam3d 的回退
路径仍走这条规则**。）

---

## 3. 工程/管道层

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| 3.1 | **`_run_subprocess` 吞掉子进程 stderr** → vipe 失败无法诊断 | `_legacy/vipe/_common.py:552`；实测失败日志只有 19 行、一句 `CalledProcessError`，无任何可定位信息，只能放弃该支线 | 子进程输出落盘到 worker 日志 |
| 3.2 | **视频路径被 `resolve()`** → 软链造的变体 take **静默变回原 take** | 做"只换参考帧"实验时 video_id 变回原 take，判"已完成"、0.03 秒结束、**不报任何错** | job 解析时若视频是符号链接，打印一行提示 |
| 3.3 | **vipe/sam3_hands 产物按 video_id 命名**，挡住"复用上游只重跑下游" | `vipe/depth/<video_id>.zip` 等；ketchup 一条要逐文件改名软链 **1414 个**。**第二次踩**（screw27 注入 CAD 时也要改名 7 个） | 文件名去掉 video_id（目录已按它分层），或提供官方 "fork take" 工具 |
| 3.4 | **不给 `--keep-interim` 就删 vipe depth / sam2 masks** | laptop 因此无法参与选帧对照实验，只能整条重跑 20 分钟 | 评测/调参场景默认保留，或至少保 `vipe/depth` + `sam2_object/` |
| 3.5 | **实例发现耗时与文档差 10–50 倍** | docstring 写"~1-2 分钟"；实测 laptop **16 分钟**、ketchup **>75 分钟**（同为 4 个 episode） | 至少改 docstring；查为何同为 4 个 episode 差 4.7 倍（推测与候选部件数/seed 重试有关） |

---

## 4. 已修 —— 我改的，请勿回退

| commit | 问题 |
|---|---|
| `8b33283` | **崩溃残留目录永久毒化该视频**：v17A 拒绝写入非空目录（`FileExistsError`），一次 CUDA 崩溃留下的半截目录让之后**每次重试都在同一秒失败**。实测 9 条视频全报废。改为：确认无 ready manifest 时先清再跑 |
| `5c59763` | `reconstruct.sh` 批量里**一条失败 `exit 1` 中止整批** → 跳过并记账，全败才失败 |
| `c3683e6` | ① `--gpu-ids` 落在 `-*` 万能分支：空格写法的值被 `[0-9]*` 吃成"跑前 N 条"，且**自动标注看不到它、固定吃 0 号卡**；② 实例发现**后段**（身份链接）失败时整步判失败，丢掉完好的逐 episode 产物 |
| （更早） | confidence marker 三处查找不一致 → 抽出 `_marker_dir()` |
| （更早） | `retarget_isaacsim.py:833` 对 `nargs="+"` 用 `os.path.basename` → **多物体 Isaac 回放一次都没跑通过** |
| （更早） | 多物体必须喂**视频级** manifest，喂 episode 级会**静默塌成单物体** → 已接 `--instance all` |

---

## 5. 你已经解决的（我确认过，不用重复做）

`ccdf6c1 feat(frame-plan)` 覆盖了我原清单里的两条，而且比我建议的更好：

- **FP/SAM3D 关键帧解耦**（原 A4：三步共用一帧是单点故障）。
  `frame_plan.json` 逐物体分 `fp_register_frame` / `sam3d_frame`，三个消费方接钩，
  越界或该帧 mask 空则回退 prompt 帧并打日志。
  **`sam3d_frame` 留空正好是选帧器的槽位** —— 我原本打算改 `label_prompt.json`，
  现在改成往这里写数字即可，不碰步骤代码。
- **`--step-arg STEP:FLAG` 通用透传**（原 A1：`register-each` 不可达）。

`ea9392b` 也修正了 fp_pose 关于 track 模式消费 mask 的描述 —— 那条是我笔记记错导致的误报，
详见第 6 节。

---

## 6. 我自己弄错过的（免得你按错的说法去改）

1. ~~"fp_pose 默认 track：1 帧配准；`register-each` 才用逐帧 mask"~~ —— **错**。
   `track` 模式**每帧都用**逐帧 SAM2 mask（质心 + 6D 卡尔曼定每帧初始位姿，
   `fp_common.py:296-317`）。正确说法是"`register-each` 才用逐帧 mask **做重新配准**"。
   参数名是 **`--pose-mode`** 不是 `--fp-mode`。**管线文档本身是对的，误导来自我的笔记。**
2. ~~"Qwen 的 `view_informative` 判不出退化视角"~~ —— **错**。是我测试的 JSON 模板里
   把示例值写成了字面 `true`，模型照着填。改成 `true/false` 后判断准确、3/3 稳定。
   **教训：给 LLM 的 JSON 模板，示例值本身就是强先验。**
3. ~~"ketchup 选成了纸盒"~~ —— 不准确，实际是**瓶身下半与纸盒粘连成一个连通块**。
4. 一开始我用**轴对齐 bbox 三轴**比形状 —— **错**，SAM3D 输出朝向任意、bbox 随朝向变。
   正确做法：PCA 对齐 + 按最长轴归一化（`tools/arctic_eval/` 外的 `mesh_shape.py`）。

---

## 7. 交给你的东西

**评测工具链**（`tools/arctic_eval/`，已入库）
- `arctic_ego_video.py` ARCTIC view0 → 可用 mp4（去畸变、丢开头暗帧、落帧号映射表）
  ⚠ **ARCTIC 只有 view 0 能用**：它是原图 0.3× 纯缩放；其余 8 机位是裁剪+拉伸到 1000×1000，
  已不是一致的相机视图，**做 3D 会错**
- `arctic_eval.py` 单 take：零拟合真实误差（分深度/横向）+ 现有 conf + 候选判据信号
- `analyze.py` 汇总，按**物体**划分 dev/held-out（同一 take 内的帧高度相关，不能随机划分）
- `arctic_vs_recon.py` / `make_viz.py` 对拍与三联可视化
- `JOURNAL.md` 全过程台账（每个坑的现场记录）

**选帧器接线**（`scale_develop/`，分支 `agent/frame-select-integration`）
- 来源 `origin/agent/v17a-scale-develop`（杜邦在开发，**未合并他的分支**以免并行冲突）
- `adapt_run_dir.py` 我们的 interim 布局 → 他要的 `runs/<id>/` 布局（软链，不拷贝上万 png）
  ⚠ 其中的 manifest 合并在身份链接失败时**有害**，见 1.2
- `qwen_client.py` 加 `QWEN_BASE_URL` / `QWEN_MODEL` / `QWEN_LOCAL_VLLM`，**默认行为不变**。
  `QWEN_LOCAL_VLLM=1` 处理两处实质差异：关思考链的参数名不同（DashScope
  `extra_body.enable_thinking` vs vLLM `chat_template_kwargs`）；走 `response_format=json_object`
  从解码层保证合法 JSON —— 不设的话 Qwen3.5 先长篇推理，被 max_tokens 截断就永远等不到 JSON
- **本地 vLLM 已验证可替代阿里云 API**：物理 GPU5，`--served-model-name vlm --port 8807`，
  双图 + JSON 单次约 14 秒。`~/bin/start_vlm.sh` 已分离 `GPU`(CUDA) / `SMI_GPU`(nvidia-smi)

**可视化**（`Output/arctic_vis/`）
`shape_laptop.jpg` `shape_ketchup.jpg`（形状比对，尺度朝向无关）、
`sel_laptop_i1.jpg` `sel_ketchup.jpg`（选帧器 top-K）、
`arctic_gt_vs_recon.mp4`（真值 vs 重建三联，含俯视图 —— 深度误差在画面里看不见，俯视图能看见）

**⚠ 安全**：`qwen_client.py:18` 硬编码了**真实阿里云 API key** 且已进 git 历史，需吊销重签。

---

## 8. 已量化的基线（后续改动的对照）

只有 laptop 这条是**物体选对了**的、可信基线；ketchup 那条因 1.1 作废。
```
laptop (828 帧, 全自动)
  形状      长宽 -2.2%     厚度 +150%（PCA 归一，尺度朝向无关）
  质心误差  中位 239mm     深度分量 236mm / 横向 38mm / 投影质心仅差 33px
  深度比    我们/真值 1.36
  ViPE 相机 位置残差 12.1mm、朝向 1.94°（相机本身很准）；但世界尺度偏小 17%
  双手      178–185mm（以横向为主）
  confidence conf_pos 84 / grade "good"  ← 真实误差是物体自身长度的 73%
```
**关键结论**：confidence 的**排序有效**（高分箱比低分箱准 2.6–4.5 倍，四种误差口径一致），
但**绝对标定完全不可信**；主误差在**深度**，而所有判据都是图像内轮廓比对 ——
把物体放远同时放大，轮廓一模一样，**这类误差在图像里结构性不可见**。

---

## 9. 如果只做三件事

1. **加"重建的是不是任务物体"的检查**（1.3）—— 全绿指标下重建错物体，目前**完全不可见**，
   比任何精度优化都优先，且用现成的 Qwen 终审就能做。
2. **修 GPU0 / 明确编号映射**（0.1）—— 不修的话所有远程作业都在随机抢卡。
3. **把 `view_informative` 提为硬性项**（2.1，转给杜邦）—— 真值证据齐全，判据已验证有效。
