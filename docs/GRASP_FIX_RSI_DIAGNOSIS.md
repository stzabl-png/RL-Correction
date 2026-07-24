# pp0 抓取失败诊断 + RSI 修复 (2026-07-20)

## 症状
pp0(洗洁精瓶,18cm 细高)两条消融(gc=几何+GraspPose+cuRobo, base=仅几何)各跑满 100M,
**成功里程碑全 0**,策略把物体**横向甩飞**(term/obj_div=100%)、**指尖接触恒 0**。对照:
clip11(矮圆台,旧代码,纯重建,低摩擦)至少能接触+尝试提。

## 系统排查结论(输入 + reward 逐项)
- **reward 没写错**:task/approach/contact/lift/traj/imit/curobo 各项数值都忠实反映"没抓住";
  离线单测全过。**不是 reward bug**——但它**假设骨干参考至少能接触物体**,pp0 崩了这个前提。
- **根因在输入**:零残差(纯参考回放)诊断 `diag_grasp.py`:
  - 指尖接触**恒 0**、指尖离物表面 3–10cm、物体从没抬起、手接近时先把物体**撞开 11cm**(碰撞非摩擦,
    改指尖-only 低摩擦后位移不变)。
- **物体钉死(--pin_object)再测,决定性**:
  - **cuRobo(anchor)**:钉死后**最多 5 指接触**——**对齐是对的**,问题是动态物体在手指合拢前被撞跑。
  - **人手重建(human)**:钉死也 0 接触、指尖 8→34cm 越走越远——**重建本身不抓握**(retarget 出的
    腕轨迹不把手带到物体)。
- **cuRobo 手-物在自身一致系里** close 帧腕-物 24cm(腕根),手长~18cm→指尖差几厘米没合上 + 物体被撞跑。
- **无坐标桥接 bug**:两个物体源仅差 1.2cm;上游**不需要**改坐标对齐(已验证)。

## 修复(采纳用户方向: DeepMimic RSI + kinematic-freeze + ConTrack 自适应加权)
1. **RSI (DeepMimic)**:`_reset_idx` 按 `rsi_prob=0.5` 从 [交互起, L-1] 采样参考帧起步(偏重
   grasp/lift/carry),手+物初始化到该帧参考状态(物体在手里/抬升态)。`_ref_t`/相位判断改逐-env
   `rsi_start`。让策略直接见到成功态 → task/contact/lift reward 有梯度。
2. **kinematic-freeze**:`_pre_physics_step` 前 `settle_steps` 步把物体钉在当前参考位姿(给参考速度,
   释放不跳变)+ 动作门控关。让抓握/沉降稳定后释放,避免接近撞飞。
3. **自适应 imitation (ConTrack)**:`train.py` PPO hook 按 `success_rate_ema` 调 `lam_imit/lam_traj`:
   初期 0.2×(放手探索),success 到 0.3 时恢复满(细化)。
4. **指尖-only SuperGrip**(防御性):手身低摩擦 0.2、指尖 3.0,防手掌粘连甩飞(非根治,但更对)。

## 验证(零残差 + m0/diag)
| | RSI 前 | RSI 后 |
|---|---|---|
| pp0_anchor 指尖接触 | 恒 0 | 均 2.5–3.6, max 5 |
| pp0_anchor 物体高度 | 0.9m 恒定 | 1.0–1.1m(抬起) |
| pp0_anchor success_rate(零残差) | 0 | **0.625** |
| pp0_anchor ep_rew/task | ≈0 | +0.99 |
**短训练冒烟**: reward 2 分钟内 -33→+13,`success_rate_ema ≥1% @ 180K steps`(此前 100M 从没到过里程碑)。

## 关键结论 / 待办
- **pp0_anchor(cuRobo 骨干)现在可训**——RSI 有效。推荐先训它。
- **pp0_human 骨干 RSI 救不了**(重建腕不抓握,即使加 grasp_prior 也 contact=0)——这是**上游 retarget
  质量问题**,要做"修好人手重建让腕真的抓到物体"才能走人手骨干(用户的 noisy-correction 目标需要它)。
- **success_rate 被 RSI 抬高**(rsi_prob=0.5 混了易起步的 RSI env);**真实从头成功率要 rsi_prob=0 单独评**。

## 改动文件
`env/correction_env.py`(RSI/freeze/指尖摩擦/相位判断)、`env/correction_env_cfg.py`(rsi_prob/settle语义)、
`train.py`(自适应 imitation hook)、新增 `diag_grasp.py`(抓握几何诊断,支持 --pin_object)。
