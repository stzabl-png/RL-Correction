# 8 卡机重启后如何继续（2026-08-11 15:41 存档）

重启目的：拯救物理 GPU 0（状态 `GPU requires reset`，CUDA 完全枚举不到它）。

---

## ★★ 重启后第一件事：重新确认 CUDA 编号映射

**重启前**，因为物理 GPU0 对 CUDA 不可见，CUDA 只枚举 7 张卡、整体偏移一位：
```
CUDA 索引 n  ==  nvidia-smi 索引 n+1
```
我们所有 `--gpu N` / `--gpu-ids N` 都按这个偏移在用。

**GPU0 修好后偏移会消失**，那时再按"减一"就**全错**。所以：

```bash
bash tools/map_gpu.sh          # 对照 nvidia-smi 与 torch 看到的已用显存
```
对上了（`CVD=n` 的已用显存 ≈ `smi n` 的已用显存）说明恢复正常，直接用 nvidia-smi 编号；
仍然错位就继续减一。**别跳过这一步** —— 昨夜 9 条视频报废、6 小时空转就是因为跑错了卡。

配套：`~/bin/start_vlm.sh` 里的 `SMI_GPU=${SMI_GPU:-$((GPU+1))}` 默认值也要跟着改回 `$GPU`。

---

## 磁盘上已保存的东西（重启不会丢）

| 内容 | 位置（UCB） |
|---|---|
| **11 条人工标注**（用户手点，最值钱） | `Output/ReconstructOutput/interim/arctic15/*/sam2_object/label_prompt.json` |
| **11 条 frame_plan.json**（fp = onset+10） | 同上目录 |
| 15fps 视频（已去畸变、嵌套布局） | `Data/arctic15/<subject>/<seq>.mp4` |
| 已完成 5 条的重建产物 | `Output/ReconstructOutput/arctic15/{s05/laptop_grab_01, s05/box_grab_01, s05/microwave_grab_01, s04/capsulemachine_grab_01, s07/ketchup_grab_01}` |
| 上游 interim（决定能否只重跑下游） | 见下表 |

**interim 保留情况**（`vipe` 和 `sam2` 都在 = 只需重跑 sam3d 之后的步骤，约 4 分钟/条）：
```
vipe=Y sam2=Y : capsulemachine  box  espressomachine  laptop  microwave  mixer  ketchup
vipe=Y sam2=N : notebook  phone
vipe=N sam2=N : scissors  waffleiron        ← 这两条要从头跑
```

**本地也已同步**（不含 masks）：`Output/ReconstructOutput/arctic15/`、
逐条评测结果 `Output/arctic_eval/rows15/*.json`。

---

## 还剩什么

已完成 **5/11**：laptop、ketchup、box、capsulemachine、microwave
待完成 **6 条**：espressomachine、mixer、notebook、phone、scissors、waffleiron
（全部因显存不足失败，产物未损坏，幂等重跑会跳过已完成步骤）

**继续的命令**（重启后先做上面的编号自检，再把 `--gpu-ids` 换成正确的值）：
```bash
ssh yanghong@169.229.192.185
# /tmp 会被重启清空 —— 脚本要重建，内容见本仓 JOURNAL.md 或下面
bash /tmp/recon_arctic15.sh     # 幂等：已完成的 take/step 自动跳过
```
脚本要点（重启后 /tmp 清空，需重建）：
```
--dataset arctic15 --root ~/Reconstruct_and_Retarget/Data/arctic15
--steps=vipe,sam3_hands,sam2_object,hawor,sam3d,sam3d_scale,fp_pose,fuse,confidence
--keep-interim --no-auto-label          # 标注已在磁盘上，别再跑 v17A
--gpu-ids <按自检结果填>
```
跑完后：
```bash
# 远端补 bridge + 接触区间（bridge 必须加 --cpu，见下）
bash /tmp/postproc15.sh
# 本地评测 + 汇总
python3 tools/arctic_eval/arctic_eval.py --take <take> --meta <meta> --audit <audit> --out Output/arctic_eval/rows15
python3 tools/arctic_eval/shape_table.py
```

---

## 重启前踩过的坑（别重复）

1. **`recon_to_replay.py` 必须加 `--cpu`**：不加会在 `import torch` 的 CUDA 初始化处崩
   （GPU0 的 ERR! 让枚举全部设备失败）。GPU0 修好后可能不再需要，但加了也无害。
2. **停 vLLM 要按 PID 杀**：真正吃显存的进程叫 `VLLM::EngineCore`，
   `pkill -f "vllm serve"` 匹配不到它。我漏杀了一个，它占 43GB 导致 4 条 take OOM 报废。
   而且 `~/bin/vlm_supervise.sh` **带自动重启**，要先停守护（tmux 会话）再杀进程。
3. **`pkill -f <关键词>` 会匹配到自己的命令行**（本会话踩了 4 次），用 PID 最稳。
4. **另一个 agent 也在用 yanghong 账户**：15:10 他起了 tmux `vlm7`（`GPU=4`，物理 GPU5），
   和我的重建撞卡。**建议约定分卡**。
5. **audit 记录按路径末两级匹配会串台**：`arctic` 与 `arctic15` 的 `<subject>/<seq>` 同名，
   会静默取到另一个数据集的 confidence（实测 15fps 的 laptop 取到了 30fps 的 84/77，
   真实是 77/71）。已改成取末三级。
6. **变体 take 的 mp4 要用硬链不能用软链**：批量会 `resolve()` 路径，
   软链会让 video_id 变回原 take，然后判"已完成"、0.03 秒结束、**不报任何错**。
7. **vipe/sam3_hands 产物按 video_id 命名**：做变体实验要逐文件改名软链（ketchup 一条 1414 个）。

---

## 已有的结论（5 条，见 JOURNAL.md 详细版）

```
物体             形状-中轴  形状-短轴   位置mm  深度mm  横向mm  深度比  confP confR
box                -0.7%      -2%      54     41     32   1.06   94.0  62.5
capsulemachine     -2.2%     +32%     387    379     71   1.67   66.0  29.0
ketchup           +22.2%     +28%     336    330     51   1.57   55.0  28.0
laptop             +2.2%    +115%     349    340     52   1.48   77.0  71.0
microwave          -2.7%     -16%     379    376     53   1.59   79.0  34.0
```
**三条 5/5 一致的结论**：① 深度误差压倒性、横向可忽略；② 形状误差集中在**最短轴**
（中轴 4/5 在 3% 以内，短轴 -16%~+115%）→ 单标量尺度原理上修不了；
③ **box 是反例且解释了根因**：规整长方体单视角就能推全三维，各项指标好一个数量级
（深度比 1.06 vs 其余 1.48–1.67）。

**confidence 的预测力：n=5 说明不了问题**（rho -0.2 ~ -0.5，需 |rho|>0.88 才显著）。
要下结论至少需要 **12 条**（6 dev / 6 held-out，按物体划分）。
