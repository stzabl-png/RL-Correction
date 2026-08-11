# 远程全链跑法（UCB 8×A6000）

> 2026-08-10 起的规矩：**跑数据一律远程，本地 4080S 16GB 只做开发**。
> 依据见 RUNTIME_LEDGER.md（16GB 边界 = sam3d/sam3d_scale/fp_pose 三步，FP 峰值 ~27GB）。

## 机器与账户

- `yanghong@169.229.192.185`（UCB 共享机，8×A6000 48GB，VPN 内，本机免密已通）
- **礼仪红线**：GPU7 = vLLM 服务专占；root 的 rl_rebuild 训练任务不能动；
  选卡前 `nvidia-smi` 看空闲，留 ≥7GB 余量（邻居显存会瞬时波动）
- 代码树 `~/Reconstruct_and_Retarget` = 本仓 Step2_NoisyRecon 分支的 git 工作区
  （中转 bare 仓 `~/repos/RL-Correction.git`）；老版编排代码在
  `_retired_recon_pipeline_20260801/` 仅留参考
- 布局：模型树物理在顶层 `third_party/`（env 的 editable 指它，别搬动），
  `ego_pipeline/Reconstruction/third_party` 与 `data` 是符号链接

## 同步代码（本地改完之后）

```bash
# 本地
git push ucb Step2_NoisyRecon        # ucb = yanghong@169.229.192.185:repos/RL-Correction.git
# 远程
ssh yanghong@169.229.192.185 "cd ~/Reconstruct_and_Retarget && git pull -q origin Step2_NoisyRecon"
```

## 跑重建（远程, 全 10 步含 v17A 自动标注、confidence、contact）

```bash
ssh yanghong@169.229.192.185
export PATH=~/miniconda3/bin:$PATH && source ~/miniconda3/etc/profile.d/conda.sh   # 非登录 shell 必须
cd ~/Reconstruct_and_Retarget/ego_pipeline
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # 挑空闲卡
./reconstruct.sh <视频或目录> --dataset egodex --root <数据root> --gpu-ids=<空闲卡>   # = 必须带
# 长任务用 tmux 包(SSH 断开不死): tmux new -d -s recon '... reconstruct.sh ...'
```

要点：
- `SAM3_VERSION=sam3` 已在 reconstruct.sh 固化，无需手动
- 步骤脚本会覆写 CUDA_VISIBLE_DEVICES —— **选卡只认 `--gpu-ids=`/`--gpu N` 传参**
- vipe 走 `conda run -n cu128 uv run --no-sync`（venv 是重建过的，实测通）
- v17A 自动标注默认开（标注缺失时自动出 mask）；`--no-auto-label` 关；`--web` 人工兜底

## 跑完拉回本地库（poseqa 以本地为权威）

```bash
# 1) 成品 take 目录(含 confidence marker)
rsync -a yanghong@169.229.192.185:~/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex/<路径>/ \
      $RR/Output/ReconstructOutput/egodex/<路径>/
# 2) CT 观测(免本地重跑 CoTracker)
rsync -a yanghong@169.229.192.185:~/Reconstruct_and_Retarget/Data/VideoPrior/poseqa/cc/ \
      $RR/Data/VideoPrior/poseqa/cc/
# 3) 本地增量入库(每条 ~15s, 复用远程 cc)
PY=/home/lyh/anaconda3/envs/hawor/bin/python
T=/home/lyh/Project/RL_Correction/steps/step2_reconstruction
P=$RR/Data/VideoPrior/poseqa
$PY $T/pose_audit.py --scene <take目录> --out $P --ct-dir $P/cc
$PY $T/rts_smoother.py --scene <take目录> --audit $P/pose_audit.json --out $P/rts
$PY $T/take_manifest.py --audit $P/pose_audit.json --out $P/TAKE_MANIFEST.json
```

远程 poseqa（`~/Reconstruct_and_Retarget/Data/VideoPrior/poseqa`）只是批量运行的工作区，
**take 裁决以本地 TAKE_MANIFEST.json 为准**（下游 RL 读本地）。

## 已踩平的环境坑（别再踩）

- conda-pack 环境的 pip shebang 坏（`python3.10` 不在 PATH）→ 装包用 `python -m pip`
- UCB hawor env 曾缺 `rtree`（已装, 2026-08-10）
- 验收基准：pour/11 远程 confidence 与本地打分**逐字节一致**（pos 93.0/rot 30.5）

## contact 步(第 10 步)在 UCB 的现状

MagicDexMate 的 .venv-isaac(dex_retargeting+pinocchio)**尚未部署到 UCB** —— contact 步
检测到依赖缺失会写 `skipped_missing_deps` 标记继续批量, 不挡整条视频。两个选择:
(a) 拉回本地后 `--force` 补跑该步(纯 CPU ~8 分钟/手); (b) 把 MagicDexMate venv
conda-pack/rsync 到 UCB 后 --force。检出 skipped 的 take:
`grep -rl skipped_missing_deps Output/ReconstructOutput --include=contact_complete.json`

## VLM 透明门(2026-08-10 接入, 规则 v2)

auto_label 内置逐实例透明判定: 只过滤"空透明"(高置信), 全滤的视频以退出码 3 从清单
剔除继续批量。依赖 GPU7 的 vlm7 服务(tmux `vlm7`, ~/bin/vlm_supervise.sh 守护,
UCB 上默认 127.0.0.1:8807 直连; 本地跑要 ssh -L 隧道 + VLM_API_BASE)。
服务没起时 fail-open 全放行并醒目警告。AUTO_LABEL_VLM_GATE=0 关门。
校准基线四案例(全对才算过): 3号清水瓶滤/pour茶瓶留/2号金属瓶留/pour灰杯留。

## UCB GPU 编号错位(已修复, 教训保留)

2026-08-11 物理 GPU0 曾 ERR → CUDA 只见 7 张、`--gpu N`=物理 N+1, 一夜 9 条视频因此报废。
当日重启修复, **编号已恢复一致**(CUDA 8 卡实测)。教训固化: 机器重启/驱动变更后, 跑任务前
先 `tools/map_gpu.sh` 自检编号 —— 错位可能再次出现, 而任何硬编码的 ±1 补偿在状态翻转后都会反噬。
