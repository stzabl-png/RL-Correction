# confidence —— 物体轨迹可信度链路（Step2 的第 9 道工序）

一条 take 重建(fuse)完成后，自动给物体轨迹逐帧打可信度分、RTS 平滑出双版本轨迹 + σ，
并汇总成 take 级裁决清单。**评分标准、smooth/filtered/σ 的用法、RL 消费规则全在
[`CONFIDENCE_GUIDE.md`](CONFIDENCE_GUIDE.md)**；接入重建 pipeline 的方案与踩坑记录在
[`PIPELINE_INTEGRATION.md`](PIPELINE_INTEGRATION.md)。

```
pose_audit.py            9 判据逐帧打分 conf_pos/conf_rot (0~100); --scene 单take增量模式
cotracker_consistency.py CoTracker 第二观察员(纯 2D 比对, 不解 PnP)
rts_smoother.py          conf→测量噪声; KF+RTS(位置)/MEKF(旋转), 出 σ 与因果/非因果双轨迹
take_manifest.py         take 级裁决(透明排除/同视频自动择优/旋转可用性分级)
heldout_eval.py          held-out 验收驱动(静止段真值红线)
conf_viz.py              可信度叠加视频
pose_calib_sheet.py      人眼校准表(阈值出处证据)
pipeline_step/           部署到 recon_pipeline/confidence/run_sequence.py 的薄包装
                         (注册进 run_pipeline.py 与 run_batch_queue.py, env: hawor, GPU 7GB)
```

部署位置（开发机）：工具链在 `RL_Correction/steps/step2_reconstruction/`，
流水线包装在 `$HV2RD_ROOT/recon_pipeline/confidence/`；本目录是入库快照。
数据产物（pose_audit.json / TAKE_MANIFEST.json / rts/ / cc/）不入库，
在 `$RR_ROOT/Data/VideoPrior/poseqa/`。
