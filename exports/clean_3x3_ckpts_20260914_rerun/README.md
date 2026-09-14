# Clean 20M 口径补充 (2026-09-14)
- `Clean3_ablBase_s42_20M_c50/`: take3 Base 从头重跑到 20M (原 run 续跑无 20M 节点), 配方 = 原 take3 Base (v1h 母带 / HAND_REF=1 / 课程 release 50→10), world.json / TB / train.log / last.pth(=20.0M)。台账 §5.26~5.27。
- `eval10_20M/<run>_20M/`: 九条 20M 节点的 eval10 JSON (10 env, jitter 0, seed 2026, Denso `common/eval10.sh`), `eval10_out/` 为日志。`Clean3_ablBase_rerun_s42_{19M,20M}` 是作废的第一次重跑 (课程起点误设 10), 仅留证。
- 汇总表: `docs/RESULTS_20M_20260914.md`。
