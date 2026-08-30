# Engineering journal

## 2026-08-30 — Sweep task bootstrap

- Goal: Add a fixed-start bimanual sweep task on top of the verified Pour17 residual-RL stack.
- Changes: Created the task-owned engineering records and output-directory policy on branch `sweep-task`.
- Reasoning: Videos, training outputs, datasets, and generated artifacts must remain outside Git history while source and design records remain reviewable.
- Validation: Baseline branch `Step4_RL_Correction` resolved to `4b1daa75ea0625dff10e387f41cfd7320c191d96`; the new branch started from a clean worktree.
- Remaining: Import Sweep2 data, build the scene/reference, validate replay, implement the tracker and training path.
