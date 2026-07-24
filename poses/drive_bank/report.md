# Pose bank report
policy: logs/debug/2026-07-03_17-52-05/stage1_nn/last.pth
task: Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-PoseBank-v1
good envs: 256/256

| pose | picked-for | dz (m) | n_eng | omega (rad/s) | spread | ratios T/I/M/R/P | |F| per finger (N) |
|---|---|---|---|---|---|---|---|
| pose1 | omega | -0.0051 | 3 | +7.31 | 0.612 | 0.45/0.13/0.00/0.43/0.00 | 5.44/1.54/0.00/5.19/0.00 |
| pose2 | neng | -0.0041 | 3 | +6.57 | 0.555 | 0.48/0.07/0.00/0.45/0.00 | 6.80/0.96/0.00/6.36/0.00 |
| pose3 | spread | -0.0033 | 3 | +4.50 | 0.663 | 0.42/0.22/0.00/0.36/0.00 | 6.36/3.38/0.00/5.54/0.00 |

Contact positions (object-relative, m) for picked poses:
- pose1: T=(+0.018,+0.014,-0.003); I=(-0.019,+0.007,-0.013); R=(+0.011,-0.020,-0.012)
- pose2: T=(+0.018,+0.015,-0.004); I=(-0.019,+0.006,-0.013); R=(+0.006,-0.021,-0.012)
- pose3: T=(+0.019,+0.014,-0.002); I=(-0.020,+0.004,-0.014); R=(+0.001,-0.021,-0.012)
