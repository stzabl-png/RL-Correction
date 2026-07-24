# Isaac ROS FoundationPose (STEP_6_pose branch only)

`recon_pipeline/fp_pose` uses the **Python FoundationPose + FP++** backend. The default mode is `--pose-mode track`, which registers the object prompt frame, then runs mask-centroid/6D-Kalman initialized `track_one(...)` forward and backward to cover every video frame and records `foundationpose_python_pp` in `fp_pose_meta.json`. It does not export Isaac scene frames or run TensorRT nodes.

The Isaac ROS + TensorRT stack lives on branch [`STEP_6_pose`](https://github.com/jiaka1chen/HumanVideo2RobotData/tree/STEP_6_pose) under `pipeline/step6_pose/foundationpose_pp_ros`. That path expects an external register pose and runs **tracking only** via `FoundationPoseTrackingNode` + `pp_tracker`.

For deployment-speed TensorRT tracking, use STEP_6_pose directly. For development and this repo’s default pipeline, use `fp_pose/run_sequence.py` with the `foundationpose` conda env (see [docs/SETUP.md](../docs/SETUP.md)). Per-frame registration remains available for diagnostics with `--pose-mode register-each`.
