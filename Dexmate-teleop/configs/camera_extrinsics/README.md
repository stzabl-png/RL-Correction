# configs/camera_extrinsics/

One `kinect_<serial>.yaml` per Azure Kinect, written by
`RobotCamCalib/extr_calib_vega_kinect.py` (schema `dexmate_teleop.camera_extrinsics.v1`), plus the
raw samples in `<yaml>.samples.npz`. Read with `magicdexmate.camera_extrinsics.load(serial)`.

* `T_base_color` — colour camera in the robot `base` frame (the calibrated result)
* `T_color_depth` — factory depth→colour
* `T_base_depth` — product; the frame `scripts/kinect_pointcloud.py --align depth` records in

Redo the calibration whenever the tripod or the robot chassis moves. Procedure:
`RobotCamCalib/docs/VEGA-kinect-extrinsics.md`.
