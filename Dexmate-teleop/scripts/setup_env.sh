#!/usr/bin/env bash
# Teleop env setup (uv). Re-run safe. From repo root:  bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv --python 3.11 .venv
P=.venv/bin/python
# cpu-only torch is all dex-retargeting needs (and faster to install)
uv pip install -p $P torch --index-url https://download.pytorch.org/whl/cpu
# dex-retargeting is vendored in-repo under third_party/ (no sibling checkout needed)
uv pip install -p $P -e third_party/dex-retargeting pyzmq pytest wuji-sdk
# 手臂线(PICO -> Pink IK -> 页面)的依赖。此前这里只装手部线,手臂线的包是
# 后来在机器上手动补的,新机器照脚本装会缺:viser(页面)、pin/pin-pink
# (逆运动学)、yourdfpy(URDF 显示)、msgpack(消息)、scipy(数学工具)。
# quadprog 是 pink 逆运动学的 QP 求解器后端 —— pin-pink 只装框架不装后端,
# 缺了它 solve_ik 直接抛 SolverNotFound(干净环境实测抓到的)。
uv pip install -p $P viser pin pin-pink yourdfpy msgpack scipy quadprog pyk4a
# 相机<->机器人外参标定(RobotCamCalib/extr_calib_vega_kinect.py):AprilTag 检测、
# 日志、OpenCV(要 contrib:自检用 cv2.aruco 合成标定板)。
uv pip install -p $P pupil-apriltags loguru opencv-contrib-python
# Optional: SAPIEN preview window for teleop_retarget.py --viz (not the sim; that's Isaac):
#   uv pip install -p $P "sapien==3.0.0b0"

$P scripts/prepare_assets.py
$P scripts/check_env.py
echo "OK. Try: $P scripts/teleop_retarget.py --source mock --viz"
