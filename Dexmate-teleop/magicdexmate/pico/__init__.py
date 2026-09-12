"""PICO VR (XRoboToolkit) input layer for Vega arm teleop.

Two halves, split by Python version:
  - producer side (py3.10 ONLY - xrobotoolkit_sdk ships a cp310 .so):
    `xr_client.py` + `scripts/teleop_pico_producer.py`, publishes raw poses
    over ZMQ. Run with GR00T's `.venv_teleop` or a local `.venv-pico`.
  - consumer side (py3.11, `.venv-isaac`): `xr_pose.py` frame math +
    `magicdexmate/sources/pico_source.py`, no SDK import.
"""
