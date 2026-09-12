# SharpaWave hand assets — attribution

The `right/` and `left/` URDFs and meshes here are derived from Sharpa's
official robot description, **© 2025 Sharpa Group, Apache License 2.0**
(see `LICENSE.txt` / `NOTICE.txt`).

Source: `sharpa-urdf-usd-xml/wave_01/{right,left}_sharpa_wave/`
(https://github.com/sharpa-robotics, internal mirror at
`~/luhr/magicsim/Sharpa/sharpa-urdf-usd-xml`).

The only modification vs. upstream: mesh references were rewritten from the ROS
package form `package://<side>_sharpa_wave/meshes/...` to repo-relative
`meshes/...` so SAPIEN/pinocchio resolve them without a ROS package path.

Regenerate (requires the source repo present):

    python scripts/prepare_assets.py
