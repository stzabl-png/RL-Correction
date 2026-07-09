# Sharpa Wave Hand Assets

This folder is the OCIR-local source of truth for Sharpa Wave hand assets.

- `sharpa_wave_right.yml`: OCIR descriptor for the supported right-hand setup.
- `urdf/right_sharpa_wave` and `urdf/left_sharpa_wave`: URDF packages and mesh folders grouped by side.
- `usd/right` and `usd/left`: USD assets grouped by side.
- `collision/curobo`: cuRobo/BODex collision sphere files.
- `contact_points/dex_hand_pose`: contact-point categories and hand pose helper configs.
- `robot_configs`: cuRobo robot configs used by BODex and other cuRobo-based tasks.
- `grasp_synthesis/bodex`: BODex grasp synthesis configs, including contact strategy and optimizer setup.

The original BODex adapter loads `sharpa_wave_right.yml` and passes the reusable
`robot_configs/curobo_sharpa_right.yml` robot config to BODex. The active grasp
config still controls which links are treated as contact links at runtime.
