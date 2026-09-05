# Sweep2 reward and success contract

The task is evaluated in the dustpan coordinate frame. The dustpan mesh axes were
checked directly: local `x` is its width, local `+y` is the upward normal in the
held scene, and local `+z` runs from handle to the open lip.

The fixed 1 cm cube begins just outside the positive-z lip. The only dense positive
task term is an earn-only potential for moving that cube across the lip while it
remains laterally aligned with the pan. Holding still and moving backward cannot
produce positive task income. One-time bonuses mark ready geometry, meaningful
broom-associated cube motion, full containment, and stable containment.

Success requires the complete cube footprint to lie within the conservative pan
interior and its velocity relative to the pan to remain below 5 cm/s for 10 control
steps (0.5 seconds at 20 Hz). Success immediately terminates the episode. Reference
tracking is a separate confidence-gated P-OBJ term; it is never accepted as task
success and cannot by itself advance the task clock.
