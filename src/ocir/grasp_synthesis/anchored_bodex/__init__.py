"""Anchored BODex: human-demonstration-guided grasp synthesis on cuRobo v2.

A sibling pipeline to :mod:`ocir.grasp_synthesis.bodex_curobo_v2` (which it
imports but never modifies). Instead of sampling seeds on the object surface,
it anchors the optimization to a reconstructed human video demonstration:

- seeds are retargeted human hand poses from frames where the hand contacts
  the object (relaxed out of contact, then jittered);
- an affordance heatmap (cached per sequence) drives an attraction energy and
  ranking metric so contacts land where the human touched the object;
- the hand contact-point set is selected per sequence from which human hand
  parts actually made contact;
- an annealed pose prior keeps each seed near its own retargeted anchor early
  in the optimization while the BODex force-closure QP decides the outcome.
"""
