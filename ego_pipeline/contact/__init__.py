"""Contact alignment + heatmap: SharpaWave hand <-> reconstructed object.

Pipeline (see tools/contact_align_heatmap.py):
  frames.py    unify coordinate frames (ref_qpos scene frame -> recon world) + sanity checks
  urdf_fk.py   SharpaWave URDF forward kinematics -> hand surface geometry in world
  observe2d.py project object mesh into the image, intersect with hand/object masks
               -> the contact region as OBSERVED in the video (depth-error free)
  align.py     [next] solve one 3-DoF translation that puts the hand pads on that region
  heatmap.py   [next] per-vertex contact weight on the object mesh, in object-local frame
"""
