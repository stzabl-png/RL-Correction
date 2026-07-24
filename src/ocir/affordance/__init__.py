"""Vendored affordance-prediction model (Sonata encoder + task head).

Self-contained inference for a per-point "expected grasp area" heatmap on an
object mesh. Vendored from the AffordanceModel repo so OCIR does not depend on
an external checkout; the checkpoint lives at ``assets/affordance/model.pt``
and is swapped by replacing that file.

Runtime deps (Sonata / spconv-cu128 / torch_scatter) are NOT in the
grasp-synthesis env; run inference in the dedicated affordance conda env (see
``envs/affordance-requirements.txt``). Kept import-light so it loads in that env
without pulling in the rest of OCIR.
"""
