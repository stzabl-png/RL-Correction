"""OCIR port of the exact BODex grasp objective onto official cuRobo v2.

The code in this package is intentionally separate from both the official
``third_party/curobo`` checkout and the vendored ``third_party/BODex``
codebase.  It ports BODex's grasp-contact objective (grasp-matrix/QP force
closure, staged contact cost, contact queries, and optimizer behavior) onto
official cuRobo v2 types and extension points, without importing or linking
against anything under ``third_party/BODex``.
"""

from ocir.grasp_synthesis.bodex_curobo_v2.backend import (
    BodexCuroboV2BackendReport,
    ExactBodexUnavailable,
    check_exact_bodex_v2_backend,
    import_official_curobo,
)

__all__ = [
    "BodexCuroboV2BackendReport",
    "ExactBodexUnavailable",
    "check_exact_bodex_v2_backend",
    "import_official_curobo",
]
