"""Backend validation for the exact BODex-on-official-cuRobo-v2 port."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_CUROBO_ROOT = REPO_ROOT / "third_party/curobo"


class ExactBodexUnavailable(RuntimeError):
    """Raised when an exact BODex dependency is unavailable."""


@dataclass(frozen=True)
class BodexCuroboV2BackendReport:
    curobo_path: Path
    missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return len(self.missing) == 0


def _resolved_sys_path() -> Iterable[tuple[str, Path | None]]:
    for path in sys.path:
        if not path:
            yield path, None
            continue
        try:
            yield path, Path(path).resolve()
        except OSError:
            yield path, None


def import_official_curobo():
    """Import cuRobo from OCIR's official NVLabs submodule.

    This deliberately removes the patched MagicSim and BODex cuRobo forks
    from ``sys.path`` before importing, and never imports anything from
    ``third_party/BODex`` (including its native extensions).
    """

    if not OFFICIAL_CUROBO_ROOT.exists():
        raise ExactBodexUnavailable(
            "official cuRobo checkout is missing. Initialize it with:\n"
            "  git submodule update --init --recursive third_party/curobo"
        )

    official_root = OFFICIAL_CUROBO_ROOT.resolve()
    blocked_roots = (
        (REPO_ROOT / "third_party/MagicSim/Third_Party/curobo").resolve(),
        (REPO_ROOT / "third_party/BODex/src").resolve(),
        (REPO_ROOT / "third_party/BODex").resolve(),
    )
    sys.path[:] = [
        original
        for original, resolved in _resolved_sys_path()
        if resolved is None or not any(str(resolved).startswith(str(root)) for root in blocked_roots)
    ]
    if str(official_root) not in [str(resolved) for _, resolved in _resolved_sys_path() if resolved]:
        sys.path.insert(0, str(official_root))

    loaded = sys.modules.get("curobo")
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if not str(loaded_path).startswith(str(official_root)):
            for module_name in list(sys.modules):
                if module_name == "curobo" or module_name.startswith("curobo."):
                    del sys.modules[module_name]

    import curobo

    curobo_path = Path(curobo.__file__).resolve()
    if not str(curobo_path).startswith(str(official_root)):
        raise ExactBodexUnavailable(
            "active 'curobo' did not resolve to OCIR's official NVLabs cuRobo submodule.\n"
            f"Expected root: {official_root}\n"
            f"Active path: {curobo_path}"
        )
    return curobo


def _missing_import(module_name: str, attr_name: str | None, description: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"{module_name}: missing module for {description} ({type(exc).__name__}: {exc})"
    if attr_name is not None and not hasattr(module, attr_name):
        return f"{module_name}.{attr_name}: missing symbol for {description}"
    return None


def inspect_exact_bodex_v2_backend() -> BodexCuroboV2BackendReport:
    curobo = import_official_curobo()
    curobo_path = Path(curobo.__file__).resolve()

    required = (
        ("curobo._src.types.device_cfg", "DeviceCfg", "official cuRobo v2 device type"),
        ("curobo._src.cost.cost_base", "BaseCost", "official cuRobo v2 custom cost base"),
        ("curobo._src.cost.cost_base_cfg", "BaseCostCfg", "official cuRobo v2 custom cost config"),
        ("curobo._src.geom.transform", "pose_multiply", "official cuRobo v2 transform utilities"),
        ("curobo._src.geom.sphere_fit.wp_mesh_query", "WarpMeshQuery", "official cuRobo v2 analytic mesh SDF query"),
        ("curobo._src.geom.sphere_fit.wp_mesh_query", "WarpSphereSDFFunction", "official cuRobo v2 analytic SDF autograd function"),
        ("curobo._src.optim.components.gradient_opt_core", "GradientOptCore", "official cuRobo v2 gradient optimizer core"),
        ("curobo._src.optim.gradient.line_search_strategy", "GreedyLineSearchStrategy", "official cuRobo v2 greedy line search base class"),
        ("curobo._src.types.robot", "RobotCfg", "official cuRobo v2 robot config"),
        ("curobo._src.robot.kinematics.kinematics", "Kinematics", "official cuRobo v2 kinematics"),
        ("coal", "distance", "standalone coal GJK/EPA convex distance query"),
        ("coal", "ConvexBase", "standalone coal convex hull construction"),
        ("warp", None, "warp for cuRobo v2 mesh SDF kernels"),
        ("ocir.grasp_synthesis.bodex_curobo_v2.qp", "BatchedReluQp", "ported BODex batched ReLU-QP solver"),
        ("ocir.grasp_synthesis.bodex_curobo_v2.grasp_energy", "QPEnergy", "ported BODex QP grasp energy"),
        ("ocir.grasp_synthesis.bodex_curobo_v2.grasp_cost", "BodexGraspCost", "ported BODex staged grasp cost"),
        ("ocir.grasp_synthesis.bodex_curobo_v2.contact_world", "SingleObjectContactWorld", "cuRobo-v2/coal exact single-object contact world"),
        ("ocir.grasp_synthesis.bodex_curobo_v2.newton_opt", "BodexNewtonOpt", "BODex-faithful momentum/per-group-normalized optimizer"),
        ("ocir.grasp_synthesis.bodex_curobo_v2.seed_generator", "HeurGraspSeedGenerator", "BODex-faithful surface-normal-facing grasp seed generator"),
    )

    missing = [message for module, attr, desc in required if (message := _missing_import(module, attr, desc))]
    return BodexCuroboV2BackendReport(curobo_path=curobo_path, missing=tuple(missing))


def check_exact_bodex_v2_backend() -> BodexCuroboV2BackendReport:
    report = inspect_exact_bodex_v2_backend()
    if report.ok:
        return report

    details = "\n  - ".join(report.missing)
    raise ExactBodexUnavailable(
        "Exact BODex-on-official-cuRobo-v2 is not available in the active runtime.\n"
        "This port uses official cuRobo v2 for device types, transforms, kinematics, "
        "and custom cost registration; the standalone 'coal' package (not BODex's own "
        "compiled extension) for exact convex mesh-mesh contact; and cuRobo v2's own "
        "WarpMeshQuery for sphere-mesh contact. It never imports third_party/BODex.\n"
        f"Active cuRobo path: {report.curobo_path}\n"
        f"Missing requirements:\n  - {details}"
    )
