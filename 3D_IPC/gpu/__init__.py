"""Our physics on the imported Newton GPU solver structure."""

from pathlib import Path
import sys


_third_party = Path(__file__).resolve().parent / "third_party"
_vendor_roots = {"project": _third_party / "newton", "reference": _third_party / "newton_reference"}
_vendor = _vendor_roots["project"]
_loaded = sys.modules.get("newton")
if _loaded is not None:
    _origin = getattr(_loaded, "__file__", None)
    _matches = [root for root in _vendor_roots.values()
                if _origin and Path(_origin).resolve().is_relative_to(root)]
    if not _matches:
        raise ImportError("An external Newton package is already loaded; start a fresh process for this GPU solver")
    _vendor = _matches[0]
if str(_vendor) not in sys.path:
    sys.path.insert(0, str(_vendor))


def select_newton_runtime(kind="project"):
    """Select an internal snapshot before Newton imports; never mix runtimes."""
    global _vendor
    if kind not in _vendor_roots:
        raise ValueError("Newton runtime must be 'project' or 'reference'")
    root = _vendor_roots[kind]
    if not (root / "newton/__init__.py").is_file():
        raise ImportError(f"Missing local Newton snapshot: {root}")
    loaded = sys.modules.get("newton")
    origin = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded is not None and (not origin or not Path(origin).resolve().is_relative_to(root)):
        raise ImportError("Start a fresh process to select another Newton snapshot")
    for value in _vendor_roots.values():
        while str(value) in sys.path:
            sys.path.remove(str(value))
    sys.path.insert(0, str(root))
    _vendor = root
    return root


__all__ = ["SolverIPC", "SolverDAT", "SolverDATReference"]


def __getattr__(name):
    if name == "SolverIPC":
        from .solver import SolverIPC
        return SolverIPC
    if name == "SolverDAT":
        select_newton_runtime("reference")
        from .solver_dat import SolverDAT
        return SolverDAT
    if name == "SolverDATReference":
        from .solver_dat_reference import SolverDATReference
        return SolverDATReference
    raise AttributeError(name)
