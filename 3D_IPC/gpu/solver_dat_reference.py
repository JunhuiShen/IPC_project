"""The exact pinned upstream DAT solver, without local source patches.

Select this snapshot in a fresh process before importing a solver or model.
No subclass, changed defaults or physics hooks are introduced here.
"""
from . import select_newton_runtime

select_newton_runtime("reference")

from newton.solvers import SolverVBD as SolverDATReference

__all__ = ["SolverDATReference"]
