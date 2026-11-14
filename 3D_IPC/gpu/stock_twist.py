"""Run the shared DAT twisting-cloth test with our energy evaluations."""


def run(arguments):
    from .solver_dat import SolverDAT
    from .dat_stock_scene import attach_stock_ipc
    from .dat_twist import run as run_twist
    import newton.solvers

    original_solver = newton.solvers.SolverVBD
    solvers = []

    def energy_solver(model, **stock_options):
        attach_stock_ipc(model, gs_predictor="none")
        solver = SolverDAT(model, **stock_options)
        solvers.append(solver)
        return solver

    try:
        newton.solvers.SolverVBD = energy_solver
        result = run_twist(arguments)
        if result:
            return result
        # Check our latched numerical errors outside the official timer.
        for solver in solvers:
            solver.check_status()
        return 0
    finally:
        newton.solvers.SolverVBD = original_solver
