"""Project level-based GPU CCD prediction for the reference VBD controller.

Only initialization changes. VBD still owns its inertia, local solve, collision
response and velocity update. Predictor candidates and schedules are private;
they never overwrite the native DAT contact rows or the VBD color groups.
"""
import math

import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionDetector
from newton._src.solvers.vbd.particle_vbd_kernels import forward_step

from .ccd import CcdBackend, resolve_cached
from .contact import ContactBackend
from .initial_guess_schedule import ParallelColors, ParallelLevels


def parameters(params):
    """Freeze and validate the predictor configuration before allocating work."""
    result = dict(params)
    if result.get('ccd_method','interval') not in ('interval','plane','direct_rounding'):
        raise ValueError('CCD method must be interval, plane, or direct_rounding')
    result.setdefault('colored_ccd_guess_iters', 1)
    result.setdefault('colored_ccd_guess_fraction', 1.0)
    sweeps, fraction = result['colored_ccd_guess_iters'], result['colored_ccd_guess_fraction']
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 0:
        raise ValueError('Colored CCD sweeps must be a nonnegative integer')
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Colored CCD fraction must be in [0,1]')
    exact = result.get('exact_computation_fallback', True)
    direct = result.get('ccd_exact_step_bound', False)
    if not isinstance(exact, bool) or not isinstance(direct, bool):
        raise TypeError('CCD exact fallback and direct-step options must be booleans')
    if direct and not exact:
        raise ValueError('Exact-step proposals require robust CCD')
    radius = result.get('node_box_max', .01)
    if isinstance(radius, bool) or not math.isfinite(radius) or radius <= 1e-10:
        raise ValueError('The CCD movement-box radius must exceed its 1e-10 inset')
    return result


def prepare(solver, vertex_capacity, edge_capacity):
    """Allocate the existing CCD implementation without selecting the GS runtime."""
    model, p = solver.model, solver._ccd_guess_params
    detector = TriMeshCollisionDetector(model, init_collision_info=True,
        vertex_collision_buffer_pre_alloc=vertex_capacity,
        edge_collision_buffer_pre_alloc=edge_capacity,
        topological_contact_filter_threshold=0)
    contacts = ContactBackend(model, detector.collision_info,
        d_hat=float(p.get('d_hat', .002)), k_barrier=0.0,
        exact_geometry=p.get('exact_computation_fallback', True),
        node_body=getattr(model.ipc, 'node_body', None),
        error=solver.error, track_query_epoch=True)
    backend = CcdBackend(contacts, node_box_radius=p.get('node_box_max', .01),
        exact_geometry=p.get('exact_computation_fallback', True),
        exact_step_bound=p.get('ccd_exact_step_bound', False),
        error=solver.error, group_query_block_dim=64)
    if not backend.exact_step_bound:
        backend._resolve = resolve_cached
    backend.prepare_initial_guess(detector, p, use_group_queries=True,
                                  independent_colors=True)
    graph = backend._initial_guess.graph
    graph.prepare_parallel_colors(ParallelColors(graph))
    graph.prepare_parallel_levels(ParallelLevels(graph))
    solver._ccd_guess_detector = detector
    solver._ccd_guess_contacts = contacts
    solver.ccd_backend = backend


def initialize(solver, state_in, state_out, contacts, dt):
    """Run one discarded probe plus updating sweeps, then rebase VBD at the guess."""
    model = solver.model
    if not model.particle_count:
        return
    # Native forward_step saves x_n and the gravity-inclusive inertial target,
    # but does not change positions. Our predictor targets x_n + fraction*h*v_n,
    # exactly as in SolverIPC; it must not start from the native published guess.
    wp.launch(forward_step, model.particle_count,
        inputs=[dt, model.gravity, model.particle_world, solver.particle_q_prev,
                state_in.particle_q, state_in.particle_qd, model.particle_inv_mass,
                state_in.particle_f, model.particle_flags],
        outputs=[solver.inertia, solver.particle_displacements], device=solver.device)
    solver.ccd_backend.initial_guess(state_in.particle_q, solver.particle_q_prev,
        state_in.particle_qd, solver.inertia, dt, solver._ccd_guess_params,
        detector=solver._ccd_guess_detector)
    # DAT displacements must be relative to its NEW accepted reference. Keeping
    # native forward_step's raw displacement would apply the predictor twice.
    solver.pos_prev_collision_detection.assign(state_in.particle_q)
    solver.particle_displacements.zero_()
    if solver.particle_enable_self_contact:
        solver._collision_detection_penetration_free(state_in, reset_reference=False)
    if solver.iterations == 0:
        wp.copy(state_out.particle_q, state_in.particle_q)
