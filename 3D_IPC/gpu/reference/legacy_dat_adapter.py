# Historical IPC-adapter DAT; not a production solver or the pristine reference.
"""Our energies and IPC/friction with Newton's actual soft DAT update filter.

The upstream timestep, structural color loop and planar truncation kernels are
used directly. Our local solve produces an absolute float32 proposal; DAT
receives its displacement from the current detection reference. Rejected
motion is never carried into the next color's accumulated displacement.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from newton import Model, State
from newton.solvers import SolverVBD
from newton._src.solvers.solver import SolverBase
from newton._src.solvers.vbd.particle_vbd_kernels import forward_step

from gpu import kernels
from gpu.contact import ContactBackend
from gpu.ccd import CcdBackend
from gpu.sdf import SdfBackend
from gpu.solver import SolverIPC


@wp.kernel
def _stage_predictor(q: wp.array[wp.vec3], inertia: wp.array[wp.vec3], reference: wp.array[wp.vec3],
                     flags: wp.array[int], mass: wp.array[float], mode: int,
                     displacement: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    target = reference[i]
    if (flags[i] & 1) != 0 and mass[i] > 0.0:
        if mode == 1:
            target = inertia[i]
        elif mode == 2:
            target = q[i]
    value = target - reference[i]
    for k in range(3):
        if not wp.isfinite(value[k]):
            wp.atomic_max(error, 0, 3)
    displacement[i] = value


@wp.kernel
def _stage_color(q: wp.array[wp.vec3], proposal: wp.array[wp.vec3], reference: wp.array[wp.vec3],
                  flags: wp.array[int], mass: wp.array[float], colors: wp.array[int], color: int,
                  displacement: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    target = q[i]
    if error[0] == 0 and (flags[i] & 1) != 0 and mass[i] > 0.0 and colors[i] == color:
        target = proposal[i]
    value = target - reference[i]
    for k in range(3):
        if not wp.isfinite(value[k]):
            wp.atomic_max(error, 0, 3)
    displacement[i] = value


@wp.kernel
def _accepted_displacements(q: wp.array[wp.vec3], reference: wp.array[wp.vec3],
                             displacement: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    value = q[i] - reference[i]
    for k in range(3):
        if not wp.isfinite(q[i][k]) or not wp.isfinite(value[k]):
            wp.atomic_max(error, 0, 3)
    displacement[i] = value


class SolverDAT(SolverVBD):
    """Own cloth/solid energies and IPC law, accepted by upstream soft DAT.

    Rigid proxies require a coupled body DAT adapter and are rejected here.
    The separate SolverIPC path retains the complete rigid GS/CCD solver.
    """

    solver_mode = "dat"

    def __init__(self, model: Model, *, iterations=None, vertex_contact_capacity=128,
                 edge_contact_capacity=256, allow_cpu=False):
        if not hasattr(model, "ipc"):
            raise TypeError("Build the actual Newton Model with gpu.scene.create_model")
        if model.body_count:
            raise ValueError("Our-energy DAT currently supports deformable cloth/solid scenes; use --solver ipc for rigid bodies")
        if not model.device.is_cuda and not allow_cpu:
            raise ValueError("The GPU DAT solver requires an explicit CUDA model")
        p = model.ipc.params
        for name in ("use_ogc", "use_cloth_grid", "use_jacobi_level_based_ccd"):
            if p.get(name, False):
                raise ValueError(f"{name} requests another algorithm; select only --solver dat")
        if not p.get("fixed_iters", False):
            raise ValueError("The captured GPU DAT solver requires --fixed_iters")
        radius = float(p["node_box_max"])
        gamma = float(p.get("dat_gamma", 0.9))
        padding = float(p.get("dat_node_box_padding", 1.2))
        requested_radius = float(p.get("dat_query_radius", 0.0))
        if not math.isfinite(radius) or radius <= 1e-10:
            raise ValueError("DAT node_box_max must exceed 1e-10")
        if not math.isfinite(gamma) or not 0.0 < gamma < 1.0:
            raise ValueError("DAT gamma must lie strictly between zero and one")
        if not math.isfinite(padding) or padding < 1.0 or not math.isfinite(requested_radius) or requested_radius < 0:
            raise ValueError("Invalid DAT query radius or node-box padding")
        interval = int(p.get("dat_collision_detection_interval", 0))
        if interval < 0:
            raise ValueError("DAT collision interval must be nonnegative")
        interval = interval if interval > 0 else int(p["node_box_update_count"])
        if interval <= 0:
            raise ValueError("DAT collision interval must be positive")
        base_iterations = p["max_global_iters"] if iterations is None else iterations
        if isinstance(base_iterations, bool) or not isinstance(base_iterations, int) or base_iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        predictor = bool(p.get("dat_inertial_predictor", True))
        requested_guess = any(bool(p.get(name, False)) for name in
            ("use_ccd_guess", "use_colored_ccd_guess", "use_colored_ccd_guess_level_based"))
        if not predictor and requested_guess:
            raise ValueError("dat_inertial_predictor=false requires all CCD initial-guess flags to be false")

        self.params, self._failed = p, False
        self.error = wp.zeros(1, int, device=model.device)
        self._box_radius, self._inertial_predictor = radius, predictor
        # The inherited DAT kernel limits each side to gamma*R/2. Keep that
        # exactly the native maximum node budget while the real BVH reach also
        # covers our IPC activation band and both sides' entire motion budget.
        dat_radius = 2.0 * radius / gamma
        margin = min(max(float(p["d_hat"]), 1e-8), dat_radius)
        self._query_radius = max(max(float(p["d_hat"]), 1e-8) + 2.0 * radius, requested_radius) * padding
        self._query_radius += 32.0 * float(np.finfo(np.float32).eps) * max(1.0, self._query_radius)
        super().__init__(model, iterations=base_iterations,
            particle_enable_self_contact=True, particle_self_contact_margin=margin,
            particle_self_contact_gap=dat_radius-margin,
            particle_vertex_contact_buffer_size=vertex_contact_capacity,
            particle_edge_contact_buffer_size=edge_contact_capacity,
            particle_topological_contact_filter_threshold=0,
            particle_rest_shape_contact_exclusion_radius=0.0,
            particle_enable_tile_solve=True, rigid_soft_enable_dat=False,
            dat_conservative_bound_relaxation=gamma,
            collision_frequency_type={SolverBase.CollisionSlot.SOFT_SELF_CONTACT: SolverBase.CollisionFrequencyType.ITERATIONS},
            collision_frequency={SolverBase.CollisionSlot.SOFT_SELF_CONTACT: interval},
            enable_fast_math=False)
        self.proposals = wp.empty_like(model.particle_q)
        self.contact_backend = ContactBackend(model, self.trimesh_collision_info,
            d_hat=float(p["d_hat"]), k_barrier=float(p["k_barrier"]),
            friction=float(p["friction_coefficient"]), eps_v=float(p["friction_velocity_epsilon"]),
            exact_geometry=bool(p["exact_computation_fallback"]), node_body=model.ipc.node_body, error=self.error)
        flags = model.particle_flags.numpy()
        dynamic = bool(np.any((flags & 1) & (model.particle_mass.numpy() > 0)))
        self._guess_requested = dynamic and bool(p.get("use_colored_ccd_guess", False) or
            p.get("use_colored_ccd_guess_level_based", False) or
            (p.get("use_ccd_guess", False) and not model.ipc.payload.get("uses_general_solver", False)))
        self.ccd_backend = None
        if self._guess_requested:
            self.ccd_backend = CcdBackend(self.contact_backend, node_box_radius=radius,
                exact_geometry=bool(p["exact_computation_fallback"]), error=self.error)
            self.ccd_backend.prepare_initial_guess(self.trimesh_collision_detector, p)
        self.sdf_backend = SdfBackend(model, self.error)
        include_tets = model.tet_count > 0
        include_triangles = bool(np.any(model.ipc.tri_active.numpy())) or bool(np.any(model.ipc.edge_active.numpy()))
        self._ipc_particles_per_block = 1 if include_tets else 2
        self._ipc_elasticity_kernel = kernels.make_solve_elasticity_tile(
            include_triangles, include_tets, not include_tets) if model.device.is_cuda else kernels.solve_elasticity_scalar
        self._active_color_counts = [int(np.count_nonzero(flags[group.numpy()] & 1)) for group in model.particle_color_groups]
        self.contact_capacities = {"vertex": vertex_contact_capacity, "edge": edge_contact_capacity}
        self.configuration = {"gamma": gamma, "maximum_checkpoint_displacement": radius,
            "dat_internal_query_radius": dat_radius, "actual_bvh_query_radius": self._query_radius,
            "collision_interval_iterations": interval, "inertial_predictor": predictor,
            "ccd_initial_guess": self._guess_requested, "contact_energy": "our IPC/friction",
            "accepted_update": "upstream planar DAT", "contact_capacities": self.contact_capacities}

    # Reuse equations, without reusing the IPC acceptance or initialization.
    _accumulate_particle_contact_terms = SolverIPC._accumulate_particle_contact_terms
    _solve_particle_elasticity_color = SolverIPC._solve_particle_elasticity_color
    prepare_capture = SolverIPC.prepare_capture
    reset_status = SolverIPC.reset_status
    check_status = SolverIPC.check_status

    def _collision_detection_penetration_free(self, current_state: State, *, reset_reference=True):
        if reset_reference:
            self.pos_prev_collision_detection.assign(current_state.particle_q)
            self.particle_displacements.zero_()
        # The same complete upstream BVH rows feed IPC forces and native DAT
        # division planes. Their min-distance cache is not used by planar DAT.
        self.contact_backend.detect(self.trimesh_collision_detector, current_state.particle_q,
                                    self._query_radius, refit=True)

    def _initialize_particles(self, state_in: State, state_out: State, contacts, dt):
        m = self.model
        if not m.particle_count:
            return
        wp.launch(forward_step, m.particle_count,
            [dt, m.gravity, m.particle_world, self.particle_q_prev, state_in.particle_q,
             state_in.particle_qd, m.particle_inv_mass, state_in.particle_f, m.particle_flags],
            [self.inertia, self.particle_displacements], device=self.device)
        self.sdf_backend.prepare(dt)
        mode = 1 if self._inertial_predictor else 0
        if self._guess_requested:
            self.ccd_backend.initial_guess(state_in.particle_q, self.particle_q_prev,
                state_in.particle_qd, self.inertia, dt, self.params, detector=self.trimesh_collision_detector)
            # Predictor sweeps replace the detector rows/BVH. Restore the old
            # checkpoint coverage before truncating motion relative to it.
            self.contact_backend.detect(self.trimesh_collision_detector, self.pos_prev_collision_detection,
                                        self._query_radius, refit=True)
            mode = 2
        wp.launch(_stage_predictor, m.particle_count,
            [state_in.particle_q, self.inertia, self.pos_prev_collision_detection, m.particle_flags,
             m.particle_mass, mode, self.particle_displacements, self.error], device=self.device)
        self._penetration_free_truncation(state_in, contacts)
        self._sync_accepted(state_in)
        if self.iterations == 0:
            wp.copy(state_out.particle_q, state_in.particle_q)

    def _sync_accepted(self, state):
        wp.launch(_accepted_displacements, self.model.particle_count,
            [state.particle_q, self.pos_prev_collision_detection, self.particle_displacements, self.error], device=self.device)

    def _finish_particle_color(self, state_in, state_out, contacts, dt, color):
        if not self._active_color_counts[color]:
            return
        m = self.model
        wp.launch(_stage_color, m.particle_count,
            [state_in.particle_q, self.proposals, self.pos_prev_collision_detection, m.particle_flags,
             m.particle_mass, m.particle_colors, color, self.particle_displacements, self.error], device=self.device)
        super()._finish_particle_color(state_in, state_out, contacts, dt, color)
        self._sync_accepted(state_in)


assert SolverDAT.step is SolverVBD.step
assert SolverDAT._solve_particle_iteration is SolverVBD._solve_particle_iteration
wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
