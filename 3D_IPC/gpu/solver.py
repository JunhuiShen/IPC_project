"""Our equations and CCD in the actual imported VBD GPU timestep.

SolverIPC deliberately inherits SolverVBD.step and its particle-color loop.
Newton owns the float32 Model/State, adjacency, BVHs, color groups and velocity
publication. Only the model-specific physics and accepted-step hooks differ.
The production particle solve retains the upstream cooperative 16-lane layout.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from . import select_newton_runtime

select_newton_runtime("project")

from newton import Model, State
from newton.solvers import SolverVBD
from newton._src.solvers.solver import SolverBase
from newton._src.solvers.vbd.particle_vbd_kernels import forward_step

from . import kernels
from .ccd import CcdBackend
from .sdf import SdfBackend


class SolverIPC(SolverVBD):
    """Our GS/IPC/CCD model plugged into the authors' concrete GPU solver.

    Float32 state and tree-reduced color updates follow the upstream execution
    model. Contact-coupled vertices of one structural color update together;
    CCD certifies their combined represented motion before it is committed.
    This is not a promise of a bit-identical serial CPU GS trajectory.
    """

    def __init__(self, model: Model, *, iterations=None, vertex_contact_capacity=128,
                 edge_contact_capacity=256, allow_cpu=False):
        if not hasattr(model, "ipc"):
            raise TypeError("Build the actual Newton Model with gpu.scene.create_model")
        if not model.device.is_cuda and not allow_cpu:
            raise ValueError("The GPU solver requires an explicit CUDA model")
        p = model.ipc.params
        for name in ("use_dat", "use_ogc", "use_cloth_grid", "use_jacobi_level_based_ccd"):
            if p.get(name, False):
                raise ValueError(f"{name} requests another algorithm; this solver uses our IPC/CCD physics")
        if not p.get("fixed_iters", False):
            raise ValueError("The captured GPU solver requires --fixed_iters, like the native GPU driver")
        for name in ("node_box_max", "node_box_min", "dt"):
            if not math.isfinite(float(p[name])) or p[name] <= 0:
                raise ValueError(f"Invalid solver parameter {name}")
        if p["node_box_max"] < p["node_box_min"] or p["node_box_update_count"] <= 0:
            raise ValueError("Invalid movement-box configuration")
        if p.get("ccd_exact_step_bound", False) and not p.get("exact_computation_fallback", True):
            raise ValueError("Exact-step proposals require robust CCD")
        self.error = wp.zeros(1, int, device=model.device)
        self._failed = False
        self.params = p
        self._box_radius = float(p["node_box_max"])
        # The detector queries current geometry. Cover both primitives' full
        # coordinatewise blue-box budgets until the next detection checkpoint.
        self._query_radius = max(float(p["d_hat"]), 1e-8) + 2.0 * self._box_radius
        self._query_radius += 32.0 * np.finfo(np.float32).eps * max(1.0, self._query_radius)
        base_iterations = p["max_global_iters"] if iterations is None else iterations
        if isinstance(base_iterations, bool) or not isinstance(base_iterations, int) or base_iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        super().__init__(model, iterations=base_iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=max(float(p["d_hat"]), 1e-8),
            particle_self_contact_gap=self._query_radius-max(float(p["d_hat"]), 1e-8),
            particle_vertex_contact_buffer_size=vertex_contact_capacity,
            particle_edge_contact_buffer_size=edge_contact_capacity,
            particle_topological_contact_filter_threshold=0,
            particle_rest_shape_contact_exclusion_radius=0.0,
            particle_enable_tile_solve=True,
            rigid_soft_enable_dat=False,
            collision_frequency_type={SolverBase.CollisionSlot.SOFT_SELF_CONTACT: SolverBase.CollisionFrequencyType.ITERATIONS},
            collision_frequency={SolverBase.CollisionSlot.SOFT_SELF_CONTACT: int(p["node_box_update_count"])},
            enable_fast_math=False)
        self.proposals = wp.empty_like(model.particle_q)
        from .contact_original import ContactBackendIPC
        self.contact_backend = ContactBackendIPC(model, self.trimesh_collision_info,
            d_hat=float(p["d_hat"]), k_barrier=float(p["k_barrier"]),
            friction=float(p["friction_coefficient"]), eps_v=float(p["friction_velocity_epsilon"]),
            exact_geometry=bool(p["exact_computation_fallback"]), node_body=model.ipc.node_body, error=self.error,
            track_query_epoch=bool(p.get("use_colored_ccd_guess", False) or
                                   p.get("use_colored_ccd_guess_level_based", False)))
        self.ccd_backend = CcdBackend(self.contact_backend, node_box_radius=self._box_radius,
            exact_geometry=bool(p["exact_computation_fallback"]), error=self.error, group_query_block_dim=64)
        if not self.ccd_backend.exact_step_bound:
            from .ccd_cached import resolve as resolve_cached
            self.ccd_backend._resolve = resolve_cached
        flags = model.particle_flags.numpy()
        has_dynamic_particles = bool(np.any((flags & 1) & (model.particle_mass.numpy() > 0)))
        self._guess_requested = has_dynamic_particles and bool(p.get("use_colored_ccd_guess", False) or
            p.get("use_colored_ccd_guess_level_based", False) or
            (p.get("use_ccd_guess", False) and not model.ipc.payload.get("uses_general_solver", False) and model.body_count == 0))
        if self._guess_requested:
            self.ccd_backend.prepare_initial_guess(self.trimesh_collision_detector, self.params, use_group_queries=True)
            graph = self.ccd_backend._initial_guess.graph
            if graph is not None:
                from .parallel_colors import ParallelColors
                graph.prepare_parallel_colors(ParallelColors(graph))
                if p.get("use_colored_ccd_guess_level_based", False):
                    from .parallel_levels import ParallelLevels
                    graph.prepare_parallel_levels(ParallelLevels(graph))
        self.sdf_backend = SdfBackend(model, self.error)
        include_tets = model.tet_count > 0
        include_triangles = bool(np.any(model.ipc.tri_active.numpy())) or bool(np.any(model.ipc.edge_active.numpy()))
        self._ipc_particles_per_block = 1 if include_tets else 2
        self._ipc_elasticity_kernel = kernels.make_solve_elasticity_tile(
            include_triangles, include_tets, not include_tets) if model.device.is_cuda else kernels.solve_elasticity_scalar
        self._active_color_counts = []
        for group in model.particle_color_groups:
            ids = group.numpy()
            self._active_color_counts.append(int(np.count_nonzero(flags[ids] & 1)))
        self._rigid = None
        if model.body_count:
            from .body_solver import BodySolver
            self._rigid = BodySolver(self)
            self._query_radius = max(self._query_radius, self._rigid.query_radius)

    # The public timestep and particle color schedule are inherited unchanged.
    # The hooks below are the necessary substitutions for our physical model.
    def _pre_initialization_detection(self, state_in, dt):
        # Upstream needs a checkpoint before its DAT-limited prediction. Our
        # predictor queries its own complete swept bounds, and initialization
        # below always rebuilds at the accepted initial guess. A second refit
        # of the same input mesh would add no candidate coverage.
        return False

    def _initialize_particles(self, state_in: State, state_out: State, contacts, dt):
        m = self.model
        if not m.particle_count:
            return
        wp.launch(forward_step, dim=m.particle_count,
            inputs=[dt, m.gravity, m.particle_world, self.particle_q_prev,
                    state_in.particle_q, state_in.particle_qd, m.particle_inv_mass,
                    state_in.particle_f, m.particle_flags],
            outputs=[self.inertia, self.particle_displacements], device=self.device)
        self.particle_displacements.zero_()
        self.sdf_backend.prepare(dt)
        if self._guess_requested:
            self.ccd_backend.initial_guess(state_in.particle_q, self.particle_q_prev,
                state_in.particle_qd, self.inertia, dt, self.params, detector=self.trimesh_collision_detector)
        self._collision_detection_penetration_free(state_in, reset_reference=True)
        if self.iterations == 0:
            # Ordinarily the inherited color loop publishes the final iterate.
            wp.copy(state_out.particle_q, state_in.particle_q)

    def _collision_detection_penetration_free(self, current_state: State, *, reset_reference=True):
        if reset_reference:
            self.pos_prev_collision_detection.assign(current_state.particle_q)
            self.particle_displacements.zero_()
        self.contact_backend.detect(self.trimesh_collision_detector, current_state.particle_q,
                                    self._query_radius, refit=True)
        if self._rigid is not None:
            self._rigid.refresh(current_state)

    def _self_contact_is_due(self, iter_num):
        return iter_num > 0 and iter_num % int(self.params["node_box_update_count"]) == 0

    def _accumulate_particle_contact_terms(self, state_in, state_out, contacts, dt, color):
        if not self._active_color_counts[color]:
            return
        self.contact_backend.accumulate(state_in.particle_q, self.particle_q_prev, color,
                                       self.particle_forces, self.particle_hessians, dt)
        if self.model.ipc.collider_data.shape[0]:
            self.sdf_backend.accumulate(state_in.particle_q, self.particle_q_prev, color,
                self.particle_forces, self.particle_hessians, dt)

    def _solve_particle_elasticity_color(self, state_in, state_out, contacts, dt, color):
        if not self._active_color_counts[color]:
            return
        m, ipc = self.model, self.model.ipc
        arguments = [dt, m.particle_color_groups[color], self.particle_q_prev,
            state_in.particle_q, m.particle_mass, self.inertia, m.particle_flags,
            m.tri_indices, m.tri_poses, m.tri_materials, m.tri_areas,
            m.edge_indices, m.edge_rest_angle, m.edge_rest_length, m.edge_bending_properties,
            m.tet_indices, m.tet_poses, m.tet_materials, self.particle_adjacency,
            self.particle_forces, self.particle_hessians,
            ipc.pin_targets, ipc.pinned, float(self.params["kpin"]), ipc.edge_coefficient,
            ipc.tet_grad_N, ipc.tet_measure, float(self.params["damping"])]
        count = m.particle_color_groups[color].size
        if self.device.is_cuda:
            per_block = self._ipc_particles_per_block
            block_dim = 16 * per_block
            dimension = ((count + per_block - 1) // per_block) * block_dim
        else:
            block_dim, dimension = 64, count
        wp.launch(self._ipc_elasticity_kernel, dim=dimension, block_dim=block_dim,
            inputs=arguments, outputs=[self.particle_displacements, self.proposals, self.error], device=self.device)

    def _finish_particle_color(self, state_in, state_out, contacts, dt, color):
        if not self._active_color_counts[color]:
            return
        self.ccd_backend.accept(state_in.particle_q, self.proposals, color, dt,
                                center=self.pos_prev_collision_detection, radius=self._box_radius,
                                bounded=True, clip_ccd=bool(self.params["use_ccd"]))

    def _initialize_rigid_bodies(self, state_in, control, contacts, dt, update_rigid):
        if self._rigid is not None:
            self._rigid.begin(state_in, dt)

    def _solve_rigid_body_iteration(self, state_in, state_out, control, contacts, dt):
        if self._rigid is not None:
            self._rigid.iterate(state_in, dt)

    def _finalize_rigid_bodies(self, state_in, state_out, dt):
        if self._rigid is not None:
            self._rigid.finish(state_in, state_out, dt)

    def _snapshot_rigid_contact_history(self, contacts, *, force=False):
        # Our lagged friction is based on the fixed step-start position array.
        return

    def prepare_capture(self):
        """All workspaces are allocated in __init__; CUDA launches warm on capture."""
        self.check_status()

    def reset_status(self):
        if self._failed:
            raise RuntimeError("A failed solver cannot be reused; restore with a fresh instance")
        self.error.zero_()

    def check_status(self):
        if self._failed:
            raise RuntimeError("The solver is invalidated by a previous GPU failure")
        try:
            wp.synchronize_device(self.device)
            # All components share this latched error word. One download also
            # checks CCD; calling each backend would download it repeatedly.
            self.contact_backend.check()
        except Exception:
            self._failed = True
            raise


assert SolverIPC.step is SolverVBD.step
assert SolverIPC._solve_particle_iteration is SolverVBD._solve_particle_iteration
