"""CPU-style GS dependencies and our equations in the imported GPU timestep.

Newton owns the float32 Model/State, material adjacency, BVHs and velocity
publication. Deformable GS rebuilds contact-aware colors at box checkpoints;
the local material solve retains the cooperative 16-lane GPU layout.
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
from .gs import MovementBoxes
from .sdf import SdfBackend


class SolverIPC(SolverVBD):
    """Our GS/IPC/CCD model plugged into the authors' concrete GPU solver.

    Adaptive movement boxes and contact-independent colors follow CPU GS.
    Float32 state and tree reductions can change the numeric trajectory. Mixed
    rigid scenes retain the inherited particle schedule and separate rigid
    block order until a unified mixed dependency graph is implemented.
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
        self._contact_aware_gs = bool(p.get("contact_aware_gs", True)) and model.body_count == 0 and model.particle_count > 0
        self._box_radius = float(p["node_box_max"])
        self._adaptive_node_boxes = bool(p.get("adaptive_node_boxes", True))
        self.movement_boxes = MovementBoxes(model, p["node_box_min"], p["node_box_max"])
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
            grouped_precision=p.get("gs_contact_precision", "float32") if self._contact_aware_gs else "float64",
            d_hat=float(p["d_hat"]), k_barrier=float(p["k_barrier"]),
            friction=float(p["friction_coefficient"]), eps_v=float(p["friction_velocity_epsilon"]),
            exact_geometry=bool(p["exact_computation_fallback"]), node_body=model.ipc.node_body, error=self.error,
            track_query_epoch=bool(self._contact_aware_gs or p.get("use_colored_ccd_guess", False) or
                                   p.get("use_colored_ccd_guess_level_based", False)))
        self.ccd_backend = CcdBackend(self.contact_backend, node_box_radius=self._box_radius,
            exact_geometry=bool(p["exact_computation_fallback"]), error=self.error, group_query_block_dim=64)
        self.ccd_backend.single_vertex_filter = bool(p.get("gs_ccd_filter", True))
        if not self.ccd_backend.exact_step_bound:
            from .ccd import resolve_cached
            self.ccd_backend._resolve = resolve_cached
        flags = model.particle_flags.numpy()
        has_dynamic_particles = bool(np.any((flags & 1) & (model.particle_mass.numpy() > 0)))
        self._guess_requested = has_dynamic_particles and bool(p.get("use_colored_ccd_guess", False) or
            p.get("use_colored_ccd_guess_level_based", False) or
            (p.get("use_ccd_guess", False) and not model.ipc.payload.get("uses_general_solver", False) and model.body_count == 0))
        if self._guess_requested:
            self.ccd_backend.prepare_initial_guess(self.trimesh_collision_detector, self.params, use_group_queries=True,
                independent_colors=bool(p.get("use_colored_ccd_guess", False))
                    and not p.get("use_colored_ccd_guess_level_based", False) and model.body_count == 0)
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
        self._gs_schedule = None
        self._box_filter = None
        if self._contact_aware_gs:
            from .gs import GsSchedule
            self._gs_schedule = GsSchedule(self.contact_backend)
            if self._adaptive_node_boxes and p.get("gs_prune_box_contacts", True):
                from .movement_box_queries import GsBoxEpochFilter
                self._box_filter = GsBoxEpochFilter(self.contact_backend)
        self._rigid = None
        if model.body_count:
            from .body_solver import BodySolver
            self._rigid = BodySolver(self)
            self._query_radius = max(self._query_radius, self._rigid.query_radius)

    # The public timestep is inherited; these hooks supply our model and GS order.
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
        if self._adaptive_node_boxes:
            self.movement_boxes.begin(state_in.particle_qd, dt)
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
        if self._adaptive_node_boxes:
            self.movement_boxes.anchor(state_in.particle_q)
        self._collision_detection_penetration_free(state_in, reset_reference=True)
        if self.iterations == 0:
            # Ordinarily the inherited color loop publishes the final iterate.
            wp.copy(state_out.particle_q, state_in.particle_q)

    def _collision_detection_penetration_free(self, current_state: State, *, reset_reference=True):
        if reset_reference:
            self.pos_prev_collision_detection.assign(current_state.particle_q)
            self.particle_displacements.zero_()
        if self._adaptive_node_boxes and self._rigid is None:
            self.contact_backend.detect_boxes(self.trimesh_collision_detector,
                current_state.particle_q, self.movement_boxes.radii, float(self.params["d_hat"]))
        else:
            # Rigid analytic motion retains its own conservative scalar reach.
            self.contact_backend.detect(self.trimesh_collision_detector, current_state.particle_q,
                                        self._query_radius, refit=True)
        if self._rigid is not None:
            self._rigid.refresh(current_state)
        if self._box_filter is not None:
            from .contact import parameters
            # A whole-box proof removes only interactions that stay inactive
            # throughout this epoch. All uncertain pairs remain dependencies.
            self._box_filter.apply(current_state.particle_q, self.particle_q_prev,
                self.contact_backend.swept_lower, self.contact_backend.swept_upper,
                parameters(dt=float(self.params["dt"]), **self.contact_backend.options))
        if self._gs_schedule is not None:
            self._gs_schedule.rebuild()

    def _self_contact_is_due(self, iter_num):
        return iter_num > 0 and iter_num % int(self.params["node_box_update_count"]) == 0

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        gs = self._gs_schedule
        if gs is None:
            return super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        self.particle_forces.zero_()
        self.particle_hessians.zero_()

        def color_body():
            self.contact_backend.accumulate(state_in.particle_q, self.particle_q_prev, -1,
                self.particle_forces, self.particle_hessians, dt,
                colors=gs.colors, current_color=gs.current_color, query_view=gs.query_view)
            if self.model.ipc.collider_data.shape[0]:
                self.sdf_backend.accumulate(state_in.particle_q, self.particle_q_prev, -1,
                    self.particle_forces, self.particle_hessians, dt,
                    colors=gs.colors, current_color=gs.current_color)
            gs.solve_elasticity(self, state_in, dt)
            self.ccd_backend.accept(state_in.particle_q, self.proposals, -1, dt,
                center=self.pos_prev_collision_detection, radius=self._box_radius,
                radii=self.movement_boxes.radii if self._adaptive_node_boxes else None,
                colors=gs.colors, current_color=gs.current_color, query_view=gs.query_view,
                independent=True, bounded=True, clip_ccd=bool(self.params["use_ccd"]))

        gs.sweep(color_body)
        wp.copy(state_out.particle_q, state_in.particle_q)

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
            # The stock-scene bridge supplies own-energy bending coefficients
            # mapped to the imported DAT scene's physical hinge stiffness.
            m.edge_indices, m.edge_rest_angle, m.edge_rest_length,
            getattr(ipc, "stock_energy_edge_properties", m.edge_bending_properties),
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
                                radii=self.movement_boxes.radii if self._adaptive_node_boxes else None,
                                bounded=True, clip_ccd=bool(self.params["use_ccd"]))

    def _finalize_particles(self, state_out: State, dt):
        if self._adaptive_node_boxes:
            self.movement_boxes.finish(state_out.particle_q)
        super()._finalize_particles(state_out, dt)

    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, world_mask=world_mask, flags=flags)
        self.movement_boxes.reset(self._normalize_reset_world_mask(world_mask))

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
