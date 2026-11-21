"""Live IPC/friction assembly for the current combined VBD color.

The BVH candidate graph and per-color incidence are shared with ColorCcd.
Candidates are evaluated at the latest accepted positions before each material
color update. Friction anchors remain the previous physical substep positions.
Only the current color receives force/Hessian contributions. Each undirected
candidate is evaluated once in its stored orientation using the project's
full barrier derivatives and frozen-friction derivatives. Geometry and energy
arithmetic use binary64 on the GPU; assembled storage remains float32.
"""

from functools import cache
import math

import numpy as np
import warp as wp

from . import contact, dat_energies as energies


@wp.kernel(module="unique", enable_backward=False)
def _pack_primitive_ids(nt: wp.array2d[int], ee: wp.array2d[int],
                        offsets: wp.array[int], primitives: wp.array[wp.vec2i],
                        error: wp.array[int]):
    row = wp.tid()
    if error[0] == 0:
        begin, end = offsets[row], offsets[row + 1]
        if begin < 0 or end < begin or end > primitives.shape[0]:
            wp.atomic_max(error, 0, 431)
        else:
            if row < nt.shape[0]:
                if end - begin > nt.shape[1]:
                    wp.atomic_max(error, 0, 431)
                else:
                    for at in range(begin, end):
                        primitives[at] = wp.vec2i(row, nt[row, at - begin])
            else:
                edge = row - nt.shape[0]
                if edge >= ee.shape[0] or end - begin > ee.shape[1]:
                    wp.atomic_max(error, 0, 431)
                else:
                    for at in range(begin, end):
                        primitives[at] = wp.vec2i(edge, ee[edge, at - begin])


@cache
def _assembly_kernel(precision):
    @wp.kernel(module="unique", enable_backward=False)
    def assemble_color_contacts(
        q: wp.array[wp.vec3], previous_q: wp.array[wp.vec3],
        triangles: wp.array2d[int], edges: wp.array2d[int],
        nodes: wp.array[wp.vec4i], count: wp.array[int], primitives: wp.array[wp.vec2i],
        candidate_offsets: wp.array[int], pair_ids: wp.array[int], color_offsets: wp.array[int],
        colors: wp.array[int], current: wp.array[int], workers: int,
        graph_epoch: wp.array[wp.uint64], expected_epoch: wp.array[wp.uint64],
        incidence_epoch: wp.array[wp.uint64], box_epoch: wp.array[wp.uint64],
        incidence_box_epoch: wp.array[wp.uint64], graph_box_epoch: wp.array[wp.uint64],
        dt: float, radius: float, friction: float, epsilon: float,
        edge_parallel_epsilon: float, damping_bidirectional: bool,
        context: energies.EnergyContext,
        forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33],
        pending: wp.array[wp.vec2i], pending_count: wp.array[int], error: wp.array[int],
    ):
        color = current[0]
        if error[0] == 0:
            if (graph_epoch[0] != expected_epoch[0]
                    or graph_epoch[0] != incidence_epoch[0]
                    or box_epoch[0] != incidence_box_epoch[0]
                    or box_epoch[0] != graph_box_epoch[0]):
                wp.atomic_max(error, 0, 432)
            elif color < 0 or color + 1 >= color_offsets.shape[0]:
                wp.atomic_max(error, 0, 431)
            else:
                begin, end = color_offsets[color], color_offsets[color + 1]
                if begin < 0 or end < begin or end > pair_ids.shape[0]:
                    wp.atomic_max(error, 0, 431)
                else:
                    for at in range(begin + wp.tid(), end, workers):
                        pair = pair_ids[at]
                        if pair < 0 or pair >= count[0] or pair >= nodes.shape[0]:
                            wp.atomic_max(error, 0, 431)
                        else:
                            corners = nodes[pair]
                            active_corner = int(-1)
                            active_count = int(0)
                            for corner in range(4):
                                if colors[corners[corner]] == color:
                                    active_corner = corner
                                    active_count += 1
                            if active_count != 1:
                                wp.atomic_max(error, 0, 433)
                            else:
                                kind = int(1)
                                if pair < candidate_offsets[q.shape[0]]:
                                    kind = 0
                                # The project packet uses exactly the shared
                                # candidate order: [point,a,b,c] or [a,b,c,d].
                                # It includes b'(d) * Hessian(d), which the
                                # native frozen-normal rank-one block omits.
                                result = contact.evaluate_self(
                                    q, previous_q, corners, kind, context.parameters)
                                status = int(result[48])
                                if status == 2:
                                    slot = wp.atomic_add(pending_count, 0, 1)
                                    if slot < pending.shape[0]:
                                        pending[slot] = wp.vec2i(pair, active_corner)
                                    else:
                                        wp.atomic_max(error, 0, 12)
                                else:
                                    _scatter_contact(result, corners[active_corner], active_corner,
                                                     forces, hessians, error)

    contact.configure_module(assemble_color_contacts)
    return assemble_color_contacts


@wp.func
def _scatter_contact(result: contact.Result, node: int, role: int,
                     forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33], error: wp.array[int]):
    status = int(result[48])
    if status != 0:
        wp.atomic_max(error, 0, 100 + status)
    elif result[49] != 0.0:
        offset = 12 * role
        force = wp.vec3(result[offset], result[offset + 1], result[offset + 2])
        hessian = wp.mat33(0.0)
        for row in range(3):
            for column in range(3):
                hessian[row, column] = result[offset + 3 + 3 * row + column]
        wp.atomic_add(forces, node, force)
        wp.atomic_add(hessians, node, hessian)


@wp.kernel(module="unique", enable_backward=False)
def _resolve_color_contacts(
    q: wp.array[wp.vec3], previous_q: wp.array[wp.vec3],
    nodes: wp.array[wp.vec4i], candidate_offsets: wp.array[int], params: contact.Parameters,
    words: wp.array[wp.uint32], slots: wp.array[int], workers: int,
    pending: wp.array[wp.vec2i], pending_count: wp.array[int],
    forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33], error: wp.array[int],
):
    worker = wp.tid()
    if error[0] == 0:
        for slot in range(worker, pending_count[0], workers):
            request = pending[slot]
            pair, role = request[0], request[1]
            corners = nodes[pair]
            kind = int(pair >= candidate_offsets[q.shape[0]])
            result = contact.evaluate_self_exact(q, previous_q, corners, kind, params, words, slots, worker)
            _scatter_contact(result, corners[role], role, forces, hessians, error)


contact.configure_module(_resolve_color_contacts, include_filter=False)


class ColorContactForces:
    """Preallocated live contact forces on the same pairs as color CCD.

    Refresh immediately after ``ColorCcd.refresh``. ``assemble`` must run before
    the material solve for every color and uses the substep-start friction anchor,
    never ColorCcd's changing per-color reference positions.
    """

    def __init__(self, model, graph, color_ccd, error, *, k_barrier,
                 precision="float32", edge_parallel_epsilon=1e-5,
                 damping_bidirectional=True):
        if color_ccd.graph is not graph or graph.model is not model:
            raise ValueError("Color contact forces require matching color CCD and candidate graph")
        if precision not in ("float32", "float64"):
            raise ValueError("Contact scalar precision must be float32 or float64")
        represented_k = np.float32(k_barrier)
        if (isinstance(k_barrier, bool) or not math.isfinite(k_barrier) or k_barrier < 0
                or not np.isfinite(represented_k) or (k_barrier > 0 and represented_k == 0)):
            raise ValueError("Barrier stiffness must be nonnegative and representable in float32")
        if not math.isfinite(edge_parallel_epsilon) or edge_parallel_epsilon < 0:
            raise ValueError("Edge parallel epsilon must be finite and nonnegative")
        self.model, self.graph, self.color_ccd, self.error = model, graph, color_ccd, error
        self.device, self.n = model.device, model.particle_count
        self.precision, self.k_barrier = "float64", float(represented_k)
        # ``precision`` is retained for constructor compatibility with the
        # native energy workspace. The project evaluator is deliberately not
        # rewritten as float32: its CPU-derived geometric Hessians use double.
        self.configuration = {
            "energy_model": "project",
            "geometry_and_derivative_precision": "float64",
            "assembled_storage_precision": "float32",
            "normal_hessian": "full geometric diagonal blocks",
            "friction": "project frozen friction gradient and Hessian",
            "exact_geometry": True,
        }
        self.edge_parallel_epsilon = float(edge_parallel_epsilon)
        self.damping_bidirectional = bool(damping_bidirectional)
        self.primitives = wp.empty(graph.capacity, wp.vec2i, device=self.device)
        self.pending = wp.empty(graph.capacity, wp.vec2i, device=self.device)
        self.pending_count = wp.zeros(1, int, device=self.device)
        self.exact_workers = min(16, self.device.sm_count) if self.device.is_cuda else 1
        self.words, self.slots = contact.allocate_exact_storage(self.device, self.exact_workers)
        self._graph_epoch = wp.zeros(1, wp.uint64, device=self.device)
        self.context = energies.EnergyContext()
        self.context.error = error
        self.kernel = _assembly_kernel(precision)
        self._ready = False

    def refresh(self):
        graph, d = self.graph, self.device
        if not graph._built or not self.color_ccd._ready:
            raise RuntimeError("Refresh color CCD before its contact-force workspace")
        rows = self.n + graph.edge_count
        if rows:
            wp.launch(_pack_primitive_ids, rows,
                      [graph.node_triangle_pairs, graph.edge_edge_pairs, graph.candidate_offsets,
                       self.primitives, self.error], device=d)
        wp.copy(self._graph_epoch, graph.ccd_epoch)
        self._ready = True

    def assemble(self, q, previous_q, dt, d_hat, friction_mu, friction_epsilon,
                 forces, hessians):
        if not self._ready:
            raise RuntimeError("Refresh contact-force candidates before assembly")
        for array, dtype in ((q, wp.vec3), (previous_q, wp.vec3),
                             (forces, wp.vec3), (hessians, wp.mat33)):
            if array.dtype != dtype or array.shape != (self.n,) or array.device != self.device:
                raise TypeError("Color contact forces require complete model arrays")
        # These are host parameter/struct assignments during ordinary execution
        # or graph capture, not device downloads or device allocations.
        self.context.parameters = contact.parameters(
            d_hat=float(np.float32(d_hat)), k_barrier=self.k_barrier,
            dt=float(np.float32(dt)), friction=float(np.float32(friction_mu)),
            eps_v=float(np.float32(friction_epsilon)))
        forces.zero_()
        hessians.zero_()
        if self.n == 0 or self.graph.capacity == 0:
            return
        g, c = self.graph, self.color_ccd
        self.pending_count.zero_()
        wp.launch(self.kernel, c.workers,
                  [q, previous_q, self.model.tri_indices, self.model.edge_indices,
                   g.candidate_nodes, g.candidate_count, self.primitives,
                   g.candidate_offsets, c.pair_ids, c.offsets, g.colors, g.current_color,
                   c.workers, g.ccd_epoch, self._graph_epoch,
                   c._graph_epoch, c.boxes.ccd_epoch, c._box_epoch, g.ccd_source_box_epoch,
                   dt, d_hat, friction_mu, friction_epsilon, self.edge_parallel_epsilon,
                   self.damping_bidirectional, self.context, forces, hessians,
                   self.pending, self.pending_count, self.error],
                  device=self.device, block_dim=64)
        wp.launch(_resolve_color_contacts, self.exact_workers,
                  [q, previous_q, g.candidate_nodes, g.candidate_offsets, self.context.parameters,
                   self.words, self.slots, self.exact_workers, self.pending, self.pending_count,
                   forces, hessians, self.error], device=self.device, block_dim=1)


energies.configure_module(_pack_primitive_ids)
