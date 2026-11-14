"""GPU GS movement history, contact-independent scheduling and local updates.

Movement budgets follow CPU displacement history. Every material or movement-box
contact hyperedge contributes its complete conflict clique; device-only repair
converges to canonical CPU greedy colors before any GS update. The local solve
reuses the existing energy laws and cooperative float32 material reductions.
"""

import math
from functools import cache

import warp as wp
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from . import energy, kernels
from .contact import decode_candidate
from .parallel_colors import ParallelColors
from .parallel_levels import _start_loop, _validate_rows
from .query_groups import QueryGroups


@wp.kernel
def _box_begin(velocity: wp.array[wp.vec3], previous: wp.array[wp.float64],
           dt: wp.float64, minimum: wp.float64, maximum: wp.float64,
           radii: wp.array[wp.float64]):
    i = wp.tid()
    v = velocity[i]
    speed = wp.sqrt(wp.float64(v[0]) * wp.float64(v[0])
                    + wp.float64(v[1]) * wp.float64(v[1])
                    + wp.float64(v[2]) * wp.float64(v[2]))
    radii[i] = wp.clamp(wp.float64(1.2) * wp.max(previous[i], dt * speed), minimum, maximum)


@wp.kernel
def _box_finish(position: wp.array[wp.vec3], start: wp.array[wp.vec3],
            previous: wp.array[wp.float64]):
    i = wp.tid()
    end = position[i]
    anchor = start[i]
    dx = wp.float64(end[0]) - wp.float64(anchor[0])
    dy = wp.float64(end[1]) - wp.float64(anchor[1])
    dz = wp.float64(end[2]) - wp.float64(anchor[2])
    previous[i] = wp.sqrt(dx * dx + dy * dy + dz * dz)


@wp.kernel
def _box_reset(previous: wp.array[wp.float64], radii: wp.array[wp.float64],
           maximum: wp.float64, worlds: wp.array[int], world_count: int,
           world_mask: wp.array[wp.bool], all_worlds: bool):
    i = wp.tid()
    world = worlds[i]
    selected = all_worlds
    if not all_worlds:
        if world == -1:
            world = world_count
        selected = world_mask[world]
    if selected:
        previous[i] = maximum
        radii[i] = maximum


class MovementBoxes:
    """Preallocated per-node budgets; every timestep operation stays on device."""

    def __init__(self, model, minimum, maximum):
        self.device = model.device
        self.count = model.particle_count
        self.minimum, self.maximum = float(minimum), float(maximum)
        if not (math.isfinite(self.minimum) and math.isfinite(self.maximum)
                and 1.0e-10 < self.minimum <= self.maximum):
            raise ValueError("Movement-box radii must be finite and larger than the CCD inset")
        self.radii = wp.full(self.count, self.maximum, wp.float64, device=self.device)
        # The CPU workspace also starts with the maximum so the first solve
        # has its full movement budget before a displacement history exists.
        self.previous_displacement = wp.full(self.count, self.maximum, wp.float64, device=self.device)
        self.start = wp.empty_like(model.particle_q)
        self._worlds = model.particle_world
        self._world_count = model.world_count

    def begin(self, velocity, dt):
        """Freeze radii from input velocity, before the predictor changes positions."""
        if self.count:
            wp.launch(_box_begin, self.count,
                [velocity, self.previous_displacement, float(dt), self.minimum, self.maximum, self.radii],
                device=self.device)

    def anchor(self, positions):
        """Anchor history after the predictor; later box checkpoints do not call this."""
        wp.copy(self.start, positions)

    def finish(self, positions):
        if self.count:
            wp.launch(_box_finish, self.count,
                [positions, self.start, self.previous_displacement], device=self.device)

    def reset(self, world_mask=None):
        """Discard history for the same worlds reset by the parent solver."""
        if self.count:
            wp.launch(_box_reset, self.count,
                [self.previous_displacement, self.radii, self.maximum, self._worlds,
                 self._world_count, world_mask, world_mask is None], device=self.device)


@wp.func_native(r"""
    unsigned long long forbidden[2]={0,0};
    for(uint64_t at=rows.data[vertex];at<rows.data[vertex+1];++at) {
        const int neighbor=adjacency.data[at];
        if(neighbor>=vertex)break;
        const int color=colors.data[neighbor];
        if(color>=0&&color<128)forbidden[color/64]|=1ull<<(color%64);
    }
    for(int word=0;word<2;++word) {
        const unsigned long long available=~forbidden[word];
        if(available) {
#if defined(__CUDA_ARCH__)
            return 64*word+__ffsll(static_cast<long long>(available))-1;
#else
            for(int bit=0;bit<64;++bit)if(available&(1ull<<bit))return 64*word+bit;
#endif
        }
    }
    // Complete graphs and arbitrary valid seeds retain an uncapped mex.
    for(int color=128;color<colors.shape[0];++color) {
        bool available=true;
        for(uint64_t at=rows.data[vertex];at<rows.data[vertex+1];++at) {
            const int neighbor=adjacency.data[at];
            if(neighbor>=vertex)break;
            if(colors.data[neighbor]==color){available=false;break;}
        }
        if(available)return color;
    }
    return -1;
""")
def _color_least_free(rows: wp.array[wp.uint64], adjacency: wp.array[int], colors: wp.array[int], vertex: int) -> int: ...


@wp.kernel
def _color_validate_seed(colors: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    if colors[vertex] < -1 or colors[vertex] >= colors.shape[0]:
        wp.atomic_max(error, 0, 391)


@wp.kernel
def _color_repair(rows: wp.array[wp.uint64], adjacency: wp.array[int], source: wp.array[int],
            target: wp.array[int], changed: wp.array[int], track: int, error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0:
        color = _color_least_free(rows, adjacency, source, vertex)
        if color < 0 or color >= source.shape[0]:
            wp.atomic_max(error, 0, 392)
        else:
            target[vertex] = color
            if track != 0 and color != source[vertex]:
                wp.atomic_or(changed, 0, 1)


@wp.kernel
def _color_repair_next(changed: wp.array[int], condition: wp.array[int], iterations: wp.array[int],
          maximum: int, error: wp.array[int]):
    iterations[0] += 1
    condition[0] = int(changed[0] != 0 and error[0] == 0)
    if condition[0] != 0 and iterations[0] > maximum:
        wp.atomic_max(error, 0, 393)
        condition[0] = 0


@wp.kernel
def _color_publish(colors: wp.array[int], trial_order: wp.array[int], metadata: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0:
        color = colors[vertex]
        if color < 0 or color >= colors.shape[0]:
            wp.atomic_max(error, 0, 392)
        else:
            wp.atomic_max(metadata, 0, color + 1)
            wp.atomic_max(metadata, 1, color + 1)
            trial_order[vertex] = vertex
        if vertex == 0:
            metadata[2] = colors.shape[0]


class WarmColors:
    """Persistent two-buffer Jacobi repair, with no host convergence readback."""
    def __init__(self, graph):
        self.graph, self.n, self.device = graph, graph.n, graph.device
        self.temporary = wp.empty(self.n, int, device=self.device)
        self.changed = wp.zeros(1, int, device=self.device)
        self.condition = wp.zeros(1, int, device=self.device)
        self.iterations = wp.zeros(1, int, device=self.device)

    def apply(self, contacts):
        g, n, d = self.graph, self.n, self.device
        if n == 0:
            raise ValueError("Empty model is not a supported GS coloring epoch")
        wp.launch(_validate_rows, n, [g.rows, g.adjacency, contacts.error], device=d)
        wp.launch(_color_validate_seed, n, [g.colors, contacts.error], device=d)
        g.metadata.zero_()
        wp.launch(_start_loop, 1, [self.condition, self.iterations, contacts.error], device=d)
        def body():
            wp.launch(_color_repair, n, [g.rows, g.adjacency, g.colors, self.temporary, self.changed, 0, contacts.error], device=d)
            self.changed.zero_()
            wp.launch(_color_repair, n, [g.rows, g.adjacency, self.temporary, g.colors, self.changed, 1, contacts.error], device=d)
            # If pass two changes nothing, F(temporary)==temporary: the
            # published result is a fixed point, even if pass one changed it.
            wp.launch(_color_repair_next, 1, [self.changed, self.condition, self.iterations, n + 1, contacts.error], device=d)
        wp.capture_while(self.condition, body)
        wp.launch(_color_publish, n, [g.colors, g.trial_order, g.metadata, contacts.error], device=d)


@wp.func
def _edge(vertex: int, neighbor: int, stride: int, bits: wp.array[wp.uint32], error: wp.array[int]):
    n = bits.shape[0] // stride
    if vertex < 0 or vertex >= n or neighbor < 0 or neighbor >= n:
        wp.atomic_max(error, 0, 381)
    elif neighbor < vertex:
        # Greedy coloring only consumes lower-ID neighbors. Storing each edge
        # once also halves the largest possible CSR allocation.
        wp.atomic_or(bits, vertex * stride + neighbor // 32, wp.uint32(1) << wp.uint32(neighbor % 32))


@wp.kernel
def _material_edges(triangles: wp.array2d[int], edges: wp.array2d[int], tets: wp.array2d[int],
                    stride: int, bits: wp.array[wp.uint32], error: wp.array[int]):
    at = wp.tid()
    if error[0] == 0:
        if at < triangles.shape[0]:
            for a in range(3):
                for b in range(3):
                    _edge(triangles[at, a], triangles[at, b], stride, bits, error)
        elif at < triangles.shape[0] + edges.shape[0]:
            hinge = at - triangles.shape[0]
            # Boundary edges contain an absent apex and no bending energy.
            if edges[hinge, 0] >= 0 and edges[hinge, 1] >= 0:
                for a in range(4):
                    for b in range(4):
                        _edge(edges[hinge, a], edges[hinge, b], stride, bits, error)
        else:
            tet = at - triangles.shape[0] - edges.shape[0]
            if tet < tets.shape[0]:
                for a in range(4):
                    for b in range(4):
                        _edge(tets[tet, a], tets[tet, b], stride, bits, error)


@wp.kernel
def _contact_edges(triangles: wp.array2d[int], edges: wp.array2d[int],
                   info: TriMeshCollisionInfo, ids: wp.array[int], count: wp.array[int],
                   workers: int, stride: int, bits: wp.array[wp.uint32], error: wp.array[int]):
    if error[0] == 0:
        if count[0] < 0 or count[0] > ids.shape[0]:
            wp.atomic_max(error, 0, 382)
        else:
            for at in range(wp.tid(), count[0], workers):
                nodes, kind = decode_candidate(ids[at], triangles, edges, info)
                for a in range(4):
                    for b in range(4):
                        _edge(nodes[a], nodes[b], stride, bits, error)


@wp.func_native(r"""
    int count=0;
    for(int word=0;word<stride;++word) {
        uint32_t value=bits.data[vertex*stride+word];
        while(value){value&=value-1;++count;}
    }
    return count;
""")
def _degree(bits: wp.array[wp.uint32], stride: int, vertex: int) -> int: ...


@wp.kernel
def _degrees(bits: wp.array[wp.uint32], stride: int, counts: wp.array[int]):
    vertex = wp.tid()
    if vertex + 1 < counts.shape[0]:
        counts[vertex] = _degree(bits, stride, vertex)
    else:
        counts[vertex] = 0


@wp.func_native(r"""
    int at=offsets.data[vertex];
    for(int word=0;word<stride;++word) {
        uint32_t value=bits.data[vertex*stride+word];
        for(int bit=0;bit<32;++bit)if(value&(uint32_t(1)<<bit)) {
            if(at>=adjacency.shape[0]){error.data[0]=382;return;}
            adjacency.data[at++]=32*word+bit;
        }
    }
""")
def _fill_row(bits: wp.array[wp.uint32], stride: int, vertex: int, offsets: wp.array[int],
              adjacency: wp.array[int], error: wp.array[int]): ...


@wp.kernel
def _rows(bits: wp.array[wp.uint32], stride: int, offsets: wp.array[int], rows: wp.array[wp.uint64],
          adjacency: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    rows[vertex] = wp.uint64(offsets[vertex])
    if vertex + 1 < rows.shape[0] and error[0] == 0:
        _fill_row(bits, stride, vertex, offsets, adjacency, error)


@wp.kernel
def _count_colors(colors: wp.array[int], counts: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0:
        color = colors[vertex]
        if color < 0 or color >= colors.shape[0]:
            wp.atomic_max(error, 0, 383)
        else:
            wp.atomic_add(counts, color, 1)


@wp.kernel
def _pack_colors(colors: wp.array[int], offsets: wp.array[int], cursors: wp.array[int],
                 order: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0:
        color = colors[vertex]
        at = offsets[color] + wp.atomic_add(cursors, color, 1)
        if at < offsets[color] or at >= offsets[color + 1] or at >= order.shape[0]:
            wp.atomic_max(error, 0, 383)
        else:
            order[at] = vertex


@wp.kernel
def _begin_color(current: wp.array[int], condition: wp.array[int], metadata: wp.array[int], error: wp.array[int]):
    current[0] = 0
    condition[0] = int(metadata[0] > 0 and error[0] == 0)


@wp.kernel
def _next_color(current: wp.array[int], condition: wp.array[int], metadata: wp.array[int], error: wp.array[int]):
    current[0] += 1
    condition[0] = int(current[0] < metadata[0] and error[0] == 0)


@wp.kernel
def _check_contacts(triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                    ids: wp.array[int], count: wp.array[int], colors: wp.array[int],
                    workers: int, error: wp.array[int]):
    if error[0] == 0:
        for at in range(wp.tid(), count[0], workers):
            nodes, kind = decode_candidate(ids[at], triangles, edges, info)
            for a in range(4):
                for b in range(a):
                    if nodes[a] != nodes[b] and colors[nodes[a]] == colors[nodes[b]]:
                        wp.atomic_max(error, 0, 384)


@cache
def make_dynamic_elasticity(include_triangles: bool, include_tets: bool, tiled: bool = True):
    """Reuse the existing material laws and half-warp reduction/3x3 solve."""
    @wp.kernel(module="unique", enable_backward=False)
    def solve(
        dt: float, order: wp.array[int], offsets: wp.array[int], current: wp.array[int],
        pos_prev: wp.array[wp.vec3], pos: wp.array[wp.vec3], mass: wp.array[float],
        inertia: wp.array[wp.vec3], flags: wp.array[int],
        triangles: wp.array2d[int], tri_poses: wp.array[wp.mat22], tri_materials: wp.array2d[float], tri_areas: wp.array[float],
        edges: wp.array2d[int], rest_angles: wp.array[float], rest_lengths: wp.array[float], bending: wp.array2d[float],
        tets: wp.array2d[int], tet_poses: wp.array[wp.mat33], tet_materials: wp.array2d[float], adjacency: kernels.MeshAdjacencyData,
        contact_forces: wp.array[wp.vec3], contact_hessians: wp.array[wp.mat33],
        pin_targets: wp.array[wp.vec3], pinned: wp.array[int], kpin: float, bending_coefficients: wp.array[float],
        tet_grad_N: wp.array2d[wp.vec3], tet_measures: wp.array[float], damping: float,
        displacements: wp.array[wp.vec3], proposals: wp.array[wp.vec3], error: wp.array[int],
    ):
        tid = wp.tid()
        slot = tid
        lane = int(0)
        step = int(1)
        if wp.static(tiled):
            slot = tid // 16
            lane = tid % 16
            step = 16
        group = current[0]
        begin = offsets[group]
        end = offsets[group + 1]
        valid = slot < end - begin and error[0] == 0
        particle = int(0)
        dynamic = False
        if valid:
            particle = order[begin + slot]
            dynamic = flags[particle] & 1 != 0 and mass[particle] > 0.0
        force = wp.vec3(0.0)
        hessian = wp.mat33(0.0)
        if dynamic:
            if wp.static(include_triangles):
                face_count = int(0)
                if triangles.shape[0] > 0:
                    face_count = kernels.get_vertex_num_adjacent_faces(adjacency, particle)
                face_at = int(lane)
                while face_at < face_count:
                    face, role = kernels.get_vertex_adjacent_face_id_order(adjacency, particle, face_at)
                    if tri_materials[face, 0] > 0.0 or tri_materials[face, 1] > 0.0:
                        value = energy.triangle(pos[triangles[face, 0]], pos[triangles[face, 1]], pos[triangles[face, 2]],
                            tri_poses[face], tri_areas[face], tri_materials[face, 0], tri_materials[face, 1], role)
                        f, h = energy.unpack(value)
                        force += f
                        hessian += h
                    face_at += step
                edge_count = int(0)
                if edges.shape[0] > 0:
                    edge_count = kernels.get_vertex_num_adjacent_edges(adjacency, particle)
                edge_at = int(lane)
                while edge_at < edge_count:
                    edge, role = kernels.get_vertex_adjacent_edge_id_order(adjacency, particle, edge_at)
                    if bending_coefficients[edge] > 0.0 and bending[edge, 0] > 0.0:
                        native_role = role + 2
                        if role >= 2:
                            native_role = role - 2
                        value = energy.hinge(pos[edges[edge, 2]], pos[edges[edge, 3]], pos[edges[edge, 0]], pos[edges[edge, 1]],
                            bending_coefficients[edge], rest_angles[edge], bending[edge, 0], native_role)
                        f, h = energy.unpack(value)
                        force += f
                        hessian += h
                    edge_at += step
            if wp.static(include_tets):
                tet_count = int(0)
                if tets.shape[0] > 0:
                    tet_count = kernels.get_vertex_num_adjacent_tets(adjacency, particle)
                tet_at = int(lane)
                while tet_at < tet_count:
                    tet, role = kernels.get_vertex_adjacent_tet_id_order(adjacency, particle, tet_at)
                    if tet_materials[tet, 0] > 0.0 or tet_materials[tet, 1] > 0.0:
                        value = energy.solid(pos[tets[tet, 0]], pos[tets[tet, 1]], pos[tets[tet, 2]], pos[tets[tet, 3]],
                            tet_poses[tet], tet_grad_N[tet, role], tet_measures[tet], tet_materials[tet, 0], tet_materials[tet, 1])
                        f, h = energy.unpack(value)
                        force += f
                        hessian += h
                    tet_at += step
        total_f = force
        total_h = hessian
        if wp.static(tiled):
            total_f = wp.vec3(kernels._half_reduce(force[0]), kernels._half_reduce(force[1]), kernels._half_reduce(force[2]))
            total_h = wp.mat33(kernels._half_reduce(hessian[0, 0]), kernels._half_reduce(hessian[0, 1]), kernels._half_reduce(hessian[0, 2]),
                kernels._half_reduce(hessian[1, 0]), kernels._half_reduce(hessian[1, 1]), kernels._half_reduce(hessian[1, 2]),
                kernels._half_reduce(hessian[2, 0]), kernels._half_reduce(hessian[2, 1]), kernels._half_reduce(hessian[2, 2]))
        if lane == 0 and valid:
            if dynamic:
                kernels._finish_particle(particle, dt, pos, mass, inertia, contact_forces, contact_hessians,
                    pin_targets, pinned, kpin, damping, total_f, total_h, displacements, proposals, error)
            else:
                displacements[particle] = wp.vec3(0.0)
                proposals[particle] = pos[particle]
    energy.configure_module(solve)
    return solve


class GsSchedule:
    """Persistent contact-aware color graph and its device-controlled sweep.

    Construct before graph capture, call ``rebuild`` immediately after each box
    candidate refresh, and call ``sweep(body)`` once per GS iteration. The body
    reads ``current_color`` and ``colors`` and uses ``query_view`` for contacts.
    """
    def __init__(self, contacts, *, warm_colors=True):
        self.contacts, self.model, self.device = contacts, contacts.model, contacts.device
        self.n = self.model.particle_count
        if self.n <= 0:
            raise ValueError("Contact-aware particle GS needs a nonempty model")
        if not getattr(contacts, "track_query_epoch", False):
            raise ValueError("Contact-aware GS requires candidate epoch tracking")
        self.stride = (self.n + 31) // 32
        material_edges = 3 * self.model.tri_count + 6 * self.model.edge_count + 6 * self.model.tet_count
        capacity = min(self.n * (self.n - 1) // 2, material_edges + 6 * contacts.capacity)
        if self.n * self.stride >= 2**31 or capacity >= 2**31:
            raise ValueError("Contact-aware GS graph exceeds its int32 GPU index capacity")
        self.material_bits = wp.zeros(self.n * self.stride, wp.uint32, device=self.device)
        self.bits = wp.empty_like(self.material_bits)
        self.degrees = wp.zeros(self.n + 1, int, device=self.device)
        self.scan = wp.empty_like(self.degrees)
        self.rows = wp.empty(self.n + 1, wp.uint64, device=self.device)
        self.adjacency = wp.empty(capacity, int, device=self.device)
        self.colors = wp.full(self.n, -1, int, device=self.device)
        self.trial_order = wp.empty(self.n, int, device=self.device)
        self.metadata = wp.zeros(3, int, device=self.device)
        self.counts = wp.zeros(self.n + 1, int, device=self.device)
        self.offsets = wp.zeros(self.n + 1, int, device=self.device)
        self.cursors = wp.zeros(self.n + 1, int, device=self.device)
        self.order = wp.empty(self.n, int, device=self.device)
        self.current_color = wp.zeros(1, int, device=self.device)
        self.condition = wp.zeros(1, int, device=self.device)
        self.parallel_colors = WarmColors(self) if warm_colors else ParallelColors(self)
        self.query_groups = QueryGroups(contacts)
        if not self.query_groups.enabled:
            raise ValueError("Contact-aware GS candidate grouping exceeds its int32 GPU index capacity")
        self.query_view = None
        primitive_count = self.model.tri_count + self.model.edge_count + self.model.tet_count
        if primitive_count:
            wp.launch(_material_edges, primitive_count,
                [self.model.tri_indices, self.model.edge_indices, self.model.tet_indices,
                 self.stride, self.material_bits, contacts.error], device=self.device)
        self._elasticity = make_dynamic_elasticity(self.model.tri_count > 0 or self.model.edge_count > 0,
                                                   self.model.tet_count > 0, self.device.is_cuda)

    def rebuild(self):
        c, m = self.contacts, self.model
        wp.copy(self.bits, self.material_bits)
        if c.capacity:
            wp.launch(_contact_edges, c.fast_workers,
                [m.tri_indices, m.edge_indices, c.info, c.candidate_ids, c.candidate_count,
                 c.fast_workers, self.stride, self.bits, c.error], device=self.device)
        wp.launch(_degrees, self.n + 1, [self.bits, self.stride, self.degrees], device=self.device)
        wp.utils.array_scan(self.degrees, self.scan, inclusive=False)
        wp.launch(_rows, self.n + 1,
            [self.bits, self.stride, self.scan, self.rows, self.adjacency, c.error], device=self.device)
        self.parallel_colors.apply(c)
        self.counts.zero_()
        self.cursors.zero_()
        wp.launch(_count_colors, self.n, [self.colors, self.counts, c.error], device=self.device)
        wp.utils.array_scan(self.counts, self.offsets, inclusive=False)
        wp.launch(_pack_colors, self.n, [self.colors, self.offsets, self.cursors, self.order, c.error], device=self.device)
        self.query_groups.build(self.colors)
        self.query_view = self.query_groups.view(self.current_color)

    def validate_independence(self):
        """Optional device check for tests/debugging of a candidate epoch."""
        c, m = self.contacts, self.model
        if c.capacity:
            wp.launch(_check_contacts, c.fast_workers,
                [m.tri_indices, m.edge_indices, c.info, c.candidate_ids, c.candidate_count,
                 self.colors, c.fast_workers, c.error], device=self.device)

    def sweep(self, body):
        if self.query_view is None:
            raise RuntimeError("Rebuild contact-aware GS before its first sweep")
        wp.launch(_begin_color, 1, [self.current_color, self.condition, self.metadata, self.contacts.error], device=self.device)
        def color_body():
            body()
            wp.launch(_next_color, 1, [self.current_color, self.condition, self.metadata, self.contacts.error], device=self.device)
        wp.capture_while(self.condition, color_body)

    def solve_elasticity(self, solver, state_in, dt):
        m, ipc = self.model, self.model.ipc
        # Match the mapped stock-scene bending bank used by DAT + our energies.
        bending = getattr(ipc, "stock_energy_edge_properties", m.edge_bending_properties)
        arguments = [dt, self.order, self.offsets, self.current_color, solver.particle_q_prev,
            state_in.particle_q, m.particle_mass, solver.inertia, m.particle_flags,
            m.tri_indices, m.tri_poses, m.tri_materials, m.tri_areas,
            m.edge_indices, m.edge_rest_angle, m.edge_rest_length, bending,
            m.tet_indices, m.tet_poses, m.tet_materials, solver.particle_adjacency,
            solver.particle_forces, solver.particle_hessians, ipc.pin_targets, ipc.pinned,
            float(solver.params["kpin"]), ipc.edge_coefficient, ipc.tet_grad_N, ipc.tet_measure,
            float(solver.params["damping"])]
        # Two independent 16-lane groups per warp, exactly as the cloth kernel.
        dimension = ((self.n + 1) // 2) * 32 if self.device.is_cuda else self.n
        block_dim = 32 if self.device.is_cuda else 64
        wp.launch(self._elasticity, dimension, block_dim=block_dim, inputs=arguments,
            outputs=[solver.particle_displacements, solver.proposals, solver.error], device=self.device)


wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
