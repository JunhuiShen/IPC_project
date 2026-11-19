"""BVH barrier candidates and material/contact graph colors, without forces.

This module depends only on Warp so it can run beside the pinned reference
Newton runtime. Graph bitsets, CSR conversion, and canonical warm greedy
coloring are adapted from ``gpu/solver.py`` (ContactAwareGS/WarmColors).
Only lower-ID adjacency is stored: its fixed-point colors are exactly the
serial, lowest-vertex-ID greedy result, regardless of the previous colors.

Candidate rows have explicit capacities. Overflow latches error 421 and
prevents coloring/sweeps; no partial candidate graph is used for an update.
Errors 422, 423, and 424 indicate graph storage, colors, and convergence.
"""

import warp as wp


@wp.func
def _worlds_match(a: int, b: int):
    return a < 0 or b < 0 or a == b


@wp.func
def _any_dynamic(nodes: wp.vec4i, flags: wp.array[int], mass: wp.array[float]):
    result = bool(False)
    for i in range(4):
        v = nodes[i]
        result = result or (flags[v] & 1 != 0 and mass[v] > 0.0)
    return result


@wp.kernel
def _query_nodes(bvh: wp.uint64, lower: wp.array[wp.vec3], upper: wp.array[wp.vec3],
                 triangles: wp.array2d[int], worlds: wp.array[int], surface: wp.array[int],
                 flags: wp.array[int], mass: wp.array[float], pairs: wp.array2d[int],
                 counts: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    found = int(0)
    if error[0] == 0 and surface[vertex] != 0:
        query = wp.bvh_query_aabb(bvh, lower[vertex], upper[vertex])
        triangle = int(0)
        while wp.bvh_query_next(query, triangle):
            a = triangles[triangle, 0]
            b = triangles[triangle, 1]
            c = triangles[triangle, 2]
            nodes = wp.vec4i(vertex, a, b, c)
            if (vertex != a and vertex != b and vertex != c
                    and _worlds_match(worlds[vertex], worlds[a])
                    and _any_dynamic(nodes, flags, mass)):
                if found < pairs.shape[1]:
                    pairs[vertex, found] = triangle
                found += 1
        if found > pairs.shape[1]:
            wp.atomic_max(error, 0, 421)
    counts[vertex] = found


@wp.kernel
def _query_edges(bvh: wp.uint64, lower: wp.array[wp.vec3], upper: wp.array[wp.vec3],
                 edges: wp.array2d[int], worlds: wp.array[int], flags: wp.array[int],
                 mass: wp.array[float], pairs: wp.array2d[int], counts: wp.array[int],
                 error: wp.array[int]):
    edge = wp.tid()
    found = int(0)
    if error[0] == 0:
        a = edges[edge, 2]
        b = edges[edge, 3]
        query = wp.bvh_query_aabb(bvh, lower[edge], upper[edge])
        other = int(0)
        while wp.bvh_query_next(query, other):
            c = edges[other, 2]
            d = edges[other, 3]
            nodes = wp.vec4i(a, b, c, d)
            if (edge < other and a != c and a != d and b != c and b != d
                    and _worlds_match(worlds[a], worlds[c])
                    and _any_dynamic(nodes, flags, mass)):
                if found < pairs.shape[1]:
                    pairs[edge, found] = other
                found += 1
        if found > pairs.shape[1]:
            wp.atomic_max(error, 0, 421)
    counts[edge] = found


@wp.func
def _edge(vertex: int, neighbor: int, stride: int, bits: wp.array[wp.uint32], error: wp.array[int]):
    n = bits.shape[0] // stride
    if vertex < 0 or vertex >= n or neighbor < 0 or neighbor >= n:
        wp.atomic_max(error, 0, 422)
    elif neighbor < vertex:
        wp.atomic_or(bits, vertex * stride + neighbor // 32,
                     wp.uint32(1) << wp.uint32(neighbor % 32))


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
def _row_counts(nt: wp.array[int], ee: wp.array[int], counts: wp.array[int], error: wp.array[int]):
    row = wp.tid()
    value = int(0)
    if error[0] == 0:
        if row < nt.shape[0]:
            value = nt[row]
        elif row < nt.shape[0] + ee.shape[0]:
            value = ee[row - nt.shape[0]]
    counts[row] = value


@wp.kernel
def _pack_candidates(triangles: wp.array2d[int], edges: wp.array2d[int],
                     nt: wp.array2d[int], ee: wp.array2d[int], offsets: wp.array[int],
                     nodes_out: wp.array[wp.vec4i], count: wp.array[int], stride: int,
                     bits: wp.array[wp.uint32], error: wp.array[int]):
    row = wp.tid()
    if row == 0:
        count[0] = offsets[offsets.shape[0] - 1]
    if error[0] == 0:
        for at in range(offsets[row], offsets[row + 1]):
            local = at - offsets[row]
            nodes = wp.vec4i(0)
            if row < nt.shape[0]:
                triangle = nt[row, local]
                nodes = wp.vec4i(row, triangles[triangle, 0], triangles[triangle, 1], triangles[triangle, 2])
            else:
                edge = row - nt.shape[0]
                other = ee[edge, local]
                nodes = wp.vec4i(edges[edge, 2], edges[edge, 3], edges[other, 2], edges[other, 3])
            if at >= nodes_out.shape[0]:
                wp.atomic_max(error, 0, 421)
            else:
                nodes_out[at] = nodes
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
            if(at>=adjacency.shape[0])return;
            adjacency.data[at++]=32*word+bit;
        }
    }
""")
def _fill_row(bits: wp.array[wp.uint32], stride: int, vertex: int, offsets: wp.array[int],
              adjacency: wp.array[int]): ...


@wp.kernel
def _rows(bits: wp.array[wp.uint32], stride: int, offsets: wp.array[int],
          rows: wp.array[wp.uint64], adjacency: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    rows[vertex] = wp.uint64(offsets[vertex])
    if offsets[offsets.shape[0] - 1] > adjacency.shape[0]:
        wp.atomic_max(error, 0, 422)
    elif vertex + 1 < rows.shape[0] and error[0] == 0:
        _fill_row(bits, stride, vertex, offsets, adjacency)


@wp.func_native(r"""
    unsigned long long forbidden[2]={0,0};
    for(uint64_t at=rows.data[vertex];at<rows.data[vertex+1];++at) {
        const int color=colors.data[adjacency.data[at]];
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
    for(int color=128;color<colors.shape[0];++color) {
        bool available=true;
        for(uint64_t at=rows.data[vertex];at<rows.data[vertex+1];++at) {
            if(colors.data[adjacency.data[at]]==color){available=false;break;}
        }
        if(available)return color;
    }
    return -1;
""")
def _least_free(rows: wp.array[wp.uint64], adjacency: wp.array[int], colors: wp.array[int], vertex: int) -> int: ...


@wp.kernel
def _repair(rows: wp.array[wp.uint64], adjacency: wp.array[int], source: wp.array[int],
            target: wp.array[int], changed: wp.array[int], track: int, error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0:
        color = _least_free(rows, adjacency, source, vertex)
        if color < 0 or color >= source.shape[0]:
            wp.atomic_max(error, 0, 423)
        else:
            target[vertex] = color
            if track != 0 and color != source[vertex]:
                wp.atomic_or(changed, 0, 1)


@wp.kernel
def _start_repair(condition: wp.array[int], iterations: wp.array[int], error: wp.array[int]):
    condition[0] = int(error[0] == 0)
    iterations[0] = 0


@wp.kernel
def _next_repair(changed: wp.array[int], condition: wp.array[int], iterations: wp.array[int],
                 maximum: int, error: wp.array[int]):
    iterations[0] += 1
    condition[0] = int(changed[0] != 0 and error[0] == 0)
    if condition[0] != 0 and iterations[0] > maximum:
        wp.atomic_max(error, 0, 424)
        condition[0] = 0


@wp.kernel
def _count_colors(colors: wp.array[int], counts: wp.array[int], metadata: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0:
        color = colors[vertex]
        if color < 0 or color >= colors.shape[0]:
            wp.atomic_max(error, 0, 423)
        else:
            wp.atomic_add(counts, color, 1)
            wp.atomic_max(metadata, 0, color + 1)
            wp.atomic_max(metadata, 1, color + 1)
        if vertex == 0:
            metadata[2] = colors.shape[0]


@wp.kernel
def _pack_colors(colors: wp.array[int], offsets: wp.array[int], cursors: wp.array[int],
                 order: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0:
        color = colors[vertex]
        at = offsets[color] + wp.atomic_add(cursors, color, 1)
        if at < offsets[color] or at >= offsets[color + 1] or at >= order.shape[0]:
            wp.atomic_max(error, 0, 423)
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
def _check_independence(rows: wp.array[wp.uint64], adjacency: wp.array[int],
                        colors: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0:
        for at in range(int(rows[vertex]), int(rows[vertex + 1])):
            if colors[vertex] == colors[adjacency[at]]:
                wp.atomic_max(error, 0, 423)


@wp.kernel
def _record_statistics(n: int, offsets: wp.array[int], rows: wp.array[wp.uint64],
                       metadata: wp.array[int], stats: wp.array[wp.int64], error: wp.array[int]):
    if error[0] == 0:
        previous = stats[0]
        stats[0] = previous + wp.int64(1)
        values = wp.vec4i(offsets[n], offsets[offsets.shape[0] - 1] - offsets[n], metadata[0], int(rows[n]))
        for i in range(4):
            value = wp.int64(values[i])
            at = 1 + 3 * i
            stats[at] = stats[at] + value
            if previous == wp.int64(0):
                stats[at + 1] = value
                stats[at + 2] = value
            else:
                stats[at + 1] = wp.min(stats[at + 1], value)
                stats[at + 2] = wp.max(stats[at + 2], value)


class BarrierColoring:
    """Persistent broad-phase pair rows, deduplicated graph, and color sweep.

    Call ``update()`` just after the associated box/BVH checkpoint. A sweep
    executes a zero-argument callback for each device-selected color, with
    vertices in ``order[offsets[c]:offsets[c+1]]``. Construction allocates all
    work arrays; update and sweep neither read back nor resize device arrays.
    """

    def __init__(self, model, boxes, error, *, vertex_capacity=128, edge_capacity=256):
        for name, value in (("vertex_capacity", vertex_capacity), ("edge_capacity", edge_capacity)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        self.model, self.boxes, self.error, self.device = model, boxes, error, model.device
        self.n, self.edge_count = model.particle_count, model.edge_count
        self.vertex_capacity, self.edge_capacity = vertex_capacity, edge_capacity
        self.capacity = self.n * vertex_capacity + self.edge_count * edge_capacity
        self.stride = (self.n + 31) // 32
        material_edges = 3 * model.tri_count + 6 * model.edge_count + 6 * model.tet_count
        adjacency_capacity = min(self.n * (self.n - 1) // 2, material_edges + 6 * self.capacity)
        if max(self.capacity, self.n * self.stride, adjacency_capacity, self.n + self.edge_count + 1) >= 2**31:
            raise ValueError("Barrier coloring exceeds int32 GPU index capacity")
        d = self.device
        self.surface = getattr(getattr(model, "ipc", None), "surface", None)
        if self.surface is None:
            self.surface = wp.ones(self.n, int, device=d)
        self.node_triangle_pairs = wp.empty((self.n, vertex_capacity), int, device=d)
        self.node_triangle_counts = wp.zeros(self.n, int, device=d)
        self.edge_edge_pairs = wp.empty((self.edge_count, edge_capacity), int, device=d)
        self.edge_edge_counts = wp.zeros(self.edge_count, int, device=d)
        self._pair_counts = wp.zeros(self.n + self.edge_count + 1, int, device=d)
        self.candidate_offsets = wp.zeros_like(self._pair_counts)
        self.candidate_nodes = wp.empty(self.capacity, wp.vec4i, device=d)
        self.candidate_count = wp.zeros(1, int, device=d)
        self.material_bits = wp.zeros(self.n * self.stride, wp.uint32, device=d)
        self.bits = wp.empty_like(self.material_bits)
        self.degrees = wp.zeros(self.n + 1, int, device=d)
        self._scan = wp.empty_like(self.degrees)
        self.rows = wp.zeros(self.n + 1, wp.uint64, device=d)
        self.adjacency = wp.empty(adjacency_capacity, int, device=d)
        self.colors = wp.full(self.n, -1, int, device=d)
        self._temporary = wp.empty_like(self.colors)
        self.metadata = wp.zeros(3, int, device=d)
        self._counts = wp.zeros(self.n + 1, int, device=d)
        self.offsets = wp.zeros_like(self._counts)
        self._cursors = wp.zeros_like(self._counts)
        self.order = wp.empty(self.n, int, device=d)
        self.current_color = wp.zeros(1, int, device=d)
        self._condition = wp.zeros(1, int, device=d)
        self._changed = wp.zeros(1, int, device=d)
        self.coloring_iterations = wp.zeros(1, int, device=d)
        self._stats = wp.zeros(13, wp.int64, device=d)
        self._built = False
        self.ccd_epoch = None
        self.ccd_source_box_epoch = None
        primitive_count = model.tri_count + model.edge_count + model.tet_count
        if primitive_count:
            wp.launch(_material_edges, primitive_count,
                      [model.tri_indices, model.edge_indices, model.tet_indices,
                       self.stride, self.material_bits, self.error], device=d)

    def update(self):
        m, b, d, n = self.model, self.boxes, self.device, self.n
        self.node_triangle_counts.zero_()
        self.edge_edge_counts.zero_()
        self.candidate_count.zero_()
        self.metadata.zero_()
        if n and b.triangle_bvh is not None:
            wp.launch(_query_nodes, n,
                      [b.triangle_bvh.id, b.node_lower, b.node_upper, m.tri_indices,
                       m.particle_world, self.surface, m.particle_flags, m.particle_mass,
                       self.node_triangle_pairs, self.node_triangle_counts, self.error], device=d)
        if self.edge_count and b.edge_bvh is not None:
            wp.launch(_query_edges, self.edge_count,
                      [b.edge_bvh.id, b.edge_query_lower, b.edge_query_upper, m.edge_indices,
                       m.particle_world, m.particle_flags, m.particle_mass,
                       self.edge_edge_pairs, self.edge_edge_counts, self.error], device=d)
        wp.launch(_row_counts, self._pair_counts.shape[0],
                  [self.node_triangle_counts, self.edge_edge_counts, self._pair_counts, self.error], device=d)
        wp.utils.array_scan(self._pair_counts, self.candidate_offsets, inclusive=False)
        wp.copy(self.bits, self.material_bits)
        if n + self.edge_count:
            wp.launch(_pack_candidates, n + self.edge_count,
                      [m.tri_indices, m.edge_indices, self.node_triangle_pairs, self.edge_edge_pairs,
                       self.candidate_offsets, self.candidate_nodes, self.candidate_count,
                       self.stride, self.bits, self.error], device=d)
        wp.launch(_degrees, n + 1, [self.bits, self.stride, self.degrees], device=d)
        wp.utils.array_scan(self.degrees, self._scan, inclusive=False)
        wp.launch(_rows, n + 1, [self.bits, self.stride, self._scan, self.rows, self.adjacency, self.error], device=d)
        self._counts.zero_()
        self._cursors.zero_()
        if n:
            wp.launch(_start_repair, 1, [self._condition, self.coloring_iterations, self.error], device=d)

            def repair_body():
                wp.launch(_repair, n, [self.rows, self.adjacency, self.colors, self._temporary,
                                      self._changed, 0, self.error], device=d)
                self._changed.zero_()
                wp.launch(_repair, n, [self.rows, self.adjacency, self._temporary, self.colors,
                                      self._changed, 1, self.error], device=d)
                wp.launch(_next_repair, 1, [self._changed, self._condition,
                                           self.coloring_iterations, n + 1, self.error], device=d)

            wp.capture_while(self._condition, repair_body)
            wp.launch(_count_colors, n, [self.colors, self._counts, self.metadata, self.error], device=d)
        wp.utils.array_scan(self._counts, self.offsets, inclusive=False)
        if n:
            wp.launch(_pack_colors, n, [self.colors, self.offsets, self._cursors, self.order, self.error], device=d)
        wp.launch(_record_statistics, 1,
                  [n, self.candidate_offsets, self.rows, self.metadata, self._stats, self.error], device=d)
        self._built = True
        if self.ccd_epoch is not None:
            from .broad_phase_boxes import _advance_ccd_epoch
            wp.copy(self.ccd_source_box_epoch, self.boxes.ccd_epoch)
            wp.launch(_advance_ccd_epoch, 1, [self.ccd_epoch], device=d)

    def sweep(self, body):
        if not self._built:
            raise RuntimeError("Update barrier coloring before its first sweep")
        wp.launch(_begin_color, 1,
                  [self.current_color, self._condition, self.metadata, self.error], device=self.device)

        def color_body():
            body()
            wp.launch(_next_color, 1,
                      [self.current_color, self._condition, self.metadata, self.error], device=self.device)

        wp.capture_while(self._condition, color_body)

    def validate_independence(self):
        if self.n:
            wp.launch(_check_independence, self.n,
                      [self.rows, self.adjacency, self.colors, self.error], device=self.device)

    def reset_statistics(self):
        """Exclude setup/capture/warmup epochs from later measurement samples."""
        self._stats.zero_()

    def statistics(self):
        """Synchronizing diagnostic readback; never call inside graph capture."""
        if self.device.is_capturing:
            raise RuntimeError("Barrier coloring statistics require execution outside graph capture")
        stats = self._stats.numpy()
        offsets = self.candidate_offsets.numpy()
        epochs = int(stats[0])
        final = [int(offsets[self.n]), int(offsets[-1] - offsets[self.n]),
                 int(self.metadata.numpy()[0]), int(self.rows.numpy()[-1])]
        result = {"query_epochs": epochs, "candidate_capacity": self.capacity,
                  "vertex_capacity": self.vertex_capacity, "edge_capacity": self.edge_capacity,
                  "node_triangle_max_row_demand": int(self.node_triangle_counts.numpy().max()) if self.n else 0,
                  "edge_edge_max_row_demand": int(self.edge_edge_counts.numpy().max()) if self.edge_count else 0,
                  "coloring_iterations": int(self.coloring_iterations.numpy()[0]),
                  "statistics_in_timed_work": True, "error": int(self.error.numpy()[0])}
        for i, name in enumerate(("node_triangle_pairs", "edge_edge_pairs", "color_count", "graph_edges")):
            at = 1 + 3 * i
            result[name] = final[i]
            result[name + "_mean"] = float(stats[at]) / epochs if epochs else 0.0
            result[name + "_min"] = int(stats[at + 1]) if epochs else 0
            result[name + "_max"] = int(stats[at + 2]) if epochs else 0
        result["candidate_count"] = final[0] + final[1]
        return result
