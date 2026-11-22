"""Our linear-CCD predictors inside the actual Newton state/BVH pipeline.

Targets are fixed at step-start plus velocity times dt, without gravity.
Global prediction uses the same conservative FP32 CCD and shared clipping
fraction as motion acceptance. Colored and level prediction use contact-only graph scheduling;
the latter discards one trial and replays component/BFS groups. All graph
counts, colors, trials, levels, geometry and acceptance remain on the device.
"""

from __future__ import annotations

import math

import warp as wp

from .ccd_defaults import LINEAR_CCD_GAP_FLOOR
from .contact import decode_candidate
from .warp_build import configure_module as _configure


HEADERS = ("level_schedule.cuh",)


def configure_module(module=None):
    return _configure(__name__ if module is None else module, HEADERS)


@wp.kernel
def _targets(q: wp.array[wp.vec3], previous: wp.array[wp.vec3], velocity: wp.array[wp.vec3],
              flags: wp.array[int], mass: wp.array[float], dt: float, fraction: float,
              target: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    target[i] = q[i]
    for k in range(3):
        if not wp.isfinite(q[i][k]) or not wp.isfinite(previous[i][k]):
            wp.atomic_max(error, 0, 302)
    if flags[i] & 1 != 0 and mass[i] > 0.0:
        raw = dt * velocity[i]
        value = previous[i] + fraction * raw
        for k in range(3):
            if not wp.isfinite(q[i][k]) or not wp.isfinite(raw[k]) or not wp.isfinite(value[k]):
                wp.atomic_max(error, 0, 302)
        target[i] = value


from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo


@wp.kernel
def _bit_edges(triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                 ids: wp.array[int], count: wp.array[int], workers: int, stride: int,
                 bits: wp.array[wp.uint32], incidence: wp.array[wp.uint64], error: wp.array[int]):
    worker = wp.tid()
    if error[0] == 0:
        for at in range(worker, count[0], workers):
            nodes, kind = decode_candidate(ids[at], triangles, edges, info)
            for a in range(4):
                v = nodes[a]
                wp.atomic_add(incidence, v, wp.uint64(1))
                for b in range(4):
                    other = nodes[b]
                    if v != other:
                        wp.atomic_or(bits, v * stride + other // 32, wp.uint32(1) << wp.uint32(other % 32))


@wp.func_native(r"""
    int count=0;
    for(int w=0;w<stride;++w) {
        uint32_t value=bits.data[vertex*stride+w];
        while(value){value&=value-1;++count;}
    }
    return count;
""")
def _row_count(bits: wp.array[wp.uint32], vertex: int, stride: int) -> int: ...


@wp.kernel
def _row_counts(bits: wp.array[wp.uint32], stride: int, counts: wp.array[int]):
    i = wp.tid()
    counts[i] = _row_count(bits, i, stride)


@wp.func_native(r"""
    int at=offsets.data[vertex];
    for(int w=0;w<stride;++w) {
        uint32_t value=bits.data[vertex*stride+w];
        for(int b=0;b<32;++b)if(value&(uint32_t(1)<<b)) {
            if(at>=adjacency.shape[0]){error.data[0]=372;return;}
            adjacency.data[at++]=32*w+b;
        }
    }
""")
def _fill_row(bits: wp.array[wp.uint32], vertex: int, stride: int, offsets: wp.array[int],
               adjacency: wp.array[int], error: wp.array[int]): ...


@wp.kernel
def _rows_fill(bits: wp.array[wp.uint32], stride: int, offsets: wp.array[int], rows: wp.array[wp.uint64],
                adjacency: wp.array[int], error: wp.array[int]):
    i = wp.tid()
    rows[i] = wp.uint64(offsets[i])
    if i + 1 < rows.shape[0] and error[0] == 0:
        _fill_row(bits, i, stride, offsets, adjacency, error)


_SCRATCH = r"""
    using namespace ipc_gpu::solver::level_predictor;
    Scratch scratch;
    scratch.colors=colors.data;scratch.seen=seen.data;scratch.levels=levels.data;
    scratch.component=component.data;scratch.queue=queue.data;scratch.trial_order=trial_order.data;
    scratch.order=order.data;scratch.offsets=offsets.data;scratch.level_offsets=level_offsets.data;
    scratch.vertices=colors.shape[0];
    Shape shape;
"""


@wp.func_native(_SCRATCH + r"""
    const auto status=color(colors.shape[0],reinterpret_cast<const size_t*>(rows.data),adjacency.data,scratch,shape);
    if(status!=Status::Ok)error.data[0]=static_cast<int>(status)+300;
    metadata.data[0]=shape.colors;metadata.data[1]=shape.colors;metadata.data[2]=colors.shape[0];
""")
def _color_native(rows: wp.array[wp.uint64], adjacency: wp.array[int], colors: wp.array[int], seen: wp.array[int],
                    levels: wp.array[int], component: wp.array[int], queue: wp.array[int], trial_order: wp.array[int],
                    order: wp.array[int], offsets: wp.array[int], level_offsets: wp.array[int], metadata: wp.array[int],
                    error: wp.array[int]): ...


@wp.kernel
def _color_graph(rows: wp.array[wp.uint64], adjacency: wp.array[int], colors: wp.array[int], seen: wp.array[int],
                   levels: wp.array[int], component: wp.array[int], queue: wp.array[int], trial_order: wp.array[int],
                   order: wp.array[int], offsets: wp.array[int], level_offsets: wp.array[int], metadata: wp.array[int],
                   error: wp.array[int]):
    if error[0] == 0:
        _color_native(rows, adjacency, colors, seen, levels, component, queue, trial_order,
                      order, offsets, level_offsets, metadata, error)


@wp.kernel
def _trial_keys(colors: wp.array[int], incidence: wp.array[wp.uint64], flags: wp.array[int],
                  keys: wp.array[wp.int64], values: wp.array[int]):
    i = wp.tid()
    weight = wp.uint64(0)
    if flags[i] & 1 != 0:
        weight = incidence[i]
    keys[i] = wp.int64(colors[i]) * wp.int64(4294967296) + wp.int64(4294967295) - wp.int64(weight)
    values[i] = i


@wp.kernel
def _trial_order(values: wp.array[int], trial_order: wp.array[int]):
    trial_order[wp.tid()] = values[wp.tid()]


@wp.func_native(r"""
    double d[3];for(int k=0;k<3;++k)d[k]=double(value[k])-double(start[k]);
    return ipc_gpu::solver::level_predictor::stable_norm3(d);
""")
def _movement_native(value: wp.vec3, start: wp.vec3) -> wp.float64: ...


@wp.kernel
def _movement(q: wp.array[wp.vec3], start: wp.array[wp.vec3], flags: wp.array[int],
                moving: wp.array[wp.uint8], movement: wp.array[wp.float64], error: wp.array[int]):
    i = wp.tid()
    moving[i] = wp.uint8(flags[i] & 1 != 0)
    movement[i] = _movement_native(q[i], start[i])
    if not wp.isfinite(movement[i]):
        wp.atomic_max(error, 0, 373)


@wp.func_native(_SCRATCH + r"""
    const auto status=schedule(colors.shape[0],reinterpret_cast<const size_t*>(rows.data),adjacency.data,
        moving.data,movement.data,scratch,shape);
    if(status!=Status::Ok)error.data[0]=static_cast<int>(status)+300;
    metadata.data[1]=shape.groups;metadata.data[2]=shape.active;
""")
def _levels_native(rows: wp.array[wp.uint64], adjacency: wp.array[int], colors: wp.array[int], seen: wp.array[int],
                     levels: wp.array[int], component: wp.array[int], queue: wp.array[int], trial_order: wp.array[int],
                     order: wp.array[int], offsets: wp.array[int], level_offsets: wp.array[int], metadata: wp.array[int],
                     moving: wp.array[wp.uint8], movement: wp.array[wp.float64], error: wp.array[int]): ...


@wp.kernel
def _levels_graph(rows: wp.array[wp.uint64], adjacency: wp.array[int], colors: wp.array[int], seen: wp.array[int],
                    levels: wp.array[int], component: wp.array[int], queue: wp.array[int], trial_order: wp.array[int],
                    order: wp.array[int], offsets: wp.array[int], level_offsets: wp.array[int], metadata: wp.array[int],
                    moving: wp.array[wp.uint8], movement: wp.array[wp.float64], error: wp.array[int]):
    if error[0] == 0:
        _levels_native(rows, adjacency, colors, seen, levels, component, queue, trial_order,
                       order, offsets, level_offsets, metadata, moving, movement, error)


@wp.func_native(r"""
    return ipc_gpu::solver::level_predictor::group_of(index,groups,offsets.data);
""")
def _group(index: int, groups: int, offsets: wp.array[int]) -> int: ...


@wp.kernel
def _group_colors(order: wp.array[int], offsets: wp.array[int], metadata: wp.array[int], groups: wp.array[int]):
    i = wp.tid()
    if i < metadata[2]:
        groups[order[i]] = _group(i, metadata[1], offsets)


@wp.kernel
def _color_begin(current: wp.array[int], condition: wp.array[int], metadata: wp.array[int], error: wp.array[int]):
    current[0] = 0
    condition[0] = int(metadata[1] > 0 and error[0] == 0)


@wp.kernel
def _color_next(current: wp.array[int], condition: wp.array[int], metadata: wp.array[int], error: wp.array[int]):
    current[0] += 1
    condition[0] = int(current[0] < metadata[1] and error[0] == 0)


@wp.kernel
def _color_proposal(q: wp.array[wp.vec3], target: wp.array[wp.vec3], flags: wp.array[int], colors: wp.array[int],
                      current: wp.array[int], proposal: wp.array[wp.vec3]):
    i = wp.tid()
    proposal[i] = q[i]
    if flags[i] & 1 != 0 and colors[i] == current[0]:
        proposal[i] = target[i]


@wp.kernel
def _sweep_begin(index: wp.array[int], condition: wp.array[int], error: wp.array[int]):
    index[0] = 0
    condition[0] = int(error[0] == 0)


@wp.kernel
def _sweep_bits(q: wp.array[wp.vec3], before: wp.array[wp.vec3], target: wp.array[wp.vec3], flags: wp.array[int],
                  bits: wp.array[int]):
    i = wp.tid()
    if flags[i] & 1 != 0:
        moved = q[i][0] != before[i][0] or q[i][1] != before[i][1] or q[i][2] != before[i][2]
        distance = wp.vec3d(q[i]) - wp.vec3d(target[i])
        value = int(0)
        if moved:
            value |= 1
        if wp.dot(distance, distance) >= wp.float64(1e-28):
            value |= 2
        wp.atomic_or(bits, 0, value)


@wp.kernel
def _sweep_next(index: wp.array[int], condition: wp.array[int], maximum: int, bits: wp.array[int], error: wp.array[int]):
    index[0] += 1
    condition[0] = int(index[0] < maximum and bits[0] == 3 and error[0] == 0)


class PredictorGraph:
    def __init__(self, contacts, *, compact_adjacency=False):
        self.device, self.n = contacts.device, contacts.model.particle_count
        self.stride = (self.n + 31) // 32
        self.adjacency_capacity = (min(12 * contacts.capacity, self.n * (self.n - 1))
                                   if compact_adjacency else 12 * contacts.capacity)
        if self.adjacency_capacity >= 2**31 or self.n * self.stride >= 2**31:
            raise ValueError("Predictor graph exceeds its GPU index capacity")
        self.bits = wp.zeros(self.n * self.stride, wp.uint32, device=self.device)
        self.counts = wp.zeros(self.n + 1, int, device=self.device)
        self.scan = wp.empty_like(self.counts)
        self.rows = wp.empty(self.n + 1, wp.uint64, device=self.device)
        self.adjacency = wp.empty(self.adjacency_capacity, int, device=self.device)
        self.incidence = wp.zeros(self.n, wp.uint64, device=self.device)
        self.colors, self.seen, self.levels, self.component, self.queue, self.trial_order, self.order = (
            wp.empty(self.n, int, device=self.device) for _ in range(7))
        self.offsets = wp.empty(self.n + 1, int, device=self.device)
        self.level_offsets = wp.empty(self.n + 1, int, device=self.device)
        self.metadata = wp.zeros(3, int, device=self.device)
        self.moving = wp.empty(self.n, wp.uint8, device=self.device)
        self.movement = wp.empty(self.n, wp.float64, device=self.device)
        self.groups = wp.empty(self.n, int, device=self.device)
        self.keys = wp.empty(2 * self.n, wp.int64, device=self.device)
        self.values = wp.empty(2 * self.n, int, device=self.device)
        self._parallel_levels = None
        self._parallel_colors = None

    def prepare_parallel_colors(self, workspace):
        """Install an explicitly prepared GS-only canonical color workspace."""
        if workspace.graph is not self:
            raise ValueError("The color workspace belongs to another predictor graph")
        self._parallel_colors = workspace

    def prepare_parallel_levels(self, workspace):
        """Install an explicitly prepared GS-only execution workspace."""
        if workspace.graph is not self:
            raise ValueError("The level workspace belongs to another predictor graph")
        self._parallel_levels = workspace

    def arguments(self):
        return [self.rows, self.adjacency, self.colors, self.seen, self.levels, self.component, self.queue,
                self.trial_order, self.order, self.offsets, self.level_offsets, self.metadata]

    def build(self, contacts):
        self.bits.zero_(); self.counts.zero_(); self.incidence.zero_()
        wp.launch(_bit_edges, contacts.fast_workers,
            [contacts.model.tri_indices, contacts.model.edge_indices, contacts.info, contacts.candidate_ids,
             contacts.candidate_count, contacts.fast_workers, self.stride, self.bits, self.incidence, contacts.error], device=self.device)
        wp.launch(_row_counts, self.n, [self.bits, self.stride, self.counts], device=self.device)
        wp.utils.array_scan(self.counts, self.scan, inclusive=False)
        wp.launch(_rows_fill, self.n + 1, [self.bits, self.stride, self.scan, self.rows, self.adjacency, contacts.error], device=self.device)
        if self._parallel_colors is None:
            wp.launch(_color_graph, 1, [*self.arguments(), contacts.error], device=self.device)
        else:
            self._parallel_colors.apply(contacts)
        wp.launch(_trial_keys, self.n,
            [self.colors, self.incidence, contacts.model.particle_flags, self.keys, self.values], device=self.device)
        wp.utils.radix_sort_pairs(self.keys, self.values, self.n)
        wp.launch(_trial_order, self.n, [self.values, self.trial_order], device=self.device)

    def finish_trial(self, q, start, contacts):
        wp.launch(_movement, self.n,
            [q, start, contacts.model.particle_flags, self.moving, self.movement, contacts.error], device=self.device)
        if self._parallel_levels is None:
            wp.launch(_levels_graph, 1, [*self.arguments(), self.moving, self.movement, contacts.error], device=self.device)
        else:
            self._parallel_levels.apply(contacts)
        self.groups.fill_(-1)
        wp.launch(_group_colors, self.n, [self.order, self.offsets, self.metadata, self.groups], device=self.device)


class InitialGuess:
    def __init__(self, ccd, detector, params=None, *, use_group_queries=False, independent_colors=False):
        if not isinstance(independent_colors, bool):
            raise TypeError("Independent predictor colors must be a boolean")
        if independent_colors and not use_group_queries:
            raise ValueError("Independent predictor colors require complete grouped queries")
        self.ccd, self.contacts, self.detector = ccd, ccd.contacts, detector
        self.independent_colors = independent_colors
        self.device, self.model = ccd.device, ccd.model
        n = self.model.particle_count
        self.start = wp.empty(n, wp.vec3, device=self.device)
        self.targets = wp.empty(n, wp.vec3, device=self.device)
        self.proposal = wp.empty(n, wp.vec3, device=self.device)
        self.box_filter = None
        self.sweep_start = wp.empty(n, wp.vec3, device=self.device)
        self.color = wp.zeros(1, int, device=self.device)
        self.color_condition = wp.zeros(1, int, device=self.device)
        self.sweep = wp.zeros(1, int, device=self.device)
        self.sweep_condition = wp.zeros(1, int, device=self.device)
        self.bits = wp.zeros(1, int, device=self.device)
        colored = params is None or params.get("use_colored_ccd_guess", False) or params.get("use_colored_ccd_guess_level_based", False)
        self.graph = PredictorGraph(self.contacts, compact_adjacency=use_group_queries) if colored else None
        self.query_groups = None
        if colored and use_group_queries:
            from .query_groups import QueryGroups
            self.query_groups = QueryGroups(self.contacts)
        if independent_colors and (self.query_groups is None or not self.query_groups.enabled):
            raise ValueError("Independent predictor colors require complete grouped queries")

    def _sweep(self, q, colors, dt, *, independent=False):
        g, c = self.graph, self.contacts
        wp.copy(self.sweep_start, q)
        wp.launch(_color_begin, 1, [self.color, self.color_condition, g.metadata, c.error], device=self.device)
        query_view = self.query_groups.view(self.color) if self.query_groups is not None else None

        def color_body():
            if independent:
                # Each swept contact contributes a clique, so at most one
                # corner moves per pair. Keep unrelated vertices' full steps,
                # as CPU colored prediction does, with unchanged gap checks.
                self.ccd.accept(q, self.targets, -1, dt, bounded=False, preserve_gap=True,
                    colors=colors, current_color=self.color, query_view=query_view, independent=True)
            else:
                wp.launch(_color_proposal, self.model.particle_count,
                    [q, self.targets, self.model.particle_flags, colors, self.color, self.proposal], device=self.device)
                self.ccd.accept(q, self.proposal, -1, dt, bounded=False, preserve_gap=True, query_view=query_view)
            wp.launch(_color_next, 1, [self.color, self.color_condition, g.metadata, c.error], device=self.device)

        wp.capture_while(self.color_condition, color_body)
        self.bits.zero_()
        wp.launch(_sweep_bits, self.model.particle_count,
            [q, self.sweep_start, self.targets, self.model.particle_flags, self.bits], device=self.device)

    def apply(self, q, previous, velocity, inertia, dt, params, *, detector=None):
        legacy = bool(params.get("use_colored_ccd_guess", False))
        level = bool(params.get("use_colored_ccd_guess_level_based", False))
        if legacy and level:
            raise ValueError("Choose one colored CCD predictor")
        if self.independent_colors and not (legacy or level):
            raise ValueError("Independent predictor colors require a colored CCD mode")
        sweeps = params.get("colored_ccd_guess_iters", 0)
        fraction = params.get("colored_ccd_guess_fraction", 1.0)
        if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 0:
            raise ValueError("Colored CCD sweeps must be a nonnegative integer")
        if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Colored CCD fraction must be in [0,1]")
        global_guess = bool(params.get("use_ccd_guess", False)) and not legacy and not level
        global_guess = global_guess and self.model.body_count == 0 and self.model.tet_count == 0
        if not legacy and not level and not global_guess:
            return q
        c = self.contacts
        wp.copy(self.start, q)
        wp.launch(_targets, self.model.particle_count,
            [q, previous, velocity, self.model.particle_flags, self.model.particle_mass, float(dt), float(fraction if legacy or level else 1.0),
             self.targets, c.error], device=self.device)
        if (legacy or level) and sweeps == 0:
            return q
        detector = detector or self.detector
        if detector is None:
            raise ValueError("The CCD predictor requires the actual Newton triangle-mesh detector")
        c.detect_swept(detector, self.start, self.targets)
        if global_guess:
            # One FP32 CCD pass computes the common conservative fraction and
            # rechecks its rounded endpoint. There is no separate FP64 TOI pass.
            return self.ccd.accept(q, self.targets, -1, dt, bounded=False)
        if self.graph is None:
            raise RuntimeError("Prepare colored predictor workspaces before CUDA graph capture")
        # The probe and first updating sweep each start at the same positions.
        # Every rounded prefix lies in the padded swept box. A whole-box
        # separation beyond the barrier radius excludes all possible contacts,
        # even when the four corners move in different groups.
        if (self.box_filter is not None and level and sweeps == 1
                and params["d_hat"] >= LINEAR_CCD_GAP_FLOOR):
            self.box_filter.apply(self.start, previous, c.swept_lower, c.swept_upper, params)
        g = self.graph
        g.build(c)
        if self.query_groups is not None:
            self.query_groups.build(g.colors)
        colors = g.colors
        if level:
            # The discarded probe must use the same per-vertex acceptance as
            # the updating sweeps: its movement determines the level ordering.
            self._sweep(q, colors, dt, independent=self.independent_colors)
            g.finish_trial(q, self.start, c)
            wp.copy(q, self.start)
            colors = g.groups
            if self.query_groups is not None:
                self.query_groups.build(colors)
        wp.launch(_sweep_begin, 1, [self.sweep, self.sweep_condition, c.error], device=self.device)

        def sweep_body():
            self._sweep(q, colors, dt, independent=self.independent_colors)
            wp.launch(_sweep_next, 1,
                [self.sweep, self.sweep_condition, sweeps, self.bits, c.error], device=self.device)

        wp.capture_while(self.sweep_condition, sweep_body)
        return q


configure_module()
