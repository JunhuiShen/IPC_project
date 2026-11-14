"""Parallel integer-only replacement for the canonical level replay schedule.

Connected components and multi-source shortest-path levels have unique integer
solutions. Exact movement maxima seed every tie, as the native oracle does.
A stable (level, trial-rank) sort then reproduces its order and group offsets.
This module never reads or writes positions or accepts a physical step.
"""
import warp as wp


@wp.kernel
def _initialize(moving: wp.array[wp.uint8], movement: wp.array[wp.float64],
                colors: wp.array[int], trial_order: wp.array[int], seen: wp.array[int],
                labels: wp.array[int], largest: wp.array[wp.float64], error: wp.array[int]):
    v = wp.tid()
    labels[v] = v
    largest[v] = wp.float64(-1.0)
    if moving[v] > wp.uint8(1) or not wp.isfinite(movement[v]) or movement[v] < wp.float64(0.0):
        wp.atomic_max(error, 0, 373 if wp.isfinite(movement[v]) == False or movement[v] < wp.float64(0.0) else 371)
    q = trial_order[v]
    if q < 0 or q >= colors.shape[0]:
        wp.atomic_max(error, 0, 371)
    else:
        if colors[q] < 0 or colors[q] >= colors.shape[0]:
            wp.atomic_max(error, 0, 371)
        old = wp.atomic_add(seen, q, 1)
        if old != 0:
            wp.atomic_max(error, 0, 371)


@wp.kernel
def _validate_rows(rows: wp.array[wp.uint64], adjacency: wp.array[int], error: wp.array[int]):
    v = wp.tid()
    n = rows.shape[0]-1
    begin = rows[v]
    end = rows[v+1]
    if (v == 0 and begin != wp.uint64(0)) or begin > end or end > wp.uint64(adjacency.shape[0]):
        wp.atomic_max(error, 0, 371)
    elif error[0] == 0:
        previous = int(-1)
        for at in range(int(begin), int(end)):
            neighbor = adjacency[at]
            if neighbor < 0 or neighbor >= n or neighbor == v or neighbor <= previous:
                wp.atomic_max(error, 0, 371)
            previous = neighbor


@wp.kernel
def _start_loop(condition: wp.array[int], iterations: wp.array[int], error: wp.array[int]):
    condition[0] = int(error[0] == 0)
    iterations[0] = 0


@wp.kernel
def _next_loop(changed: wp.array[int], condition: wp.array[int], iterations: wp.array[int],
               maximum: int, error: wp.array[int]):
    iterations[0] += 1
    condition[0] = int(changed[0] != 0 and error[0] == 0)
    if condition[0] != 0 and iterations[0] > maximum:
        wp.atomic_max(error, 0, 371)
        condition[0] = 0


@wp.kernel
def _component_pass(rows: wp.array[wp.uint64], adjacency: wp.array[int],
                    source: wp.array[int], target: wp.array[int], changed: wp.array[int], error: wp.array[int]):
    v = wp.tid()
    if error[0] == 0:
        best = source[v]
        for at in range(int(rows[v]), int(rows[v+1])):
            best = wp.min(best, source[adjacency[at]])
        best = wp.min(best, source[best])
        target[v] = best
        if best != source[v]:
            wp.atomic_or(changed, 0, 1)


@wp.kernel
def _component_maximum(labels: wp.array[int], moving: wp.array[wp.uint8],
                        movement: wp.array[wp.float64], largest: wp.array[wp.float64], error: wp.array[int]):
    v = wp.tid()
    if error[0] == 0 and moving[v] != wp.uint8(0):
        wp.atomic_max(largest, labels[v], movement[v])


@wp.kernel
def _seed_levels(labels: wp.array[int], moving: wp.array[wp.uint8], movement: wp.array[wp.float64],
                  largest: wp.array[wp.float64], levels: wp.array[int], error: wp.array[int]):
    v = wp.tid()
    if error[0] == 0:
        maximum = largest[labels[v]]
        levels[v] = levels.shape[0]
        if maximum < wp.float64(0.0) or (moving[v] != wp.uint8(0) and movement[v] == maximum):
            levels[v] = 0


@wp.kernel
def _level_pass(rows: wp.array[wp.uint64], adjacency: wp.array[int], source: wp.array[int],
                target: wp.array[int], changed: wp.array[int], error: wp.array[int]):
    v = wp.tid()
    if error[0] == 0:
        best = source[v]
        for at in range(int(rows[v]), int(rows[v+1])):
            value = source[adjacency[at]]
            if value < source.shape[0]:
                best = wp.min(best, value+1)
        target[v] = best
        if best != source[v]:
            wp.atomic_or(changed, 0, 1)


@wp.kernel
def _active_keys(trial_order: wp.array[int], colors: wp.array[int], levels: wp.array[int],
                  moving: wp.array[wp.uint8], keys: wp.array[wp.int64], values: wp.array[int],
                  histogram: wp.array[int], active: wp.array[int], error: wp.array[int]):
    rank = wp.tid()
    if error[0] == 0:
        v = trial_order[rank]
        values[rank] = v
        if moving[v] != wp.uint8(0):
            level = levels[v]
            if level < 0 or level >= levels.shape[0]:
                wp.atomic_max(error, 0, 371)
            else:
                keys[rank] = wp.int64(level)*wp.int64(levels.shape[0])+wp.int64(rank)
                wp.atomic_add(histogram, level, 1)
                wp.atomic_add(active, 0, 1)
        else:
            keys[rank] = wp.int64(9223372036854775807)


@wp.kernel
def _boundaries(values: wp.array[int], levels: wp.array[int], colors: wp.array[int], active: wp.array[int],
                 order: wp.array[int], boundaries: wp.array[int], error: wp.array[int]):
    i = wp.tid()
    boundaries[i] = 0
    if error[0] == 0 and i < active[0]:
        v = values[i]
        order[i] = v
        boundary = int(i == 0)
        if i > 0:
            previous = values[i-1]
            boundary = int(levels[v] != levels[previous] or colors[v] != colors[previous])
        boundaries[i] = boundary


@wp.kernel
def _publish(boundaries: wp.array[int], group_scan: wp.array[int], active: wp.array[int],
              offsets: wp.array[int], metadata: wp.array[int], error: wp.array[int]):
    i = wp.tid()
    if error[0] == 0:
        if i < active[0] and boundaries[i] != 0:
            offsets[group_scan[i]-1] = i
        if i == 0:
            groups = int(0)
            if active[0] != 0:
                groups = group_scan[active[0]-1]
            metadata[1] = groups
            metadata[2] = active[0]
            offsets[groups] = active[0]


class ParallelLevels:
    def __init__(self, graph):
        self.graph, self.n, self.device = graph, graph.n, graph.device
        self.labels = [wp.empty(self.n, int, device=self.device) for _ in range(2)]
        self.level_temporary = wp.empty(self.n, int, device=self.device)
        self.seen = wp.zeros(self.n, int, device=self.device)
        self.largest = wp.empty(self.n, wp.float64, device=self.device)
        self.histogram = wp.zeros(self.n+1, int, device=self.device)
        self.boundaries = wp.zeros(self.n, int, device=self.device)
        self.group_scan = wp.empty(self.n, int, device=self.device)
        self.active = wp.zeros(1, int, device=self.device)
        self.changed = wp.zeros(1, int, device=self.device)
        self.condition = wp.zeros(1, int, device=self.device)
        self.iterations = wp.zeros(1, int, device=self.device)

    def apply(self, contacts):
        g, n, d = self.graph, self.n, self.device
        if n == 0:
            raise ValueError('Empty model is not a supported predictor epoch')
        self.seen.zero_(); self.histogram.zero_(); self.active.zero_()
        wp.launch(_initialize,n,[g.moving,g.movement,g.colors,g.trial_order,self.seen,self.labels[0],self.largest,contacts.error],device=d)
        wp.launch(_validate_rows,n,[g.rows,g.adjacency,contacts.error],device=d)
        wp.launch(_start_loop,1,[self.condition,self.iterations,contacts.error],device=d)
        def component_body():
            self.changed.zero_()
            wp.launch(_component_pass,n,[g.rows,g.adjacency,self.labels[0],self.labels[1],self.changed,contacts.error],device=d)
            wp.launch(_component_pass,n,[g.rows,g.adjacency,self.labels[1],self.labels[0],self.changed,contacts.error],device=d)
            wp.launch(_next_loop,1,[self.changed,self.condition,self.iterations,n,contacts.error],device=d)
        wp.capture_while(self.condition,component_body)
        wp.launch(_component_maximum,n,[self.labels[0],g.moving,g.movement,self.largest,contacts.error],device=d)
        wp.launch(_seed_levels,n,[self.labels[0],g.moving,g.movement,self.largest,g.levels,contacts.error],device=d)
        wp.launch(_start_loop,1,[self.condition,self.iterations,contacts.error],device=d)
        def level_body():
            self.changed.zero_()
            wp.launch(_level_pass,n,[g.rows,g.adjacency,g.levels,self.level_temporary,self.changed,contacts.error],device=d)
            wp.launch(_level_pass,n,[g.rows,g.adjacency,self.level_temporary,g.levels,self.changed,contacts.error],device=d)
            wp.launch(_next_loop,1,[self.changed,self.condition,self.iterations,n,contacts.error],device=d)
        wp.capture_while(self.condition,level_body)
        wp.launch(_active_keys,n,[g.trial_order,g.colors,g.levels,g.moving,g.keys,g.values,self.histogram,self.active,contacts.error],device=d)
        wp.utils.radix_sort_pairs(g.keys,g.values,n)
        wp.utils.array_scan(self.histogram,g.level_offsets,inclusive=True)
        wp.launch(_boundaries,n,[g.values,g.levels,g.colors,self.active,g.order,self.boundaries,contacts.error],device=d)
        wp.utils.array_scan(self.boundaries,self.group_scan,inclusive=True)
        wp.launch(_publish,n,[self.boundaries,self.group_scan,self.active,g.offsets,g.metadata,contacts.error],device=d)


wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})
