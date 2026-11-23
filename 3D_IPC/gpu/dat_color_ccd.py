"""Independent, pair-parallel CCD for each material VBD color.

The combined graph gives every pair at most one moving corner. Pair workers
reduce a separate fraction for that corner, using the selected FP32 direct or
plane proposal and its rounded-endpoint certificate. The per-color method is
independent of the level CCD predictor method.
Positions stay frozen until every pair finishes. No contact energy is added.
"""

from functools import cache

import numpy as np
import warp as wp
from newton._src.geometry.kernels import (
    TRI_CONTACT_FEATURE_FACE_INTERIOR, closest_point_line_segment, triangle_closest_point,
)

from .ccd import (configure_module, direct_pair_proposal, direct_pair_certificate,
                  direct_multi_axis_pair_proposal, direct_multi_axis_pair_certificate,
                  plane_pair_proposal, plane_pair_certificate,
                  _plane_mover, radius_float32, round_inside, get_independent_shortener)
from .ccd_defaults import MOTION_PROJECTION_POINT_GAP
from .query_groups import _unique_groups
from .warp_build import configure_module as configure_headers


@wp.func
def _closest_triangle_offset(a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """Return the closest point relative to the moving point (the origin)."""
    normal = wp.cross(b - a, c - a)
    if wp.dot(normal, normal) > 0.0:
        witness, _, feature = triangle_closest_point(a, b, c, wp.vec3(0.0))
        if feature == TRI_CONTACT_FEATURE_FACE_INTERIOR:
            # Reconstruct in the normal direction. A barycentric weighted sum
            # can introduce a spurious tangential component in a face witness.
            return normal * (wp.dot(a, normal) / wp.length_sq(normal))
        return witness

    # A collinear triangle is its longest segment, including a single point.
    ab, ac, bc = wp.length_sq(b - a), wp.length_sq(c - a), wp.length_sq(c - b)
    if ab >= ac and ab >= bc:
        if ab == 0.0:
            return a
        return closest_point_line_segment(a, b, wp.vec3(0.0))
    if ac >= bc:
        return closest_point_line_segment(a, c, wp.vec3(0.0))
    return closest_point_line_segment(b, c, wp.vec3(0.0))


@wp.func
def _point_moved(p: wp.vec3, target: wp.vec3):
    return p[0] != target[0] or p[1] != target[1] or p[2] != target[2]


@wp.func
def _motion_projection_proposal(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3],
                                nodes: wp.vec4i, kind: int):
    # The caller has already checked that exactly one corner is moving.
    p = q[nodes[0]]
    if kind == 0 and _point_moved(p, candidate[nodes[0]]):
        offset = _closest_triangle_offset(q[nodes[1]] - p, q[nodes[2]] - p, q[nodes[3]] - p)
        motion = candidate[nodes[0]] - p
        length = wp.length(motion)
        distance = wp.length(offset)
        if not wp.isfinite(distance) or not wp.isfinite(length):
            return wp.vec3(0.0, 2.0, 0.0)
        if distance <= MOTION_PROJECTION_POINT_GAP or length == 0.0:
            return wp.vec3(0.0)
        toward = wp.dot(offset, motion)
        if toward <= 0.0:
            # The initial closest point supplies a supporting plane. The
            # proposed motion moves tangentially to it or farther away.
            return wp.vec3(1.0, 0.0, 1.0)
        projection = toward / length
        alpha = wp.clamp((projection - MOTION_PROJECTION_POINT_GAP) / length, 0.0, 1.0)
        return wp.vec3(alpha, 0.0, 0.0)
    return direct_multi_axis_pair_proposal(q, candidate, nodes, kind)


@wp.func
def _motion_projection_certificate(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3],
                                   nodes: wp.vec4i, kind: int):
    # This experiment uses only the projection rule for a moving point. Other
    # mover roles retain the multi-axis control's rounded endpoint check.
    if kind == 0 and _point_moved(q[nodes[0]], candidate[nodes[0]]):
        return wp.vec3(1.0, 0.0, 1.0)
    return direct_multi_axis_pair_certificate(q, candidate, nodes, kind)


@wp.kernel(module="unique", enable_backward=False)
def _count_pairs(nodes: wp.array[wp.vec4i], count: wp.array[int], colors: wp.array[int],
                 workers: int, counts: wp.array[int], error: wp.array[int]):
    if error[0] == 0:
        if count[0] < 0 or count[0] > nodes.shape[0]:
            wp.atomic_max(error, 0, 12)
        else:
            for pair in range(wp.tid(), count[0], workers):
                keys = _unique_groups(nodes[pair], colors, error)
                for corner in range(4):
                    if keys[corner] >= 0:
                        wp.atomic_add(counts, keys[corner], 1)


@wp.kernel(module="unique", enable_backward=False)
def _fill_pairs(nodes: wp.array[wp.vec4i], count: wp.array[int], colors: wp.array[int],
                workers: int, offsets: wp.array[int], cursors: wp.array[int],
                ids: wp.array[int], error: wp.array[int]):
    if error[0] == 0:
        for pair in range(wp.tid(), count[0], workers):
            keys = _unique_groups(nodes[pair], colors, error)
            for corner in range(4):
                color = keys[corner]
                if color >= 0:
                    at = offsets[color] + wp.atomic_add(cursors, color, 1)
                    if at < offsets[color] or at >= offsets[color + 1] or at >= ids.shape[0]:
                        wp.atomic_max(error, 0, 12)
                    else:
                        ids[at] = pair


@wp.kernel(module="unique", enable_backward=False)
def _prepare(q: wp.array[wp.vec3], reference: wp.array[wp.vec3], displacement: wp.array[wp.vec3],
             center: wp.array[wp.vec3], radii: wp.array[wp.float64],
             lower: wp.array[wp.vec3], upper: wp.array[wp.vec3],
             colors: wp.array[int], current: wp.array[int], flags: wp.array[int], mass: wp.array[float],
             box_epoch: wp.array[wp.uint64], graph_epoch: wp.array[wp.uint64],
             graph_box_epoch: wp.array[wp.uint64],
             saved_box_epoch: wp.array[wp.uint64], saved_graph_epoch: wp.array[wp.uint64],
             candidate: wp.array[wp.vec3], fractions: wp.array[float],
             rejected: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    fractions[vertex] = 1.0
    rejected[vertex] = 0
    if (box_epoch[0] != saved_box_epoch[0] or graph_epoch[0] != saved_graph_epoch[0]
            or box_epoch[0] != graph_box_epoch[0]
            or current[0] < 0 or current[0] >= colors.shape[0]):
        wp.atomic_max(error, 0, 11)
    point = q[vertex]
    valid = bool(True)
    for axis in range(3):
        valid = (valid and wp.isfinite(point[axis])
                 and point[axis] >= lower[vertex][axis] and point[axis] <= upper[vertex][axis])
    if error[0] == 0 and colors[vertex] == current[0] and flags[vertex] & 1 != 0 and mass[vertex] > 0.0:
        point = round_inside(q[vertex], reference[vertex] + displacement[vertex],
                             center[vertex], radius_float32(radii[vertex]))
    for axis in range(3):
        valid = (valid and wp.isfinite(point[axis])
                 and point[axis] >= lower[vertex][axis] and point[axis] <= upper[vertex][axis])
    if not valid:
        wp.atomic_max(error, 0, 202)
    candidate[vertex] = point


@cache
def get_color_ccd_kernels(method="direct_rounding"):
    """Bind only the pair acceptance rule; scheduling and reduction are shared."""
    rules = {
        "direct_rounding": (direct_pair_proposal, direct_pair_certificate),
        "direct_rounding_multi_axis": (direct_multi_axis_pair_proposal, direct_multi_axis_pair_certificate),
        "motion_projection": (_motion_projection_proposal, _motion_projection_certificate),
        "plane": (plane_pair_proposal, plane_pair_certificate),
    }
    if method not in rules:
        raise ValueError(f"Color CCD method must be one of {', '.join(rules)}")
    pair_proposal, pair_certificate = rules[method]

    @wp.kernel(module="unique", enable_backward=False)
    def _propose(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], nodes: wp.array[wp.vec4i],
                 pair_count: wp.array[int], primitive_offsets: wp.array[int],
                 ids: wp.array[int], offsets: wp.array[int], current: wp.array[int], workers: int,
                 fractions: wp.array[float], error: wp.array[int]):
        color = current[0]
        if error[0] == 0:
            if color < 0 or color + 1 >= offsets.shape[0]:
                wp.atomic_max(error, 0, 11)
            else:
                begin, end = offsets[color], offsets[color + 1]
                if begin < 0 or end < begin or end > ids.shape[0]:
                    wp.atomic_max(error, 0, 12)
                else:
                    for at in range(begin + wp.tid(), end, workers):
                        pair = ids[at]
                        if pair < 0 or pair >= pair_count[0] or pair >= nodes.shape[0]:
                            wp.atomic_max(error, 0, 12)
                        else:
                            corners = nodes[pair]
                            mover, invalid = _plane_mover(q, candidate, corners)
                            if invalid != 0:
                                wp.atomic_max(error, 0, invalid)
                            elif mover >= 0:
                                kind = int(pair >= primitive_offsets[q.shape[0]])
                                result = pair_proposal(q, candidate, corners, kind)
                                if result[1] != 0.0:
                                    wp.atomic_max(error, 0, 202)
                                else:
                                    wp.atomic_min(fractions, mover, result[0])


    @wp.kernel(module="unique", enable_backward=False)
    def _verify(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], nodes: wp.array[wp.vec4i],
                primitive_offsets: wp.array[int], ids: wp.array[int], offsets: wp.array[int],
                current: wp.array[int], workers: int, fractions: wp.array[float],
                rejected: wp.array[int], error: wp.array[int]):
        # The propose launch validated this immutable span and its pair indices.
        if error[0] == 0:
            color = current[0]
            for at in range(offsets[color] + wp.tid(), offsets[color + 1], workers):
                pair = ids[at]
                corners = nodes[pair]
                mover, invalid = _plane_mover(q, candidate, corners)
                if invalid != 0:
                    wp.atomic_max(error, 0, invalid)
                elif mover >= 0:
                    if fractions[mover] < 1.0:
                        kind = int(pair >= primitive_offsets[q.shape[0]])
                        result = pair_certificate(q, candidate, corners, kind)
                        if result[2] != 1.0:
                            wp.atomic_max(rejected, mover, 1)
    for kernel in (_propose, _verify):
        if method in ("direct_rounding_multi_axis", "motion_projection"):
            configure_headers(kernel, ("ccd_direct_multi_axis.cuh",))
        else:
            configure_module(kernel)
    return _propose, _verify


@wp.kernel(module="unique", enable_backward=False)
def _publish(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], colors: wp.array[int],
             current: wp.array[int], rejected: wp.array[int], fractions: wp.array[float],
             lower: wp.array[wp.vec3], upper: wp.array[wp.vec3],
             reference: wp.array[wp.vec3], displacement: wp.array[wp.vec3], error: wp.array[int],
             proposed: wp.array[wp.vec3], accept_uncertified: bool):
    vertex = wp.tid()
    if error[0] == 0 and colors[vertex] == current[0]:
        point = candidate[vertex]
        moved = bool(False)
        # FP32 endpoint reconstruction must not escape candidate coverage,
        # including vertices with no incident pairs to recertify.
        for axis in range(3):
            moved = moved or point[axis] != q[vertex][axis]
            if (not wp.isfinite(point[axis])
                    or point[axis] < lower[vertex][axis]
                    or point[axis] > upper[vertex][axis]):
                rejected[vertex] = 1
        if accept_uncertified and (rejected[vertex] != 0 or (fractions[vertex] < 1.0 and not moved)):
            # _prepare already checked this endpoint against the movement box.
            point = proposed[vertex]
            fractions[vertex] = 1.0
            rejected[vertex] = 0
        if rejected[vertex] == 0:
            q[vertex] = point
        else:
            fractions[vertex] = 0.0
        # Rebase exactly rather than subtracting/adding accepted displacements.
        # Rejected motion cannot accumulate or reappear on another color.
        reference[vertex] = q[vertex]
        displacement[vertex] = wp.vec3(0.0)


class ColorCcd:
    """Preallocated pair incidence and per-vertex fractions for one VBD solver.

    Call refresh immediately after a box and combined-color update. apply only
    publishes graph.current_color, and runs entirely on the selected device.
    """

    def __init__(self, model, boxes, graph, error, *, method="direct_rounding", accept_uncertified=False):
        if not isinstance(accept_uncertified, bool):
            raise TypeError("accept_uncertified must be a bool")
        if accept_uncertified and method != "plane":
            raise ValueError("accept_uncertified requires the plane CCD method")
        self._propose, self._verify = get_color_ccd_kernels(method)
        self.method = method
        self.accept_uncertified = accept_uncertified
        if 4 * graph.capacity >= 2**31:
            raise ValueError("Color CCD pair incidence exceeds int32 capacity")
        minimum = np.nextafter(np.float32(boxes.minimum), np.float32(-np.inf))
        if not np.isfinite(minimum) or minimum <= np.float32(1e-10) or not np.isfinite(np.float32(boxes.maximum)):
            raise ValueError("Color CCD box radii must be representable in float32 above 1e-10")
        if boxes.ccd_epoch is not None or graph.ccd_epoch is not None:
            raise ValueError("Color CCD requires exclusive ownership of its box/color checkpoints")
        self.model, self.boxes, self.graph, self.error = model, boxes, graph, error
        self.device, self.n = model.device, model.particle_count
        d, n = self.device, self.n
        self.workers = max(128, d.sm_count * 128) if d.is_cuda else 128
        self.center = wp.clone(model.particle_q)
        self.radii = wp.clone(boxes.radii)
        self.candidate = wp.empty(n, wp.vec3, device=d)
        self.proposed = wp.empty_like(self.candidate) if accept_uncertified else self.candidate
        self.fractions = wp.ones(n, float, device=d)
        self.rejected = wp.zeros(n, int, device=d)
        self.pair_ids = wp.empty(4 * graph.capacity, int, device=d)
        self.offsets = wp.zeros(n + 1, int, device=d)
        self._counts = wp.zeros_like(self.offsets)
        self._cursors = wp.zeros_like(self.offsets)
        boxes.ccd_epoch = wp.zeros(1, wp.uint64, device=d)
        boxes.ccd_centers = wp.clone(model.particle_q)
        boxes.ccd_radii = wp.clone(boxes.radii)
        graph.ccd_epoch = wp.zeros(1, wp.uint64, device=d)
        graph.ccd_source_box_epoch = wp.zeros(1, wp.uint64, device=d)
        self._box_epoch = wp.zeros(1, wp.uint64, device=d)
        self._graph_epoch = wp.zeros(1, wp.uint64, device=d)
        self._ready = False
        # Establish actual checkpoint snapshots before a caller can refresh.
        boxes.update(model.particle_q)

    def refresh(self, q):
        g, d = self.graph, self.device
        if not g._built:
            raise RuntimeError("Build combined candidate coloring before ColorCcd.refresh")
        if q.dtype != wp.vec3 or q.shape != (self.n,) or q.device != d:
            raise TypeError("Color CCD requires complete model position arrays")
        g.validate_independence()
        wp.copy(self.center, self.boxes.ccd_centers)
        wp.copy(self.radii, self.boxes.ccd_radii)
        wp.copy(self._box_epoch, self.boxes.ccd_epoch)
        wp.copy(self._graph_epoch, g.ccd_epoch)
        self._counts.zero_()
        self._cursors.zero_()
        if g.capacity:
            args = [g.candidate_nodes, g.candidate_count, g.colors, self.workers]
            wp.launch(_count_pairs, self.workers, args + [self._counts, self.error], device=d)
        wp.utils.array_scan(self._counts, self.offsets, inclusive=False)
        if g.capacity:
            wp.launch(_fill_pairs, self.workers,
                      args + [self.offsets, self._cursors, self.pair_ids, self.error], device=d)
        self._ready = True

    def apply(self, q, reference, displacement):
        if not self._ready:
            raise RuntimeError("Refresh color CCD before its first update")
        for array in (q, reference, displacement):
            if array.dtype != wp.vec3 or array.shape != (self.n,) or array.device != self.device:
                raise TypeError("Color CCD requires complete model position arrays")
        if self.n == 0:
            return
        b, g, d = self.boxes, self.graph, self.device
        wp.launch(_prepare, self.n,
                  [q, reference, displacement, self.center, self.radii, b.node_lower, b.node_upper,
                   g.colors, g.current_color, self.model.particle_flags, self.model.particle_mass,
                   b.ccd_epoch, g.ccd_epoch, g.ccd_source_box_epoch, self._box_epoch, self._graph_epoch,
                   self.candidate, self.fractions, self.rejected, self.error], device=d)
        if self.accept_uncertified:
            wp.copy(self.proposed, self.candidate)
        if g.capacity:
            wp.launch(self._propose, self.workers,
                      [q, self.candidate, g.candidate_nodes, g.candidate_count, g.candidate_offsets,
                       self.pair_ids, self.offsets, g.current_color, self.workers,
                       self.fractions, self.error], device=d, block_dim=64)
        wp.launch(get_independent_shortener(), self.n,
                  [q, self.candidate, self.fractions, self.error], device=d)
        if g.capacity:
            wp.launch(self._verify, self.workers,
                      [q, self.candidate, g.candidate_nodes, g.candidate_offsets, self.pair_ids,
                       self.offsets, g.current_color, self.workers, self.fractions,
                       self.rejected, self.error], device=d, block_dim=64)
        wp.launch(_publish, self.n,
                  [q, self.candidate, g.colors, g.current_color, self.rejected,
                   self.fractions, b.node_lower, b.node_upper, reference, displacement, self.error,
                   self.proposed, self.accept_uncertified], device=d)


for _kernel in (_count_pairs, _fill_pairs, _prepare, _publish):
    configure_module(_kernel)
