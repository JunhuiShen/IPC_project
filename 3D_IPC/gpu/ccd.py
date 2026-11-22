"""Represented-trajectory CCD, predictors' acceptance, and GS independent steps.

Trajectory certificates, backtracking and interpolation use FP32. Direct
rounding certifies the proposed fraction and all shorter rounded prefixes.
Its optional endpoint recheck is controlled by the exact fallback scene flag.
Legacy resolver names remain API aliases.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from .contact import decode_candidate
from .warp_build import configure_module as _configure, flatten_headers


HEADERS = ("ccd_plane_fp32.cuh", "ccd_direct_fp32.cuh")
PairResult = wp.vec3  # FP32 fraction, status, full represented path


def _preamble(mode, include_filter):
    if mode not in ("fast", "original", "cached", "bound", "single_vertex", "float32"):
        raise ValueError("Unknown CCD compilation mode")
    return flatten_headers(HEADERS)[0]


def configure_module(module=None, *, include_filter=True, mode="float32"):
    """Compile only the FP32 CCD arithmetic, including for legacy mode names."""
    return _configure(__name__ if module is None else module, (),
                      extra_preamble=_preamble(mode, include_filter))


@wp.func_native(r"""
    // Shared broad-phase radii are stored in binary64. This boundary performs
    // only a conversion, directed inward, before any CCD arithmetic begins.
#if defined(__CUDA_ARCH__)
    return __double2float_rd(radius);
#else
    const float value=static_cast<float>(radius);
    if(!ipc_gpu::ccd32::detail::finite(value))return value;
    return ipc_gpu::ccd32::detail::neighbor(value,false);
#endif
""")
def radius_float32(radius: wp.float64) -> float: ...


@wp.func_native(r"""
    using namespace ipc_gpu::ccd32::detail;
    wp::vec_t<3,float> result;
    if(!finite(radius) || radius<=1e-10f) {
        for(int k=0;k<3;++k)result[k]=NAN;
        return result;
    }
    for(int k=0;k<3;++k) {
        // Inward-rounded bounds enclose only points inside the shared box.
        const auto low=subtract(point(center[k]),point(radius));
        const auto high=add(point(center[k]),point(radius));
        const float lo=add(point(low.hi),point(1e-10f)).hi;
        const float hi=subtract(point(high.lo),point(1e-10f)).lo;
        if(!finite(current[k]) || !finite(proposed[k]) || !finite(center[k])) {
            result[k]=NAN;continue;
        }
        if(!(lo<=hi) || current[k]<low.hi || current[k]>high.lo) {
            // A radius below the coordinate ULP can leave no provable inner
            // endpoint. The exact represented center is still inside the box.
            result[k]=same_value(current[k],center[k])?current[k]:NAN;
            continue;
        }
        result[k]=maximum(lo,minimum(hi,proposed[k]));
    }
    return result;
""")
def round_inside(current: wp.vec3, proposed: wp.vec3, center: wp.vec3, radius: float) -> wp.vec3: ...


@wp.func_native(r"""
    wp::vec_t<3,float> result;
    for(int k=0;k<3;++k)
        result[k]=ipc_gpu::ccd32::detail::interpolate(start[k],end[k],alpha);
    return result;
""")
def interpolate_positions(start: wp.vec3, end: wp.vec3, alpha: float) -> wp.vec3: ...


_INPUT = r"""
    wp::vec_t<3,float> result;
    result[0]=1.0f;result[1]=0.0f;result[2]=0.0f;
    ipc_gpu::ccd32::LinearQuery query;
    query.vertex_face=kind==0;
    if(kind!=0&&kind!=1) {result[1]=2.0f;return result;}
    for(int i=0;i<4;++i) {
        if(nodes[i]<0 || nodes[i]>=positions.shape[0] || nodes[i]>=proposed.shape[0]) {
            result[1]=2.0f;return result;
        }
        for(int k=0;k<3;++k) {
            query.start[i][k]=positions.data[nodes[i]][k];
            query.end[i][k]=proposed.data[nodes[i]][k];
        }
    }
"""


@wp.func_native(_INPUT + r"""
    const auto status=ipc_gpu::ccd32::validate_linear(query);
    if(status==ipc_gpu::ccd32::Status::Invalid)result[1]=2.0f;
    else if(status==ipc_gpu::ccd32::Status::Safe)result[2]=1.0f;
    else result[1]=5.0f;
    return result;
""")
def pair_certificate(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
                     kind: int, exact_geometry: int) -> PairResult: ...


@wp.kernel(module="unique", enable_backward=False)
def _prepare(q: wp.array[wp.vec3], proposed: wp.array[wp.vec3], center: wp.array[wp.vec3],
              radius: float, radii: wp.array[wp.float64], use_radii: int,
              bounded: int, colors: wp.array[int], flags: wp.array[int], mass: wp.array[float],
              color: int, candidate: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    if error[0] == 0:
        point = q[i]
        if flags[i] & 1 != 0 and mass[i] > 0.0 and (color < 0 or colors[i] == color):
            point = proposed[i]
        changed = point[0] != q[i][0] or point[1] != q[i][1] or point[2] != q[i][2]
        if bounded != 0 and changed:
            budget = radius
            if use_radii != 0:
                budget = radius_float32(radii[i])
            point = round_inside(q[i], point, center[i], budget)
        for k in range(3):
            if not wp.isfinite(q[i][k]) or not wp.isfinite(point[k]):
                wp.atomic_max(error, 0, 202)
        candidate[i] = point





@wp.kernel(module="unique", enable_backward=False)
def _begin(condition: wp.array[int], attempts: wp.array[int], verified: wp.array[int], error: wp.array[int]):
    attempts[0] = 0
    verified[0] = 0
    condition[0] = int(error[0] == 0)


@wp.kernel(module="unique", enable_backward=False)
def _begin_round(fraction: wp.array[float], pending_count: wp.array[int]):
    fraction[0] = float(1.0)
    pending_count[0] = 0


@wp.kernel(module="unique", enable_backward=False)
def _certify(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], ids: wp.array[int], count: wp.array[int],
              triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
              exact_geometry: int, workers: int, pending: wp.array[int], pending_count: wp.array[int], error: wp.array[int]):
    worker = wp.tid()
    if error[0] == 0:
        for at in range(worker, count[0], workers):
            nodes, kind = decode_candidate(ids[at], triangles, edges, info)
            result = pair_certificate(q, candidate, nodes, kind, exact_geometry)
            if int(result[1]) == 5:
                slot = wp.atomic_add(pending_count, 0, 1)
                if slot < pending.shape[0]:
                    pending[slot] = at
                else:
                    wp.atomic_max(error, 0, 12)
            elif int(result[1]) != 0:
                wp.atomic_max(error, 0, 200 + int(result[1]))





@wp.kernel(module="unique", enable_backward=False)
def _shorten(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], fraction: wp.array[float], error: wp.array[int]):
    i = wp.tid()
    alpha = fraction[0]
    if error[0] == 0 and alpha < float(1.0):
        candidate[i] = interpolate_positions(q[i], candidate[i], alpha)


@wp.kernel(module="unique", enable_backward=False)
def _next(condition: wp.array[int], attempts: wp.array[int], verified: wp.array[int],
           fraction: wp.array[float], maximum: int, error: wp.array[int]):
    attempts[0] += 1
    if error[0] != 0:
        condition[0] = 0
    elif fraction[0] == float(1.0):
        verified[0] = 1
        condition[0] = 0
    elif attempts[0] >= maximum:
        wp.atomic_max(error, 0, 204)
        condition[0] = 0
    else:
        condition[0] = 1


@wp.kernel(module="unique", enable_backward=False)
def _commit(candidate: wp.array[wp.vec3], q: wp.array[wp.vec3], verified: wp.array[int], error: wp.array[int]):
    if error[0] == 0 and verified[0] != 0:
        i = wp.tid()
        q[i] = candidate[i]


@wp.kernel(module="unique", enable_backward=False)
def _apply_direct_fraction(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], fractions: wp.array[float],
                   rejected: wp.array[int], attempts: wp.array[int],
                   verified: wp.array[int], publish: bool, error: wp.array[int]):
    i = wp.tid()
    if i == 0:
        attempts[0] = 1
        verified[0] = 1
    if i < q.shape[0]:
        full_step = rejected[i] == 0
        rejected[i] = 0  # reused for the optional shortened-endpoint check
        proposal = candidate[i]
        if error[0] == 0:
            point = proposal
            if full_step:
                fractions[i] = 1.0
            elif fractions[i] < 1.0:
                point = interpolate_positions(q[i], proposal, fractions[i])
            candidate[i] = point
            if publish:
                q[i] = point


class CcdBackend:
    def __init__(self, contacts, *, node_box_radius=0.01, exact_geometry=True, max_rounds=128,
                 exact_step_bound=None, error=None, group_query_block_dim=32):
        if not math.isfinite(node_box_radius) or node_box_radius <= 1.0e-10:
            raise ValueError("The movement-box radius must exceed its 1e-10 inset")
        if not isinstance(exact_geometry, bool) or isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
            raise ValueError("Invalid CCD mode or retry budget")
        if not isinstance(group_query_block_dim, int) or isinstance(group_query_block_dim, bool) or group_query_block_dim not in (32, 64, 128, 256):
            raise ValueError("Grouped query block size must be 32, 64, 128 or 256")
        if exact_step_bound is None:
            options = getattr(getattr(contacts.model, "ipc", None), "params", {})
            exact_step_bound = options.get("ccd_exact_step_bound", False)
        if not isinstance(exact_step_bound, bool) or exact_step_bound and not exact_geometry:
            raise ValueError("Exact-step proposals require robust CCD")
        self.contacts, self.model, self.device = contacts, contacts.model, contacts.device
        self.method = getattr(getattr(self.model, "ipc", None), "params", {}).get("ccd_method", "interval")
        if self.method not in ("interval", "plane", "direct_rounding"):
            raise ValueError("CCD method must be interval, plane, or direct_rounding")
        self.radius, self.exact_geometry, self.max_rounds = node_box_radius, exact_geometry, max_rounds
        self.precision = "float32"
        self.exact_step_bound = exact_step_bound
        self.group_query_block_dim = group_query_block_dim
        self.single_vertex_filter = False
        self.error = error if error is not None else contacts.error
        self.candidate = wp.empty(self.model.particle_count, wp.vec3, device=self.device)
        self.reference = wp.clone(self.model.particle_q)
        self.condition = wp.zeros(1, int, device=self.device)
        self.attempts = wp.zeros(1, int, device=self.device)
        self.verified = wp.zeros(1, int, device=self.device)
        self.fraction = wp.ones(1, float, device=self.device)
        self.node_fractions = wp.ones(self.model.particle_count, float, device=self.device)
        self.plane_rejected = wp.zeros(self.model.particle_count, int, device=self.device) if self.method != "interval" else None
        self._empty_radii = wp.empty(0, wp.float64, device=self.device)
        self.pending = wp.empty(contacts.capacity, int, device=self.device)
        self.pending_count = wp.zeros(1, int, device=self.device)
        self._resolve = get_resolver("bound" if exact_step_bound else "original")

    def set_reference(self, q):
        wp.copy(self.reference, q)

    def accept(self, q, proposed, color, dt, *, center=None, radius=None, bounded=True, colors=None, preserve_gap=False,
               clip_ccd=True, query_view=None, radii=None, current_color=None, independent=False,
               unclipped=None):
        if q.dtype != wp.vec3 or proposed.dtype != wp.vec3 or q.shape != proposed.shape:
            raise TypeError("CCD requires matching upstream float32 particle arrays")
        if unclipped is not None:
            if not (independent and clip_ccd):
                raise ValueError("Saving pre-CCD proposals requires independent CCD updates")
            if (unclipped.dtype != wp.vec3 or unclipped.shape != q.shape or unclipped.device != self.device
                    or unclipped.ptr in (q.ptr, proposed.ptr, self.candidate.ptr)):
                raise ValueError("Pre-CCD proposals require separate float32 storage")
        radius = self.radius if radius is None else radius
        if not math.isfinite(radius) or radius <= 1.0e-10:
            raise ValueError("Invalid CCD movement radius")
        # Host configuration conversion must not enlarge a movement box.
        radius32 = np.float32(radius)
        if not np.isfinite(radius32):
            raise ValueError("CCD movement radius must be representable in float32")
        if float(radius32) > radius:
            radius32 = np.nextafter(radius32, np.float32(-np.inf))
        reference = self.reference if center is None else center
        if radii is not None and (radii.dtype != wp.float64 or radii.shape != q.shape or radii.device != self.device):
            raise TypeError("CCD per-node radii must match the model's device float64 layout")
        radius_array = self._empty_radii if radii is None else radii
        if current_color is not None and (query_view is None or current_color is not query_view.current_group):
            raise ValueError("Device-selected CCD colors require their matching grouped query view")
        if independent and query_view is None:
            raise ValueError("Independent GS acceptance requires a complete grouped candidate view")
        if clip_ccd and self.method != "interval" and not independent:
            raise ValueError("Direct CCD requires contact-independent grouped vertex updates")
        selected_colors = self.model.particle_colors if colors is None else colors
        c = self.contacts
        ids = c.candidate_ids
        if query_view is not None:
            from .query_groups import prepare_grouped, certify_grouped
            if q.shape[0] != self.model.particle_count:
                raise TypeError("Grouped CCD requires complete model position arrays")
            query_view.validate(c)
            ids, offsets, current_group, view_groups = (query_view.ids, query_view.offsets,
                query_view.current_group, query_view.node_groups)
            view_epoch = query_view.epoch
            wp.launch(prepare_grouped, q.shape[0],
                [q, proposed, reference, float(radius32), radius_array, int(radii is not None),
                 int(bounded), selected_colors, self.model.particle_flags,
                 self.model.particle_mass, int(color), view_groups, current_group, 1, int(current_color is not None), view_epoch, c.query_epoch,
                 self.candidate, self.node_fractions if clip_ccd and self.method != "interval" else None,
                 self.error], device=self.device)
        else:
            wp.launch(_prepare, q.shape[0],
                [q, proposed, reference, float(radius32), radius_array, int(radii is not None),
                 int(bounded), selected_colors, self.model.particle_flags,
                 self.model.particle_mass, int(color), self.candidate, self.error], device=self.device)
        if not clip_ccd:
            self.verified.fill_(1)
            wp.launch(_commit, q.shape[0], [self.candidate, q, self.verified, self.error], device=self.device)
            return q
        direct_commit = self.method == "direct_rounding" and not self.exact_geometry
        # Optional output for callers inspecting the proposal before CCD.
        if unclipped is not None:
            wp.copy(unclipped, self.candidate)
        if self.method != "interval":
            propose, verify, publish = get_plane_kernels(self.method)
            args = [q, self.candidate, ids, self.model.tri_indices, self.model.edge_indices,
                    c.info, offsets, current_group, c.fast_workers, self.node_fractions]
            self.plane_rejected.zero_()
            if c.capacity:
                wp.launch(propose, c.fast_workers, args + [self.plane_rejected, self.error], device=self.device,
                          block_dim=self.group_query_block_dim)
            if self.method == "direct_rounding":
                wp.launch(_apply_direct_fraction, max(1, q.shape[0]),
                          [q, self.candidate, self.node_fractions, self.plane_rejected,
                           self.attempts, self.verified, direct_commit, self.error], device=self.device)
                if direct_commit:
                    return q
            else:
                wp.launch(get_independent_shortener(), q.shape[0],
                          [q, self.candidate, self.node_fractions, self.error], device=self.device)
                self.attempts.fill_(1)
                self.verified.fill_(1)
            if c.capacity:
                wp.launch(verify, c.fast_workers, args + [self.plane_rejected, self.error],
                          device=self.device, block_dim=self.group_query_block_dim)
            # Independent candidates have at most one mover, so dropping any
            # rejected vertex restores that pair's reference without affecting
            # certificates for other moving vertices in this group.
            wp.launch(publish, q.shape[0], [q, self.candidate, self.plane_rejected, self.error], device=self.device)
            return q
        wp.launch(_begin, 1, [self.condition, self.attempts, self.verified, self.error], device=self.device)
        if independent:
            independent_resolve = get_independent_resolver(self.exact_step_bound)
            independent_certify = get_independent_certifier()
            independent_shorten = get_independent_shortener()

        def round_body():
            wp.launch(_begin_round, 1, [self.fraction, self.pending_count], device=self.device)
            if independent:
                self.node_fractions.fill_(1.0)
            if c.capacity:
                if independent:
                    wp.launch(independent_certify, c.fast_workers,
                        [q, self.candidate, ids, self.model.tri_indices, self.model.edge_indices,
                         c.info, offsets, current_group, int(self.exact_geometry), int(self.single_vertex_filter), c.fast_workers,
                         self.pending, self.pending_count, self.error], device=self.device,
                        block_dim=self.group_query_block_dim)
                elif query_view is not None:
                    wp.launch(certify_grouped, c.fast_workers,
                        [q, self.candidate, ids, c.candidate_count, self.model.tri_indices,
                         self.model.edge_indices, c.info, offsets, current_group, 1,
                         int(self.exact_geometry), c.fast_workers,
                         self.pending, self.pending_count, self.error], device=self.device,
                        block_dim=self.group_query_block_dim)
                else:
                    wp.launch(_certify, c.fast_workers,
                        [q, self.candidate, ids, c.candidate_count, self.model.tri_indices,
                         self.model.edge_indices, c.info, int(self.exact_geometry), c.fast_workers,
                         self.pending, self.pending_count, self.error], device=self.device)
                resolve = independent_resolve if independent else self._resolve
                reductions = [self.node_fractions, self.fraction] if independent else [self.fraction]
                wp.launch(resolve, c.fast_workers,
                    [q, self.candidate, ids, self.model.tri_indices, self.model.edge_indices,
                     c.info, int(self.exact_geometry), int(preserve_gap), c.words, c.slots, c.fast_workers, self.pending,
                     self.pending_count, *reductions, self.error], device=self.device, block_dim=self.group_query_block_dim)
            if independent:
                wp.launch(independent_shorten, q.shape[0],
                    [q, self.candidate, self.node_fractions, self.error], device=self.device)
            else:
                wp.launch(_shorten, q.shape[0], [q, self.candidate, self.fraction, self.error], device=self.device)
            wp.launch(_next, 1,
                [self.condition, self.attempts, self.verified, self.fraction, self.max_rounds, self.error], device=self.device)

        wp.capture_while(self.condition, round_body)
        wp.launch(_commit, q.shape[0], [self.candidate, q, self.verified, self.error], device=self.device)
        return q

    def initial_guess(self, q, previous, velocity, inertia, dt, params, *, detector=None):
        from .policies import InitialGuess
        if not hasattr(self, "_initial_guess"):
            # Construct before graph capture. Parent normally calls prepare()
            # or creates this policy explicitly in SolverIPC.__init__.
            self._initial_guess = InitialGuess(self, detector)
        return self._initial_guess.apply(q, previous, velocity, inertia, dt, params, detector=detector)

    def prepare_initial_guess(self, detector, params=None, *, use_group_queries=False, independent_colors=False):
        from .policies import InitialGuess
        self._initial_guess = InitialGuess(self, detector, params=params, use_group_queries=use_group_queries,
                                          independent_colors=independent_colors)

    def check(self):
        code = int(self.error.numpy()[0])
        if 200 <= code < 300:
            names = {201: "initial intersection", 202: "invalid represented motion", 203: "invalid CCD arithmetic",
                     204: "CCD rounded-endpoint reproof exhausted", 205: "contact-dependent GS batch"}
            raise RuntimeError("IPC CCD: " + names.get(code, str(code)))
        self.contacts.check()


# Legacy mode names all compile the same FP32 conservative backtracking.
# Scratch arguments are retained for existing callers, but never accessed.
@cache
def get_pair_clip(mode="original"):
    if mode not in ("original", "cached", "bound", "float32"):
        raise ValueError("Unknown CCD clipping mode")

    @wp.func_native(_INPUT + r"""
        const auto out=ipc_gpu::ccd32::clip_linear(query);
        result[0]=out.fraction;
        result[1]=out.status==ipc_gpu::ccd32::Status::Invalid?2.0f:0.0f;
        result[2]=out.full_step?1.0f:0.0f;
        return result;
    """)
    def clip(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
             kind: int, exact_geometry: int, preserve_gap: int, words: wp.array[wp.uint32],
             slots: wp.array[int], worker: int) -> PairResult: ...
    return clip


@cache
def get_resolver(mode="original"):
    """An isolated FP32 resolver; legacy mode names select identical arithmetic."""
    pair_clip = get_pair_clip(mode)

    @wp.kernel(module="unique", enable_backward=False)
    def resolve(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], ids: wp.array[int],
                triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                exact_geometry: int, preserve_gap: int, words: wp.array[wp.uint32], slots: wp.array[int],
                workers: int, pending: wp.array[int], pending_count: wp.array[int],
                fraction: wp.array[float], error: wp.array[int]):
        worker = wp.tid()
        if error[0] == 0:
            for index in range(worker, pending_count[0], workers):
                nodes, kind = decode_candidate(ids[pending[index]], triangles, edges, info)
                result = pair_clip(q, candidate, nodes, kind, exact_geometry, preserve_gap, words, slots, worker)
                if int(result[1]) != 0:
                    wp.atomic_max(error, 0, 200 + int(result[1]))
                else:
                    wp.atomic_min(fraction, 0, result[0])

    configure_module(resolve, include_filter=False, mode=mode)
    return resolve


@cache
def get_independent_resolver(exact_step_bound=False):
    mode = "bound" if exact_step_bound else "cached"
    pair_clip = get_pair_clip(mode)

    @wp.kernel(module="unique", enable_backward=False)
    def resolve(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], ids: wp.array[int],
                triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                exact_geometry: int, preserve_gap: int, words: wp.array[wp.uint32],
                slots: wp.array[int], workers: int, pending: wp.array[int], pending_count: wp.array[int],
                node_fractions: wp.array[float], fraction: wp.array[float], error: wp.array[int]):
        worker = wp.tid()
        if error[0] == 0:
            for index in range(worker, pending_count[0], workers):
                nodes, kind = decode_candidate(ids[pending[index]], triangles, edges, info)
                result = pair_clip(q, candidate, nodes, kind, exact_geometry, preserve_gap, words, slots, worker)
                if int(result[1]) != 0:
                    wp.atomic_max(error, 0, 200 + int(result[1]))
                else:
                    for role in range(4):
                        node = nodes[role]
                        changed = False
                        for k in range(3):
                            changed = changed or candidate[node][k] != q[node][k]
                        if changed:
                            wp.atomic_min(node_fractions, node, result[0])
                    wp.atomic_min(fraction, 0, result[0])

    configure_module(resolve, include_filter=False, mode=mode)
    return resolve


@cache
def get_independent_certifier():
    @wp.kernel(module="unique", enable_backward=False)
    def certify(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], ids: wp.array[int],
                triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                offsets: wp.array[int], current: wp.array[int], exact_geometry: int, use_filter: int, workers: int,
                pending: wp.array[int], pending_count: wp.array[int], error: wp.array[int]):
        group = current[0]
        if error[0] == 0:
            if group < 0 or group + 1 >= offsets.shape[0]:
                wp.atomic_max(error, 0, 11)
            else:
                begin = offsets[group]
                end = offsets[group + 1]
                if begin < 0 or end < begin or end > ids.shape[0]:
                    wp.atomic_max(error, 0, 12)
                else:
                    for at in range(begin + wp.tid(), end, workers):
                        nodes, kind = decode_candidate(ids[at], triangles, edges, info)
                        moved = int(0)
                        valid = True
                        for role in range(4):
                            node = nodes[role]
                            changed = False
                            if node < 0 or node >= q.shape[0]:
                                valid = False
                                wp.atomic_max(error, 0, 202)
                            else:
                                for k in range(3):
                                    changed = changed or candidate[node][k] != q[node][k]
                            moved += int(changed)
                        if moved > 1:
                            wp.atomic_max(error, 0, 205)
                        elif valid and moved == 1:
                            proved = False
                            if use_filter != 0:
                                proved = single_vertex_safe(q, candidate, nodes, kind)
                            if not proved:
                                result = pair_certificate(q, candidate, nodes, kind, exact_geometry)
                                if int(result[1]) == 5:
                                    slot = wp.atomic_add(pending_count, 0, 1)
                                    if slot < pending.shape[0]:
                                        pending[slot] = at
                                    else:
                                        wp.atomic_max(error, 0, 12)
                                elif int(result[1]) != 0:
                                    wp.atomic_max(error, 0, 200 + int(result[1]))

    configure_module(certify, mode="single_vertex")
    return certify


@cache
def get_independent_shortener():
    @wp.kernel(module="unique", enable_backward=False)
    def shorten(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3],
                node_fractions: wp.array[float], error: wp.array[int]):
        i = wp.tid()
        alpha = node_fractions[i]
        if error[0] == 0 and alpha < float(1.0):
            candidate[i] = interpolate_positions(q[i], candidate[i], alpha)

    configure_module(shorten, mode="fast")
    return shorten


@wp.func_native(_INPUT + r"""
    const float fraction=ipc_gpu::ccd32plane::propose_fraction(query);
    if(fraction<0.0f)result[1]=2.0f;
    else result[0]=fraction;
    return result;
""")
def plane_pair_proposal(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
                        kind: int) -> PairResult: ...


@wp.func_native(_INPUT + r"""
    const auto status=ipc_gpu::ccd32plane::validate_linear(query);
    result[2]=status==ipc_gpu::ccd32::Status::Safe?1.0f:0.0f;
    return result;
""")
def plane_pair_certificate(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
                           kind: int) -> PairResult: ...


@wp.func_native(_INPUT + r"""
    const float fraction=ipc_gpu::ccd32direct::propose_fraction(query);
    if(fraction<0.0f)result[1]=2.0f;
    else result[0]=fraction;
    return result;
""")
def direct_pair_proposal(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
                        kind: int) -> PairResult: ...


@wp.func_native(_INPUT + r"""
    bool full = false;
    float prefix = ipc_gpu::ccd32direct::propose_fraction(query, &full);
    if (prefix == 0.0f) {
        bool refined_full = false;
        prefix = ipc_gpu::ccd_clearance::grouped_fraction(query, refined_full);
        full = full || refined_full;
    }
    if (prefix < 0.0f) result[1] = 2.0f;
    else {
        result[0] = prefix;
        result[2] = full ? 1.0f : 0.0f;
    }
    return result;
""")
def direct_group_proposal(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
                          kind: int) -> PairResult: ...


@wp.func_native(_INPUT + r"""
    const auto status=ipc_gpu::ccd32direct::validate_linear(query);
    result[2]=status==ipc_gpu::ccd32::Status::Safe?1.0f:0.0f;
    return result;
""")
def direct_pair_certificate(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
                           kind: int) -> PairResult: ...


@wp.func
def _plane_mover(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], nodes: wp.vec4i):
    mover = int(-1)
    invalid = int(0)
    for role in range(4):
        node = nodes[role]
        if node < 0 or node >= q.shape[0]:
            invalid = 202
        else:
            changed = False
            for k in range(3):
                changed = changed or q[node][k] != candidate[node][k]
            if changed:
                if mover >= 0:
                    invalid = 205
                mover = node
    return mover, invalid


@cache
def get_plane_kernels(method="plane"):
    if method not in ("plane", "direct_rounding"):
        raise ValueError("Expected a bounded direct CCD method")
    direct = method == "direct_rounding"
    pair_proposal = direct_group_proposal if direct else plane_pair_proposal
    pair_certificate = plane_pair_certificate if method == "plane" else direct_pair_certificate
    @wp.kernel(module="unique", enable_backward=False)
    def propose(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], ids: wp.array[int],
                triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                offsets: wp.array[int], current: wp.array[int], workers: int,
                fractions: wp.array[float], full_rejected: wp.array[int], error: wp.array[int]):
        group = current[0]
        if error[0] == 0:
            if group < 0 or group + 1 >= offsets.shape[0]:
                wp.atomic_max(error, 0, 11)
            else:
                begin, end = offsets[group], offsets[group + 1]
                if begin < 0 or end < begin or end > ids.shape[0]:
                    wp.atomic_max(error, 0, 12)
                else:
                    for at in range(begin + wp.tid(), end, workers):
                        nodes, kind = decode_candidate(ids[at], triangles, edges, info)
                        mover, invalid = _plane_mover(q, candidate, nodes)
                        if invalid != 0:
                            wp.atomic_max(error, 0, invalid)
                        elif mover >= 0:
                            result = pair_proposal(q, candidate, nodes, kind)
                            if result[1] != 0.0:
                                wp.atomic_max(error, 0, 202)
                            else:
                                if result[0] < 1.0:
                                    wp.atomic_min(fractions, mover, result[0])
                                if wp.static(direct):
                                    if result[2] == 0.0:
                                        wp.atomic_max(full_rejected, mover, 1)

    @wp.kernel(module="unique", enable_backward=False)
    def verify(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], ids: wp.array[int],
               triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
               offsets: wp.array[int], current: wp.array[int], workers: int,
               fractions: wp.array[float], rejected: wp.array[int], error: wp.array[int]):
        # propose has checked this view, and its offsets are immutable between
        # the two launches. A full fraction reused the exact original endpoint;
        # only reconstructed, shortened endpoints need a second certificate.
        if error[0] == 0:
            group = current[0]
            for at in range(offsets[group] + wp.tid(), offsets[group + 1], workers):
                nodes, kind = decode_candidate(ids[at], triangles, edges, info)
                mover, invalid = _plane_mover(q, candidate, nodes)
                if invalid != 0:
                    wp.atomic_max(error, 0, invalid)
                elif mover >= 0:
                    if fractions[mover] < 1.0:
                        result = pair_certificate(q, candidate, nodes, kind)
                        if result[2] != 1.0:
                            wp.atomic_max(rejected, mover, 1)

    @wp.kernel(module="unique", enable_backward=False)
    def publish(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], rejected: wp.array[int], error: wp.array[int]):
        i = wp.tid()
        if error[0] == 0:
            if rejected[i] == 0:
                q[i] = candidate[i]

    for kernel in (propose, verify, publish):
        configure_module(kernel)
    if direct:
        _configure(propose, ("ccd_clearance.cuh",))
    return propose, verify, publish


@wp.func_native(_INPUT.replace("return result;", "return false;") + r"""
    return ipc_gpu::ccd32::detail::valid(query)
        && ipc_gpu::ccd32::detail::separated(query,0.0f,1.0f,query.gap_floor);
""")
def single_vertex_safe(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
                       kind: int) -> bool: ...


def install_our_instance(solver):
    """Explicit external-probe opt-in, before building/capturing its FrameGraph."""
    from .solver import SolverIPC
    original_resolve = get_resolver("original")
    if type(solver) is not SolverIPC:
        raise TypeError("The optional cold cache applies only to an actual SolverIPC instance")
    backend = solver.ccd_backend
    if backend.exact_step_bound or backend._resolve is not original_resolve:
        raise ValueError("The optional cold cache requires the original linear clip resolver")
    if wp.get_device(backend.device).is_capturing:
        raise RuntimeError("Install the optional cold cache before graph capture")
    backend._resolve = get_resolver("cached")
    return original_resolve


def __getattr__(name):
    clips = {"pair_clip": "original", "pair_clip_cached": "cached", "pair_clip_bound": "bound"}
    resolvers = {"resolve_original": "original", "resolve_cached": "cached", "resolve_bound": "bound"}
    if name in clips:
        return get_pair_clip(clips[name])
    if name in resolvers:
        return get_resolver(resolvers[name])
    raise AttributeError(name)


# Fast kernels share this source file but never a cold compilation unit.
for _kernel in (_prepare, _begin, _begin_round, _certify, _shorten, _next, _commit, _apply_direct_fraction):
    configure_module(_kernel, mode="fast")
