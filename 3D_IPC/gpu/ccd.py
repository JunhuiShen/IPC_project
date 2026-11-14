"""Represented-trajectory CCD, predictors' acceptance, and GS independent steps.

Original, cached and direct-bound cold kernels are created in isolated Warp
modules. Fast certificates, box clipping and per-node reductions share this
source file while keeping their compilation separate from exact arithmetic.
Every rounded shortened endpoint is certified again before commit.
"""

from __future__ import annotations

import math
import re
from functools import cache
from pathlib import Path

import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from .contact import decode_candidate
from .warp_build import configure_module as _configure, flatten_headers


HEADERS = ("ccd.cuh", "exact.cuh", "original_ccd.cuh", "original_rigid.cuh", "global_guess.cuh")
PairResult = wp.vec3d  # fraction, status, full represented path


def _preamble(mode, include_filter):
    if mode not in ("fast", "original", "cached", "bound", "single_vertex"):
        raise ValueError("Unknown CCD compilation mode")
    core = Path(__file__).with_suffix(".cuh").read_text(encoding="utf-8")
    sections = {}
    # Shared math headers include ccd.cuh themselves. Expand them against its
    # core only, then append selected helpers after their prerequisite types.
    # This keeps the physical header standalone without guarded include cycles
    # in Warp's stricter source expander.
    for section in ("cache", "single_vertex"):
        pattern = (r"// IPC_CCD_SECTION_BEGIN " + section + r"\n.*?"
                   r"// IPC_CCD_SECTION_END " + section + r"(?:\n|$)")
        matches = re.findall(pattern, core, flags=re.DOTALL)
        if len(matches) != 1:
            raise RuntimeError("CCD source section markers changed: " + section)
        sections[section] = matches[0]
        core = re.sub(pattern, "", core, flags=re.DOTALL)
    headers = ["ccd.cuh"]
    if mode in ("original", "cached", "bound"):
        headers.extend(HEADERS[1:])
    if mode == "bound":
        headers.append("direct_bound.cuh")
    if include_filter or mode == "single_vertex":
        headers.append("separation.cuh")
    source, _ = flatten_headers(headers, source_overrides={"ccd.cuh": core})
    for section, selected in (("cache", mode == "cached"),
                              ("single_vertex", mode == "single_vertex")):
        if selected:
            # These prerequisite includes were expanded above. No other
            # includes are permitted inside the selected helper fragments.
            fragment = sections[section]
            for header in ("exact.cuh", "separation.cuh"):
                fragment = fragment.replace('#include "' + header + '"\n', "")
            if re.search(r"^\s*#\s*include\b", fragment, flags=re.MULTILINE):
                raise RuntimeError("Unexpected CCD helper dependency: " + section)
            source += "\n" + fragment
    return source


def configure_module(module=None, *, include_filter=True, mode="original"):
    """Configure one caller without merging fast and cold compilation units.

    The default retains the original public helper's complete math preamble.
    Production fast and cold factories select their required mode explicitly.
    Expanded selected source alone contributes to Warp's portable cache hash.
    """
    return _configure(__name__ if module is None else module, (),
                      extra_preamble=_preamble(mode, include_filter))


@wp.func_native(r"""
    wp::vec_t<3,float> result;
    if(!ipc_gpu::ccd::detail::finite(radius) || radius<=1e-10) {
        for(int k=0;k<3;++k)result[k]=static_cast<float>(NAN);
        return result;
    }
    for(int k=0;k<3;++k) {
        const double lo=double(center[k])-radius+1e-10;
        const double hi=double(center[k])+radius-1e-10;
        const double raw=double(proposed[k]);
        if(!(lo<=hi) || !ipc_gpu::ccd::detail::finite(raw)
            || !ipc_gpu::ccd::detail::finite(double(current[k]))
            || double(current[k])<double(center[k])-radius
            || double(current[k])>double(center[k])+radius) {
            result[k]=static_cast<float>(NAN);continue;
        }
        const double clipped=fmin(hi,fmax(lo,raw));
        const double delta=clipped-double(current[k]);
        float value=static_cast<float>(double(current[k])+delta);
        // Round inward to the represented float32 interval, not merely to
        // a safe double endpoint which could cross the boundary on cast.
        union FloatBits { float f; uint32_t u; } bits;
        bits.f=value;
        if(double(value)<lo) {
            if(value==0.0f)bits.u=1u;
            else if(value>0.0f)++bits.u;else --bits.u;
        }
        value=bits.f;
        if(double(value)>hi) {
            if(value==0.0f)bits.u=0x80000001u;
            else if(value>0.0f)--bits.u;else ++bits.u;
        }
        value=bits.f;
        result[k]=(double(value)>=lo && double(value)<=hi)?value:static_cast<float>(NAN);
    }
    return result;
""")
def round_inside(current: wp.vec3, proposed: wp.vec3, center: wp.vec3, radius: wp.float64) -> wp.vec3: ...


_INPUT = r"""
    wp::vec_t<3,double> result;
    result[0]=1;result[1]=0;result[2]=0;
    ipc_gpu::ccd::LinearQuery query;
    query.vertex_face=kind==0;
    int moved=0;
    double scale=0;
    for(int i=0;i<4;++i) {
        if(nodes[i]<0 || nodes[i]>=positions.shape[0] || nodes[i]>=proposed.shape[0]) {
            result[1]=2;return result;
        }
        bool changed=false;
        for(int k=0;k<3;++k) {
            query.start[i][k]=double(positions.data[nodes[i]][k]);
            query.end[i][k]=double(proposed.data[nodes[i]][k]);
            if(!ipc_gpu::ccd::detail::finite(query.start[i][k]) || !ipc_gpu::ccd::detail::finite(query.end[i][k])) {
                result[1]=2;return result;
            }
            changed=changed || query.start[i][k]!=query.end[i][k];
            scale=fmax(scale,fabs(query.start[i][k]));
        }
        moved+=changed;
    }
    query.gap_floor=fmax(1e-8,128.0*DBL_EPSILON*scale);
    if(!moved) {result[2]=1;return result;}
"""


@wp.func_native(_INPUT + r"""
    // Only a strict outward-rounded hull proof can bypass the original
    // binary64 certificate. Uncertain F32 geometry follows it unchanged.
    if(ipc_gpu::separation::safe(query.start,query.end,query.vertex_face,query.gap_floor)) {
        result[2]=1;return result;
    }
    const auto status=ipc_gpu::ccd::validate_linear(query);
    if(status==ipc_gpu::ccd::Status::Invalid)result[1]=2;
    else if(status==ipc_gpu::ccd::Status::Safe
        &&(exact_geometry||ipc_gpu::ccd::detail::separated(query,0,0,1e-10)))result[2]=1;
    else result[1]=5;
    return result;
""")
def pair_certificate(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
                     kind: int, exact_geometry: int) -> PairResult: ...


@wp.kernel(module="unique", enable_backward=False)
def _prepare(q: wp.array[wp.vec3], proposed: wp.array[wp.vec3], center: wp.array[wp.vec3],
              radius: wp.float64, radii: wp.array[wp.float64], use_radii: int,
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
                budget = radii[i]
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
def _begin_round(fraction: wp.array[wp.float64], pending_count: wp.array[int]):
    fraction[0] = wp.float64(1.0)
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
def _shorten(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], fraction: wp.array[wp.float64], error: wp.array[int]):
    i = wp.tid()
    alpha = fraction[0]
    if error[0] == 0 and alpha < wp.float64(1.0):
        start = q[i]
        end = candidate[i]
        result = wp.vec3(0.0)
        for k in range(3):
            value = wp.float64(start[k]) + alpha * (wp.float64(end[k]) - wp.float64(start[k]))
            result[k] = wp.float32(value)
        candidate[i] = result


@wp.kernel(module="unique", enable_backward=False)
def _next(condition: wp.array[int], attempts: wp.array[int], verified: wp.array[int],
           fraction: wp.array[wp.float64], maximum: int, error: wp.array[int]):
    attempts[0] += 1
    if error[0] != 0:
        condition[0] = 0
    elif fraction[0] == wp.float64(1.0):
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
        self.radius, self.exact_geometry, self.max_rounds = node_box_radius, exact_geometry, max_rounds
        self.exact_step_bound = exact_step_bound
        self.group_query_block_dim = group_query_block_dim
        self.single_vertex_filter = False
        self.error = error if error is not None else contacts.error
        self.candidate = wp.empty(self.model.particle_count, wp.vec3, device=self.device)
        self.reference = wp.clone(self.model.particle_q)
        self.condition = wp.zeros(1, int, device=self.device)
        self.attempts = wp.zeros(1, int, device=self.device)
        self.verified = wp.zeros(1, int, device=self.device)
        self.fraction = wp.ones(1, wp.float64, device=self.device)
        self.node_fractions = wp.ones(self.model.particle_count, wp.float64, device=self.device)
        self._empty_radii = wp.empty(0, wp.float64, device=self.device)
        self.pending = wp.empty(contacts.capacity, int, device=self.device)
        self.pending_count = wp.zeros(1, int, device=self.device)
        self._resolve = get_resolver("bound" if exact_step_bound else "original")

    def set_reference(self, q):
        wp.copy(self.reference, q)

    def accept(self, q, proposed, color, dt, *, center=None, radius=None, bounded=True, colors=None, preserve_gap=False,
               clip_ccd=True, query_view=None, radii=None, current_color=None, independent=False):
        if q.dtype != wp.vec3 or proposed.dtype != wp.vec3 or q.shape != proposed.shape:
            raise TypeError("CCD requires matching upstream float32 particle arrays")
        radius = self.radius if radius is None else radius
        if not math.isfinite(radius) or radius <= 1.0e-10:
            raise ValueError("Invalid CCD movement radius")
        reference = self.reference if center is None else center
        if radii is not None and (radii.dtype != wp.float64 or radii.shape != q.shape or radii.device != self.device):
            raise TypeError("CCD per-node radii must match the model's device float64 layout")
        radius_array = self._empty_radii if radii is None else radii
        if current_color is not None and (query_view is None or current_color is not query_view.current_group):
            raise ValueError("Device-selected CCD colors require their matching grouped query view")
        if independent and query_view is None:
            raise ValueError("Independent GS acceptance requires a complete grouped candidate view")
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
                [q, proposed, reference, float(radius), radius_array, int(radii is not None),
                 int(bounded), selected_colors, self.model.particle_flags,
                 self.model.particle_mass, int(color), view_groups, current_group, 1, int(current_color is not None), view_epoch, c.query_epoch,
                 self.candidate, self.error], device=self.device)
        else:
            wp.launch(_prepare, q.shape[0],
                [q, proposed, reference, float(radius), radius_array, int(radii is not None),
                 int(bounded), selected_colors, self.model.particle_flags,
                 self.model.particle_mass, int(color), self.candidate, self.error], device=self.device)
        if not clip_ccd:
            self.verified.fill_(1)
            wp.launch(_commit, q.shape[0], [self.candidate, q, self.verified, self.error], device=self.device)
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
                wp.launch(resolve, c.workers,
                    [q, self.candidate, ids, self.model.tri_indices, self.model.edge_indices,
                     c.info, int(self.exact_geometry), int(preserve_gap), c.words, c.slots, c.workers, self.pending,
                     self.pending_count, *reductions, self.error], device=self.device, block_dim=1)
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
            names = {201: "initial intersection", 202: "invalid represented motion", 203: "exact CCD capacity exhausted",
                     204: "CCD rounded-endpoint reproof exhausted", 205: "contact-dependent GS batch"}
            raise RuntimeError("IPC CCD: " + names.get(code, str(code)))
        self.contacts.check()


# Cold native snippets retain the original arithmetic and retry policies.
_CLIP = r"""
#if defined(__CUDA_ARCH__)
    ipc_gpu::exact::DevicePool pool;
    ipc_gpu::exact::initialize_pool(pool,
        words.data+size_t(worker)*2048*512,slots.data+size_t(worker)*512,2048,512);
#endif
    const auto clean=[&]() {
#if defined(__CUDA_ARCH__)
        return !pool.failed && pool.available==pool.slot_count;
#else
        return true;
#endif
    };
    double fraction=1.0;
    int failure=4;
    if(!exact_geometry && moved==1) {
        ipc_gpu::original_ccd::Query original;
        original.vertex_face=query.vertex_face;
        for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
            original.positions[i][k]=query.start[i][k];
            original.motion[i][k]=query.end[i][k]-query.start[i][k];
        }
        auto hit=ipc_gpu::original_ccd::evaluate_fast(original);
        if(hit.status==ipc_gpu::original_ccd::Status::NeedsExact) {
#if defined(__CUDA_ARCH__)
            hit=ipc_gpu::original_ccd::evaluate_exact<2048>(original);
#else
            hit=ipc_gpu::original_ccd::evaluate_exact<256>(original);
            if(hit.status==ipc_gpu::original_ccd::Status::CapacityExceeded)
                hit=ipc_gpu::original_ccd::evaluate_exact<2048>(original);
#endif
        }
        if(!clean() || hit.status==ipc_gpu::original_ccd::Status::CapacityExceeded)failure=3;
        else if(hit.status!=ipc_gpu::original_ccd::Status::Ok)failure=2;
        else if(hit.collision)fraction=0.9*hit.toi;
    }
    if(failure==4)for(int attempt=0;attempt<=64;++attempt) {
        double trial[4][3];
        for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
            const double value=attempt==64?query.start[i][k]
                :fraction==1.0?query.end[i][k]
                :query.start[i][k]+fraction*(query.end[i][k]-query.start[i][k]);
            trial[i][k]=double(static_cast<float>(value));
        }
        ipc_gpu::exact::Status verdict=ipc_gpu::exact::Status::Invalid;
        if(exact_geometry && moved<=1) {
#if defined(__CUDA_ARCH__)
            verdict=ipc_gpu::exact::validate_step<2048>(query.start,trial,query.vertex_face,query.gap_floor);
#else
            verdict=ipc_gpu::exact::validate_step<256>(query.start,trial,query.vertex_face,query.gap_floor);
            if(verdict==ipc_gpu::exact::Status::CapacityExceeded)
                verdict=ipc_gpu::exact::validate_step<2048>(query.start,trial,query.vertex_face,query.gap_floor);
#endif
        } else {
            using R=ipc_gpu::original_rigid::RepresentationStatus;
#if defined(__CUDA_ARCH__)
            auto path=ipc_gpu::original_rigid::represented_path<2048>(query.start,trial,query.vertex_face);
#else
            auto path=ipc_gpu::original_rigid::represented_path<256>(query.start,trial,query.vertex_face);
            if(path==R::CapacityExceeded)
                path=ipc_gpu::original_rigid::represented_path<2048>(query.start,trial,query.vertex_face);
#endif
            verdict=path==R::Safe?ipc_gpu::exact::Status::Safe
                :path==R::InitialContact?ipc_gpu::exact::Status::InitialContact
                :path==R::CapacityExceeded?ipc_gpu::exact::Status::CapacityExceeded
                :path==R::Invalid?ipc_gpu::exact::Status::Invalid:ipc_gpu::exact::Status::Unsafe;
            if((exact_geometry || preserve_gap) && verdict==ipc_gpu::exact::Status::Safe) {
#if defined(__CUDA_ARCH__)
                verdict=ipc_gpu::exact::validate_endpoint_gap<2048>(query.start,trial,query.vertex_face,query.gap_floor);
#else
                verdict=ipc_gpu::exact::validate_endpoint_gap<256>(query.start,trial,query.vertex_face,query.gap_floor);
                if(verdict==ipc_gpu::exact::Status::CapacityExceeded)
                    verdict=ipc_gpu::exact::validate_endpoint_gap<2048>(query.start,trial,query.vertex_face,query.gap_floor);
#endif
            }
        }
        if(!clean() || verdict==ipc_gpu::exact::Status::CapacityExceeded) {failure=3;break;}
        if(verdict==ipc_gpu::exact::Status::InitialContact) {failure=1;break;}
        if(verdict==ipc_gpu::exact::Status::Invalid) {failure=2;break;}
        if(verdict==ipc_gpu::exact::Status::Safe) {
            failure=0;result[0]=attempt==64?0.0:fraction;result[2]=fraction==1.0?1.0:0.0;break;
        }
        fraction*=0.5;
    }
    result[1]=failure;
#if defined(__CUDA_ARCH__)
    if(!clean())result[1]=3;
    ipc_gpu::exact::release_pool();
#endif
    return result;
"""

_BOUND = r"""
        if(attempt==0 && exact_geometry && moved==1) {
            const auto bound=ipc_gpu::direct_bound::separation_step_bound<2048>(
                query.start,query.end,query.vertex_face,query.gap_floor);
            if(!clean() || bound.status==ipc_gpu::direct_bound::BoundStatus::InvalidResourceContext) {
                failure=3;break;
            }
            if(bound.status==ipc_gpu::direct_bound::BoundStatus::Computed
                &&bound.fraction>0.0&&bound.fraction<1.0
                &&ipc_gpu::ccd::detail::finite(bound.fraction)) {
                // A double nextafter can disappear when positions are stored
                // as float32. Move the proposal fraction inward by one float32
                // value, then independently certify the actual stored points.
                union FractionBits { float value; uint32_t bits; } hint;
                hint.value=static_cast<float>(bound.fraction);
                if(hint.value>0.0f)--hint.bits;
                const double alpha=double(hint.value);
                double direct_trial[4][3];
                bool changed=false,different=false;
                for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
                    direct_trial[i][k]=double(static_cast<float>(
                        query.start[i][k]+alpha*(query.end[i][k]-query.start[i][k])));
                    changed=changed||direct_trial[i][k]!=query.start[i][k];
                    different=different||direct_trial[i][k]!=query.end[i][k];
                }
                if(alpha>0.0&&alpha<1.0&&changed&&different) {
#if defined(__CUDA_ARCH__)
                    auto checked=ipc_gpu::exact::validate_step<2048>(
                        query.start,direct_trial,query.vertex_face,query.gap_floor);
#else
                    auto checked=ipc_gpu::exact::validate_step<256>(
                        query.start,direct_trial,query.vertex_face,query.gap_floor);
                    if(checked==ipc_gpu::exact::Status::CapacityExceeded)
                        checked=ipc_gpu::exact::validate_step<2048>(
                            query.start,direct_trial,query.vertex_face,query.gap_floor);
#endif
                    if(!clean()||checked==ipc_gpu::exact::Status::CapacityExceeded) {failure=3;break;}
                    if(checked==ipc_gpu::exact::Status::InitialContact) {failure=1;break;}
                    if(checked==ipc_gpu::exact::Status::Invalid) {failure=2;break;}
                    if(checked==ipc_gpu::exact::Status::Safe) {
                        result[0]=alpha;result[2]=0.0;failure=0;break;
                    }
                }
            }
        }
"""

@cache
def get_pair_clip(mode="original"):
    """Return a stable native clip function for one unchanged arithmetic mode."""
    if mode not in ("original", "cached", "bound"):
        raise ValueError("Unknown CCD clipping mode")
    source = _CLIP
    if mode == "cached":
        source = source.replace("ipc_gpu::original_rigid::represented_path",
                                "ipc_gpu::gs_ccd_cache::represented_path")
    elif mode == "bound":
        backtrack = "        fraction*=0.5;"
        if source.count(backtrack) != 1:
            raise RuntimeError("The exact CCD fallback insertion point changed")
        source = source.replace(backtrack, _BOUND + backtrack)

    # Warp infers native return values from the syntactic decorator in the
    # stub's AST. Dynamically decorating a bare stub would generate void.
    # Distinct stub names also keep mode registration and identities separate.
    if mode == "original":
        @wp.func_native(_INPUT + source)
        def original(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
                     kind: int, exact_geometry: int, preserve_gap: int, words: wp.array[wp.uint32],
                     slots: wp.array[int], worker: int) -> PairResult: ...
        return original
    if mode == "cached":
        @wp.func_native(_INPUT + source)
        def cached(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
                   kind: int, exact_geometry: int, preserve_gap: int, words: wp.array[wp.uint32],
                   slots: wp.array[int], worker: int) -> PairResult: ...
        return cached
    @wp.func_native(_INPUT + source)
    def bound(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
              kind: int, exact_geometry: int, preserve_gap: int, words: wp.array[wp.uint32],
              slots: wp.array[int], worker: int) -> PairResult: ...
    return bound


@cache
def get_resolver(mode="original"):
    """One cached, isolated cold kernel per original/cached/direct mode."""
    pair_clip = get_pair_clip(mode)

    @wp.kernel(module="unique", enable_backward=False)
    def resolve(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], ids: wp.array[int],
                triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                exact_geometry: int, preserve_gap: int, words: wp.array[wp.uint32], slots: wp.array[int],
                workers: int, pending: wp.array[int], pending_count: wp.array[int],
                fraction: wp.array[wp.float64], error: wp.array[int]):
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
                node_fractions: wp.array[wp.float64], fraction: wp.array[wp.float64], error: wp.array[int]):
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
                node_fractions: wp.array[wp.float64], error: wp.array[int]):
        i = wp.tid()
        alpha = node_fractions[i]
        if error[0] == 0 and alpha < wp.float64(1.0):
            start = q[i]
            end = candidate[i]
            result = wp.vec3()
            for k in range(3):
                result[k] = wp.float32(wp.float64(start[k]) + alpha * (wp.float64(end[k]) - wp.float64(start[k])))
            candidate[i] = result

    configure_module(shorten, mode="fast")
    return shorten


@wp.func_native(r"""
    if(kind!=0&&kind!=1)return false;
    double start[4][3],end[4][3],scale=0.0;
    for(int i=0;i<4;++i) {
        if(nodes[i]<0||nodes[i]>=positions.shape[0]||nodes[i]>=proposed.shape[0])return false;
        for(int k=0;k<3;++k) {
            start[i][k]=double(positions.data[nodes[i]][k]);
            end[i][k]=double(proposed.data[nodes[i]][k]);
            if(!ipc_gpu::ccd::detail::finite(start[i][k])||!ipc_gpu::ccd::detail::finite(end[i][k]))return false;
            scale=fmax(scale,fabs(start[i][k]));
        }
    }
    const double gap=fmax(1e-8,128.0*DBL_EPSILON*scale);
    return ipc_gpu::gs_ccd_filter::safe(start,end,kind==0,gap);
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
for _kernel in (_prepare, _begin, _begin_round, _certify, _shorten, _next, _commit):
    configure_module(_kernel, mode="fast")
