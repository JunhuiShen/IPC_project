"""Our linear CCD acceptance inside the inherited VBD color loop.

Upstream structural colors can contain several vertices of a new contact. A
single-vertex CCD test is therefore not sufficient. This backend certifies
the complete represented float32 trajectory of every candidate and uses one
common fraction for the current color. Any rounded shorter proposal is tested
again. CUDA graph conditional loops keep this reproof on the GPU and terminate
as soon as every complete proposal is certified.
"""

from __future__ import annotations

import math

import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from .contact import decode_candidate
from .warp_build import configure_module as _configure


HEADERS = ("ccd.cuh", "exact.cuh", "original_ccd.cuh", "original_rigid.cuh", "global_guess.cuh")
PairResult = wp.vec3d  # fraction, status, full represented path


def configure_module(module=None, *, include_filter=True):
    headers = (*HEADERS, "separation.cuh") if include_filter else HEADERS
    return _configure(__name__ if module is None else module, headers)


@wp.func_native(r"""
    wp::vec_t<3,float> result;
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


@wp.kernel
def _prepare(q: wp.array[wp.vec3], proposed: wp.array[wp.vec3], center: wp.array[wp.vec3],
              radius: wp.float64, bounded: int, colors: wp.array[int], flags: wp.array[int], mass: wp.array[float],
              color: int, candidate: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    if error[0] == 0:
        point = q[i]
        if flags[i] & 1 != 0 and mass[i] > 0.0 and (color < 0 or colors[i] == color):
            point = proposed[i]
        changed = point[0] != q[i][0] or point[1] != q[i][1] or point[2] != q[i][2]
        if bounded != 0 and changed:
            point = round_inside(q[i], point, center[i], radius)
        for k in range(3):
            if not wp.isfinite(q[i][k]) or not wp.isfinite(point[k]):
                wp.atomic_max(error, 0, 202)
        candidate[i] = point





@wp.kernel
def _begin(condition: wp.array[int], attempts: wp.array[int], verified: wp.array[int], error: wp.array[int]):
    attempts[0] = 0
    verified[0] = 0
    condition[0] = int(error[0] == 0)


@wp.kernel
def _begin_round(fraction: wp.array[wp.float64], pending_count: wp.array[int]):
    fraction[0] = wp.float64(1.0)
    pending_count[0] = 0


@wp.kernel
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





@wp.kernel
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


@wp.kernel
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


@wp.kernel
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
        self.error = error if error is not None else contacts.error
        self.candidate = wp.empty(self.model.particle_count, wp.vec3, device=self.device)
        self.reference = wp.clone(self.model.particle_q)
        self.condition = wp.zeros(1, int, device=self.device)
        self.attempts = wp.zeros(1, int, device=self.device)
        self.verified = wp.zeros(1, int, device=self.device)
        self.fraction = wp.ones(1, wp.float64, device=self.device)
        self.pending = wp.empty(contacts.capacity, int, device=self.device)
        self.pending_count = wp.zeros(1, int, device=self.device)
        if exact_step_bound:
            from .ccd_bound import resolve
        else:
            from .ccd_exact import resolve
        self._resolve = resolve

    def set_reference(self, q):
        wp.copy(self.reference, q)

    def accept(self, q, proposed, color, dt, *, center=None, radius=None, bounded=True, colors=None, preserve_gap=False,
               clip_ccd=True, query_view=None):
        if q.dtype != wp.vec3 or proposed.dtype != wp.vec3 or q.shape != proposed.shape:
            raise TypeError("CCD requires matching upstream float32 particle arrays")
        radius = self.radius if radius is None else radius
        if not math.isfinite(radius) or radius <= 1.0e-10:
            raise ValueError("Invalid CCD movement radius")
        reference = self.reference if center is None else center
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
                [q, proposed, reference, float(radius), int(bounded), selected_colors, self.model.particle_flags,
                 self.model.particle_mass, int(color), view_groups, current_group, 1, view_epoch, c.query_epoch,
                 self.candidate, self.error], device=self.device)
        else:
            wp.launch(_prepare, q.shape[0],
                [q, proposed, reference, float(radius), int(bounded), selected_colors, self.model.particle_flags,
                 self.model.particle_mass, int(color), self.candidate, self.error], device=self.device)
        if not clip_ccd:
            self.verified.fill_(1)
            wp.launch(_commit, q.shape[0], [self.candidate, q, self.verified, self.error], device=self.device)
            return q
        wp.launch(_begin, 1, [self.condition, self.attempts, self.verified, self.error], device=self.device)

        def round_body():
            wp.launch(_begin_round, 1, [self.fraction, self.pending_count], device=self.device)
            if c.capacity:
                if query_view is not None:
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
                wp.launch(self._resolve, c.workers,
                    [q, self.candidate, ids, self.model.tri_indices, self.model.edge_indices,
                     c.info, int(self.exact_geometry), int(preserve_gap), c.words, c.slots, c.workers, self.pending,
                     self.pending_count, self.fraction, self.error], device=self.device, block_dim=1)
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

    def prepare_initial_guess(self, detector, params=None, *, use_group_queries=False):
        from .policies import InitialGuess
        self._initial_guess = InitialGuess(self, detector, params=params, use_group_queries=use_group_queries)

    def check(self):
        code = int(self.error.numpy()[0])
        if 200 <= code < 300:
            names = {201: "initial intersection", 202: "invalid represented motion", 203: "exact CCD capacity exhausted",
                     204: "CCD rounded-endpoint reproof exhausted"}
            raise RuntimeError("IPC CCD: " + names.get(code, str(code)))
        self.contacts.check()


def __getattr__(name):
    if name == "pair_clip":
        from .ccd_exact import pair_clip
        return pair_clip
    raise AttributeError(name)


configure_module()
