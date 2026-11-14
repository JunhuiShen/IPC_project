"""Optional exact separating-plane proposal, before unchanged CCD fallback.

This module is loaded only when ccd_exact_step_bound is enabled. The proposal
supports one moving contact corner. Coupled colors retain the general CCD
path, and every accepted color still recertifies its actual rounded endpoint.
"""

import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from .ccd import PairResult, _INPUT, configure_module
from .ccd_exact import _CLIP
from .contact import decode_candidate
from .warp_build import configure_module as _configure


HEADERS = ("direct_bound.cuh",)


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

# Preserve the complete base resolver, including its original fallback range,
# retry budget, exact policy and error mapping. Insert only this opt-in hint.
_BACKTRACK = "        fraction*=0.5;"
if _CLIP.count(_BACKTRACK) != 1:
    raise RuntimeError("The exact CCD fallback insertion point changed")


@wp.func_native(_INPUT + _CLIP.replace(_BACKTRACK, _BOUND + _BACKTRACK))
def pair_clip_bound(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
                    kind: int, exact_geometry: int, preserve_gap: int, words: wp.array[wp.uint32], slots: wp.array[int],
                    worker: int) -> PairResult: ...


@wp.kernel
def resolve(q: wp.array[wp.vec3], candidate: wp.array[wp.vec3], ids: wp.array[int],
              triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
              exact_geometry: int, preserve_gap: int, words: wp.array[wp.uint32], slots: wp.array[int], workers: int,
              pending: wp.array[int], pending_count: wp.array[int], fraction: wp.array[wp.float64], error: wp.array[int]):
    worker = wp.tid()
    if error[0] == 0:
        for index in range(worker, pending_count[0], workers):
            nodes, kind = decode_candidate(ids[pending[index]], triangles, edges, info)
            result = pair_clip_bound(q, candidate, nodes, kind, exact_geometry, preserve_gap, words, slots, worker)
            if int(result[1]) != 0:
                wp.atomic_max(error, 0, 200 + int(result[1]))
            else:
                wp.atomic_min(fraction, 0, result[0])


configure_module(__name__, include_filter=False)
_configure(__name__, HEADERS)
