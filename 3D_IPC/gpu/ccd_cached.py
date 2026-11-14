"""GS-only optional execution cache; original clip arithmetic is unchanged."""

import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from .ccd import PairResult, _INPUT, HEADERS as _BASE_HEADERS
from .warp_build import configure_module as _configure
from .contact import decode_candidate


HEADERS = (*_BASE_HEADERS, "ccd_cache.cuh")


def configure_module(module=None):
    return _configure(__name__ if module is None else module, HEADERS)


def install_our_instance(solver):
    """Explicit external-probe opt-in, before building/capturing its FrameGraph."""
    from .solver import SolverIPC
    from .ccd_exact import resolve as original_resolve
    if type(solver) is not SolverIPC:
        raise TypeError("The optional cold cache applies only to an actual SolverIPC instance")
    backend = solver.ccd_backend
    if backend.exact_step_bound or backend._resolve is not original_resolve:
        raise ValueError("The optional cold cache requires the original linear clip resolver")
    if wp.get_device(backend.device).is_capturing:
        raise RuntimeError("Install the optional cold cache before graph capture")
    backend._resolve = resolve
    return original_resolve


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
            auto path=ipc_gpu::gs_ccd_cache::represented_path<2048>(query.start,trial,query.vertex_face);
#else
            auto path=ipc_gpu::gs_ccd_cache::represented_path<256>(query.start,trial,query.vertex_face);
            if(path==R::CapacityExceeded)
                path=ipc_gpu::gs_ccd_cache::represented_path<2048>(query.start,trial,query.vertex_face);
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


@wp.func_native(_INPUT + _CLIP)
def pair_clip(positions: wp.array[wp.vec3], proposed: wp.array[wp.vec3], nodes: wp.vec4i,
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
            result = pair_clip(q, candidate, nodes, kind, exact_geometry, preserve_gap, words, slots, worker)
            if int(result[1]) != 0:
                wp.atomic_max(error, 0, 200 + int(result[1]))
            else:
                wp.atomic_min(fraction, 0, result[0])


configure_module()
