"""Cold IPC contact kernels, compiled separately from BVH and fast assembly.

Only uncertain robust contacts enter this fixed worker queue. The kernel uses
one thread per block and a disjoint exact arena per worker; keeping it in its
own Warp module avoids compiling its large integer code for fast block sizes.
"""

import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from .contact import Parameters, Result, _EXACT, _INPUT, _PACK, configure_module, decode_candidate, scatter_self


@wp.func_native(_INPUT + _EXACT + _PACK)
def evaluate_self_exact(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3], nodes: wp.vec4i,
                        kind: int, params: Parameters, words: wp.array[wp.uint32], slots: wp.array[int],
                        worker: int) -> Result: ...


@wp.kernel
def resolve(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3],
             triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
             ids: wp.array[int], colors: wp.array[int], flags: wp.array[int], mass: wp.array[float],
             color: int, params: Parameters, words: wp.array[wp.uint32], slots: wp.array[int],
             workers: int, pending: wp.array[int], pending_count: wp.array[int],
             forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33], error: wp.array[int]):
    worker = wp.tid()
    if error[0] == 0:
        for index in range(worker, pending_count[0], workers):
            nodes, kind = decode_candidate(ids[pending[index]], triangles, edges, info)
            result = evaluate_self_exact(positions, previous, nodes, kind, params, words, slots, worker)
            status = int(result[48])
            if status != 0:
                wp.atomic_max(error, 0, 100 + status)
            elif result[49] != 0.0:
                scatter_self(result, nodes, colors, flags, mass, color, forces, hessians)


configure_module(__name__, include_filter=False)
