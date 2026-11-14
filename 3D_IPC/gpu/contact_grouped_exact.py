"""Cold exact contact replay for a device-selected dynamic GS color.

Keep this kernel separate from ordinary BVH/assembly block-size variants so
the exact arithmetic is compiled only for its one-thread worker blocks.
"""

import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from .contact import Parameters, configure_module, decode_candidate, scatter_self
from .contact_exact import evaluate_self_exact


@wp.kernel
def resolve(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3],
             triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
             ids: wp.array[int], colors: wp.array[int], flags: wp.array[int], mass: wp.array[float],
             current_color: wp.array[int], params: Parameters, words: wp.array[wp.uint32], slots: wp.array[int],
             workers: int, pending: wp.array[int], pending_count: wp.array[int],
             forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33], error: wp.array[int]):
    worker = wp.tid()
    if error[0] == 0:
        color = current_color[0]
        for index in range(worker, pending_count[0], workers):
            nodes, kind = decode_candidate(ids[pending[index]], triangles, edges, info)
            result = evaluate_self_exact(positions, previous, nodes, kind, params, words, slots, worker)
            status = int(result[48])
            if status != 0:
                wp.atomic_max(error, 0, 100 + status)
            elif result[49] != 0.0:
                scatter_self(result, nodes, colors, flags, mass, color, forces, hessians)


configure_module(__name__, include_filter=False)
