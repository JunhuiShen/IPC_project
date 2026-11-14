"""Our IPC contact law inside Newton's float32 GPU collision architecture.

The candidate stream comes directly from ``TriMeshCollisionInfo``. Detection
rows are compacted on the device; ordinary contact evaluation is fused with
scatter into the VBD force/Hessian buffers. Only exact requests use a second
queue. Geometry is evaluated from the exact represented float32 coordinates
using our maintained binary64/exact arithmetic, then physical local systems
are accumulated in the upstream float32 storage.

No queue count or numerical array is downloaded during a color update. All
workspaces are persistent and CUDA graph capture keeps the same launch shape.
``check`` belongs outside a captured frame; an error is latched until an
explicit outside-frame ``reset_status``.
"""

from __future__ import annotations

import math

import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from .warp_build import configure_module as _configure
from .movement_box_queries import build_nodes as _build_box_nodes


HEADERS = ("contact.cuh", "contact_exact.cuh", "original_contact.cuh", "exact.cuh")
Result = wp.types.vector(length=50, dtype=wp.float32)
FullPacket = wp.types.vector(length=178, dtype=wp.float64)


def configure_module(module=None, *, include_filter=True):
    headers = (*HEADERS, "separation.cuh") if include_filter else HEADERS
    return _configure(__name__ if module is None else module, headers)


@wp.struct
class Parameters:
    d_hat: wp.float64
    k_barrier: wp.float64
    dt: wp.float64
    friction: wp.float64
    eps_v: wp.float64
    eps: wp.float64
    exact_geometry: int


def parameters(*, d_hat, k_barrier, dt, friction=0.0, eps_v=0.001, exact_geometry=True):
    for name, value in (("d_hat", d_hat), ("k_barrier", k_barrier), ("friction", friction)):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name, value in (("dt", dt), ("eps_v", eps_v)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(exact_geometry, bool):
        raise TypeError("exact_geometry must be a boolean")
    result = Parameters()
    result.d_hat, result.k_barrier, result.dt = d_hat, k_barrier, dt
    result.friction, result.eps_v, result.eps = friction, eps_v, 1.0e-12
    result.exact_geometry = int(exact_geometry)
    return result


@wp.func
def decode_candidate(encoded: int, triangles: wp.array2d[int], edges: wp.array2d[int],
                     info: TriMeshCollisionInfo):
    nodes = wp.vec4i(0)
    kind = int(0)
    if encoded >= 0:
        vertex = info.vertex_colliding_triangles[2 * encoded]
        triangle = info.vertex_colliding_triangles[2 * encoded + 1]
        nodes = wp.vec4i(vertex, triangles[triangle, 0], triangles[triangle, 1], triangles[triangle, 2])
    else:
        slot = -encoded - 1
        first = info.edge_colliding_edges[2 * slot]
        second = info.edge_colliding_edges[2 * slot + 1]
        nodes = wp.vec4i(edges[first, 2], edges[first, 3], edges[second, 2], edges[second, 3])
        kind = 1
    return nodes, kind


@wp.func
def _same_body(nodes: wp.vec4i, owners: wp.array[int]):
    owner = owners[nodes[0]]
    return owner >= 0 and owner == owners[nodes[1]] and owner == owners[nodes[2]] and owner == owners[nodes[3]]


@wp.func
def _worlds_match(first: int, second: int):
    return first < 0 or second < 0 or first == second


@wp.func
def _query_padding(position: wp.vec3, radius: float):
    scale = wp.max(wp.abs(position), wp.vec3(wp.abs(radius)))
    return scale * 0.00000095367431640625 + wp.vec3(1.401298464324817e-45)


@wp.kernel
def _rigid_coordinate_margin(positions: wp.array[wp.vec3], margin: wp.array[float], error: wp.array[int]):
    point = positions[wp.tid()]
    if wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2]):
        scale = wp.max(wp.abs(point[0]), wp.max(wp.abs(point[1]), wp.abs(point[2])))
        wp.atomic_max(margin, 0, scale * 0.00000762939453125)
    else:
        wp.atomic_max(error, 0, 202)


@wp.kernel
def _detect_vertices(bvh: wp.uint64, positions: wp.array[wp.vec3], triangles: wp.array2d[int],
                      worlds: wp.array[int], owners: wp.array[int], surface: wp.array[int], radius: float,
                      rigid_margin: wp.array[float], info: TriMeshCollisionInfo, error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0 and surface[vertex] != 0:
        p = positions[vertex]
        pad = _query_padding(p, radius) + wp.vec3(rigid_margin[0])
        query = wp.bvh_query_aabb(bvh, p - wp.vec3(radius) - pad, p + wp.vec3(radius) + pad)
        found = int(0)
        tri = int(0)
        offset = info.vertex_colliding_triangles_offsets[vertex]
        capacity = info.vertex_colliding_triangles_buffer_sizes[vertex]
        while wp.bvh_query_next(query, tri):
            a = triangles[tri, 0]
            b = triangles[tri, 1]
            c = triangles[tri, 2]
            nodes = wp.vec4i(vertex, a, b, c)
            if vertex != a and vertex != b and vertex != c and _worlds_match(worlds[vertex], worlds[a]) and not _same_body(nodes, owners):
                if found < capacity:
                    info.vertex_colliding_triangles[2 * (offset + found)] = vertex
                    info.vertex_colliding_triangles[2 * (offset + found) + 1] = tri
                found += 1
        info.vertex_colliding_triangles_count[vertex] = found
        info.vertex_colliding_triangles_min_dist[vertex] = radius
        if found > capacity:
            wp.atomic_max(error, 0, 10)
    elif surface[vertex] == 0:
        info.vertex_colliding_triangles_count[vertex] = 0


@wp.kernel
def _detect_edges(bvh: wp.uint64, positions: wp.array[wp.vec3], edges: wp.array2d[int],
                  worlds: wp.array[int], owners: wp.array[int], radius: float,
                  rigid_margin: wp.array[float], info: TriMeshCollisionInfo, error: wp.array[int]):
    edge = wp.tid()
    if error[0] == 0:
        a = edges[edge, 2]
        b = edges[edge, 3]
        lo = wp.min(positions[a], positions[b])
        hi = wp.max(positions[a], positions[b])
        pad = wp.max(_query_padding(lo, radius), _query_padding(hi, radius)) + wp.vec3(rigid_margin[0])
        query = wp.bvh_query_aabb(bvh, lo - wp.vec3(radius) - pad, hi + wp.vec3(radius) + pad)
        found = int(0)
        other = int(0)
        offset = info.edge_colliding_edges_offsets[edge]
        capacity = info.edge_colliding_edges_buffer_sizes[edge]
        while wp.bvh_query_next(query, other):
            c = edges[other, 2]
            d = edges[other, 3]
            nodes = wp.vec4i(a, b, c, d)
            if a != c and a != d and b != c and b != d and _worlds_match(worlds[a], worlds[c]) and not _same_body(nodes, owners):
                if found < capacity:
                    info.edge_colliding_edges[2 * (offset + found)] = edge
                    info.edge_colliding_edges[2 * (offset + found) + 1] = other
                found += 1
        info.edge_colliding_edges_count[edge] = found
        info.edge_colliding_edges_min_dist[edge] = radius
        if found > capacity:
            wp.atomic_max(error, 0, 10)


@wp.kernel
def _swept_nodes(start: wp.array[wp.vec3], end: wp.array[wp.vec3], reach: float,
                  lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    lo = wp.min(start[i], end[i])
    hi = wp.max(start[i], end[i])
    pad = wp.max(_query_padding(lo, reach), _query_padding(hi, reach))
    lower[i] = lo - wp.vec3(reach) - pad
    upper[i] = hi + wp.vec3(reach) + pad
    for k in range(3):
        if not wp.isfinite(lower[i][k]) or not wp.isfinite(upper[i][k]):
            wp.atomic_max(error, 0, 202)


@wp.kernel
def _swept_triangles(triangles: wp.array2d[int], nlo: wp.array[wp.vec3], nhi: wp.array[wp.vec3],
                      lower: wp.array[wp.vec3], upper: wp.array[wp.vec3]):
    i = wp.tid()
    a = triangles[i, 0]
    b = triangles[i, 1]
    c = triangles[i, 2]
    lower[i] = wp.min(nlo[a], wp.min(nlo[b], nlo[c]))
    upper[i] = wp.max(nhi[a], wp.max(nhi[b], nhi[c]))


@wp.kernel
def _swept_edges(edges: wp.array2d[int], nlo: wp.array[wp.vec3], nhi: wp.array[wp.vec3],
                  lower: wp.array[wp.vec3], upper: wp.array[wp.vec3]):
    i = wp.tid()
    a = edges[i, 2]
    b = edges[i, 3]
    lower[i] = wp.min(nlo[a], nlo[b])
    upper[i] = wp.max(nhi[a], nhi[b])


@wp.kernel
def _detect_swept_vertices(bvh: wp.uint64, lower: wp.array[wp.vec3], upper: wp.array[wp.vec3],
                           triangles: wp.array2d[int], worlds: wp.array[int], owners: wp.array[int], surface: wp.array[int],
                           info: TriMeshCollisionInfo, error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0 and surface[vertex] != 0:
        query = wp.bvh_query_aabb(bvh, lower[vertex], upper[vertex])
        found = int(0)
        tri = int(0)
        offset = info.vertex_colliding_triangles_offsets[vertex]
        capacity = info.vertex_colliding_triangles_buffer_sizes[vertex]
        while wp.bvh_query_next(query, tri):
            a = triangles[tri, 0]
            b = triangles[tri, 1]
            c = triangles[tri, 2]
            nodes = wp.vec4i(vertex, a, b, c)
            if vertex != a and vertex != b and vertex != c and _worlds_match(worlds[vertex], worlds[a]) and not _same_body(nodes, owners):
                if found < capacity:
                    info.vertex_colliding_triangles[2 * (offset + found)] = vertex
                    info.vertex_colliding_triangles[2 * (offset + found) + 1] = tri
                found += 1
        info.vertex_colliding_triangles_count[vertex] = found
        info.vertex_colliding_triangles_min_dist[vertex] = 0.0
        if found > capacity:
            wp.atomic_max(error, 0, 10)
    elif surface[vertex] == 0:
        info.vertex_colliding_triangles_count[vertex] = 0


@wp.kernel
def _detect_swept_edges(bvh: wp.uint64, lower: wp.array[wp.vec3], upper: wp.array[wp.vec3],
                       worlds: wp.array[int], owners: wp.array[int], edges: wp.array2d[int],
                       info: TriMeshCollisionInfo, error: wp.array[int]):
    edge = wp.tid()
    if error[0] == 0:
        a = edges[edge, 2]
        b = edges[edge, 3]
        lo = wp.min(lower[a], lower[b])
        hi = wp.max(upper[a], upper[b])
        query = wp.bvh_query_aabb(bvh, lo, hi)
        found = int(0)
        other = int(0)
        offset = info.edge_colliding_edges_offsets[edge]
        capacity = info.edge_colliding_edges_buffer_sizes[edge]
        while wp.bvh_query_next(query, other):
            c = edges[other, 2]
            d = edges[other, 3]
            nodes = wp.vec4i(a, b, c, d)
            if a != c and a != d and b != c and b != d and _worlds_match(worlds[a], worlds[c]) and not _same_body(nodes, owners):
                if found < capacity:
                    info.edge_colliding_edges[2 * (offset + found)] = edge
                    info.edge_colliding_edges[2 * (offset + found) + 1] = other
                found += 1
        info.edge_colliding_edges_count[edge] = found
        info.edge_colliding_edges_min_dist[edge] = 0.0
        if found > capacity:
            wp.atomic_max(error, 0, 10)


@wp.kernel
def _compact_rows(info: TriMeshCollisionInfo, triangles: wp.array2d[int], edges: wp.array2d[int],
                  owners: wp.array[int], ids: wp.array[int], count: wp.array[int], error: wp.array[int]):
    tid = wp.tid()
    row = tid // 4
    lane = tid % 4
    nv = info.vertex_colliding_triangles_count.shape[0]
    ne = info.edge_colliding_edges_count.shape[0]
    if error[0] == 0:
        if row < nv:
            found = info.vertex_colliding_triangles_count[row]
            capacity = info.vertex_colliding_triangles_buffer_sizes[row]
            if found < 0 or found > capacity:
                wp.atomic_max(error, 0, 10)
            for local in range(lane, wp.min(found, capacity), 4):
                slot = info.vertex_colliding_triangles_offsets[row] + local
                vertex = info.vertex_colliding_triangles[2 * slot]
                tri = info.vertex_colliding_triangles[2 * slot + 1]
                if vertex < 0 or vertex >= owners.shape[0] or tri < 0 or tri >= triangles.shape[0]:
                    wp.atomic_max(error, 0, 11)
                else:
                    nodes = wp.vec4i(vertex, triangles[tri, 0], triangles[tri, 1], triangles[tri, 2])
                    if vertex != nodes[1] and vertex != nodes[2] and vertex != nodes[3] and not _same_body(nodes, owners):
                        at = wp.atomic_add(count, 0, 1)
                        if at < ids.shape[0]:
                            ids[at] = slot
                        else:
                            wp.atomic_max(error, 0, 12)
        elif row - nv < ne:
            edge_row = row - nv
            found = info.edge_colliding_edges_count[edge_row]
            capacity = info.edge_colliding_edges_buffer_sizes[edge_row]
            if found < 0 or found > capacity:
                wp.atomic_max(error, 0, 10)
            for local in range(lane, wp.min(found, capacity), 4):
                slot = info.edge_colliding_edges_offsets[edge_row] + local
                first = info.edge_colliding_edges[2 * slot]
                second = info.edge_colliding_edges[2 * slot + 1]
                if first < 0 or first >= edges.shape[0] or second < 0 or second >= edges.shape[0]:
                    wp.atomic_max(error, 0, 11)
                elif first < second:
                    nodes = wp.vec4i(edges[first, 2], edges[first, 3], edges[second, 2], edges[second, 3])
                    shared = nodes[0] == nodes[2] or nodes[0] == nodes[3] or nodes[1] == nodes[2] or nodes[1] == nodes[3]
                    if not shared and not _same_body(nodes, owners):
                        at = wp.atomic_add(count, 0, 1)
                        if at < ids.shape[0]:
                            ids[at] = -slot - 1
                        else:
                            wp.atomic_max(error, 0, 12)





_INPUT = r"""
    using namespace ipc_gpu::solver;
    ContactInput in;
    for(int i=0;i<4;++i) {
        if(nodes[i]<0 || nodes[i]>=positions.shape[0] || nodes[i]>=previous.shape[0]) {
            wp::vec_t<50,float> invalid; for(int k=0;k<50;++k)invalid[k]=0;
            invalid[48]=1;return invalid;
        }
        for(int k=0;k<3;++k) {
            in.positions[i][k]=double(positions.data[nodes[i]][k]);
            in.previous[i][k]=double(previous.data[nodes[i]][k]);
        }
    }
    in.kind=kind==0?ContactKind::PointTriangle:ContactKind::EdgeEdge;
    in.d_hat=params.d_hat;in.k_barrier=params.k_barrier;in.dt=params.dt;
    in.friction=params.friction;in.eps_v=params.eps_v;in.eps=params.eps;
    in.hessian_blocks=0x8421u;
"""

_PACK = r"""
    wp::vec_t<50,float> packed;
    for(int k=0;k<50;++k)packed[k]=0;
    packed[48]=static_cast<float>(static_cast<int>(out.status));packed[49]=static_cast<float>(out.active);
    if(out.status==ContactStatus::Ok && out.active) {
        const double inverse_dt2=1.0/(params.dt*params.dt);
        for(int role=0;role<4;++role) {
            const double weight=out.weights[role];
            for(int k=0;k<3;++k) {
                const double g=params.k_barrier*out.barrier_gradient[role][k]
                    +(weight*out.friction_relative_gradient[k])*inverse_dt2;
                packed[12*role+k]=static_cast<float>(-g);
                for(int l=0;l<3;++l) {
                    const double h=params.k_barrier*out.barrier_hessian[role][role][3*k+l]
                        +((weight*out.friction_relative_hessian[3*k+l])*weight)*inverse_dt2;
                    packed[12*role+3+3*k+l]=static_cast<float>(h);
                }
            }
        }
        for(int k=0;k<48;++k)if(!isfinite(packed[k])) {
            packed[48]=4;packed[49]=0;break;
        }
    }
    return packed;
"""


@wp.func_native(_INPUT + r"""
    ContactOutput out;
    if(params.d_hat>0 && params.k_barrier>0) {
        // An inactive-geometry proof must not bypass the original validation
        // of lagged friction state or caller-supplied scalar parameters.
        bool filterable=ipc_gpu::separation::finite(in.k_barrier)
            &&ipc_gpu::separation::finite(in.dt)&&in.dt>0
            &&ipc_gpu::separation::finite(in.friction)&&in.friction>=0
            &&ipc_gpu::separation::finite(in.eps_v)&&in.eps_v>0
            &&ipc_gpu::separation::finite(in.eps)&&in.eps>0;
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)
            filterable=filterable&&ipc_gpu::separation::finite(in.previous[i][k]);
        if(!filterable||!ipc_gpu::separation::safe(in.positions,in.positions,kind==0,params.d_hat))
            out=params.exact_geometry?evaluate_contact(in):evaluate_contact_original(in);
    }
""" + _PACK)
def evaluate_self(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3], nodes: wp.vec4i,
                  kind: int, params: Parameters) -> Result: ...


_EXACT = r"""
    ContactOutput out;
#if defined(__CUDA_ARCH__)
    ipc_gpu::exact::DevicePool pool;
    ipc_gpu::exact::initialize_pool(pool,
        words.data+size_t(worker)*2048*512,slots.data+size_t(worker)*512,2048,512);
    // CUDA integers use global arena storage and loop over their used limbs.
    // The final capacity has no extra per-operation cost; compiling a second
    // lower-capacity specialization only duplicates this large cold path.
    out=evaluate_contact_exact<2048>(in);
    if(pool.failed || pool.available!=pool.slot_count) {
        out=ContactOutput{};out.status=ContactStatus::ExactCapacityExceeded;
    }
    ipc_gpu::exact::release_pool();
#else
    // CPU integers contain fixed arrays, so retain the smaller first attempt.
    out=evaluate_contact_exact<256>(in);
    if(out.status==ContactStatus::ExactCapacityExceeded)
        out=evaluate_contact_exact<2048>(in);
#endif
"""


_FULL_INPUT = _INPUT.replace(
    "wp::vec_t<50,float> invalid; for(int k=0;k<50;++k)invalid[k]=0;",
    "wp::vec_t<178,double> invalid; for(int k=0;k<178;++k)invalid[k]=0;",
).replace("invalid[48]=1", "invalid[176]=1").replace("in.hessian_blocks=0x8421u", "in.hessian_blocks=0xffffu")
_FULL_PACK = r"""
    wp::vec_t<178,double> packed;
    for(int k=0;k<178;++k)packed[k]=0;
    packed[0]=out.distance;
    for(int i=0;i<4;++i)packed[1+i]=out.weights[i];
    for(int k=0;k<3;++k)packed[5+k]=out.normal[k];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)packed[8+3*i+k]=out.barrier_gradient[i][k];
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int k=0;k<9;++k)
        packed[20+36*i+9*j+k]=out.barrier_hessian[i][j][k];
    for(int k=0;k<3;++k)packed[164+k]=out.friction_relative_gradient[k];
    for(int k=0;k<9;++k)packed[167+k]=out.friction_relative_hessian[k];
    packed[176]=static_cast<int>(out.status);packed[177]=out.active;
    return packed;
"""


@wp.func_native(_FULL_INPUT + r"""
    ContactOutput out;
    if(params.d_hat>0 && params.k_barrier>0)
        out=params.exact_geometry?evaluate_contact(in):evaluate_contact_original(in);
""" + _FULL_PACK)
def evaluate_full(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3], nodes: wp.vec4i,
                  kind: int, params: Parameters) -> FullPacket: ...


@wp.func_native(_FULL_INPUT + _EXACT + _FULL_PACK)
def evaluate_full_exact(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3], nodes: wp.vec4i,
                        kind: int, params: Parameters, words: wp.array[wp.uint32], slots: wp.array[int],
                        worker: int) -> FullPacket: ...


@wp.func
def scatter_self(result: Result, nodes: wp.vec4i, colors: wp.array[int], flags: wp.array[int],
                 mass: wp.array[float], color: int, forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    for role in range(4):
        vertex = nodes[role]
        if colors[vertex] == color and flags[vertex] & 1 != 0 and mass[vertex] > 0.0:
            force = wp.vec3(result[12 * role], result[12 * role + 1], result[12 * role + 2])
            hessian = wp.mat33(0.0)
            for r in range(3):
                for c in range(3):
                    hessian[r, c] = result[12 * role + 3 + 3 * r + c]
            wp.atomic_add(forces, vertex, force)
            wp.atomic_add(hessians, vertex, hessian)


@wp.kernel
def _evaluate_fast(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                   triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                   ids: wp.array[int], count: wp.array[int], colors: wp.array[int], flags: wp.array[int],
                   mass: wp.array[float], color: int, params: Parameters, workers: int,
                   pending: wp.array[int], pending_count: wp.array[int],
                   forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33], error: wp.array[int]):
    worker = wp.tid()
    if error[0] == 0:
        for at in range(worker, count[0], workers):
            nodes, kind = decode_candidate(ids[at], triangles, edges, info)
            touches = False
            for role in range(4):
                v = nodes[role]
                if colors[v] == color and flags[v] & 1 != 0 and mass[v] > 0.0:
                    touches = True
            if touches:
                result = evaluate_self(positions, previous, nodes, kind, params)
                status = int(result[48])
                if status == 2 and params.exact_geometry != 0:
                    slot = wp.atomic_add(pending_count, 0, 1)
                    if slot < pending.shape[0]:
                        pending[slot] = at
                    else:
                        wp.atomic_max(error, 0, 12)
                elif status != 0:
                    wp.atomic_max(error, 0, 100 + status)
                elif result[49] != 0.0:
                    scatter_self(result, nodes, colors, flags, mass, color, forces, hessians)


@wp.kernel
def _evaluate_grouped(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                      triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                      ids: wp.array[int], count: wp.array[int], colors: wp.array[int], flags: wp.array[int],
                      mass: wp.array[float], current_color: wp.array[int], offsets: wp.array[int], use_view: int,
                      view_epoch: wp.array[wp.uint64], query_epoch: wp.array[wp.uint64],
                      params: Parameters, workers: int, pending: wp.array[int], pending_count: wp.array[int],
                      forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33], error: wp.array[int]):
    worker = wp.tid()
    color = current_color[0]
    valid = color >= 0 and color < colors.shape[0]
    begin = int(0)
    end = count[0]
    if use_view != 0 and valid:
        valid = view_epoch[0] == query_epoch[0]
        begin = offsets[color]
        end = offsets[color + 1]
    valid = valid and begin >= 0 and end >= begin and end <= ids.shape[0]
    if not valid:
        wp.atomic_max(error, 0, 11)
    if error[0] == 0 and valid:
        for at in range(begin + worker, end, workers):
            nodes, kind = decode_candidate(ids[at], triangles, edges, info)
            touches = False
            for role in range(4):
                node = nodes[role]
                if colors[node] == color and flags[node] & 1 != 0 and mass[node] > 0.0:
                    touches = True
            if touches:
                result = evaluate_self(positions, previous, nodes, kind, params)
                status = int(result[48])
                if status == 2 and params.exact_geometry != 0:
                    slot = wp.atomic_add(pending_count, 0, 1)
                    if slot < pending.shape[0]:
                        pending[slot] = at
                    else:
                        wp.atomic_max(error, 0, 12)
                elif status != 0:
                    wp.atomic_max(error, 0, 100 + status)
                elif result[49] != 0.0:
                    scatter_self(result, nodes, colors, flags, mass, color, forces, hessians)


class ContactBackend:
    """Persistent graph-safe IPC assembly consuming Newton's actual BVH rows."""

    def __init__(self, model, info, *, d_hat, k_barrier, friction=0.0, eps_v=0.001,
                 exact_geometry=True, node_body=None, workers=None, error=None, track_query_epoch=False):
        self.model, self.info, self.device = model, info, wp.get_device(model.device)
        self.options = dict(d_hat=d_hat, k_barrier=k_barrier, friction=friction,
                            eps_v=eps_v, exact_geometry=exact_geometry)
        self.error = error if error is not None else wp.zeros(1, int, device=self.device)
        nv = info.vertex_colliding_triangles_count.shape[0]
        ne = info.edge_colliding_edges_count.shape[0]
        self.capacity = info.vertex_colliding_triangles.shape[0] // 2 + info.edge_colliding_edges.shape[0] // 2
        if self.capacity < 0 or self.capacity >= 2**31:
            raise ValueError("Newton collision storage exceeds the IPC queue index capacity")
        self.row_threads = (nv + ne) * 4
        self.candidate_ids = wp.empty(self.capacity, int, device=self.device)
        self.candidate_count = wp.zeros(1, int, device=self.device)
        self.track_query_epoch = bool(track_query_epoch)
        self.query_epoch = wp.zeros(1, wp.uint64, device=self.device) if self.track_query_epoch else None
        self._compact_epoch = None
        if self.track_query_epoch:
            from .query_groups import compact_rows_epoch
            self._compact_epoch = compact_rows_epoch
        self.query_generation = 0
        self.pending = wp.empty(self.capacity, int, device=self.device)
        self.pending_count = wp.zeros(1, int, device=self.device)
        self.node_body = node_body if node_body is not None else wp.full(model.particle_count, -1, int, device=self.device)
        self.surface = getattr(getattr(model, "ipc", None), "surface", None)
        if self.surface is None:
            self.surface = wp.ones(model.particle_count, int, device=self.device)
        self.workers = workers if workers is not None else (min(16, self.device.sm_count) if self.device.is_cuda else 1)
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers <= 0:
            raise ValueError("Exact workers must be a positive integer")
        self.words = wp.empty(self.workers * 2048 * 512 if self.device.is_cuda else 0, wp.uint32, device=self.device)
        self.slots = wp.empty(self.workers * 512 if self.device.is_cuda else 0, int, device=self.device)
        self.fast_workers = max(128, self.device.sm_count * 128) if self.device.is_cuda else 128
        self.swept_lower = wp.empty(model.particle_count, wp.vec3, device=self.device)
        self.swept_upper = wp.empty(model.particle_count, wp.vec3, device=self.device)
        # Swept and movement-box detection execute at distinct checkpoints.
        # Reuse their unexpanded node bounds; retain separate expanded query
        # bounds so the BVH stores unions of the complete blue boxes.
        self.box_query_lower = wp.empty(model.particle_count, wp.vec3, device=self.device)
        self.box_query_upper = wp.empty(model.particle_count, wp.vec3, device=self.device)
        self._has_rigid_bodies = getattr(model, "body_count", 0) > 0
        self.rigid_coordinate_margin = wp.zeros(1, float, device=self.device)
        # Keep the large exact fallback out of every fast/detection module
        # block-size variant. The cold module has one block-size-one kernel.
        from .contact_exact import resolve
        self._resolve = resolve
        from .contact_grouped_exact import resolve as resolve_grouped
        self._resolve_grouped = resolve_grouped
        parameters(dt=1.0, **self.options)

    def refresh(self, info=None):
        if info is not None:
            capacity = info.vertex_colliding_triangles.shape[0] // 2 + info.edge_colliding_edges.shape[0] // 2
            if capacity != self.capacity:
                raise RuntimeError("Collision storage changed; rebuild the IPC backend before graph capture")
            self.info = info
        if self.track_query_epoch:
            self.query_generation += 1
        self.candidate_count.zero_()
        if self.row_threads:
            arguments = [self.info, self.model.tri_indices, self.model.edge_indices, self.node_body,
                         self.candidate_ids, self.candidate_count, self.error]
            if self.track_query_epoch:
                wp.launch(self._compact_epoch, self.row_threads, [*arguments, self.query_epoch], device=self.device)
            else:
                wp.launch(_compact_rows, self.row_threads, arguments, device=self.device)

    def detect(self, detector, positions, query_radius, *, refit=True):
        """Conservative query mode on the original detector's actual BVHs.

        Approximate closest-distance tests and structural-ring exclusions are
        inappropriate for our complete-trajectory CCD candidate set. The BVH
        traversal therefore records every nonincident AABB hit in the same
        upstream bounded rows. ``query_radius`` must cover both primitives'
        whole allowed checkpoint motions and the desired contact/CCD reach.
        """
        if not math.isfinite(query_radius) or query_radius <= 0:
            raise ValueError("Conservative collision reach must be finite and positive")
        if self._has_rigid_bodies:
            # Let G bound current proxy coordinates and rho bound the fixed
            # reference radii. A COM is at most G+rho from the origin. Float32
            # projection residuals are O(eps*(G+rho)); an analytic rotation
            # can turn a large Y residual into X motion. Per-axis query padding
            # cannot enclose that. Use 64*eps*G uniformly in every axis, while
            # BodySolver's retained constructor margin covers rho and initial
            # projection error. The factor encloses both primitives' residuals
            # and their curved/rounded endpoint joins. Cloth avoids this scan.
            self.rigid_coordinate_margin.zero_()
            wp.launch(_rigid_coordinate_margin, self.model.particle_count,
                [positions, self.rigid_coordinate_margin, self.error], device=self.device)
        if refit:
            # The upstream detector constructs empty BVHs too. Refitting an
            # empty Warp BVH can dereference absent CPU storage, so update only
            # the actual nonempty structures while retaining this detector.
            detector.vertex_positions = positions
            if self.model.tri_count:
                detector.refit_triangles()
            if self.model.edge_count and detector.bvh_edges is not None:
                detector.refit_edges()
        reverse = self.info.triangle_colliding_vertices_count
        if reverse is not None:
            reverse.zero_()
        if self.model.tri_count:
            wp.launch(_detect_vertices, self.model.particle_count,
                [detector.bvh_tris.id, positions, self.model.tri_indices, self.model.particle_world,
                 self.node_body, self.surface, float(query_radius), self.rigid_coordinate_margin,
                 self.info, self.error], device=self.device, block_dim=16)
        else:
            self.info.vertex_colliding_triangles_count.zero_()
        if self.model.edge_count and detector.bvh_edges is not None:
            wp.launch(_detect_edges, self.model.edge_count,
                [detector.bvh_edges.id, positions, self.model.edge_indices, self.model.particle_world,
                 self.node_body, float(query_radius), self.rigid_coordinate_margin,
                 self.info, self.error], device=self.device, block_dim=16)
        else:
            self.info.edge_colliding_edges_count.zero_()
        self.refresh()

    def detect_boxes(self, detector, centers, radii, d_hat):
        """Record every deformable pair reachable within its checkpoint boxes.

        Primitive BVHs contain unions of the outward-rounded vertex boxes.
        Queries expand only one side by ``d_hat``: blue vertex/edge boxes
        against triangle/edge unions. This is the CPU GS red/green-box
        construction with the expansion moved to the query side for NT.
        All storage and BVHs already exist before CUDA graph capture.
        """
        if self._has_rigid_bodies:
            raise ValueError("Rigid trajectories require the scalar conservative detector")
        if isinstance(d_hat, bool) or not math.isfinite(d_hat) or d_hat < 0:
            raise ValueError("Movement-box contact reach must be finite and nonnegative")
        count = self.model.particle_count
        if centers.dtype != wp.vec3 or centers.shape != (count,) or centers.device != self.device:
            raise TypeError("Movement-box centers must match the model float32 particle array")
        if radii.dtype != wp.float64 or radii.shape != (count,) or radii.device != self.device:
            raise TypeError("Movement-box radii must match the model binary64 particle array")
        reverse = self.info.triangle_colliding_vertices_count
        if reverse is not None:
            reverse.zero_()
        if count:
            wp.launch(_build_box_nodes, count,
                [centers, radii, float(d_hat), self.swept_lower, self.swept_upper,
                 self.box_query_lower, self.box_query_upper, self.error], device=self.device)
        if self.model.tri_count:
            wp.launch(_swept_triangles, self.model.tri_count,
                [self.model.tri_indices, self.swept_lower, self.swept_upper,
                 detector.lower_bounds_tris, detector.upper_bounds_tris], device=self.device)
            detector.bvh_tris.refit()
            wp.launch(_detect_swept_vertices, count,
                [detector.bvh_tris.id, self.box_query_lower, self.box_query_upper,
                 self.model.tri_indices, self.model.particle_world, self.node_body,
                 self.surface, self.info, self.error], device=self.device, block_dim=16)
        else:
            self.info.vertex_colliding_triangles_count.zero_()
        if self.model.edge_count and detector.bvh_edges is not None:
            wp.launch(_swept_edges, self.model.edge_count,
                [self.model.edge_indices, self.swept_lower, self.swept_upper,
                 detector.lower_bounds_edges, detector.upper_bounds_edges], device=self.device)
            detector.bvh_edges.refit()
            wp.launch(_detect_swept_edges, self.model.edge_count,
                [detector.bvh_edges.id, self.box_query_lower, self.box_query_upper,
                 self.model.particle_world, self.node_body, self.model.edge_indices,
                 self.info, self.error], device=self.device, block_dim=16)
        else:
            self.info.edge_colliding_edges_count.zero_()
        self.refresh()

    def detect_swept(self, detector, start, end, reach=1.0e-8):
        """Reuse the same upstream BVHs with whole initial-guess motion boxes.

        The caller must refit the detector to actual accepted positions before
        later ordinary detection; SolverIPC's post-initialization pass does so.
        """
        if not math.isfinite(reach) or reach < 0:
            raise ValueError("Swept collision reach must be finite and nonnegative")
        reverse = self.info.triangle_colliding_vertices_count
        if reverse is not None:
            reverse.zero_()
        wp.launch(_swept_nodes, self.model.particle_count,
            [start, end, float(reach), self.swept_lower, self.swept_upper, self.error], device=self.device)
        if self.model.tri_count:
            wp.launch(_swept_triangles, self.model.tri_count,
                [self.model.tri_indices, self.swept_lower, self.swept_upper,
                 detector.lower_bounds_tris, detector.upper_bounds_tris], device=self.device)
            detector.bvh_tris.refit()
            wp.launch(_detect_swept_vertices, self.model.particle_count,
                [detector.bvh_tris.id, self.swept_lower, self.swept_upper, self.model.tri_indices,
                 self.model.particle_world, self.node_body, self.surface, self.info, self.error], device=self.device, block_dim=16)
        else:
            self.info.vertex_colliding_triangles_count.zero_()
        if self.model.edge_count and detector.bvh_edges is not None:
            wp.launch(_swept_edges, self.model.edge_count,
                [self.model.edge_indices, self.swept_lower, self.swept_upper,
                 detector.lower_bounds_edges, detector.upper_bounds_edges], device=self.device)
            detector.bvh_edges.refit()
            wp.launch(_detect_swept_edges, self.model.edge_count,
                [detector.bvh_edges.id, self.swept_lower, self.swept_upper, self.model.particle_world,
                 self.node_body, self.model.edge_indices, self.info, self.error], device=self.device, block_dim=16)
        else:
            self.info.edge_colliding_edges_count.zero_()
        self.refresh()

    def accumulate(self, positions, previous, color, forces, hessians, dt, *,
                   colors=None, current_color=None, query_view=None):
        if current_color is not None:
            return self.accumulate_grouped(positions, previous, color, forces, hessians, dt,
                colors=colors, current_color=current_color, query_view=query_view)
        if colors is not None or query_view is not None:
            raise TypeError("Dynamic contact colors require a device current_color")
        p = parameters(dt=float(dt), **self.options)
        self.pending_count.zero_()
        if self.capacity:
            wp.launch(_evaluate_fast, self.fast_workers,
                [positions, previous, self.model.tri_indices, self.model.edge_indices, self.info,
                 self.candidate_ids, self.candidate_count, self.model.particle_colors, self.model.particle_flags,
                 self.model.particle_mass, int(color), p, self.fast_workers,
                 self.pending, self.pending_count, forces, hessians, self.error], device=self.device)
            wp.launch(self._resolve, self.workers,
                [positions, previous, self.model.tri_indices, self.model.edge_indices, self.info,
                 self.candidate_ids, self.model.particle_colors, self.model.particle_flags, self.model.particle_mass,
                 int(color), p, self.words, self.slots, self.workers, self.pending, self.pending_count,
                 forces, hessians, self.error], device=self.device, block_dim=1)

    def accumulate_grouped(self, positions, previous, color, forces, hessians, dt, *,
                           colors, current_color, query_view=None):
        """Assemble one device-selected GS color from its complete pair span.

        ``query_view`` owns an immutable snapshot of the coloring used to
        construct its spans. Its snapshot also controls scatter, keeping
        evaluation and candidate ownership in the same epoch. Without a view
        the kernel scans the original complete candidate bank.
        """
        count = self.model.particle_count
        if colors is None or colors.dtype != wp.int32 or colors.shape != (count,) or colors.device != self.device:
            raise TypeError("Dynamic contact colors must match the model device int array")
        if (current_color is None or current_color.dtype != wp.int32 or current_color.shape != (1,)
                or current_color.device != self.device):
            raise TypeError("Dynamic contact selection requires one device int")
        if not self.track_query_epoch:
            raise ValueError("Dynamic GS contact assembly requires tracked query epochs")
        ids, offsets, epoch = self.candidate_ids, self.candidate_count, self.query_epoch
        selected_colors = colors
        if query_view is not None:
            query_view.validate(self)
            if query_view.current_group is not current_color:
                raise ValueError("Contact query span and current_color must share their device selector")
            ids, offsets, epoch = query_view.ids, query_view.offsets, query_view.epoch
            selected_colors = query_view.node_groups
        p = parameters(dt=float(dt), **self.options)
        self.pending_count.zero_()
        if self.capacity:
            wp.launch(_evaluate_grouped, self.fast_workers,
                [positions, previous, self.model.tri_indices, self.model.edge_indices, self.info,
                 ids, self.candidate_count, selected_colors, self.model.particle_flags,
                 self.model.particle_mass, current_color, offsets, int(query_view is not None),
                 epoch, self.query_epoch, p, self.fast_workers, self.pending, self.pending_count,
                 forces, hessians, self.error], device=self.device)
            wp.launch(self._resolve_grouped, self.workers,
                [positions, previous, self.model.tri_indices, self.model.edge_indices, self.info,
                 ids, selected_colors, self.model.particle_flags, self.model.particle_mass,
                 current_color, p, self.words, self.slots, self.workers, self.pending,
                 self.pending_count, forces, hessians, self.error], device=self.device, block_dim=1)

    def reset_status(self):
        """Call outside captured frames, after the preceding check succeeded."""
        self.error.zero_()

    def check(self):
        code = int(self.error.numpy()[0])
        if code:
            names = {10: "upstream collision-row overflow", 11: "invalid collision topology",
                     12: "IPC queue capacity exhausted", 101: "invalid contact input",
                     102: "unresolved original contact geometry", 103: "zero contact distance",
                     104: "nonfinite contact derivatives", 105: "exact contact capacity exhausted",
                     201: "CCD initial intersection", 202: "invalid represented CCD motion",
                     203: "exact CCD capacity exhausted", 204: "CCD rounded-endpoint reproof exhausted",
                     302: "nonfinite initial-guess target", 350: "invalid initial-guess geometry",
                     351: "invalid initial-guess interval", 353: "initial-guess dyadic capacity exhausted",
                     354: "initial-guess heap capacity exhausted", 371: "invalid predictor graph",
                     372: "predictor graph capacity exhausted", 373: "nonfinite predictor trial",
                     602: "invalid rigid primitive ownership", 603: "invalid rigid finite pin",
                     604: "rigid local LDLT solve failed", 605: "nonfinite rigid proposal",
                     611: "rigid CCD initial intersection", 612: "invalid rigid curved/represented CCD",
                     613: "rigid exact capacity or arena ownership failure", 614: "rigid CCD backtracking exhausted"}
            raise RuntimeError("IPC collision backend: " + names.get(code, str(code)))


def __getattr__(name):
    if name == "evaluate_self_exact":
        from .contact_exact import evaluate_self_exact
        return evaluate_self_exact
    raise AttributeError(name)


configure_module()
