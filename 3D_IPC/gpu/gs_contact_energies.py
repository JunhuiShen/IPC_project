"""Native DAT IPC/friction equations on our GS candidate and color schedule.

The detector and CCD retain their own conservative geometry. Energy geometry
uses DAT's float32 closest points, frozen normal/weights and PSD local blocks.
EE candidates are undirected in GS, so both native directed evaluations are
performed once. No candidate, color or epoch data is downloaded during a sweep.
"""

from functools import cache
import math

import numpy as np
import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo
from newton._src.solvers.vbd import particle_vbd_kernels as stock

from . import contact, dat_energies


@wp.func
def _preflight(q: wp.array[wp.vec3], previous: wp.array[wp.vec3], nodes: wp.vec4i,
               kind: int, parallel_epsilon: float):
    status = int(0)
    for role in range(4):
        vertex = nodes[role]
        if vertex < 0 or vertex >= q.shape[0] or vertex >= previous.shape[0]:
            status = 101
        else:
            for axis in range(3):
                if not wp.isfinite(q[vertex][axis]) or not wp.isfinite(previous[vertex][axis]):
                    status = 101
    if status == 0:
        distance = float(0.0)
        if kind == 0:
            closest, bary, feature = stock.triangle_closest_point(
                q[nodes[1]], q[nodes[2]], q[nodes[3]], q[nodes[0]])
            distance = wp.length(q[nodes[0]] - closest)
        else:
            st = wp.closest_point_edge_edge(q[nodes[0]], q[nodes[1]],
                q[nodes[2]], q[nodes[3]], parallel_epsilon)
            distance = st[2]
        if not wp.isfinite(distance) or distance < 0.0:
            status = 102
        elif distance == 0.0:
            status = 103
    return status


@wp.func
def _scatter(vertex: int, force: wp.vec3, hessian: wp.mat33,
             colors: wp.array[int], flags: wp.array[int], mass: wp.array[float],
             color: int, forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    if colors[vertex] == color and flags[vertex] & 1 != 0 and mass[vertex] > 0.0:
        wp.atomic_add(forces, vertex, force)
        wp.atomic_add(hessians, vertex, hessian)


@cache
def make_accumulate(precision="float32"):
    edge_energy, triangle_energy = dat_energies._energy_wrappers(precision)

    @wp.kernel(module="unique", enable_backward=False)
    def accumulate(q: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                   triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                   ids: wp.array[int], count: wp.array[int], colors: wp.array[int], flags: wp.array[int],
                   mass: wp.array[float], host_color: int, current_color: wp.array[int], dynamic: int,
                   offsets: wp.array[int], use_view: int, view_epoch: wp.array[wp.uint64],
                   query_epoch: wp.array[wp.uint64], parallel_epsilon: float,
                   ctx: dat_energies.EnergyContext, workers: int,
                   forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33], error: wp.array[int]):
        color = host_color
        if dynamic != 0:
            color = current_color[0]
        begin = int(0)
        end = count[0]
        valid = color >= 0 and color < colors.shape[0]
        if use_view != 0 and valid:
            valid = view_epoch[0] == query_epoch[0]
            begin = offsets[color]
            end = offsets[color + 1]
        valid = valid and begin >= 0 and end >= begin and end <= ids.shape[0]
        if not valid:
            wp.atomic_max(error, 0, 11)
        if valid and error[0] == 0:
            for at in range(begin + wp.tid(), end, workers):
                encoded = ids[at]
                nodes, kind = contact.decode_candidate(encoded, triangles, edges, info)
                touches = False
                for role in range(4):
                    node = nodes[role]
                    if colors[node] == color and flags[node] & 1 != 0 and mass[node] > 0.0:
                        touches = True
                if touches and ctx.parameters.k_barrier > wp.float64(0.0) and ctx.parameters.d_hat > wp.float64(0.0):
                    status = _preflight(q, previous, nodes, kind, parallel_epsilon)
                    if status != 0:
                        wp.atomic_max(error, 0, status)
                    else:
                        radius = float(ctx.parameters.d_hat)
                        friction = float(ctx.parameters.friction)
                        epsilon = float(ctx.parameters.eps_v)
                        dt = float(ctx.parameters.dt)
                        if kind == 0:
                            vertex = info.vertex_colliding_triangles[2 * encoded]
                            triangle = info.vertex_colliding_triangles[2 * encoded + 1]
                            active, f0, f1, f2, f3, h0, h1, h2, h3 = triangle_energy(
                                vertex, triangle, q, previous, triangles, radius, 0.0, 0.0,
                                friction, epsilon, dt, True, ctx)
                            if active and error[0] == 0:
                                # The native VT return order is [a,b,c,vertex].
                                _scatter(nodes[0], f3, h3, colors, flags, mass, color, forces, hessians)
                                _scatter(nodes[1], f0, h0, colors, flags, mass, color, forces, hessians)
                                _scatter(nodes[2], f1, h1, colors, flags, mass, color, forces, hessians)
                                _scatter(nodes[3], f2, h2, colors, flags, mass, color, forces, hessians)
                        else:
                            slot = -encoded - 1
                            first = info.edge_colliding_edges[2 * slot]
                            second = info.edge_colliding_edges[2 * slot + 1]
                            active0, f0, f1, h0, h1 = edge_energy(first, second, q, previous,
                                edges, radius, 0.0, 0.0, friction, epsilon, dt, parallel_epsilon, True, ctx)
                            active1, f2, f3, h2, h3 = edge_energy(second, first, q, previous,
                                edges, radius, 0.0, 0.0, friction, epsilon, dt, parallel_epsilon, True, ctx)
                            if error[0] == 0:
                                if active0:
                                    _scatter(nodes[0], f0, h0, colors, flags, mass, color, forces, hessians)
                                    _scatter(nodes[1], f1, h1, colors, flags, mass, color, forces, hessians)
                                if active1:
                                    _scatter(nodes[2], f2, h2, colors, flags, mass, color, forces, hessians)
                                    _scatter(nodes[3], f3, h3, colors, flags, mass, color, forces, hessians)

    dat_energies.configure_module(accumulate)
    return accumulate


@wp.func_native(r"""
    wp::vec_t<178,double> out;
    // Construct the large value in native storage. Warp's indexed vector
    // updates otherwise generate a long SSA copy chain inside rigid kernels.
#if defined(__CUDA_ARCH__)
    #pragma unroll 1
#endif
    for(int k=0;k<178;++k)out[k]=0.0;
    out[0]=static_cast<double>(distance);
#if defined(__CUDA_ARCH__)
    #pragma unroll 1
#endif
    for(int role=0;role<4;++role){
        out[1+role]=static_cast<double>(weights[role]);
        for(int row=0;row<3;++row){
            // Keep every interior operation in float and the original
            // multiplication grouping before its final promotion to double.
            const float gradient=weights[role]*(first*normal[row]);
            out[8+3*role+row]=static_cast<double>(gradient);
            for(int other=0;other<4;++other)for(int column=0;column<3;++column){
                const float weight=weights[role]*weights[other];
                const float normal_product=(second*normal[row])*normal[column];
                out[20+36*role+9*other+3*row+column]=static_cast<double>(weight*normal_product);
            }
        }
    }
    for(int row=0;row<3;++row){
        out[5+row]=static_cast<double>(normal[row]);
        out[164+row]=(static_cast<double>(wp::neg(friction_force[row]))*dt)*dt;
        for(int column=0;column<3;++column)
            out[167+3*row+column]=(static_cast<double>(friction_hessian.data[row][column])*dt)*dt;
    }
    out[177]=1.0;
#if defined(__CUDA_ARCH__)
    #pragma unroll 1
#endif
    for(int index=0;index<176;++index)if(!isfinite(out[index])){
        out[176]=4.0;
        out[177]=0.0;
    }
    return out;
""")
def _pack_full(distance: float, weights: wp.vec4, normal: wp.vec3, first: float, second: float,
               friction_force: wp.vec3, friction_hessian: wp.mat33, dt: wp.float64) -> contact.FullPacket: ...


@cache
def make_full_evaluator(precision="float32"):
    """Return the same frozen-geometry model for rigid Jacobian pullbacks.

    FullPacket stores unit-stiffness barrier terms and dt²-scaled friction
    energy terms. Its sixteen world Hessian blocks are the PSD outer product
    of the frozen weighted contact normal, with no distance curvature.

    An EE packet uses the first edge's directed closest-point frame. For
    parallel overlapping segments the native reverse query can choose another
    equally close pair. A rigid body's assembly must orient its own edge first
    to match that body's native directed DAT terms. The packet's opposite-edge
    blocks remain a coherent, symmetric PSD extension of this chosen frame;
    they need not equal a separate reverse-direction native query.
    """
    if precision not in ("float32", "float64"):
        raise ValueError("DAT contact scalar precision must be float32 or float64")
    barrier = (dat_energies.evaluate_ipc_force_norm if precision == "float32"
               else dat_energies.evaluate_ipc_force_norm64)

    @wp.func
    def evaluate(q: wp.array[wp.vec3], previous: wp.array[wp.vec3], nodes: wp.vec4i,
                 kind: int, params: contact.Parameters):
        out = contact.FullPacket()
        if params.d_hat > wp.float64(0.0) and params.k_barrier > wp.float64(0.0):
            status = _preflight(q, previous, nodes, kind, 1.0e-5)
            if status != 0:
                out[176] = wp.float64(status - 100)
            else:
                weights = wp.vec4(0.0)
                difference = wp.vec3(0.0)
                motion = wp.vec3(0.0)
                distance = float(0.0)
                if kind == 0:
                    a = q[nodes[1]]
                    b = q[nodes[2]]
                    c = q[nodes[3]]
                    point = q[nodes[0]]
                    closest, barycentric, feature = stock.triangle_closest_point(a, b, c, point)
                    difference = point - closest
                    distance = wp.length(difference)
                    weights = wp.vec4(1.0, -barycentric[0], -barycentric[1], -barycentric[2])
                    closest_previous = (barycentric[0] * previous[nodes[1]]
                                        + barycentric[1] * previous[nodes[2]]
                                        + barycentric[2] * previous[nodes[3]])
                    motion = (point - previous[nodes[0]]) - (closest - closest_previous)
                else:
                    a = q[nodes[0]]
                    b = q[nodes[1]]
                    c = q[nodes[2]]
                    d = q[nodes[3]]
                    st = wp.closest_point_edge_edge(a, b, c, d, 1.0e-5)
                    s = st[0]
                    t = st[1]
                    closest_first = a + (b - a) * s
                    closest_second = c + (d - c) * t
                    difference = closest_first - closest_second
                    distance = st[2]
                    weights = wp.vec4(1.0 - s, s, -1.0 + t, -t)
                    previous_first = previous[nodes[0]] + (previous[nodes[1]] - previous[nodes[0]]) * s
                    previous_second = previous[nodes[2]] + (previous[nodes[3]] - previous[nodes[2]]) * t
                    motion = (closest_first - previous_first) - (closest_second - previous_second)
                out[0] = wp.float64(distance)
                if distance < float(params.d_hat):
                    normal = difference / distance
                    first, second = barrier(distance, float(params.d_hat), 1.0)
                    scaled_first, scaled_second = barrier(distance, float(params.d_hat), float(params.k_barrier))
                    axis0, axis1 = stock.orthonormal_basis(normal)
                    tangent = stock.mat32(axis0[0], axis1[0], axis0[1], axis1[1], axis0[2], axis1[2])
                    slip = wp.transpose(tangent) * motion
                    dt = float(params.dt)
                    friction_force, friction_hessian = stock.compute_friction(
                        float(params.friction), -scaled_first, tangent, slip, float(params.eps_v) * dt)
                    out = _pack_full(distance, weights, normal, first, second,
                        friction_force, friction_hessian, params.dt)
        return out

    return evaluate


class GsDatContact:
    """Persistent native-energy adapter; candidate and CCD ownership stay in GS."""

    def __init__(self, contacts, precision):
        self.contacts = contacts
        self.workspace = dat_energies.EnergyWorkspace(contacts.info, device=contacts.device,
            k_barrier=contacts.options["k_barrier"], error=contacts.error, precision=precision)
        self.kernel = make_accumulate(precision)
        self.current = wp.zeros(1, int, device=contacts.device)
        self.epoch = wp.zeros(1, wp.uint64, device=contacts.device)

    def accumulate(self, q, previous, color, forces, hessians, dt, *,
                   colors=None, current_color=None, query_view=None):
        c, model = self.contacts, self.contacts.model
        dynamic = current_color is not None
        if dynamic:
            n = model.particle_count
            if colors is None or colors.dtype != wp.int32 or colors.shape != (n,) or colors.device != c.device:
                raise TypeError("Dynamic contact colors must match the model device int array")
            if current_color.dtype != wp.int32 or current_color.shape != (1,) or current_color.device != c.device:
                raise TypeError("Dynamic contact selection requires one device int")
            if not c.track_query_epoch:
                raise ValueError("Dynamic GS contact assembly requires tracked query epochs")
        elif colors is not None or query_view is not None:
            raise TypeError("Dynamic contact colors require a device current_color")
        else:
            colors = model.particle_colors
            current_color = self.current
        ids, offsets = c.candidate_ids, c.candidate_count
        epoch = c.query_epoch if c.track_query_epoch else self.epoch
        view_epoch = epoch
        if query_view is not None:
            query_view.validate(c)
            if query_view.current_group is not current_color:
                raise ValueError("Contact query span and current_color must share their device selector")
            ids, offsets, view_epoch = query_view.ids, query_view.offsets, query_view.epoch
            colors = query_view.node_groups
        # Keep all host parameter validation, including representability, before
        # a captured launch. Context uses the same represented values as DAT.
        contact.parameters(dt=float(dt), **c.options)
        stiffness = c.options["k_barrier"]
        represented = np.float32(stiffness)
        if (isinstance(stiffness, bool) or not math.isfinite(stiffness) or stiffness < 0
                or not np.isfinite(represented) or (stiffness > 0 and represented == 0)):
            raise ValueError("Own k_barrier must be finite, nonnegative and representable in float32")
        self.workspace.k_barrier = float(represented)
        ctx = self.workspace.begin(dt, c.options["d_hat"], c.options["friction"], c.options["eps_v"])
        c.pending_count.zero_()
        if c.capacity:
            wp.launch(self.kernel, c.fast_workers,
                [q, previous, model.tri_indices, model.edge_indices, c.info, ids, c.candidate_count,
                 colors, model.particle_flags, model.particle_mass, int(color), current_color, int(dynamic),
                 offsets, int(query_view is not None), view_epoch, epoch, 1.0e-5, ctx, c.fast_workers,
                 forces, hessians, c.error], device=c.device)
