# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The imported VBD cooperative elasticity layout with our energy functions.

The 16-lane vertex groups, half-warp packing, adjacency gathers and reduction
tree follow Newton's ``make_solve_elasticity_tile``. Material evaluations,
finite pins, damped GS solve and proposal publication implement our model.
Every evolving array and the local reduction/solve use float32. The separate
``energy`` reference functions retain our original binary64 arithmetic for tests.

Only proposals for the current color are written. Collision kernels must use
current positions for other colors, rather than reading stale proposals.
"""

import functools

import warp as wp

from newton import ParticleFlags
from newton._src.utils.mesh import (
    MeshAdjacencyData,
    get_vertex_adjacent_edge_id_order,
    get_vertex_adjacent_face_id_order,
    get_vertex_adjacent_tet_id_order,
    get_vertex_num_adjacent_edges,
    get_vertex_num_adjacent_faces,
    get_vertex_num_adjacent_tets,
)

from . import energy


TILE_SIZE = 16


@wp.func_native("""
    #if defined(__CUDA_ARCH__)
    float r=value;
    const unsigned mask=blockDim.x==16?0x0000ffffu:0xffffffffu;
    r+=__shfl_xor_sync(mask,r,8,16);
    r+=__shfl_xor_sync(mask,r,4,16);
    r+=__shfl_xor_sync(mask,r,2,16);
    r+=__shfl_xor_sync(mask,r,1,16);
    return r;
    #else
    return value;
    #endif
""")
def _half_reduce(value: float) -> float: ...


@wp.func
def _finish_particle(particle: int, dt: float, pos: wp.array[wp.vec3], mass: wp.array[float],
                     inertia: wp.array[wp.vec3], contact_forces: wp.array[wp.vec3],
                     contact_hessians: wp.array[wp.mat33], pin_targets: wp.array[wp.vec3],
                     pin_mask: wp.array[int], kpin: float, damping: float,
                     force: wp.vec3, hessian: wp.mat33,
                     displacements: wp.array[wp.vec3], proposals: wp.array[wp.vec3], error: wp.array[int]):
    inv_dt2 = 1.0 / (dt * dt)
    f = force + mass[particle] * inv_dt2 * (inertia[particle] - pos[particle])
    h = hessian + mass[particle] * inv_dt2 * wp.identity(n=3,dtype=float)
    if pin_mask[particle] != 0:
        f -= kpin * (pos[particle] - pin_targets[particle])
        h += kpin * wp.identity(n=3,dtype=float)
    f += contact_forces[particle]
    h += contact_hessians[particle]
    finite = True
    for a in range(3):
        if not wp.isfinite(f[a]):
            finite = False
        for b in range(3):
            if not wp.isfinite(h[a,b]):
                finite = False
    if not finite:
        wp.atomic_max(error,0,1)
        proposals[particle] = pos[particle]
    else:
        delta, valid = energy.solve(h,f)
        if valid == 0:
            wp.atomic_max(error,0,2)
            proposals[particle] = pos[particle]
        else:
            movement = damping * delta
            target = pos[particle] + movement
            for a in range(3):
                if not wp.isfinite(target[a]):
                    wp.atomic_max(error,0,3)
            proposals[particle] = target
            displacements[particle] = displacements[particle] + movement


@functools.cache
def make_solve_elasticity_tile(include_triangles: bool, include_tets: bool, two_particles_per_warp: bool):
    """Return a cooperative kernel with the upstream argument order plus IPC data.

    After ``particle_hessians`` append ``pin_targets, pin_mask, kpin,
    bending_coefficients, tet_grad_N, tet_measures, damping``. Outputs are
    ``particle_displacements, proposals, error``. Pass own/native rest poses
    through the ordinary ``tri_poses``/``tet_poses`` slots. Two-particle variants
    require 32-thread blocks; the one-particle variant requires 16-thread blocks.
    """

    @wp.kernel(module="unique",enable_backward=False)
    def solve_elasticity_tile_kernel(
        dt: float,
        particle_ids_in_color: wp.array[wp.int32],
        pos_prev: wp.array[wp.vec3],
        pos: wp.array[wp.vec3],
        mass: wp.array[float],
        inertia: wp.array[wp.vec3],
        particle_flags: wp.array[wp.int32],
        tri_indices: wp.array2d[wp.int32],
        tri_poses: wp.array[wp.mat22],
        tri_materials: wp.array2d[float],
        tri_areas: wp.array[float],
        edge_indices: wp.array2d[wp.int32],
        edge_rest_angles: wp.array[float],
        edge_rest_length: wp.array[float],
        edge_bending_properties: wp.array2d[float],
        tet_indices: wp.array2d[wp.int32],
        tet_poses: wp.array[wp.mat33],
        tet_materials: wp.array2d[float],
        particle_adjacency: MeshAdjacencyData,
        particle_forces: wp.array[wp.vec3],
        particle_hessians: wp.array[wp.mat33],
        pin_targets: wp.array[wp.vec3],
        pin_mask: wp.array[wp.int32],
        kpin: float,
        bending_coefficients: wp.array[float],
        tet_grad_N: wp.array2d[wp.vec3],
        tet_measures: wp.array[float],
        damping: float,
        particle_displacements: wp.array[wp.vec3],
        proposals: wp.array[wp.vec3],
        error: wp.array[wp.int32],
    ):
        tid=wp.tid()
        slot=tid//TILE_SIZE
        lane=tid%TILE_SIZE
        valid_slot=True
        if wp.static(two_particles_per_warp):
            valid_slot=slot<particle_ids_in_color.shape[0]
        particle=wp.int32(0)
        dynamic=False
        if valid_slot:
            particle=particle_ids_in_color[slot]
            dynamic=(particle_flags[particle]&ParticleFlags.ACTIVE)!=0 and mass[particle]!=0.0
        if wp.static(not two_particles_per_warp):
            if not dynamic:
                if lane==0:
                    particle_displacements[particle]=wp.vec3(0.0)
                    proposals[particle]=pos[particle]
                return
        f=wp.vec3(0.0)
        h=wp.mat33(0.0)
        if dynamic:
            if wp.static(include_triangles):
                face_count=wp.int32(0)
                if tri_indices.shape[0]>0:
                    face_count=get_vertex_num_adjacent_faces(particle_adjacency,particle)
                at=wp.int32(lane)
                while at<face_count:
                    face,role=get_vertex_adjacent_face_id_order(particle_adjacency,particle,at)
                    if tri_materials[face,0]>0.0 or tri_materials[face,1]>0.0:
                        value=energy.triangle(pos[tri_indices[face,0]],pos[tri_indices[face,1]],pos[tri_indices[face,2]],
                            tri_poses[face],tri_areas[face],tri_materials[face,0],tri_materials[face,1],role)
                        ef,eh=energy.unpack(value)
                        f+=ef
                        h+=eh
                    at+=TILE_SIZE
                edge_count=wp.int32(0)
                if edge_indices.shape[0]>0:
                    edge_count=get_vertex_num_adjacent_edges(particle_adjacency,particle)
                edge_at=wp.int32(lane)
                while edge_at<edge_count:
                    edge,edge_role=get_vertex_adjacent_edge_id_order(particle_adjacency,particle,edge_at)
                    if bending_coefficients[edge]>0.0 and edge_bending_properties[edge,0]>0.0:
                        native_role=edge_role+2
                        if edge_role>=2:
                            native_role=edge_role-2
                        bend=energy.hinge(pos[edge_indices[edge,2]],pos[edge_indices[edge,3]],
                            pos[edge_indices[edge,0]],pos[edge_indices[edge,1]],bending_coefficients[edge],
                            edge_rest_angles[edge],edge_bending_properties[edge,0],native_role)
                        bf,bh=energy.unpack(bend)
                        f+=bf
                        h+=bh
                    edge_at+=TILE_SIZE
            if wp.static(include_tets):
                tet_count=wp.int32(0)
                if tet_indices.shape[0]>0:
                    tet_count=get_vertex_num_adjacent_tets(particle_adjacency,particle)
                tet_at=wp.int32(lane)
                while tet_at<tet_count:
                    tet,tet_role=get_vertex_adjacent_tet_id_order(particle_adjacency,particle,tet_at)
                    if tet_materials[tet,0]>0.0 or tet_materials[tet,1]>0.0:
                        solid_value=energy.solid(pos[tet_indices[tet,0]],pos[tet_indices[tet,1]],
                            pos[tet_indices[tet,2]],pos[tet_indices[tet,3]],tet_poses[tet],tet_grad_N[tet,tet_role],
                            tet_measures[tet],tet_materials[tet,0],tet_materials[tet,1])
                        sf,sh=energy.unpack(solid_value)
                        f+=sf
                        h+=sh
                    tet_at+=TILE_SIZE
        # Both supported launch layouts are half-warp groups. An explicit
        # butterfly avoids a shared-memory temporary for the single-slot case.
        total_f=wp.vec3(_half_reduce(f[0]),_half_reduce(f[1]),_half_reduce(f[2]))
        total_h=wp.mat33(_half_reduce(h[0,0]),_half_reduce(h[0,1]),_half_reduce(h[0,2]),
            _half_reduce(h[1,0]),_half_reduce(h[1,1]),_half_reduce(h[1,2]),
            _half_reduce(h[2,0]),_half_reduce(h[2,1]),_half_reduce(h[2,2]))
        if lane==0 and valid_slot:
            if not dynamic:
                particle_displacements[particle]=wp.vec3(0.0)
                proposals[particle]=pos[particle]
            else:
                _finish_particle(particle,dt,pos,mass,inertia,particle_forces,particle_hessians,
                    pin_targets,pin_mask,kpin,damping,total_f,total_h,particle_displacements,proposals,error)

    energy.configure_module(solve_elasticity_tile_kernel)
    return solve_elasticity_tile_kernel


@wp.kernel
def solve_elasticity_scalar(
    dt: float, particle_ids_in_color: wp.array[wp.int32], pos_prev: wp.array[wp.vec3], pos: wp.array[wp.vec3],
    mass: wp.array[float], inertia: wp.array[wp.vec3], particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32], tri_poses: wp.array[wp.mat22], tri_materials: wp.array2d[float],
    tri_areas: wp.array[float], edge_indices: wp.array2d[wp.int32], edge_rest_angles: wp.array[float],
    edge_rest_length: wp.array[float], edge_bending_properties: wp.array2d[float], tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33], tet_materials: wp.array2d[float], particle_adjacency: MeshAdjacencyData,
    particle_forces: wp.array[wp.vec3], particle_hessians: wp.array[wp.mat33], pin_targets: wp.array[wp.vec3],
    pin_mask: wp.array[wp.int32], kpin: float, bending_coefficients: wp.array[float],
    tet_grad_N: wp.array2d[wp.vec3], tet_measures: wp.array[float], damping: float, particle_displacements: wp.array[wp.vec3],
    proposals: wp.array[wp.vec3], error: wp.array[wp.int32],
):
    particle=particle_ids_in_color[wp.tid()]
    if (particle_flags[particle]&ParticleFlags.ACTIVE)==0 or mass[particle]==0.0:
        particle_displacements[particle]=wp.vec3(0.0)
        proposals[particle]=pos[particle]
        return
    f=wp.vec3(0.0)
    h=wp.mat33(0.0)
    face_count=wp.int32(0)
    if tri_indices.shape[0]>0:
        face_count=get_vertex_num_adjacent_faces(particle_adjacency,particle)
    for at in range(face_count):
        face,role=get_vertex_adjacent_face_id_order(particle_adjacency,particle,at)
        if tri_materials[face,0]>0.0 or tri_materials[face,1]>0.0:
            ef,eh=energy.unpack(energy.triangle(pos[tri_indices[face,0]],pos[tri_indices[face,1]],pos[tri_indices[face,2]],
                tri_poses[face],tri_areas[face],tri_materials[face,0],tri_materials[face,1],role))
            f+=ef
            h+=eh
    edge_count=wp.int32(0)
    if edge_indices.shape[0]>0:
        edge_count=get_vertex_num_adjacent_edges(particle_adjacency,particle)
    for edge_at in range(edge_count):
        edge,edge_role=get_vertex_adjacent_edge_id_order(particle_adjacency,particle,edge_at)
        if bending_coefficients[edge]>0.0 and edge_bending_properties[edge,0]>0.0:
            native_role=edge_role+2
            if edge_role>=2:native_role=edge_role-2
            bf,bh=energy.unpack(energy.hinge(pos[edge_indices[edge,2]],pos[edge_indices[edge,3]],
                pos[edge_indices[edge,0]],pos[edge_indices[edge,1]],bending_coefficients[edge],edge_rest_angles[edge],
                edge_bending_properties[edge,0],native_role))
            f+=bf
            h+=bh
    tet_count=wp.int32(0)
    if tet_indices.shape[0]>0:
        tet_count=get_vertex_num_adjacent_tets(particle_adjacency,particle)
    for tet_at in range(tet_count):
        tet,tet_role=get_vertex_adjacent_tet_id_order(particle_adjacency,particle,tet_at)
        if tet_materials[tet,0]>0.0 or tet_materials[tet,1]>0.0:
            sf,sh=energy.unpack(energy.solid(pos[tet_indices[tet,0]],pos[tet_indices[tet,1]],
                pos[tet_indices[tet,2]],pos[tet_indices[tet,3]],tet_poses[tet],tet_grad_N[tet,tet_role],
                tet_measures[tet],tet_materials[tet,0],tet_materials[tet,1]))
            f+=sf
            h+=sh
    _finish_particle(particle,dt,pos,mass,inertia,particle_forces,particle_hessians,
        pin_targets,pin_mask,kpin,damping,f,h,particle_displacements,proposals,error)


energy.configure_module(__name__)
