"""Our GPU Gauss-Seidel controller, schedules and cooperative material assembly.

Newton provides the project Model/State, adjacency, BVHs and timestep. This
module owns movement budgets, contact-independent colors, level schedules,
local material proposals and the GS hooks. Shared material equations remain
in energy.py and dat_energies.py; GS and DAT share the material laws and
PSD approximations without importing either runtime-dependent controller.
"""
from __future__ import annotations

import functools
from functools import cache
import math

import numpy as np
import warp as wp

from . import select_newton_runtime

select_newton_runtime("project")

from newton import Model, State, ParticleFlags
from newton.solvers import SolverVBD
from newton._src.solvers.solver import SolverBase
from newton._src.solvers.vbd.particle_vbd_kernels import forward_step
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo
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
from .ccd import CcdBackend
from .contact import decode_candidate
from .query_groups import QueryGroups
from .sdf import SdfBackend
from . import dat_energies as material_energy


# Cooperative material reductions and proposals.

# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0







TILE_SIZE = 16


@wp.func
def _membrane_terms(face: int, role: int, pos: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                    triangles: wp.array2d[int], inverse: wp.mat22, area: float,
                    mu: float, lam: float, dt: float):
    # Share DAT's corotated force, native PSD quasi-Newton Hessian and
    # degeneracy guards. Material damping is omitted by both energy adapters.
    return material_energy.evaluate_corotated_force_native_hessian(
        face, role, pos, previous, triangles, inverse, area, mu, lam, 0.0, dt)


@wp.func
def _bending_terms(edge: int, role: int, pos: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                   edges: wp.array2d[int], rest_angles: wp.array[float], rest_lengths: wp.array[float],
                   properties: wp.array2d[float], coefficients: wp.array[float], dt: float):
    # The GS bank stores our unweighted kB, just like DAT's own_stiffness.
    # Native E=(ke*L0/2)*delta_theta^2; ours E=kB*c_e*delta_theta^2.
    if edges[edge, 0] < 0 or edges[edge, 1] < 0 or properties[edge, 0] <= 0.0 or coefficients[edge] <= 0.0:
        return wp.vec3(0.0), wp.mat33(0.0)
    length = rest_lengths[edge]
    if not wp.isfinite(length) or length <= 0.0:
        return wp.vec3(0.0), wp.mat33(0.0)
    stiffness = ((2.0 * properties[edge, 0]) * coefficients[edge]) / length
    if not wp.isfinite(stiffness) or stiffness <= 0.0:
        return wp.vec3(0.0), wp.mat33(0.0)
    return material_energy.stock.evaluate_dihedral_angle_based_bending_force_hessian(
        edge, role, pos, previous, edges, rest_angles, rest_lengths, stiffness, 0.0, dt)


@wp.kernel(module="unique", enable_backward=False)
def _mask_material_banks(original_triangles: wp.array2d[float], original_bending: wp.array2d[float],
                         own_triangles: wp.array2d[float], own_bending: wp.array[float],
                         mask_triangles: int, mask_bending: int,
                         effective_triangles: wp.array2d[float], effective_bending: wp.array[float]):
    element = wp.tid()
    if mask_triangles != 0 and element < own_triangles.shape[0]:
        active = original_triangles[element, 0] > 0.0 or original_triangles[element, 1] > 0.0
        for parameter in range(2):
            value = float(0.0)
            if active:
                value = own_triangles[element, parameter]
            effective_triangles[element, parameter] = value
    if mask_bending != 0 and element < own_bending.shape[0]:
        value = float(0.0)
        if original_bending[element, 0] > 0.0:
            value = own_bending[element]
        effective_bending[element] = value


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
                        ef,eh=_membrane_terms(face,role,pos,pos_prev,tri_indices,
                            tri_poses[face],tri_areas[face],tri_materials[face,0],tri_materials[face,1],dt)
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
                        bf,bh=_bending_terms(edge,edge_role,pos,pos_prev,edge_indices,
                            edge_rest_angles,edge_rest_length,edge_bending_properties,bending_coefficients,dt)
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
            ef,eh=_membrane_terms(face,role,pos,pos_prev,tri_indices,
                tri_poses[face],tri_areas[face],tri_materials[face,0],tri_materials[face,1],dt)
            f+=ef
            h+=eh
    edge_count=wp.int32(0)
    if edge_indices.shape[0]>0:
        edge_count=get_vertex_num_adjacent_edges(particle_adjacency,particle)
    for edge_at in range(edge_count):
        edge,edge_role=get_vertex_adjacent_edge_id_order(particle_adjacency,particle,edge_at)
        if bending_coefficients[edge]>0.0 and edge_bending_properties[edge,0]>0.0:
            bf,bh=_bending_terms(edge,edge_role,pos,pos_prev,edge_indices,
                edge_rest_angles,edge_rest_length,edge_bending_properties,bending_coefficients,dt)
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

# Canonical parallel component and level schedules.

@wp.kernel
def _initialize(moving: wp.array[wp.uint8], movement: wp.array[wp.float64],
                colors: wp.array[int], trial_order: wp.array[int], seen: wp.array[int],
                labels: wp.array[int], largest: wp.array[wp.float64], error: wp.array[int]):
    v = wp.tid()
    labels[v] = v
    largest[v] = wp.float64(-1.0)
    if moving[v] > wp.uint8(1) or not wp.isfinite(movement[v]) or movement[v] < wp.float64(0.0):
        wp.atomic_max(error, 0, 373 if wp.isfinite(movement[v]) == False or movement[v] < wp.float64(0.0) else 371)
    q = trial_order[v]
    if q < 0 or q >= colors.shape[0]:
        wp.atomic_max(error, 0, 371)
    else:
        if colors[q] < 0 or colors[q] >= colors.shape[0]:
            wp.atomic_max(error, 0, 371)
        old = wp.atomic_add(seen, q, 1)
        if old != 0:
            wp.atomic_max(error, 0, 371)


@wp.kernel
def _validate_rows(rows: wp.array[wp.uint64], adjacency: wp.array[int], error: wp.array[int]):
    v = wp.tid()
    n = rows.shape[0]-1
    begin = rows[v]
    end = rows[v+1]
    if (v == 0 and begin != wp.uint64(0)) or begin > end or end > wp.uint64(adjacency.shape[0]):
        wp.atomic_max(error, 0, 371)
    elif error[0] == 0:
        previous = int(-1)
        for at in range(int(begin), int(end)):
            neighbor = adjacency[at]
            if neighbor < 0 or neighbor >= n or neighbor == v or neighbor <= previous:
                wp.atomic_max(error, 0, 371)
            previous = neighbor


@wp.kernel
def _start_loop(condition: wp.array[int], iterations: wp.array[int], error: wp.array[int]):
    condition[0] = int(error[0] == 0)
    iterations[0] = 0


@wp.kernel
def _next_loop(changed: wp.array[int], condition: wp.array[int], iterations: wp.array[int],
               maximum: int, error: wp.array[int]):
    iterations[0] += 1
    condition[0] = int(changed[0] != 0 and error[0] == 0)
    if condition[0] != 0 and iterations[0] > maximum:
        wp.atomic_max(error, 0, 371)
        condition[0] = 0


@wp.kernel
def _component_pass(rows: wp.array[wp.uint64], adjacency: wp.array[int],
                    source: wp.array[int], target: wp.array[int], changed: wp.array[int], error: wp.array[int]):
    v = wp.tid()
    if error[0] == 0:
        best = source[v]
        for at in range(int(rows[v]), int(rows[v+1])):
            best = wp.min(best, source[adjacency[at]])
        best = wp.min(best, source[best])
        target[v] = best
        if best != source[v]:
            wp.atomic_or(changed, 0, 1)


@wp.kernel
def _component_maximum(labels: wp.array[int], moving: wp.array[wp.uint8],
                        movement: wp.array[wp.float64], largest: wp.array[wp.float64], error: wp.array[int]):
    v = wp.tid()
    if error[0] == 0 and moving[v] != wp.uint8(0):
        wp.atomic_max(largest, labels[v], movement[v])


@wp.kernel
def _seed_levels(labels: wp.array[int], moving: wp.array[wp.uint8], movement: wp.array[wp.float64],
                  largest: wp.array[wp.float64], levels: wp.array[int], error: wp.array[int]):
    v = wp.tid()
    if error[0] == 0:
        maximum = largest[labels[v]]
        levels[v] = levels.shape[0]
        if maximum < wp.float64(0.0) or (moving[v] != wp.uint8(0) and movement[v] == maximum):
            levels[v] = 0


@wp.kernel
def _level_pass(rows: wp.array[wp.uint64], adjacency: wp.array[int], source: wp.array[int],
                target: wp.array[int], changed: wp.array[int], error: wp.array[int]):
    v = wp.tid()
    if error[0] == 0:
        best = source[v]
        for at in range(int(rows[v]), int(rows[v+1])):
            value = source[adjacency[at]]
            if value < source.shape[0]:
                best = wp.min(best, value+1)
        target[v] = best
        if best != source[v]:
            wp.atomic_or(changed, 0, 1)


@wp.kernel
def _active_keys(trial_order: wp.array[int], colors: wp.array[int], levels: wp.array[int],
                  moving: wp.array[wp.uint8], keys: wp.array[wp.int64], values: wp.array[int],
                  histogram: wp.array[int], active: wp.array[int], error: wp.array[int]):
    rank = wp.tid()
    if error[0] == 0:
        v = trial_order[rank]
        values[rank] = v
        if moving[v] != wp.uint8(0):
            level = levels[v]
            if level < 0 or level >= levels.shape[0]:
                wp.atomic_max(error, 0, 371)
            else:
                keys[rank] = wp.int64(level)*wp.int64(levels.shape[0])+wp.int64(rank)
                wp.atomic_add(histogram, level, 1)
                wp.atomic_add(active, 0, 1)
        else:
            keys[rank] = wp.int64(9223372036854775807)


@wp.kernel
def _boundaries(values: wp.array[int], levels: wp.array[int], colors: wp.array[int], active: wp.array[int],
                 order: wp.array[int], boundaries: wp.array[int], error: wp.array[int]):
    i = wp.tid()
    boundaries[i] = 0
    if error[0] == 0 and i < active[0]:
        v = values[i]
        order[i] = v
        boundary = int(i == 0)
        if i > 0:
            previous = values[i-1]
            boundary = int(levels[v] != levels[previous] or colors[v] != colors[previous])
        boundaries[i] = boundary


@wp.kernel
def _publish(boundaries: wp.array[int], group_scan: wp.array[int], active: wp.array[int],
              offsets: wp.array[int], metadata: wp.array[int], error: wp.array[int]):
    i = wp.tid()
    if error[0] == 0:
        if i < active[0] and boundaries[i] != 0:
            offsets[group_scan[i]-1] = i
        if i == 0:
            groups = int(0)
            if active[0] != 0:
                groups = group_scan[active[0]-1]
            metadata[1] = groups
            metadata[2] = active[0]
            offsets[groups] = active[0]


class ParallelLevels:
    def __init__(self, graph):
        self.graph, self.n, self.device = graph, graph.n, graph.device
        self.labels = [wp.empty(self.n, int, device=self.device) for _ in range(2)]
        self.level_temporary = wp.empty(self.n, int, device=self.device)
        self.seen = wp.zeros(self.n, int, device=self.device)
        self.largest = wp.empty(self.n, wp.float64, device=self.device)
        self.histogram = wp.zeros(self.n+1, int, device=self.device)
        self.boundaries = wp.zeros(self.n, int, device=self.device)
        self.group_scan = wp.empty(self.n, int, device=self.device)
        self.active = wp.zeros(1, int, device=self.device)
        self.changed = wp.zeros(1, int, device=self.device)
        self.condition = wp.zeros(1, int, device=self.device)
        self.iterations = wp.zeros(1, int, device=self.device)

    def apply(self, contacts):
        g, n, d = self.graph, self.n, self.device
        if n == 0:
            raise ValueError('Empty model is not a supported predictor epoch')
        self.seen.zero_(); self.histogram.zero_(); self.active.zero_()
        wp.launch(_initialize,n,[g.moving,g.movement,g.colors,g.trial_order,self.seen,self.labels[0],self.largest,contacts.error],device=d)
        wp.launch(_validate_rows,n,[g.rows,g.adjacency,contacts.error],device=d)
        wp.launch(_start_loop,1,[self.condition,self.iterations,contacts.error],device=d)
        def component_body():
            self.changed.zero_()
            wp.launch(_component_pass,n,[g.rows,g.adjacency,self.labels[0],self.labels[1],self.changed,contacts.error],device=d)
            wp.launch(_component_pass,n,[g.rows,g.adjacency,self.labels[1],self.labels[0],self.changed,contacts.error],device=d)
            wp.launch(_next_loop,1,[self.changed,self.condition,self.iterations,n,contacts.error],device=d)
        wp.capture_while(self.condition,component_body)
        wp.launch(_component_maximum,n,[self.labels[0],g.moving,g.movement,self.largest,contacts.error],device=d)
        wp.launch(_seed_levels,n,[self.labels[0],g.moving,g.movement,self.largest,g.levels,contacts.error],device=d)
        wp.launch(_start_loop,1,[self.condition,self.iterations,contacts.error],device=d)
        def level_body():
            self.changed.zero_()
            wp.launch(_level_pass,n,[g.rows,g.adjacency,g.levels,self.level_temporary,self.changed,contacts.error],device=d)
            wp.launch(_level_pass,n,[g.rows,g.adjacency,self.level_temporary,g.levels,self.changed,contacts.error],device=d)
            wp.launch(_next_loop,1,[self.changed,self.condition,self.iterations,n,contacts.error],device=d)
        wp.capture_while(self.condition,level_body)
        wp.launch(_active_keys,n,[g.trial_order,g.colors,g.levels,g.moving,g.keys,g.values,self.histogram,self.active,contacts.error],device=d)
        wp.utils.radix_sort_pairs(g.keys,g.values,n)
        wp.utils.array_scan(self.histogram,g.level_offsets,inclusive=True)
        wp.launch(_boundaries,n,[g.values,g.levels,g.colors,self.active,g.order,self.boundaries,contacts.error],device=d)
        wp.utils.array_scan(self.boundaries,self.group_scan,inclusive=True)
        wp.launch(_publish,n,[self.boundaries,self.group_scan,self.active,g.offsets,g.metadata,contacts.error],device=d)

# Canonical parallel greedy colors.

@wp.func_native(r"""
    unsigned long long forbidden[2]={0,0};
    for(uint64_t at=rows.data[vertex];at<rows.data[vertex+1];++at) {
        const int neighbor=adjacency.data[at];
        if(neighbor>=vertex)break;
        const int color=colors.data[neighbor];
        if(color<0)return -1;
        if(color<128)forbidden[color/64]|=1ull<<(color%64);
    }
    for(int word=0;word<2;++word) {
        const unsigned long long available=~forbidden[word];
        if(available) {
#if defined(__CUDA_ARCH__)
            return 64*word+__ffsll(static_cast<long long>(available))-1;
#else
            for(int bit=0;bit<64;++bit)if(available&(1ull<<bit))return 64*word+bit;
#endif
        }
    }
    // No color-count cap: graphs requiring more than 128 colors retain the
    // same exact minimum-free-color search as the original implementation.
    for(int color=128;color<colors.shape[0];++color) {
        bool available=true;
        for(uint64_t at=rows.data[vertex];at<rows.data[vertex+1];++at) {
            const int neighbor=adjacency.data[at];
            if(neighbor>=vertex)break;
            if(colors.data[neighbor]==color){available=false;break;}
        }
        if(available)return color;
    }
    return -1;
""")
def _ready_color(rows:wp.array[wp.uint64],adjacency:wp.array[int],colors:wp.array[int],vertex:int)->int: ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    const unsigned mask=__activemask();
    const unsigned votes=__ballot_sync(mask,pending!=0);
    return votes!=0 && static_cast<int>(threadIdx.x%32)==__ffs(mask)-1;
#else
    return pending!=0;
#endif
""")
def _pending_leader(pending:int)->bool: ...


@wp.kernel
def _color_pass(rows:wp.array[wp.uint64],adjacency:wp.array[int],source:wp.array[int],
                target:wp.array[int],pending:wp.array[int],error:wp.array[int]):
    vertex=wp.tid()
    remaining=int(0)
    if error[0]==0:
        color=source[vertex]
        if color<0:color=_ready_color(rows,adjacency,source,vertex)
        target[vertex]=color
        remaining=int(color<0)
    if _pending_leader(remaining):wp.atomic_or(pending,0,1)


@wp.kernel
def _publish_colors(colors:wp.array[int],trial_order:wp.array[int],metadata:wp.array[int],error:wp.array[int]):
    vertex=wp.tid()
    if error[0]==0:
        color=colors[vertex]
        if color<0 or color>=colors.shape[0]:wp.atomic_max(error,0,371)
        else:
            wp.atomic_max(metadata,0,color+1)
            wp.atomic_max(metadata,1,color+1)
            trial_order[vertex]=vertex
        if vertex==0:metadata[2]=colors.shape[0]


class ParallelColors:
    def __init__(self,graph):
        self.graph,self.n,self.device=graph,graph.n,graph.device
        self.temporary=wp.empty(self.n,int,device=self.device)
        self.pending=wp.zeros(1,int,device=self.device)
        self.condition=wp.zeros(1,int,device=self.device)
        self.iterations=wp.zeros(1,int,device=self.device)

    def apply(self,contacts):
        g,n,d=self.graph,self.n,self.device
        if n==0:raise ValueError('Empty model is not a supported predictor epoch')
        wp.launch(_validate_rows,n,[g.rows,g.adjacency,contacts.error],device=d)
        g.colors.fill_(-1);g.metadata.zero_()
        wp.launch(_start_loop,1,[self.condition,self.iterations,contacts.error],device=d)
        def body():
            self.pending.zero_()
            wp.launch(_color_pass,n,[g.rows,g.adjacency,g.colors,self.temporary,self.pending,contacts.error],device=d)
            wp.launch(_color_pass,n,[g.rows,g.adjacency,self.temporary,g.colors,self.pending,contacts.error],device=d)
            wp.launch(_next_loop,1,[self.pending,self.condition,self.iterations,n,contacts.error],device=d)
        wp.capture_while(self.condition,body)
        wp.launch(_publish_colors,n,[g.colors,g.trial_order,g.metadata,contacts.error],device=d)

# Movement budgets and contact-aware GS scheduling.

@wp.kernel
def _box_begin(velocity: wp.array[wp.vec3], previous: wp.array[wp.float64],
           dt: wp.float64, minimum: wp.float64, maximum: wp.float64,
           radii: wp.array[wp.float64]):
    i = wp.tid()
    v = velocity[i]
    speed = wp.sqrt(wp.float64(v[0]) * wp.float64(v[0])
                    + wp.float64(v[1]) * wp.float64(v[1])
                    + wp.float64(v[2]) * wp.float64(v[2]))
    radii[i] = wp.clamp(wp.float64(1.2) * wp.max(previous[i], dt * speed), minimum, maximum)


@wp.kernel
def _box_finish(position: wp.array[wp.vec3], start: wp.array[wp.vec3],
            previous: wp.array[wp.float64]):
    i = wp.tid()
    end = position[i]
    anchor = start[i]
    dx = wp.float64(end[0]) - wp.float64(anchor[0])
    dy = wp.float64(end[1]) - wp.float64(anchor[1])
    dz = wp.float64(end[2]) - wp.float64(anchor[2])
    previous[i] = wp.sqrt(dx * dx + dy * dy + dz * dz)


@wp.kernel
def _box_reset(previous: wp.array[wp.float64], radii: wp.array[wp.float64],
           maximum: wp.float64, worlds: wp.array[int], world_count: int,
           world_mask: wp.array[wp.bool], all_worlds: bool):
    i = wp.tid()
    world = worlds[i]
    selected = all_worlds
    if not all_worlds:
        if world == -1:
            world = world_count
        selected = world_mask[world]
    if selected:
        previous[i] = maximum
        radii[i] = maximum


class MovementBoxes:
    """Preallocated per-node budgets; every timestep operation stays on device."""

    def __init__(self, model, minimum, maximum):
        self.device = model.device
        self.count = model.particle_count
        self.minimum, self.maximum = float(minimum), float(maximum)
        if not (math.isfinite(self.minimum) and math.isfinite(self.maximum)
                and 1.0e-10 < self.minimum <= self.maximum):
            raise ValueError("Movement-box radii must be finite and larger than the CCD inset")
        self.radii = wp.full(self.count, self.maximum, wp.float64, device=self.device)
        # The CPU workspace also starts with the maximum so the first solve
        # has its full movement budget before a displacement history exists.
        self.previous_displacement = wp.full(self.count, self.maximum, wp.float64, device=self.device)
        self.start = wp.empty_like(model.particle_q)
        self._worlds = model.particle_world
        self._world_count = model.world_count

    def begin(self, velocity, dt):
        """Freeze radii from input velocity, before the predictor changes positions."""
        if self.count:
            wp.launch(_box_begin, self.count,
                [velocity, self.previous_displacement, float(dt), self.minimum, self.maximum, self.radii],
                device=self.device)

    def anchor(self, positions):
        """Anchor history after the predictor; later box checkpoints do not call this."""
        wp.copy(self.start, positions)

    def finish(self, positions):
        if self.count:
            wp.launch(_box_finish, self.count,
                [positions, self.start, self.previous_displacement], device=self.device)

    def reset(self, world_mask=None):
        """Discard history for the same worlds reset by the parent solver."""
        if self.count:
            wp.launch(_box_reset, self.count,
                [self.previous_displacement, self.radii, self.maximum, self._worlds,
                 self._world_count, world_mask, world_mask is None], device=self.device)


@wp.func_native(r"""
    unsigned long long forbidden[2]={0,0};
    for(uint64_t at=rows.data[vertex];at<rows.data[vertex+1];++at) {
        const int neighbor=adjacency.data[at];
        if(neighbor>=vertex)break;
        const int color=colors.data[neighbor];
        if(color>=0&&color<128)forbidden[color/64]|=1ull<<(color%64);
    }
    for(int word=0;word<2;++word) {
        const unsigned long long available=~forbidden[word];
        if(available) {
#if defined(__CUDA_ARCH__)
            return 64*word+__ffsll(static_cast<long long>(available))-1;
#else
            for(int bit=0;bit<64;++bit)if(available&(1ull<<bit))return 64*word+bit;
#endif
        }
    }
    // Complete graphs and arbitrary valid seeds retain an uncapped mex.
    for(int color=128;color<colors.shape[0];++color) {
        bool available=true;
        for(uint64_t at=rows.data[vertex];at<rows.data[vertex+1];++at) {
            const int neighbor=adjacency.data[at];
            if(neighbor>=vertex)break;
            if(colors.data[neighbor]==color){available=false;break;}
        }
        if(available)return color;
    }
    return -1;
""")
def _color_least_free(rows: wp.array[wp.uint64], adjacency: wp.array[int], colors: wp.array[int], vertex: int) -> int: ...


@wp.kernel
def _color_validate_seed(colors: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    if colors[vertex] < -1 or colors[vertex] >= colors.shape[0]:
        wp.atomic_max(error, 0, 391)


@wp.kernel
def _color_repair(rows: wp.array[wp.uint64], adjacency: wp.array[int], source: wp.array[int],
            target: wp.array[int], changed: wp.array[int], track: int, error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0:
        color = _color_least_free(rows, adjacency, source, vertex)
        if color < 0 or color >= source.shape[0]:
            wp.atomic_max(error, 0, 392)
        else:
            target[vertex] = color
            if track != 0 and color != source[vertex]:
                wp.atomic_or(changed, 0, 1)


@wp.kernel
def _color_repair_next(changed: wp.array[int], condition: wp.array[int], iterations: wp.array[int],
          maximum: int, error: wp.array[int]):
    iterations[0] += 1
    condition[0] = int(changed[0] != 0 and error[0] == 0)
    if condition[0] != 0 and iterations[0] > maximum:
        wp.atomic_max(error, 0, 393)
        condition[0] = 0


@wp.kernel
def _color_publish(colors: wp.array[int], trial_order: wp.array[int], metadata: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0:
        color = colors[vertex]
        if color < 0 or color >= colors.shape[0]:
            wp.atomic_max(error, 0, 392)
        else:
            wp.atomic_max(metadata, 0, color + 1)
            wp.atomic_max(metadata, 1, color + 1)
            trial_order[vertex] = vertex
        if vertex == 0:
            metadata[2] = colors.shape[0]


class WarmColors:
    """Persistent two-buffer Jacobi repair, with no host convergence readback."""
    def __init__(self, graph):
        self.graph, self.n, self.device = graph, graph.n, graph.device
        self.temporary = wp.empty(self.n, int, device=self.device)
        self.changed = wp.zeros(1, int, device=self.device)
        self.condition = wp.zeros(1, int, device=self.device)
        self.iterations = wp.zeros(1, int, device=self.device)

    def apply(self, contacts):
        g, n, d = self.graph, self.n, self.device
        if n == 0:
            raise ValueError("Empty model is not a supported GS coloring epoch")
        wp.launch(_validate_rows, n, [g.rows, g.adjacency, contacts.error], device=d)
        wp.launch(_color_validate_seed, n, [g.colors, contacts.error], device=d)
        g.metadata.zero_()
        wp.launch(_start_loop, 1, [self.condition, self.iterations, contacts.error], device=d)
        def body():
            wp.launch(_color_repair, n, [g.rows, g.adjacency, g.colors, self.temporary, self.changed, 0, contacts.error], device=d)
            self.changed.zero_()
            wp.launch(_color_repair, n, [g.rows, g.adjacency, self.temporary, g.colors, self.changed, 1, contacts.error], device=d)
            # If pass two changes nothing, F(temporary)==temporary: the
            # published result is a fixed point, even if pass one changed it.
            wp.launch(_color_repair_next, 1, [self.changed, self.condition, self.iterations, n + 1, contacts.error], device=d)
        wp.capture_while(self.condition, body)
        wp.launch(_color_publish, n, [g.colors, g.trial_order, g.metadata, contacts.error], device=d)


@wp.func
def _edge(vertex: int, neighbor: int, stride: int, bits: wp.array[wp.uint32], error: wp.array[int]):
    n = bits.shape[0] // stride
    if vertex < 0 or vertex >= n or neighbor < 0 or neighbor >= n:
        wp.atomic_max(error, 0, 381)
    elif neighbor < vertex:
        # Greedy coloring only consumes lower-ID neighbors. Storing each edge
        # once also halves the largest possible CSR allocation.
        wp.atomic_or(bits, vertex * stride + neighbor // 32, wp.uint32(1) << wp.uint32(neighbor % 32))


@wp.kernel
def _material_edges(triangles: wp.array2d[int], edges: wp.array2d[int], tets: wp.array2d[int],
                    stride: int, bits: wp.array[wp.uint32], error: wp.array[int]):
    at = wp.tid()
    if error[0] == 0:
        if at < triangles.shape[0]:
            for a in range(3):
                for b in range(3):
                    _edge(triangles[at, a], triangles[at, b], stride, bits, error)
        elif at < triangles.shape[0] + edges.shape[0]:
            hinge = at - triangles.shape[0]
            # Boundary edges contain an absent apex and no bending energy.
            if edges[hinge, 0] >= 0 and edges[hinge, 1] >= 0:
                for a in range(4):
                    for b in range(4):
                        _edge(edges[hinge, a], edges[hinge, b], stride, bits, error)
        else:
            tet = at - triangles.shape[0] - edges.shape[0]
            if tet < tets.shape[0]:
                for a in range(4):
                    for b in range(4):
                        _edge(tets[tet, a], tets[tet, b], stride, bits, error)


@wp.kernel
def _contact_edges(triangles: wp.array2d[int], edges: wp.array2d[int],
                   info: TriMeshCollisionInfo, ids: wp.array[int], count: wp.array[int],
                   workers: int, stride: int, bits: wp.array[wp.uint32], error: wp.array[int]):
    if error[0] == 0:
        if count[0] < 0 or count[0] > ids.shape[0]:
            wp.atomic_max(error, 0, 382)
        else:
            for at in range(wp.tid(), count[0], workers):
                nodes, kind = decode_candidate(ids[at], triangles, edges, info)
                for a in range(4):
                    for b in range(4):
                        _edge(nodes[a], nodes[b], stride, bits, error)


@wp.func_native(r"""
    int count=0;
    for(int word=0;word<stride;++word) {
        uint32_t value=bits.data[vertex*stride+word];
        while(value){value&=value-1;++count;}
    }
    return count;
""")
def _degree(bits: wp.array[wp.uint32], stride: int, vertex: int) -> int: ...


@wp.kernel
def _degrees(bits: wp.array[wp.uint32], stride: int, counts: wp.array[int]):
    vertex = wp.tid()
    if vertex + 1 < counts.shape[0]:
        counts[vertex] = _degree(bits, stride, vertex)
    else:
        counts[vertex] = 0


@wp.func_native(r"""
    int at=offsets.data[vertex];
    for(int word=0;word<stride;++word) {
        uint32_t value=bits.data[vertex*stride+word];
        for(int bit=0;bit<32;++bit)if(value&(uint32_t(1)<<bit)) {
            if(at>=adjacency.shape[0]){error.data[0]=382;return;}
            adjacency.data[at++]=32*word+bit;
        }
    }
""")
def _fill_row(bits: wp.array[wp.uint32], stride: int, vertex: int, offsets: wp.array[int],
              adjacency: wp.array[int], error: wp.array[int]): ...


@wp.kernel
def _rows(bits: wp.array[wp.uint32], stride: int, offsets: wp.array[int], rows: wp.array[wp.uint64],
          adjacency: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    rows[vertex] = wp.uint64(offsets[vertex])
    if vertex + 1 < rows.shape[0] and error[0] == 0:
        _fill_row(bits, stride, vertex, offsets, adjacency, error)


@wp.kernel
def _count_colors(colors: wp.array[int], counts: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0:
        color = colors[vertex]
        if color < 0 or color >= colors.shape[0]:
            wp.atomic_max(error, 0, 383)
        else:
            wp.atomic_add(counts, color, 1)


@wp.kernel
def _pack_colors(colors: wp.array[int], offsets: wp.array[int], cursors: wp.array[int],
                 order: wp.array[int], error: wp.array[int]):
    vertex = wp.tid()
    if error[0] == 0:
        color = colors[vertex]
        at = offsets[color] + wp.atomic_add(cursors, color, 1)
        if at < offsets[color] or at >= offsets[color + 1] or at >= order.shape[0]:
            wp.atomic_max(error, 0, 383)
        else:
            order[at] = vertex


@wp.kernel
def _begin_color(current: wp.array[int], condition: wp.array[int], metadata: wp.array[int], error: wp.array[int]):
    current[0] = 0
    condition[0] = int(metadata[0] > 0 and error[0] == 0)


@wp.kernel
def _next_color(current: wp.array[int], condition: wp.array[int], metadata: wp.array[int], error: wp.array[int]):
    current[0] += 1
    condition[0] = int(current[0] < metadata[0] and error[0] == 0)


@wp.kernel
def _check_contacts(triangles: wp.array2d[int], edges: wp.array2d[int], info: TriMeshCollisionInfo,
                    ids: wp.array[int], count: wp.array[int], colors: wp.array[int],
                    workers: int, error: wp.array[int]):
    if error[0] == 0:
        for at in range(wp.tid(), count[0], workers):
            nodes, kind = decode_candidate(ids[at], triangles, edges, info)
            for a in range(4):
                for b in range(a):
                    if nodes[a] != nodes[b] and colors[nodes[a]] == colors[nodes[b]]:
                        wp.atomic_max(error, 0, 384)


@cache
def make_dynamic_elasticity(include_triangles: bool, include_tets: bool, tiled: bool = True):
    """Solve compact color slots with unchanged material arithmetic.

    A positive budget runs a fixed-size pool over just the device-resident
    active group. Every pool round includes both half-warps, even for padded
    or inactive slots, so the original width-16 shuffle participation holds.
    Zero retains the full-size launch used by the numerical oracle.
    """
    @wp.kernel(module="unique", enable_backward=False)
    def solve(
        dt: float, order: wp.array[int], offsets: wp.array[int], current: wp.array[int],
        pos_prev: wp.array[wp.vec3], pos: wp.array[wp.vec3], mass: wp.array[float],
        inertia: wp.array[wp.vec3], flags: wp.array[int],
        triangles: wp.array2d[int], tri_poses: wp.array[wp.mat22], tri_materials: wp.array2d[float], tri_areas: wp.array[float],
        edges: wp.array2d[int], rest_angles: wp.array[float], rest_lengths: wp.array[float], bending: wp.array2d[float],
        tets: wp.array2d[int], tet_poses: wp.array[wp.mat33], tet_materials: wp.array2d[float], adjacency: MeshAdjacencyData,
        contact_forces: wp.array[wp.vec3], contact_hessians: wp.array[wp.mat33],
        pin_targets: wp.array[wp.vec3], pinned: wp.array[int], kpin: float, bending_coefficients: wp.array[float],
        tet_grad_N: wp.array2d[wp.vec3], tet_measures: wp.array[float], damping: float, worker_slots: int,
        displacements: wp.array[wp.vec3], proposals: wp.array[wp.vec3], error: wp.array[int],
    ):
        tid = wp.tid()
        slot = tid
        lane = int(0)
        step = int(1)
        if wp.static(tiled):
            slot = tid // 16
            lane = tid % 16
            step = 16
        first_slot = slot
        group = current[0]
        begin = int(0)
        end = int(0)
        if group < 0 or group + 1 >= offsets.shape[0]:
            if tid == 0:
                wp.atomic_max(error, 0, 385)
        else:
            begin = offsets[group]
            end = offsets[group + 1]
            if begin < 0 or end < begin or end > order.shape[0]:
                begin = 0
                end = 0
                if tid == 0:
                    wp.atomic_max(error, 0, 385)
        active_count = end - begin
        if worker_slots < 0:
            active_count = 0
            if tid == 0:
                wp.atomic_max(error, 0, 385)
        if wp.static(tiled):
            if worker_slots % 2 != 0:
                active_count = 0
                if tid == 0:
                    wp.atomic_max(error, 0, 385)
        slot = first_slot
        warp_first_slot = first_slot
        if wp.static(tiled):
            warp_first_slot = (first_slot // 2) * 2
        # The predicate is uniform across both half-warps, even with wider
        # thread blocks. An odd final slot keeps its padded half in every
        # shuffle; whole unused warps exit without energy or reduction work.
        # Additive grid strides avoid a runtime division and round counter.
        while warp_first_slot < active_count:
            valid = slot < active_count and error[0] == 0
            particle = int(0)
            dynamic = False
            if valid:
                particle = order[begin + slot]
                if particle < 0 or particle >= pos.shape[0]:
                    valid = False
                    if lane == 0:
                        wp.atomic_max(error, 0, 385)
                else:
                    dynamic = flags[particle] & 1 != 0 and mass[particle] > 0.0
            force = wp.vec3(0.0)
            hessian = wp.mat33(0.0)
            if dynamic:
                if wp.static(include_triangles):
                    face_count = int(0)
                    if triangles.shape[0] > 0:
                        face_count = get_vertex_num_adjacent_faces(adjacency, particle)
                    face_at = int(lane)
                    while face_at < face_count:
                        face, role = get_vertex_adjacent_face_id_order(adjacency, particle, face_at)
                        if tri_materials[face, 0] > 0.0 or tri_materials[face, 1] > 0.0:
                            f, h = _membrane_terms(face, role, pos, pos_prev, triangles,
                                tri_poses[face], tri_areas[face], tri_materials[face, 0], tri_materials[face, 1], dt)
                            force += f
                            hessian += h
                        face_at += step
                    edge_count = int(0)
                    if edges.shape[0] > 0:
                        edge_count = get_vertex_num_adjacent_edges(adjacency, particle)
                    edge_at = int(lane)
                    while edge_at < edge_count:
                        edge, role = get_vertex_adjacent_edge_id_order(adjacency, particle, edge_at)
                        if bending_coefficients[edge] > 0.0 and bending[edge, 0] > 0.0:
                            f, h = _bending_terms(edge, role, pos, pos_prev, edges,
                                rest_angles, rest_lengths, bending, bending_coefficients, dt)
                            force += f
                            hessian += h
                        edge_at += step
                if wp.static(include_tets):
                    tet_count = int(0)
                    if tets.shape[0] > 0:
                        tet_count = get_vertex_num_adjacent_tets(adjacency, particle)
                    tet_at = int(lane)
                    while tet_at < tet_count:
                        tet, role = get_vertex_adjacent_tet_id_order(adjacency, particle, tet_at)
                        if tet_materials[tet, 0] > 0.0 or tet_materials[tet, 1] > 0.0:
                            value = energy.solid(pos[tets[tet, 0]], pos[tets[tet, 1]], pos[tets[tet, 2]], pos[tets[tet, 3]],
                                tet_poses[tet], tet_grad_N[tet, role], tet_measures[tet], tet_materials[tet, 0], tet_materials[tet, 1])
                            f, h = energy.unpack(value)
                            force += f
                            hessian += h
                        tet_at += step
            total_f = force
            total_h = hessian
            if wp.static(tiled):
                total_f = wp.vec3(_half_reduce(force[0]), _half_reduce(force[1]), _half_reduce(force[2]))
                total_h = wp.mat33(_half_reduce(hessian[0, 0]), _half_reduce(hessian[0, 1]), _half_reduce(hessian[0, 2]),
                    _half_reduce(hessian[1, 0]), _half_reduce(hessian[1, 1]), _half_reduce(hessian[1, 2]),
                    _half_reduce(hessian[2, 0]), _half_reduce(hessian[2, 1]), _half_reduce(hessian[2, 2]))
            if lane == 0 and valid:
                if dynamic:
                    _finish_particle(particle, dt, pos, mass, inertia, contact_forces, contact_hessians,
                        pin_targets, pinned, kpin, damping, total_f, total_h, displacements, proposals, error)
                else:
                    displacements[particle] = wp.vec3(0.0)
                    proposals[particle] = pos[particle]
            if worker_slots == 0:
                break
            slot += worker_slots
            warp_first_slot += worker_slots
    energy.configure_module(solve)
    return solve


class GsSchedule:
    """Persistent contact-aware color graph and its device-controlled sweep.

    Construct before graph capture, call ``rebuild`` immediately after each box
    candidate refresh, and call ``sweep(body)`` once per GS iteration. The body
    reads ``current_color`` and ``colors`` and uses ``query_view`` for contacts.
    """
    def __init__(self, contacts, *, warm_colors=True, material_slot_budget=None):
        self.contacts, self.model, self.device = contacts, contacts.model, contacts.device
        self.n = self.model.particle_count
        if self.n <= 0:
            raise ValueError("Contact-aware particle GS needs a nonempty model")
        if not getattr(contacts, "track_query_epoch", False):
            raise ValueError("Contact-aware GS requires candidate epoch tracking")
        self.stride = (self.n + 31) // 32
        material_edges = 3 * self.model.tri_count + 6 * self.model.edge_count + 6 * self.model.tet_count
        capacity = min(self.n * (self.n - 1) // 2, material_edges + 6 * contacts.capacity)
        if self.n * self.stride >= 2**31 or capacity >= 2**31:
            raise ValueError("Contact-aware GS graph exceeds its int32 GPU index capacity")
        self.material_bits = wp.zeros(self.n * self.stride, wp.uint32, device=self.device)
        self.bits = wp.empty_like(self.material_bits)
        self.degrees = wp.zeros(self.n + 1, int, device=self.device)
        self.scan = wp.empty_like(self.degrees)
        self.rows = wp.empty(self.n + 1, wp.uint64, device=self.device)
        self.adjacency = wp.empty(capacity, int, device=self.device)
        self.colors = wp.full(self.n, -1, int, device=self.device)
        self.trial_order = wp.empty(self.n, int, device=self.device)
        self.metadata = wp.zeros(3, int, device=self.device)
        self.counts = wp.zeros(self.n + 1, int, device=self.device)
        self.offsets = wp.zeros(self.n + 1, int, device=self.device)
        self.cursors = wp.zeros(self.n + 1, int, device=self.device)
        self.order = wp.empty(self.n, int, device=self.device)
        self.current_color = wp.zeros(1, int, device=self.device)
        self.condition = wp.zeros(1, int, device=self.device)
        self.parallel_colors = WarmColors(self) if warm_colors else ParallelColors(self)
        self.query_groups = QueryGroups(contacts)
        if not self.query_groups.enabled:
            raise ValueError("Contact-aware GS candidate grouping exceeds its int32 GPU index capacity")
        self.query_view = None
        primitive_count = self.model.tri_count + self.model.edge_count + self.model.tet_count
        if primitive_count:
            wp.launch(_material_edges, primitive_count,
                [self.model.tri_indices, self.model.edge_indices, self.model.tet_indices,
                 self.stride, self.material_bits, contacts.error], device=self.device)
        if material_slot_budget is None:
            # Two 32-thread blocks per SM, capped at 256 blocks, with two
            # vertex slots per block. Larger groups reuse this bounded pool.
            material_slot_budget = (2 * min((self.n + 1) // 2, 256, 2 * max(1, self.device.sm_count))
                                    if self.device.is_cuda else 0)
        if isinstance(material_slot_budget, bool) or not isinstance(material_slot_budget, int) or material_slot_budget < 0:
            raise ValueError("Material slot budget must be a nonnegative integer")
        if self.device.is_cuda and material_slot_budget % 2:
            raise ValueError("Cooperative material pools require an even slot budget")
        self.material_slot_budget = material_slot_budget
        self._elasticity = make_dynamic_elasticity(self.model.tri_count > 0 or self.model.edge_count > 0,
                                                   self.model.tet_count > 0, self.device.is_cuda)

    def rebuild(self):
        c, m = self.contacts, self.model
        wp.copy(self.bits, self.material_bits)
        if c.capacity:
            wp.launch(_contact_edges, c.fast_workers,
                [m.tri_indices, m.edge_indices, c.info, c.candidate_ids, c.candidate_count,
                 c.fast_workers, self.stride, self.bits, c.error], device=self.device)
        wp.launch(_degrees, self.n + 1, [self.bits, self.stride, self.degrees], device=self.device)
        wp.utils.array_scan(self.degrees, self.scan, inclusive=False)
        wp.launch(_rows, self.n + 1,
            [self.bits, self.stride, self.scan, self.rows, self.adjacency, c.error], device=self.device)
        self.parallel_colors.apply(c)
        self.counts.zero_()
        self.cursors.zero_()
        wp.launch(_count_colors, self.n, [self.colors, self.counts, c.error], device=self.device)
        wp.utils.array_scan(self.counts, self.offsets, inclusive=False)
        wp.launch(_pack_colors, self.n, [self.colors, self.offsets, self.cursors, self.order, c.error], device=self.device)
        self.query_groups.build(self.colors)
        self.query_view = self.query_groups.view(self.current_color)

    def validate_independence(self):
        """Optional device check for tests/debugging of a candidate epoch."""
        c, m = self.contacts, self.model
        if c.capacity:
            wp.launch(_check_contacts, c.fast_workers,
                [m.tri_indices, m.edge_indices, c.info, c.candidate_ids, c.candidate_count,
                 self.colors, c.fast_workers, c.error], device=self.device)

    def sweep(self, body):
        if self.query_view is None:
            raise RuntimeError("Rebuild contact-aware GS before its first sweep")
        wp.launch(_begin_color, 1, [self.current_color, self.condition, self.metadata, self.contacts.error], device=self.device)
        def color_body():
            body()
            wp.launch(_next_color, 1, [self.current_color, self.condition, self.metadata, self.contacts.error], device=self.device)
        wp.capture_while(self.condition, color_body)

    def solve_elasticity(self, solver, state_in, dt):
        m, materials = self.model, solver._energy_materials
        arguments = [dt, self.order, self.offsets, self.current_color, solver.particle_q_prev,
            state_in.particle_q, m.particle_mass, solver.inertia, m.particle_flags,
            m.tri_indices, m.tri_poses, solver._energy_triangle_materials, m.tri_areas,
            m.edge_indices, m.edge_rest_angle, m.edge_rest_length, solver._energy_bending,
            m.tet_indices, m.tet_poses, m.tet_materials, solver.particle_adjacency,
            solver.particle_forces, solver.particle_hessians, materials.pin_targets, materials.pin_mask,
            materials.pin_stiffness, materials.edge_coefficients, materials.tet_gradients, materials.tet_measures,
            float(solver.params["damping"]), self.material_slot_budget]
        # The fixed-size pool reads the active count on device. No launch
        # dimension or color count is copied to the host during a timestep.
        if self.device.is_cuda:
            slots = self.material_slot_budget or 2 * ((self.n + 1) // 2)
            dimension = slots * 16
        else:
            dimension = self.material_slot_budget or self.n
        block_dim = 32 if self.device.is_cuda else 64
        wp.launch(self._elasticity, dimension, block_dim=block_dim, inputs=arguments,
            outputs=[solver.particle_displacements, solver.proposals, solver.error], device=self.device)

# Newton timestep integration.

class SolverIPC(SolverVBD):
    """Our GS/IPC/CCD model plugged into the authors' concrete GPU solver.

    Adaptive movement boxes and contact-independent colors follow CPU GS.
    Float32 state and tree reductions can change the numeric trajectory. Mixed
    rigid scenes retain the inherited particle schedule and separate rigid
    block order until a unified mixed dependency graph is implemented.
    """

    def __init__(self, model: Model, *, iterations=None, vertex_contact_capacity=128,
                 edge_contact_capacity=256, allow_cpu=False):
        if not hasattr(model, "ipc"):
            raise TypeError("Build the actual Newton Model with gpu.scene.create_model")
        if not model.device.is_cuda and not allow_cpu:
            raise ValueError("The GPU solver requires an explicit CUDA model")
        p = model.ipc.params
        for name in ("use_dat", "use_ogc", "use_cloth_grid", "use_jacobi_level_based_ccd"):
            if p.get(name, False):
                raise ValueError(f"{name} requests another algorithm; this solver uses our IPC/CCD physics")
        if not p.get("fixed_iters", False):
            raise ValueError("The captured GPU solver requires --fixed_iters, like the native GPU driver")
        for name in ("node_box_max", "node_box_min", "dt"):
            if not math.isfinite(float(p[name])) or p[name] <= 0:
                raise ValueError(f"Invalid solver parameter {name}")
        if p["node_box_max"] < p["node_box_min"] or p["node_box_update_count"] <= 0:
            raise ValueError("Invalid movement-box configuration")
        if p.get("ccd_exact_step_bound", False) and not p.get("exact_computation_fallback", True):
            raise ValueError("Exact-step proposals require robust CCD")
        self.error = wp.zeros(1, int, device=model.device)
        self._failed = False
        self.params = p
        self._contact_aware_gs = bool(p.get("contact_aware_gs", True)) and model.body_count == 0 and model.particle_count > 0
        self._box_radius = float(p["node_box_max"])
        self._adaptive_node_boxes = bool(p.get("adaptive_node_boxes", True))
        self.movement_boxes = MovementBoxes(model, p["node_box_min"], p["node_box_max"])
        # The detector queries current geometry. Cover both primitives' full
        # coordinatewise blue-box budgets until the next detection checkpoint.
        self._query_radius = max(float(p["d_hat"]), 1e-8) + 2.0 * self._box_radius
        self._query_radius += 32.0 * np.finfo(np.float32).eps * max(1.0, self._query_radius)
        base_iterations = p["max_global_iters"] if iterations is None else iterations
        if isinstance(base_iterations, bool) or not isinstance(base_iterations, int) or base_iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        super().__init__(model, iterations=base_iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=max(float(p["d_hat"]), 1e-8),
            particle_self_contact_gap=self._query_radius-max(float(p["d_hat"]), 1e-8),
            particle_vertex_contact_buffer_size=vertex_contact_capacity,
            particle_edge_contact_buffer_size=edge_contact_capacity,
            particle_topological_contact_filter_threshold=0,
            particle_rest_shape_contact_exclusion_radius=0.0,
            particle_enable_tile_solve=True,
            rigid_soft_enable_dat=False,
            collision_frequency_type={SolverBase.CollisionSlot.SOFT_SELF_CONTACT: SolverBase.CollisionFrequencyType.ITERATIONS},
            collision_frequency={SolverBase.CollisionSlot.SOFT_SELF_CONTACT: int(p["node_box_update_count"])},
            enable_fast_math=False)
        self._prepare_material_banks()
        self.proposals = wp.empty_like(model.particle_q)
        from .contact import ContactBackendIPC
        self.contact_backend = ContactBackendIPC(model, self.trimesh_collision_info,
            grouped_precision=p.get("gs_contact_precision", "float32"), energy_model="dat",
            d_hat=float(p["d_hat"]), k_barrier=float(p["k_barrier"]),
            friction=float(p["friction_coefficient"]), eps_v=float(p["friction_velocity_epsilon"]),
            exact_geometry=bool(p["exact_computation_fallback"]), node_body=model.ipc.node_body, error=self.error,
            track_query_epoch=bool(self._contact_aware_gs or p.get("use_colored_ccd_guess", False) or
                                   p.get("use_colored_ccd_guess_level_based", False)))
        self.ccd_backend = CcdBackend(self.contact_backend, node_box_radius=self._box_radius,
            exact_geometry=bool(p["exact_computation_fallback"]), error=self.error, group_query_block_dim=64)
        self.ccd_backend.single_vertex_filter = bool(p.get("gs_ccd_filter", True))
        if not self.ccd_backend.exact_step_bound:
            from .ccd import resolve_cached
            self.ccd_backend._resolve = resolve_cached
        flags = model.particle_flags.numpy()
        has_dynamic_particles = bool(np.any((flags & 1) & (model.particle_mass.numpy() > 0)))
        self._guess_requested = has_dynamic_particles and bool(p.get("use_colored_ccd_guess", False) or
            p.get("use_colored_ccd_guess_level_based", False) or
            (p.get("use_ccd_guess", False) and not model.ipc.payload.get("uses_general_solver", False) and model.body_count == 0))
        if self._guess_requested:
            self.ccd_backend.prepare_initial_guess(self.trimesh_collision_detector, self.params, use_group_queries=True,
                independent_colors=bool(p.get("use_colored_ccd_guess", False))
                    and not p.get("use_colored_ccd_guess_level_based", False) and model.body_count == 0)
            graph = self.ccd_backend._initial_guess.graph
            if graph is not None:
                graph.prepare_parallel_colors(ParallelColors(graph))
                if p.get("use_colored_ccd_guess_level_based", False):
                    graph.prepare_parallel_levels(ParallelLevels(graph))
        self.sdf_backend = SdfBackend(model, self.error)
        self.configuration = {
            "algorithm": "project GS schedule and CCD acceptance",
            "energy_membrane": {"force": "corotated", "hessian": "native PSD approximation",
                                "singular_geometry": "native area-ratio floor"},
            "energy_bending": {"evaluator": "native with stiffness mapping", "degeneracy": "native guards"},
            "energy_solid": "shared corotated force and frozen-rotation PSD Hessian",
            "energy_contact": dict(self.contact_backend.configuration),
            "energy_sdf": {"precision": self.sdf_backend.precision,
                           "include_curvature": bool(self.sdf_backend.params.include_curvature),
                           "normal_hessian": "full spatial curvature" if self.sdf_backend.params.include_curvature else "PSD normal outer product",
                           "geometry": "float32 with binary64 replay" if self.sdf_backend.precision == "float32" else "binary64",
                           "material_transport": "binary64",
                           "friction_hessian": "frozen tangent PSD; zero at zero slip"},
            "rigid_contact_hessian": "frozen contact geometry and Jacobian PSD pullback",
            "local_solve": "project rank-revealing solve and configured damping",
        }
        include_tets = model.tet_count > 0
        include_triangles = bool(np.any(model.ipc.tri_active.numpy())) or bool(np.any(model.ipc.edge_active.numpy()))
        self._ipc_particles_per_block = 1 if include_tets else 2
        self._ipc_elasticity_kernel = make_solve_elasticity_tile(
            include_triangles, include_tets, not include_tets) if model.device.is_cuda else solve_elasticity_scalar
        self._active_color_counts = []
        for group in model.particle_color_groups:
            ids = group.numpy()
            self._active_color_counts.append(int(np.count_nonzero(flags[ids] & 1)))
        self._gs_schedule = None
        self._box_filter = None
        self.skip_empty_mesh_contacts = True
        if self._contact_aware_gs:
            self._gs_schedule = GsSchedule(self.contact_backend)
            if self._adaptive_node_boxes and p.get("gs_prune_box_contacts", True):
                from .movement_box_queries import GsBoxEpochFilter
                self._box_filter = GsBoxEpochFilter(self.contact_backend)
        self._rigid = None
        if model.body_count:
            from .rigid import BodySolver
            self._rigid = BodySolver(self)
            self._query_radius = max(self._query_radius, self._rigid.query_radius)

    def _prepare_material_banks(self):
        model = self.model
        # Read the same immutable material banks as DAT. The reshape aliases
        # the stiffness array, so GPU-authored material updates remain visible.
        self._energy_materials = material_energy.prepare(model)
        materials = self._energy_materials
        # DAT guards activity with original model slots before reading separate
        # energy banks. Model materials/topology are immutable scene inputs;
        # separately supplied own banks may be updated on the GPU between sweeps.
        self._mask_triangle_materials = False
        if hasattr(model.ipc, "dat_triangle_materials") and model.tri_count:
            original_triangles = model.tri_materials.numpy()
            self._mask_triangle_materials = bool(np.any(
                (original_triangles[:, 0] <= 0.0) & (original_triangles[:, 1] <= 0.0)))
        self._mask_bending_materials = False
        if hasattr(model.ipc, "dat_bending_stiffness") and model.edge_count:
            # Zero rest coefficients already suppress boundary/non-hinge rows
            # in both adapters, regardless of future own-stiffness updates.
            self._mask_bending_materials = bool(np.any(
                (model.edge_bending_properties.numpy()[:, 0] <= 0.0) &
                (materials.edge_coefficients.numpy() > 0.0)))
        self._energy_triangle_materials = (wp.empty_like(materials.triangle_materials)
            if self._mask_triangle_materials else materials.triangle_materials)
        self._energy_bending_stiffness = (wp.empty_like(materials.bending_stiffness)
            if self._mask_bending_materials else materials.bending_stiffness)
        self._energy_bending = self._energy_bending_stiffness.reshape((model.edge_count, 1))
        self._refresh_material_banks()

    def _refresh_material_banks(self):
        if self._mask_triangle_materials or self._mask_bending_materials:
            m, materials = self.model, self._energy_materials
            wp.launch(_mask_material_banks, max(m.tri_count, m.edge_count),
                [m.tri_materials, m.edge_bending_properties, materials.triangle_materials,
                 materials.bending_stiffness, int(self._mask_triangle_materials), int(self._mask_bending_materials),
                 self._energy_triangle_materials, self._energy_bending_stiffness], device=self.device)

    # The public timestep is inherited; these hooks supply our model and GS order.
    def _pre_initialization_detection(self, state_in, dt):
        # Upstream needs a checkpoint before its DAT-limited prediction. Our
        # predictor queries its own complete swept bounds, and initialization
        # below always rebuilds at the accepted initial guess. A second refit
        # of the same input mesh would add no candidate coverage.
        return False

    def _initialize_particles(self, state_in: State, state_out: State, contacts, dt):
        m = self.model
        if not m.particle_count:
            return
        if self._adaptive_node_boxes:
            self.movement_boxes.begin(state_in.particle_qd, dt)
        wp.launch(forward_step, dim=m.particle_count,
            inputs=[dt, m.gravity, m.particle_world, self.particle_q_prev,
                    state_in.particle_q, state_in.particle_qd, m.particle_inv_mass,
                    state_in.particle_f, m.particle_flags],
            outputs=[self.inertia, self.particle_displacements], device=self.device)
        self.particle_displacements.zero_()
        self.sdf_backend.prepare(dt)
        if self._guess_requested:
            self.ccd_backend.initial_guess(state_in.particle_q, self.particle_q_prev,
                state_in.particle_qd, self.inertia, dt, self.params, detector=self.trimesh_collision_detector)
        if self._adaptive_node_boxes:
            self.movement_boxes.anchor(state_in.particle_q)
        self._collision_detection_penetration_free(state_in, reset_reference=True)
        if self.iterations == 0:
            # Ordinarily the inherited color loop publishes the final iterate.
            wp.copy(state_out.particle_q, state_in.particle_q)

    def _collision_detection_penetration_free(self, current_state: State, *, reset_reference=True):
        if reset_reference:
            self.pos_prev_collision_detection.assign(current_state.particle_q)
            self.particle_displacements.zero_()
        if self._adaptive_node_boxes and self._rigid is None:
            self.contact_backend.detect_boxes(self.trimesh_collision_detector,
                current_state.particle_q, self.movement_boxes.radii, float(self.params["d_hat"]))
        else:
            # Rigid analytic motion retains its own conservative scalar reach.
            self.contact_backend.detect(self.trimesh_collision_detector, current_state.particle_q,
                                        self._query_radius, refit=True)
        if self._rigid is not None:
            self._rigid.refresh(current_state)
        if self._box_filter is not None:
            from .contact import parameters
            # A whole-box proof removes only interactions that stay inactive
            # throughout this epoch. All uncertain pairs remain dependencies.
            self._box_filter.apply(current_state.particle_q, self.particle_q_prev,
                self.contact_backend.swept_lower, self.contact_backend.swept_upper,
                parameters(dt=float(self.params["dt"]), **self.contact_backend.options))
        if self._gs_schedule is not None:
            self._gs_schedule.rebuild()

    def _self_contact_is_due(self, iter_num):
        return iter_num > 0 and iter_num % int(self.params["node_box_update_count"]) == 0

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        self._refresh_material_banks()
        gs = self._gs_schedule
        if gs is None:
            return super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        self.particle_forces.zero_()
        self.particle_hessians.zero_()

        def color_body(include_mesh):
            if include_mesh:
                self.contact_backend.accumulate(state_in.particle_q, self.particle_q_prev, -1,
                    self.particle_forces, self.particle_hessians, dt,
                    colors=gs.colors, current_color=gs.current_color, query_view=gs.query_view)
            if self.model.ipc.collider_data.shape[0]:
                self.sdf_backend.accumulate(state_in.particle_q, self.particle_q_prev, -1,
                    self.particle_forces, self.particle_hessians, dt,
                    colors=gs.colors, current_color=gs.current_color)
            gs.solve_elasticity(self, state_in, dt)
            self.ccd_backend.accept(state_in.particle_q, self.proposals, -1, dt,
                center=self.pos_prev_collision_detection, radius=self._box_radius,
                radii=self.movement_boxes.radii if self._adaptive_node_boxes else None,
                colors=gs.colors, current_color=gs.current_color, query_view=gs.query_view,
                independent=True, bounded=True, clip_ccd=bool(self.params["use_ccd"]))

        if self.skip_empty_mesh_contacts and self.device.is_cuda and self.device.is_capturing:
            # Check parameters even when the device branch skips mesh
            # evaluation. Vertex/epoch checks still run in CCD preparation.
            from .contact import parameters
            parameters(dt=float(dt), **self.contact_backend.options)

            def with_mesh():
                gs.sweep(lambda: color_body(True))

            def without_mesh():
                fast = getattr(self.contact_backend, "_grouped_fast", None)
                if fast is not None:
                    fast.ordinary_count.zero_()
                self.contact_backend.pending_count.zero_()
                gs.sweep(lambda: color_body(False))

            wp.capture_if(self.contact_backend.candidate_count, on_true=with_mesh, on_false=without_mesh)
        else:
            gs.sweep(lambda: color_body(True))
        wp.copy(state_out.particle_q, state_in.particle_q)

    def _accumulate_particle_contact_terms(self, state_in, state_out, contacts, dt, color):
        if not self._active_color_counts[color]:
            return
        self.contact_backend.accumulate(state_in.particle_q, self.particle_q_prev, color,
                                       self.particle_forces, self.particle_hessians, dt)
        if self.model.ipc.collider_data.shape[0]:
            self.sdf_backend.accumulate(state_in.particle_q, self.particle_q_prev, color,
                self.particle_forces, self.particle_hessians, dt)

    def _solve_particle_elasticity_color(self, state_in, state_out, contacts, dt, color):
        if not self._active_color_counts[color]:
            return
        m, materials = self.model, self._energy_materials
        arguments = [dt, m.particle_color_groups[color], self.particle_q_prev,
            state_in.particle_q, m.particle_mass, self.inertia, m.particle_flags,
            m.tri_indices, m.tri_poses, self._energy_triangle_materials, m.tri_areas,
            m.edge_indices, m.edge_rest_angle, m.edge_rest_length, self._energy_bending,
            m.tet_indices, m.tet_poses, m.tet_materials, self.particle_adjacency,
            self.particle_forces, self.particle_hessians,
            materials.pin_targets, materials.pin_mask, materials.pin_stiffness, materials.edge_coefficients,
            materials.tet_gradients, materials.tet_measures, float(self.params["damping"])]
        count = m.particle_color_groups[color].size
        if self.device.is_cuda:
            per_block = self._ipc_particles_per_block
            block_dim = 16 * per_block
            dimension = ((count + per_block - 1) // per_block) * block_dim
        else:
            block_dim, dimension = 64, count
        wp.launch(self._ipc_elasticity_kernel, dim=dimension, block_dim=block_dim,
            inputs=arguments, outputs=[self.particle_displacements, self.proposals, self.error], device=self.device)

    def _finish_particle_color(self, state_in, state_out, contacts, dt, color):
        if not self._active_color_counts[color]:
            return
        self.ccd_backend.accept(state_in.particle_q, self.proposals, color, dt,
                                center=self.pos_prev_collision_detection, radius=self._box_radius,
                                radii=self.movement_boxes.radii if self._adaptive_node_boxes else None,
                                bounded=True, clip_ccd=bool(self.params["use_ccd"]))

    def _finalize_particles(self, state_out: State, dt):
        if self._adaptive_node_boxes:
            self.movement_boxes.finish(state_out.particle_q)
        super()._finalize_particles(state_out, dt)

    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, world_mask=world_mask, flags=flags)
        self.movement_boxes.reset(self._normalize_reset_world_mask(world_mask))

    def _initialize_rigid_bodies(self, state_in, control, contacts, dt, update_rigid):
        if self._rigid is not None:
            self._rigid.begin(state_in, dt)

    def _solve_rigid_body_iteration(self, state_in, state_out, control, contacts, dt):
        if self._rigid is not None:
            self._rigid.iterate(state_in, dt)

    def _finalize_rigid_bodies(self, state_in, state_out, dt):
        if self._rigid is not None:
            self._rigid.finish(state_in, state_out, dt)

    def _snapshot_rigid_contact_history(self, contacts, *, force=False):
        # Our lagged friction is based on the fixed step-start position array.
        return

    def prepare_capture(self):
        """All workspaces are allocated in __init__; CUDA launches warm on capture."""
        self.check_status()

    def reset_status(self):
        if self._failed:
            raise RuntimeError("A failed solver cannot be reused; restore with a fresh instance")
        self.error.zero_()

    def check_status(self):
        if self._failed:
            raise RuntimeError("The solver is invalidated by a previous GPU failure")
        try:
            wp.synchronize_device(self.device)
            # All components share this latched error word. One download also
            # checks CCD; calling each backend would download it repeatedly.
            self.contact_backend.check()
        except Exception:
            self._failed = True
            raise


assert SolverIPC.step is SolverVBD.step


energy.configure_module(__name__)
wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
