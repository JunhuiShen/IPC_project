# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Stock kernel bodies retained with explicitly guarded energy-only substitutions.

"""Stock tiled VBD layout with only our material/pin energy terms substituted.

The original inverse, determinant threshold, accumulative displacement update,
16-lane reduction and packing remain verbatim. No proposal or DAT-sync stages.
"""
import ast
import copy
import functools
import inspect
import textwrap

import warp as wp
from newton._src.solvers.vbd import particle_vbd_kernels as stock
from . import energy
from .dat_materials import (evaluate_own_membrane_force_hessian, evaluate_own_bending_force_hessian,
    evaluate_own_solid_force_hessian, evaluate_own_pin_force_hessian)

MeshAdjacencyData=stock.MeshAdjacencyData
ParticleFlags=stock.ParticleFlags
TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE=stock.TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
VBD_DEBUG_PRINTING_OPTIONS=stock.VBD_DEBUG_PRINTING_OPTIONS
_warp_half_reduce_sum=stock._warp_half_reduce_sum
get_vertex_num_adjacent_faces=stock.get_vertex_num_adjacent_faces
get_vertex_adjacent_face_id_order=stock.get_vertex_adjacent_face_id_order
get_vertex_num_adjacent_edges=stock.get_vertex_num_adjacent_edges
get_vertex_adjacent_edge_id_order=stock.get_vertex_adjacent_edge_id_order
get_vertex_num_adjacent_tets=stock.get_vertex_num_adjacent_tets
get_vertex_adjacent_tet_id_order=stock.get_vertex_adjacent_tet_id_order


def _configure_stock_execution(kernel, triangles, tets, two):
    reference=stock.make_solve_elasticity_tile(triangles,tets,two)
    options=wp.get_module_options(reference.module)
    energy.configure_module(kernel)
    # Do not change non-energy arithmetic/compiler policy. The solver must
    # likewise register the same execution options on its substituted kernel.
    wp.set_module_options({key:options[key] for key in ('fast_math','fuse_fp') if key in options},module=kernel.module)


_ENERGY_ARGUMENTS=('energy_triangle_materials','energy_edge_coefficients','energy_bending_stiffness','energy_pin_mask','energy_pin_targets',
    'energy_pin_stiffness','energy_tet_gradients','energy_tet_measures')
_PIN_TILE='''if energy_pin_mask[particle_index] != 0:
    f_pin, h_pin = evaluate_own_pin_force_hessian(
        particle_index, pos, energy_pin_targets, energy_pin_stiffness
    )
    f_total += f_pin
    h_total += h_pin'''
_PIN_SCALAR=_PIN_TILE.replace('f_total +=','f +=').replace('h_total +=','h +=')


def assert_stock_layout():
    """Fail if anything beyond the explicit energy substitutions drifts."""
    def source(function):
        return ast.parse(textwrap.dedent(inspect.getsource(getattr(function,'func',function))))
    def named(tree,name):
        return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    def dump(node):return ast.dump(node,include_attributes=False)
    expected_tile=named(source(stock.make_solve_elasticity_tile),'solve_elasticity_tile_kernel')
    candidate_tile=named(source(make_solve_elasticity_tile),'solve_elasticity_tile_kernel')
    expected_scalar=named(source(stock.solve_elasticity),'solve_elasticity')
    candidate_scalar=named(source(solve_elasticity),'solve_elasticity')
    expected_helper=named(source(stock.accumulate_elasticity_force_and_hessian),'accumulate_elasticity_force_and_hessian')
    candidate_helper=named(source(accumulate_elasticity_force_and_hessian),'accumulate_elasticity_force_and_hessian')
    replacements={
        'evaluate_own_membrane_force_hessian':('evaluate_neo_hookean_membrane_force_hessian',('energy_triangle_materials',)),
        'evaluate_own_bending_force_hessian':('evaluate_dihedral_angle_based_bending_force_hessian',('energy_edge_coefficients','energy_bending_stiffness')),
        'evaluate_own_solid_force_hessian':('evaluate_volumetric_neo_hookean_force_and_hessian',('energy_tet_gradients','energy_tet_measures')),
        'accumulate_elasticity_force_and_hessian':('accumulate_elasticity_force_and_hessian',('energy_triangle_materials','energy_edge_coefficients','energy_bending_stiffness','energy_tet_gradients','energy_tet_measures'))}
    class Reverse(ast.NodeTransformer):
        def __init__(self,pin):self.pin=dump(ast.parse(pin).body[0]) if pin else None;self.pins=0;self.calls={}
        def visit_If(self,node):
            if self.pin and dump(node)==self.pin:self.pins+=1;return None
            return self.generic_visit(node)
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id in replacements:
                name=node.func.id;original,extra=replacements[name]
                if extra:
                    actual=tuple(x.id if isinstance(x,ast.Name) else None for x in node.args[-len(extra):])
                    if actual!=extra:raise AssertionError('Changed appended energy arguments: '+name)
                    node.args=node.args[:-len(extra)]
                node.func.id=original;self.calls[name]=self.calls.get(name,0)+1
            return node
    receipts={}
    for label,expected,candidate,pin,extra in [
        ('tile',expected_tile,candidate_tile,_PIN_TILE,_ENERGY_ARGUMENTS),
        ('scalar',expected_scalar,candidate_scalar,_PIN_SCALAR,_ENERGY_ARGUMENTS),
        ('scalar_accumulate',expected_helper,candidate_helper,None,('energy_triangle_materials','energy_edge_coefficients','energy_bending_stiffness','energy_tet_gradients','energy_tet_measures'))]:
        node=copy.deepcopy(candidate)
        actual=tuple(a.arg for a in node.args.args if a.arg in _ENERGY_ARGUMENTS)
        if actual!=extra:raise AssertionError('Changed energy input signature: '+label)
        node.args.args=[a for a in node.args.args if a.arg not in extra]
        reverse=Reverse(pin);node=reverse.visit(node)
        if reverse.pins!=(1 if pin else 0):raise AssertionError('Changed finite pin energy block: '+label)
        if dump(node)!=dump(expected):raise AssertionError('Stock non-energy body/layout changed: '+label)
        receipts[label]={'unchanged_non_energy_ast':True,'energy_call_counts':reverse.calls,'pin_blocks':reverse.pins}
    return receipts


@functools.cache
def make_solve_elasticity_tile(include_triangles: bool, include_tets: bool, two_particles_per_warp: bool):
    """Build the tiled per-particle elasticity kernel, specialized at code generation.

    One kernel source serves every tiled variant: ``include_triangles`` / ``include_tets``
    statically strip the triangle+edge or tetrahedron adjacency loops (the stripped
    code is absent from the generated kernel, matching a hand-specialized one), and
    ``two_particles_per_warp`` packs two independent 16-lane particle solves into one
    32-thread warp, reducing with :func:`_warp_half_reduce_sum` to preserve the legacy
    16-lane reduction order bit for bit. Variants are memoized per flag combination.
    """

    # _warp_half_reduce_sum hardcodes a width-16 shuffle ladder.
    assert TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE == 16

    @wp.kernel(module="unique", enable_backward=False)
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
        energy_triangle_materials: wp.array2d[float],
        energy_edge_coefficients: wp.array[float],
        energy_bending_stiffness: wp.array[float],
        energy_pin_mask: wp.array[wp.int32],
        energy_pin_targets: wp.array[wp.vec3],
        energy_pin_stiffness: float,
        energy_tet_gradients: wp.array2d[wp.vec3],
        energy_tet_measures: wp.array[float],
        # output
        particle_displacements: wp.array[wp.vec3],
    ):
        tid = wp.tid()
        particle_slot = tid // TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        thread_idx = tid % TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE

        particle_slot_valid = True
        if wp.static(two_particles_per_warp):
            # Odd color sizes leave the second half-warp slot empty; it must still reach the
            # warp-wide reductions, so it cannot early-return.
            particle_slot_valid = particle_slot < particle_ids_in_color.shape[0]

        particle_index = wp.int32(0)
        particle_is_dynamic = False
        if particle_slot_valid:
            particle_index = particle_ids_in_color[particle_slot]
            particle_is_dynamic = (
                particle_flags[particle_index] & ParticleFlags.ACTIVE != 0 and mass[particle_index] != 0.0
            )

        if wp.static(not two_particles_per_warp):
            # Whole 16-lane block shares one particle: safe to retire inactive blocks early.
            if not particle_is_dynamic:
                if thread_idx == 0:
                    particle_displacements[particle_index] = wp.vec3(0.0)
                return

        f = wp.vec3(0.0)
        h = wp.mat33(0.0)

        if particle_is_dynamic:
            if wp.static(include_triangles):
                if tri_indices.shape[0] > 0:
                    batch_counter = wp.int32(0)
                    num_adj_faces = get_vertex_num_adjacent_faces(particle_adjacency, particle_index)
                    # loop through all the adjacent triangles using the whole 16-lane group
                    while batch_counter + thread_idx < num_adj_faces:
                        adj_tri_counter = thread_idx + batch_counter
                        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
                        tri_index, vertex_order = get_vertex_adjacent_face_id_order(
                            particle_adjacency, particle_index, adj_tri_counter
                        )

                        # fmt: off
                        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
                            wp.printf(
                                "particle: %d | num_adj_faces: %d | ",
                                particle_index,
                                get_vertex_num_adjacent_faces(particle_adjacency, particle_index),
                            )
                            wp.printf("i_face: %d | face id: %d | v_order: %d | ", adj_tri_counter, tri_index, vertex_order)
                            wp.printf(
                                "face: %d %d %d\n",
                                tri_indices[tri_index, 0],
                                tri_indices[tri_index, 1],
                                tri_indices[tri_index, 2],
                            )
                        # fmt: on

                        if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
                            f_tri, h_tri = evaluate_own_membrane_force_hessian(
                                tri_index,
                                vertex_order,
                                pos,
                                pos_prev,
                                tri_indices,
                                tri_poses[tri_index],
                                tri_areas[tri_index],
                                tri_materials[tri_index, 0],
                                tri_materials[tri_index, 1],
                                tri_materials[tri_index, 2],
                                dt,
                                energy_triangle_materials,
                            )

                            f += f_tri
                            h += h_tri

                if edge_indices.shape[0] > 0:
                    batch_counter = wp.int32(0)
                    num_adj_edges = get_vertex_num_adjacent_edges(particle_adjacency, particle_index)
                    while batch_counter + thread_idx < num_adj_edges:
                        adj_edge_counter = batch_counter + thread_idx
                        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
                        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
                            particle_adjacency, particle_index, adj_edge_counter
                        )
                        if edge_bending_properties[nei_edge_index, 0] > 0.0:
                            f_edge, h_edge = evaluate_own_bending_force_hessian(
                                nei_edge_index,
                                vertex_order_on_edge,
                                pos,
                                pos_prev,
                                edge_indices,
                                edge_rest_angles,
                                edge_rest_length,
                                edge_bending_properties[nei_edge_index, 0],
                                edge_bending_properties[nei_edge_index, 1],
                                dt,
                                energy_edge_coefficients,
                                energy_bending_stiffness,
                            )

                            f += f_edge
                            h += h_edge

            if wp.static(include_tets):
                if tet_indices.shape[0] > 0:
                    batch_counter = wp.int32(0)
                    num_adj_tets = get_vertex_num_adjacent_tets(particle_adjacency, particle_index)
                    while batch_counter + thread_idx < num_adj_tets:
                        adj_tet_counter = batch_counter + thread_idx
                        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
                        nei_tet_index, vertex_order_on_tet = get_vertex_adjacent_tet_id_order(
                            particle_adjacency, particle_index, adj_tet_counter
                        )
                        if tet_materials[nei_tet_index, 0] > 0.0 or tet_materials[nei_tet_index, 1] > 0.0:
                            f_tet, h_tet = evaluate_own_solid_force_hessian(
                                nei_tet_index,
                                vertex_order_on_tet,
                                pos_prev,
                                pos,
                                tet_indices,
                                tet_poses[nei_tet_index],
                                tet_materials[nei_tet_index, 0],
                                tet_materials[nei_tet_index, 1],
                                tet_materials[nei_tet_index, 2],
                                dt,
                                energy_tet_gradients,
                                energy_tet_measures,
                            )

                            f += f_tet
                            h += h_tet

        if wp.static(two_particles_per_warp):
            # The legacy 16-thread tile reduction uses the same 8,4,2,1 butterfly tree for lane
            # zero. Scalar shuffles retain that arithmetic order while allowing the two half-warps
            # to solve independent particles in one full warp.
            f_total = wp.vec3(
                _warp_half_reduce_sum(f[0]),
                _warp_half_reduce_sum(f[1]),
                _warp_half_reduce_sum(f[2]),
            )
            h_total = wp.mat33(
                _warp_half_reduce_sum(h[0, 0]),
                _warp_half_reduce_sum(h[0, 1]),
                _warp_half_reduce_sum(h[0, 2]),
                _warp_half_reduce_sum(h[1, 0]),
                _warp_half_reduce_sum(h[1, 1]),
                _warp_half_reduce_sum(h[1, 2]),
                _warp_half_reduce_sum(h[2, 0]),
                _warp_half_reduce_sum(h[2, 1]),
                _warp_half_reduce_sum(h[2, 2]),
            )
        else:
            f_tile = wp.tile(f, preserve_type=True)
            h_tile = wp.tile(h, preserve_type=True)

            f_total = wp.tile_reduce(wp.add, f_tile)[0]
            h_total = wp.tile_reduce(wp.add, h_tile)[0]

        if thread_idx == 0 and particle_slot_valid:
            if wp.static(two_particles_per_warp):
                if not particle_is_dynamic:
                    particle_displacements[particle_index] = wp.vec3(0.0)
                    return
            if energy_pin_mask[particle_index] != 0:
                f_pin, h_pin = evaluate_own_pin_force_hessian(
                    particle_index, pos, energy_pin_targets, energy_pin_stiffness
                )
                f_total += f_pin
                h_total += h_pin
            dt_sqr_reciprocal = 1.0 / (dt * dt)
            h_total = (
                h_total
                + mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)
                + particle_hessians[particle_index]
            )
            if abs(wp.determinant(h_total)) > 1e-8:
                h_inv = wp.inverse(h_total)
                f_total = (
                    f_total
                    + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * dt_sqr_reciprocal
                    + particle_forces[particle_index]
                )
                particle_displacements[particle_index] = particle_displacements[particle_index] + h_inv * f_total

    _configure_stock_execution(solve_elasticity_tile_kernel, include_triangles, include_tets, two_particles_per_warp)
    return solve_elasticity_tile_kernel



@wp.func
def accumulate_elasticity_force_and_hessian(
    particle_index: int,
    dt: float,
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
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
    f: wp.vec3,
    h: wp.mat33,
    energy_triangle_materials: wp.array2d[float],
    energy_edge_coefficients: wp.array[float],
    energy_bending_stiffness: wp.array[float],
    energy_tet_gradients: wp.array2d[wp.vec3],
    energy_tet_measures: wp.array[float],
):
    """Add the membrane, bending, and tet elasticity force and Hessian incident to a particle."""
    if tri_indices.shape[0] > 0:
        # elastic force and hessian
        for i_adj_tri in range(get_vertex_num_adjacent_faces(particle_adjacency, particle_index)):
            tri_index, vertex_order = get_vertex_adjacent_face_id_order(particle_adjacency, particle_index, i_adj_tri)

            # fmt: off
            if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
                wp.printf(
                    "particle: %d | num_adj_faces: %d | ",
                    particle_index,
                    get_vertex_num_adjacent_faces(particle_adjacency, particle_index),
                )
                wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_index, vertex_order)
                wp.printf(
                    "face: %d %d %d\n",
                    tri_indices[tri_index, 0],
                    tri_indices[tri_index, 1],
                    tri_indices[tri_index, 2],
                )
            # fmt: on

            if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
                f_tri, h_tri = evaluate_own_membrane_force_hessian(
                    tri_index,
                    vertex_order,
                    pos,
                    pos_prev,
                    tri_indices,
                    tri_poses[tri_index],
                    tri_areas[tri_index],
                    tri_materials[tri_index, 0],
                    tri_materials[tri_index, 1],
                    tri_materials[tri_index, 2],
                    dt,
                    energy_triangle_materials,
                )

                f = f + f_tri
                h = h + h_tri

    if edge_indices.shape[0] > 0:
        for i_adj_edge in range(get_vertex_num_adjacent_edges(particle_adjacency, particle_index)):
            nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
                particle_adjacency, particle_index, i_adj_edge
            )
            # vertex is on the edge; otherwise it only effects the bending energy n
            if edge_bending_properties[nei_edge_index, 0] > 0.0:
                f_edge, h_edge = evaluate_own_bending_force_hessian(
                    nei_edge_index,
                    vertex_order_on_edge,
                    pos,
                    pos_prev,
                    edge_indices,
                    edge_rest_angles,
                    edge_rest_length,
                    edge_bending_properties[nei_edge_index, 0],
                    edge_bending_properties[nei_edge_index, 1],
                    dt,
                    energy_edge_coefficients,
                    energy_bending_stiffness,
                )

                f = f + f_edge
                h = h + h_edge

    if tet_indices.shape[0] > 0:
        # solve tet elasticity
        num_adj_tets = get_vertex_num_adjacent_tets(particle_adjacency, particle_index)
        for adj_tet_counter in range(num_adj_tets):
            nei_tet_index, vertex_order_on_tet = get_vertex_adjacent_tet_id_order(
                particle_adjacency, particle_index, adj_tet_counter
            )
            if tet_materials[nei_tet_index, 0] > 0.0 or tet_materials[nei_tet_index, 1] > 0.0:
                f_tet, h_tet = evaluate_own_solid_force_hessian(
                    nei_tet_index,
                    vertex_order_on_tet,
                    pos_prev,
                    pos,
                    tet_indices,
                    tet_poses[nei_tet_index],
                    tet_materials[nei_tet_index, 0],
                    tet_materials[nei_tet_index, 1],
                    tet_materials[nei_tet_index, 2],
                    dt,
                    energy_tet_gradients,
                    energy_tet_measures,
                )

                f += f_tet
                h += h_tet

    return f, h


@wp.kernel
def solve_elasticity(
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
    energy_triangle_materials: wp.array2d[float],
    energy_edge_coefficients: wp.array[float],
    energy_bending_stiffness: wp.array[float],
    energy_pin_mask: wp.array[wp.int32],
    energy_pin_targets: wp.array[wp.vec3],
    energy_pin_stiffness: float,
    energy_tet_gradients: wp.array2d[wp.vec3],
    energy_tet_measures: wp.array[float],
    # output
    particle_displacements: wp.array[wp.vec3],
):
    t_id = wp.tid()

    particle_index = particle_ids_in_color[t_id]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE or mass[particle_index] == 0:
        particle_displacements[particle_index] = wp.vec3(0.0)
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # fmt: off
    if wp.static("inertia_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "particle: %d after accumulate inertia\nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )

    f, h = accumulate_elasticity_force_and_hessian(
        particle_index,
        dt,
        pos_prev,
        pos,
        tri_indices,
        tri_poses,
        tri_materials,
        tri_areas,
        edge_indices,
        edge_rest_angles,
        edge_rest_length,
        edge_bending_properties,
        tet_indices,
        tet_poses,
        tet_materials,
        particle_adjacency,
        f,
        h,
        energy_triangle_materials,
        energy_edge_coefficients,
        energy_bending_stiffness,
        energy_tet_gradients,
        energy_tet_measures,
    )

    # fmt: off
    if wp.static("overall_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "vertex: %d final\noverall force:\n %f %f %f, \noverall hessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )

    # fmt: on
    if energy_pin_mask[particle_index] != 0:
        f_pin, h_pin = evaluate_own_pin_force_hessian(
            particle_index, pos, energy_pin_targets, energy_pin_stiffness
        )
        f += f_pin
        h += h_pin
    h = h + particle_hessians[particle_index]
    f = f + particle_forces[particle_index]

    if abs(wp.determinant(h)) > 1e-8:
        h_inv = wp.inverse(h)
        particle_displacements[particle_index] = particle_displacements[particle_index] + h_inv * f


_scalar_stock_options=wp.get_module_options(stock.solve_elasticity.module)
energy.configure_module(__name__)
wp.set_module_options({key:_scalar_stock_options[key] for key in ('fast_math','fuse_fp')
    if key in _scalar_stock_options},module=__name__)
