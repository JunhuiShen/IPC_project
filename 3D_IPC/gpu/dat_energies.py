# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Our material and IPC energies on native DAT execution and kernel layouts.

Reference closest points, directed contact ownership, friction and PSD local
solve matrices are reused with our corotated stress and scalar IPC barrier.
Source guards and reversible substitutions protect the pinned native bodies.
"""

import ast
import copy
from dataclasses import dataclass
import functools
from functools import cache
import hashlib
import inspect
import linecache
import math
from pathlib import Path
import textwrap

import numpy as np
import warp as wp
from newton._src.solvers.vbd import particle_vbd_kernels as stock
from . import contact, energy


# IPC contact laws and native row assembly.
@wp.func
def evaluate_ipc_force_norm(dis: float, radius: float, stiffness: float):
    """Derivatives of -k*(d-r)^2*log(d/r), for 0 < d < r."""
    if stiffness == 0.0:
        return 0.0, 0.0
    u = (radius - dis) / dis
    logarithm = wp.log(dis / radius)
    first = -stiffness * dis * u * (u - 2.0 * logarithm)
    second = stiffness * (u * (u + 4.0) - 2.0 * logarithm)
    return first, second


@wp.func
def evaluate_ipc_force_norm64(dis: float, radius: float, stiffness: float):
    if stiffness == 0.0:
        return 0.0, 0.0
    d = wp.float64(dis)
    r = wp.float64(radius)
    k = wp.float64(stiffness)
    u = (r - d) / d
    logarithm = wp.log(d / r)
    first = -k * d * u * (u - wp.float64(2.0) * logarithm)
    second = k * (u * (u + wp.float64(4.0)) - wp.float64(2.0) * logarithm)
    return float(first), float(second)


_STOCK = {
    'ee': ('evaluate_edge_edge_contact_2_vertices',
           'a4097124d8f9e52ad3da136def31a08f21e0cbac1e8e4c77e71555d2e6c89593'),
    'vt': ('evaluate_vertex_triangle_collision_force_hessian_4_vertices',
           'c77065982e01697e429e1ebc37cea6436d46a9467492e1f7a92a0cacee479752'),
}
_DAMPING_TARGETS = {'st_anchor', 'closest_p_anchor', '_bary_anchor',
                    '_feature_type_anchor', 'gap_rate', 'damping_force', 'damping_hessian'}


def _targets(node):
    return {item.id for target in node.targets for item in ast.walk(target)
            if isinstance(item, ast.Name)}


def _failed_tuple(kind):
    count = 2 if kind == 'ee' else 4
    return 'return False, ' + ', '.join(['zero_f'] * count + ['zero_h'] * count)


@cache
def native_contact_source(kind='vt', precision='float32'):
    """Return the callable clone source and its reversible substitution proof."""
    if kind not in _STOCK or precision not in ('float32', 'float64'):
        raise ValueError('Native DAT contact expects vt/ee and float32/float64')
    name, expected = _STOCK[kind]
    original = inspect.getsource(getattr(stock, name).func)
    digest = hashlib.sha256(original.encode()).hexdigest()
    if digest != expected:
        raise RuntimeError('Pinned native DAT contact evaluator changed: ' + name)
    tree = ast.parse(original)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    lines = original.splitlines(keepends=True)
    source = original
    substitutions = []

    def replace(before, after):
        nonlocal source
        if source.count(before) != 1 or not before or not after:
            raise RuntimeError('Unexpected native contact substitution layout: ' + name)
        source = source.replace(before, after, 1)
        substitutions.append((before, after))

    clone_name = '_native_' + kind + '_' + precision
    replace('def ' + name + '(\n', 'def ' + clone_name + '(\n')
    replace('    contact_damping_bidirectional: bool,\n',
            '    contact_damping_bidirectional: bool,\n    error: wp.array[int],\n')
    callback = 'evaluate_ipc_force_norm' if precision == 'float32' else 'evaluate_ipc_force_norm64'
    replace('evaluate_self_contact_force_norm(', callback + '(')

    removed = 0
    friction_edits = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and _targets(node) & _DAMPING_TARGETS:
            before = ''.join(lines[node.lineno-1:node.end_lineno])
            indent = ' ' * node.col_offset
            replace(before, indent + '# Native IPC adapter: omit normal damping (' + str(removed) + ').\n')
            removed += 1
        elif isinstance(node, ast.AugAssign) and isinstance(node.value, ast.BinOp):
            value = node.value
            if isinstance(value.op, ast.Add) and isinstance(value.left, ast.Name) and value.left.id in ('damping_force', 'damping_hessian'):
                before = ''.join(lines[node.lineno-1:node.end_lineno])
                replace(before, before.replace(value.left.id + ' + ', '', 1))
                friction_edits += 1
    expected_counts = (4, 4) if kind == 'ee' else (6, 8)
    if (removed, friction_edits) != expected_counts:
        raise RuntimeError('Unexpected native contact normal-damping sites: ' + name)

    failure = _failed_tuple(kind)
    distance_guard = ('    if not wp.isfinite(dis) or dis < 0.0:\n'
                      '        wp.atomic_max(error, 0, 101)\n'
                      '        zero_f = wp.vec3(0.0)\n'
                      '        zero_h = wp.mat33(0.0)\n'
                      '        ' + failure + '\n')
    replace('    collision_normal = diff / dis\n',
            distance_guard + '    collision_normal = diff / dis\n')
    derivative_guard = ('        if not wp.isfinite(dEdD) or not wp.isfinite(d2E_dDdD):\n'
                        '            wp.atomic_max(error, 0, 104)\n'
                        '            zero_f = wp.vec3(0.0)\n'
                        '            zero_h = wp.mat33(0.0)\n'
                        '            ' + failure + '\n')
    replace('        collision_force = -dEdD * collision_normal\n',
            derivative_guard + '        collision_force = -dEdD * collision_normal\n')

    restored = source
    for before, after in reversed(substitutions):
        if restored.count(after) != 1:
            raise RuntimeError('Native DAT contact substitution is not reversible')
        restored = restored.replace(after, before, 1)
    if restored != original:
        raise RuntimeError('Unapproved native DAT contact source change')
    return source, {'stock_source_sha256': digest,
                    'clone_sha256': hashlib.sha256(source.encode()).hexdigest(),
                    'restored_byte_identical': True, 'geometry_unchanged': True,
                    'normal_hessian': 'native rank-one', 'friction': 'native compute_friction',
                    'normal_damping_assignments_removed': removed,
                    'friction_assembly_edits': friction_edits,
                    'substitutions': substitutions, 'function_name': clone_name}


@cache
def make_evaluators(precision='float32'):
    result = []
    for kind in ('ee', 'vt'):
        source, proof = native_contact_source(kind, precision)
        filename = str(Path(__file__).resolve()) + '.' + kind + '.' + precision + '.py'
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        namespace = dict(vars(stock))
        namespace.update(__name__=__name__, evaluate_ipc_force_norm=evaluate_ipc_force_norm,
                         evaluate_ipc_force_norm64=evaluate_ipc_force_norm64)
        exec(compile(source, filename, 'exec'), namespace)
        result.append(namespace[proof['function_name']])
    return tuple(result)


def configure_module(module=None):
    target = __name__ if module is None else getattr(module, 'module', module)
    options = wp.get_module_options(stock.accumulate_self_contact_force_and_hessian.module)
    wp.set_module_options({'enable_backward': False,
                          **{key: options[key] for key in ('fast_math', 'fuse_fp') if key in options}},
                          module=target)


STOCK_SOURCE_SHA256 = "d79a0cdaf257f5dfd20992afd25904f33421f924da661a57ec74d14463cf79f7"


@wp.struct
class EnergyContext:
    parameters: contact.Parameters
    error: wp.array[int]


@wp.func
def _valid_context(radius: float, friction: float, epsilon: float, dt: float, ctx: EnergyContext):
    p = ctx.parameters
    valid = p.d_hat == wp.float64(radius) and p.friction == wp.float64(friction)
    valid = valid and p.eps_v == wp.float64(epsilon) and p.dt == wp.float64(dt)
    if not valid:
        wp.atomic_max(ctx.error, 0, 101)
    return valid and ctx.error[0] == 0


@wp.func
def _finite_block(force: wp.vec3, hessian: wp.mat33):
    # Evaluate every component without a short-circuit chain or runtime index.
    valid = (
        int(wp.isfinite(force[0]))
        & int(wp.isfinite(force[1]))
        & int(wp.isfinite(force[2]))
        & int(wp.isfinite(hessian[0, 0]))
        & int(wp.isfinite(hessian[0, 1]))
        & int(wp.isfinite(hessian[0, 2]))
        & int(wp.isfinite(hessian[1, 0]))
        & int(wp.isfinite(hessian[1, 1]))
        & int(wp.isfinite(hessian[1, 2]))
        & int(wp.isfinite(hessian[2, 0]))
        & int(wp.isfinite(hessian[2, 1]))
        & int(wp.isfinite(hessian[2, 2]))
    )
    return valid != 0


def evaluate_edge_edge_contact_2_vertices_our_energy(
    e1: int, e2: int, pos: wp.array[wp.vec3], pos_anchor: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32], collision_radius: float,
    collision_stiffness: float, collision_damping: float,
    friction_coefficient: float, friction_epsilon: float, dt: float,
    edge_edge_parallel_epsilon: float, contact_damping_bidirectional: bool,
    energy_context: EnergyContext,
):
    active = False
    f0 = wp.vec3(0.0); f1 = wp.vec3(0.0)
    h0 = wp.mat33(0.0); h1 = wp.mat33(0.0)
    if _valid_context(collision_radius, friction_coefficient, friction_epsilon, dt, energy_context):
        if e1 < 0 or e2 < 0 or e1 >= edge_indices.shape[0] or e2 >= edge_indices.shape[0]:
            wp.atomic_max(energy_context.error, 0, 101)
        else:
            active, f0, f1, h0, h1 = _native_edge(
                e1, e2, pos, pos_anchor, edge_indices, collision_radius,
                float(energy_context.parameters.k_barrier), 0.0,
                friction_coefficient, friction_epsilon, dt,
                edge_edge_parallel_epsilon, contact_damping_bidirectional, energy_context.error)
            if active and (not _finite_block(f0,h0) or not _finite_block(f1,h1)):
                wp.atomic_max(energy_context.error, 0, 104)
    if energy_context.error[0] != 0:
        return False, wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0)
    return active, f0, f1, h0, h1


def evaluate_vertex_triangle_collision_force_hessian_4_vertices_our_energy(
    v: int, tri: int, pos: wp.array[wp.vec3], pos_anchor: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32], collision_radius: float,
    collision_stiffness: float, collision_damping: float,
    friction_coefficient: float, friction_epsilon: float, dt: float,
    contact_damping_bidirectional: bool, energy_context: EnergyContext,
):
    active = False
    f0=wp.vec3(0.0); f1=wp.vec3(0.0); f2=wp.vec3(0.0); f3=wp.vec3(0.0)
    h0=wp.mat33(0.0); h1=wp.mat33(0.0); h2=wp.mat33(0.0); h3=wp.mat33(0.0)
    if _valid_context(collision_radius, friction_coefficient, friction_epsilon, dt, energy_context):
        if tri < 0 or tri >= tri_indices.shape[0]:
            wp.atomic_max(energy_context.error, 0, 101)
        else:
            active,f0,f1,f2,f3,h0,h1,h2,h3 = _native_triangle(
                v,tri,pos,pos_anchor,tri_indices,collision_radius,
                float(energy_context.parameters.k_barrier),0.0,
                friction_coefficient,friction_epsilon,dt,
                contact_damping_bidirectional,energy_context.error)
            if active and (not _finite_block(f0,h0) or not _finite_block(f1,h1) or not _finite_block(f2,h2) or not _finite_block(f3,h3)):
                wp.atomic_max(energy_context.error, 0, 104)
    if energy_context.error[0] != 0:
        return False,wp.vec3(0.0),wp.vec3(0.0),wp.vec3(0.0),wp.vec3(0.0),wp.mat33(0.0),wp.mat33(0.0),wp.mat33(0.0),wp.mat33(0.0)
    return active,f0,f1,f2,f3,h0,h1,h2,h3


@cache
def _energy_wrappers(precision):
    edge, triangle = make_evaluators(precision)
    namespace = dict(globals(), _native_edge=edge, _native_triangle=triangle)
    result = []
    for template in (evaluate_edge_edge_contact_2_vertices_our_energy,
                     evaluate_vertex_triangle_collision_force_hessian_4_vertices_our_energy):
        name = template.__name__ + '_' + precision
        source = '@wp.func\n' + inspect.getsource(template).replace('def '+template.__name__+'(', 'def '+name+'(', 1)
        filename = str(Path(__file__).resolve()) + '.' + name + '.py'
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        exec(compile(source, filename, 'exec'), namespace)
        result.append(namespace[name])
    return tuple(result)


def cloned_source():
    """Return the source clone and reversible proof of the allowed substitutions."""
    original = inspect.getsource(stock.accumulate_self_contact_force_and_hessian.func)
    actual = hashlib.sha256(original.encode()).hexdigest()
    if actual != STOCK_SOURCE_SHA256:
        raise RuntimeError("Stock contact accumulation source changed; review the clone before use")
    source = original
    substitutions = [
        ("@wp.kernel\n", '@wp.kernel(module="unique", enable_backward=False)\n'),
        ("def accumulate_self_contact_force_and_hessian(\n", "def accumulate_dat_contact_energy(\n"),
        ("    # outputs: particle force and hessian\n", "    energy_context: EnergyContext,\n    # outputs: particle force and hessian\n"),
        ("evaluate_edge_edge_contact_2_vertices(", "evaluate_edge_edge_contact_2_vertices_our_energy("),
        ("evaluate_vertex_triangle_collision_force_hessian_4_vertices(", "evaluate_vertex_triangle_collision_force_hessian_4_vertices_our_energy("),
        ("\n                            contact_damping_bidirectional,\n", "\n                            contact_damping_bidirectional,\n                            energy_context,\n"),
        ("\n                        contact_damping_bidirectional,\n", "\n                        contact_damping_bidirectional,\n                        energy_context,\n"),
    ]
    for before, after in substitutions:
        if source.count(before) != 1:
            raise RuntimeError("Unexpected stock energy call/signature layout")
        source = source.replace(before, after, 1)
    restored = source
    for before, after in reversed(substitutions):
        if restored.count(after) != 1:
            raise RuntimeError("Non-reversible stock energy substitution")
        restored = restored.replace(after, before, 1)
    if restored != original:
        raise RuntimeError("The stock contact traversal/atomic body was modified")
    return source, {"stock_source_sha256": actual, "clone_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "restored_byte_identical": True, "allowed_substitutions": substitutions}


@cache
def make_accumulate_self_contact_force_and_hessian(precision="float64"):
    if precision not in ("float32", "float64"):
        raise ValueError("DAT contact scalar precision must be float32 or float64")
    edge,triangle = _energy_wrappers(precision)
    source,_ = cloned_source()
    filename = str(Path(__file__).resolve()) + '.' + precision + '.stock_contact_energy.py'
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(vars(stock))
    namespace.update(__name__=__name__, EnergyContext=EnergyContext,
        evaluate_edge_edge_contact_2_vertices_our_energy=edge,
        evaluate_vertex_triangle_collision_force_hessian_4_vertices_our_energy=triangle)
    exec(compile(source,filename,'exec'),namespace)
    kernel = namespace['accumulate_dat_contact_energy']
    configure_module(kernel)
    return kernel


class EnergyWorkspace:
    """Native contact assembly; scalar precision never changes geometry."""
    def __init__(self,collision_info,*,device=None,k_barrier=100.0,
                 error=None,precision="float64"):
        self.device = wp.get_device(device if device is not None else collision_info.vertex_colliding_triangles.device)
        represented_k = np.float32(k_barrier)
        if isinstance(k_barrier,bool) or not math.isfinite(k_barrier) or k_barrier < 0 or not np.isfinite(represented_k) or (k_barrier>0 and represented_k==0):
            raise ValueError("Own k_barrier must be finite, nonnegative and representable in float32")
        if precision not in ("float32","float64"):raise ValueError("DAT contact scalar precision must be float32 or float64")
        self.precision = precision
        self.k_barrier = float(represented_k)
        self.context = EnergyContext()
        self.context.error = error if error is not None else wp.zeros(1,int,device=self.device)
        self.kernel = make_accumulate_self_contact_force_and_hessian(precision)
        configure_module(self.kernel)

    def begin(self,dt,collision_radius,friction_mu,friction_epsilon):
        self.context.parameters = contact.parameters(d_hat=float(np.float32(collision_radius)),
            k_barrier=self.k_barrier,dt=float(np.float32(dt)),friction=float(np.float32(friction_mu)),
            eps_v=float(np.float32(friction_epsilon)),exact_geometry=False)
        return self.context

    def check(self):
        code = int(self.context.error.numpy()[0])
        if code:raise RuntimeError(f"DAT contact energy failed with status {code}")


# Material laws and immutable additional model inputs.
_MEMBRANE_SOURCE_SHA256 = "0c4b307ecc14786fca13aba999ee10a3c3d77616be7b359d861961246687d1e8"
_MEMBRANE_FUNCTION_NAME = "evaluate_neo_hookean_membrane_force_hessian"
_MEMBRANE_CLONE_NAME = "evaluate_corotated_force_native_hessian"


def membrane_source():
    """Return native source with only reversible corotated stress substitutions."""
    original = inspect.getsource(getattr(stock, _MEMBRANE_FUNCTION_NAME).func)
    digest = hashlib.sha256(original.encode()).hexdigest()
    if digest != _MEMBRANE_SOURCE_SHA256:
        raise RuntimeError("Pinned native DAT membrane evaluator changed")
    substitutions = [
        ("def " + _MEMBRANE_FUNCTION_NAME + "(\n", "def " + _MEMBRANE_CLONE_NAME + "(\n"),
        ("    P_col0 = mu_nh * f0 + s * g0\n",
         "    stretch_trace = wp.sqrt(wp.length_sq(f0) + wp.length_sq(f1) + 2.0 * J_s)\n"
         "    R_col0 = (f0 + g0) / stretch_trace\n"
         "    P_col0 = 2.0 * mu * (f0 - R_col0) + lmbd * (J_s - 1.0) * g0\n"),
        ("    P_col1 = mu_nh * f1 + s * g1\n",
         "    R_col1 = (f1 + g1) / stretch_trace\n"
         "    P_col1 = 2.0 * mu * (f1 - R_col1) + lmbd * (J_s - 1.0) * g1\n"),
    ]
    source = original
    for before, after in substitutions:
        if source.count(before) != 1:
            raise RuntimeError("Unexpected native DAT membrane stress layout")
        source = source.replace(before, after, 1)
    restored = source
    for before, after in reversed(substitutions):
        if restored.count(after) != 1:
            raise RuntimeError("Native DAT membrane stress substitution is not reversible")
        restored = restored.replace(after, before, 1)
    if restored != original:
        raise RuntimeError("Unapproved native DAT membrane source change")
    return source, {"stock_source_sha256": digest,
                    "clone_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "restored_byte_identical": True, "geometry_unchanged": True,
                    "hessian_unchanged": True, "force": "corotated",
                    "hessian": "native PSD quasi-Newton approximation",
                    "singular_geometry": "native area-ratio floor",
                    "function_name": _MEMBRANE_CLONE_NAME, "substitutions": substitutions}


@cache
def make_membrane_evaluator():
    source, proof = membrane_source()
    filename = str(Path(__file__).resolve()) + ".membrane.py"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(vars(stock))
    namespace["__name__"] = __name__
    exec(compile(source, filename, "exec"), namespace)
    return namespace[proof["function_name"]]


evaluate_corotated_force_native_hessian = make_membrane_evaluator()


@wp.func
def evaluate_own_membrane_force_hessian(face:int,role:int,pos:wp.array[wp.vec3],pos_anchor:wp.array[wp.vec3],
    triangles:wp.array2d[int],inverse:wp.mat22,area:float,mu:float,lam:float,viscous_damping:float,dt:float,
    own_materials:wp.array2d[float]):
    own_mu=own_materials[face,0]
    own_lam=own_materials[face,1]
    if own_mu==0.0 and own_lam==0.0:
        return wp.vec3(0.0),wp.mat33(0.0)
    return evaluate_corotated_force_native_hessian(
        face,role,pos,pos_anchor,triangles,inverse,area,own_mu,own_lam,0.0,dt)


@wp.func
def evaluate_own_bending_force_hessian(edge:int,role:int,pos:wp.array[wp.vec3],pos_anchor:wp.array[wp.vec3],
    edges:wp.array2d[int],rest_angles:wp.array[float],rest_lengths:wp.array[float],stiffness:float,
    viscous_damping:float,dt:float,coefficients:wp.array[float],own_stiffness:wp.array[float]):
    if edges[edge,0]<0 or edges[edge,1]<0 or own_stiffness[edge]<=0.0 or coefficients[edge]<=0.0:
        return wp.vec3(0.0),wp.mat33(0.0)
    rest_length=rest_lengths[edge]
    if not wp.isfinite(rest_length) or rest_length<=0.0:
        return wp.vec3(0.0),wp.mat33(0.0)
    mapped_stiffness=((2.0*own_stiffness[edge])*coefficients[edge])/rest_length
    if not wp.isfinite(mapped_stiffness) or mapped_stiffness<=0.0:
        return wp.vec3(0.0),wp.mat33(0.0)
    # Native E=(ke*L0/2)*delta_theta^2; ours E=kB*c_e*delta_theta^2.
    # Delegate geometry, degeneracy guards and the PSD Hessian to native DAT.
    return stock.evaluate_dihedral_angle_based_bending_force_hessian(
        edge,role,pos,pos_anchor,edges,rest_angles,rest_lengths,mapped_stiffness,0.0,dt)


@wp.func
def evaluate_own_solid_force_hessian(tet:int,role:int,pos_anchor:wp.array[wp.vec3],pos:wp.array[wp.vec3],
    tets:wp.array2d[int],inverse:wp.mat33,mu:float,lam:float,viscous_damping:float,dt:float,
    gradients:wp.array2d[wp.vec3],measures:wp.array[float]):
    value=energy.solid(pos[tets[tet,0]],pos[tets[tet,1]],pos[tets[tet,2]],pos[tets[tet,3]],
        inverse,gradients[tet,role],measures[tet],mu,lam)
    return energy.unpack(value)


@wp.func
def evaluate_own_pin_force_hessian(particle:int,pos:wp.array[wp.vec3],targets:wp.array[wp.vec3],stiffness:float):
    return -stiffness*(pos[particle]-targets[particle]),stiffness*wp.identity(n=3,dtype=float)


@dataclass(frozen=True)
class MaterialData:
    triangle_materials: object
    edge_coefficients: object
    bending_stiffness: object
    pin_mask: object
    pin_targets: object
    pin_stiffness: float
    tet_gradients: object
    tet_measures: object

    def arguments(self):
        return [self.triangle_materials,self.edge_coefficients,self.bending_stiffness,self.pin_mask,self.pin_targets,float(self.pin_stiffness),
            self.tet_gradients,self.tet_measures]


def prepare(model,*,triangle_materials=None,edge_coefficients=None,bending_stiffness=None,pin_mask=None,pin_targets=None,pin_stiffness=None,
            tet_gradients=None,tet_measures=None):
    """Read model/rest metadata and own only the appended energy inputs.

    Existing IPC energy metadata is reused when present. A stock model has no
    finite pins by default; its inactive driven vertices stay hard constraints.
    Motion targets may be GPU-authored, so an existing target array is aliased.
    """
    device=model.device;meta=getattr(model,'ipc',None);n=model.particle_count
    def fallback(value,name):return getattr(meta,name,None) if value is None else value
    triangle_materials=fallback(triangle_materials,'dat_triangle_materials')
    edge_coefficients=fallback(edge_coefficients,'edge_coefficient')
    bending_stiffness=fallback(bending_stiffness,'dat_bending_stiffness')
    pin_mask=fallback(pin_mask,'pinned');pin_targets=fallback(pin_targets,'pin_targets')
    tet_gradients=fallback(tet_gradients,'tet_grad_N');tet_measures=fallback(tet_measures,'tet_measure')
    if pin_stiffness is None:pin_stiffness=float(getattr(meta,'params',{}).get('kpin',0.0))
    if not np.isfinite(pin_stiffness) or pin_stiffness<0:raise ValueError('Invalid finite-pin stiffness')
    def array(value,dtype,shape,name):
        if isinstance(value,wp.array):
            if value.dtype!=dtype or value.shape!=shape or value.device!=device:raise ValueError('Invalid '+name+' layout')
            return value
        values=np.asarray(value)
        if values.shape!=shape or not np.isfinite(values).all():raise ValueError('Invalid '+name)
        return wp.array(values,dtype=dtype,device=device)
    if edge_coefficients is None:
        x=model.particle_q.numpy().astype(np.float64)
        edges=model.edge_indices.numpy() if model.edge_count else np.empty((0,4),np.int32)
        values=np.zeros(model.edge_count,np.float32)
        for i,(a,b,e0,e1) in enumerate(edges):
            if a<0 or b<0:continue
            edge=x[e1]-x[e0]
            area=.5*(np.linalg.norm(np.cross(edge,x[a]-x[e0]))+np.linalg.norm(np.cross(edge,x[b]-x[e0])))
            if area>0:values[i]=np.dot(edge,edge)/area
        edge_coefficients=values
    if pin_mask is None:pin_mask=np.zeros(n,np.int32)
    if triangle_materials is None:
        triangle_materials=model.tri_materials.numpy()[:,:2].copy() if model.tri_count else np.empty((0,2),np.float32)
    if bending_stiffness is None:
        bending_stiffness=model.edge_bending_properties.numpy()[:,0].copy() if model.edge_count else np.empty(0,np.float32)
    if pin_targets is None:pin_targets=model.particle_q
    if tet_gradients is None or tet_measures is None:
        inverse=model.tet_poses.numpy().astype(np.float64) if model.tet_count else np.empty((0,3,3))
        gradients=np.empty((model.tet_count,4,3),np.float32);measure=np.empty(model.tet_count,np.float32)
        for i,dm in enumerate(inverse):
            gradients[i,1:]=dm;gradients[i,0]=-dm.sum(axis=0)
            determinant=np.linalg.det(dm)
            if not np.isfinite(determinant) or determinant==0:raise ValueError('Invalid stock tetrahedron rest matrix')
            measure[i]=1.0/(6.0*abs(determinant))
        if tet_gradients is None:tet_gradients=wp.array(gradients,dtype=wp.vec3,ndim=2,device=device)
        if tet_measures is None:tet_measures=measure
    # wp.vec3 array shapes omit the value's vector dimension.
    if not isinstance(pin_targets,wp.array):
        values=np.asarray(pin_targets)
        if values.shape!=(n,3) or not np.isfinite(values).all():raise ValueError('Invalid pin targets')
        pin_targets=wp.array(values,dtype=wp.vec3,device=device)
    if not isinstance(tet_gradients,wp.array):
        values=np.asarray(tet_gradients)
        if values.shape!=(model.tet_count,4,3) or not np.isfinite(values).all():raise ValueError('Invalid tetrahedron gradients')
        tet_gradients=wp.array(values,dtype=wp.vec3,ndim=2,device=device)
    coefficients_host=edge_coefficients.numpy() if isinstance(edge_coefficients,wp.array) else np.asarray(edge_coefficients)
    mask_host=pin_mask.numpy() if isinstance(pin_mask,wp.array) else np.asarray(pin_mask)
    stiffness_host=bending_stiffness.numpy() if isinstance(bending_stiffness,wp.array) else np.asarray(bending_stiffness)
    materials_host=triangle_materials.numpy() if isinstance(triangle_materials,wp.array) else np.asarray(triangle_materials)
    if not np.isfinite(coefficients_host).all() or np.any(coefficients_host<0):raise ValueError('Invalid bending coefficients')
    if not np.isin(mask_host,[0,1]).all():raise ValueError('Invalid finite-pin mask')
    if not np.isfinite(stiffness_host).all() or np.any(stiffness_host<0):raise ValueError('Invalid bending stiffness')
    if not np.isfinite(materials_host).all():raise ValueError('Invalid triangle materials')
    return MaterialData(array(triangle_materials,wp.float32,(model.tri_count,2),'triangle materials'),
        array(edge_coefficients,wp.float32,(model.edge_count,),'edge coefficients'),
        array(bending_stiffness,wp.float32,(model.edge_count,),'bending stiffness'),
        array(pin_mask,wp.int32,(n,),'pin mask'),array(pin_targets,wp.vec3,(n,),'pin targets'),float(pin_stiffness),
        array(tet_gradients,wp.vec3,(model.tet_count,4),'tet gradients'),
        array(tet_measures,wp.float32,(model.tet_count,),'tet measures'))


# Guarded native tiled and scalar material solve kernels.
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

# The shared scalar/material module needs the project arithmetic preamble.
# Restore native execution flags after that preamble's strict defaults.
# Unique tiled material kernels retain their own _configure_stock_execution().
energy.configure_module(__name__)
configure_module()
