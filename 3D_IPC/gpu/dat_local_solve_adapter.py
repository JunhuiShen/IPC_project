"""Opt-in FP32 CPU-style local solve; leave energy/scheduling arithmetic intact."""
import ast
import copy
from functools import cache
import hashlib
import inspect
import linecache
from pathlib import Path
import textwrap

import warp as wp
from . import dat_energies as energies
from . import dat_local_solve


_TILE_OLD = '''if abs(wp.determinant(h_total)) > 1e-8:
    h_inv = wp.inverse(h_total)
    f_total = (
        f_total
        + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * dt_sqr_reciprocal
        + particle_forces[particle_index]
    )
    particle_displacements[particle_index] = particle_displacements[particle_index] + h_inv * f_total'''
_TILE_NEW = '''f_total = (
    f_total
    + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * dt_sqr_reciprocal
    + particle_forces[particle_index]
)
particle_displacements[particle_index] = particle_displacements[particle_index] + cpu_local_solve(h_total, f_total, LOCAL_DAMPING)'''
_SCALAR_OLD = '''if abs(wp.determinant(h)) > 1e-8:
    h_inv = wp.inverse(h)
    particle_displacements[particle_index] = particle_displacements[particle_index] + h_inv * f'''
_SCALAR_NEW = '''particle_displacements[particle_index] = particle_displacements[particle_index] + cpu_local_solve(h, f, LOCAL_DAMPING)'''


@cache
def source(tiled, scheduled, damping):
    from .dat_color_schedule import schedule_source, _substitute_statements
    if scheduled:
        original = schedule_source(tiled)[0]
    else:
        function = energies.make_solve_elasticity_tile if tiled else energies.solve_elasticity.func
        original = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(original)
    before, after = (_TILE_OLD, _TILE_NEW) if tiled else (_SCALAR_OLD, _SCALAR_NEW)
    after = after.replace('LOCAL_DAMPING', repr(float(damping)))
    _substitute_statements(tree, before, after)
    restored = copy.deepcopy(tree)
    _substitute_statements(restored, after, before)
    if ast.dump(restored, include_attributes=False) != ast.dump(ast.parse(original), include_attributes=False):
        raise RuntimeError('Local solve adaptation changed unrelated arithmetic')
    # Separate scalar variants have separate compiler modules; no global
    # native kernel or upstream solver is patched.
    if not tiled:
        tree.body[0].decorator_list = ast.parse('@wp.kernel(module="unique", enable_backward=False)\ndef f(): pass').body[0].decorator_list
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    return result, {'restored_ast_identical': True,
                    'original_source_sha256': hashlib.sha256(original.encode()).hexdigest(),
                    'adapted_source_sha256': hashlib.sha256(result.encode()).hexdigest()}


@cache
def definition(tiled, scheduled, damping):
    code, _ = source(tiled, scheduled, damping)
    filename = str(Path(__file__).resolve()) + f'.{int(tiled)}.{int(scheduled)}.{damping}.py'
    linecache.cache[filename] = (len(code), None, code.splitlines(keepends=True), filename)
    namespace = dict(vars(energies))
    namespace.update(__name__=__name__, cpu_local_solve=dat_local_solve.solve)
    exec(compile(code, filename, 'exec'), namespace)
    return namespace['make_solve_elasticity_tile' if tiled else 'solve_elasticity']


def configure(kernel, reference):
    options = wp.get_module_options(reference.module)
    if kernel.module is not reference.module:
        wp.set_module_options(options, module=kernel.module)
    dat_local_solve.configure_module(kernel)
    # Preserve existing material evaluation/reduction compiler policy.
    wp.set_module_options({k: options[k] for k in ('fast_math', 'fuse_fp') if k in options}, module=kernel.module)


def scalar(damping, *, scheduled=False):
    kernel = definition(False, scheduled, damping)
    configure(kernel, energies.solve_elasticity)
    return kernel


def tiled(triangles, tets, two, damping, *, scheduled=False):
    kernel = definition(True, scheduled, damping)(triangles, tets, two)
    configure(kernel, energies.make_solve_elasticity_tile(triangles, tets, two))
    return kernel
