"""Device color scheduling for the reference DAT material kernels.

Only the color-slot lookup is adapted.  Each active particle executes the
unchanged ``dat_energies`` numerical body, including its native local inverse
and, on CUDA, the native 16-lane reduction. The optional ColorCcd sidecar replaces
publication after each color; the predictor remains owned by ``SolverDAT``.
"""

import ast
import copy
from functools import cache
import hashlib
import inspect
import linecache
from pathlib import Path
import textwrap

import warp as wp
from newton._src.solvers.vbd.solver_vbd import _is_tet_only_elasticity_model

from . import dat_energies as energies


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _substitute_statements(tree, before, after):
    """Replace exactly one statement sequence, including nested blocks."""
    before = ast.parse(textwrap.dedent(before)).body
    after = ast.parse(textwrap.dedent(after)).body
    expected = [_dump(node) for node in before]
    count = 0

    def visit(node):
        nonlocal count
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                result = []
                index = 0
                while index < len(value):
                    items = value[index:index + len(before)]
                    if (len(items) == len(before)
                            and all(isinstance(item, ast.AST) for item in items)
                            and [_dump(item) for item in items] == expected):
                        result.extend(copy.deepcopy(after))
                        index += len(before)
                        count += 1
                    else:
                        item = value[index]
                        if isinstance(item, ast.AST):
                            visit(item)
                        result.append(item)
                        index += 1
                setattr(node, field, result)
            elif isinstance(value, ast.AST):
                visit(value)

    visit(tree)
    if count != 1:
        raise RuntimeError(f"Unexpected material schedule substitution count: {count}")


_TILE_EDITS = (
    ("""particle_slot_valid = True
if wp.static(two_particles_per_warp):
    particle_slot_valid = particle_slot < particle_ids_in_color.shape[0]
""", """particle_slot_valid = particle_slot < (
    schedule_offsets[schedule_current_color[0] + 1]
    - schedule_offsets[schedule_current_color[0]])
"""),
    ("particle_index = particle_ids_in_color[particle_slot]",
     "particle_index = particle_ids_in_color[schedule_offsets[schedule_current_color[0]] + particle_slot]"),
    ("""if thread_idx == 0:
    particle_displacements[particle_index] = wp.vec3(0.0)
""", """if thread_idx == 0 and particle_slot_valid:
    particle_displacements[particle_index] = wp.vec3(0.0)
"""),
)
_SCALAR_EDITS = (
    ("""t_id = wp.tid()
particle_index = particle_ids_in_color[t_id]
""", """t_id = wp.tid()
if t_id >= schedule_offsets[schedule_current_color[0] + 1] - schedule_offsets[schedule_current_color[0]]:
    return
particle_index = particle_ids_in_color[schedule_offsets[schedule_current_color[0]] + t_id]
"""),
)


@cache
def schedule_source(tiled):
    """Return the generated source and a reversible, non-numerical edit proof."""
    function = energies.make_solve_elasticity_tile if tiled else energies.solve_elasticity.func
    original_source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(original_source)
    original_tree = copy.deepcopy(tree)
    name = "solve_elasticity_tile_kernel" if tiled else "solve_elasticity"
    kernel = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    if kernel.args.args[1].arg != "particle_ids_in_color":
        raise RuntimeError("Material kernel color argument changed")
    extra_args = ast.parse("def schedule(schedule_offsets: wp.array[wp.int32], "
                           "schedule_current_color: wp.array[wp.int32]): pass").body[0].args.args
    kernel.args.args[2:2] = extra_args
    edits = _TILE_EDITS if tiled else _SCALAR_EDITS
    for before, after in edits:
        _substitute_statements(kernel, before, after)

    # Reverse every permitted scheduling edit and compare the complete AST,
    # including the factory's execution options and each arithmetic expression.
    restored = copy.deepcopy(tree)
    restored_kernel = next(node for node in ast.walk(restored)
                           if isinstance(node, ast.FunctionDef) and node.name == name)
    del restored_kernel.args.args[2:4]
    for before, after in reversed(edits):
        _substitute_statements(restored_kernel, after, before)
    if _dump(restored) != _dump(original_tree):
        raise RuntimeError("Dynamic coloring changed the material numerical body")
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    proof = {
        "source_sha256": hashlib.sha256(original_source.encode()).hexdigest(),
        "scheduled_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "unchanged_numerical_ast": True,
        "restored_ast_identical": True,
        "scheduling_substitutions": len(edits),
    }
    return source, proof


@cache
def _scheduled_definition(tiled):
    source, _ = schedule_source(tiled)
    filename = str(Path(__file__).resolve()) + (".tile.py" if tiled else ".scalar.py")
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(vars(energies))
    namespace["__name__"] = __name__
    exec(compile(source, filename, "exec"), namespace)
    result = namespace["make_solve_elasticity_tile" if tiled else "solve_elasticity"]
    if not tiled:
        # Keep the project's arithmetic preamble and the original scalar
        # compiler policy, including the native fast-math and FMA settings.
        options = wp.get_module_options(energies.solve_elasticity.module)
        wp.set_module_options(options, module=result.module)
    return result


def prepare(solver):
    """Select the schedule kernel and reject omitted native terms at setup."""
    model = solver.model
    if solver.enable_planar_dat or solver.particle_enable_self_contact or solver.rigid_soft_enable_dat:
        raise ValueError("Dynamic barrier coloring requires all collision processing and DAT disabled")
    if model.spring_count:
        raise ValueError("Dynamic barrier coloring does not yet schedule native spring energies")
    if getattr(model, "attachment_particle_body_count", 0):
        raise ValueError("Dynamic barrier coloring does not yet schedule particle-body attachments")
    energies.assert_stock_layout()
    tiled = solver.use_particle_tile_solve
    if tiled:
        include_tets = model.tet_count > 0
        # This is the same specialization used by reference SolverVBD.
        include_triangles = not _is_tet_only_elasticity_model(model)
        two = not include_tets
        solver._dynamic_elasticity_kernel = _scheduled_definition(True)(include_triangles, include_tets, two)
        if solver.local_solve=='cpu_fp32':
            from . import dat_local_solve_adapter as local_adapter
            solver._dynamic_elasticity_kernel=local_adapter.tiled(include_triangles,include_tets,two,
                solver.local_solve_damping,scheduled=True)
        per_block = 2 if two else 1
        solver._dynamic_elasticity_block_dim = per_block * energies.TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        blocks = (model.particle_count + per_block - 1) // per_block
        solver._dynamic_elasticity_dim = blocks * solver._dynamic_elasticity_block_dim
    else:
        solver._dynamic_elasticity_kernel = _scheduled_definition(False)
        if solver.local_solve=='cpu_fp32':
            from . import dat_local_solve_adapter as local_adapter
            solver._dynamic_elasticity_kernel=local_adapter.scalar(solver.local_solve_damping,scheduled=True)
        solver._dynamic_elasticity_block_dim = None
        solver._dynamic_elasticity_dim = model.particle_count
    # Generated kernels are cached across solver instances. Register each
    # instance's matching material policy so native step() restores it when
    # another solver has changed the shared module's compilation options.
    source_kernel = solver._tiled_elasticity_kernel if tiled else solver._own_scalar_elasticity_kernel
    options = wp.get_module_options(source_kernel.module)
    solver._set_module_options(
        {key: options[key] for key in ("fast_math", "fuse_fp") if key in options},
        module=solver._dynamic_elasticity_kernel.module,
    )
    solver._dynamic_color_proof = dict(schedule_source(tiled)[1])
    if solver.local_solve=='cpu_fp32':
        solver._dynamic_color_proof['local_solve_substitution']=local_adapter.source(tiled,True,solver.local_solve_damping)[1]
    return solver._dynamic_color_proof


def solve_iteration(solver, state_in, state_out, dt):
    """Execute a native material sweep over the current device color schedule."""
    model = solver.model
    if model.particle_count == 0:
        return
    graph = solver.barrier_coloring
    solver.particle_forces.zero_()
    solver.particle_hessians.zero_()
    inputs = [
        dt, graph.order, graph.offsets, graph.current_color,
        solver.particle_q_prev, state_in.particle_q, model.particle_mass,
        solver.inertia, model.particle_flags, model.tri_indices, model.tri_poses,
        model.tri_materials, model.tri_areas, model.edge_indices,
        model.edge_rest_angle, model.edge_rest_length, model.edge_bending_properties,
        model.tet_indices, model.tet_poses, model.tet_materials,
        solver.particle_adjacency, solver.particle_forces, solver.particle_hessians,
        *solver._energy_materials.arguments(),
    ]

    def body():
        if solver.color_contact_forces is not None:
            solver.color_contact_forces.assemble(
                state_in.particle_q,solver.particle_q_prev,dt,
                solver._energy_dhat,solver._energy_friction,solver._energy_eps_v,
                solver.particle_forces,solver.particle_hessians)
        launch_options = {}
        if solver._dynamic_elasticity_block_dim is not None:
            launch_options["block_dim"] = solver._dynamic_elasticity_block_dim
        wp.launch(solver._dynamic_elasticity_kernel, dim=solver._dynamic_elasticity_dim,
                  inputs=inputs, outputs=[solver.particle_displacements],
                  device=solver.device, **launch_options)
        if solver.color_ccd is not None:
            solver.color_ccd.apply(state_in.particle_q, solver.pos_prev_collision_detection,
                                   solver.particle_displacements)
        else:
            solver._penetration_free_truncation(state_in, None)

    graph.sweep(body)
    wp.copy(state_out.particle_q, state_in.particle_q)
