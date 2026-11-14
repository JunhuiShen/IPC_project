"""Our IPC/friction energy evaluated in Newton's unchanged stock contact rows.

Only the two material evaluator calls in the stock accumulation kernel are
substituted. Traversal, four-lane primitive ownership, color tests and atomics
are copied verbatim and guarded against the pinned source. The stock directed
EE rows remain directed: each invocation adds only the caller's two vertices.

Own k_barrier is an explicit coefficient (default 100), not a conversion of
Newton's soft_contact_ke. Stock contact normal damping is absent from our law.
Collision margin, friction, friction epsilon and dt retain the stock inputs.
Robust ambiguities enter an energy-only exact replay; no detection, DAT or CCD
initialization/acceptance policy is changed by this module.
"""
from functools import cache
import hashlib
import inspect
import linecache
import math
from pathlib import Path

import numpy as np
import warp as wp
from newton._src.solvers.vbd import particle_vbd_kernels as stock

from . import contact
from . import dat_contact_fast


STOCK_SOURCE_SHA256 = "d79a0cdaf257f5dfd20992afd25904f33421f924da661a57ec74d14463cf79f7"


@wp.struct
class EnergyContext:
    parameters: contact.Parameters
    error: wp.array[int]
    pending_nodes: wp.array[wp.vec4i]
    pending_kind: wp.array[int]
    pending_side: wp.array[int]
    pending_count: wp.array[int]
    exact_indices: wp.array[int]
    exact_count: wp.array[int]
    ordinary_workers: int
    words: wp.array[wp.uint32]
    slots: wp.array[int]
    workers: int


@wp.func
def _force(packet: contact.Result, role: int):
    return wp.vec3(packet[12 * role], packet[12 * role + 1], packet[12 * role + 2])


@wp.func
def _hessian(packet: contact.Result, role: int):
    return wp.mat33(packet[12*role+3], packet[12*role+4], packet[12*role+5],
                    packet[12*role+6], packet[12*role+7], packet[12*role+8],
                    packet[12*role+9], packet[12*role+10], packet[12*role+11])


@wp.func
def _evaluate(pos: wp.array[wp.vec3], previous: wp.array[wp.vec3], nodes: wp.vec4i,
              kind: int, side: int, collision_radius: float, friction: float,
              epsilon: float, dt: float, ctx: EnergyContext):
    result = contact.Result()
    if ctx.error[0] == 0:
        p = ctx.parameters
        # One kernel launch has common material/time inputs. Validate its
        # snapshot before queuing, so the exact pass cannot use stale values.
        if p.d_hat != wp.float64(collision_radius) or p.friction != wp.float64(friction) or p.eps_v != wp.float64(epsilon) or p.dt != wp.float64(dt):
            wp.atomic_max(ctx.error, 0, 101)
        else:
            result = contact.evaluate_self(pos, previous, nodes, kind, p)
            status = int(result[48])
            if status == 2 and p.exact_geometry != 0:
                at = wp.atomic_add(ctx.pending_count, 0, 1)
                if at < ctx.pending_nodes.shape[0]:
                    ctx.pending_nodes[at] = nodes
                    ctx.pending_kind[at] = kind
                    ctx.pending_side[at] = side
                else:
                    wp.atomic_max(ctx.error, 0, 12)
            elif status != 0:
                wp.atomic_max(ctx.error, 0, 100 + status)
    return result


@wp.func
def _evaluate_float32(pos: wp.array[wp.vec3], previous: wp.array[wp.vec3], nodes: wp.vec4i,
              kind: int, side: int, collision_radius: float, friction: float,
              epsilon: float, dt: float, ctx: EnergyContext):
    result = contact.Result()
    if ctx.error[0] == 0:
        p = ctx.parameters
        if p.d_hat != wp.float64(collision_radius) or p.friction != wp.float64(friction) or p.eps_v != wp.float64(epsilon) or p.dt != wp.float64(dt):
            wp.atomic_max(ctx.error, 0, 101)
        else:
            result = dat_contact_fast.evaluate_self(pos, previous, nodes, kind, p)
            status = int(result[48])
            if status == 6:
                at = wp.atomic_add(ctx.pending_count, 0, 1)
                if at < ctx.pending_nodes.shape[0]:
                    ctx.pending_nodes[at] = nodes
                    ctx.pending_kind[at] = kind
                    ctx.pending_side[at] = side
                else:
                    wp.atomic_max(ctx.error, 0, 12)
            elif status != 0:
                wp.atomic_max(ctx.error, 0, 100 + status)
    return result


@wp.func
def evaluate_edge_edge_contact_2_vertices_our_energy(
    e1: int, e2: int, pos: wp.array[wp.vec3], pos_anchor: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32], collision_radius: float,
    collision_stiffness: float, collision_damping: float,
    friction_coefficient: float, friction_epsilon: float, dt: float,
    edge_edge_parallel_epsilon: float, contact_damping_bidirectional: bool,
    energy_context: EnergyContext,
):
    force0 = wp.vec3(0.0); force1 = wp.vec3(0.0)
    hessian0 = wp.mat33(0.0); hessian1 = wp.mat33(0.0)
    active = False
    if e1 < 0 or e2 < 0 or e1 >= edge_indices.shape[0] or e2 >= edge_indices.shape[0]:
        wp.atomic_max(energy_context.error, 0, 101)
    else:
        first = wp.min(e1, e2); second = wp.max(e1, e2)
        nodes = wp.vec4i(edge_indices[first,2], edge_indices[first,3],
                        edge_indices[second,2], edge_indices[second,3])
        side = int(0)
        if e1 != first:
            side = 2
        packet = _evaluate(pos, pos_anchor, nodes, 1, side, collision_radius,
                           friction_coefficient, friction_epsilon, dt, energy_context)
        active = packet[48] == 0.0 and packet[49] != 0.0
        if active:
            force0 = _force(packet, side); force1 = _force(packet, side+1)
            hessian0 = _hessian(packet, side); hessian1 = _hessian(packet, side+1)
    return active, force0, force1, hessian0, hessian1


@wp.func
def evaluate_vertex_triangle_collision_force_hessian_4_vertices_our_energy(
    v: int, tri: int, pos: wp.array[wp.vec3], pos_anchor: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32], collision_radius: float,
    collision_stiffness: float, collision_damping: float,
    friction_coefficient: float, friction_epsilon: float, dt: float,
    contact_damping_bidirectional: bool, energy_context: EnergyContext,
):
    f0 = wp.vec3(0.0); f1 = wp.vec3(0.0); f2 = wp.vec3(0.0); f3 = wp.vec3(0.0)
    h0 = wp.mat33(0.0); h1 = wp.mat33(0.0); h2 = wp.mat33(0.0); h3 = wp.mat33(0.0)
    active = False
    if tri < 0 or tri >= tri_indices.shape[0]:
        wp.atomic_max(energy_context.error, 0, 101)
    else:
        # Our energy packet is (v,a,b,c); Newton's output tuple is (a,b,c,v).
        nodes = wp.vec4i(v, tri_indices[tri,0], tri_indices[tri,1], tri_indices[tri,2])
        packet = _evaluate(pos, pos_anchor, nodes, 0, -1, collision_radius,
                           friction_coefficient, friction_epsilon, dt, energy_context)
        active = packet[48] == 0.0 and packet[49] != 0.0
        if active:
            f0 = _force(packet,1); f1 = _force(packet,2); f2 = _force(packet,3); f3 = _force(packet,0)
            h0 = _hessian(packet,1); h1 = _hessian(packet,2); h2 = _hessian(packet,3); h3 = _hessian(packet,0)
    return active, f0, f1, f2, f3, h0, h1, h2, h3


# Separate Warp functions keep the large ordinary-double evaluator out of the
# float32 contact module's call graph. A runtime precision branch would retain
# its register/local-memory cost even when the branch is rarely taken.
def _float32_evaluator(function):
    name = function.func.__name__
    source = inspect.getsource(function.func)
    source = source.replace(name + "(", name + "_float32(", 1)
    source = source.replace("packet = _evaluate(", "packet = _evaluate_float32(")
    if name == "evaluate_edge_edge_contact_2_vertices_our_energy":
        # A runtime vector index makes Warp materialize four by-value copies
        # of the 50-float packet. Literal roles preserve every calculation and
        # overflow check while allowing the compiler to keep results in registers.
        dynamic = """        if active:
            force0 = _force(packet, side); force1 = _force(packet, side+1)
            hessian0 = _hessian(packet, side); hessian1 = _hessian(packet, side+1)
"""
        literal = """        if active:
            if side == 0:
                force0 = _force(packet, 0); force1 = _force(packet, 1)
                hessian0 = _hessian(packet, 0); hessian1 = _hessian(packet, 1)
            else:
                force0 = _force(packet, 2); force1 = _force(packet, 3)
                hessian0 = _hessian(packet, 2); hessian1 = _hessian(packet, 3)
"""
        if source.count(dynamic) != 1:
            raise RuntimeError("Unexpected DAT edge result extraction layout")
        source = source.replace(dynamic, literal, 1)
    filename = str(Path(__file__).resolve()) + "." + name + "_float32.py"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(globals())
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name + "_float32"]


_edge_float32 = _float32_evaluator(evaluate_edge_edge_contact_2_vertices_our_energy)
_triangle_float32 = _float32_evaluator(evaluate_vertex_triangle_collision_force_hessian_4_vertices_our_energy)


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
        raise ValueError("DAT contact precision must be float32 or float64")
    source, _ = cloned_source()
    filename = str(Path(__file__).resolve()) + ".stock_contact_energy.py"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(vars(stock))
    namespace.update(__name__=__name__, EnergyContext=EnergyContext,
        evaluate_edge_edge_contact_2_vertices_our_energy=_edge_float32 if precision == "float32" else evaluate_edge_edge_contact_2_vertices_our_energy,
        evaluate_vertex_triangle_collision_force_hessian_4_vertices_our_energy=_triangle_float32 if precision == "float32" else evaluate_vertex_triangle_collision_force_hessian_4_vertices_our_energy)
    exec(compile(source, filename, "exec"), namespace)
    kernel = namespace["accumulate_dat_contact_energy"]
    contact.configure_module(kernel)
    if precision == "float32":
        dat_contact_fast.configure_module(kernel)
    return kernel


@cache
def make_resolve_exact():
    from .contact_exact import evaluate_self_exact

    @wp.kernel(module="unique", enable_backward=False)
    def resolve_exact(pos: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                      colors: wp.array[int], color: int, ctx: EnergyContext,
                      forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
        if ctx.error[0] == 0:
            for at in range(wp.tid(), wp.min(ctx.pending_count[0], ctx.pending_nodes.shape[0]), ctx.workers):
                side = ctx.pending_side[at]
                kind = ctx.pending_kind[at]
                if (kind == 0 and side == -1) or (kind == 1 and (side == 0 or side == 2)):
                    nodes = ctx.pending_nodes[at]
                    packet = evaluate_self_exact(pos, previous, nodes, kind, ctx.parameters,
                                                 ctx.words, ctx.slots, wp.tid())
                    status = int(packet[48])
                    if status != 0:
                        wp.atomic_max(ctx.error, 0, 100 + status)
                    elif packet[49] != 0.0:
                        for role in range(4):
                            # Match the stock kernel: equality of color alone,
                            # including inactive particles in that color.
                            if (side == -1 or (role >= side and role < side+2)) and colors[nodes[role]] == color:
                                wp.atomic_add(forces, nodes[role], _force(packet,role))
                                wp.atomic_add(hessians, nodes[role], _hessian(packet,role))
                else:
                    wp.atomic_max(ctx.error, 0, 101)
    contact.configure_module(resolve_exact, include_filter=False)
    return resolve_exact


@cache
def make_resolve_precision():
    """Compile parallel ordinary replay separately from the rare exact path."""
    from .contact_exact import evaluate_self_exact

    @wp.kernel(module="unique", enable_backward=False)
    def resolve_ordinary(pos: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                      colors: wp.array[int], color: int, ctx: EnergyContext,
                      forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
        if ctx.error[0] == 0:
            for at in range(wp.tid(), wp.min(ctx.pending_count[0], ctx.pending_nodes.shape[0]), ctx.ordinary_workers):
                side = ctx.pending_side[at]
                kind = ctx.pending_kind[at]
                if (kind == 0 and side == -1) or (kind == 1 and (side == 0 or side == 2)):
                    nodes = ctx.pending_nodes[at]
                    packet = contact.evaluate_self(pos, previous, nodes, kind, ctx.parameters)
                    status = int(packet[48])
                    if status == 2 and ctx.parameters.exact_geometry != 0:
                        exact_at = wp.atomic_add(ctx.exact_count, 0, 1)
                        if exact_at < ctx.exact_indices.shape[0]:
                            ctx.exact_indices[exact_at] = at
                        else:
                            wp.atomic_max(ctx.error, 0, 12)
                    elif status != 0:
                        wp.atomic_max(ctx.error, 0, 100 + status)
                    elif packet[49] != 0.0:
                        for role in range(4):
                            if (side == -1 or (role >= side and role < side+2)) and colors[nodes[role]] == color:
                                wp.atomic_add(forces, nodes[role], _force(packet,role))
                                wp.atomic_add(hessians, nodes[role], _hessian(packet,role))
                else:
                    wp.atomic_max(ctx.error, 0, 101)
    contact.configure_module(resolve_ordinary)

    @wp.kernel(module="unique", enable_backward=False)
    def resolve_precision_exact(pos: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                      colors: wp.array[int], color: int, ctx: EnergyContext,
                      forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
        if ctx.error[0] == 0:
            for exact_at in range(wp.tid(), wp.min(ctx.exact_count[0], ctx.exact_indices.shape[0]), ctx.workers):
                at = ctx.exact_indices[exact_at]
                if at < 0 or at >= wp.min(ctx.pending_count[0], ctx.pending_nodes.shape[0]):
                    wp.atomic_max(ctx.error, 0, 101)
                else:
                    side = ctx.pending_side[at]
                    kind = ctx.pending_kind[at]
                    if (kind == 0 and side == -1) or (kind == 1 and (side == 0 or side == 2)):
                        nodes = ctx.pending_nodes[at]
                        packet = evaluate_self_exact(pos, previous, nodes, kind, ctx.parameters,
                                                     ctx.words, ctx.slots, wp.tid())
                        status = int(packet[48])
                        if status != 0:
                            wp.atomic_max(ctx.error, 0, 100 + status)
                        elif packet[49] != 0.0:
                            for role in range(4):
                                if (side == -1 or (role >= side and role < side+2)) and colors[nodes[role]] == color:
                                    wp.atomic_add(forces, nodes[role], _force(packet,role))
                                    wp.atomic_add(hessians, nodes[role], _hessian(packet,role))
                    else:
                        wp.atomic_max(ctx.error, 0, 101)
    contact.configure_module(resolve_precision_exact, include_filter=False)
    return resolve_ordinary, resolve_precision_exact


class EnergyWorkspace:
    """Persistent energy-only queue; use begin → stock clone → resolve per color.

    Kernel inputs are the original stock inputs, then ``context``, then the
    original force/H outputs. Use the original launch dimension/max_blocks.
    ``begin`` and ``resolve`` are capture-safe and never read device counts.
    """
    def __init__(self, collision_info, *, device=None, k_barrier=100.0,
                 exact_geometry=True, workers=None, error=None, precision="float64"):
        self.device = wp.get_device(device if device is not None else collision_info.vertex_colliding_triangles.device)
        if isinstance(k_barrier,bool) or not math.isfinite(k_barrier) or k_barrier < 0:
            raise ValueError("Own k_barrier must be finite and nonnegative")
        if not isinstance(exact_geometry,bool):
            raise TypeError("exact_geometry must be a boolean")
        if precision not in ("float32", "float64"):
            raise ValueError("DAT contact precision must be float32 or float64")
        self.precision = precision
        self.k_barrier, self.exact_geometry = float(k_barrier), exact_geometry
        capacity = collision_info.vertex_colliding_triangles.shape[0]//2 + collision_info.edge_colliding_edges.shape[0]//2
        if capacity < 0 or capacity >= 2**31:
            raise ValueError("Stock directed row capacity exceeds the energy queue index range")
        self.capacity = capacity
        ctx = EnergyContext()
        ctx.error = error if error is not None else wp.zeros(1,int,device=self.device)
        ctx.pending_nodes = wp.empty(capacity,wp.vec4i,device=self.device)
        ctx.pending_kind = wp.empty(capacity,int,device=self.device)
        ctx.pending_side = wp.empty(capacity,int,device=self.device)
        ctx.pending_count = wp.zeros(1,int,device=self.device)
        # The high-precision ordinary law is independent across rows. Only its
        # status-2 requests enter the small, arena-backed exact worker queue.
        ctx.exact_indices = wp.empty(capacity if precision == "float32" and exact_geometry else 0,int,device=self.device)
        ctx.exact_count = wp.zeros(1,int,device=self.device)
        ctx.ordinary_workers = self.device.sm_count*128 if self.device.is_cuda else 1
        worker_count = workers if workers is not None else (min(16,self.device.sm_count) if self.device.is_cuda else 1)
        if isinstance(worker_count,bool) or not isinstance(worker_count,int) or worker_count <= 0:
            raise ValueError("Exact energy workers must be positive")
        ctx.workers = worker_count
        ctx.words = wp.empty(ctx.workers*2048*512 if self.device.is_cuda and exact_geometry else 0,wp.uint32,device=self.device)
        ctx.slots = wp.empty(ctx.workers*512 if self.device.is_cuda and exact_geometry else 0,int,device=self.device)
        self.context = ctx
        self.kernel = make_accumulate_self_contact_force_and_hessian(precision)
        if precision == "float32":
            self._resolve, self._resolve_precision_exact = make_resolve_precision()
            if not exact_geometry:
                self._resolve_precision_exact = None
        else:
            self._resolve = make_resolve_exact() if exact_geometry else None
            self._resolve_precision_exact = None

    def begin(self, dt, collision_radius, friction_mu, friction_epsilon):
        # The stock kernel signature stores these inputs as float32; the own
        # F64 law consumes those exact represented values in both passes.
        self.context.parameters = contact.parameters(d_hat=float(np.float32(collision_radius)),
            k_barrier=self.k_barrier,dt=float(np.float32(dt)),friction=float(np.float32(friction_mu)),
            eps_v=float(np.float32(friction_epsilon)),exact_geometry=self.exact_geometry)
        self.context.pending_count.zero_()
        if self.precision == "float32":
            self.context.exact_count.zero_()
        return self.context

    def resolve(self, pos, previous, colors, color, forces, hessians):
        if self._resolve is not None and self.capacity:
            ordinary = self.precision == "float32"
            wp.launch(self._resolve,self.context.ordinary_workers if ordinary else self.context.workers,
                [pos,previous,colors,int(color),self.context,forces,hessians],
                device=self.device,block_dim=128 if ordinary else 1)
            if self._resolve_precision_exact is not None:
                wp.launch(self._resolve_precision_exact,self.context.workers,
                    [pos,previous,colors,int(color),self.context,forces,hessians],
                    device=self.device,block_dim=1)

    def check(self):
        code = int(self.context.error.numpy()[0])
        if code:
            raise RuntimeError(f"DAT contact energy failed with status {code}")


contact.configure_module(__name__)
