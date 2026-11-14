"""Compact physical SDF accumulation with high-precision replay.

The diagnostic/rigid 42-double packet remains in :mod:`gpu.sdf`. This path
returns only float32 force, full local Hessian, status and selected collider.
Signed distances, union ordering and moving material transport retain float64.
"""
from functools import cache

import warp as wp

from . import sdf
from .warp_build import configure_module as _configure


Result = wp.types.vector(length=14, dtype=wp.float32)


def configure_module(module=None):
    return _configure(__name__ if module is None else module, ("sdf_fast.cuh",))


_EVALUATE_SOURCE = r"""
    wp::vec_t<14,float> packet;
    for(int k=0;k<14;++k)packet[k]=0.0f;
    packet[13]=-1.0f;
    ipc_gpu::sdf_fast::Input in;
    for(int k=0;k<3;++k){in.x[k]=position[k];in.previous[k]=previous[k];}
    in.stiffness=params.stiffness;in.epsilon=params.epsilon;in.dt=params.dt;
    in.friction=params.friction;in.eps_v=params.eps_v;in.include_curvature=params.include_curvature;
    in.prepared_poses=0;
    const int status=ipc_gpu::sdf_fast::evaluate(in,
        reinterpret_cast<const double*>(colliders.data),sizeof(colliders.data[0])/sizeof(double),
        colliders.shape[0],&packet[0]);
    if(status) {
        for(int k=0;k<12;++k)packet[k]=0.0f;
        packet[12]=static_cast<float>(status);
    }
    return packet;
"""


@wp.func_native(_EVALUATE_SOURCE)
def evaluate(position: wp.vec3, previous: wp.vec3, colliders: wp.array[sdf.Collider],
             params: sdf.Parameters) -> Result: ...


@wp.func_native(_EVALUATE_SOURCE.replace("in.prepared_poses=0;", "in.prepared_poses=1;"))
def _evaluate_prepared(position: wp.vec3, previous: wp.vec3, colliders: wp.array[sdf.Collider],
                       params: sdf.Parameters) -> Result: ...


@wp.struct
class Context:
    pending: wp.array[int]
    count: wp.array[int]
    error: wp.array[int]
    workers: int


@wp.func
def _finite_terms(force: wp.vec3, hessian: wp.mat33):
    valid = True
    for a in range(3):
        valid = valid and wp.isfinite(force[a])
        for b in range(3):
            valid = valid and wp.isfinite(hessian[a, b])
    return valid


@wp.func
def _terms(packet: Result):
    return (wp.vec3(packet[0], packet[1], packet[2]),
            wp.mat33(packet[3], packet[4], packet[5], packet[6], packet[7], packet[8],
                     packet[9], packet[10], packet[11]))


@wp.kernel(module="unique", enable_backward=False)
def accumulate(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3], flags: wp.array[int],
               colors: wp.array[int], color: int, solid: wp.array[int], surface: wp.array[int],
               colliders: wp.array[sdf.Collider], params: sdf.Parameters,
               forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33], ctx: Context):
    i = wp.tid()
    if ctx.error[0] == 0 and (flags[i] & 1) != 0 and colors[i] == color and (solid[i] == 0 or surface[i] != 0):
        # Checked path: preparation may have skipped material validation when
        # friction was zero and the caller enabled it later without prepare().
        packet = evaluate(positions[i], previous[i], colliders, params)
        status = int(packet[12])
        if status == 6:
            at = wp.atomic_add(ctx.count, 0, 1)
            if at < ctx.pending.shape[0]:
                ctx.pending[at] = i
            else:
                wp.atomic_max(ctx.error, 0, 49)
        elif status != 0:
            wp.atomic_max(ctx.error, 0, 49)
        else:
            force, hessian = _terms(packet)
            forces[i] += force
            hessians[i] += hessian


configure_module(accumulate)


@wp.kernel(module="unique", enable_backward=False)
def accumulate_prepared(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3], flags: wp.array[int],
                        colors: wp.array[int], color: int, solid: wp.array[int], surface: wp.array[int],
                        colliders: wp.array[sdf.Collider], params: sdf.Parameters,
                        forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33], ctx: Context):
    i = wp.tid()
    if ctx.error[0] == 0 and (flags[i] & 1) != 0 and colors[i] == color and (solid[i] == 0 or surface[i] != 0):
        # Selected only after prepare_colliders checked material poses. Its
        # preceding ordered launch latches error 48 before this guard runs.
        packet = _evaluate_prepared(positions[i], previous[i], colliders, params)
        status = int(packet[12])
        if status == 6:
            at = wp.atomic_add(ctx.count, 0, 1)
            if at < ctx.pending.shape[0]:
                ctx.pending[at] = i
            else:
                wp.atomic_max(ctx.error, 0, 49)
        elif status != 0:
            wp.atomic_max(ctx.error, 0, 49)
        else:
            force, hessian = _terms(packet)
            forces[i] += force
            hessians[i] += hessian


configure_module(accumulate_prepared)


@cache
def make_replay():
    @wp.kernel(module="unique", enable_backward=False)
    def replay(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3],
               colliders: wp.array[sdf.Collider], params: sdf.Parameters,
               forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33], ctx: Context):
        if ctx.error[0] == 0:
            for at in range(wp.tid(), wp.min(ctx.count[0], ctx.pending.shape[0]), ctx.workers):
                i = ctx.pending[at]
                packet = sdf.evaluate(positions[i], previous[i], colliders, params)
                if packet[41] != wp.float64(0.0):
                    wp.atomic_max(ctx.error, 0, 49)
                else:
                    force, hessian = sdf.vertex_terms(packet, wp.float32(params.dt))
                    if _finite_terms(force, hessian):
                        forces[i] += force
                        hessians[i] += hessian
                    else:
                        wp.atomic_max(ctx.error, 0, 49)
    sdf.configure_module(replay)
    return replay


configure_module()
