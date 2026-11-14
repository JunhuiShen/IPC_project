"""Own IPC normal derivatives with the original Warp friction evaluator.

This DAT-only entry point leaves the fast evaluator shared with GS untouched.
A guarded clone emits normal derivatives and friction inputs from one
feature evaluation. The existing precision/exact replays remain the fallback.
"""

import ast
from pathlib import Path

import warp as wp

from . import contact, dat_contact_fast
from .dat_friction import evaluate_reference_friction_checked
from .warp_build import configure_module as _configure


NormalPacket = wp.types.vector(length=62, dtype=wp.float32)


def _sources():
    root = Path(__file__).resolve().parent
    header = (root / "dat_contact_fast.cuh").read_text()
    start = header.index("IPC_DAT_FAST_HD inline int evaluate(const Input& in,float* packet) {")
    end_marker = "\n}\n\n}} // namespace ipc_gpu::dat_contact_fast"
    if header.count(end_marker) != 1:
        raise RuntimeError("DAT fast evaluator is no longer the final header function")
    original = header[start:header.index(end_marker)+2]
    signature = "inline int evaluate(const Input& in,float* packet) {"
    marker = "    // Exact expansion of the minimized-distance Schur complement."
    extra = """    // Retain the original scalar guards before using native friction.
    for(int k=0;k<3;++k) {
        packet[50+k]=feature.normal[k];
        packet[58+k]=displacement[k];
    }
    for(int role=0;role<4;++role)packet[53+role]=feature.w[role];
    packet[57]=-in.stiffness*bp;
    packet[61]=in.dt*in.eps_v;
    // The native Warp routine adds friction after these normal derivatives.
    friction_scale=0.0f;
"""
    changes = [
        (signature, signature.replace("evaluate(", "evaluate_reference_components(")),
        (marker, extra + marker),
    ]
    transformed = original
    for before, after in changes:
        if transformed.count(before) != 1:
            raise RuntimeError("DAT contact layout changed; review reference-friction bridge")
        transformed = transformed.replace(before, after, 1)
    restored = transformed
    for before, after in reversed(changes):
        restored = restored.replace(after, before, 1)
    if restored != original:
        raise RuntimeError("Non-friction normal derivative change detected")

    tree = ast.parse((root / "dat_contact_fast.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_self")
    decorator = next(n for n in function.decorator_list if isinstance(n, ast.Call))
    code = ast.literal_eval(decorator.args[0])
    for before, after in [
        ("wp::vec_t<50,float> packed;", "wp::vec_t<62,float> packed;"),
        ("k<50;", "k<62;"),
        ("status=evaluate(in,&packed[0]);", "status=evaluate_reference_components(in,&packed[0]);"),
    ]:
        if code.count(before) != 1:
            raise RuntimeError("DAT fast input validation changed; review bridge")
        code = code.replace(before, after, 1)
    transformed = transformed.replace("IPC_DAT_FAST_HD inline", "IPC_DAT_REFERENCE_HD inline", 1)
    preamble = """#ifndef IPC_DAT_REFERENCE_COMPONENTS
#define IPC_DAT_REFERENCE_COMPONENTS
#if defined(__CUDACC__)
#define IPC_DAT_REFERENCE_HD __host__ __device__
#else
#define IPC_DAT_REFERENCE_HD
#endif
namespace ipc_gpu { namespace dat_contact_fast {
""" + transformed + """
}}
#undef IPC_DAT_REFERENCE_HD
#endif
"""
    return code, preamble


_CODE, _PREAMBLE = _sources()


def configure_module(module=None):
    return _configure(__name__ if module is None else module,
                      ("dat_contact_fast.cuh",), extra_preamble=_PREAMBLE)


@wp.func_native(_CODE)
def _normal_components(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                       nodes: wp.vec4i, kind: int, params: contact.Parameters) -> NormalPacket: ...


@wp.func
def evaluate_self(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                  nodes: wp.vec4i, kind: int, params: contact.Parameters) -> contact.Result:
    components = _normal_components(positions, previous, nodes, kind, params)
    result = contact.Result()
    for i in range(50):
        result[i] = components[i]
    if result[48] == 0.0 and result[49] != 0.0 and params.friction > 0.0:
        normal = wp.vec3(components[50], components[51], components[52])
        motion = wp.vec3(components[58], components[59], components[60])
        eps_u = components[61]
        force, hessian, needs_replay = evaluate_reference_friction_checked(
            normal, motion, components[57], float(params.friction), eps_u)
        if needs_replay != 0:
            result[48] = 6.0
            result[49] = 0.0
        else:
            # All roles are checked, even when a particular color uses only
            # one. A failed native intermediate replays the unchanged F64 law.
            for role in range(4):
                weight = components[53+role]
                for k in range(3):
                    result[12*role+k] += weight*force[k]
                    for l in range(3):
                        result[12*role+3+3*k+l] += (weight*hessian[k,l])*weight
            for i in range(48):
                if not wp.isfinite(result[i]):
                    result[48] = 6.0
                    result[49] = 0.0
    if result[48] != 0.0:
        for i in range(48):
            result[i] = 0.0
    return result


configure_module()
