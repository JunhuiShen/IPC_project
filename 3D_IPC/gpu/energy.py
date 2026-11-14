"""Our material equations specialized for Newton's float32 GPU state.

The fast specialization changes scalar storage/math precision, not the energy
model: membrane and tetrahedral elasticity remain corotated, and bending is
the native dihedral energy with its native PSD Hessian approximation. Original
binary64 arithmetic remains compiled in a separate namespace for references.
"""

import re

import warp as wp

from . import warp_build as build


HEADERS = ("kernels.h", "solid_types.h", "cloth_math.cuh", "solid_math.cuh", "solve.cuh")
Pack = wp.types.vector(length=12, dtype=wp.float32)


def _float32_source():
    source, dependencies = build.flatten_headers(HEADERS)
    source = source[len(build._CRT):]
    # Unsuffixed C++ decimal literals otherwise promote float operations to
    # double. Integer literals and all original integer/control flow stay intact.
    literal = re.compile(r"(?<![\w.])(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])|(?<![\w.])\d+[eE][+-]?\d+(?![\w.])")
    protected = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',re.DOTALL)

    def specialize(code):
        code = code.replace("IPC_WARP_HEADER_", "IPC_WARP_F32_HEADER_")
        code = re.sub(r"\bipc_gpu\b", "ipc_gpu_f32", code)
        code = re.sub(r"\bdouble\b", "float", code)
        # Adapt machine-precision stopping criteria in the QR/polar routines.
        code = code.replace("4503599627370496.0", "8388608.0")
        code = code.replace("2.2250738585072014e-308", "1.1754943508222875e-38")
        code = code.replace("2.2737367544323206e-13", "1.0e-6")
        code = code.replace("DBL_EPSILON", "FLT_EPSILON").replace("DBL_MIN", "FLT_MIN")
        for name in ("fma", "sqrt", "fabs", "hypot", "copysign", "atan2", "pow", "sin", "cos", "fmax", "fmin"):
            code = re.sub(r"::" + name + r"\s*\(", "::" + name + "f(", code)
            # The local QR solve uses unqualified C math calls. Bind these
            # explicitly as well; Warp's SDK-free CPU CRT has double overloads.
            code = re.sub(r"(?<![\w:])" + name + r"\s*\(", name + "f(", code)
        return literal.sub(lambda match: match.group(0) + "f", code)

    chunks=[];at=0
    for match in protected.finditer(source):
        chunks.append(specialize(source[at:match.start()]))
        chunks.append(match.group(0))  # Retain licenses, comments and strings verbatim.
        at=match.end()
    chunks.append(specialize(source[at:]))
    source="".join(chunks)
    return "// Project corotated material specialization: IEEE float32\n" + source, dependencies


_F32_SOURCE, _F32_DEPENDENCIES = _float32_source()


def configure_module(module=None):
    return build.configure_module(__name__ if module is None else module, HEADERS,
                                  extra_preamble=_F32_SOURCE, extra_dependencies=_F32_DEPENDENCIES)


_PACK32 = """
    wp::vec_t<12,float> packed;
    for (int i=0;i<3;++i) packed[i]=-out.gradient[i];
    for (int i=0;i<9;++i) packed[3+i]=out.hessian[i];
    return packed;
"""
_PACK64 = """
    wp::vec_t<12,float> packed;
    for (int i=0;i<3;++i) packed[i]=static_cast<float>(-out.gradient[i]);
    for (int i=0;i<9;++i) packed[3+i]=static_cast<float>(out.hessian[i]);
    return packed;
"""


def _triangle_input(namespace):
    return """
    NS::Triangle t{}; NS::Parameters p{};
    for(int a=0;a<3;++a){t.positions[a]=p0[a];t.positions[3+a]=p1[a];t.positions[6+a]=p2[a];}
    for(int a=0;a<2;++a)for(int b=0;b<2;++b)t.dm_inverse[2*a+b]=inverse.data[a][b];
    for(int a=0;a<2;++a)t.shape[a]=role==0?-t.dm_inverse[a]-t.dm_inverse[2+a]:t.dm_inverse[2*(role-1)+a];
    t.area=area; p.mu=mu; p.lambda=lam;
    const auto out=NS::evaluate(t,p);
    """.replace("NS", namespace)


@wp.func_native(_triangle_input("ipc_gpu_f32") + _PACK32)
def triangle(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, inverse: wp.mat22,
             area: float, mu: float, lam: float, role: int) -> Pack: ...


@wp.func_native(_triangle_input("ipc_gpu") + _PACK64)
def triangle_reference(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, inverse: wp.mat22,
                       area: float, mu: float, lam: float, role: int) -> Pack: ...


def _hinge_input(namespace):
    return """
    NS::Hinge t{}; NS::Parameters p{};
    for(int a=0;a<3;++a){t.positions[a]=p0[a];t.positions[3+a]=p1[a];t.positions[6+a]=p2[a];t.positions[9+a]=p3[a];}
    t.coefficient=coefficient; t.rest_angle=rest_angle; t.active_node=role; p.bending_stiffness=stiffness;
    const auto out=NS::evaluate(t,p);
    """.replace("NS", namespace)


@wp.func_native(_hinge_input("ipc_gpu_f32") + _PACK32)
def hinge(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3,
          coefficient: float, rest_angle: float, stiffness: float, role: int) -> Pack: ...


@wp.func_native(_hinge_input("ipc_gpu") + _PACK64)
def hinge_reference(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3,
                    coefficient: float, rest_angle: float, stiffness: float, role: int) -> Pack: ...


def _solid_input(namespace):
    return """
    NS::Solid t{}; NS::SolidParameters p{};
    for(int a=0;a<3;++a){
      t.positions[a]=p0[a];t.positions[3+a]=p1[a];t.positions[6+a]=p2[a];t.positions[9+a]=p3[a];
      t.shape[a]=shape[a];for(int b=0;b<3;++b)t.dm_inverse[3*a+b]=inverse.data[a][b];
    }
    t.measure=measure;p.mu=mu;p.lambda=lam;
    const auto out=NS::evaluate(t,p);
    """.replace("NS", namespace)


@wp.func_native(_solid_input("ipc_gpu_f32") + _PACK32)
def solid(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, inverse: wp.mat33,
          shape: wp.vec3, measure: float, mu: float, lam: float) -> Pack: ...


@wp.func_native(_solid_input("ipc_gpu") + _PACK64)
def solid_reference(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, inverse: wp.mat33,
                    shape: wp.vec3, measure: float, mu: float, lam: float) -> Pack: ...


@wp.func
def unpack(value: Pack):
    force = wp.vec3(value[0], value[1], value[2])
    hessian = wp.mat33(value[3],value[4],value[5],value[6],value[7],value[8],value[9],value[10],value[11])
    return force, hessian


@wp.func_native("""
    float result[3];
    valid=ipc_gpu_f32::solver::solve3(&hessian.data[0][0],force.c,result)?1:0;
    for(int i=0;i<3;++i)delta[i]=result[i];
""")
def _solve(hessian: wp.mat33, force: wp.vec3, delta: wp.ref[wp.vec3], valid: wp.ref[wp.int32]): ...


@wp.func
def solve(hessian: wp.mat33, force: wp.vec3):
    delta = wp.vec3(0.0)
    valid = wp.int32(0)
    _solve(hessian, force, delta, valid)
    return delta, valid


configure_module()
