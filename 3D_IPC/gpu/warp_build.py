"""Compose project device arithmetic with Warp's SDK-free C++/CUDA compiler.

Only arithmetic headers under gpu/ are expanded. Host containers, CUDA runtime
calls and unknown system headers are rejected. The original files remain the
source of truth; their expanded contents participate in Warp's compilation-cache
hashes without checkout-specific paths.
"""

import hashlib
from pathlib import Path
import re

import warp as wp


GPU_ROOT = Path(__file__).resolve().parent
_ALLOWED = {"cstddef", "cstdint", "cmath", "cfloat", "cstring", "cstdlib", "limits", "type_traits"}
_INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"](?:\s*//.*)?\s*$')

_CRT = r"""
#ifndef IPC_WARP_ARITHMETIC_CRT
#define IPC_WARP_ARITHMETIC_CRT
#undef DBL_EPSILON
#define DBL_EPSILON 2.220446049250313080847263336181640625e-16
#ifndef DBL_TRUE_MIN
#define DBL_TRUE_MIN 4.940656458412465441765687928682213724e-324
#endif
#ifndef UINT64_C
#define UINT64_C(x) x##ULL
#define INT64_C(x) x##LL
#define UINT32_C(x) x##U
#define INT32_C(x) x
#endif
#ifndef offsetof
#if defined(__CUDACC_RTC__)
// NVRTC's EDG front end exposes this expression-form intrinsic instead of
// GCC/Clang's __builtin_offsetof. The wrapper makes the address expression a
// true integer constant expression, retaining the device ABI static_asserts.
#define offsetof(type, member) __INTADDR__((size_t)&(((type*)0)->member))
#else
#define offsetof(type, member) __builtin_offsetof(type, member)
#endif
#endif
#if !defined(__CUDACC__)
extern "C" {
double hypot(double,double); double copysign(double,double);
double fmax(double,double); double fmin(double,double);
double frexp(double,int*); double scalbn(double,int); double nextafter(double,double);
float hypotf(float,float); float copysignf(float,float);
float fmaxf(float,float); float fminf(float,float);
float sqrtf(float); float fabsf(float); float fmaf(float,float,float);
float atan2f(float,float); float powf(float,float); float sinf(float); float cosf(float);
}
#endif
namespace std {
using ::size_t;
using ::int8_t; using ::int16_t; using ::int32_t; using ::int64_t;
using ::uint8_t; using ::uint16_t; using ::uint32_t; using ::uint64_t;
using ::memcpy; using ::memset; using ::fma;
using ::isfinite; using ::isnan; using ::isinf;
template<class T> struct is_trivially_copyable { static constexpr bool value=__is_trivially_copyable(T); };
template<class T> struct is_standard_layout { static constexpr bool value=__is_standard_layout(T); };
template<class T> struct numeric_limits {
 static constexpr bool is_integer=true, is_signed=T(-1)<T(0);
 static constexpr int radix=2,digits=sizeof(T)*8-(is_signed?1:0);
 CUDA_CALLABLE static constexpr T max(){return is_signed?T((~uint64_t(0))>>(65-sizeof(T)*8)):T(~T(0));}
 CUDA_CALLABLE static constexpr T min(){return is_signed?T(-max()-T(1)):T(0);}
};
template<> struct numeric_limits<double> {
 static constexpr bool is_integer=false,is_signed=true,is_iec559=true;
 static constexpr int radix=2,digits=53,min_exponent=-1021,max_exponent=1024;
 CUDA_CALLABLE static constexpr double min(){return DBL_MIN;}
 CUDA_CALLABLE static constexpr double max(){return DBL_MAX;}
 CUDA_CALLABLE static constexpr double epsilon(){return DBL_EPSILON;}
 CUDA_CALLABLE static constexpr double denorm_min(){return DBL_TRUE_MIN;}
 CUDA_CALLABLE static double infinity(){return INFINITY;}
 CUDA_CALLABLE static double quiet_NaN(){return NAN;}
};
template<> struct numeric_limits<float> {
 static constexpr bool is_integer=false,is_signed=true,is_iec559=true;
 static constexpr int radix=2,digits=24,min_exponent=-125,max_exponent=128;
 CUDA_CALLABLE static constexpr float min(){return FLT_MIN;}
 CUDA_CALLABLE static constexpr float max(){return FLT_MAX;}
 CUDA_CALLABLE static constexpr float epsilon(){return FLT_EPSILON;}
 CUDA_CALLABLE static float infinity(){return INFINITY;}
 CUDA_CALLABLE static float quiet_NaN(){return NAN;}
};
}
#endif
"""


def flatten_headers(headers):
    """Expand arithmetic headers in memory, retaining include guards and licenses."""
    dependencies = []

    def expand(value, stack=()):
        path = Path(value)
        if not path.is_absolute():
            path = GPU_ROOT / path
        path = path.resolve()
        if not path.is_relative_to(GPU_ROOT) or not path.is_file():
            raise ValueError(f"Not a project device arithmetic header: {path}")
        if path in stack:
            raise ValueError(f"Recursive project include: {path}")
        if path not in dependencies:
            dependencies.append(path)
        relative = path.relative_to(GPU_ROOT).as_posix()
        guard = "IPC_WARP_HEADER_" + re.sub(r"[^A-Za-z0-9]", "_", relative).upper()
        lines = [f"#ifndef {guard}", f"#define {guard}", f"// Source: gpu/{relative}"]
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _INCLUDE.match(line)
            if match:
                style, name = match.groups()
                if style == '"':
                    lines.append(expand(path.parent / name, (*stack, path)))
                elif name not in _ALLOWED:
                    raise ValueError(f"Unsupported system header <{name}> in {path}")
            elif re.match(r"\s*#\s*pragma\s+once\s*$", line):
                continue
            elif re.match(r"\s*#\s*include\b", line):
                raise ValueError(f"Unsupported include in {path}: {line}")
            else:
                lines.append(line)
        lines.append(f"#endif // {guard}")
        return "\n".join(lines)

    return _CRT + "\n" + "\n".join(expand(path) for path in headers), tuple(dependencies)


def _module(value):
    if value is None:
        return wp.get_module(__name__)
    value = getattr(value, "module", value)
    if isinstance(value, str):
        return wp.get_module(value)
    if hasattr(value, "__name__"):
        return wp.get_module(value.__name__)
    return value


def configure_module(module=None, headers=(), *, extra_preamble="", extra_dependencies=()):
    """Merge independent physics components without overwriting their preambles."""
    module = _module(module)
    source, dependencies = flatten_headers(headers)
    # Warp hashes dependency realpaths as well as their contents. Expanded
    # headers are already hashed through the preamble, so including their paths
    # would unnecessarily rebuild identical code in relocated checkouts.
    seen = set(dependencies)
    fingerprints = []
    for value in extra_dependencies:
        path = Path(value)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"extra_dependencies entry must be an absolute existing file: {path}")
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            fingerprints.append(f"// IPC extra build dependency SHA256: {digest}")
    preamble = "\n".join((source, extra_preamble, *fingerprints))
    options = wp.get_module_options(module)
    existing = options.get("extra_build_options") or wp.ModuleBuildOptions()
    extra = wp.ModuleBuildOptions(
        extra_cpu_preamble=preamble,
        extra_cuda_preamble=preamble,
    )
    wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False,
                           "extra_build_options": existing.merged(extra)}, module=module)
    return dependencies
