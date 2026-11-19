"""Optional CPU-style local inverse/pseudoinverse policy, entirely binary32.

The force argument is minus the energy gradient: the returned displacement is
``damping * H^+ * force``. No PSD projection or descent safeguard is introduced.
Call ``configure_module(caller_kernel)`` after its other native preambles have
been installed. CUDA uses explicit round-to-nearest PTX without flush-to-zero,
so the caller's material fast-math/FMA options remain unchanged. The host
reference uses explicit float temporaries and should be built without fast math.
"""

import warp as wp

from . import warp_build as build


HEADERS = ("local_solve_fp32.cuh",)

# Nonnegative statuses return a valid displacement. Negative statuses return
# an all-zero displacement; they are exposed so diagnostics can count them.
ADJUGATE = 0
MINIMUM_NORM_QR = 1
ZERO = 2
INVALID_INPUT = -1
NUMERICAL_FAILURE = -2


def configure_module(module=None):
    """Append this header while preserving material headers and compiler flags."""
    selected = build._module(__name__ if module is None else module)
    options = wp.get_module_options(selected)
    flags = {key: options[key] for key in ("fast_math", "fuse_fp") if key in options}
    dependencies = build.configure_module(selected, HEADERS)
    if flags:
        wp.set_module_options(flags, module=selected)
    return dependencies


@wp.func_native(r"""
    float displacement[3];
    const int status = ipc_gpu::local_solve_fp32::solve3(
        &hessian.data[0][0], force.c, damping, displacement);
    wp::vec_t<4,float> result;
    for (int i = 0; i < 3; ++i) result[i] = displacement[i];
    result[3] = static_cast<float>(status);
    return result;
""")
def solve_with_status(hessian: wp.mat33, force: wp.vec3, damping: float) -> wp.vec4: ...


@wp.func
def solve(hessian: wp.mat33, force: wp.vec3, damping: float):
    """Return the signed local correction; no force/gradient sign conversion."""
    result = solve_with_status(hessian, force, damping)
    return wp.vec3(result[0], result[1], result[2])


configure_module()
