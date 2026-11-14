"""Reference DAT friction with our zero-slip limiting derivative.

The float32 path calls Newton's actual ``compute_friction`` Warp function.
Only its zero-slip branch is completed to retain our nonzero limiting
tangential Hessian. The normal force is supplied by our IPC barrier.

Callers own geometry/scalar validation, float32 admissibility decisions and
packet overflow checks. They must retain the ordinary/exact replay when
those checks fail. This helper does not narrow the float64/exact path.
"""

import warp as wp
from newton._src.solvers.vbd import particle_vbd_kernels as stock


# Direct aliases, rather than copies of the native function implementations.
reference_compute_friction = stock.compute_friction
reference_orthonormal_basis = stock.orthonormal_basis
mat32 = stock.mat32


_FLOAT32_MIN = wp.constant(1.1754943508222875e-38)


@wp.func
def _lost_product(first: float, second: float, product: float):
    return first != 0.0 and second != 0.0 and (not wp.isfinite(product) or wp.abs(product) < _FLOAT32_MIN)


@wp.func
def evaluate_reference_friction_checked(
    normal: wp.vec3,
    relative_displacement: wp.vec3,
    normal_force: float,
    coefficient: float,
    epsilon_displacement: float,
):
    """Return ``(world_force, world_Hessian, needs_replay)``.

    A true replay flag rejects native float32 arithmetic whose intermediate
    values overflow, become subnormal, or lose a nonzero factor to zero.
    Callers must not commit flagged results; they retain the existing
    ordinary-double/exact evaluator. Geometry/scalar guards remain the
    caller's responsibility as well.
    """
    force = wp.vec3(0.0)
    hessian = wp.mat33(0.0)
    needs_replay = int(0)
    valid = wp.isfinite(normal_force) and normal_force >= 0.0
    valid = valid and wp.isfinite(coefficient) and coefficient >= 0.0
    valid = valid and wp.isfinite(epsilon_displacement) and epsilon_displacement >= _FLOAT32_MIN
    for i in range(3):
        valid = valid and wp.isfinite(normal[i]) and wp.isfinite(relative_displacement[i])
    normal_squared = wp.dot(normal, normal)
    valid = valid and wp.isfinite(normal_squared) and wp.abs(normal_squared - 1.0) <= 0.00000762939453125
    if not valid:
        return force, hessian, int(1)
    if coefficient == 0.0 or normal_force == 0.0:
        return force, hessian, needs_replay

    first, second = reference_orthonormal_basis(normal)
    tangent = mat32(first[0], second[0], first[1], second[1], first[2], second[2])
    slip = wp.transpose(tangent) * relative_displacement
    slip_norm = wp.length(slip)
    strength = coefficient * normal_force
    if _lost_product(coefficient, normal_force, strength):
        needs_replay = 1
    if not wp.isfinite(slip_norm):
        return force, hessian, int(1)

    if slip_norm == 0.0:
        # This includes norm-underflow with represented nonzero components.
        # Preserve the limiting force, but ask the full-precision path to
        # decide such a case instead of silently treating it as zero motion.
        if slip[0] != 0.0 or slip[1] != 0.0:
            needs_replay = 1
        limit = 2.0 / epsilon_displacement
        if not wp.isfinite(limit) or limit < _FLOAT32_MIN:
            needs_replay = 1
        scale = strength * limit
        if _lost_product(strength, limit, scale):
            needs_replay = 1
        # Combine the two scalar factors BEFORE multiplying tiny motion. The
        # native ordering limit*slip can underflow even when this force fits.
        force = -scale * (tangent * slip)
        hessian = scale * (tangent * wp.transpose(tangent))
    else:
        # This duplicates only the scalar factor for admissibility guards;
        # the accepted force/Hessian still come from the reference function.
        mollifier = float(0.0)
        if slip_norm > epsilon_displacement:
            mollifier = 1.0 / slip_norm
        else:
            mollifier = (-slip_norm / epsilon_displacement + 2.0) / epsilon_displacement
        if not wp.isfinite(mollifier) or mollifier < _FLOAT32_MIN:
            needs_replay = 1
        for j in range(2):
            scaled_slip = mollifier * slip[j]
            if _lost_product(mollifier, slip[j], scaled_slip):
                needs_replay = 1
            for i in range(3):
                scaled_tangent = strength * tangent[i, j]
                if _lost_product(strength, tangent[i, j], scaled_tangent):
                    needs_replay = 1
                # Also reject loss in the subsequent force contraction or
                # Hessian scaling, including components that are later summed.
                force_term = scaled_tangent * scaled_slip
                if _lost_product(scaled_tangent, scaled_slip, force_term):
                    needs_replay = 1
                hessian_term = scaled_tangent * mollifier
                if _lost_product(scaled_tangent, mollifier, hessian_term):
                    needs_replay = 1
        if needs_replay == 0:
            force, hessian = reference_compute_friction(
                coefficient, normal_force, tangent, slip, epsilon_displacement
            )

    for i in range(3):
        if not wp.isfinite(force[i]):
            needs_replay = 1
        for j in range(3):
            if not wp.isfinite(hessian[i, j]):
                needs_replay = 1
    return force, hessian, needs_replay
