#pragma once

// ===========================================================================
// rigid_dat_trajectory.h
//
// Dependency-free (C++17, standard headers only) rigid-body Divide-and-Truncate
// trajectory evaluation and *conservative* prefix certification.
//
// Mathematical model (all parameters are binary64):
//
//     phi(t) = center + t * delta_center
//              + exp(skew(dt * (omega + t * delta_omega))) * R(q0) * reference
//
// `q0` is a scalar-first quaternion (w, x, y, z).  It is interpreted exactly as
// supplied: it is *not* silently normalized.  It must already satisfy
// | |q0| - 1 | <= kQuaternionTolerance (1e-8); otherwise the input is rejected.
// The rotation it represents is the full homogeneous map
//
//     R(q0) * v = q0 * (0, v) * conj(q0)
//
// (i.e. R = [[ww+xx-yy-zz, 2(xy-wz), 2(xz+wy)], ...]), not the unit-only
// simplification with diagonal entries 1 - 2(yy+zz).  Consequently a small,
// accepted q0 norm error scales the rotated reference exactly as in the
// repository convention instead of being silently ignored or normalized away.
//
// Numerical contract
// ------------------
// * `evaluate` is a point evaluator using a Rodrigues formula (a Taylor series
//   near theta = 0).  It is only a *proposal* of a pose at a proposed time; it
//   is not a certificate and no libm sin/cos accuracy bound is claimed here.
//   Callers that need a represented-pose guarantee must separately certify the
//   returned endpoint/pose (and any safety radius) themselves.
// * `trajectory_bounds` returns an axis-aligned box that contains the complete
//   mathematical path phi([a,b]) for the supplied binary64 parameters, not
//   merely its endpoints.
// * Every set computation in `trajectory_bounds`, `plane_step` and
//   `radius_step` uses outward-rounded binary64 interval arithmetic
//   (std::nextafter at every rounded operation).  The rotation enclosure is
//   obtained by scaling and squaring the matrix exponential of skew(theta):
//   exp(skew(theta)) = exp(skew(theta)/2^s)^(2^s), with 2^s chosen by directed
//   powers-of-two so that the spectral norm of the scaled argument is at most
//   1/8.  The norm bound is the proved inequality
//   ||skew(theta)||_2 = ||theta||_2 <= |theta_x| + |theta_y| + |theta_z|,
//   evaluated with upward rounding on the interval component maxima (no
//   unrounded std::hypot is used for the scaling proof).  The scaled
//   exponential is enclosed by its degree-18 Taylor polynomial plus a
//   mathematically derived tail bound.  For ||B|| <= 1/8,
//
//       || exp(B) - sum_{k=0}^{18} B^k/k! ||_2
//          <= sum_{k=19}^inf (1/8)^k / k!
//          <= (1/8)^19 / 19! / (1 - 1/(8*20))
//          <  6e-35  <  kTaylorRemainder (1e-33).
//
//   The constant 1e-33 used below is a conservative upper bound derived from
//   that geometric tail estimate, not a heuristic epsilon.  Taylor
//   coefficients 1/k! (k <= 18, so 18! is exactly representable) are enclosed
//   in one-ulp intervals.  Repeated interval squaring then encloses exp of the
//   full argument for *every* theta in the interval box.  Consequently the
//   bounds become tight as interval widths shrink, remain valid for zero
//   rotation, and never rely on the accuracy of sin/cos.
// * Genuine positive/negative overflow is kept outward: a lower endpoint that
//   overflows to +infinity is saturated to DBL_MAX and an upper endpoint that
//   overflows to -infinity is saturated to -DBL_MAX (the opposite infinities
//   remain unbounded).  A NaN endpoint fails closed (never certifies
//   separation).  Intervals whose finite enclosure cannot be represented
//   throw std::overflow_error rather than being accepted.
// * Sample points and root bisection only *propose* times.  A prefix is
//   accepted only after the complete interval [0,t] has been certified by
//   adaptive interval subdivision.  An interval that cannot be certified is
//   reported as unsafe (fail closed), never as collision-free.
// * `plane_step` orients the plane from the signed distance at t = 0.  If the
//   initial distance is zero or numerically indistinguishable from zero, no
//   positive motion is accepted.  When the path first reaches the plane at a
//   time T <= 1, the returned relaxed step is gamma*T (gamma applied once);
//   if the whole path [0,1] is certified strictly separated, the result is 1.
// * `radius_step` returns the largest certified prefix t in [0,1] for which
//   |phi(s) - phi(0)| <= radius for all s in [0,t].  No additional gamma
//   relaxation is applied; 1 is returned for a fully certified path.
// * A mathematically no-motion path (zero delta_center and a rotation that
//   cannot vary because dt == 0, delta_omega == 0, or reference == 0) returns
//   1 from both query functions once the initial half-space separation is
//   strict.
// * All public functions are inline.  Invalid/non-finite input, bad
//   quaternions, zero/non-finite normals, out-of-range gamma, and
//   out-of-range budgets throw std::invalid_argument (or std::domain_error /
//   std::overflow_error for unrepresentable rotation angles).
// * Certification assumes IEEE-754 binary64 with round-to-nearest and does not
//   support -ffast-math reassociation; -ffp-contract=off is recommended.
//   Adaptive subdivision is depth-capped (256) and node-budgeted (<= 4e6).
// ===========================================================================

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace rigid_dat {

using Vector = std::array<double, 3>;
using Quaternion = std::array<double, 4>;

struct Trajectory {
    Vector center{}, delta_center{}, omega{}, delta_omega{}, reference{};
    Quaternion q0{{1, 0, 0, 0}};
    double dt = 1;
};

struct Box { Vector lo{}, hi{}; };

namespace detail {

static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "rigid_dat interval certification requires IEEE-754 binary64");

// Documented quaternion roundoff tolerance, applied to | |q0| - 1 |.
inline constexpr double kQuaternionTolerance = 1e-8;

// Taylor order of the scaled exponential.  For ||B|| <= 1/8 the omitted tail
// is below 6e-35; see the header comment for the derivation.
inline constexpr int kTaylorOrder = 18;
inline constexpr double kTaylorRemainder = 1e-33;

// Largest scaling exponent used for exp(skew(theta)).  This supports rotation
// norms (and hence angles) just below 2^57 ~= 1.44e17 radians while keeping
// the interval computation finite and bounded; larger arguments throw
// std::domain_error rather than being silently approximated.
inline constexpr int kMaxScalingExponent = 60;

// Budget limits for sampling/bisection.
inline constexpr int kMaxSamples = 1 << 20;
inline constexpr int kMaxIterations = 1 << 20;
inline constexpr int kMaxDepth = 256;
inline constexpr long long kMaxBoxBudget = 4000000;

struct Interval {
    double lo;
    double hi;
};

inline double round_down(double value) {
    if (std::isnan(value)) return value;
    if (value == std::numeric_limits<double>::infinity())
        return std::numeric_limits<double>::max();
    if (value == -std::numeric_limits<double>::infinity()) return value;
    return std::nextafter(value, -std::numeric_limits<double>::infinity());
}

inline double round_up(double value) {
    if (std::isnan(value)) return value;
    if (value == -std::numeric_limits<double>::infinity())
        return -std::numeric_limits<double>::max();
    if (value == std::numeric_limits<double>::infinity()) return value;
    return std::nextafter(value, std::numeric_limits<double>::infinity());
}

inline Interval make_interval(double value) { return Interval{value, value}; }

inline bool is_zero(Interval value) {
    return value.lo == 0.0 && value.hi == 0.0;
}

inline Interval add(Interval a, Interval b) {
    // Exact-zero additions do not need widening and must not manufacture a
    // subnormal-width interval (important for exactly stationary paths).
    if (is_zero(b)) return a;
    if (is_zero(a)) return b;
    double lo = a.lo + b.lo;
    double hi = a.hi + b.hi;
    if (std::isnan(lo)) lo = -std::numeric_limits<double>::infinity();
    if (std::isnan(hi)) hi = std::numeric_limits<double>::infinity();
    return Interval{round_down(lo), round_up(hi)};
}

inline Interval sub(Interval a, Interval b) {
    if (is_zero(b)) return a;
    double lo = a.lo - b.hi;
    double hi = a.hi - b.lo;
    if (std::isnan(lo)) lo = -std::numeric_limits<double>::infinity();
    if (std::isnan(hi)) hi = std::numeric_limits<double>::infinity();
    return Interval{round_down(lo), round_up(hi)};
}

inline double safe_product(double a, double b) {
    // Avoid the indeterminate 0 * inf form; an exact zero factor contributes
    // an exact zero in interval arithmetic.
    if (a == 0.0 || b == 0.0) return 0.0;
    return a * b;
}

inline Interval mul(Interval a, Interval b) {
    if (is_zero(a) || is_zero(b)) return make_interval(0.0);
    const double products[4] = {
        safe_product(a.lo, b.lo), safe_product(a.lo, b.hi),
        safe_product(a.hi, b.lo), safe_product(a.hi, b.hi)
    };
    bool saw_nan = false;
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (double p : products) {
        if (std::isnan(p)) {
            saw_nan = true;
        } else {
            if (p < lo) lo = p;
            if (p > hi) hi = p;
        }
    }
    if (saw_nan || lo > hi) {
        return Interval{-std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()};
    }
    return Interval{round_down(lo), round_up(hi)};
}

inline Interval negate(Interval a) { return Interval{-a.hi, -a.lo}; }

inline Interval scale_pow2(Interval a, int exponent) {
    if (is_zero(a)) return a;
    if (exponent == 0) return a;
    return Interval{round_down(std::ldexp(a.lo, -exponent)),
                    round_up(std::ldexp(a.hi, -exponent))};
}

inline double max_abs(Interval a) {
    // A NaN endpoint means a corrupted computation: fail closed with an upper
    // bound that cannot certify any separation.
    if (std::isnan(a.lo) || std::isnan(a.hi))
        return std::numeric_limits<double>::infinity();
    const double lo_abs = std::fabs(a.lo);
    const double hi_abs = std::fabs(a.hi);
    return lo_abs > hi_abs ? lo_abs : hi_abs;
}

// Upward upper bound for the spectral norm of skew(theta) over an interval
// box.  For every member theta, ||skew(theta)||_2 = ||theta||_2 <=
// |theta_x| + |theta_y| + |theta_z|, and each maximum is an upper bound for
// the corresponding absolute component.  All additions are rounded upward.
inline double rotation_norm_upper(const std::array<Interval, 3>& theta) {
    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double component = max_abs(theta[i]);
        if (!std::isfinite(component))
            return std::numeric_limits<double>::infinity();
        sum = round_up(sum + component);
    }
    return sum;
}

// Same bound for a concrete (point) rotation vector, used only to enforce the
// same supported argument range in point evaluation.
inline double rotation_norm_upper_point(const Vector& theta) {
    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double component = std::fabs(theta[i]);
        if (!std::isfinite(component))
            return std::numeric_limits<double>::infinity();
        sum = round_up(sum + component);
    }
    return sum;
}

// Smallest scaling exponent s >= 0 (up to the supported cap) such that
// rho_upper / 2^s <= 1/8.  std::frexp avoids forming rho_upper * 8, which
// could overflow even when the required scaling itself is representable.
inline int scaling_exponent_for_norm(double rho_upper) {
    if (!(rho_upper >= 0.0) || !std::isfinite(rho_upper)) {
        throw std::overflow_error(
            "rigid_dat: rotation angle is not representable");
    }
    if (rho_upper == 0.0) return 0;
    int exponent = 0;
    std::frexp(rho_upper, &exponent);
    int scale = exponent + 3;
    if (scale < 0) scale = 0;
    if (scale > kMaxScalingExponent) {
        throw std::domain_error(
            "rigid_dat: rotation angle exceeds the supported interval range");
    }
    return scale;
}

struct IntervalMatrix {
    Interval m[3][3];
};

inline IntervalMatrix zero_matrix() {
    IntervalMatrix r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r.m[i][j] = make_interval(0.0);
    return r;
}

inline IntervalMatrix identity_matrix() {
    IntervalMatrix r = zero_matrix();
    r.m[0][0] = make_interval(1.0);
    r.m[1][1] = make_interval(1.0);
    r.m[2][2] = make_interval(1.0);
    return r;
}

inline IntervalMatrix matrix_multiply(const IntervalMatrix& a,
                                      const IntervalMatrix& b) {
    IntervalMatrix c = zero_matrix();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Interval sum = make_interval(0.0);
            for (int k = 0; k < 3; ++k)
                sum = add(sum, mul(a.m[i][k], b.m[k][j]));
            c.m[i][j] = sum;
        }
    }
    return c;
}

// Intervals containing the exact coefficients 1/k!, k = 0..18.  Factorials up
// to 18! are exactly representable in binary64; the correctly rounded
// reciprocal lies within one ulp of the exact value, so one outward step on
// each side is a rigorous enclosure of 1/k!.
inline const std::array<Interval, kTaylorOrder + 1>& taylor_coefficients() {
    static const std::array<Interval, kTaylorOrder + 1> coeffs = [] {
        std::array<Interval, kTaylorOrder + 1> c{};
        double factorial = 1.0;
        c[0] = make_interval(1.0);
        for (int k = 1; k <= kTaylorOrder; ++k) {
            factorial *= static_cast<double>(k);
            const double inverse = 1.0 / factorial;
            c[k] = Interval{round_down(inverse), round_up(inverse)};
        }
        return c;
    }();
    return coeffs;
}

// Enclose exp(skew(theta)) * v for all theta in the supplied 3-D interval box
// and all v in the supplied 3-D interval box.
inline std::array<Interval, 3> rotation_action_interval(
    const std::array<Interval, 3>& theta,
    const std::array<Interval, 3>& v) {

    if (is_zero(v[0]) && is_zero(v[1]) && is_zero(v[2])) return v;

    const double rho = rotation_norm_upper(theta);
    if (!std::isfinite(rho)) {
        throw std::overflow_error("rigid_dat: rotation angle is not representable");
    }
    if (rho == 0.0) return v;

    // Choose s with ||theta|| / 2^s <= 1/8 for every theta in the box.  The
    // power-of-two scaling is outward-rounded, so verify the bound on the
    // scaled interval itself and, if one rounding ulp pushed it above 1/8,
    // increase s (each increase halves the bound).
    int scale = scaling_exponent_for_norm(rho);
    std::array<Interval, 3> scaled_theta{};
    for (;;) {
        for (int i = 0; i < 3; ++i)
            scaled_theta[i] = scale_pow2(theta[i], scale);
        if (rotation_norm_upper(scaled_theta) <= 0.125) break;
        ++scale;
        if (scale > kMaxScalingExponent) {
            throw std::domain_error(
                "rigid_dat: rotation angle exceeds the supported interval "
                "range");
        }
    }

    // B = skew(theta) / 2^scale, as an interval matrix.
    IntervalMatrix b = zero_matrix();
    b.m[0][1] = negate(scaled_theta[2]);
    b.m[0][2] = scaled_theta[1];
    b.m[1][0] = scaled_theta[2];
    b.m[1][2] = negate(scaled_theta[0]);
    b.m[2][0] = negate(scaled_theta[1]);
    b.m[2][1] = scaled_theta[0];

    // Degree-18 Taylor enclosure of exp(B).
    const auto& coeffs = taylor_coefficients();
    IntervalMatrix term = identity_matrix();
    IntervalMatrix poly = zero_matrix();
    for (int k = 0; k <= kTaylorOrder; ++k) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                poly.m[i][j] = add(poly.m[i][j], mul(coeffs[k], term.m[i][j]));
        if (k < kTaylorOrder) term = matrix_multiply(term, b);
    }

    // Add the rigorous tail bound entrywise (spectral norm bounds entrywise
    // magnitude).
    const Interval tail{-kTaylorRemainder, kTaylorRemainder};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            poly.m[i][j] = add(poly.m[i][j], tail);

    // exp(theta) = exp(theta/2^s)^(2^s); interval squaring encloses every
    // member of the scaled family.
    for (int k = 0; k < scale; ++k) poly = matrix_multiply(poly, poly);

    std::array<Interval, 3> result{};
    for (int i = 0; i < 3; ++i) {
        Interval sum = make_interval(0.0);
        for (int j = 0; j < 3; ++j)
            sum = add(sum, mul(poly.m[i][j], v[j]));
        result[i] = sum;
    }
    return result;
}

inline Vector cross(const Vector& a, const Vector& b) {
    return Vector{a[1] * b[2] - a[2] * b[1],
                  a[2] * b[0] - a[0] * b[2],
                  a[0] * b[1] - a[1] * b[0]};
}

inline Vector quaternion_apply(const Quaternion& q, const Vector& v) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double ww = w * w, xx = x * x, yy = y * y, zz = z * z;
    const double xy = x * y, xz = x * z, yz = y * z;
    const double wx = w * x, wy = w * y, wz = w * z;
    return Vector{
        (ww + xx - yy - zz) * v[0] + 2.0 * (xy - wz) * v[1] +
            2.0 * (xz + wy) * v[2],
        2.0 * (xy + wz) * v[0] + (ww - xx + yy - zz) * v[1] +
            2.0 * (yz - wx) * v[2],
        2.0 * (xz - wy) * v[0] + 2.0 * (yz + wx) * v[1] +
            (ww - xx - yy + zz) * v[2]};
}

inline std::array<Interval, 3> quaternion_apply_interval(const Quaternion& q,
                                                         const Vector& v) {
    const Interval w = make_interval(q[0]);
    const Interval x = make_interval(q[1]);
    const Interval y = make_interval(q[2]);
    const Interval z = make_interval(q[3]);
    const Interval two = make_interval(2.0);

    const Interval ww = mul(w, w);
    const Interval xx = mul(x, x);
    const Interval yy = mul(y, y);
    const Interval zz = mul(z, z);
    const Interval xy = mul(x, y);
    const Interval xz = mul(x, z);
    const Interval yz = mul(y, z);
    const Interval wx = mul(w, x);
    const Interval wy = mul(w, y);
    const Interval wz = mul(w, z);

    const Interval r00 = sub(add(ww, xx), add(yy, zz));
    const Interval r01 = mul(two, sub(xy, wz));
    const Interval r02 = mul(two, add(xz, wy));
    const Interval r10 = mul(two, add(xy, wz));
    const Interval r11 = sub(add(ww, yy), add(xx, zz));
    const Interval r12 = mul(two, sub(yz, wx));
    const Interval r20 = mul(two, sub(xz, wy));
    const Interval r21 = mul(two, add(yz, wx));
    const Interval r22 = sub(add(ww, zz), add(xx, yy));

    const Interval vx = make_interval(v[0]);
    const Interval vy = make_interval(v[1]);
    const Interval vz = make_interval(v[2]);

    std::array<Interval, 3> out{};
    out[0] = add(add(mul(r00, vx), mul(r01, vy)), mul(r02, vz));
    out[1] = add(add(mul(r10, vx), mul(r11, vy)), mul(r12, vz));
    out[2] = add(add(mul(r20, vx), mul(r21, vy)), mul(r22, vz));
    return out;
}

// Point evaluation of exp(skew(theta(t))) * u0 plus the center term.  `u0` is
// already R(q0) * reference.
inline Vector evaluate_core(const Trajectory& path, double t,
                            const Vector& u0) {
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(u0[i])) {
            throw std::overflow_error(
                "rigid_dat: rotated reference is not representable");
        }
    }

    Vector theta{};
    if (path.dt != 0.0) {
        theta = Vector{
            path.dt * (path.omega[0] + t * path.delta_omega[0]),
            path.dt * (path.omega[1] + t * path.delta_omega[1]),
            path.dt * (path.omega[2] + t * path.delta_omega[2])};
    }
    if (!std::isfinite(theta[0]) || !std::isfinite(theta[1]) ||
        !std::isfinite(theta[2])) {
        throw std::overflow_error("rigid_dat: rotation angle is not representable");
    }

    // Point evaluation supports the same rotation-angle range as the interval
    // exponential (that is where the scaling/squaring certificate is valid).
    (void)scaling_exponent_for_norm(rotation_norm_upper_point(theta));

    const double angle =
        std::hypot(theta[0], std::hypot(theta[1], theta[2]));
    if (!std::isfinite(angle)) {
        throw std::overflow_error("rigid_dat: rotation angle is not representable");
    }

    const Vector first = cross(theta, u0);
    const Vector second = cross(theta, first);
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(first[i]) || !std::isfinite(second[i])) {
            throw std::overflow_error(
                "rigid_dat: rotation evaluation is not representable");
        }
    }

    double c1 = 0.0;
    double c2 = 0.0;
    if (angle < 1e-4) {
        // sin(a)/a and (1-cos(a))/a^2 via Taylor series; stable as a -> 0.
        const double a2 =
            theta[0] * theta[0] + theta[1] * theta[1] + theta[2] * theta[2];
        c1 = 1.0 +
             a2 * (-1.0 / 6.0 + a2 * (1.0 / 120.0 + a2 * (-1.0 / 5040.0)));
        c2 = 0.5 +
             a2 * (-1.0 / 24.0 + a2 * (1.0 / 720.0 + a2 * (-1.0 / 40320.0)));
    } else {
        const double sine = std::sin(angle);
        const double cosine = std::cos(angle);
        c1 = sine / angle;
        c2 = (1.0 - cosine) / (angle * angle);
    }
    if (!std::isfinite(c1) || !std::isfinite(c2)) {
        throw std::overflow_error(
            "rigid_dat: rotation coefficients are not representable");
    }

    Vector rotated{};
    for (int i = 0; i < 3; ++i) {
        rotated[i] = u0[i] + c1 * first[i] + c2 * second[i];
        if (!std::isfinite(rotated[i])) {
            throw std::overflow_error(
                "rigid_dat: rotation evaluation is not representable");
        }
    }

    Vector result{};
    for (int i = 0; i < 3; ++i) {
        result[i] = path.center[i] + t * path.delta_center[i] + rotated[i];
        if (!std::isfinite(result[i])) {
            throw std::overflow_error(
                "rigid_dat: trajectory evaluation is not representable");
        }
    }
    return result;
}

inline void validate_trajectory(const Trajectory& path) {
    const double values[20] = {
        path.center[0], path.center[1], path.center[2],
        path.delta_center[0], path.delta_center[1], path.delta_center[2],
        path.omega[0], path.omega[1], path.omega[2],
        path.delta_omega[0], path.delta_omega[1], path.delta_omega[2],
        path.reference[0], path.reference[1], path.reference[2],
        path.q0[0], path.q0[1], path.q0[2], path.q0[3],
        path.dt};
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "rigid_dat: trajectory parameters must be finite");
        }
    }
    const double qnorm = std::hypot(std::hypot(path.q0[0], path.q0[1]),
                                    std::hypot(path.q0[2], path.q0[3]));
    if (!std::isfinite(qnorm) ||
        std::fabs(qnorm - 1.0) > kQuaternionTolerance) {
        throw std::invalid_argument(
            "rigid_dat: q0 must be a unit quaternion (scalar first)");
    }
}

inline Interval dot_box(const Vector& normal, const Box& box) {
    Interval sum = make_interval(0.0);
    for (int i = 0; i < 3; ++i)
        sum = add(sum, mul(make_interval(normal[i]),
                           Interval{box.lo[i], box.hi[i]}));
    return sum;
}

inline Interval dot_scalar_interval(const Vector& a, const Vector& b) {
    Interval sum = make_interval(0.0);
    for (int i = 0; i < 3; ++i)
        sum = add(sum, mul(make_interval(a[i]), make_interval(b[i])));
    return sum;
}

inline double dot_double(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline bool all_zero(const Vector& v) {
    return v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0;
}

inline Box bounds_core(const Trajectory& path, double a, double b,
                       const std::array<Interval, 3>& v0) {
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(v0[i].lo) || !std::isfinite(v0[i].hi)) {
            throw std::overflow_error(
                "rigid_dat: rotated reference is not representable");
        }
    }
    const Interval t_interval{a, b};
    std::array<Interval, 3> theta{};
    for (int i = 0; i < 3; ++i) {
        const Interval linear =
            add(make_interval(path.omega[i]),
                mul(t_interval, make_interval(path.delta_omega[i])));
        theta[i] = mul(make_interval(path.dt), linear);
    }

    const std::array<Interval, 3> rotated =
        rotation_action_interval(theta, v0);

    Box box{};
    for (int i = 0; i < 3; ++i) {
        const Interval center_term =
            add(make_interval(path.center[i]),
                mul(t_interval, make_interval(path.delta_center[i])));
        const Interval value = add(center_term, rotated[i]);
        if (!std::isfinite(value.lo) || !std::isfinite(value.hi)) {
            throw std::overflow_error(
                "rigid_dat: trajectory bounds overflow binary64");
        }
        box.lo[i] = value.lo;
        box.hi[i] = value.hi;
    }
    return box;
}

struct PlanePrefixScan {
    const Trajectory* path = nullptr;
    Vector normal{};
    Interval normal_dot_point{};
    std::array<Interval, 3> v0{};
    long long budget = 0;
    int max_depth = 1;
    double last_safe = 0.0;
    bool stopped = false;

    bool interval_is_safe(double a, double b) const {
        const Box box = bounds_core(*path, a, b, v0);
        const Interval distance = sub(dot_box(normal, box), normal_dot_point);
        return distance.lo > 0.0;
    }

    // Depth-first, time-ordered subdivision.  A safe leaf extends the proven
    // prefix; the first leaf that cannot be proven safe stops the scan.
    bool scan(double a, double b, int depth) {
        if (stopped) return false;
        if (budget <= 0) {
            stopped = true;
            return false;
        }
        --budget;
        if (interval_is_safe(a, b)) {
            last_safe = b;
            return true;
        }
        if (depth >= max_depth) {
            stopped = true;
            return false;
        }
        const double mid = a + 0.5 * (b - a);
        if (!(mid > a && mid < b)) {
            stopped = true;
            return false;
        }
        if (!scan(a, mid, depth + 1)) return false;
        return scan(mid, b, depth + 1);
    }
};

struct RadiusPrefixScan {
    const Trajectory* path = nullptr;
    std::array<Interval, 3> v0{};
    Box reference_box{};
    double radius = 0.0;
    long long budget = 0;
    int max_depth = 1;
    double last_safe = 0.0;
    bool stopped = false;

    bool interval_is_safe(double a, double b) const {
        const Box box = bounds_core(*path, a, b, v0);
        Interval squared_sum = make_interval(0.0);
        for (int i = 0; i < 3; ++i) {
            const Interval difference =
                sub(Interval{box.lo[i], box.hi[i]},
                    Interval{reference_box.lo[i], reference_box.hi[i]});
            const double component_bound = max_abs(difference);
            if (std::isnan(component_bound)) return false;
            const Interval square =
                mul(make_interval(component_bound),
                    make_interval(component_bound));
            squared_sum = add(squared_sum, square);
        }
        if (!(squared_sum.hi >= 0.0) ||
            !std::isfinite(squared_sum.hi)) {
            return false;
        }
        const double distance_upper =
            round_up(std::sqrt(squared_sum.hi));
        return distance_upper <= radius;
    }

    bool scan(double a, double b, int depth) {
        if (stopped) return false;
        if (budget <= 0) {
            stopped = true;
            return false;
        }
        --budget;
        if (interval_is_safe(a, b)) {
            last_safe = b;
            return true;
        }
        if (depth >= max_depth) {
            stopped = true;
            return false;
        }
        const double mid = a + 0.5 * (b - a);
        if (!(mid > a && mid < b)) {
            stopped = true;
            return false;
        }
        if (!scan(a, mid, depth + 1)) return false;
        return scan(mid, b, depth + 1);
    }
};

}  // namespace detail

// Point proposal only: see the numerical contract above.  This is not a
// certificate of any represented endpoint or of the path between samples.
inline Vector evaluate(const Trajectory& path, double t) {
    detail::validate_trajectory(path);
    if (!std::isfinite(t)) {
        throw std::invalid_argument("rigid_dat: time must be finite");
    }
    const Vector u0 = detail::quaternion_apply(path.q0, path.reference);
    return detail::evaluate_core(path, t, u0);
}

inline Box trajectory_bounds(const Trajectory& path, double a, double b) {
    detail::validate_trajectory(path);
    if (!std::isfinite(a) || !std::isfinite(b)) {
        throw std::invalid_argument(
            "rigid_dat: interval endpoints must be finite");
    }
    if (a > b) {
        throw std::invalid_argument(
            "rigid_dat: interval requires a <= b");
    }
    const std::array<detail::Interval, 3> v0 =
        detail::quaternion_apply_interval(path.q0, path.reference);
    return detail::bounds_core(path, a, b, v0);
}

inline double plane_step(const Trajectory& path, const Vector& normal,
                         const Vector& plane_point, double gamma,
                         int samples = 8, int iterations = 40) {
    detail::validate_trajectory(path);
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(normal[i]) || !std::isfinite(plane_point[i])) {
            throw std::invalid_argument(
                "rigid_dat: plane data must be finite");
        }
    }
    const double normal_length =
        std::hypot(normal[0], std::hypot(normal[1], normal[2]));
    if (!(normal_length > 0.0) || !std::isfinite(normal_length)) {
        throw std::invalid_argument(
            "rigid_dat: plane normal must be finite and nonzero");
    }
    if (!std::isfinite(gamma) || gamma < 0.0 || gamma > 1.0) {
        throw std::invalid_argument("rigid_dat: gamma must lie in [0,1]");
    }
    if (samples < 1 || samples > detail::kMaxSamples) {
        throw std::invalid_argument(
            "rigid_dat: samples outside supported range");
    }
    if (iterations < 1 || iterations > detail::kMaxIterations) {
        throw std::invalid_argument(
            "rigid_dat: iterations outside supported range");
    }
    if (gamma == 0.0) return 0.0;

    const std::array<detail::Interval, 3> v0 =
        detail::quaternion_apply_interval(path.q0, path.reference);

    // Orient the half-space using a certified enclosure of the initial signed
    // distance.  Boundary or uncertain initial contact fails closed.
    const Box initial_box = detail::bounds_core(path, 0.0, 0.0, v0);
    const detail::Interval initial_dot_point =
        detail::dot_scalar_interval(normal, plane_point);
    const detail::Interval initial_distance =
        detail::sub(detail::dot_box(normal, initial_box), initial_dot_point);

    Vector oriented_normal = normal;
    if (initial_distance.lo > 0.0) {
        // Keep the supplied orientation.
    } else if (initial_distance.hi < 0.0) {
        oriented_normal = Vector{-normal[0], -normal[1], -normal[2]};
    } else {
        return 0.0;
    }
    const detail::Interval oriented_dot_point =
        detail::dot_scalar_interval(oriented_normal, plane_point);

    // Exact no-motion paths: no translation and a rotation that cannot vary
    // (dt == 0, delta_omega == 0, or a zero reference vector).
    if (detail::all_zero(path.delta_center) &&
        (path.dt == 0.0 || detail::all_zero(path.delta_omega) ||
         detail::all_zero(path.reference))) {
        return 1.0;
    }

    detail::PlanePrefixScan scan{};
    scan.path = &path;
    scan.normal = oriented_normal;
    scan.normal_dot_point = oriented_dot_point;
    scan.v0 = v0;
    scan.max_depth = iterations < detail::kMaxDepth
                         ? iterations
                         : detail::kMaxDepth;
    scan.budget = 1024 + 64LL * (scan.max_depth + 1) +
                  8LL * static_cast<long long>(samples);
    if (scan.budget > detail::kMaxBoxBudget) scan.budget = detail::kMaxBoxBudget;

    // Paper-like sampling only proposes an upper time limit.  The scan below
    // still certifies the complete prefix; if the proposal turns out to be
    // causally wrong (roundoff), the remaining range is scanned as well.
    const Vector u0 = detail::quaternion_apply(path.q0, path.reference);
    double proposed_limit = 1.0;
    for (int i = 1; i <= samples; ++i) {
        const double sample_time =
            static_cast<double>(i) / static_cast<double>(samples);
        const Vector sample_position =
            detail::evaluate_core(path, sample_time, u0);
        const double signed_distance =
            detail::dot_double(oriented_normal, sample_position) -
            detail::dot_double(oriented_normal, plane_point);
        if (signed_distance <= 0.0) {
            proposed_limit = sample_time;
            break;
        }
    }

    bool fully_certified = scan.scan(0.0, proposed_limit, 0);
    if (fully_certified && proposed_limit < 1.0)
        fully_certified = scan.scan(proposed_limit, 1.0, 0);
    if (fully_certified) return 1.0;

    // Return the relaxed last proven safe prefix.  gamma is applied once; the
    // represented endpoint is kept strictly below the certified prefix.
    const double last_safe = scan.last_safe;
    if (!(last_safe > 0.0)) return 0.0;
    double relaxed = gamma * last_safe;
    if (!(relaxed < last_safe)) {
        relaxed = std::nextafter(last_safe, 0.0);
    }
    if (!(relaxed > 0.0)) return 0.0;
    return relaxed;
}

inline double radius_step(const Trajectory& path, double radius,
                          int iterations = 40) {
    detail::validate_trajectory(path);
    if (!std::isfinite(radius) || radius < 0.0) {
        throw std::invalid_argument(
            "rigid_dat: radius must be finite and nonnegative");
    }
    if (iterations < 1 || iterations > detail::kMaxIterations) {
        throw std::invalid_argument(
            "rigid_dat: iterations outside supported range");
    }

    // No-motion paths are safe for every radius (including a zero radius).
    if (detail::all_zero(path.delta_center) &&
        (path.dt == 0.0 || detail::all_zero(path.delta_omega) ||
         detail::all_zero(path.reference))) {
        return 1.0;
    }

    const std::array<detail::Interval, 3> v0 =
        detail::quaternion_apply_interval(path.q0, path.reference);

    detail::RadiusPrefixScan scan{};
    scan.path = &path;
    scan.v0 = v0;
    scan.reference_box = detail::bounds_core(path, 0.0, 0.0, v0);
    scan.radius = radius;
    scan.max_depth = iterations < detail::kMaxDepth
                         ? iterations
                         : detail::kMaxDepth;
    scan.budget = 1024 + 64LL * (scan.max_depth + 1);
    if (scan.budget > detail::kMaxBoxBudget) scan.budget = detail::kMaxBoxBudget;

    if (scan.scan(0.0, 1.0, 0)) return 1.0;
    return scan.last_safe;
}

}  // namespace rigid_dat
