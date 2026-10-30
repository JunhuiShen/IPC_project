#pragma once

// Independent ORIGINAL linear-CCD component.
//
// This header deliberately does not alter, wrap, or replace ccd.cuh's robust
// represented-endpoint certificate. It ports the CPU `original=true` backend:
// the scaled/local floating kernels with their exact ambiguity guards, plus
// the ORIGINAL exact-rational resolver (including its fixed 1e-10 world-space
// initial/event membership band). The fast entry point is allowed to return
// NeedsExact; it is never allowed to turn uncertainty into "no hit".
//
// device use:
//   * evaluate_fast is allocation-free and needs no integer pool;
//   * evaluate_exact<Capacity> uses ipc_gpu::exact's bounded thread-private
//     DevicePool. Install it (exact::initialize_pool) in the worker before the
//     query and release it after the query. A limb-capacity CapacityExceeded
//     result is retried with evaluate_exact<2048> AFTER the first pass has
//     released every slot without a pool failure; an allocator failure or
//     ownership leak is an explicit resource error, never a blind retry.
//
// Host use:
//   * evaluate_exact<Capacity> uses exact::Integer's normal host storage.
//
// Semantics intentionally preserved from ccd.cpp:
//   * positions/motions are converted to exact integers from the original
//     binary64 values BEFORE subtraction or scaling;
//   * initial intersection is distance(0) <= 1e-10, so a positive gap at or
//     below 1e-10 is an initial contact (t=0);
//   * the non-coplanar candidate is the affine tetrahedron-determinant root;
//   * a coplanar step evaluates all original incidence/edge/triple roots over
//     [0,1] and returns the EARLIEST intersecting event;
//   * the returned TOI is the CPU contact_result conversion: nearest binary64,
//     then one representable value toward zero if that rounded upward.
//
// No Boost, Eigen, host callback, dynamic allocation, or device allocation is
// used here. No shared exact.cuh file is edited.

#include "exact.cuh"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__CUDACC__)
#define IPC_ORIGINAL_HD __host__ __device__
#else
#define IPC_ORIGINAL_HD
#endif

namespace ipc_gpu {
namespace original_ccd {

enum class Status : unsigned { Ok, InvalidInput, NeedsExact, CapacityExceeded };

struct Query {
    double positions[4][3]{};
    double motion[4][3]{};
    // true: NT with positions[0] as the point and [1..3] as the triangle.
    // false: SS with positions [a,b,c,d].
    bool vertex_face = true;
    double eps = 1e-12;
};

// ---------------------------------------------------------------------------
// Result: fully defined 16-byte transfer record
// ---------------------------------------------------------------------------
// Host and device callers move whole Result arrays with
//     cudaMemcpy(..., count * sizeof(Result), ...)
// That transfer reads the COMPLETE object representation of every element, not
// only the three semantic fields. The natural layout
//     { Status status; bool collision; double toi; }
// is 16 bytes with `toi` at offset 8, and therefore has three IMPLICIT padding
// bytes at offsets 5..7. Aggregate initialization of `Result{s, b, t}` does
// not give those bytes a value, so a fieldwise device store (4-byte status,
// 1-byte collision, 8-byte toi) leaves them uninitialized in the cudaMalloc'ed
// result array. The DtoH cudaMemcpy then copies 16 indeterminate bytes per
// record, which compute-sanitizer (initcheck) reports as an uninitialized
// device-memory read of count*16 bytes even though every semantic field was
// written and is numerically correct.
//
// Fix (representation only, no numeric change): the three bytes are declared
// as a real, always-zero-initialized member that occupies exactly the ABI
// padding slot. Every construction path and every fieldwise or wide struct
// copy then covers all 16 bytes. The layout the public API and the CUDA
// transfer depend on is unchanged:
//
//   offset 0  status    4 bytes
//   offset 4  collision 1 byte
//   offset 5  reserved  3 bytes (never semantic, never compared)
//   offset 8  toi       8 bytes
//   sizeof == 16, alignof == 8
//
// Because `reserved` sits between `collision` and `toi`, the type is no longer
// an aggregate; the three-argument constructor below keeps the existing
// `Result{status, collision, toi}` spelling (host and device) valid, and the
// default case (Ok / no hit / toi 1.0) is unchanged. Standard-layout and
// trivial-copyability are preserved, so the type is still cudaMemcpy-safe.
// `reserved` is never read by evaluate_fast/evaluate_exact, by any status
// transition, or by any host/device comparison.
struct Result {
    Status status = Status::Ok;
    bool collision = false;
    unsigned char reserved[3] = {0, 0, 0};
    // no hit is represented by 1.0. CPU no-hit is NaN; compare `collision`
    // before ever comparing this field against a CPU TOI.
    double toi = 1.0;

    // `reserved` is zeroed explicitly in both constructors (in addition to its
    // default member initializer) so every construction path, host or device,
    // produces a record whose full 16-byte object representation is defined.
    IPC_ORIGINAL_HD Result() : reserved{0, 0, 0} {}
    IPC_ORIGINAL_HD Result(Status status_value, bool collision_value,
                           double toi_value)
        : status(status_value), collision(collision_value),
          reserved{0, 0, 0}, toi(toi_value) {}
};

// Pin the transfer contract the sanitizer-clean DtoH copy relies on. These
// assertions fail the build if a future field, reorder or packing pragma
// reintroduces implicit padding bytes into a Result array transfer.
static_assert(sizeof(Result) == 16,
              "Result must stay a 16-byte host/device transfer record");
static_assert(alignof(Result) == 8, "Result alignment changed");
static_assert(offsetof(Result, status) == 0, "Result::status offset changed");
static_assert(offsetof(Result, collision) == 4,
              "Result::collision offset changed");
static_assert(offsetof(Result, reserved) == 5,
              "Result::reserved must occupy the former ABI padding bytes");
static_assert(offsetof(Result, toi) == 8, "Result::toi offset changed");
static_assert(std::is_standard_layout<Result>::value,
              "Result must stay standard-layout");
static_assert(std::is_trivially_copyable<Result>::value,
              "Result must stay trivially copyable for cudaMemcpy");

// TEST-ONLY diagnostics copied back from the real CUDA harness. This type is
// never consulted by evaluate_fast/evaluate_exact and is not part of the
// production component contract. It lets the host test distinguish a genuine
// 256->2048 limb-capacity retry from a corrupt/exhausted DevicePool without
// weakening any production guard.
namespace test {

struct DeviceDiagnostics {
    unsigned long long exact_queries = 0;
    unsigned long long exact_ok = 0;
    unsigned long long exact_capacity_exceeded = 0;
    unsigned long long exact_invalid = 0;
    unsigned long long exact_other = 0;

    // First evaluate_exact<256> pass on every query.
    unsigned long long first_pass_failed = 0;
    unsigned long long first_pass_available_mismatch = 0;

    // Only a healthy first-pass pool (no failure, all slots returned) permits
    // the 256->2048 retry.
    unsigned long long retries_2048 = 0;
    unsigned long long retry_resource_errors = 0;
    unsigned long long ownership_errors = 0;

    // State after the evaluate_exact<2048> retry.
    unsigned long long final_pass_failed = 0;
    unsigned long long final_pass_available_mismatch = 0;
    unsigned long long final_pass_errors = 0;

    // Separate small-limb probe: capacity 2 must fail arithmetically while the
    // arena itself remains healthy, then capacity 2048 must recover.
    unsigned long long small_capacity_limb_failures = 0;
    unsigned long long small_capacity_pool_failures = 0;
    unsigned long long small_capacity_available_mismatch = 0;
    unsigned long long small_capacity_retry_ok = 0;
    unsigned long long small_capacity_retry_failures = 0;
    unsigned long long small_capacity_retry_pool_failures = 0;
    unsigned long long small_capacity_retry_available_mismatch = 0;

    int peak_slots = 0;
    int slot_count = 0;
};

} // namespace test

namespace float_detail {

IPC_ORIGINAL_HD inline bool finite(double x) { return x - x == 0.0; }
IPC_ORIGINAL_HD inline double absv(double x) { return x < 0.0 ? -x : x; }
// Match std::min/std::max's argument-order behavior, including NaN.
IPC_ORIGINAL_HD inline double minimum(double a, double b) {
    return b < a ? b : a;
}
IPC_ORIGINAL_HD inline double maximum(double a, double b) {
    return a < b ? b : a;
}
IPC_ORIGINAL_HD inline double clamp01(double x) {
    // CPU uses std::clamp(t, 0, 1).
    if (x < 0.0) return 0.0;
    if (1.0 < x) return 1.0;
    return x;
}
IPC_ORIGINAL_HD inline double max3(double a, double b, double c) {
    return maximum(a, maximum(b, c));
}
IPC_ORIGINAL_HD inline double max4(double a, double b, double c, double d) {
    return maximum(maximum(a, b), maximum(c, d));
}
IPC_ORIGINAL_HD inline double orig_fma(double a, double b, double c) {
    return ::fma(a, b, c);
}
IPC_ORIGINAL_HD inline double orig_sqrt(double x) { return ::sqrt(x); }
IPC_ORIGINAL_HD inline double orig_scalbn(double x, int e) {
    return ::scalbn(x, e);
}
IPC_ORIGINAL_HD inline double orig_ldexp(double x, int e) {
    return ::ldexp(x, e);
}
IPC_ORIGINAL_HD inline double orig_frexp(double x, int* e) {
    return ::frexp(x, e);
}

IPC_ORIGINAL_HD inline std::uint64_t double_bits(double x) {
#if defined(__CUDA_ARCH__)
    return static_cast<std::uint64_t>(__double_as_longlong(x));
#else
    std::uint64_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
#endif
}

IPC_ORIGINAL_HD inline double bits_double(std::uint64_t bits) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(static_cast<long long>(bits));
#else
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

// Next representable binary64 in the requested direction. Used only for the
// CPU-compatible downward adjustment of a rounded exact TOI.
IPC_ORIGINAL_HD inline double neighbor(double x, bool upward) {
    if (!finite(x)) return x;
    if (x == 0.0) return upward ? bits_double(UINT64_C(1))
                                : bits_double(UINT64_C(0x8000000000000001));
    std::uint64_t bits = double_bits(x);
    if (upward == (x > 0.0)) ++bits; else --bits;
    return bits_double(bits);
}

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct Vec3 {
    double v[3]{};
};

IPC_ORIGINAL_HD inline Vec2 vec2(double x, double y) {
    Vec2 r; r.x = x; r.y = y; return r;
}
IPC_ORIGINAL_HD inline Vec3 vec3(double x, double y, double z) {
    Vec3 r; r.v[0] = x; r.v[1] = y; r.v[2] = z; return r;
}
IPC_ORIGINAL_HD inline Vec3 vec3_sub(const Vec3& a, const Vec3& b) {
    return vec3(a.v[0] - b.v[0], a.v[1] - b.v[1], a.v[2] - b.v[2]);
}
IPC_ORIGINAL_HD inline Vec3 vec3_add(const Vec3& a, const Vec3& b) {
    return vec3(a.v[0] + b.v[0], a.v[1] + b.v[1], a.v[2] + b.v[2]);
}
IPC_ORIGINAL_HD inline Vec3 vec3_mul(const Vec3& a, double s) {
    return vec3(a.v[0] * s, a.v[1] * s, a.v[2] * s);
}
IPC_ORIGINAL_HD inline Vec3 vec3_div(const Vec3& a, double s) {
    return vec3(a.v[0] / s, a.v[1] / s, a.v[2] / s);
}
IPC_ORIGINAL_HD inline double vec3_dot(const Vec3& a, const Vec3& b) {
    return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2];
}
IPC_ORIGINAL_HD inline double vec3_squared_norm(const Vec3& a) {
    return vec3_dot(a, a);
}
IPC_ORIGINAL_HD inline double vec3_lp1(const Vec3& a) {
    return absv(a.v[0]) + absv(a.v[1]) + absv(a.v[2]);
}
IPC_ORIGINAL_HD inline double vec3_max_abs(const Vec3& a) {
    return max3(absv(a.v[0]), absv(a.v[1]), absv(a.v[2]));
}
IPC_ORIGINAL_HD inline bool vec3_all_finite(const Vec3& a) {
    return finite(a.v[0]) && finite(a.v[1]) && finite(a.v[2]);
}
// Eigen-compatible stableNorm: scale by the largest component first.
IPC_ORIGINAL_HD inline double vec3_stable_norm(const Vec3& a) {
    const double scale = vec3_max_abs(a);
    if (scale == 0.0) return 0.0;
    double sum = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double y = a.v[k] / scale;
        sum += y * y;
    }
    return scale * orig_sqrt(sum);
}
IPC_ORIGINAL_HD inline void vec3_max_abs_index(const Vec3& a, int& index) {
    index = 0;
    double best = absv(a.v[0]);
    for (int k = 1; k < 3; ++k) {
        const double value = absv(a.v[k]);
        if (value > best) { best = value; index = k; }
    }
}

IPC_ORIGINAL_HD inline Vec2 vec2_sub(const Vec2& a, const Vec2& b) {
    return vec2(a.x - b.x, a.y - b.y);
}

// Exact recovery of the rounding error of c*d. Same expression/order as the
// CPU linear backend.
IPC_ORIGINAL_HD inline double difference_of_products(double a, double b,
                                                     double c, double d) {
    const double cd = c * d;
    return orig_fma(a, b, -cd) + orig_fma(-c, d, cd);
}

IPC_ORIGINAL_HD inline double orient2(const Vec2& a, const Vec2& b) {
    return difference_of_products(a.x, b.y, a.y, b.x);
}

IPC_ORIGINAL_HD inline Vec3 stable_cross(const Vec3& a, const Vec3& b) {
    return vec3(difference_of_products(a.v[1], b.v[2], a.v[2], b.v[1]),
                difference_of_products(a.v[2], b.v[0], a.v[0], b.v[2]),
                difference_of_products(a.v[0], b.v[1], a.v[1], b.v[0]));
}

constexpr double kRoundoff = 64.0 * DBL_EPSILON;
constexpr double kLinearTimePrecision = 1.0e-8;
constexpr double kLinearValidationDistance = 1.0e-10;

struct LinearCCDGuard {
    bool uncertain = false;
    double separation = 0.0;
    double error = 0.0;
    double time_error = 0.0;

    IPC_ORIGINAL_HD double orientation(const Vec2& a, const Vec2& b) {
        const double value = orient2(a, b);
        error = kRoundoff * (absv(a.x * b.y) + absv(a.y * b.x));
        if (error > 0.0 && absv(value) <= error) uncertain = true;
        return value;
    }

    IPC_ORIGINAL_HD double triple(const Vec3& a, const Vec3& b, const Vec3& c,
                                  const Vec3& normal) {
        const double value = vec3_dot(normal, c);
        const double magnitude =
            (absv(a.v[1] * b.v[2]) + absv(a.v[2] * b.v[1])) * absv(c.v[0]) +
            (absv(a.v[2] * b.v[0]) + absv(a.v[0] * b.v[2])) * absv(c.v[1]) +
            (absv(a.v[0] * b.v[1]) + absv(a.v[1] * b.v[0])) * absv(c.v[2]);
        error = kRoundoff * magnitude;
        if (!finite(value) || (magnitude > 0.0 && absv(value) <= error)) {
            uncertain = true;
        }
        if (!uncertain) {
            const double edge_max =
                maximum(vec3_max_abs(a), vec3_max_abs(b));
            if (!(vec3_max_abs(normal) > 4.0 * separation * edge_max)) {
                const double area = vec3_stable_norm(normal);
                if (area > 0.0
                    && area <= separation * maximum(vec3_stable_norm(a),
                                                    vec3_stable_norm(b))) {
                    uncertain = true;
                }
            }
        }
        return value;
    }

    IPC_ORIGINAL_HD double root(double d, double c, double d_error) {
        const double t = -d / c;
        const double bound = d_error + absv(t) * error;
        if (bound > kLinearTimePrecision * absv(c)) uncertain = true;
        if (finite(t)) time_error = maximum(time_error, bound / absv(c));
        return t;
    }
};

IPC_ORIGINAL_HD inline double orientation_root(
        const Vec2& u, const Vec2& du, const Vec2& v, const Vec2& dv,
        const Vec2& point, const Vec2& dp, LinearCCDGuard& guard) {
    const double intercept = guard.orientation(vec2_sub(v, u),
                                               vec2_sub(point, u));
    const double intercept_error = guard.error;
    double slope;
    const bool u_moves = maximum(absv(du.x), absv(du.y)) != 0.0;
    const bool v_moves = maximum(absv(dv.x), absv(dv.y)) != 0.0;
    const bool p_moves = maximum(absv(dp.x), absv(dp.y)) != 0.0;
    if (u_moves) {
        slope = guard.orientation(du, vec2_sub(v, point));
    } else if (v_moves) {
        slope = guard.orientation(dv, vec2_sub(point, u));
    } else if (p_moves) {
        slope = guard.orientation(vec2_sub(v, u), dp);
    } else {
        slope = guard.orientation(vec2_sub(v, u), dp);
    }
    return slope == 0.0 ? NAN : guard.root(intercept, slope, intercept_error);
}

// Faithful port of ccd.cpp's LinearQuery. Values are first put into a local
// frame and then uniformly rescaled by an exact power of two.
struct LinearQuery {
    double x[4][3]{};
    double dx[4][3]{};
    int exponent = 0;
    bool requires_exact = false;
    double validation_distance = 0.0;

    IPC_ORIGINAL_HD LinearQuery(const double positions[4][3],
                                const double motion[4][3]) {
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 3; ++k) {
                x[i][k] = positions[i][k];
                dx[i][k] = motion[i][k];
            }
        const double origin[3] = {x[3][0], x[3][1], x[3][2]};
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 3; ++k) x[i][k] -= origin[k];

        bool all_finite = true;
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 3; ++k) all_finite = all_finite && finite(x[i][k]);
        if (!all_finite) {
            requires_exact = true;
            double world_max = 0.0;
            for (int i = 0; i < 4; ++i)
                for (int k = 0; k < 3; ++k)
                    world_max = maximum(world_max, absv(positions[i][k]));
            orig_frexp(world_max, &exponent);
            for (int i = 0; i < 4; ++i)
                for (int k = 0; k < 3; ++k) {
                    x[i][k] = orig_scalbn(positions[i][k], -exponent)
                        - orig_scalbn(origin[k], -exponent);
                    dx[i][k] = orig_scalbn(motion[i][k], -exponent);
                }
        }

        double magnitude = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 3; ++k) {
                magnitude = maximum(magnitude, absv(x[i][k]));
                magnitude = maximum(magnitude, absv(dx[i][k]));
            }

        int local_exponent = 0;
        bool normal_scale = false;
        double scale = 1.0;
        const std::uint64_t magnitude_bits = double_bits(magnitude);
        const int biased_exponent =
            static_cast<int>((magnitude_bits >> 52) & UINT64_C(0x7ff));
        if (biased_exponent >= 1 && biased_exponent <= 2044) {
            local_exponent = biased_exponent - 1022;
            const std::uint64_t scale_bits =
                static_cast<std::uint64_t>(2045 - biased_exponent) << 52;
            scale = bits_double(scale_bits);
            normal_scale = true;
        }
        if (!normal_scale) {
            orig_frexp(magnitude, &local_exponent);
            normal_scale = local_exponent >= -1023 && local_exponent <= 1022;
            if (normal_scale) scale = orig_scalbn(1.0, -local_exponent);
        }
        exponent += local_exponent;
        constexpr double minimum_safe_component = 0x1p-200;
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 3; ++k) {
                x[i][k] = normal_scale ? x[i][k] * scale
                                       : orig_scalbn(x[i][k], -local_exponent);
                dx[i][k] = normal_scale ? dx[i][k] * scale
                                        : orig_scalbn(dx[i][k], -local_exponent);
                if ((x[i][k] != 0.0 && absv(x[i][k]) < minimum_safe_component)
                    || (dx[i][k] != 0.0
                        && absv(dx[i][k]) < minimum_safe_component)) {
                    requires_exact = true;
                }
                if (motion[i][k] != 0.0 && dx[i][k] == 0.0) requires_exact = true;
                for (int j = 0; j < i; ++j) {
                    if (positions[i][k] != positions[j][k]
                        && x[i][k] == x[j][k]) requires_exact = true;
                }
            }
        }
        validation_distance = normal_scale && exponent == local_exponent
            ? kLinearValidationDistance * scale
            : orig_scalbn(kLinearValidationDistance, -exponent);
    }
};

IPC_ORIGINAL_HD inline bool in_unit_interval(double t, double eps) {
    return t >= -eps && t <= 1.0 + eps;
}

IPC_ORIGINAL_HD inline Vec2 project_drop_axis(const Vec3& v, int drop_axis) {
    if (drop_axis == 0) return vec2(v.v[1], v.v[2]);
    if (drop_axis == 1) return vec2(v.v[0], v.v[2]);
    return vec2(v.v[0], v.v[1]);
}

IPC_ORIGINAL_HD inline int dominant_drop_axis(const Vec3& n0,
                                              const Vec3& n1) {
    const Vec3 n = vec3(absv(n0.v[0]) + absv(n1.v[0]),
                        absv(n0.v[1]) + absv(n1.v[1]),
                        absv(n0.v[2]) + absv(n1.v[2]));
    if (n.v[0] >= n.v[1] && n.v[0] >= n.v[2]) return 0;
    if (n.v[1] >= n.v[0] && n.v[1] >= n.v[2]) return 1;
    return 2;
}

// Exact copy of ccd.cpp's swept-hull separation certificate. It can certify
// separation for a whole step, but it never proves a collision.
IPC_ORIGINAL_HD inline bool swept_axis_separates(
        const double points[2][4][3], int split, Vec3 axis,
        double separation) {
    const double magnitude = vec3_max_abs(axis);
    if (!(magnitude > 0.0) || !vec3_all_finite(axis)) return false;
    for (int k = 0; k < 3; ++k) axis.v[k] /= magnitude;
    double first_lo = INFINITY;
    double first_hi = -first_lo;
    double second_lo = first_lo;
    double second_hi = first_hi;
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        for (int i = 0; i < 4; ++i) {
            const Vec3 point = vec3(points[endpoint][i][0],
                                    points[endpoint][i][1],
                                    points[endpoint][i][2]);
            const double value = vec3_dot(axis, point);
            if (i < split) {
                first_lo = minimum(first_lo, value);
                first_hi = maximum(first_hi, value);
            } else {
                second_lo = minimum(second_lo, value);
                second_hi = maximum(second_hi, value);
            }
        }
    }
    const double margin =
        (4.0 * kRoundoff + 2.0 * separation) * vec3_lp1(axis);
    return maximum(second_lo - first_hi, first_lo - second_hi) > margin;
}

IPC_ORIGINAL_HD inline bool swept_hulls_separated(const LinearQuery& q,
                                                  bool vertex_face,
                                                  double separation) {
    double points[2][4][3]{};
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k) {
            points[0][i][k] = q.x[i][k];
            points[1][i][k] = q.x[i][k] + q.dx[i][k];
        }
    const int split = vertex_face ? 1 : 2;

    for (int k = 0; k < 3; ++k) {
        Vec3 axis = vec3(0.0, 0.0, 0.0);
        axis.v[k] = 1.0;
        if (swept_axis_separates(points, split, axis, separation)) return true;
    }

    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const double (*p)[3] = points[endpoint];
        const Vec3 p0 = vec3(p[0][0], p[0][1], p[0][2]);
        const Vec3 p1 = vec3(p[1][0], p[1][1], p[1][2]);
        const Vec3 p2 = vec3(p[2][0], p[2][1], p[2][2]);
        const Vec3 p3 = vec3(p[3][0], p[3][1], p[3][2]);
        if (vertex_face) {
            const Vec3 normal = stable_cross(vec3_sub(p2, p1),
                                             vec3_sub(p3, p1));
            if (swept_axis_separates(points, split, normal, separation))
                return true;
            for (int edge = 0; edge < 3; ++edge) {
                const int next_index = 1 + ((edge + 1) % 3);
                const int current_index = 1 + edge;
                const Vec3 edge_vector = vec3_sub(
                    vec3(p[next_index][0], p[next_index][1], p[next_index][2]),
                    vec3(p[current_index][0], p[current_index][1],
                         p[current_index][2]));
                const Vec3 axis = stable_cross(normal, edge_vector);
                if (swept_axis_separates(points, split, axis, separation))
                    return true;
            }
        } else {
            const Vec3 u = vec3_sub(p1, p0);
            const Vec3 v = vec3_sub(p3, p2);
            const Vec3 w = vec3_sub(p2, p0);
            if (swept_axis_separates(points, split, stable_cross(u, v),
                                     separation)
                || swept_axis_separates(points, split,
                    stable_cross(stable_cross(u, w), u), separation)
                || swept_axis_separates(points, split,
                    stable_cross(stable_cross(v, w), v), separation))
                return true;
            for (int endpoint2 = 0; endpoint2 < 4; ++endpoint2) {
                const int other = endpoint2 < 2 ? 2 : 0;
                const Vec3 edge = vec3_sub(
                    vec3(p[other + 1][0], p[other + 1][1], p[other + 1][2]),
                    vec3(p[other][0], p[other][1], p[other][2]));
                const Vec3 offset = vec3_sub(
                    vec3(p[endpoint2][0], p[endpoint2][1], p[endpoint2][2]),
                    vec3(p[other][0], p[other][1], p[other][2]));
                const double length_squared = vec3_squared_norm(edge);
                const double t = length_squared > 0.0
                    ? clamp01(vec3_dot(offset, edge) / length_squared) : 0.0;
                const Vec3 axis = vec3_sub(offset, vec3_mul(edge, t));
                if (swept_axis_separates(points, split, axis, separation))
                    return true;
            }
        }
    }
    return false;
}

IPC_ORIGINAL_HD inline bool segments_intersect(
        const Vec3& a0, const Vec3& a1, const Vec3& b0, const Vec3& b1,
        double eps, LinearCCDGuard& guard, bool coplanar = false,
        double evaluation_error = 0.0, bool known_collinear = false) {
    const Vec3 u = vec3_sub(a1, a0);
    const Vec3 v = vec3_sub(b1, b0);
    const Vec3 w = vec3_sub(b0, a0);
    if (vec3_max_abs(u) == 0.0 || vec3_max_abs(v) == 0.0) {
        guard.uncertain = true;
        return false;
    }
    const Vec3 normal = stable_cross(u, v);
    Vec3 zero = vec3(0.0, 0.0, 0.0);
    const int axis = dominant_drop_axis(normal, zero);
    if (normal.v[axis] == 0.0) {
        const Vec3 offset = stable_cross(w, u);
        if (!known_collinear && vec3_max_abs(offset) != 0.0) {
            const double scale = vec3_stable_norm(w) * vec3_stable_norm(u);
            if (vec3_max_abs(offset) <= kRoundoff * scale
                + (evaluation_error + 2.0 * guard.separation)
                    * vec3_stable_norm(u)) {
                guard.uncertain = true;
            }
            return false;
        }
        int along = 0;
        vec3_max_abs_index(u, along);
        double A = 0.0, B = u.v[along], C = w.v[along],
            D = w.v[along] + v.v[along];
        if (A > B) { const double tmp = A; A = B; B = tmp; }
        if (C > D) { const double tmp = C; C = D; D = tmp; }
        if (A > D || C > B) {
            const double gap = maximum(A - D, C - B);
            const double tolerance =
                eps * max3(absv(B), absv(C), absv(D));
            if (gap <= tolerance + 2.0 * guard.separation)
                guard.uncertain = true;
            return false;
        }
        return true;
    }
    if (!coplanar) {
        const double plane = guard.triple(u, v, w, normal);
        if (plane != 0.0) {
            if (absv(plane) <= 2.0 * guard.separation
                    * vec3_stable_norm(normal)) {
                guard.uncertain = true;
            }
            return false;
        }
    }
    const Vec2 pu = project_drop_axis(u, axis);
    const Vec2 pv = project_drop_axis(v, axis);
    const double denom = orient2(pu, pv);
    const double denom_magnitude =
        absv(pu.x * pv.y) + absv(pu.y * pv.x);
    if (kRoundoff * denom_magnitude
        > kLinearTimePrecision * absv(denom)) guard.uncertain = true;
    const Vec2 pw = project_drop_axis(w, axis);
    const double alpha = orient2(pw, pv) / denom;
    const double beta = orient2(pw, pu) / denom;
    if (!finite(alpha) || !finite(beta)) {
        guard.uncertain = true;
        return false;
    }
    const double outside =
        max4(-alpha, alpha - 1.0, -beta, beta - 1.0);
    if (outside > 0.0) {
        const double coordinate_error = evaluation_error
            + kRoundoff * max4(vec3_max_abs(a0), vec3_max_abs(a1),
                               vec3_max_abs(b0), vec3_max_abs(b1));
        const double bound = (8.0 * coordinate_error
                              + 2.0 * guard.separation)
            * (vec3_lp1(u) + vec3_lp1(v) + vec3_lp1(w)
               + coordinate_error);
        if (outside * absv(denom)
            <= maximum(bound, eps * absv(denom))) guard.uncertain = true;
        return false;
    }
    return true;
}

IPC_ORIGINAL_HD inline bool clip_affine_to_interval(
        double value0, double slope, double lower, double upper,
        double& t_min, double& t_max) {
    if (slope == 0.0) return value0 >= lower && value0 <= upper;
    double lower_time = (lower - value0) / slope;
    double upper_time = (upper - value0) / slope;
    if (lower_time > upper_time) {
        const double tmp = lower_time;
        lower_time = upper_time;
        upper_time = tmp;
    }
    t_min = maximum(t_min, lower_time);
    t_max = minimum(t_max, upper_time);
    return t_min <= t_max;
}

IPC_ORIGINAL_HD inline void insertion_sort(double* values, int count) {
    for (int i = 1; i < count; ++i) {
        const double value = values[i];
        int j = i - 1;
        while (j >= 0 && values[j] > value) {
            values[j + 1] = values[j];
            --j;
        }
        values[j + 1] = value;
    }
}

struct NTPredicate {
    const double (*x)[3];
    const double (*dx)[3];
    LinearCCDGuard* guard;
    double eps;

    IPC_ORIGINAL_HD bool operator()(double t, bool coplanar = false) const {
        const Vec3 point = vec3(x[0][0] + dx[0][0] * t,
                                x[0][1] + dx[0][1] * t,
                                x[0][2] + dx[0][2] * t);
        const Vec3 a = vec3(x[1][0] + dx[1][0] * t,
                            x[1][1] + dx[1][1] * t,
                            x[1][2] + dx[1][2] * t);
        const Vec3 edge1 = vec3_sub(vec3(x[2][0] + dx[2][0] * t,
                                         x[2][1] + dx[2][1] * t,
                                         x[2][2] + dx[2][2] * t), a);
        const Vec3 edge2 = vec3_sub(vec3(x[3][0] + dx[3][0] * t,
                                         x[3][1] + dx[3][1] * t,
                                         x[3][2] + dx[3][2] * t), a);
        const Vec3 offset = vec3_sub(point, a);
        const Vec3 normal = stable_cross(edge1, edge2);
        const Vec3 zero = vec3(0.0, 0.0, 0.0);
        const int axis = dominant_drop_axis(normal, zero);
        const Vec2 e1 = project_drop_axis(edge1, axis);
        const Vec2 e2 = project_drop_axis(edge2, axis);
        const double area = orient2(e1, e2);
        const double area_magnitude =
            absv(e1.x * e2.y) + absv(e1.y * e2.x);
        if (area == 0.0
            || kRoundoff * area_magnitude
                > kLinearTimePrecision * absv(area)) {
            guard->uncertain = true;
            return false;
        }
        if (!coplanar) {
            const double plane = guard->triple(edge1, edge2, offset, normal);
            if (plane != 0.0) {
                if (absv(plane) <= 2.0 * guard->separation
                        * vec3_stable_norm(normal)) {
                    guard->uncertain = true;
                }
                return false;
            }
        }
        const Vec2 q = project_drop_axis(offset, axis);
        const double lambda2 = orient2(q, e2) / area;
        const double lambda3 = orient2(e1, q) / area;
        const double lambda1 =
            orient2(vec2_sub(e1, q), vec2_sub(e2, q)) / area;
        const double min_lambda =
            minimum(lambda1, minimum(lambda2, lambda3));
        if (!finite(min_lambda)) {
            guard->uncertain = true;
            return false;
        }
        if (min_lambda < 0.0) {
            const double position_scale =
                max4(vec3_max_abs(vec3(x[0][0], x[0][1], x[0][2])),
                     vec3_max_abs(vec3(x[1][0], x[1][1], x[1][2])),
                     vec3_max_abs(vec3(x[2][0], x[2][1], x[2][2])),
                     vec3_max_abs(vec3(x[3][0], x[3][1], x[3][2])));
            const double motion_scale =
                max4(vec3_max_abs(vec3(dx[0][0], dx[0][1], dx[0][2])),
                     vec3_max_abs(vec3(dx[1][0], dx[1][1], dx[1][2])),
                     vec3_max_abs(vec3(dx[2][0], dx[2][1], dx[2][2])),
                     vec3_max_abs(vec3(dx[3][0], dx[3][1], dx[3][2])));
            const double coordinate_error =
                kRoundoff * (position_scale + t * motion_scale)
                + guard->time_error * motion_scale;
            const double bound = (8.0 * coordinate_error
                                  + 2.0 * guard->separation)
                * (vec3_lp1(edge1) + vec3_lp1(edge2)
                   + vec3_lp1(offset) + coordinate_error);
            if (-min_lambda * absv(area)
                <= maximum(bound, eps * absv(area))) {
                guard->uncertain = true;
            }
            return false;
        }
        return true;
    }
};

// Local result mirroring the CPU CCDResult fields used by the kernels. It is
// intentionally not the public Result: the public API normalizes no-hit to 1.
struct CCDResult {
    bool collision = false;
    double t = NAN;
};

template <typename Predicate>
IPC_ORIGINAL_HD inline CCDResult first_coplanar_contact(
        double* roots, int count, const Predicate& intersects,
        LinearCCDGuard& guard) {
    insertion_sort(roots, count);
    for (int i = 0; i < count; ++i) {
        const double t = roots[i];
        const bool prior_uncertainty = guard.uncertain;
        const bool boundary_hit = intersects(t);
        const bool boundary_uncertainty = guard.uncertain;
        if (boundary_hit && !boundary_uncertainty) {
            CCDResult result;
            result.collision = true;
            result.t = t;
            return result;
        }
        const double next = i + 1 < count ? roots[i + 1] : 1.0;
        if (next > t) {
            guard.uncertain = prior_uncertainty;
            if (intersects(t + 0.5 * (next - t))) {
                CCDResult result;
                result.collision = true;
                result.t = t;
                return result;
            }
            guard.uncertain = guard.uncertain || boundary_uncertainty;
        }
        if (boundary_hit) {
            CCDResult result;
            result.collision = true;
            result.t = t;
            return result;
        }
    }
    CCDResult result;
    if (intersects(1.0)) {
        result.collision = true;
        result.t = 1.0;
    }
    return result;
}

struct SSPredicate {
    const double (*x)[3];
    const double (*dx)[3];
    LinearCCDGuard* guard;
    double eps;

    IPC_ORIGINAL_HD bool operator()(double t, bool coplanar = false) const {
        const Vec3 a0 = vec3(x[0][0] + dx[0][0] * t,
                             x[0][1] + dx[0][1] * t,
                             x[0][2] + dx[0][2] * t);
        const Vec3 a1 = vec3(x[1][0], x[1][1], x[1][2]);
        const Vec3 b0 = vec3(x[2][0], x[2][1], x[2][2]);
        const Vec3 b1 = vec3(x[3][0], x[3][1], x[3][2]);
        return segments_intersect(
            a0, a1, b0, b1, eps, *guard, coplanar,
            (kRoundoff * t + guard->time_error)
                * vec3_max_abs(vec3(dx[0][0], dx[0][1], dx[0][2])));
    }
};

struct TranslatingPredicate {
    const double (*x)[3];
    const double (*dx)[3];
    LinearCCDGuard* guard;
    double eps;

    IPC_ORIGINAL_HD bool operator()(double t, bool coplanar = false) const {
        const Vec3 a0 = vec3(x[0][0] + dx[0][0] * t,
                             x[0][1] + dx[0][1] * t,
                             x[0][2] + dx[0][2] * t);
        const Vec3 a1 = vec3(x[1][0] + dx[0][0] * t,
                             x[1][1] + dx[0][1] * t,
                             x[1][2] + dx[0][2] * t);
        const Vec3 b0 = vec3(x[2][0], x[2][1], x[2][2]);
        const Vec3 b1 = vec3(x[3][0], x[3][1], x[3][2]);
        return segments_intersect(
            a0, a1, b0, b1, eps, *guard, coplanar,
            (kRoundoff * t + guard->time_error)
                * vec3_max_abs(vec3(dx[0][0], dx[0][1], dx[0][2])));
    }
};

IPC_ORIGINAL_HD inline void add_orientation_root(
        const Vec2& u0, const Vec2& du, const Vec2& v0, const Vec2& dv,
        const Vec2& query0, const Vec2& dquery, double eps,
        double* roots, int& root_count, LinearCCDGuard& guard) {
    double t = orientation_root(u0, du, v0, dv, query0, dquery, guard);
    if (!in_unit_interval(t, eps)) return;
    roots[root_count++] = clamp01(t);
}

IPC_ORIGINAL_HD inline CCDResult node_triangle_linear_ccd(
        const double x[4][3], const double dx[4][3], double eps,
        LinearCCDGuard& guard) {
    CCDResult result;
    const double geometry_relative_eps = maximum(eps, kRoundoff);
    NTPredicate intersects{x, dx, &guard, geometry_relative_eps};

    if (intersects(0.0)) {
        result.collision = true;
        result.t = 0.0;
        return result;
    }
    if (guard.uncertain) return CCDResult{};

    int moving = 0;
    for (int i = 0; i < 4; ++i) {
        const Vec3 motion = vec3(dx[i][0], dx[i][1], dx[i][2]);
        if (vec3_max_abs(motion) != 0.0) moving = i;
    }
    int fixed[3] = {0, 0, 0};
    int count = 0;
    for (int i = 0; i < 4; ++i) if (i != moving) fixed[count++] = i;
    const Vec3 positions_moving = vec3(x[moving][0], x[moving][1],
                                       x[moving][2]);
    const Vec3 fixed0 = vec3(x[fixed[0]][0], x[fixed[0]][1],
                             x[fixed[0]][2]);
    const Vec3 p = vec3_sub(vec3(x[fixed[1]][0], x[fixed[1]][1],
                                 x[fixed[1]][2]), fixed0);
    const Vec3 q = vec3_sub(vec3(x[fixed[2]][0], x[fixed[2]][1],
                                 x[fixed[2]][2]), fixed0);
    const Vec3 plane_normal = stable_cross(p, q);
    const double d = guard.triple(p, q, vec3_sub(positions_moving, fixed0),
                                  plane_normal);
    const double d_error = guard.error;
    const Vec3 motion_moving = vec3(dx[moving][0], dx[moving][1],
                                    dx[moving][2]);
    const double c = guard.triple(p, q, motion_moving, plane_normal);
    const bool coplanar_for_entire_step = c == 0.0 && d == 0.0;
    if (guard.uncertain) return CCDResult{};

    if (c != 0.0) {
        const double t = guard.root(d, c, d_error);
        if (guard.uncertain) return CCDResult{};
        if (in_unit_interval(t, eps)) {
            const double candidate_t = clamp01(t);
            if (intersects(candidate_t, candidate_t == t)) {
                result.collision = true;
                result.t = candidate_t;
            }
        }
    }
    if (!coplanar_for_entire_step) return result;

    const Vec3 normal0 = stable_cross(
        vec3_sub(vec3(x[2][0], x[2][1], x[2][2]),
                 vec3(x[1][0], x[1][1], x[1][2])),
        vec3_sub(vec3(x[3][0], x[3][1], x[3][2]),
                 vec3(x[1][0], x[1][1], x[1][2])));
    const Vec3 normal1 = stable_cross(
        vec3((x[2][0] - x[1][0]) + dx[2][0] - dx[1][0],
             (x[2][1] - x[1][1]) + dx[2][1] - dx[1][1],
             (x[2][2] - x[1][2]) + dx[2][2] - dx[1][2]),
        vec3((x[3][0] - x[1][0]) + dx[3][0] - dx[1][0],
             (x[3][1] - x[1][1]) + dx[3][1] - dx[1][1],
             (x[3][2] - x[1][2]) + dx[3][2] - dx[1][2]));
    const int drop_axis = dominant_drop_axis(normal0, normal1);

    const Vec2 point0 = project_drop_axis(
        vec3(x[0][0], x[0][1], x[0][2]), drop_axis);
    const Vec2 dpoint = project_drop_axis(
        vec3(dx[0][0], dx[0][1], dx[0][2]), drop_axis);
    const Vec2 a0 = project_drop_axis(
        vec3(x[1][0], x[1][1], x[1][2]), drop_axis);
    const Vec2 da = project_drop_axis(
        vec3(dx[1][0], dx[1][1], dx[1][2]), drop_axis);
    const Vec2 b0 = project_drop_axis(
        vec3(x[2][0], x[2][1], x[2][2]), drop_axis);
    const Vec2 db = project_drop_axis(
        vec3(dx[2][0], dx[2][1], dx[2][2]), drop_axis);
    const Vec2 c0 = project_drop_axis(
        vec3(x[3][0], x[3][1], x[3][2]), drop_axis);
    const Vec2 dc = project_drop_axis(
        vec3(dx[3][0], dx[3][1], dx[3][2]), drop_axis);

    double roots[3]{};
    int root_count = 0;
    add_orientation_root(a0, da, b0, db, point0, dpoint, eps,
                         roots, root_count, guard);
    add_orientation_root(b0, db, c0, dc, point0, dpoint, eps,
                         roots, root_count, guard);
    add_orientation_root(c0, dc, a0, da, point0, dpoint, eps,
                         roots, root_count, guard);
    if (guard.uncertain) return CCDResult{};
    return first_coplanar_contact(roots, root_count, intersects, guard);
}

IPC_ORIGINAL_HD inline bool point_between_1d(
        double point0, double dpoint, double a0, double da,
        double b0, double db, double eps, double t) {
    const double ra = (point0 - a0) + t * (dpoint - da);
    const double rb = (point0 - b0) + t * (dpoint - db);
    const double local_length = maximum(absv(ra), absv(rb));
    return ra * rb <= eps * local_length * local_length;
}

IPC_ORIGINAL_HD inline double earliest_point_in_interval_time(
        double point0, double dpoint, double a0, double da,
        double b0, double db, double eps) {
    if (point_between_1d(point0, dpoint, a0, da, b0, db, eps, 0.0))
        return 0.0;

    double roots[2]{};
    int root_count = 0;
    const double slopes[2] = {dpoint - da, dpoint - db};
    const double intercepts[2] = {point0 - a0, point0 - b0};
    for (int i = 0; i < 2; ++i) {
        if (slopes[i] == 0.0) continue;
        double t = -intercepts[i] / slopes[i];
        if (!in_unit_interval(t, eps)) continue;
        roots[root_count++] = clamp01(t);
    }
    insertion_sort(roots, root_count);
    for (int i = 0; i < root_count; ++i) {
        const double t = roots[i];
        if (point_between_1d(point0, dpoint, a0, da, b0, db, eps, t))
            return t;
        const double next_t = i + 1 < root_count ? roots[i + 1] : 1.0;
        if (next_t > t
            && point_between_1d(point0, dpoint, a0, da, b0, db, eps,
                                t + 0.5 * (next_t - t)))
            return t;
    }
    // The CPU function intentionally returns 1.0 in both the final-hit and
    // no-final-hit cases; the caller re-validates the represented geometry.
    return 1.0;
}

IPC_ORIGINAL_HD inline double earliest_collinear_time(
        const double x[4][3], const double dx[4][3], double eps) {
    const Vec3 direction0 = vec3_sub(
        vec3(x[1][0], x[1][1], x[1][2]),
        vec3(x[0][0], x[0][1], x[0][2]));
    const Vec3 direction1 = vec3_sub(direction0,
        vec3(dx[0][0], dx[0][1], dx[0][2]));
    const double ax = absv(direction0.v[0]) + absv(direction1.v[0]);
    const double ay = absv(direction0.v[1]) + absv(direction1.v[1]);
    const double az = absv(direction0.v[2]) + absv(direction1.v[2]);
    int axis = (ax >= ay && ax >= az) ? 0 : (ay >= ax && ay >= az ? 1 : 2);
    const double A0 = x[0][axis], dA = dx[0][axis];
    const double B0 = x[1][axis];
    const double C0 = x[2][axis];
    const double D0 = x[3][axis];
    double best = 1.0;
    best = minimum(best, earliest_point_in_interval_time(
        A0, dA, C0, 0.0, D0, 0.0, eps));
    best = minimum(best, earliest_point_in_interval_time(
        B0, 0.0, C0, 0.0, D0, 0.0, eps));
    best = minimum(best, earliest_point_in_interval_time(
        C0, 0.0, A0, dA, B0, 0.0, eps));
    best = minimum(best, earliest_point_in_interval_time(
        D0, 0.0, A0, dA, B0, 0.0, eps));
    return best;
}

// Faithful port of segment_segment_linear_ccd. The moving endpoint is stored
// in x[0]/dx[0], x[1] is its fixed partner, and [2,3] is the fixed segment.
IPC_ORIGINAL_HD inline CCDResult segment_segment_linear_ccd(
        const double x[4][3], const double dx[4][3], double eps,
        LinearCCDGuard& guard) {
    CCDResult result;
    const double geometry_relative_eps = maximum(eps, kRoundoff);
    SSPredicate intersects{x, dx, &guard, geometry_relative_eps};

    if (intersects(0.0)) {
        result.collision = true;
        result.t = 0.0;
        return result;
    }
    if (guard.uncertain) return CCDResult{};

    const Vec3 vx0 = vec3(x[0][0], x[0][1], x[0][2]);
    const Vec3 vx1 = vec3(x[1][0], x[1][1], x[1][2]);
    const Vec3 vx2 = vec3(x[2][0], x[2][1], x[2][2]);
    const Vec3 vx3 = vec3(x[3][0], x[3][1], x[3][2]);
    const Vec3 vdx0 = vec3(dx[0][0], dx[0][1], dx[0][2]);
    const Vec3 h = vec3_sub(vx1, vx2);
    const Vec3 b = vec3_sub(vx3, vx2);
    const Vec3 y = vec3_sub(vx0, vx2);
    const Vec3 plane_normal = stable_cross(h, b);
    const double d = -guard.triple(h, b, y, plane_normal);
    const double d_error = guard.error;
    const double c = -guard.triple(h, b, vdx0, plane_normal);
    const bool coplanar_for_entire_step = c == 0.0 && d == 0.0;
    if (guard.uncertain) return CCDResult{};

    if (c != 0.0) {
        const double t = guard.root(d, c, d_error);
        if (guard.uncertain) return CCDResult{};
        if (in_unit_interval(t, eps)) {
            const double candidate_t = clamp01(t);
            if (intersects(candidate_t, candidate_t == t)) {
                result.collision = true;
                result.t = candidate_t;
            }
        }
    }
    if (!coplanar_for_entire_step) return result;

    const Vec3 direction0 = vec3_sub(vx1, vx0);
    const Vec3 direction1 = vec3_sub(direction0, vdx0);
    const Vec3 static_direction = vec3_sub(vx3, vx2);
    const Vec3 normal0 = stable_cross(direction0, static_direction);
    const Vec3 normal1 = stable_cross(direction1, static_direction);
    if (vec3_max_abs(normal0) == 0.0 && vec3_max_abs(normal1) == 0.0) {
        const double collinear_t = earliest_collinear_time(x, dx, eps);
        if (intersects(collinear_t)) {
            result.collision = true;
            result.t = collinear_t;
        }
        return result;
    }

    const int drop_axis = dominant_drop_axis(normal0, normal1);
    const Vec2 a0 = project_drop_axis(vx0, drop_axis);
    const Vec2 da = project_drop_axis(vdx0, drop_axis);
    const Vec2 b0 = project_drop_axis(vx1, drop_axis);
    const Vec2 c0 = project_drop_axis(vx2, drop_axis);
    const Vec2 d0 = project_drop_axis(vx3, drop_axis);
    const Vec2 zero = vec2(0.0, 0.0);

    double roots[4]{};
    int root_count = 0;
    add_orientation_root(a0, da, b0, zero, c0, zero, eps,
                         roots, root_count, guard);
    add_orientation_root(a0, da, b0, zero, d0, zero, eps,
                         roots, root_count, guard);
    add_orientation_root(c0, zero, d0, zero, a0, da, eps,
                         roots, root_count, guard);
    add_orientation_root(c0, zero, d0, zero, b0, zero, eps,
                         roots, root_count, guard);
    if (guard.uncertain) return CCDResult{};
    return first_coplanar_contact(roots, root_count, intersects, guard);
}

IPC_ORIGINAL_HD inline CCDResult translating_segment_linear_ccd(
        const double x[4][3], const double dx[4][3], double eps,
        LinearCCDGuard& guard) {
    CCDResult result;
    const double geometry_relative_eps = maximum(eps, kRoundoff);
    TranslatingPredicate intersects{x, dx, &guard, geometry_relative_eps};

    if (intersects(0.0)) {
        result.collision = true;
        result.t = 0.0;
        return result;
    }
    if (guard.uncertain) return CCDResult{};

    const Vec3 x1 = vec3(x[0][0], x[0][1], x[0][2]);
    const Vec3 x2 = vec3(x[1][0], x[1][1], x[1][2]);
    const Vec3 x3 = vec3(x[2][0], x[2][1], x[2][2]);
    const Vec3 x4 = vec3(x[3][0], x[3][1], x[3][2]);
    const Vec3 vdx = vec3(dx[0][0], dx[0][1], dx[0][2]);

    const Vec3 a = vec3_sub(x2, x1);
    const Vec3 b = vec3_sub(x4, x3);
    const Vec3 r = vec3_sub(x3, x1);
    const double a2 = vec3_squared_norm(a);
    const double b2 = vec3_squared_norm(b);
    if (a2 == 0.0 || b2 == 0.0) {
        guard.uncertain = true;
        return result;
    }

    const Vec3 normal = stable_cross(a, b);
    const int drop_axis = dominant_drop_axis(normal, vec3(0.0, 0.0, 0.0));
    const double denominator =
        orient2(project_drop_axis(a, drop_axis),
                project_drop_axis(b, drop_axis));
    const bool parallel = denominator == 0.0;

    if (!parallel) {
        const double d = guard.triple(a, b, r, normal);
        const double d_error = guard.error;
        const double c = -guard.triple(a, b, vdx, normal);
        const bool coplanar_for_entire_step = c == 0.0 && d == 0.0;
        if (guard.uncertain) return CCDResult{};

        if (c != 0.0) {
            const double t = guard.root(d, c, d_error);
            if (guard.uncertain) return CCDResult{};
            if (in_unit_interval(t, eps)) {
                const double candidate_t = clamp01(t);
                if (intersects(candidate_t, candidate_t == t)) {
                    result.collision = true;
                    result.t = candidate_t;
                    return result;
                }
            }
        }
        if (!coplanar_for_entire_step) return result;

        const Vec2 pa = project_drop_axis(a, drop_axis);
        const Vec2 pb = project_drop_axis(b, drop_axis);
        const Vec2 pr = project_drop_axis(r, drop_axis);
        const Vec2 pdx = project_drop_axis(vdx, drop_axis);
        const double alpha0 = orient2(pr, pb) / denominator;
        const double alpha_slope = -orient2(pdx, pb) / denominator;
        const double beta0 = orient2(pr, pa) / denominator;
        const double beta_slope = -orient2(pdx, pa) / denominator;
        double t_min = 0.0;
        double t_max = 1.0;
        if (!clip_affine_to_interval(alpha0, alpha_slope, 0.0, 1.0,
                                     t_min, t_max)
            || !clip_affine_to_interval(beta0, beta_slope, 0.0, 1.0,
                                        t_min, t_max)) {
            return result;
        }
        if (in_unit_interval(t_min, eps)) {
            const double candidate_t = clamp01(t_min);
            if (intersects(candidate_t, true)) {
                result.collision = true;
                result.t = candidate_t;
            }
        }
        return result;
    }

    const Vec3 direction = vec3_stable_norm(a) >= vec3_stable_norm(b) ? a : b;
    const Vec3 transverse_offset = stable_cross(r, direction);
    const Vec3 transverse_velocity = stable_cross(vdx, direction);
    int transverse_axis = 0;
    vec3_max_abs_index(transverse_velocity, transverse_axis);
    const double transverse_speed = absv(transverse_velocity.v[transverse_axis]);
    const bool no_transverse_motion = transverse_speed == 0.0;

    if (!no_transverse_motion) {
        const double offset_size = vec3_max_abs(transverse_offset);
        bool known_collinear = offset_size == 0.0;
        if (!known_collinear) {
            const Vec3 u = vec3_div(transverse_offset, offset_size);
            const Vec3 v = vec3_div(transverse_velocity, transverse_speed);
            const Vec3 mismatch = stable_cross(u, v);
            known_collinear = vec3_max_abs(mismatch) == 0.0;
            if (!known_collinear) {
                if (vec3_max_abs(mismatch) <= kRoundoff)
                    guard.uncertain = true;
                return result;
            }
        }
        const double t = transverse_offset.v[transverse_axis]
            / transverse_velocity.v[transverse_axis];
        if (in_unit_interval(t, eps)) {
            const double candidate_t = clamp01(t);
            if (segments_intersect(
                    vec3(x1.v[0] + vdx.v[0] * candidate_t,
                         x1.v[1] + vdx.v[1] * candidate_t,
                         x1.v[2] + vdx.v[2] * candidate_t),
                    vec3(x2.v[0] + vdx.v[0] * candidate_t,
                         x2.v[1] + vdx.v[1] * candidate_t,
                         x2.v[2] + vdx.v[2] * candidate_t),
                    x3, x4, geometry_relative_eps, guard, false, 0.0,
                    known_collinear && candidate_t == t)) {
                result.collision = true;
                result.t = candidate_t;
            }
        }
        return result;
    }

    const bool collinear_for_entire_step =
        vec3_max_abs(transverse_offset) == 0.0;
    if (!collinear_for_entire_step) return result;

    const double ax = absv(direction.v[0]);
    const double ay = absv(direction.v[1]);
    const double az = absv(direction.v[2]);
    const int axis = (ax >= ay && ax >= az) ? 0 : (ay >= ax && ay >= az ? 1 : 2);
    double moving_min = x1.v[axis];
    double moving_max = x2.v[axis];
    double fixed_min = x3.v[axis];
    double fixed_max = x4.v[axis];
    if (moving_min > moving_max) {
        const double tmp = moving_min; moving_min = moving_max; moving_max = tmp;
    }
    if (fixed_min > fixed_max) {
        const double tmp = fixed_min; fixed_min = fixed_max; fixed_max = tmp;
    }
    const double speed = vdx.v[axis];
    double t_min = 0.0;
    double t_max = 1.0;
    if (!clip_affine_to_interval(moving_min - fixed_max, speed,
                                 -INFINITY, 0.0, t_min, t_max)
        || !clip_affine_to_interval(fixed_min - moving_max, -speed,
                                    -INFINITY, 0.0, t_min, t_max)) {
        return result;
    }
    if (in_unit_interval(t_min, eps)) {
        const double candidate_t = clamp01(t_min);
        if (intersects(candidate_t)) {
            result.collision = true;
            result.t = candidate_t;
        }
    }
    return result;
}

IPC_ORIGINAL_HD inline bool all_zero3(const double a[3]) {
    return a[0] == 0.0 && a[1] == 0.0 && a[2] == 0.0;
}

IPC_ORIGINAL_HD inline bool same_motion(const double a[3],
                                        const double b[3]) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

IPC_ORIGINAL_HD inline int moving_count(const Query& q) {
    int count = 0;
    for (int i = 0; i < 4; ++i) if (!all_zero3(q.motion[i])) ++count;
    return count;
}

IPC_ORIGINAL_HD inline bool is_first_edge_translation(const Query& q) {
    return same_motion(q.motion[0], q.motion[1])
        && all_zero3(q.motion[2]) && all_zero3(q.motion[3]);
}

IPC_ORIGINAL_HD inline bool valid_query(const Query& q) {
    if (!finite(q.eps)) return false;
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k)
            if (!finite(q.positions[i][k]) || !finite(q.motion[i][k]))
                return false;
    const int count = moving_count(q);
    if (q.vertex_face) return count <= 1;
    if (count <= 1) return true;
    // EE may translate its first edge, with both displacement vectors exactly
    // equal and the other edge fixed. No other multi-vertex motion is
    // supported by the CPU original linear backend.
    return count == 2 && is_first_edge_translation(q)
        && !all_zero3(q.motion[0]);
}

IPC_ORIGINAL_HD inline Result invalid_result() {
    return Result{Status::InvalidInput, false, 1.0};
}
IPC_ORIGINAL_HD inline Result needs_exact_result() {
    return Result{Status::NeedsExact, false, 1.0};
}
IPC_ORIGINAL_HD inline Result no_hit_result() {
    return Result{Status::Ok, false, 1.0};
}
IPC_ORIGINAL_HD inline Result hit_result(double t) {
    return Result{Status::Ok, true, t};
}

} // namespace float_detail

// Guarded floating original backend. It never invents a hit and never turns
// uncertainty into a miss: an inconclusive query is returned as NeedsExact.
IPC_ORIGINAL_HD inline Result evaluate_fast(const Query& query) {
    using namespace float_detail;
    if (!valid_query(query)) return invalid_result();

    LinearQuery q(query.positions, query.motion);
    if (q.requires_exact) {
        // Same CPU certificate before requesting the cold exact resolver.
        if (swept_hulls_separated(q, query.vertex_face,
                                  q.validation_distance)) {
            return no_hit_result();
        }
        return needs_exact_result();
    }

    LinearCCDGuard guard{false, q.validation_distance, 0.0, 0.0};
    CCDResult local;
    if (query.vertex_face) {
        local = node_triangle_linear_ccd(q.x, q.dx, query.eps, guard);
    } else if (is_first_edge_translation(query)
               && !all_zero3(query.motion[0])) {
        local = translating_segment_linear_ccd(q.x, q.dx, query.eps, guard);
    } else {
        int moving = 0;
        for (int i = 0; i < 4; ++i)
            if (!all_zero3(query.motion[i])) moving = i;
        if (moving == 0) {
            local = segment_segment_linear_ccd(q.x, q.dx, query.eps, guard);
        } else {
            // Reorder so the moving endpoint is x[0] and its edge partner is
            // x[1], as required by segment_segment_linear_ccd.
            int order[4] = {0, 1, 2, 3};
            if (moving == 1) {
                order[0] = 1; order[1] = 0; order[2] = 2; order[3] = 3;
            } else if (moving == 2) {
                order[0] = 2; order[1] = 3; order[2] = 0; order[3] = 1;
            } else if (moving == 3) {
                order[0] = 3; order[1] = 2; order[2] = 0; order[3] = 1;
            }
            double rx[4][3]{}, rdx[4][3]{};
            for (int i = 0; i < 4; ++i)
                for (int k = 0; k < 3; ++k) {
                    rx[i][k] = q.x[order[i]][k];
                    rdx[i][k] = q.dx[order[i]][k];
                }
            local = segment_segment_linear_ccd(rx, rdx, query.eps, guard);
        }
    }

    if (!guard.uncertain) {
        return local.collision ? hit_result(local.t) : no_hit_result();
    }
    if (swept_hulls_separated(q, query.vertex_face, guard.separation))
        return no_hit_result();
    return needs_exact_result();
}

namespace exact_detail {

template <int C>
using Integer = exact::Integer<C>;
template <int C>
using Arithmetic = exact::Arithmetic<C>;
template <int C>
using Points = exact::Points<C>;

template <int C>
IPC_ORIGINAL_HD inline Result exact_capacity() {
    return Result{Status::CapacityExceeded, false, 1.0};
}

// Compare distance(p)^2 <= validation_squared, with all values on the same
// dyadic integer grid. This is the CPU initial-time membership predicate.
template <int C>
IPC_ORIGINAL_HD inline bool initial_intersects(
        Arithmetic<C>& a, const Points<C>& p, bool face,
        const Integer<C>& validation_squared) {
    const auto distance = exact::distance(a, p, face);
    const auto threshold = a.mul(distance.denominator, validation_squared);
    return a.cmp(distance.numerator, threshold) <= 0;
}

// Event positions are H_i = Q*x_i + N*dx_i = Q*p(N/Q). Their squared distance
// is distance(H)^2 / Q^2, hence the extra Q^2 in the comparison.
template <int C>
IPC_ORIGINAL_HD inline bool event_intersects(
        Arithmetic<C>& a, const Points<C>& x, const Points<C>& motion,
        bool face, const Integer<C>& validation_squared,
        const Integer<C>& numerator, const Integer<C>& denominator) {
    Points<C> event;
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k) {
            event.p[i].v[k] = a.sum(
                a.mul(denominator, x.p[i].v[k]),
                a.mul(numerator, motion.p[i].v[k]));
        }
    const auto distance = exact::distance(a, event, face);
    const auto q2 = a.mul(denominator, denominator);
    const auto threshold = a.mul(a.mul(distance.denominator, q2),
                                 validation_squared);
    return a.cmp(distance.numerator, threshold) <= 0;
}

// r > numerator/denominator, exactly. `r` is a finite nonnegative binary64.
template <int C>
IPC_ORIGINAL_HD inline bool double_greater_ratio(
        Arithmetic<C>& a, double r, const Integer<C>& numerator,
        const Integer<C>& denominator) {
    const exact::Dyadic value = exact::decompose(r);
    if (!value.valid) { a.ok = false; return false; }
    if (!value.sign) return false;
    if (a.sign(numerator) == 0) return true;
    const Integer<C> mantissa(static_cast<std::int64_t>(value.mantissa));
    const auto scaled_denominator = a.mul(mantissa, denominator);
    if (value.exponent >= 0) {
        const auto lhs = exact::shift_left(scaled_denominator, value.exponent);
        if (!lhs.valid) { a.ok = false; return false; }
        return a.cmp(lhs, numerator) > 0;
    }
    const auto rhs = exact::shift_left(numerator, -value.exponent);
    if (!rhs.valid) { a.ok = false; return false; }
    return a.cmp(scaled_denominator, rhs) > 0;
}

// CPU contact_result: nearest conversion, then one representable value toward
// zero when that conversion rounded upward.
template <int C>
IPC_ORIGINAL_HD inline double rounded_down_time(
        Arithmetic<C>& a, const Integer<C>& numerator,
        const Integer<C>& denominator) {
    bool ok = true;
    double rounded = exact::ratio_to_double(numerator, denominator, 0, &ok);
    if (!ok || !float_detail::finite(rounded)) {
        a.ok = false;
        return NAN;
    }
    if (double_greater_ratio(a, rounded, numerator, denominator))
        rounded = float_detail::neighbor(rounded, false);
    return rounded;
}

// Evaluate one candidate event and update the exact earliest hit. Returning
// true means the candidate lies in the CPU event range and intersects.
template <int C>
IPC_ORIGINAL_HD inline bool consider_candidate(
        Arithmetic<C>& a, const Points<C>& x, const Points<C>& motion,
        bool face, const Integer<C>& validation_squared,
        Integer<C> numerator, Integer<C> denominator, bool allow_zero_time,
        Integer<C>& best_numerator, Integer<C>& best_denominator,
        bool& have_best) {
    if (a.sign(denominator) == 0) return false;
    if (a.sign(denominator) < 0) {
        numerator = exact::negate(numerator);
        denominator = exact::negate(denominator);
    }
    const int numerator_sign = a.sign(numerator);
    const bool in_range =
        (allow_zero_time ? numerator_sign >= 0 : numerator_sign > 0)
        && a.cmp(numerator, denominator) <= 0;
    if (!a.ok || !in_range) return false;
    if (!event_intersects(a, x, motion, face, validation_squared,
                          numerator, denominator)) {
        return false;
    }
    if (!a.ok) return false;
    if (!have_best) {
        best_numerator = numerator;
        best_denominator = denominator;
        have_best = true;
        return true;
    }
    const auto lhs = a.mul(numerator, best_denominator);
    const auto rhs = a.mul(best_numerator, denominator);
    if (!a.ok) return true;
    if (a.cmp(lhs, rhs) < 0) {
        best_numerator = numerator;
        best_denominator = denominator;
    }
    return true;
}

template <int C>
IPC_ORIGINAL_HD inline Result resolve_original_exact(const Query& query) {
    using float_detail::all_zero3;
    using float_detail::moving_count;
    using float_detail::valid_query;
    if (!valid_query(query)) return float_detail::invalid_result();

    // Common dyadic grid. Include the exact 1e-10 source-of-truth threshold.
    int unit = 0;
    bool have_unit = false;
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k) {
            const exact::Dyadic p = exact::decompose(query.positions[i][k]);
            const exact::Dyadic d = exact::decompose(query.motion[i][k]);
            if (!p.valid || !d.valid) return float_detail::invalid_result();
            if (p.sign && (!have_unit || p.exponent < unit)) {
                unit = p.exponent; have_unit = true;
            }
            if (d.sign && (!have_unit || d.exponent < unit)) {
                unit = d.exponent; have_unit = true;
            }
        }
    const exact::Dyadic validation_dyadic = exact::decompose(1.0e-10);
    if (!validation_dyadic.valid) return float_detail::invalid_result();
    if (!have_unit || validation_dyadic.exponent < unit)
        unit = validation_dyadic.exponent;

    Arithmetic<C> a;
    Points<C> x, motion, end;
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k) {
            x.p[i].v[k] = exact::from_double<C>(query.positions[i][k], unit);
            motion.p[i].v[k] = exact::from_double<C>(query.motion[i][k], unit);
            a.ok = a.ok && x.p[i].v[k].valid && motion.p[i].v[k].valid;
            end.p[i].v[k] = a.sum(x.p[i].v[k], motion.p[i].v[k]);
        }
    const auto validation = exact::from_double<C>(1.0e-10, unit);
    const auto validation_squared = a.mul(validation, validation);
    if (!a.ok || !validation.valid || !validation_squared.valid)
        return exact_capacity<C>();

    // ORIGINAL initial policy: a positive gap <= 1e-10 is an initial contact.
    const bool initial = initial_intersects(a, x, query.vertex_face,
                                           validation_squared);
    if (!a.ok) return exact_capacity<C>();
    if (initial) return float_detail::hit_result(0.0);
    if (moving_count(query) == 0) return float_detail::no_hit_result();

    const auto intercept = exact::determinant(a, x);
    const auto end_determinant = exact::determinant(a, end);
    const auto slope = a.sub(end_determinant, intercept);
    if (!a.ok) return exact_capacity<C>();

    Integer<C> best_numerator, best_denominator;
    bool have_best = false;

    if (a.sign(slope) != 0) {
        // CPU accepts time >= 0 here; an initial contact was already handled.
        const auto numerator = exact::negate(intercept);
        consider_candidate(a, x, motion, query.vertex_face,
                           validation_squared, numerator, slope, true,
                           best_numerator, best_denominator, have_best);
        if (!a.ok) return exact_capacity<C>();
        if (!have_best) return float_detail::no_hit_result();
        const double time = rounded_down_time(a, best_numerator,
                                              best_denominator);
        if (!a.ok) return exact_capacity<C>();
        return float_detail::hit_result(time);
    }
    if (a.sign(intercept) != 0) return float_detail::no_hit_result();

    // Coplanar step: the original event policy. Event 1 is unconditional;
    // every edge-component and triple-product-component root in (0,1] is
    // retained. We scan all events and retain the exact earliest hit, which is
    // equivalent to the CPU sort/dedup/first-intersecting scan.
    const Integer<C> one(1);
    consider_candidate(a, x, motion, query.vertex_face, validation_squared,
                       one, one, true, best_numerator, best_denominator,
                       have_best);
    if (!a.ok) return exact_capacity<C>();

    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            const auto edge = exact::difference(a, x.p[j], x.p[i]);
            const auto velocity = exact::difference(a, motion.p[j], motion.p[i]);
            if (!a.ok) return exact_capacity<C>();
            for (int axis = 0; axis < 3; ++axis) {
                consider_candidate(a, x, motion, query.vertex_face,
                    validation_squared, exact::negate(edge.v[axis]),
                    velocity.v[axis], false,
                    best_numerator, best_denominator, have_best);
                if (!a.ok) return exact_capacity<C>();
            }
            for (int k = j + 1; k < 4; ++k) {
                const auto other = exact::difference(a, x.p[k], x.p[i]);
                const auto other_velocity =
                    exact::difference(a, motion.p[k], motion.p[i]);
                const auto value = exact::cross(a, edge, other);
                const auto first = exact::cross(a, velocity, other);
                const auto second = exact::cross(a, edge, other_velocity);
                if (!a.ok) return exact_capacity<C>();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto derivative =
                        a.sum(first.v[axis], second.v[axis]);
                    consider_candidate(a, x, motion, query.vertex_face,
                        validation_squared, exact::negate(value.v[axis]),
                        derivative, false,
                        best_numerator, best_denominator, have_best);
                    if (!a.ok) return exact_capacity<C>();
                }
            }
        }
    }
    if (!have_best) return float_detail::no_hit_result();
    const double time = rounded_down_time(a, best_numerator,
                                          best_denominator);
    if (!a.ok) return exact_capacity<C>();
    return float_detail::hit_result(time);
}

} // namespace exact_detail

template <int Capacity>
IPC_ORIGINAL_HD inline Result evaluate_exact(const Query& query) {
    static_assert(Capacity >= 2, "at least two 32-bit limbs required");
    const Result result = exact_detail::resolve_original_exact<Capacity>(query);
#if defined(__CUDA_ARCH__)
    // Every temporary has been destroyed by the time the resolver returned.
    // A failed or leaking worker pool is an explicit resource failure, never a
    // no-hit result. Callers retry with evaluate_exact<2048> after resetting
    // and reinstalling their thread-private pool.
    if (result.status != Status::InvalidInput) {
        const exact::DevicePool* pool = exact::current_pool();
        if (!pool || pool->failed || pool->available != pool->slot_count)
            return exact_detail::exact_capacity<Capacity>();
    }
#endif
    return result;
}

} // namespace original_ccd
} // namespace ipc_gpu

#undef IPC_ORIGINAL_HD
