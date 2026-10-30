#include "original_ccd.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(IPC_GPU_PRODUCTION_REFERENCE)
#include "ccd.h"
#endif

#if defined(IPC_GPU_CUDA_TEST)
// The .cu translation unit owns the real kernel launches and device scratch.
void original_ccd_cuda_evaluate(const ipc_gpu::original_ccd::Query* queries,
                                std::size_t count,
                                ipc_gpu::original_ccd::Result* fast,
                                ipc_gpu::original_ccd::Result* exact,
                                const ipc_gpu::original_ccd::Query*
                                    small_capacity_query,
                                ipc_gpu::original_ccd::test::
                                    DeviceDiagnostics* diagnostics);
#endif

namespace orig = ipc_gpu::original_ccd;
namespace {

void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

struct Entry {
    orig::Query query;
    std::string name;
    bool analytic = false;
    bool analytic_collision = false;
    double analytic_toi = 1.0;
    double analytic_tolerance = 1e-12;
    // Directed exact-rational conversion oracle (not a fast-path assertion).
    bool exact_toi_bits = false;
    double exact_toi_expected = 0.0;
};

struct A3 {
    double v[3]{};
    A3(const double (&a)[3]) { v[0] = a[0]; v[1] = a[1]; v[2] = a[2]; }
    A3(double x, double y, double z) { v[0] = x; v[1] = y; v[2] = z; }
    A3(std::initializer_list<double> values) {
        int i = 0;
        for (double value : values) if (i < 3) v[i++] = value;
    }
};

orig::Query nt(const A3& point, const A3& point_motion,
               const A3& a, const A3& b, const A3& c,
               double eps = 1e-12) {
    orig::Query q;
    q.vertex_face = true;
    q.eps = eps;
    for (int k = 0; k < 3; ++k) {
        q.positions[0][k] = point.v[k];
        q.motion[0][k] = point_motion.v[k];
        q.positions[1][k] = a.v[k];
        q.positions[2][k] = b.v[k];
        q.positions[3][k] = c.v[k];
    }
    return q;
}

orig::Query ss(const A3& a, const A3& da, const A3& b, const A3& db,
               const A3& c, const A3& d,
               double eps = 1e-12) {
    orig::Query q;
    q.vertex_face = false;
    q.eps = eps;
    for (int k = 0; k < 3; ++k) {
        q.positions[0][k] = a.v[k]; q.motion[0][k] = da.v[k];
        q.positions[1][k] = b.v[k]; q.motion[1][k] = db.v[k];
        q.positions[2][k] = c.v[k];
        q.positions[3][k] = d.v[k];
    }
    return q;
}

Entry analytic(orig::Query q, std::string name, bool collision, double toi,
               double tolerance = 1e-12) {
    Entry e;
    e.query = q;
    e.name = std::move(name);
    e.analytic = true;
    e.analytic_collision = collision;
    e.analytic_toi = toi;
    e.analytic_tolerance = tolerance;
    return e;
}

Entry plain(orig::Query q, std::string name) {
    Entry e;
    e.query = q;
    e.name = std::move(name);
    return e;
}

#if defined(IPC_GPU_PRODUCTION_REFERENCE)
bool is_translation(const orig::Query& q) {
    if (q.vertex_face) return false;
    bool first_same = true;
    for (int k = 0; k < 3; ++k)
        first_same = first_same && q.motion[0][k] == q.motion[1][k];
    bool second_fixed = true;
    for (int k = 0; k < 3; ++k)
        second_fixed = second_fixed && q.motion[2][k] == 0.0
            && q.motion[3][k] == 0.0;
    bool moving = false;
    for (int k = 0; k < 3; ++k) moving = moving || q.motion[0][k] != 0.0;
    return first_same && second_fixed && moving;
}

int single_moving_index(const orig::Query& q) {
    int moving = 0;
    for (int i = 0; i < 4; ++i) {
        bool nonzero = false;
        for (int k = 0; k < 3; ++k) nonzero = nonzero || q.motion[i][k] != 0.0;
        if (nonzero) moving = i;
    }
    return moving;
}
#endif

[[maybe_unused]] int ulp_distance(double a, double b) {
    if (a == b) return 0;
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return (std::isnan(a) && std::isnan(b)) ? 0
            : std::numeric_limits<int>::max();
    }
    std::uint64_t ia = 0, ib = 0;
    std::memcpy(&ia, &a, sizeof(ia));
    std::memcpy(&ib, &b, sizeof(ib));
    if (a < 0.0 || b < 0.0)
        return std::numeric_limits<int>::max();
    const std::uint64_t difference = ia > ib ? ia - ib : ib - ia;
    return difference > std::uint64_t(std::numeric_limits<int>::max())
        ? std::numeric_limits<int>::max() : static_cast<int>(difference);
}

// The CPU oracle is the source of truth. For a nonzero CPU TOI use a
// time-relative bound; a true t=0 event admits no absolute slack.
bool toi_matches_cpu(double toi, double cpu_toi) {
    if (!std::isfinite(cpu_toi)) return false;
    if (cpu_toi == 0.0) return toi == 0.0;
    const double tolerance = 1e-12 * std::abs(cpu_toi);
    return std::isfinite(toi) && std::abs(toi - cpu_toi) <= tolerance;
}

orig::Result host_exact(const orig::Query& q, int& retries) {
    orig::Result result = orig::evaluate_exact<256>(q);
    if (result.status == orig::Status::CapacityExceeded) {
        ++retries;
        result = orig::evaluate_exact<2048>(q);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Transfer-representation regression
// ---------------------------------------------------------------------------
// The CUDA harness moves whole Result arrays with a single
//     cudaMemcpy(count * sizeof(Result))
// in each direction. That transfer reads the COMPLETE object representation of
// every record, including the three non-semantic bytes at offsets 5..7 that a
// naive {Status, bool, double} layout leaves as implicit padding. Those bytes
// are now an explicit, always-zero-initialized member, so every record produced
// by this component (host code or the real device kernels) must transport with
// bytes 5..7 fully defined and equal to zero, and the bytewise transport must
// preserve all three semantic fields. Nothing here relaxes a numeric check.
void require_transfer_record(const std::string& name,
                             const orig::Result& result) {
    static_assert(sizeof(orig::Result) == 16,
                  "Result transfer record must stay 16 bytes");
    static_assert(offsetof(orig::Result, status) == 0, "status offset");
    static_assert(offsetof(orig::Result, collision) == 4, "collision offset");
    static_assert(offsetof(orig::Result, reserved) == 5, "reserved offset");
    static_assert(offsetof(orig::Result, toi) == 8, "toi offset");

    unsigned char raw[sizeof(orig::Result)] = {};
    std::memcpy(raw, &result, sizeof(raw));
    for (std::size_t byte = 5; byte < 8; ++byte)
        require(raw[byte] == 0,
                name + ": Result transfer byte " + std::to_string(byte)
                    + " was not initialized by the producing code path");

    // Reproduce exactly what cudaMemcpy does and confirm the bytewise copy
    // preserves every semantic field.
    orig::Result transported;
    std::memcpy(&transported, raw, sizeof(transported));
    require(transported.status == result.status
                && transported.collision == result.collision
                && transported.toi == result.toi,
            name + ": bytewise Result transfer changed a semantic field");
}

// Host-side simulation of the whole-array device transfer, in both directions.
void transfer_representation_tests(const std::vector<orig::Result>& fast,
                                   const std::vector<orig::Result>& exact) {
    require(!fast.empty() && fast.size() == exact.size(),
            "transfer regression needs paired non-empty result arrays");
    for (std::size_t i = 0; i < fast.size(); ++i) {
        require_transfer_record("host fast[" + std::to_string(i) + "]", fast[i]);
        require_transfer_record("host exact[" + std::to_string(i) + "]",
                                exact[i]);
    }

    std::vector<unsigned char> wire(exact.size() * sizeof(orig::Result));
    std::memcpy(wire.data(), exact.data(), wire.size());
    for (std::size_t i = 0; i < exact.size(); ++i) {
        const unsigned char* record = wire.data() + i * sizeof(orig::Result);
        for (std::size_t byte = 5; byte < 8; ++byte)
            require(record[byte] == 0,
                    "wire exact[" + std::to_string(i)
                        + "]: reserved byte not defined in the transferred "
                          "record");
    }
    std::vector<orig::Result> round_trip(exact.size());
    std::memcpy(round_trip.data(), wire.data(), wire.size());
    for (std::size_t i = 0; i < exact.size(); ++i)
        require(round_trip[i].status == exact[i].status
                    && round_trip[i].collision == exact[i].collision
                    && round_trip[i].toi == exact[i].toi,
                "wire exact[" + std::to_string(i)
                    + "]: array transfer changed a semantic field");

    // The default/3-argument forms must both be transfer-clean: the old
    // `Result` default case, and the exact brace spelling the device wrapper
    // uses for explicit resource failures.
    const orig::Result default_case;
    require_transfer_record("default Result", default_case);
    require(default_case.status == orig::Status::Ok
                && !default_case.collision && default_case.toi == 1.0,
            "default Result must stay Ok / no hit / toi 1");
    require_transfer_record(
        "capacity Result",
        orig::Result{orig::Status::CapacityExceeded, false, 1.0});
}

struct CpuOutcome {
    bool collision = false;
    double toi = std::numeric_limits<double>::quiet_NaN();
};

#if defined(IPC_GPU_PRODUCTION_REFERENCE)
CpuOutcome cpu_original(const orig::Query& q) {
    Vec3 p[4], d[4];
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k) {
            p[i][k] = q.positions[i][k];
            d[i][k] = q.motion[i][k];
        }
    CCDResult result;
    if (q.vertex_face) {
        result = node_triangle_only_one_node_moves(
            p[0], d[0], p[1], d[1], p[2], d[2], p[3], d[3],
            q.eps, false, true);
    } else if (is_translation(q)) {
        result = segment_segment_same_displacement_linear_ccd(
            p[0], d[0], p[1], d[1], p[2], p[3], q.eps, true);
    } else {
        int order[4] = {0, 1, 2, 3};
        const int moving = single_moving_index(q);
        if (moving == 1) {
            order[0] = 1; order[1] = 0;
        } else if (moving == 2) {
            order[0] = 2; order[1] = 3; order[2] = 0; order[3] = 1;
        } else if (moving == 3) {
            order[0] = 3; order[1] = 2; order[2] = 0; order[3] = 1;
        }
        result = segment_segment_only_one_node_moves(
            p[order[0]], d[order[0]], p[order[1]],
            p[order[2]], p[order[3]], q.eps, false, true);
    }
    CpuOutcome outcome;
    outcome.collision = result.collision;
    if (result.collision) outcome.toi = result.t;
    return outcome;
}
#endif

// Compare BOTH caller-visible routes against the real CPU original dispatcher:
//   * fast.status==Ok is consumed directly by a future caller;
//   * fast.status==NeedsExact hands the same query to evaluate_exact.
// The exact entry is additionally checked against its own rational contract.
CpuOutcome compare_route_with_cpu(const Entry& entry,
                                  const orig::Result& fast,
                                  const orig::Result& exact) {
#if defined(IPC_GPU_PRODUCTION_REFERENCE)
    const CpuOutcome cpu = cpu_original(entry.query);
    require(exact.status == orig::Status::Ok,
            entry.name + ": exact status must be Ok");
    require(exact.collision == cpu.collision,
            entry.name + ": exact collision bool differs from CPU original");

    const orig::Result& route =
        fast.status == orig::Status::Ok ? fast : exact;
    require(route.status == orig::Status::Ok,
            entry.name + ": caller route must resolve to Ok");
    require(route.collision == cpu.collision,
            entry.name + ": caller route collision differs from CPU original");
    if (cpu.collision) {
        // The CPU dispatcher may return its guarded fast root, while the exact
        // entry returns the rational original root. They can differ by ulps,
        // but the time-relative bound is tight and t=0 is exact.
        require(toi_matches_cpu(route.toi, cpu.toi),
                entry.name + ": caller route TOI differs from CPU original");
        require(toi_matches_cpu(exact.toi, cpu.toi),
                entry.name + ": exact TOI differs from CPU original");
    } else {
        require(route.toi == 1.0,
                entry.name + ": caller route no-hit TOI must be 1");
        require(exact.toi == 1.0,
                entry.name + ": exact no-hit TOI must be 1");
    }
    return cpu;
#else
    (void)entry;
    (void)fast;
    (void)exact;
    return CpuOutcome{};
#endif
}

void compare_analytic(const Entry& entry, const orig::Result& exact) {
    if (!entry.analytic) return;
    require(exact.status == orig::Status::Ok,
            entry.name + ": analytic exact status");
    require(exact.collision == entry.analytic_collision,
            entry.name + ": analytic collision mismatch");
    if (entry.analytic_collision) {
        const double tolerance = entry.analytic_tolerance;
        require(std::abs(exact.toi - entry.analytic_toi) <= tolerance,
                entry.name + ": analytic TOI mismatch");
    } else {
        require(exact.toi == 1.0, entry.name + ": no-hit TOI must be 1");
    }
}

void compare_directed_exact_bits(const Entry& entry, const orig::Result& exact,
                                 const char* route) {
    if (!entry.exact_toi_bits) return;
    require(exact.status == orig::Status::Ok,
            entry.name + ": directed exact status on " + route);
    require(exact.collision,
            entry.name + ": directed exact must be a collision on " + route);
    require(exact.toi == entry.exact_toi_expected,
            entry.name + ": directed exact TOI must be the nearest-down "
            "binary64 of 1/10 on " + route);
}

// Deterministic dyadic generator: every coordinate is k/8 for a small integer
// k, so all corpus inputs and conditioned displacements are exact binary64
// values with a bounded exponent span. Conditioned vectors reach the other
// primitive at t_star=1/2, 1/4 or 1; pattern 3 leaves the motion random.
double dyadic_coord(std::mt19937_64& rng) {
    const std::int64_t value = static_cast<std::int64_t>(rng() % 97) - 48;
    return static_cast<double>(value) / 8.0;
}

bool motion_nonzero(const double motion[3]) {
    return motion[0] != 0.0 || motion[1] != 0.0 || motion[2] != 0.0;
}

double contact_time(int pattern) {
    if (pattern == 0) return 0.5;
    if (pattern == 1) return 0.25;
    return 1.0;
}

orig::Query dyadic_nt(std::mt19937_64& rng, int role, int pattern) {
    double p[4][3];
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k) p[i][k] = dyadic_coord(rng);
    double motion[3] = {0.0, 0.0, 0.0};
    if (pattern < 3) {
        const double t_star = contact_time(pattern);
        double target[3];
        if (role == 0) {
            // Point reaches a strictly interior barycentric combination of the
            // fixed triangle at t_star.
            for (int k = 0; k < 3; ++k)
                target[k] = 0.5 * p[1][k] + 0.25 * p[2][k]
                    + 0.25 * p[3][k];
        } else {
            // Moving triangle vertex reaches a position at which the fixed
            // query point is interior with barycentric (1/2, 1/4, 1/4).
            const int f1 = role == 1 ? 2 : 1;
            const int f2 = role == 2 ? 3 : (role == 3 ? 2 : 3);
            for (int k = 0; k < 3; ++k)
                target[k] = 2.0 * p[0][k] - 0.5 * p[f1][k]
                    - 0.5 * p[f2][k];
        }
        for (int k = 0; k < 3; ++k)
            motion[k] = (target[k] - p[role][k]) / t_star;
    } else {
        for (int k = 0; k < 3; ++k) motion[k] = 2.0 * dyadic_coord(rng);
    }
    if (!motion_nonzero(motion)) motion[0] = 0.125;

    orig::Query q;
    q.vertex_face = true;
    q.eps = 1e-12;
    for (int k = 0; k < 3; ++k)
        for (int i = 0; i < 4; ++i) q.positions[i][k] = p[i][k];
    for (int k = 0; k < 3; ++k) q.motion[role][k] = motion[k];
    return q;
}

orig::Query dyadic_ss_one(std::mt19937_64& rng, int role, int pattern) {
    double p[4][3];
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k) p[i][k] = dyadic_coord(rng);
    double motion[3] = {0.0, 0.0, 0.0};
    if (pattern < 3) {
        const double t_star = contact_time(pattern);
        const double weight = pattern == 0 ? 0.5 : (pattern == 1 ? 0.25 : 0.75);
        const int other = role < 2 ? 2 : 0;
        double target[3];
        for (int k = 0; k < 3; ++k)
            target[k] = p[other][k]
                + weight * (p[other + 1][k] - p[other][k]);
        for (int k = 0; k < 3; ++k)
            motion[k] = (target[k] - p[role][k]) / t_star;
    } else {
        for (int k = 0; k < 3; ++k) motion[k] = 2.0 * dyadic_coord(rng);
    }
    if (!motion_nonzero(motion)) motion[1] = 0.125;

    orig::Query q;
    q.vertex_face = false;
    q.eps = 1e-12;
    for (int k = 0; k < 3; ++k)
        for (int i = 0; i < 4; ++i) q.positions[i][k] = p[i][k];
    for (int k = 0; k < 3; ++k) q.motion[role][k] = motion[k];
    return q;
}

orig::Query dyadic_ss_translation(std::mt19937_64& rng, int pattern) {
    double p[4][3];
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k) p[i][k] = dyadic_coord(rng);
    double motion[3] = {0.0, 0.0, 0.0};
    if (pattern < 3) {
        const double t_star = contact_time(pattern);
        const double weight = pattern == 0 ? 0.5 : (pattern == 1 ? 0.25 : 0.75);
        double target[3];
        for (int k = 0; k < 3; ++k)
            target[k] = p[2][k] + weight * (p[3][k] - p[2][k]);
        for (int k = 0; k < 3; ++k)
            motion[k] = (target[k] - p[0][k]) / t_star;
    } else {
        for (int k = 0; k < 3; ++k) motion[k] = 2.0 * dyadic_coord(rng);
    }
    if (!motion_nonzero(motion)) motion[0] = 0.125;

    orig::Query q;
    q.vertex_face = false;
    q.eps = 1e-12;
    for (int k = 0; k < 3; ++k)
        for (int i = 0; i < 4; ++i) q.positions[i][k] = p[i][k];
    for (int k = 0; k < 3; ++k) {
        q.motion[0][k] = motion[k];
        q.motion[1][k] = motion[k];
    }
    return q;
}

std::vector<Entry> fixtures() {
    std::vector<Entry> entries;
    const double origin[3] = {0, 0, 0};
    const double bx[3] = {1, 0, 0};
    const double by[3] = {0, 1, 0};
    const double zero[3] = {0, 0, 0};

    {
        const double point[3] = {0.25, 0.25, 1.0};
        const double motion[3] = {0, 0, -2.0};
        entries.push_back(analytic(
            nt(point, motion, origin, bx, by), "NT point down cross",
            true, 0.5));
    }
    {
        const double point[3] = {0.25, 0.25, 1.0};
        const double motion[3] = {0, 0, 2.0};
        entries.push_back(analytic(
            nt(point, motion, origin, bx, by), "NT point up miss",
            false, 1.0));
    }
    {
        const double point[3] = {0.25, 0.25, 5e-11};
        const double motion[3] = {0, 0, 1.0};
        entries.push_back(analytic(
            nt(point, motion, origin, bx, by), "NT positive gap <=1e-10",
            true, 0.0));
    }
    {
        // The floating local frame underflows this displacement component,
        // so the fast path must request exact evaluation rather than guess.
        const double point[3] = {0.25, 0.25,
                                 std::numeric_limits<double>::denorm_min()};
        const double motion[3] = {0, 0, 0};
        entries.push_back(analytic(
            nt(point, motion, origin, bx, by), "NT NeedsExact initial band",
            true, 0.0));
    }
    {
        const double point[3] = {0.25, 0.25, 0.0};
        const double motion[3] = {0, 0, 0};
        entries.push_back(analytic(
            nt(point, motion, origin, bx, by), "NT initial boundary",
            true, 0.0));
    }
    {
        const double point[3] = {-1.0, 0.3, 0.0};
        const double motion[3] = {2.0, 0, 0};
        entries.push_back(analytic(
            nt(point, motion, origin, bx, by), "NT coplanar crossing",
            true, 0.5));
    }
    {
        const double point[3] = {-1.0, 2.0, 0.0};
        const double motion[3] = {2.0, 0, 0};
        entries.push_back(analytic(
            nt(point, motion, origin, bx, by), "NT coplanar miss",
            false, 1.0));
    }
    {
        const double point[3] = {1.0, 1.0, 1.0};
        const double motion[3] = {-1.0, -2.0, -2.0};
        entries.push_back(analytic(
            nt(point, motion, origin, bx, by), "NT noncoplanar diagonal",
            true, 0.5));
    }
    {
        const double point[3] = {0.25, 0.25, 1.0};
        const double motion[3] = {0, 0, -2.0};
        entries.push_back(plain(
            nt(point, motion, origin, origin, origin), "NT degenerate triangle"));
    }

    // All four NT moving roles. These are compared against the real CPU
    // original dispatcher, including its exact fallback.
    for (int role = 0; role < 4; ++role) {
        const double point[3] = {0.2, 0.2, 0.5};
        const double triangle[3][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        orig::Query q = nt(point, zero, triangle[0], triangle[1], triangle[2]);
        const double motion[3] = {0.05 * (role + 1), -0.03, 0.4};
        for (int k = 0; k < 3; ++k) q.motion[role][k] = motion[k];
        entries.push_back(plain(q, "NT moving role " + std::to_string(role)));
    }

    // SS one-vertex crossing and misses.
    {
        const double a[3] = {0, 0, -1};
        const double da[3] = {0, 0, 2};
        const double b[3] = {1, 0, -1};
        const double c[3] = {0, 0, 0};
        const double d[3] = {0, 1, 0};
        entries.push_back(analytic(
            ss(a, da, b, zero, c, d), "SS endpoint crossing", true, 0.5));
    }
    {
        const double a[3] = {0, 0, -1};
        const double da[3] = {0, 0, -2};
        const double b[3] = {1, 0, -1};
        const double c[3] = {0, 0, 0};
        const double d[3] = {0, 1, 0};
        entries.push_back(analytic(
            ss(a, da, b, zero, c, d), "SS endpoint miss", false, 1.0));
    }
    {
        const double a[3] = {0, 0, 5e-11};
        const double da[3] = {0, 0, 1.0};
        const double b[3] = {1, 0, 5e-11};
        const double c[3] = {0, 0, 0};
        const double d[3] = {1, 0, 0};
        entries.push_back(analytic(
            ss(a, da, b, zero, c, d), "SS positive gap <=1e-10",
            true, 0.0));
    }
    {
        const double a[3] = {0, 0, 0};
        const double da[3] = {0, 0, 0};
        const double b[3] = {1, 0, 0};
        const double c[3] = {0, 0, 0};
        const double d[3] = {0, 2, 0};
        entries.push_back(analytic(
            ss(a, da, b, zero, c, d), "SS initial endpoint touch", true, 0.0));
    }
    {
        const double a[3] = {0, 0, 1};
        const double da[3] = {0, 0, 0};
        const double b[3] = {1, 0, 1};
        const double c[3] = {0, 1, 0};
        const double d[3] = {1, 1, 0};
        entries.push_back(analytic(
            ss(a, da, b, zero, c, d), "SS parallel separated", false, 1.0));
    }
    {
        // Coplanar collinear overlap in 1D, no transverse motion.
        const double a[3] = {-2, 0, 0};
        const double da[3] = {4, 0, 0};
        const double b[3] = {-1, 0, 0};
        const double c[3] = {0, 0, 0};
        const double d[3] = {1, 0, 0};
        entries.push_back(plain(
            ss(a, da, b, zero, c, d), "SS collinear translation"));
    }

    // Translating first edge: both first-edge vertices share EXACT motion.
    {
        const double a[3] = {-1, -2, 0};
        const double da[3] = {0, 4, 0};
        const double b[3] = {1, -2, 0};
        const double c[3] = {0, 0, 0};
        const double d[3] = {0, 2, 0};
        entries.push_back(analytic(
            ss(a, da, b, da, c, d), "SS translating edge crossing",
            true, 0.5));
    }
    {
        const double a[3] = {-1, -2, 0};
        const double da[3] = {0, 4, 0};
        const double b[3] = {1, -2, 0};
        const double c[3] = {0, 3, 0};
        const double d[3] = {0, 4, 0};
        entries.push_back(analytic(
            ss(a, da, b, da, c, d), "SS translating edge miss",
            false, 1.0));
    }

    // All four EE one-moving-vertex roles. In each one the moving endpoint
    // reaches the midpoint of the other segment at t=0.5.
    for (int role = 0; role < 4; ++role) {
        const double moving[3] = {0.5, 1.0, 1.0};
        const double partner[3] = {0.0, 1.0, 0.0};
        const double fixed_a[3] = {0.0, 0.0, 0.0};
        const double fixed_b[3] = {1.0, 0.0, 0.0};
        double p[4][3];
        for (int k = 0; k < 3; ++k) {
            if (role == 0) {
                p[0][k] = moving[k]; p[1][k] = partner[k];
                p[2][k] = fixed_a[k]; p[3][k] = fixed_b[k];
            } else if (role == 1) {
                p[0][k] = partner[k]; p[1][k] = moving[k];
                p[2][k] = fixed_a[k]; p[3][k] = fixed_b[k];
            } else if (role == 2) {
                p[0][k] = fixed_a[k]; p[1][k] = fixed_b[k];
                p[2][k] = moving[k]; p[3][k] = partner[k];
            } else {
                p[0][k] = fixed_a[k]; p[1][k] = fixed_b[k];
                p[2][k] = partner[k]; p[3][k] = moving[k];
            }
        }
        orig::Query q = ss(p[0], zero, p[1], zero, p[2], p[3]);
        q.motion[role][0] = 0.0;
        q.motion[role][1] = -2.0;
        q.motion[role][2] = -2.0;
        entries.push_back(analytic(
            q, "EE moving role " + std::to_string(role), true, 0.5));
    }

    // Translation of the whole scene, exact power-of-two scaling, and the
    // full binary64 exponent range.
    const double offset[3] = {1e9, -2e9, 3e8};
    {
        double point[3] = {0.25, 0.25, 1.0};
        double motion[3] = {0, 0, -2.0};
        double o[3], one[3], two[3];
        for (int k = 0; k < 3; ++k) {
            o[k] = origin[k] + offset[k];
            one[k] = bx[k] + offset[k];
            two[k] = by[k] + offset[k];
            point[k] += offset[k];
        }
        entries.push_back(analytic(
            nt(point, motion, o, one, two), "NT translated world", true, 0.5));
    }
    for (int exponent : {-200, -80, 80, 200}) {
        double point[3] = {0.25, 0.25, 1.0};
        double motion[3] = {0, 0, -2.0};
        double o[3], one[3], two[3];
        for (int k = 0; k < 3; ++k) {
            point[k] = std::ldexp(point[k], exponent);
            motion[k] = std::ldexp(motion[k], exponent);
            o[k] = std::ldexp(origin[k], exponent);
            one[k] = std::ldexp(bx[k], exponent);
            two[k] = std::ldexp(by[k], exponent);
        }
        entries.push_back(plain(
            nt(point, motion, o, one, two),
            "NT scaled 2^" + std::to_string(exponent)));
    }
    {
        // Deliberately mixed extreme magnitudes. The floating local frame
        // cannot be formed; the exact resolver must still be explicit.
        orig::Query q = nt(origin, zero, origin, bx, by);
        q.positions[0][0] = std::numeric_limits<double>::max();
        q.positions[1][0] = -std::numeric_limits<double>::max();
        q.positions[2][0] = std::numeric_limits<double>::denorm_min();
        q.positions[3][1] = -std::numeric_limits<double>::denorm_min();
        entries.push_back(plain(q, "NT full binary64 range miss"));
    }
    {
        orig::Query q = ss(origin, zero, bx, zero, by, zero);
        q.positions[0][0] = std::numeric_limits<double>::max();
        q.positions[1][0] = -std::numeric_limits<double>::max();
        q.positions[2][0] = std::numeric_limits<double>::denorm_min();
        q.positions[3][1] = -std::numeric_limits<double>::denorm_min();
        entries.push_back(plain(q, "EE full binary64 range"));
    }

    // Directed rational TOI conversion. The exact root is 1/10, whose nearest
    // binary64 (0.1) is ABOVE 1/10; contact_result must therefore return
    // nextafter(0.1, 0) exactly. This is asserted for the exact entry only:
    // the guarded floating dispatcher legitimately returns 0.1.
    {
        const double point[3] = {0.25, 0.25, 1.0};
        const double motion[3] = {0, 0, -10.0};
        const double expected = std::nextafter(0.1, 0.0);
        Entry entry = analytic(
            nt(point, motion, origin, bx, by),
            "NT directed exact 1/10 nearest-down", true, expected, 0.0);
        entry.exact_toi_bits = true;
        entry.exact_toi_expected = expected;
        entries.push_back(entry);
    }
    // Earliest of several coplanar incidence events: the point enters the
    // triangle at t=1/3 and leaves later. The earliest event must win.
    {
        const double point[3] = {-1.0, 0.25, 0.0};
        const double motion[3] = {3.0, 0, 0};
        entries.push_back(analytic(
            nt(point, motion, origin, bx, by),
            "NT coplanar multiple events earliest", true, 1.0 / 3.0));
    }
    // Explicit t=1 endpoint: contact occurs exactly at the end of the step.
    {
        const double point[3] = {0.25, 0.25, 1.0};
        const double motion[3] = {0, 0, -1.0};
        entries.push_back(analytic(
            nt(point, motion, origin, bx, by),
            "NT endpoint contact t=1", true, 1.0));
    }

    // Original 1e-10 initial-membership band crossed with tiny and huge
    // speeds. A positive gap at or below 1e-10 is t=0; larger gaps follow the
    // affine root exactly. Upward motion is a miss unless the initial band
    // already contains the point.
    for (double gap : {1.0, 1e-9, 1e-11}) {
        for (double speed : {-2.0, -1e10, 2.0}) {
            const double point[3] = {0.25, 0.25, gap};
            const double motion[3] = {0, 0, speed};
            const bool collision = gap <= 1e-10 || speed < 0.0;
            const double toi = gap <= 1e-10 ? 0.0
                : (speed < 0.0 ? gap / (-speed) : 1.0);
            // Analytic TOI check follows the same rule as the CPU bound: a
            // true zero event must be exactly zero; nonzero times are
            // time-relative.
            const double tolerance =
                toi == 0.0 ? 0.0 : 1e-12 * std::abs(toi);
            entries.push_back(analytic(
                nt(point, motion, origin, bx, by),
                "NT gap " + std::to_string(gap) + " speed "
                    + std::to_string(speed),
                collision, toi, tolerance));
        }
    }

    // 1536 deterministic conditioned dyadic vectors:
    //   * 512 NT, evenly covering all four moving roles;
    //   * 512 EE one-vertex, evenly covering all four moving roles;
    //   * 512 translating first-edge EE queries.
    // Each block uses the four contact-time/random patterns evenly. The exact
    // CPU original dispatcher is the oracle; only the analytic fixtures above
    // hard-code a TOI.
    std::mt19937_64 dyadic_rng(424242);
    for (int n = 0; n < 512; ++n) {
        const int role = n % 4;
        const int pattern = (n / 4) % 4;
        entries.push_back(plain(
            dyadic_nt(dyadic_rng, role, pattern),
            "dyadic NT n=" + std::to_string(n)));
    }
    for (int n = 0; n < 512; ++n) {
        const int role = n % 4;
        const int pattern = (n / 4) % 4;
        entries.push_back(plain(
            dyadic_ss_one(dyadic_rng, role, pattern),
            "dyadic EE n=" + std::to_string(n) + " role="
                + std::to_string(role)));
    }
    for (int n = 0; n < 512; ++n) {
        const int pattern = n % 4;
        entries.push_back(plain(
            dyadic_ss_translation(dyadic_rng, pattern),
            "dyadic EE translation n=" + std::to_string(n)));
    }

    // Deterministic conditioned random vectors: force contacts at t=0.5 for a
    // large fraction, while keeping misses, degeneracies and both primitive
    // kinds in the mix.
    std::mt19937_64 rng(20261030);
    std::uniform_real_distribution<double> random(-1.5, 1.5);
    for (int n = 0; n < 160; ++n) {
        const bool face = n % 2 == 0;
        if (face) {
            double p[4][3];
            for (int i = 0; i < 4; ++i)
                for (int k = 0; k < 3; ++k) p[i][k] = random(rng);
            const int moving = n % 4;
            double motion[3];
            if (n % 3 == 0 && moving == 0) {
                for (int k = 0; k < 3; ++k)
                    motion[k] = 2.0 * ((p[1][k] + p[2][k] + p[3][k]) / 3.0
                                       - p[0][k]);
            } else {
                for (int k = 0; k < 3; ++k) motion[k] = 2.5 * random(rng);
            }
            orig::Query q = nt(p[0], motion, p[1], p[2], p[3]);
            for (int k = 0; k < 3; ++k) {
                q.motion[0][k] = 0.0;
                q.motion[moving][k] = motion[k];
            }
            entries.push_back(plain(
                q, "random NT " + std::to_string(n)));
        } else {
            double p[4][3];
            for (int i = 0; i < 4; ++i)
                for (int k = 0; k < 3; ++k) p[i][k] = random(rng);
            if (n % 5 == 1) {
                for (int k = 0; k < 3; ++k) p[1][k] = p[0][k] + 1.0;
                for (int k = 0; k < 3; ++k) p[3][k] = p[2][k] + 1.0;
                double motion[3];
                for (int k = 0; k < 3; ++k)
                    motion[k] = 2.0 * ((p[2][k] + p[3][k]) / 2.0
                                       - (p[0][k] + p[1][k]) / 2.0);
                entries.push_back(plain(
                    ss(p[0], motion, p[1], motion, p[2], p[3]),
                    "random translating EE " + std::to_string(n)));
            } else {
                const int moving = n % 4;
                double motion[3];
                if (n % 3 == 1) {
                    const int other_edge = moving < 2 ? 2 : 0;
                    for (int k = 0; k < 3; ++k)
                        motion[k] = 2.0 * ((p[other_edge][k]
                            + p[other_edge + 1][k]) / 2.0 - p[moving][k]);
                } else {
                    for (int k = 0; k < 3; ++k) motion[k] = 2.5 * random(rng);
                }
                orig::Query q = ss(p[0], zero, p[1], zero, p[2], p[3]);
                for (int k = 0; k < 3; ++k) q.motion[moving][k] = motion[k];
                entries.push_back(plain(
                    q, "random EE " + std::to_string(n)));
            }
        }
    }
    return entries;
}

void invalid_input_tests() {
    orig::Query q = nt({0.25, 0.25, 1.0}, {0, 0, -1}, {0, 0, 0},
                       {1, 0, 0}, {0, 1, 0});
    q.motion[1][0] = 1.0;
    require(orig::evaluate_fast(q).status == orig::Status::InvalidInput,
            "NT two moving vertices must be InvalidInput");
    require(orig::evaluate_exact<64>(q).status == orig::Status::InvalidInput,
            "exact NT two moving vertices must be InvalidInput");

    orig::Query e = ss({0, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0},
                       {0, 1, 0}, {0, 2, 0});
    e.motion[0][0] = 1.0;
    e.motion[1][0] = 2.0;
    require(orig::evaluate_fast(e).status == orig::Status::InvalidInput,
            "unmatched EE motion must be InvalidInput");
    require(orig::evaluate_exact<64>(e).status == orig::Status::InvalidInput,
            "unmatched EE exact motion must be InvalidInput");

    orig::Query n = nt({0, 0, 0}, {0, 0, 0}, {0, 0, 0},
                       {1, 0, 0}, {0, 1, 0});
    n.positions[0][0] = std::numeric_limits<double>::infinity();
    require(orig::evaluate_fast(n).status == orig::Status::InvalidInput,
            "nonfinite fast input");
    require(orig::evaluate_exact<64>(n).status == orig::Status::InvalidInput,
            "nonfinite exact input");
}

orig::Query capacity_crossing_query() {
    return nt({0.25, 0.25, 1.0}, {0, 0, -2.0}, {0, 0, 0},
              {1, 0, 0}, {0, 1, 0});
}

void capacity_and_fallback_tests(std::vector<Entry>& entries) {
    // A normal noncoplanar crossing already needs more than two 32-bit limbs
    // for the exact distance polynomial. Capacity exhaustion is explicit and
    // is retried, never turned into a miss.
    Entry crossing = analytic(
        capacity_crossing_query(),
        "capacity retry crossing", true, 0.5);
    require(orig::evaluate_exact<2>(crossing.query).status
            == orig::Status::CapacityExceeded,
            "tiny exact capacity must fail explicitly");
    int retries = 0;
    const orig::Result retried = host_exact(crossing.query, retries);
    require(retried.status == orig::Status::Ok && retried.collision,
            "capacity retry must resolve");
    entries.push_back(crossing);

    // A query whose floating local frame cannot be formed must report
    // NeedsExact rather than a guess. The exact resolver is the next stage.
    orig::Query mixed = nt({0, 0, 0}, {0, 0, 0}, {0, 0, 0},
                            {1, 0, 0}, {0, 1, 0});
    mixed.positions[0][0] = std::numeric_limits<double>::max();
    mixed.positions[1][0] = -std::numeric_limits<double>::max();
    mixed.positions[2][0] = std::numeric_limits<double>::denorm_min();
    const orig::Result fast = orig::evaluate_fast(mixed);
    require(fast.status == orig::Status::NeedsExact
            || (fast.status == orig::Status::Ok && !fast.collision),
            "extreme fast result must be closed");
    if (fast.status == orig::Status::NeedsExact) {
        int local_retries = 0;
        const orig::Result exact = host_exact(mixed, local_retries);
        require(exact.status == orig::Status::Ok,
                "NeedsExact must be resolved by exact");
    }
}

} // namespace

int main() {
    try {
        invalid_input_tests();
        const orig::Query small_capacity_query = capacity_crossing_query();
        std::vector<Entry> entries = fixtures();
        capacity_and_fallback_tests(entries);

        int host_retries = 0;
        int fast_ok = 0, fast_needs_exact = 0, fast_invalid = 0;
        std::vector<orig::Result> host_fast_results;
        std::vector<orig::Result> host_exact_results;
        std::vector<CpuOutcome> cpu_outcomes;
        host_fast_results.reserve(entries.size());
        host_exact_results.reserve(entries.size());
        cpu_outcomes.reserve(entries.size());
        for (const Entry& entry : entries) {
            const orig::Result fast = orig::evaluate_fast(entry.query);
            host_fast_results.push_back(fast);
            if (fast.status == orig::Status::Ok) ++fast_ok;
            else if (fast.status == orig::Status::NeedsExact) ++fast_needs_exact;
            else if (fast.status == orig::Status::InvalidInput) ++fast_invalid;
            require(fast.status == orig::Status::Ok
                    || fast.status == orig::Status::NeedsExact,
                    entry.name + ": fast status must be Ok or NeedsExact");
            if (fast.status == orig::Status::NeedsExact) {
                require(!fast.collision && fast.toi == 1.0,
                        entry.name + ": NeedsExact must not look like a hit");
            }
            if (fast.status == orig::Status::Ok && fast.collision) {
                require(fast.toi >= 0.0 && fast.toi <= 1.0,
                        entry.name + ": fast TOI range");
            }

            const orig::Result exact =
                host_exact(entry.query, host_retries);
            host_exact_results.push_back(exact);
            require(exact.status != orig::Status::NeedsExact,
                    entry.name + ": exact must not return NeedsExact");
            require(exact.status == orig::Status::Ok,
                    entry.name + ": exact status " +
                    std::to_string(static_cast<unsigned>(exact.status)));
            require(exact.collision || exact.toi == 1.0,
                    entry.name + ": exact no-hit TOI must be 1");
            if (exact.collision) {
                require(exact.toi >= 0.0 && exact.toi <= 1.0,
                        entry.name + ": exact TOI range");
            }
            compare_analytic(entry, exact);
            compare_directed_exact_bits(entry, exact, "host");
            cpu_outcomes.push_back(
                compare_route_with_cpu(entry, fast, exact));
        }

        // The fast path is permitted to be inconclusive, but no inconclusive
        // query may be counted as a no-hit by callers. This check exercises
        // the explicit hand-off.
        require(fast_invalid == 0, "valid fixtures produced InvalidInput");
        require(fast_needs_exact > 0,
                "at least one fixture must exercise the NeedsExact hand-off");

        // Storage-layout/transfer regression for the Result records the CUDA
        // harness moves as raw 16-byte objects.
        transfer_representation_tests(host_fast_results, host_exact_results);
        std::cout << "Result transfer records verified: "
                  << host_fast_results.size() << " fast + "
                  << host_exact_results.size() << " exact\n";

#if defined(IPC_GPU_CUDA_TEST)
        std::vector<orig::Query> queries;
        queries.reserve(entries.size());
        for (const Entry& entry : entries) queries.push_back(entry.query);
        std::vector<orig::Result> fast(queries.size());
        std::vector<orig::Result> exact(queries.size());
        orig::test::DeviceDiagnostics diagnostics;
        original_ccd_cuda_evaluate(queries.data(), queries.size(),
                                   fast.data(), exact.data(),
                                   &small_capacity_query, &diagnostics);

        require(diagnostics.exact_queries == queries.size(),
                "CUDA: not every query executed");
        require(diagnostics.exact_ok == queries.size(),
                "CUDA: every valid fixture must resolve exactly");
        require(diagnostics.exact_capacity_exceeded == 0
                && diagnostics.exact_invalid == 0
                && diagnostics.exact_other == 0,
                "CUDA: unresolved or invalid exact status");
        require(diagnostics.retries_2048 > 0,
                "CUDA: wide fixtures must exercise a real 256->2048 retry");
        require(diagnostics.first_pass_failed == 0,
                "CUDA: first-pass DevicePool failure");
        require(diagnostics.first_pass_available_mismatch == 0,
                "CUDA: first-pass DevicePool ownership leak");
        require(diagnostics.retry_resource_errors == 0,
                "CUDA: allocator failure was not a legitimate retry");
        require(diagnostics.ownership_errors == 0,
                "CUDA: ownership error was hidden");
        require(diagnostics.final_pass_failed == 0,
                "CUDA: final-pass DevicePool failure");
        require(diagnostics.final_pass_available_mismatch == 0,
                "CUDA: final-pass DevicePool ownership leak");
        require(diagnostics.final_pass_errors == 0,
                "CUDA: final wide pass did not recover the arena");
        require(diagnostics.small_capacity_limb_failures == 1,
                "CUDA: capacity-2 limb exhaustion not observed");
        require(diagnostics.small_capacity_pool_failures == 0,
                "CUDA: capacity-2 probe corrupted the arena");
        require(diagnostics.small_capacity_available_mismatch == 0,
                "CUDA: capacity-2 probe leaked arena slots");
        require(diagnostics.small_capacity_retry_ok == 1,
                "CUDA: capacity-2048 recovery after limb failure failed");
        require(diagnostics.small_capacity_retry_failures == 0,
                "CUDA: capacity-2048 probe did not resolve");
        require(diagnostics.small_capacity_retry_pool_failures == 0,
                "CUDA: capacity-2048 probe corrupted the arena");
        require(diagnostics.small_capacity_retry_available_mismatch == 0,
                "CUDA: capacity-2048 probe leaked arena slots");
        require(diagnostics.slot_count == 512,
                "CUDA: unexpected bounded arena slot count");
        require(diagnostics.peak_slots > 0
                && diagnostics.peak_slots <= diagnostics.slot_count,
                "CUDA: impossible peak arena occupancy");

        int fast_status_differences = 0;
        int device_fast_ok = 0, device_fast_needs_exact = 0;
        for (std::size_t i = 0; i < queries.size(); ++i) {
            // These records reached the host through the real
            // cudaMemcpy(count * sizeof(Result)) transfer, so their reserved
            // bytes must be defined (zero) rather than inherited ABI padding.
            require_transfer_record(entries[i].name + " (device exact)",
                                    exact[i]);
            require_transfer_record(entries[i].name + " (device fast)",
                                    fast[i]);
            require(exact[i].status == orig::Status::Ok,
                    entries[i].name + ": CUDA exact status");
            compare_directed_exact_bits(entries[i], exact[i], "device");
            const orig::Result& host = host_exact_results[i];
            require(exact[i].collision == host.collision,
                    entries[i].name + ": CUDA collision mismatch");
            if (host.collision)
                require(ulp_distance(exact[i].toi, host.toi) <= 8,
                        entries[i].name + ": CUDA TOI mismatch");
            require(fast[i].status == orig::Status::Ok
                    || fast[i].status == orig::Status::NeedsExact,
                    entries[i].name + ": CUDA fast status");
            if (fast[i].status == orig::Status::Ok) ++device_fast_ok;
            else ++device_fast_needs_exact;
            if (fast[i].status == orig::Status::Ok && fast[i].collision) {
                require(fast[i].toi >= 0.0 && fast[i].toi <= 1.0,
                        entries[i].name + ": CUDA fast TOI range");
            }
            if (fast[i].status == orig::Status::NeedsExact) {
                require(!fast[i].collision && fast[i].toi == 1.0,
                        entries[i].name
                        + ": CUDA NeedsExact must not look like a hit");
            }
            if (fast[i].status != host_fast_results[i].status)
                ++fast_status_differences;
            else if (fast[i].status == orig::Status::Ok) {
                require(fast[i].collision == host_fast_results[i].collision,
                        entries[i].name
                        + ": CUDA fast collision differs from host fast");
                if (fast[i].collision)
                    require(toi_matches_cpu(fast[i].toi,
                                            host_fast_results[i].toi),
                            entries[i].name
                            + ": CUDA fast TOI differs from host fast");
            }
#if defined(IPC_GPU_PRODUCTION_REFERENCE)
            const CpuOutcome& cpu = cpu_outcomes[i];
            require(exact[i].collision == cpu.collision,
                    entries[i].name
                    + ": CUDA exact collision differs from CPU original");
            if (cpu.collision)
                require(toi_matches_cpu(exact[i].toi, cpu.toi),
                        entries[i].name
                        + ": CUDA exact TOI differs from CPU original");
            // Every fast Ok result is consumed by a caller and therefore must
            // be checked against the CPU original, not merely range-checked.
            if (fast[i].status == orig::Status::Ok) {
                require(fast[i].collision == cpu.collision,
                        entries[i].name
                        + ": CUDA fast collision differs from CPU original");
                if (cpu.collision)
                    require(toi_matches_cpu(fast[i].toi, cpu.toi),
                            entries[i].name
                            + ": CUDA fast TOI differs from CPU original");
                else
                    require(fast[i].toi == 1.0,
                            entries[i].name
                            + ": CUDA fast no-hit TOI must be 1");
            }
#endif
        }
        std::cout << "CUDA original-CCD kernels: " << queries.size()
                  << " queries; device 256->2048 retries="
                  << diagnostics.retries_2048
                  << "; exact Ok=" << diagnostics.exact_ok
                  << "; fast Ok/NeedsExact=" << device_fast_ok << "/"
                  << device_fast_needs_exact
                  << "; peak slots=" << diagnostics.peak_slots << "/"
                  << diagnostics.slot_count
                  << "; arena errors=" << diagnostics.retry_resource_errors
                  << "/" << diagnostics.final_pass_errors
                  << "; fast host-status differences="
                  << fast_status_differences << '\n';
#else
        std::cout << "HOSTONLY: CUDA original-CCD kernel test not compiled\n";
#endif

#if defined(IPC_GPU_PRODUCTION_REFERENCE)
        std::cout << "CPU original oracle available\n";
#else
        std::cout << "HOSTONLY: real CPU ccd.cpp oracle not linked\n";
#endif
        std::cout << "Passed " << entries.size()
                  << " original CCD vectors; fast Ok=" << fast_ok
                  << " NeedsExact=" << fast_needs_exact
                  << " host wide retries=" << host_retries << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
