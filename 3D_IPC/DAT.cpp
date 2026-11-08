#include "DAT.h"

#if defined(IPC_DAT_DIRECTED_ROUNDING_TEST)

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iomanip>

#if defined(__clang__)
#pragma STDC FENV_ACCESS ON
#endif

namespace {

std::uint64_t bits(double x) {
    std::uint64_t result;
    std::memcpy(&result, &x, sizeof(result));
    return result;
}

double value(std::uint64_t pattern) {
    double result;
    std::memcpy(&result, &pattern, sizeof(result));
    return result;
}

// A volatile function pointer prevents folding the oracle into the candidate
// or replacing platform libm calls with compile-time special-case constants.
double (*volatile platform_nextafter)(double, double) = std::nextafter;

double reference_up(double x) {
    return platform_nextafter(x, std::numeric_limits<double>::infinity());
}
double reference_down(double x) {
    return platform_nextafter(x, -std::numeric_limits<double>::infinity());
}
double reference_zero(double x) { return platform_nextafter(x, 0.0); }

using Unary = double (*)(double);
struct Operation { const char* name; Unary reference; Unary candidate; };
const std::array<Operation, 3> operations{{
    {"next_up", reference_up, directed_rounding::next_up},
    {"next_down", reference_down, directed_rounding::next_down},
    {"toward_positive_zero", reference_zero, directed_rounding::toward_positive_zero}
}};

class ScopedEnvironment {
    std::fenv_t environment_;
    int saved_errno_ = errno;
    bool saved_;
public:
    ScopedEnvironment() : saved_(std::fegetenv(&environment_) == 0) {}
    ~ScopedEnvironment() {
        if (saved_) std::fesetenv(&environment_);
        errno = saved_errno_;
    }
};

struct Observation {
    std::uint64_t output;
    int error;
    int exceptions;
};

Observation observe(Unary operation, double input, int initial_errno, int initial_exceptions) {
    std::feclearexcept(FE_ALL_EXCEPT);
    if (initial_exceptions) std::feraiseexcept(initial_exceptions);
    errno = initial_errno;
    const auto output = bits(operation(input));
    const int error = errno;
    const int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    return {output, error, exceptions};
}

::testing::AssertionResult same_observation(const Operation& operation,
    std::uint64_t input, int initial_errno, int initial_exceptions) {
    const auto expected = observe(operation.reference, value(input), initial_errno, initial_exceptions);
    const auto actual = observe(operation.candidate, value(input), initial_errno, initial_exceptions);
    if (expected.output == actual.output && expected.error == actual.error
        && expected.exceptions == actual.exceptions)
        return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << operation.name << " input=0x" << std::hex << input
        << " output expected=0x" << expected.output << " actual=0x" << actual.output
        << std::dec << " errno expected=" << expected.error << " actual=" << actual.error
        << " exceptions expected=" << expected.exceptions << " actual=" << actual.exceptions;
}

constexpr std::array<std::uint64_t, 18> boundary_magnitudes{{
    0, 1, 2, 3, UINT64_C(0x000ffffffffffffe), UINT64_C(0x000fffffffffffff),
    UINT64_C(0x0010000000000000), UINT64_C(0x0010000000000001),
    UINT64_C(0x3ff0000000000000), UINT64_C(0x3ff0000000000001),
    UINT64_C(0x7feffffffffffffe), UINT64_C(0x7fefffffffffffff),
    UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff0000000000001),
    UINT64_C(0x7ff0000000001234), UINT64_C(0x7ff7ffffffffffff),
    UINT64_C(0x7ff8000000000000), UINT64_C(0x7fffffffffffffff)
}};
constexpr std::array<int, 4> rounding_modes{{FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}};

std::uint64_t next_pattern(std::uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

} // namespace

TEST(DirectedRounding, SignedZerosAndBasicNeighbors) {
    ScopedEnvironment restore;
    for (std::uint64_t zero : {UINT64_C(0), UINT64_C(0x8000000000000000)}) {
        EXPECT_EQ(bits(directed_rounding::next_up(value(zero))), UINT64_C(1));
        EXPECT_EQ(bits(directed_rounding::next_down(value(zero))), UINT64_C(0x8000000000000001));
        EXPECT_EQ(bits(directed_rounding::toward_positive_zero(value(zero))), UINT64_C(0));
    }
    EXPECT_EQ(bits(directed_rounding::next_up(1.0)), UINT64_C(0x3ff0000000000001));
    EXPECT_EQ(bits(directed_rounding::next_down(1.0)), UINT64_C(0x3fefffffffffffff));
    EXPECT_EQ(bits(directed_rounding::next_up(-1.0)), UINT64_C(0xbfefffffffffffff));
    EXPECT_EQ(bits(directed_rounding::next_down(-1.0)), UINT64_C(0xbff0000000000001));
    EXPECT_EQ(bits(directed_rounding::toward_positive_zero(-1.0)), UINT64_C(0xbfefffffffffffff));
}

TEST(DirectedRounding, BoundaryResultsErrnoAndExceptionsMatchInEveryRoundingMode) {
    ScopedEnvironment restore;
    for (int mode : rounding_modes) {
        ASSERT_EQ(std::fesetround(mode), 0);
        for (auto magnitude : boundary_magnitudes)
            for (auto sign : {UINT64_C(0), UINT64_C(0x8000000000000000)})
                for (const auto& operation : operations)
                    for (int initial_errno : {0, 123})
                        for (int initial_exceptions : {0, FE_DIVBYZERO})
                            ASSERT_TRUE(same_observation(operation, magnitude | sign,
                                initial_errno, initial_exceptions)) << "rounding mode=" << mode;
    }
}

TEST(DirectedRounding, PowerOfTwoTransitionsMatchThePlatformBitwise) {
    ScopedEnvironment restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    for (std::uint64_t exponent = 1; exponent < 2047; ++exponent)
        for (auto mantissa : {UINT64_C(0), UINT64_C(1),
                UINT64_C(0x000ffffffffffffe), UINT64_C(0x000fffffffffffff)})
            for (auto sign : {UINT64_C(0), UINT64_C(0x8000000000000000)}) {
                const auto input = sign | (exponent << 52) | mantissa;
                for (const auto& operation : operations)
                    ASSERT_EQ(bits(operation.candidate(value(input))), bits(operation.reference(value(input))))
                        << operation.name << " input=0x" << std::hex << input;
            }
}

TEST(DirectedRounding, RandomBinary64PatternsMatchThePlatformBitwise) {
    ScopedEnvironment restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    std::uint64_t state = UINT64_C(0x8247a52c1a18f903);
    for (int sample = 0; sample < 250000; ++sample) {
        const auto input = next_pattern(state);
        for (const auto& operation : operations)
            ASSERT_EQ(bits(operation.candidate(value(input))), bits(operation.reference(value(input))))
                << operation.name << " input=0x" << std::hex << input;
    }
}

TEST(DirectedRounding, RandomResultsAndEnvironmentMatchInEveryRoundingMode) {
    ScopedEnvironment restore;
    std::uint64_t state = UINT64_C(0x9a8c00f27487c1e3);
    for (int mode : rounding_modes) {
        ASSERT_EQ(std::fesetround(mode), 0);
        for (int sample = 0; sample < 4096; ++sample) {
            const auto input = next_pattern(state);
            for (const auto& operation : operations)
                ASSERT_TRUE(same_observation(operation, input, 123, FE_DIVBYZERO))
                    << "rounding mode=" << mode;
        }
    }
}

#else

#include "broad_phase.h"
#include "ccd.h"
#include "friction_energy.h"
#include "parallel_helper.h"
#include "solid_ipc.h"
#include "segment_segment_distance.h"
#include "rigid_dat_trajectory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

struct DATEndpointConstraint {
    std::array<int,4> nodes{};
    Vec3 normal=Vec3::Zero();
    int split=0;
};

struct DATWorkspace::Impl {
    BroadPhase broad_phase;
    std::vector<AABB> boxes;
    std::vector<std::atomic<double>> factors, body_factors, radius_caps;
    std::vector<int> local, rigid_nodes;
    std::vector<double> rigid_radius_factors;
    std::vector<rigid_dat::Trajectory> paths;
    std::vector<std::optional<rigid_dat::PreparedTrajectory>> prepared_paths;
    std::vector<Vec3> updates;
    std::vector<std::atomic<int>> ownership_claims;
    std::vector<std::atomic<unsigned char>> interior, bad_node, bad_body;
    std::vector<std::size_t> body_offsets, radius_chunk_offsets;
    std::vector<double> radius_chunk_minima, node_factors;
    std::vector<DATEndpointConstraint> endpoint_constraints;
};
DATWorkspace::DATWorkspace() : impl(std::make_unique<Impl>()) {}
DATWorkspace::~DATWorkspace() = default;

template<class Value>
static void dat_resize_atomic(std::vector<std::atomic<Value>>& values, std::size_t count) {
    if (values.size()!=count) values=std::vector<std::atomic<Value>>(count);
}

// An outward-rounded coordinate gap is a lower bound on Euclidean distance.
// Reject only with that certificate; avoid squared norms and their overflow.
static bool dat_aabb_separated(const std::array<int,4>& nodes, int split,
    const std::vector<Vec3>& x, double radius) {
    for (int k=0; k<3; ++k) {
        double alo=INFINITY, ahi=-INFINITY, blo=INFINITY, bhi=-INFINITY;
        for (int i=0; i<4; ++i) {
            const double value=x[nodes[i]][k];
            if (i<split) {alo=std::min(alo,value);ahi=std::max(ahi,value);}
            else {blo=std::min(blo,value);bhi=std::max(bhi,value);}
        }
        const double gap=std::max(directed_rounding::next_down(blo-ahi),
                                  directed_rounding::next_down(alo-bhi));
        if (gap>=radius) return true;
    }
    return false;
}

static void dat_atomic_min(std::atomic<double>& destination, double value)
{
    double previous = destination.load(std::memory_order_relaxed);
    while (value < previous && !destination.compare_exchange_weak(
        previous, value, std::memory_order_relaxed)) {}
}

template<class Function>
static void dat_for_pairs(std::size_t count, const Function& function)
{
    if (count < 128) {
        for (std::size_t pair = 0; pair < count; ++pair) function(pair);
        return;
    }
    std::exception_ptr error;
    // Cheap AABB rejects and curved-path certificates can differ greatly in
    // cost even for adjacent pairs. Keep pair indices stable while balancing
    // those certificates dynamically; only atomic minima combine their output.
    #pragma omp parallel for schedule(dynamic,32)
    for (std::size_t pair = 0; pair < count; ++pair) {
        try {
            function(pair);
        } catch (...) {
            // Exceptions must not escape an OpenMP worker region.
            #pragma omp critical(dat_pair_error)
            {
                if (!error) error = std::current_exception();
            }
        }
    }
    if (error) std::rethrow_exception(error);
}

// Keep tiny fixtures on the direct serial path. Large trajectory batches use
// bounded dynamic chunks because fixed and moving paths have different costs.
// A chunk reduction supplies its underlying work count and schedules one chunk
// per task, so a large body still uses workers when it has few reduction tasks.
// All workers join before any reduction/read.
template<class Function>
static void dat_for_independent(std::size_t count, const Function& function,
    std::size_t underlying_work = 0, int chunk = 64)
{
    if (count < 128 && underlying_work < 128) {
        for (std::size_t item = 0; item < count; ++item) function(item);
        return;
    }
    std::exception_ptr error;
    #pragma omp parallel for schedule(dynamic,chunk)
    for (std::size_t item = 0; item < count; ++item) {
        try {
            function(item);
        } catch (...) {
            #pragma omp critical(dat_independent_error)
            {
                if (!error) error = std::current_exception();
            }
        }
    }
    if (error) std::rethrow_exception(error);
}

static Vec3 dat_capped_update(const Vec3& update, double limit)
{
    const double largest = update.cwiseAbs().maxCoeff();
    if (largest == 0) return update;
    // Avoid overflow even when the finite input's norm exceeds DBL_MAX.
    const Vec3 scaled = update / largest;
    const double length = scaled.norm();
    if (largest <= limit / length) return update;
    return scaled * (limit / length);
}

static double dat_norm_upper(const Vec3& component_bounds)
{
    const double scale=component_bounds.maxCoeff();
    if(scale==0)return 0;
    if(!std::isfinite(scale))return INFINITY;
    double squared_sum=0;
    for(int k=0;k<3;++k)if(component_bounds[k]>0) {
        const double ratio=directed_rounding::next_up(component_bounds[k]/scale);
        const double square=directed_rounding::next_up(ratio*ratio);
        squared_sum=directed_rounding::next_up(squared_sum+square);
    }
    const double root=directed_rounding::next_up(std::sqrt(squared_sum));
    return directed_rounding::next_up(scale*root);
}

static std::array<double,2> dat_plane_projection(const rigid_dat::Box& box,
    const Vec3& normal,const Vec3& point)
{
    double lo=0,hi=0;
    for(int k=0;k<3;++k)if(normal[k]!=0) {
        const double dl=directed_rounding::next_down(box.lo[k]-point[k]);
        const double dh=directed_rounding::next_up(box.hi[k]-point[k]);
        const double pl=normal[k]>=0?normal[k]*dl:normal[k]*dh;
        const double ph=normal[k]>=0?normal[k]*dh:normal[k]*dl;
        lo=directed_rounding::next_down(lo+directed_rounding::next_down(pl));
        hi=directed_rounding::next_up(hi+directed_rounding::next_up(ph));
    }
    return {lo,hi};
}

// Work relative to the query vertex instead of subtracting rounded world-space
// closest points. For an exterior projection, search all finite triangle edges:
// barycentric sign regions alone need not pick the closest edge of an obtuse tri.
static Vec3 dat_vertex_triangle_separation(
    const Vec3& vertex, const Vec3& a, const Vec3& b, const Vec3& c)
{
    std::array<Vec3, 3> local{a - vertex, b - vertex, c - vertex};
    double scale = 0;
    for (const auto& p : local) if (!p.allFinite())
        throw std::domain_error("planar_DAT: relative geometry overflow");
    for (const auto& p : local) scale = std::max(scale, p.cwiseAbs().maxCoeff());
    if (scale == 0) return Vec3::Zero();
    // Uniform scaling and stableNorm avoid underflow/overflow of face normals.
    for (auto& p : local) p /= scale;
    const Vec3 cross = (local[1] - local[0]).cross(local[2] - local[0]);
    const double normal_length = cross.stableNorm();
    if (normal_length > 0) {
        const Vec3 normal = cross / normal_length;
        const Vec3 projection = local[0].dot(normal) * normal;
        bool interior = true;
        // Oriented edge tests avoid the squared Gram determinant, which can
        // cancel to zero for a thin, but nondegenerate, triangle.
        for (int edge = 0; edge < 3; ++edge) {
            const Vec3 direction = local[(edge + 1) % 3] - local[edge];
            if (direction.cross(projection - local[edge]).dot(normal) < 0)
                interior = false;
        }
        if (interior) return projection * scale;
    }
    Vec3 closest = Vec3::Zero();
    double minimum = std::numeric_limits<double>::infinity();
    for (int edge = 0; edge < 3; ++edge) {
        double parameter;
        const Vec3 candidate = segment_closest_point(
            Vec3::Zero(), local[edge], local[(edge + 1) % 3], parameter);
        const double distance = candidate.stableNorm();
        if (distance < minimum) {
            minimum = distance;
            closest = candidate;
        }
    }
    if (!std::isfinite(minimum))
        throw std::domain_error("planar_DAT: unresolved vertex/triangle distance");
    return closest * scale;
}

struct DATPairPlane {
    Vec3 normal = Vec3::Zero();
    std::array<double, 4> coordinate{};
    double left = -INFINITY, right = INFINITY;
    bool outside_query = false;
};

// Validate the supporting direction against ALL finite-primitive corners.
// A tiny closest residual from the fast EE solve can contain a tangential
// error; normalization amplifies it until its plane no longer separates.
// Keep the ordinary allocation coordinates unchanged, but certify their
// direction with outward-rounded projections and retry exact geometry only
// when that certificate fails. Both DAT paths use this same selection.
static DATPairPlane dat_pair_plane(const std::array<int, 4>& nodes, int split,
    const Vec3& candidate, const std::vector<Vec3>& positions, double query_radius)
{
    DATPairPlane plane;
    const Vec3& origin = positions[nodes[0]];
    const auto select = [&](const Vec3& separation) {
        const double distance = separation.stableNorm();
        if (!separation.allFinite() || !(distance > 0) || !std::isfinite(distance))
            return false;
        plane.normal = separation / distance;
        plane.left = -INFINITY;
        plane.right = INFINITY;
        double certified_left = -INFINITY, certified_right = INFINITY;
        for (int i = 0; i < 4; ++i) {
            const Vec3& point = positions[nodes[i]];
            plane.coordinate[i] = (point - origin).dot(plane.normal);
            if (!std::isfinite(plane.coordinate[i])) return false;
            const std::array<double, 3> x{{point[0], point[1], point[2]}};
            const auto projection = dat_plane_projection({x, x}, plane.normal, origin);
            if (!std::isfinite(projection[0]) || !std::isfinite(projection[1]))
                return false;
            if (i < split) {
                plane.left = std::max(plane.left, plane.coordinate[i]);
                certified_left = std::max(certified_left, projection[1]);
            } else {
                plane.right = std::min(plane.right, plane.coordinate[i]);
                certified_right = std::min(certified_right, projection[0]);
            }
        }
        const double gap = plane.right - plane.left;
        const double certified_gap = directed_rounding::next_down(
            certified_right - certified_left);
        if (!(gap > 0) || !std::isfinite(gap) || !(certified_gap > 0))
            return false;
        // A proposed residual's norm is not a lower distance bound. The
        // certified projection gap is, after allowing for the represented
        // normal's norm. Do not discard a pair merely from a rounded distance.
        const double threshold = directed_rounding::next_up(query_radius
            * dat_norm_upper(plane.normal.cwiseAbs()));
        plane.outside_query = certified_gap >= threshold;
        return true;
    };
    if (select(candidate)) return plane;

    const std::array<Vec3, 4> points{{positions[nodes[0]], positions[nodes[1]],
        positions[nodes[2]], positions[nodes[3]]}};
    const Vec3 exact = -contact_closest_residual_exact(points, split == 1);
    if (select(exact)) return plane;
    if (exact.isZero(0) && compare_contact_distance_exact(points, split == 1, 0) == 0)
        throw std::domain_error("DAT: zero-gap pair has no separating plane");
    throw std::domain_error("DAT: cannot certify a separating closest-point normal");
}

// The first side_size entries form primitive A; the remainder form B.
// separation points from A to B. Every corner of both primitives contributes,
// not just the closest points, so the whole deforming primitive stays on its side.
static void dat_constrain_pair(
    const std::array<int, 4>& nodes, int side_size, const Vec3& separation,
    double gamma, double query_radius, const std::vector<Vec3>& positions,
    const std::vector<Vec3>& updates, std::vector<std::atomic<double>>& factors)
{
    const auto plane = dat_pair_plane(nodes, side_size, separation, positions, query_radius);
    if (plane.outside_query) return;
    const auto& coordinate = plane.coordinate;
    const double left = plane.left, right = plane.right;
    std::array<double, 4> velocity{};
    double approach_a = 0, approach_b = 0;
    for (int i = 0; i < 4; ++i) {
        velocity[i] = updates[nodes[i]].dot(plane.normal);
        if (!std::isfinite(velocity[i]))
            throw std::domain_error("planar_DAT: plane projection overflow");
        if (i < side_size) {
            approach_a = std::max(approach_a, velocity[i]);
        } else {
            approach_b = std::max(approach_b, -velocity[i]);
        }
    }
    const double gap = right - left;

    // Use the paper's proportional-gap rule (Sec. 3.3), expressed without its
    // inconsistent printed lambda/sign conventions. For A -> B normal:
    // plane = q_A + a/(a+b) * (q_B-q_A). Side A gets a/(a+b) of the gap.
    // When one side is stationary/retreating its share may be zero; that side
    // is not truncated. This avoids freezing motion toward a stationary face.
    const double largest = std::max(approach_a, approach_b);
    const double a = largest > 0 ? approach_a / largest : 1;
    const double b = largest > 0 ? approach_b / largest : 1;
    const double budget_a = gap * (a / (a + b));
    const double budget_b = gap * (b / (a + b));
    for (int i = 0; i < 4; ++i) {
        const bool side_a = i < side_size;
        const double approach = side_a ? velocity[i] : -velocity[i];
        if (approach <= 0) continue; // parallel or moving away from the plane
        const double margin = side_a
            ? (left - coordinate[i]) + budget_a
            : (coordinate[i] - right) + budget_b;
        // Eq. 16: min(1, gamma*t_hit), applying relaxation exactly once.
        // Compare first to avoid overflow for an almost tangential update.
        const double allowed = gamma * margin;
        if (approach > allowed) {
            const double factor = allowed / approach;
            dat_atomic_min(factors[nodes[i]], std::clamp(factor, 0.0, 1.0));
        }
    }
}

std::vector<Vec3> planar_DAT(
    double gamma, double query_radius, const RefMesh& mesh,
    const std::vector<Vec3>& incremental_update,
    const std::vector<Vec3>& current_positions, double node_box_padding,
    const BroadPhase* candidates, DATWorkspace* workspace)
{
    const std::size_t count = current_positions.size();
    if (!std::isfinite(gamma) || gamma <= 0 || gamma >= 1
        || !std::isfinite(query_radius) || query_radius <= 0
        || !std::isfinite(node_box_padding) || node_box_padding < 1)
        throw std::invalid_argument("planar_DAT: require 0 < gamma < 1, r_q > 0, and padding >= 1");
    if (incremental_update.size() != count
        || count > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || (mesh.num_positions != 0 && mesh.num_positions != count)
        || mesh.tris.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || mesh.tris.size() % 3 != 0)
        throw std::invalid_argument("planar_DAT: inconsistent vertex or triangle counts");
    const auto valid_node = [count](int node) {
        return node >= 0 && static_cast<std::size_t>(node) < count;
    };
    std::optional<DATWorkspace> local_workspace;
    if(!workspace) {local_workspace.emplace(); workspace=&*local_workspace;}
    auto& storage=*workspace->impl;
    dat_for_independent(mesh.tris.size()/3, [&](std::size_t triangle) {
        const std::size_t i=3*triangle;
        const int a = mesh.tris[i], b = mesh.tris[i + 1], c = mesh.tris[i + 2];
        if (!valid_node(a) || !valid_node(b) || !valid_node(c)
            || a == b || b == c || a == c)
            throw std::invalid_argument("planar_DAT: invalid triangle connectivity");
    });
    auto& interior=storage.interior;
    if (!mesh.tet_nodes.empty()) {
        dat_resize_atomic(interior,count);
        dat_for_independent(count, [&](std::size_t v) {interior[v].store(0,std::memory_order_relaxed);});
    }
    dat_for_independent(mesh.tet_nodes.size(), [&](std::size_t i) {
        const int node=mesh.tet_nodes[i];
        if (!valid_node(node))
            throw std::invalid_argument("planar_DAT: invalid tet-node classification");
        // Classification lists may contain duplicates. Atomic stores avoid
        // races without changing the ordered solid-then-surface precedence.
        interior[node].store(1,std::memory_order_relaxed);
    });
    dat_for_independent(mesh.surface_nodes.size(), [&](std::size_t i) {
        const int node=mesh.surface_nodes[i];
        if (!valid_node(node))
            throw std::invalid_argument("planar_DAT: invalid surface-node classification");
        if (!mesh.tet_nodes.empty())interior[node].store(0,std::memory_order_relaxed);
    });
    if (!mesh.tet_nodes.empty()) {
        dat_for_independent(mesh.tris.size(), [&](std::size_t i) {
            if (interior[mesh.tris[i]].load(std::memory_order_relaxed))
                throw std::invalid_argument("planar_DAT: triangle vertex classified as solid interior");
        });
    }
    const double limit = (0.5 * gamma) * query_radius;
    const double radius = query_radius * node_box_padding;
    if (!(limit > 0) || !std::isfinite(radius))
        throw std::invalid_argument("planar_DAT: displacement limit or box radius out of range");

    std::vector<Vec3> updates(count);
    auto& boxes=storage.boxes; boxes.resize(count);
    auto& factors=storage.factors; dat_resize_atomic(factors,count);
    dat_for_independent(count, [&](std::size_t i) {
        if (!current_positions[i].allFinite() || !incremental_update[i].allFinite())
            throw std::invalid_argument("planar_DAT: nonfinite position or update");
        updates[i] = dat_capped_update(incremental_update[i], limit);
        boxes[i] = AABB(current_positions[i] - Vec3::Constant(radius),
                        current_positions[i] + Vec3::Constant(radius));
        if (!boxes[i].min.allFinite() || !boxes[i].max.allFinite())
            throw std::invalid_argument("planar_DAT: node box overflow");
        factors[i].store(1.0, std::memory_order_relaxed);
    });
    if (count == 0) return updates;

    // Reuse topology within a solve. Green triangles and red edges are unions
    // of these blue boxes. Zero additional d_hat padding: r_q already defines
    // the query neighborhood; there is no barrier stiffness/energy here.
    auto& broad_phase=storage.broad_phase;
    if (!candidates) {
        if (!mesh.tets.empty() || !mesh.tet_nodes.empty())
            broad_phase.initialize_surface_nodes(boxes, mesh, 0.0,
                BroadPhase::InitializationMode::DeformableSolver, false);
        else
            broad_phase.initialize(boxes, mesh, 0.0,
                BroadPhase::InitializationMode::DeformableSolver, false);
    }
    const auto& cache = candidates?candidates->cache():broad_phase.cache();
    dat_for_pairs(cache.nt_pairs.size()+cache.ss_pairs.size(), [&](std::size_t i) {
        if (i<cache.nt_pairs.size()) {
            const auto& pair = cache.nt_pairs[i];
            const std::array<int, 4> nodes{
                pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
            if (dat_aabb_separated(nodes,1,current_positions,query_radius)) return;
            const Vec3 separation = dat_vertex_triangle_separation(
                current_positions[nodes[0]], current_positions[nodes[1]],
                current_positions[nodes[2]], current_positions[nodes[3]]);
            dat_constrain_pair(nodes, 1, separation, gamma, query_radius,
                current_positions, updates, factors);
        } else {
            const auto& pair = cache.ss_pairs[i-cache.nt_pairs.size()];
            const std::array<int, 4> nodes{pair.v[0], pair.v[1], pair.v[2], pair.v[3]};
            if (dat_aabb_separated(nodes,2,current_positions,query_radius)) return;
            const auto distance = segment_segment_distance(
                current_positions[nodes[0]], current_positions[nodes[1]],
                current_positions[nodes[2]], current_positions[nodes[3]]);
            dat_constrain_pair(nodes, 2, -distance.separation, gamma, query_radius,
                current_positions, updates, factors);
        }
    });
    #pragma omp parallel for schedule(static) if(count >= 128)
    for (std::size_t i = 0; i < count; ++i)
        updates[i] *= factors[i].load(std::memory_order_relaxed);
    return updates;
}

GeneralDATResult general_planar_DAT(double gamma, double query_radius,
    const RefMesh& mesh, const std::vector<Vec3>& particle_updates,
    const std::vector<Vec3>& positions, const std::vector<Vec3>& centers,
    const std::vector<Vec3>& omega, const std::vector<Vec4>& previous_orientations,
    const std::vector<Vec3>& delta_centers, const std::vector<Vec3>& delta_omega,
    double dt, double node_box_padding,
    const BroadPhase* candidates, DATWorkspace* workspace)
{
    const std::size_t n = positions.size(), bodies = mesh.rb_nodes.size();
    if (bodies == 0) {
        if (!centers.empty() || !omega.empty() || !previous_orientations.empty()
            || !delta_centers.empty() || !delta_omega.empty())
            throw std::invalid_argument("general_DAT: unexpected rigid coordinates");
        GeneralDATResult result;
        const auto displacements=planar_DAT(gamma,query_radius,mesh,particle_updates,
            positions,node_box_padding,candidates,workspace);
        result.positions.resize(n);
        dat_for_independent(n, [&](std::size_t v) {
            result.positions[v]=positions[v]+displacements[v];
            if(!result.positions[v].allFinite())throw std::domain_error("general_DAT: endpoint overflow");
        });
        return result;
    }
    if (!(gamma > 0 && gamma < 1) || !std::isfinite(gamma)
        || !(query_radius > 0) || !std::isfinite(query_radius)
        || !(node_box_padding >= 1) || !std::isfinite(node_box_padding)
        || !(dt > 0) || !std::isfinite(dt) || !(dt*dt>0) || !std::isfinite(dt*dt))
        throw std::invalid_argument("general_DAT: invalid filter parameters");
    if (particle_updates.size() != n || mesh.node_to_rb.size() != n
        || (mesh.num_positions && mesh.num_positions != n)
        || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) || bodies>n
        || mesh.tris.size()>static_cast<std::size_t>(std::numeric_limits<int>::max())
        || centers.size() != bodies || omega.size() != bodies
        || previous_orientations.size() != bodies || delta_centers.size() != bodies
        || delta_omega.size() != bodies || mesh.ref_positions.size() != bodies
        || mesh.tris.size() % 3 || mesh.tets.size() % 4)
        throw std::invalid_argument("general_DAT: inconsistent mesh/coordinate sizes");
    const auto valid_node = [n](int v) { return v >= 0 && std::size_t(v) < n; };
    std::optional<DATWorkspace> local_workspace;
    if(!workspace) {local_workspace.emplace(); workspace=&*local_workspace;}
    auto& storage=*workspace->impl;
    auto& local=storage.local; local.resize(n);
    auto& ownership_claims=storage.ownership_claims; dat_resize_atomic(ownership_claims,n);
    auto& interior=storage.interior; dat_resize_atomic(interior,n);
    dat_for_independent(n, [&](std::size_t v) {
        local[v]=-1;
        ownership_claims[v].store(-1,std::memory_order_relaxed);
        interior[v].store(0,std::memory_order_relaxed);
    });
    auto& body_offsets=storage.body_offsets; body_offsets.resize(bodies+1);
    body_offsets[0]=0;
    for(std::size_t b=0;b<bodies;++b) {
        if(mesh.rb_nodes[b].size()>n-body_offsets[b])
            throw std::invalid_argument("general_DAT: too many rigid proxy nodes");
        body_offsets[b+1]=body_offsets[b]+mesh.rb_nodes[b].size();
    }
    const auto array = [](const Vec3& x) { return rigid_dat::Vector{x[0],x[1],x[2]}; };
    // Only rigid vertices need trajectories, including in large mixed scenes.
    auto& paths=storage.paths; paths.resize(body_offsets.back());
    auto& rigid_nodes=storage.rigid_nodes; rigid_nodes.resize(paths.size());
    auto& radius_factors=storage.rigid_radius_factors; radius_factors.resize(paths.size());
    auto& prepared_paths=storage.prepared_paths; prepared_paths.resize(paths.size());
    const auto path_index=[&](int v) {return body_offsets[mesh.node_to_rb[v]]+std::size_t(local[v]);};
    dat_for_independent(bodies, [&](std::size_t b) {
        if (mesh.rb_nodes[b].empty() || mesh.rb_nodes[b].size() != mesh.ref_positions[b].size()
            || !centers[b].allFinite() || !omega[b].allFinite()
            || !delta_centers[b].allFinite() || !delta_omega[b].allFinite()
            || omega[b].cwiseAbs().maxCoeff()>std::sqrt(std::numeric_limits<double>::max())/8
            || delta_omega[b].cwiseAbs().maxCoeff()>std::sqrt(std::numeric_limits<double>::max())/8
            || !previous_orientations[b].allFinite()
            || std::abs(previous_orientations[b].squaredNorm()-1) > 1e-10)
            throw std::invalid_argument("general_DAT: invalid rigid body");
    });
    dat_for_independent(rigid_nodes.size(), [&](std::size_t index) {
        const std::size_t b=std::size_t(std::upper_bound(body_offsets.begin(),body_offsets.end(),index)
            -body_offsets.begin()-1);
        const std::size_t j=index-body_offsets[b];
        const int v=mesh.rb_nodes[b][j];
        if (!valid_node(v) || mesh.node_to_rb[v] != int(b)
            || !mesh.ref_positions[b][j].allFinite())
            throw std::invalid_argument("general_DAT: inconsistent rigid ownership");
        int unclaimed=-1;
        // Even malformed duplicate entries must not race on local[v]. Only
        // the successful claim writes it; every other writer reports an error
        // before the joined validation phase can read an incomplete mapping.
        if(!ownership_claims[v].compare_exchange_strong(unclaimed,int(j),std::memory_order_relaxed))
            throw std::invalid_argument("general_DAT: inconsistent rigid ownership");
        local[v]=int(j);
        rigid_nodes[index]=v;
    });
    dat_for_independent(n, [&](std::size_t v) {
        const int owner=mesh.node_to_rb[v];
        if (owner < -1 || owner >= int(bodies) || ((owner>=0) != (local[v]>=0))
            || !positions[v].allFinite() || !particle_updates[v].allFinite())
            throw std::invalid_argument("general_DAT: invalid node ownership/state");
        if (owner>=0) {
            const Vec3 expected=world_space_position(mesh.ref_positions[owner][local[v]],
                centers[owner],previous_orientations[owner],omega[owner],dt);
            const double tolerance=128*std::numeric_limits<double>::epsilon()
                *std::max({1.0,expected.cwiseAbs().maxCoeff(),positions[v].cwiseAbs().maxCoeff()});
            if (!expected.allFinite() || (expected-positions[v]).cwiseAbs().maxCoeff()>tolerance)
                throw std::invalid_argument("general_DAT: rigid proxies disagree with pose");
        }
    });
    dat_for_independent(mesh.tris.size()/3, [&](std::size_t triangle) {
        const std::size_t i=3*triangle;
        const int a=mesh.tris[i],b=mesh.tris[i+1],c=mesh.tris[i+2];
        if (!valid_node(a)||!valid_node(b)||!valid_node(c)||a==b||b==c||a==c
            || mesh.node_to_rb[a]!=mesh.node_to_rb[b] || mesh.node_to_rb[a]!=mesh.node_to_rb[c])
            throw std::invalid_argument("general_DAT: invalid or mixed-owner triangle");
    });
    dat_for_independent(mesh.tets.size(), [&](std::size_t i) {
        const int v=mesh.tets[i];
        if(!valid_node(v)||mesh.node_to_rb[v]>=0)
            throw std::invalid_argument("general_DAT: invalid or rigid tetrahedron");
    });
    dat_for_independent(mesh.tet_nodes.size(), [&](std::size_t i) {
        const int v=mesh.tet_nodes[i];
        if(!valid_node(v)||mesh.node_to_rb[v]>=0) throw std::invalid_argument("general_DAT: invalid solid classification");
        interior[v].store(1,std::memory_order_relaxed);
    });
    dat_for_independent(mesh.surface_nodes.size(), [&](std::size_t i) {
        const int v=mesh.surface_nodes[i];
        if(!valid_node(v)) throw std::invalid_argument("general_DAT: invalid surface classification");
        interior[v].store(0,std::memory_order_relaxed);
    });
    dat_for_independent(mesh.tris.size(), [&](std::size_t i) {
        if(interior[mesh.tris[i]].load(std::memory_order_relaxed))
            throw std::invalid_argument("general_DAT: interior node on surface");
    });

    const double limit=.5*gamma*query_radius, radius=query_radius*node_box_padding;
    if (!(limit>0)||!std::isfinite(radius))
        throw std::invalid_argument("general_DAT: invalid query extent");
    GeneralDATResult result;
    result.body_factors.resize(bodies);
    auto& radius_caps=storage.radius_caps; dat_resize_atomic(radius_caps,bodies);
    dat_for_independent(bodies, [&](std::size_t b) {
        radius_caps[b].store(1.0,std::memory_order_relaxed);
    });
    // The far-pair bound is a full curved-path sphere, never endpoint distance.
    dat_for_independent(rigid_nodes.size(), [&](std::size_t index) {
        const int v=rigid_nodes[index],b=mesh.node_to_rb[v];
        auto& path=paths[index];
        path.center=array(centers[b]);path.delta_center=array(delta_centers[b]);
        path.omega=array(omega[b]);path.delta_omega=array(delta_omega[b]);
        path.reference=array(mesh.ref_positions[b][local[v]]);path.dt=dt;
        for(int k=0;k<4;++k)path.q0[k]=previous_orientations[b][k];
        prepared_paths[index].emplace(paths[index]);
        const auto& initial_box=prepared_paths[index]->initial_box;
        Vec3 initial_offset;
        for(int k=0;k<3;++k) initial_offset[k]=std::max(
            directed_rounding::next_up(std::abs(initial_box.lo[k]-positions[v][k])),
            directed_rounding::next_up(std::abs(initial_box.hi[k]-positions[v][k])));
        // Query neighborhoods are anchored at represented positions. The
        // mathematical phi(0) enclosure can differ by evaluation roundoff;
        // reserve that offset over the entire arc, not only at its endpoint.
        const double offset=dat_norm_upper(initial_offset);
        const double budget=offset<limit?directed_rounding::toward_positive_zero(limit-offset):0;
        if(!(budget>0))throw std::domain_error("general_DAT: unresolved query-anchor roundoff");
        const double cap=radius_caps[b].load(std::memory_order_relaxed);
        radius_factors[index]=rigid_dat::radius_step_capped(*prepared_paths[index],budget,cap);
        // Every observed cap is a minimum of already-certified factors. A
        // capped result cannot go below the original global minimum, and its
        // limiting proxy still returns that exact minimum. The ordered chunk
        // reduction below therefore retains the original body factor bitwise.
        dat_atomic_min(radius_caps[b],radius_factors[index]);
    });
    // Reduce large bodies in independent contiguous chunks, then combine each
    // body's chunk minima in original proxy order. std::min keeps the earliest
    // equal minimum, preserving the serial reduction even for signed zero.
    constexpr std::size_t radius_chunk_size=256;
    auto& radius_chunk_offsets=storage.radius_chunk_offsets; radius_chunk_offsets.resize(bodies+1);
    radius_chunk_offsets[0]=0;
    for(std::size_t b=0;b<bodies;++b)
        radius_chunk_offsets[b+1]=radius_chunk_offsets[b]
            +(mesh.rb_nodes[b].size()+radius_chunk_size-1)/radius_chunk_size;
    auto& radius_chunk_minima=storage.radius_chunk_minima;
    radius_chunk_minima.resize(radius_chunk_offsets.back());
    dat_for_independent(radius_chunk_minima.size(), [&](std::size_t chunk) {
        const std::size_t b=std::size_t(std::upper_bound(radius_chunk_offsets.begin(),radius_chunk_offsets.end(),chunk)
            -radius_chunk_offsets.begin()-1);
        const std::size_t begin=body_offsets[b]+(chunk-radius_chunk_offsets[b])*radius_chunk_size;
        const std::size_t end=std::min(begin+radius_chunk_size,body_offsets[b+1]);
        double minimum=1;
        for(std::size_t index=begin;index<end;++index)minimum=std::min(minimum,radius_factors[index]);
        radius_chunk_minima[chunk]=minimum;
    },rigid_nodes.size(),1);
    dat_for_independent(bodies, [&](std::size_t b) {
        double minimum=1;
        for(std::size_t chunk=radius_chunk_offsets[b];chunk<radius_chunk_offsets[b+1];++chunk)
            minimum=std::min(minimum,radius_chunk_minima[chunk]);
        result.body_factors[b]=minimum;
    });
    auto& updates=storage.updates; updates.resize(n);
    auto& boxes=storage.boxes; boxes.resize(n);
    auto& factors=storage.factors; dat_resize_atomic(factors,n);
    auto& body_factors=storage.body_factors; dat_resize_atomic(body_factors,bodies);
    dat_for_independent(bodies, [&](std::size_t b) {body_factors[b].store(result.body_factors[b]);});
    dat_for_independent(n, [&](std::size_t v) {
        const int b=mesh.node_to_rb[v];
        if(b<0) updates[v]=dat_capped_update(particle_updates[v],limit);
        else if(delta_omega[b].cwiseAbs().maxCoeff()==0)
            updates[v]=result.body_factors[b]*delta_centers[b];
        else {
            // Plane allocation depends on local motion, not subtraction of two
            // rounded, potentially very large, world-space endpoints.
            auto local_path=paths[path_index(v)]; local_path.center={0,0,0};
            const auto endpoint=rigid_dat::detail::evaluate_core(local_path,result.body_factors[b],prepared_paths[path_index(v)]->u0);
            const auto start=rigid_dat::detail::evaluate_core(local_path,0,prepared_paths[path_index(v)]->u0);
            updates[v]=Vec3(endpoint[0]-start[0],endpoint[1]-start[1],endpoint[2]-start[2]);
        }
        boxes[v]=AABB(positions[v]-Vec3::Constant(radius),positions[v]+Vec3::Constant(radius));
        if(!updates[v].allFinite()||!boxes[v].min.allFinite()||!boxes[v].max.allFinite())
            throw std::invalid_argument("general_DAT: node extent overflow");
        factors[v].store(1);
    });
    auto& phase=storage.broad_phase;
    if (!candidates) {
        if(!mesh.tets.empty()||!mesh.tet_nodes.empty()) phase.initialize_surface_nodes(
            boxes,mesh,0,BroadPhase::InitializationMode::GeneralSolver, false);
        else phase.initialize(boxes,mesh,0,BroadPhase::InitializationMode::GeneralSolver, false);
    }
    const auto& cache=candidates?candidates->cache():phase.cache();
    auto& endpoint_constraints=storage.endpoint_constraints;
    endpoint_constraints.resize(cache.nt_pairs.size()+cache.ss_pairs.size());
    const auto constrain=[&](const std::array<int,4>& nodes,int split,const Vec3& separation,std::size_t index) {
        const auto plane=dat_pair_plane(nodes,split,separation,positions,query_radius);
        if(plane.outside_query)return;
        const Vec3& normal=plane.normal;
        const Vec3& origin=positions[nodes[0]];
        endpoint_constraints[index]={nodes,normal,split};
        const double left=plane.left,right=plane.right;
        double a=0,b=0;
        bool rigid_pair=false;
        const auto& coordinate=plane.coordinate;
        std::array<double,4> velocity{};
        for(int i=0;i<4;++i) {
            const int v=nodes[i],owner=mesh.node_to_rb[v];
            velocity[i]=updates[v].dot(normal);
            rigid_pair|=owner>=0;
            if(!std::isfinite(velocity[i]))
                throw std::domain_error("general_DAT: plane overflow");
            if(i<split)a=std::max(a,velocity[i]);
            else b=std::max(b,-velocity[i]);
        }
        const double gap=right-left;
        const double scale=std::max(a,b);
        a=scale>0?a/scale:1; b=scale>0?b/scale:1;
        // A curved vertex can return toward a plane despite retreating endpoint
        // motion. Give every rotating side an interior plane, then certify it.
        if(rigid_pair) {a=std::max(a,1e-3);b=std::max(b,1e-3);}
        const double budget_a=gap*(a/(a+b)),budget_b=gap*(b/(a+b));
        const Vec3 plane_point=origin+normal*(left+budget_a);
        std::array<double,4> certified_margins{};
        std::array<unsigned char,4> initial_side_safe{};
        if(rigid_pair) {
            // Materializing a plane point can move the represented plane out
            // of a narrow gap. Certify its actual sides with outward-rounded
            // projections before allowing plane_step to infer an initial side.
            for(int i=0;i<4;++i) {
                const auto x=array(positions[nodes[i]]);
                const auto projection=dat_plane_projection({x,x},normal,plane_point);
                const double lo=projection[0],hi=projection[1];
                if(!std::isfinite(lo)||!std::isfinite(hi)||(i<split?!(hi<0):!(lo>0)))
                    throw std::domain_error("general_DAT: rounded division plane is not separating");
                certified_margins[i]=i<split?-hi:lo;
                initial_side_safe[i]=1;
                if(mesh.node_to_rb[nodes[i]]>=0) {
                    const auto initial=dat_plane_projection(prepared_paths[path_index(nodes[i])]->initial_box,normal,plane_point);
                    initial_side_safe[i]=i<split?initial[1]<0:initial[0]>0;
                    certified_margins[i]=std::min(certified_margins[i],i<split?-initial[1]:initial[0]);
                }
            }
        }
        for(int i=0;i<4;++i) {
            const int v=nodes[i],owner=mesh.node_to_rb[v];
            if(owner>=0&&!initial_side_safe[i]) {
                throw std::domain_error("general_DAT: initial rigid trajectory is not on its intended plane side");
            }
            if(owner>=0 && delta_omega[owner].cwiseAbs().maxCoeff()>0) {
                const double alpha=rigid_dat::plane_step(*prepared_paths[path_index(v)],array(normal),array(plane_point),gamma);
                dat_atomic_min(body_factors[owner],alpha);
            } else {
                double approach=i<split?velocity[i]:-velocity[i];
                if(rigid_pair) {
                    // Both sides must use the same represented division plane.
                    // Bound the normal speed upwards, including a rigid body's
                    // radius cap, rather than trusting a rounded dot product.
                    double upper=0;
                    for(int k=0;k<3;++k) {
                        const double speed=owner<0?updates[v][k]:delta_centers[owner][k];
                        const double coefficient=i<split?normal[k]:-normal[k];
                        const double product=directed_rounding::next_up(coefficient*speed);
                        upper=directed_rounding::next_up(upper+product);
                    }
                    approach=owner<0?upper:directed_rounding::next_up(upper*result.body_factors[owner]);
                }
                if(approach<=0)continue;
                const double margin=rigid_pair?certified_margins[i]
                    :(i<split?left-coordinate[i]+budget_a:coordinate[i]-right+budget_b);
                const double allowed=rigid_pair
                    ?directed_rounding::toward_positive_zero(gamma*margin):gamma*margin;
                if(approach>allowed) {
                    double factor=std::clamp(allowed/approach,0.0,1.0);
                    if(rigid_pair)factor=directed_rounding::toward_positive_zero(factor);
                    if(owner<0)dat_atomic_min(factors[v],factor);
                    else dat_atomic_min(body_factors[owner],
                        directed_rounding::toward_positive_zero(result.body_factors[owner]*factor));
                }
            }
        }
    };
    dat_for_pairs(endpoint_constraints.size(),[&](std::size_t i){
        // Scratch survives the call. Clear before BOTH possible early exits
        // (AABB rejection and query-radius rejection inside constrain).
        endpoint_constraints[i].split=0;
        if(i<cache.nt_pairs.size()) {
            const auto& p=cache.nt_pairs[i]; const std::array<int,4> nodes{p.node,p.tri_v[0],p.tri_v[1],p.tri_v[2]};
            if(dat_aabb_separated(nodes,1,positions,query_radius))return;
            constrain(nodes,1,dat_vertex_triangle_separation(positions[nodes[0]],positions[nodes[1]],positions[nodes[2]],positions[nodes[3]]),i);
        } else {
            const auto& p=cache.ss_pairs[i-cache.nt_pairs.size()]; const std::array<int,4> nodes{p.v[0],p.v[1],p.v[2],p.v[3]};
            if(dat_aabb_separated(nodes,2,positions,query_radius))return;
            constrain(nodes,2,-segment_segment_distance(positions[nodes[0]],positions[nodes[1]],positions[nodes[2]],positions[nodes[3]]).separation,i);
        }
    });
    auto& node_factors=storage.node_factors; node_factors.resize(n);
    dat_for_independent(n, [&](std::size_t v) {node_factors[v]=factors[v].load();});
    dat_for_independent(bodies, [&](std::size_t b) {result.body_factors[b]=body_factors[b].load();});
    // Certify represented endpoints as well as the mathematical curved path.
    // COM addition and quaternion placement can otherwise round a positive gap
    // to zero. Backtracking only shortens already-certified trajectory prefixes.
    // Returned positions must own their buffer. Write their certified values
    // directly there instead of allocating a second reusable scratch and then
    // copying it (or moving its allocation out of the workspace every call).
    result.positions.resize(n);
    auto& endpoints=result.positions;
    result.centers.resize(bodies);result.omega.resize(bodies);result.orientations.resize(bodies);
    auto& bad_node=storage.bad_node; dat_resize_atomic(bad_node,n);
    auto& bad_body=storage.bad_body; dat_resize_atomic(bad_body,bodies);
    constexpr int endpoint_backtracks=std::numeric_limits<double>::digits+2;
    for(int attempt=0;;++attempt) {
        dat_for_independent(n, [&](std::size_t v) {
            if(mesh.node_to_rb[v]<0) {
                const Vec3 displacement=updates[v]*node_factors[v];
                endpoints[v]=positions[v]+displacement;
            }
        });
        dat_for_independent(bodies, [&](std::size_t b) {
            bad_body[b].store(0,std::memory_order_relaxed);
            const double alpha=result.body_factors[b];
            result.centers[b]=centers[b]+alpha*delta_centers[b];
            result.omega[b]=omega[b]+alpha*delta_omega[b];
            result.orientations[b]=quaternion_from_angular_velocity(previous_orientations[b],result.omega[b],dt);
        });
        dat_for_independent(rigid_nodes.size(), [&](std::size_t index) {
            const int v=rigid_nodes[index],b=mesh.node_to_rb[v];
            endpoints[v]=world_space_position(mesh.ref_positions[b][local[v]],result.centers[b],result.orientations[b]);
        });
        std::atomic<bool> failed{false};
        const auto mark=[&](int v){
            if(mesh.node_to_rb[v]<0)bad_node[v].store(1,std::memory_order_relaxed);
            else bad_body[mesh.node_to_rb[v]].store(1,std::memory_order_relaxed);
            // Fuse the global OR into the already parallel certificate pass.
            // Its decision is read only after all flag writers have joined.
            failed.store(true,std::memory_order_relaxed);
        };
        dat_for_independent(n, [&](std::size_t v) {
            // No constraint worker runs until this per-node phase joins.
            bad_node[v].store(0,std::memory_order_relaxed);
            if(!endpoints[v].allFinite()) {mark(int(v));return;}
            Vec3 extent=Vec3::Zero();
            for(int k=0;k<3;++k) {
                if(endpoints[v][k]!=positions[v][k])
                    extent[k]=directed_rounding::next_up(std::abs(endpoints[v][k]-positions[v][k]));
            }
            if(dat_norm_upper(extent)>limit) mark(int(v));
        });
        dat_for_independent(endpoint_constraints.size(), [&](std::size_t index) {
            const auto& constraint=endpoint_constraints[index];
            if(!constraint.split)return;
            double left=-INFINITY,right=INFINITY;
            const Vec3& origin=endpoints[constraint.nodes[0]];
            for(int i=0;i<4;++i) {
                double lo=0,hi=0;
                for(int k=0;k<3;++k) {
                    const double d=endpoints[constraint.nodes[i]][k]-origin[k];
                    const double dl=directed_rounding::next_down(d),dh=directed_rounding::next_up(d);
                    const double normal=constraint.normal[k];
                    lo=directed_rounding::next_down(lo+directed_rounding::next_down(normal*(normal>=0?dl:dh)));
                    hi=directed_rounding::next_up(hi+directed_rounding::next_up(normal*(normal>=0?dh:dl)));
                }
                if(i<constraint.split)left=std::max(left,hi);
                else right=std::min(right,lo);
            }
            if(!(right>left)) {
                for(int v:constraint.nodes)mark(v);
            }
        });
        // Failure flags may have many writers, but all writes are atomic and
        // joined here. The decision/retry order remains body/node deterministic.
        if(!failed.load(std::memory_order_relaxed)) {
            return result;
        }
        if(attempt==endpoint_backtracks)throw std::domain_error("general_DAT: cannot certify represented endpoints");
        dat_for_independent(bodies, [&](std::size_t b) {
            if(bad_body[b].load(std::memory_order_relaxed))
                result.body_factors[b]=attempt==endpoint_backtracks-1?0:.5*result.body_factors[b];
        });
        dat_for_independent(n, [&](std::size_t v) {
            if(bad_node[v].load(std::memory_order_relaxed))
                node_factors[v]=attempt==endpoint_backtracks-1?0:.5*node_factors[v];
        });
    }
}

namespace {

template<class Function>
void parallel_for(std::size_t count, bool parallel, const Function& function) {
    std::exception_ptr error;
    #pragma omp parallel for schedule(static) if(parallel && count >= 128)
    for (std::size_t i = 0; i < count; ++i) {
        try {
            function(i);
        } catch (...) {
            #pragma omp critical(ipc_dat_colored_contact_error)
            {
                if (!error) error = std::current_exception();
            }
        }
    }
    if (error) std::rethrow_exception(error);
}

std::array<Vec3, 4> pair_positions(const NodeTrianglePair& pair,
    const std::vector<Vec3>& positions) {
    return {positions[pair.node], positions[pair.tri_v[0]],
        positions[pair.tri_v[1]], positions[pair.tri_v[2]]};
}

std::array<Vec3, 4> pair_positions(const SegmentSegmentPair& pair,
    const std::vector<Vec3>& positions) {
    return {positions[pair.v[0]], positions[pair.v[1]],
        positions[pair.v[2]], positions[pair.v[3]]};
}

struct ContactBlock {
    Vec3 normal_gradient = Vec3::Zero();
    Mat33 normal_hessian = Mat33::Zero();
    Vec3 friction_gradient = Vec3::Zero();
    Mat33 friction_hessian = Mat33::Zero();
};

} // namespace

struct DATColoredStepWorkspace::Impl {
    // Compare connectivity VALUES so in-place topology/ownership edits also
    // invalidate the cached elastic colors. Contact pairs never affect them.
    std::vector<int> topology;
    std::vector<int> next_topology;
    std::vector<std::size_t> incident_offsets;
    std::vector<const IncidentTriangles*> adjacency_rows;
    std::vector<std::vector<int>> colors;
    GreedyColoringWorkspace coloring;
    std::vector<IncidentTriangles> incident;
    std::vector<ShapeGrads> rest_gradients;
    std::vector<unsigned char> solid_nodes, surface_nodes, rigid_nodes;
    PinMap pin_map;
    std::vector<ContactBlock> contact_blocks;
    std::vector<NodeTriangleContactEvaluation> nt;
    std::vector<SegmentSegmentContactEvaluation> ss;
    std::vector<FrozenFrictionContact> nt_friction, ss_friction;
    std::vector<int> selected_nodes;
    std::vector<std::uint64_t> frozen_nodes;
    std::uint64_t freeze_generation = 0;
    const BroadPhase* candidates = nullptr;
    SimParams noncontact_params = SimParams::zeros();
    std::size_t count = 0, body_count = 0;
    bool selected_freeze = false;
    bool prepared = false, frozen = false, has_particles = false;
};

DATColoredStepWorkspace::DATColoredStepWorkspace() : impl_(std::make_unique<Impl>()) {}
DATColoredStepWorkspace::~DATColoredStepWorkspace() = default;

const std::vector<ShapeGrads>& DATColoredStepWorkspace::rest_gradients() const {
    return impl_->rest_gradients;
}

const std::vector<std::vector<int>>& DATColoredStepWorkspace::colors() const {
    return impl_->colors;
}

void DATColoredStepWorkspace::prepare(const RefMesh& mesh,
    const VertexTriangleMap& adjacency, const std::vector<Pin>& pins,
    const SimParams& params, std::size_t count, bool hold_rigid_proxies) {
    auto& workspace = *impl_;
    workspace.prepared = workspace.frozen = false;
    const std::size_t body_count = hold_rigid_proxies ? mesh.rb_nodes.size() : 0;
    if (count > std::size_t(std::numeric_limits<int>::max())
        || body_count > std::size_t(std::numeric_limits<int>::max()) - count
        || (hold_rigid_proxies && mesh.node_to_rb.size() != count)
        || mesh.tris.size() % 3 || mesh.tets.size() % 4)
        throw std::invalid_argument("Colored DAT: inconsistent topology extent");
    const auto valid = [count](int node) { return node >= 0 && std::size_t(node) < count; };
    workspace.count = count;
    workspace.body_count = body_count;
    workspace.frozen_nodes.resize(count);
    workspace.freeze_generation = 0;
    workspace.solid_nodes.resize(count);
    workspace.surface_nodes.resize(count);
    workspace.rigid_nodes.resize(count);
    workspace.incident.resize(count);
    workspace.adjacency_rows.resize(count);
    workspace.pin_map.resize(count);
    parallel_for(count, params.use_parallel, [&](std::size_t node) {
        workspace.solid_nodes[node] = 0;
        workspace.surface_nodes[node] = 0;
        workspace.rigid_nodes[node] = 0;
        workspace.incident[node].clear();
        workspace.adjacency_rows[node] = nullptr;
        workspace.pin_map[node] = -1;
        workspace.frozen_nodes[node] = 0;
        if (hold_rigid_proxies) {
            if (mesh.node_to_rb[node] >= 0 && std::size_t(mesh.node_to_rb[node]) >= body_count)
                throw std::invalid_argument("Colored DAT: invalid rigid owner");
            workspace.rigid_nodes[node] = mesh.node_to_rb[node] >= 0;
        }
    });
    // Connectivity can repeat a node. Atomic stores make these membership
    // marks safe without changing their union semantics or allocating per-team
    // masks. Each index is validated before it is used as a destination.
    const std::size_t topology_extent = std::max({mesh.tris.size(), mesh.tets.size(),
        mesh.surface_nodes.size(), mesh.hinges.size(), pins.size()});
    parallel_for(topology_extent, params.use_parallel, [&](std::size_t index) {
        if (index < mesh.tris.size() && !valid(mesh.tris[index]))
            throw std::invalid_argument("Colored DAT: invalid triangle node");
        if (index < mesh.tets.size()) {
            const int node = mesh.tets[index];
            if (!valid(node)) throw std::invalid_argument("Colored DAT: invalid tetrahedron node");
            #pragma omp atomic write
            workspace.solid_nodes[node] = 1;
        }
        if (index < mesh.surface_nodes.size()) {
            const int node = mesh.surface_nodes[index];
            if (!valid(node)) throw std::invalid_argument("Colored DAT: invalid surface node");
            #pragma omp atomic write
            workspace.surface_nodes[node] = 1;
        }
        if (index < mesh.hinges.size())
            for (int node : mesh.hinges[index].v) if (!valid(node))
                throw std::invalid_argument("Colored DAT: invalid hinge node");
        if (index < pins.size() && !valid(pins[index].vertex_index))
            throw std::invalid_argument("Colored DAT: invalid pin node");
    });
    workspace.has_particles = std::any_of(workspace.rigid_nodes.begin(),
        workspace.rigid_nodes.end(), [](unsigned char rigid) { return !rigid; });
    for (const auto& [node, row] : adjacency) {
        if (!valid(node)) throw std::invalid_argument("Colored DAT: invalid adjacency node");
        workspace.adjacency_rows[node] = &row;
    }
    parallel_for(count, params.use_parallel, [&](std::size_t node) {
        const auto* row_ptr = workspace.adjacency_rows[node];
        if (!row_ptr) return;
        const auto& row = *row_ptr;
        for (const auto& [triangle, role] : row)
            if (triangle < 0 || std::size_t(triangle) >= mesh.tris.size() / 3
                || role < 0 || role >= 3 || mesh.tris[3 * triangle + role] != node)
                throw std::invalid_argument("Colored DAT: inconsistent triangle incidence");
        if (!workspace.solid_nodes[node] && !workspace.rigid_nodes[node])
            workspace.incident[node] = row;
    });
    workspace.rest_gradients.resize(mesh.Dm_inverse.size());
    parallel_for(mesh.Dm_inverse.size(), params.use_parallel, [&](std::size_t triangle) {
        workspace.rest_gradients[triangle] = shape_function_gradients(mesh.Dm_inverse[triangle]);
    });
    // Preserve build_pin_map's last-pin-wins behavior for repeated vertices.
    for (std::size_t pin = 0; pin < pins.size(); ++pin)
        workspace.pin_map[pins[pin].vertex_index] = static_cast<int>(pin);
    workspace.noncontact_params = params;
    workspace.noncontact_params.d_hat = 0;
    workspace.noncontact_params.k_barrier = 0;
    (void)workspace.noncontact_params.dt2();

    auto& topology = workspace.next_topology;
    const std::size_t triangle_begin = 2;
    const std::size_t tetrahedron_begin = triangle_begin + mesh.tris.size() + 1;
    const std::size_t hinge_begin = tetrahedron_begin + mesh.tets.size() + 1;
    const std::size_t owner_begin = hinge_begin + mesh.hinges.size() * 4 + 2;
    workspace.incident_offsets.resize(count + 1);
    workspace.incident_offsets[0] = owner_begin + count;
    for (std::size_t node = 0; node < count; ++node)
        workspace.incident_offsets[node + 1] = workspace.incident_offsets[node]
            + 1 + 2 * workspace.incident[node].size();
    topology.resize(workspace.incident_offsets[count]);
    topology[0] = static_cast<int>(count);
    // Negative separators cannot occur in validated connectivity. Retain the
    // same key layout while writing each independent range in parallel.
    topology[triangle_begin - 1] = -1;
    topology[tetrahedron_begin - 1] = -1;
    topology[hinge_begin - 1] = -1;
    topology[owner_begin - 2] = -1;
    topology[owner_begin - 1] = static_cast<int>(body_count);
    // Actual incident rows influence the live elastic gather. Include them
    // deterministically, independent of unordered-map traversal order.
    parallel_for(std::max({mesh.tris.size(), mesh.tets.size(), mesh.hinges.size(), count}),
        params.use_parallel, [&](std::size_t index) {
        if (index < mesh.tris.size()) topology[triangle_begin + index] = mesh.tris[index];
        if (index < mesh.tets.size()) topology[tetrahedron_begin + index] = mesh.tets[index];
        if (index < mesh.hinges.size())
            std::copy(mesh.hinges[index].v, mesh.hinges[index].v + 4,
                topology.begin() + hinge_begin + 4 * index);
        if (index < count) {
            topology[owner_begin + index] = workspace.rigid_nodes[index] ? mesh.node_to_rb[index] + 1 : 0;
            std::size_t offset = workspace.incident_offsets[index];
            topology[offset++] = -1;
            for (const auto& [triangle, role] : workspace.incident[index]) {
                topology[offset++] = triangle;
                topology[offset++] = role;
            }
        }
    });
    if (topology != workspace.topology) {
        std::vector<int> node_to_block(count, -1), blocks;
        for (std::size_t node = 0; node < count; ++node)
            if (!workspace.rigid_nodes[node]) {
                node_to_block[node] = static_cast<int>(blocks.size());
                blocks.push_back(static_cast<int>(node));
            }
        // Each rigid body is one independent generalized block. This model has
        // no elastic terms between bodies, so all body blocks share color zero.
        for (std::size_t body = 0; body < body_count; ++body)
            blocks.push_back(static_cast<int>(count + body));
        std::vector<std::vector<int>> graph(blocks.size());
        const auto clique = [&](const int* element, int size) {
            for (int i = 0; i < size; ++i) {
                if (workspace.rigid_nodes[element[i]]) continue;
                const int a = node_to_block[element[i]];
                for (int j = 0; j < size; ++j) {
                    if (workspace.rigid_nodes[element[j]]) continue;
                    const int b = node_to_block[element[j]];
                    if (a != b) graph[a].push_back(b);
                }
            }
        };
        for (std::size_t triangle = 0; triangle < mesh.tris.size(); triangle += 3)
            clique(mesh.tris.data() + triangle, 3);
        // In particular, include the opposite hinge-apex pair and EVERY
        // tetrahedron edge, which a surface-only elastic graph can omit.
        for (const auto& hinge : mesh.hinges) clique(hinge.v, 4);
        for (std::size_t tet = 0; tet < mesh.tets.size(); tet += 4)
            clique(mesh.tets.data() + tet, 4);
        parallel_for(graph.size(), params.use_parallel, [&](std::size_t index) {
            auto& row = graph[index];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        });
        greedy_color_conflict_graph(graph, workspace.colors, &workspace.coloring);
        parallel_for(workspace.colors.size(), params.use_parallel, [&](std::size_t index) {
            auto& color = workspace.colors[index];
            for (int& block : color) block = blocks[block];
        });
        workspace.topology.swap(topology);
    }
    if (workspace.has_particles) workspace.contact_blocks.resize(count);
    workspace.candidates = nullptr;
    workspace.prepared = true;
}

void DATColoredStepWorkspace::freeze_contacts(const SimParams& params,
    const std::vector<Vec3>& snapshot,
    const std::vector<Vec3>* previous, const BroadPhase& candidates,
    const std::vector<int>* selected_blocks) {
    auto& workspace = *impl_;
    workspace.frozen = false;
    if (!workspace.prepared || snapshot.size() != workspace.count)
        throw std::invalid_argument("Colored DAT: workspace requires prepare()");
    const bool friction = params.friction_coefficient > 0;
    if (friction && (!previous || previous->size() != workspace.count))
        throw std::invalid_argument("Colored DAT: missing friction history");
    workspace.candidates = &candidates;
    workspace.selected_freeze = selected_blocks != nullptr;
    auto& selected_nodes = workspace.selected_nodes;
    if (selected_blocks) {
        selected_nodes.clear();
        for (int block : *selected_blocks) {
            if (block < 0 || std::size_t(block) >= workspace.count + workspace.body_count)
                throw std::invalid_argument("Colored DAT: invalid selected block");
            if (std::size_t(block) < workspace.count && !workspace.rigid_nodes[block])
                selected_nodes.push_back(block);
        }
        // The solver's colors are unique. Deduplicating here also makes the
        // public selected path safe for repeated entries in a supplied list.
        if (!std::is_sorted(selected_nodes.begin(), selected_nodes.end()))
            std::sort(selected_nodes.begin(), selected_nodes.end());
        selected_nodes.erase(std::unique(selected_nodes.begin(), selected_nodes.end()), selected_nodes.end());
        if (++workspace.freeze_generation == 0) {
            std::fill(workspace.frozen_nodes.begin(), workspace.frozen_nodes.end(), 0);
            ++workspace.freeze_generation;
        }
    }
    // A rigid-only color sweep assembles its generalized contact blocks
    // elsewhere and never reads the per-particle blocks.
    if (!workspace.has_particles || (selected_blocks && selected_nodes.empty())) {
        workspace.frozen = true;
        return;
    }
    const std::size_t node_count = selected_blocks ? selected_nodes.size() : workspace.count;
    const auto node_at = [&](std::size_t index) {
        return selected_blocks ? std::size_t(selected_nodes[index]) : index;
    };
    // Initialize every block which can be read by this color, including when
    // its previously active contacts have disappeared. Unselected stale blocks
    // cannot be read through proposal() after a selected freeze.
    parallel_for(node_count, params.use_parallel, [&](std::size_t index) {
        const std::size_t node = node_at(index);
        workspace.contact_blocks[node] = ContactBlock{};
        if (selected_blocks) workspace.frozen_nodes[node] = workspace.freeze_generation;
    });
    if (!(params.d_hat > 0 && params.k_barrier > 0)) {
        workspace.frozen = true;
        return;
    }
    const auto& cache = candidates.cache();
    if (cache.vertex_nt.size() != workspace.count || cache.vertex_ss.size() != workspace.count)
        throw std::invalid_argument("Colored DAT: missing particle contact incidence");
    workspace.nt.resize(cache.nt_pairs.size());
    workspace.ss.resize(cache.ss_pairs.size());
    if (friction) {
        workspace.nt_friction.resize(cache.nt_pairs.size());
        workspace.ss_friction.resize(cache.ss_pairs.size());
    }
    const double dt = params.dt(), scale = params.dt2() * params.k_barrier;
    const double d_hat2 = params.d_hat * params.d_hat;
    // Selection stamps are complete before either joined pair pass starts.
    // Test endpoint membership directly instead of copying/sorting incidence
    // unions; each needed pair is still evaluated exactly once for this color.
    parallel_for(cache.nt_pairs.size(), params.use_parallel, [&](std::size_t index) {
        const auto& pair = cache.nt_pairs[index];
        if (selected_blocks && workspace.frozen_nodes[pair.node] != workspace.freeze_generation
            && std::none_of(pair.tri_v,pair.tri_v+3,[&](int node) {
                return workspace.frozen_nodes[node] == workspace.freeze_generation;
            })) return;
        auto& evaluation = workspace.nt[index];
        evaluation = NodeTriangleContactEvaluation{};
        if (workspace.rigid_nodes[pair.node] && workspace.rigid_nodes[pair.tri_v[0]]
            && workspace.rigid_nodes[pair.tri_v[1]] && workspace.rigid_nodes[pair.tri_v[2]]) return;
        if (workspace.solid_nodes[pair.node] && !workspace.surface_nodes[pair.node]) return;
        for (int node : pair.tri_v)
            if (workspace.solid_nodes[node] && !workspace.surface_nodes[node]) return;
        const auto points = pair_positions(pair, snapshot);
        if (!node_triangle_aabbs_within_distance(points[0], points[1], points[2], points[3], d_hat2)) return;
        evaluation = make_node_triangle_contact_evaluation(points, params.d_hat,
            params.k_barrier, 1e-12, nullptr, params.exact_computation_fallback);
        if (friction && evaluation.active)
            workspace.nt_friction[index] = make_node_triangle_frozen_friction_contact(
                points, pair_positions(pair, *previous), evaluation, dt, params.friction_velocity_epsilon);
    });
    parallel_for(cache.ss_pairs.size(), params.use_parallel, [&](std::size_t index) {
        const auto& pair = cache.ss_pairs[index];
        if (selected_blocks && std::none_of(pair.v,pair.v+4,[&](int node) {
                return workspace.frozen_nodes[node] == workspace.freeze_generation;
            })) return;
        auto& evaluation = workspace.ss[index];
        evaluation = SegmentSegmentContactEvaluation{};
        if (std::all_of(pair.v, pair.v + 4,
                [&](int node) { return workspace.rigid_nodes[node] != 0; })) return;
        for (int node : pair.v)
            if (workspace.solid_nodes[node] && !workspace.surface_nodes[node]) return;
        const auto points = pair_positions(pair, snapshot);
        if (!segment_aabbs_within_distance(points[0], points[1], points[2], points[3], d_hat2)) return;
        evaluation = make_segment_segment_contact_evaluation(points, params.d_hat,
            params.k_barrier, 1e-12, nullptr, params.exact_computation_fallback);
        if (friction && evaluation.active)
            workspace.ss_friction[index] = make_segment_segment_frozen_friction_contact(
                points, pair_positions(pair, *previous), evaluation, dt, params.friction_velocity_epsilon);
    });
    parallel_for(node_count, params.use_parallel, [&](std::size_t selected_index) {
        const std::size_t node = node_at(selected_index);
        if (workspace.rigid_nodes[node]) return;
        auto& block = workspace.contact_blocks[node];
        const auto add = [&](const auto& evaluation, const auto& pair, int role,
                             const FrozenFrictionContact* friction_contact) {
            if (!evaluation.active) return;
            const auto points = pair_positions(pair, snapshot);
            const auto [gradient, hessian] = [&] {
                if constexpr (std::is_same_v<std::decay_t<decltype(evaluation)>, NodeTriangleContactEvaluation>)
                    return node_triangle_barrier_self_gradient_and_hessian(points[0], points[1], points[2], points[3], role, evaluation);
                else
                    return segment_segment_barrier_self_gradient_and_hessian(points[0], points[1], points[2], points[3], role, evaluation);
            }();
            block.normal_gradient += scale * gradient;
            block.normal_hessian += scale * hessian;
            if (friction_contact) {
                const auto derivatives = frozen_friction_role_gradient_and_hessian(
                    *friction_contact, role, params.friction_coefficient, params.dt2());
                block.friction_gradient += derivatives.first;
                block.friction_hessian += derivatives.second;
            }
        };
        for (const auto& entry : cache.vertex_nt[node])
            add(workspace.nt[entry.pair_index], cache.nt_pairs[entry.pair_index], entry.dof,
                friction ? &workspace.nt_friction[entry.pair_index] : nullptr);
        for (const auto& entry : cache.vertex_ss[node])
            add(workspace.ss[entry.pair_index], cache.ss_pairs[entry.pair_index], entry.dof,
                friction ? &workspace.ss_friction[entry.pair_index] : nullptr);
        if (!block.normal_gradient.allFinite() || !block.normal_hessian.allFinite()
            || !block.friction_gradient.allFinite() || !block.friction_hessian.allFinite())
            throw std::runtime_error("Colored DAT: nonfinite frozen contact system");
    });
    workspace.frozen = true;
}

Vec3 DATColoredStepWorkspace::proposal(int node, const RefMesh& mesh,
    const VertexTriangleMap& adjacency, const std::vector<Pin>& pins,
    const SimParams& params, const std::vector<Vec3>& live,
    const std::vector<Vec3>& target, const std::vector<Vec3>* previous) const {
    const auto& workspace = *impl_;
    if (!workspace.frozen || node < 0 || std::size_t(node) >= workspace.count
        || live.size() != workspace.count || target.size() != workspace.count)
        throw std::invalid_argument("Colored DAT: contact snapshot is not prepared");
    if (workspace.rigid_nodes[node]) return Vec3::Zero();
    if (workspace.selected_freeze && workspace.frozen_nodes[node] != workspace.freeze_generation)
        throw std::invalid_argument("Colored DAT: particle was not selected for this contact snapshot");
    const auto local = [&] {
        if (workspace.solid_nodes[node])
            return solid_ipc_detail::compute_solid_local_gradient_and_block_unchecked(
                node, mesh, pins, workspace.noncontact_params, live, target,
                *workspace.candidates, &workspace.solid_nodes, &workspace.surface_nodes,
                &workspace.pin_map, previous);
        return physics_detail::compute_local_gradient_and_hessian_no_barrier_unchecked(
            node, mesh, adjacency, pins, params, live, target, &workspace.pin_map,
            &workspace.incident[node], &workspace.rest_gradients, previous);
    }();
    const auto& contact = workspace.contact_blocks[node];
    Vec3 gradient = local.first;
    Mat33 hessian = local.second;
    gradient += contact.normal_gradient;
    hessian += contact.normal_hessian;
    gradient += contact.friction_gradient;
    hessian += contact.friction_hessian;
    if (!gradient.allFinite() || !hessian.allFinite())
        throw std::runtime_error("Colored DAT: nonfinite local system at vertex " + std::to_string(node));
    const Vec3 update = params.damping == 0 ? Vec3::Zero().eval()
        : (-params.damping * (matrix3d_inverse(hessian) * gradient)).eval();
    if (!update.allFinite())
        throw std::runtime_error("Colored DAT: nonfinite displacement at vertex " + std::to_string(node));
    return update;
}

#endif // IPC_DAT_DIRECTED_ROUNDING_TEST
