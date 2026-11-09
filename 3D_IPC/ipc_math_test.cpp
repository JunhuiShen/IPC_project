#include "IPC_math.h"
#include "physics.h"
#include "broad_phase.h"
#include "make_shape.h"
#include "mesh_utils.h"

#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

// ====================================================================
//  matrix3d_inverse
// ====================================================================

TEST(Matrix3dInverse, IdentityInverse) {
    Mat33 I = Mat33::Identity();
    Mat33 inv = matrix3d_inverse(I);
    EXPECT_NEAR((inv - I).lpNorm<Eigen::Infinity>(), 0.0, 1e-14);
}

TEST(Matrix3dInverse, KnownInverse) {
    Mat33 A;
    A << 2, 1, 0,
         0, 3, 1,
         1, 0, 2;
    Mat33 inv = matrix3d_inverse(A);
    Mat33 prod = A * inv;
    EXPECT_NEAR((prod - Mat33::Identity()).lpNorm<Eigen::Infinity>(), 0.0, 1e-12);
}

TEST(Matrix3dInverse, SingularMatrixReturnsFinitePseudoInverse) {
    Mat33 A = Mat33::Zero();
    A(0, 0) = 2.0;
    A(1, 1) = 3.0;

    Mat33 inv = matrix3d_inverse(A);

    EXPECT_TRUE(inv.allFinite());
    EXPECT_NEAR(inv(0, 0), 0.5, 1e-14);
    EXPECT_NEAR(inv(1, 1), 1.0 / 3.0, 1e-14);
    EXPECT_NEAR(inv(2, 2), 0.0, 1e-14);
    EXPECT_NEAR((A * inv * A - A).lpNorm<Eigen::Infinity>(), 0.0, 1e-12);
}

// ====================================================================
//  index symbols
// ====================================================================

TEST(IndexSymbols, KroneckerDelta) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(kronecker_delta(i, j), i == j ? 1.0 : 0.0);
        }
    }
}

TEST(IndexSymbols, LeviCivita) {
    EXPECT_EQ(levi_civita(0, 1, 2), 1);
    EXPECT_EQ(levi_civita(1, 2, 0), 1);
    EXPECT_EQ(levi_civita(2, 0, 1), 1);
    EXPECT_EQ(levi_civita(0, 2, 1), -1);
    EXPECT_EQ(levi_civita(2, 1, 0), -1);
    EXPECT_EQ(levi_civita(1, 0, 2), -1);
    EXPECT_EQ(levi_civita(0, 0, 1), 0);
    EXPECT_EQ(levi_civita(2, 1, 2), 0);
}

// ====================================================================
//  segment_closest_point
// ====================================================================

TEST(SegmentClosestPoint, InteriorProjection) {
    Vec3 a(0,0,0), b(2,0,0), x(1,1,0);
    double t;
    Vec3 cp = segment_closest_point(x, a, b, t);
    EXPECT_NEAR(t, 0.5, 1e-14);
    EXPECT_NEAR((cp - Vec3(1,0,0)).norm(), 0.0, 1e-14);
}

TEST(SegmentClosestPoint, ClampToEndpoints) {
    Vec3 a(0,0,0), b(1,0,0);
    double t;
    segment_closest_point(Vec3(-1,0,0), a, b, t);
    EXPECT_DOUBLE_EQ(t, 0.0);
    segment_closest_point(Vec3(5,0,0), a, b, t);
    EXPECT_DOUBLE_EQ(t, 1.0);
}

TEST(SegmentClosestPoint, DegenerateZeroLength) {
    Vec3 a(1,2,3), b(1,2,3);
    double t;
    Vec3 cp = segment_closest_point(Vec3(5,5,5), a, b, t);
    EXPECT_DOUBLE_EQ(t, 0.0);
    EXPECT_NEAR((cp - a).norm(), 0.0, 1e-14);
}

// ====================================================================
//  cross_product_in_2d
// ====================================================================

TEST(CrossProduct2D, KnownValue) {
    Vec2 a(1, 0), b(0, 1);
    EXPECT_DOUBLE_EQ(cross_product_in_2d(a, b), 1.0);
    EXPECT_DOUBLE_EQ(cross_product_in_2d(b, a), -1.0);
}

TEST(CrossProduct2D, ParallelVectorsGiveZero) {
    Vec2 a(2, 4), b(1, 2);
    EXPECT_DOUBLE_EQ(cross_product_in_2d(a, b), 0.0);
}

// ====================================================================
//  triangle_plane_barycentric_coordinates
// ====================================================================

TEST(Barycentric, VerticesReturnCorrectCoords) {
    Vec3 x1(0,0,0), x2(1,0,0), x3(0,1,0);
    auto b1 = triangle_plane_barycentric_coordinates(x1, x1, x2, x3);
    EXPECT_NEAR(b1[0], 1.0, 1e-12);
    EXPECT_NEAR(b1[1], 0.0, 1e-12);
    EXPECT_NEAR(b1[2], 0.0, 1e-12);

    auto b2 = triangle_plane_barycentric_coordinates(x2, x1, x2, x3);
    EXPECT_NEAR(b2[1], 1.0, 1e-12);

    auto b3 = triangle_plane_barycentric_coordinates(x3, x1, x2, x3);
    EXPECT_NEAR(b3[2], 1.0, 1e-12);
}

TEST(Barycentric, CentroidIsOneThird) {
    Vec3 x1(0,0,0), x2(3,0,0), x3(0,3,0);
    Vec3 centroid = (x1 + x2 + x3) / 3.0;
    auto b = triangle_plane_barycentric_coordinates(centroid, x1, x2, x3);
    EXPECT_NEAR(b[0], 1.0/3.0, 1e-12);
    EXPECT_NEAR(b[1], 1.0/3.0, 1e-12);
    EXPECT_NEAR(b[2], 1.0/3.0, 1e-12);
}

TEST(Barycentric, DegenerateTriangleReturnsZero) {
    Vec3 x1(0,0,0), x2(1,0,0), x3(2,0,0); // collinear
    auto b = triangle_plane_barycentric_coordinates(Vec3(0.5, 1.0, 0.0), x1, x2, x3);
    EXPECT_DOUBLE_EQ(b[0], 0.0);
    EXPECT_DOUBLE_EQ(b[1], 0.0);
    EXPECT_DOUBLE_EQ(b[2], 0.0);
}

// ====================================================================
//  BroadPhase::set_mesh_topology reuse matches fresh build
// ====================================================================

TEST(SetMeshTopology, CachedTopologyMatchesFreshBuild) {
    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    clear_model(ref_mesh, state, X, pins);
    build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3::Zero());

    const int nv = static_cast<int>(state.deformed_positions.size());
    std::vector<Vec3> v(nv, Vec3(0, -1, 0));
    const double dhat = 0.1;

    // Build without pre-cached topology
    BroadPhase bp_fresh;
    bp_fresh.initialize(state.deformed_positions, v, ref_mesh, 1.0/30.0, dhat);

    // Build with pre-cached topology
    BroadPhase bp_cached;
    bp_cached.set_mesh_topology(ref_mesh, nv);
    EXPECT_TRUE(bp_cached.has_topology());
    bp_cached.initialize(state.deformed_positions, v, ref_mesh, 1.0/30.0, dhat);

    // Same pairs
    EXPECT_EQ(bp_fresh.nt_pairs().size(), bp_cached.nt_pairs().size());
    EXPECT_EQ(bp_fresh.ss_pairs().size(), bp_cached.ss_pairs().size());

    // Verify topology fields match
    const auto& fc = bp_fresh.cache();
    const auto& cc = bp_cached.cache();
    EXPECT_EQ(fc.edges.size(), cc.edges.size());
    for (size_t i = 0; i < fc.edges.size(); ++i) {
        EXPECT_EQ(fc.edges[i][0], cc.edges[i][0]);
        EXPECT_EQ(fc.edges[i][1], cc.edges[i][1]);
    }
}

// ====================================================================
//  stable_norm3
// ====================================================================

namespace {

std::uint64_t sn_bits(double value) {
    std::uint64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

double sn_from_bits(std::uint64_t value) {
    double result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

// Independent oracle: extended-precision libm hypot, sharing no scaling or
// reduction code with the candidate implementation.
long double sn_reference(const Vec3& value) {
    return std::hypot(std::hypot(static_cast<long double>(value[0]),
        static_cast<long double>(value[1])), static_cast<long double>(value[2]));
}

bool sn_accurate(const Vec3& input, double result) {
    const long double reference = sn_reference(input);
    if (reference == 0) return sn_bits(result) == 0;
    const long double max_double = std::numeric_limits<double>::max();
    const long double overflow_threshold =
        std::ldexp(1.0L, 1024) - std::ldexp(1.0L, 970);
    if (reference >= overflow_threshold)
        return std::isinf(result) && result > 0;
    const double rounded_reference = reference > max_double
        ? std::numeric_limits<double>::max() : static_cast<double>(reference);
    if (std::isinf(rounded_reference))
        return std::isinf(result) && result > 0;
    if (!std::isfinite(result) || !(result > 0)) return false;
    const auto result_bits = sn_bits(result), reference_bits = sn_bits(rounded_reference);
    const auto ulps = result_bits > reference_bits
        ? result_bits - reference_bits : reference_bits - result_bits;
    const long double relative = std::abs(static_cast<long double>(result) - reference) / reference;
    return ulps <= 4 || relative <= 8 * std::numeric_limits<double>::epsilon();
}

constexpr std::array<std::size_t, 8> sn_offsets{{0, 8, 16, 24, 32, 40, 48, 56}};

// Place the vector at every packet-relevant offset and require bit-identical
// results across addresses, repetitions, and intervening call history.
double sn_check_placement_invariance(const Vec3& input) {
    alignas(64) unsigned char memory[64 + sizeof(Vec3)];
    Vec3* first = new (memory + sn_offsets[0]) Vec3(input);
    const double expected = stable_norm3(*first);
    first->~Vec3();
    for (std::size_t offset : sn_offsets) {
        Vec3* placed = new (memory + offset) Vec3(input);
        for (int repetition = 0; repetition < 3; ++repetition) {
            const double actual = stable_norm3(*placed);
            EXPECT_EQ(sn_bits(actual), sn_bits(expected))
                << "offset=" << offset << " repetition=" << repetition;
        }
        placed->~Vec3();
    }
    return expected;
}

struct SnGenerator {
    std::uint64_t state = UINT64_C(0x08547ac1badf2934);
    std::uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    double component() {
        const double mantissa = 0.5 + double(next() >> 11) * 0x1p-53;
        const int exponent = int(next() % 2001) - 1000;
        const bool negative = (next() & 1) != 0;
        return std::ldexp(negative ? -mantissa : mantissa, exponent);
    }
    double finite_pattern() {
        const auto pattern = next();
        const auto exponent = next() % 2047;
        return sn_from_bits((pattern & UINT64_C(0x800fffffffffffff))
            | (exponent << 52));
    }
    Vec3 vector(bool patterns) {
        Vec3 result;
        for (int axis = 0; axis < 3; ++axis)
            result[axis] = patterns ? finite_pattern() : component();
        return result;
    }
};

std::vector<Vec3> sn_fixtures() {
    const double tiny = std::numeric_limits<double>::denorm_min();
    const double minimum = std::numeric_limits<double>::min();
    const double largest = std::numeric_limits<double>::max();
    const Vec3 sensitive(0x1.e28f6320108f3p+10,
        0x1.a6a22134e7c66p-1, 0x1.dd76710825405p-12);
    std::vector<Vec3> result{
        sensitive, sensitive * 0x1p-600, sensitive * 0x1p+600,
        Vec3::Zero(), Vec3(-0.0, 0.0, -0.0),
        Vec3(tiny, 0, 0), Vec3(0, tiny, 0), Vec3(0, 0, tiny),
        Vec3(tiny, tiny, tiny), Vec3(2 * tiny, 3 * tiny, 4 * tiny),
        Vec3(minimum, tiny, -minimum),
        Vec3(std::nextafter(minimum, 0.0), minimum, tiny),
        Vec3(1, 2, 3), Vec3(-1, 2, -3), Vec3(3, 4, 0),
        Vec3(largest, 0, 0), Vec3(0, -largest, 0),
        Vec3(largest / 2, largest / 2, largest / 2),
        Vec3(largest, largest, largest),
        Vec3(0x1p-1000, -0x1p+1000, 0x1p-500),
        Vec3(0x1p-1000, 0x1p-1000, 0x1p-1000)
    };
    const std::size_t original_size = result.size();
    for (std::size_t index = 0; index < original_size; ++index) {
        const Vec3 value = result[index];
        result.emplace_back(value[1], value[2], value[0]);
        result.emplace_back(value[2], value[0], value[1]);
    }
    return result;
}

} // namespace

TEST(StableNorm3, KnownAddressSensitiveTripleIsInvariantAndAccurate) {
    const Vec3 sensitive(0x1.e28f6320108f3p+10,
        0x1.a6a22134e7c66p-1, 0x1.dd76710825405p-12);
    const double result = sn_check_placement_invariance(sensitive);
    EXPECT_TRUE(sn_accurate(sensitive, result));
    EXPECT_TRUE(result > 0);
}

TEST(StableNorm3, WideExponentFixturesAreInvariantAndAccurate) {
    for (const Vec3& value : sn_fixtures()) {
        const double result = sn_check_placement_invariance(value);
        EXPECT_TRUE(sn_accurate(value, result))
            << "x=" << value[0] << " y=" << value[1] << " z=" << value[2];
    }
}

TEST(StableNorm3, RandomWideExponentVectorsAreInvariantAndAccurate) {
    SnGenerator generator;
    for (int sample = 0; sample < 4096; ++sample) {
        const Vec3 value = generator.vector((sample & 1) != 0);
        const double result = sn_check_placement_invariance(value);
        EXPECT_TRUE(sn_accurate(value, result))
            << "sample=" << sample;
    }
}

TEST(StableNorm3, AllSignedZerosProducePositiveZero) {
    EXPECT_EQ(sn_bits(stable_norm3(Vec3::Zero())), UINT64_C(0));
    EXPECT_EQ(sn_bits(stable_norm3(Vec3(-0.0, 0.0, -0.0))), UINT64_C(0));
    EXPECT_EQ(sn_bits(stable_norm3(Vec3(-0.0, -0.0, 0.0))), UINT64_C(0));
}

TEST(StableNorm3, SubnormalInputsRemainPositiveAndRepresentable) {
    const double tiny = std::numeric_limits<double>::denorm_min();
    const double minimum = std::numeric_limits<double>::min();
    const std::array<Vec3, 5> cases{{
        Vec3(tiny, 0, 0), Vec3(0, 2 * tiny, 0), Vec3(tiny, tiny, tiny),
        Vec3(minimum, tiny, -minimum),
        Vec3(std::nextafter(minimum, 0.0), minimum, tiny)
    }};
    for (const Vec3& value : cases) {
        const double result = sn_check_placement_invariance(value);
        EXPECT_TRUE(std::isfinite(result) && result > 0);
        EXPECT_TRUE(sn_accurate(value, result));
    }
}

TEST(StableNorm3, ConventionalHypotNonfiniteSemantics) {
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::array<Vec3, 10> cases{{
        Vec3(infinity, 1, 2), Vec3(1, -infinity, 2), Vec3(1, 2, infinity),
        Vec3(infinity, -infinity, infinity),
        Vec3(nan, 1, 2), Vec3(1, nan, 2), Vec3(1, 2, nan),
        Vec3(nan, infinity, 2), Vec3(infinity, nan, 2), Vec3(nan, 1, -infinity)
    }};
    for (const Vec3& value : cases) {
        const double result = sn_check_placement_invariance(value);
        const bool has_infinity = std::isinf(value[0]) || std::isinf(value[1])
            || std::isinf(value[2]);
        if (has_infinity) {
            EXPECT_TRUE(std::isinf(result) && result > 0);
        } else {
            EXPECT_TRUE(std::isnan(result));
        }
    }
}
