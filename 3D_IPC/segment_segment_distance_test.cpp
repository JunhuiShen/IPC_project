#include "segment_segment_distance.h"
#include "segment_segment_distance_batch.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <gtest/gtest.h>
#include <omp.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

    constexpr double kTol = 1e-10;

    bool approx(double a, double b, double tol = kTol){
        return std::abs(a - b) <= tol;
    }

    bool approx_vec(const Vec3& a, const Vec3& b, double tol = kTol){
        return (a - b).norm() <= tol;
    }

    void print_result(const std::string& name, const SegmentSegmentDistanceResult& r){
        std::cout << name << "\n";
        std::cout << "  region     = " << to_string(r.region) << "\n";
        std::cout << "  s          = " << r.s << "\n";
        std::cout << "  t          = " << r.t << "\n";
        std::cout << "  distance   = " << r.distance << "\n";
        std::cout << "  closest_1  = " << r.closest_point_1.transpose() << "\n";
        std::cout << "  closest_2  = " << r.closest_point_2.transpose() << "\n";
    }

    // Exhaustive exact oracle: keep the original candidate order and strict
    // squared-distance comparison, independently of the production KKT tests.
    SegmentSegmentDistanceResult exhaustive_exact_distance(
            const std::array<Vec3, 4>& x) {
        using Rational = boost::multiprecision::cpp_rational;
        using Vector = std::array<Rational, 3>;
        Vector p, a, b, c;
        for (int axis = 0; axis < 3; ++axis) {
            p[axis] = Rational(x[0][axis]);
            a[axis] = Rational(x[1][axis]) - p[axis];
            b[axis] = Rational(x[3][axis]) - Rational(x[2][axis]);
            c[axis] = p[axis] - Rational(x[2][axis]);
        }
        const auto dot = [](const Vector& u, const Vector& v) -> Rational {
            return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
        };
        const auto clamp = [](const Rational& value) -> Rational {
            if (value < 0) return Rational(0);
            if (value > 1) return Rational(1);
            return value;
        };
        const Rational A = dot(a, a), B = dot(a, b), C = dot(b, b);
        const Rational D = dot(a, c), E = dot(b, c);
        const Rational determinant = A*C - B*B;
        Rational best_s = 0, best_t = 0, best_squared = 0;
        Vector residual;
        bool have_best = false;
        const auto consider = [&](const Rational& s, const Rational& t) {
            const Vector candidate{{c[0]+s*a[0]-t*b[0],
                c[1]+s*a[1]-t*b[1], c[2]+s*a[2]-t*b[2]}};
            const Rational squared = dot(candidate, candidate);
            if (!have_best || squared < best_squared) {
                best_s = s; best_t = t; best_squared = squared;
                residual = candidate; have_best = true;
            }
        };
        if (determinant > 0) {
            const Rational s = (B*E-C*D)/determinant;
            const Rational t = (A*E-B*D)/determinant;
            if (s >= 0 && s <= 1 && t >= 0 && t <= 1) consider(s, t);
        }
        consider(Rational(0), C > 0 ? clamp(E/C) : Rational(0));
        consider(Rational(1), C > 0 ? clamp((E+B)/C) : Rational(0));
        consider(A > 0 ? clamp(-D/A) : Rational(0), Rational(0));
        consider(A > 0 ? clamp((B-D)/A) : Rational(0), Rational(1));

        const auto region = [](const Rational& s, const Rational& t) {
            if (s == 0 && t == 0) return SegmentSegmentRegion::Corner_s0t0;
            if (s == 0 && t == 1) return SegmentSegmentRegion::Corner_s0t1;
            if (s == 1 && t == 0) return SegmentSegmentRegion::Corner_s1t0;
            if (s == 1 && t == 1) return SegmentSegmentRegion::Corner_s1t1;
            if (s == 0) return SegmentSegmentRegion::Edge_s0;
            if (s == 1) return SegmentSegmentRegion::Edge_s1;
            if (t == 0) return SegmentSegmentRegion::Edge_t0;
            if (t == 1) return SegmentSegmentRegion::Edge_t1;
            return SegmentSegmentRegion::Interior;
        };
        SegmentSegmentDistanceResult out;
        out.s = best_s.convert_to<double>();
        out.t = best_t.convert_to<double>();
        const std::array<Rational, 4> weights{{
            Rational(1)-best_s, best_s, best_t-Rational(1), -best_t}};
        for (int role = 0; role < 4; ++role)
            out.weights[role] = weights[role].convert_to<double>();
        for (int axis = 0; axis < 3; ++axis) {
            out.closest_point_1[axis] = Rational(p[axis]+best_s*a[axis]).convert_to<double>();
            out.closest_point_2[axis] = Rational(Rational(x[2][axis])+best_t*b[axis]).convert_to<double>();
            out.separation[axis] = residual[axis].convert_to<double>();
        }
        out.distance = out.separation.stableNorm();
        out.active_region = region(best_s, best_t);
        out.region = determinant == 0 ? SegmentSegmentRegion::ParallelSegments : out.active_region;
        out.robust = true;
        return out;
    }

    void expect_same_exact_result(const SegmentSegmentDistanceResult& actual,
                                  const SegmentSegmentDistanceResult& expected) {
        const auto bits = [](double value) {
            std::uint64_t out;
            std::memcpy(&out, &value, sizeof(out));
            return out;
        };
        EXPECT_EQ(actual.robust, expected.robust);
        EXPECT_EQ(actual.region, expected.region);
        EXPECT_EQ(actual.active_region, expected.active_region);
        EXPECT_EQ(bits(actual.s), bits(expected.s));
        EXPECT_EQ(bits(actual.t), bits(expected.t));
        EXPECT_EQ(bits(actual.distance), bits(expected.distance));
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_EQ(bits(actual.closest_point_1[axis]), bits(expected.closest_point_1[axis]));
            EXPECT_EQ(bits(actual.closest_point_2[axis]), bits(expected.closest_point_2[axis]));
            EXPECT_EQ(bits(actual.separation[axis]), bits(expected.separation[axis]));
        }
        for (int role = 0; role < 4; ++role)
            EXPECT_EQ(bits(actual.weights[role]), bits(expected.weights[role]));
    }

    void expect_batches_match_scalar(const std::vector<std::array<Vec3, 4>>& fixtures,
                                     double forced_eps = 128.0) {
        std::vector<Vec3> positions;
        std::vector<std::array<int, 2>> edges, pairs;
        std::map<std::array<std::uint64_t, 3>, int> vertex_indices;
        for (const auto& fixture : fixtures) {
            std::array<int, 4> vertices;
            for (std::size_t corner = 0; corner < fixture.size(); ++corner) {
                // Share identical source vertices across distinct edges, while
                // retaining signed zeros and NaN payloads as separate inputs.
                std::array<std::uint64_t, 3> key;
                std::memcpy(key.data(), fixture[corner].data(), sizeof(key));
                const auto entry = vertex_indices.emplace(key, static_cast<int>(positions.size()));
                if (entry.second) positions.push_back(fixture[corner]);
                vertices[corner] = entry.first->second;
            }
            const int edge = static_cast<int>(edges.size());
            edges.push_back({vertices[0], vertices[1]});
            edges.push_back({vertices[2], vertices[3]});
            // Repeat/reverse the pair to exercise shared-edge reuse without
            // assuming symmetry of tied closest-feature witnesses.
            pairs.push_back({edge, edge + 1});
            pairs.push_back({edge + 1, edge});
            pairs.push_back({edge, edge + 1});
        }
        SegmentSegmentDistanceBatchWorkspace workspace;
        std::vector<SegmentSegmentDistanceResult> actual;
        const int saved_dynamic = omp_get_dynamic();
        omp_set_dynamic(0);
        for (int threads : {1, 4}) {
            int epoch = 0;
            for (bool exact : {true, false}) {
                for (double eps : {1.0e-12, forced_eps}) {
                    SCOPED_TRACE(::testing::Message() << threads << ':' << exact << ':' << eps);
                    // Reuse the workspace across changed frozen snapshots.
                    // prepare() must not read positions before this workshare.
                    std::vector<Vec3> frozen(positions.size(), Vec3::Zero());
                    const Vec3 translation(0.125 * epoch, -0.25 * epoch, 0.5 * epoch);
                    ++epoch;
                    workspace.prepare(frozen, edges, pairs, eps, exact, actual);
                    #pragma omp parallel num_threads(threads)
                    {
                        #pragma omp for schedule(static)
                        for (std::size_t i = 0; i < positions.size(); ++i)
                            frozen[i] = positions[i] + translation;
                        workspace.compute_collective();
                    }
                    workspace.rethrow_if_failed();
                    EXPECT_FALSE(workspace.failed());
                    std::vector<unsigned char> used(edges.size(), 0);
                    std::size_t fallback_count = 0;
                    for (std::size_t pair = 0; pair < pairs.size(); ++pair) {
                        SCOPED_TRACE(pair);
                        const auto& a = edges[pairs[pair][0]];
                        const auto& b = edges[pairs[pair][1]];
                        // Eigen stableNorm's aligned-prefix split makes the
                        // legacy norm address-dependent. C++17 directs this
                        // scalar return into explicitly aligned storage, as
                        // in the preserved OGC call site; compare every bit.
                        alignas(16) const auto expected = segment_segment_distance(
                            frozen[a[0]], frozen[a[1]], frozen[b[0]], frozen[b[1]], eps, exact);
                        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(expected.separation.data()) % 16u, 0u);
                        expect_same_exact_result(actual[pair], expected);
                        if (expected.robust) {
                            ++fallback_count;
                            used[pairs[pair][0]] = used[pairs[pair][1]] = 1;
                        }
                    }
                    EXPECT_EQ(workspace.fallback_pair_count(), fallback_count);
                    std::size_t used_count = 0;
                    for (unsigned char value : used) used_count += value;
                    EXPECT_EQ(workspace.prepared_edge_count(), used_count);
                }
            }
            // Clear a populated epoch, then reuse its retained capacities.
            const std::vector<Vec3> empty_positions;
            const std::vector<std::array<int, 2>> empty_edges;
            workspace.prepare(empty_positions, empty_edges, empty_edges, 1.0e-12, true, actual);
            #pragma omp parallel num_threads(threads)
            { workspace.compute_collective(); }
            workspace.rethrow_if_failed();
            EXPECT_TRUE(actual.empty());
            EXPECT_EQ(workspace.fallback_pair_count(), 0u);
            EXPECT_EQ(workspace.prepared_edge_count(), 0u);
        }
        omp_set_dynamic(saved_dynamic);
    }

} // namespace

TEST(SegmentSegmentDistance, Interior){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -1.0, 1.0);
    const Vec3 x4(0.5, 1.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Interior case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Interior) << "Interior case: wrong region";
    EXPECT_NEAR(r.distance, 1.0, kTol) << "Interior case: wrong distance";
    EXPECT_NEAR(r.s, 0.5, kTol) << "Interior case: wrong s";
    EXPECT_NEAR(r.t, 0.5, kTol) << "Interior case: wrong t";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0))) << "Interior case: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(0.5, 0.0, 1.0))) << "Interior case: wrong closest_2";
}

TEST(SegmentSegmentDistance, EdgeS0){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(-1.0, -1.0, 1.0);
    const Vec3 x4(-1.0, 1.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Edge s=0 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Edge_s0) << "Edge s=0: wrong region";
    EXPECT_NEAR(r.s, 0.0, kTol) << "Edge s=0: wrong s";
    EXPECT_NEAR(r.t, 0.5, kTol) << "Edge s=0: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(2.0), kTol) << "Edge s=0: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(0.0, 0.0, 0.0))) << "Edge s=0: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(-1.0, 0.0, 1.0))) << "Edge s=0: wrong closest_2";
}

TEST(SegmentSegmentDistance, EdgeS1){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(2.0, -1.0, 1.0);
    const Vec3 x4(2.0, 1.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Edge s=1 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Edge_s1) << "Edge s=1: wrong region";
    EXPECT_NEAR(r.s, 1.0, kTol) << "Edge s=1: wrong s";
    EXPECT_NEAR(r.t, 0.5, kTol) << "Edge s=1: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(2.0), kTol) << "Edge s=1: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(1.0, 0.0, 0.0))) << "Edge s=1: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(2.0, 0.0, 1.0))) << "Edge s=1: wrong closest_2";
}

TEST(SegmentSegmentDistance, EdgeT0){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, 1.0, 1.0);
    const Vec3 x4(0.5, 2.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Edge t=0 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Edge_t0) << "Edge t=0: wrong region";
    EXPECT_NEAR(r.s, 0.5, kTol) << "Edge t=0: wrong s";
    EXPECT_NEAR(r.t, 0.0, kTol) << "Edge t=0: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(2.0), kTol) << "Edge t=0: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0))) << "Edge t=0: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(0.5, 1.0, 1.0))) << "Edge t=0: wrong closest_2";
}

TEST(SegmentSegmentDistance, EdgeT1){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, -2.0, 1.0);
    const Vec3 x4(0.5, -1.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Edge t=1 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Edge_t1) << "Edge t=1: wrong region";
    EXPECT_NEAR(r.s, 0.5, kTol) << "Edge t=1: wrong s";
    EXPECT_NEAR(r.t, 1.0, kTol) << "Edge t=1: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(2.0), kTol) << "Edge t=1: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0))) << "Edge t=1: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(0.5, -1.0, 1.0))) << "Edge t=1: wrong closest_2";
}

TEST(SegmentSegmentDistance, CornerS0T0){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(-1.0, -1.0, 1.0);
    const Vec3 x4(-1.0, -2.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Corner s=0,t=0 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Corner_s0t0) << "Corner s0t0: wrong region";
    EXPECT_NEAR(r.s, 0.0, kTol) << "Corner s0t0: wrong s";
    EXPECT_NEAR(r.t, 0.0, kTol) << "Corner s0t0: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(3.0), kTol) << "Corner s0t0: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, x1)) << "Corner s0t0: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, x3)) << "Corner s0t0: wrong closest_2";
}

TEST(SegmentSegmentDistance, CornerS0T1){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(-2.0, -2.0, 1.0);
    const Vec3 x4(-1.0, -1.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Corner s=0,t=1 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Corner_s0t1) << "Corner s0t1: wrong region";
    EXPECT_NEAR(r.s, 0.0, kTol) << "Corner s0t1: wrong s";
    EXPECT_NEAR(r.t, 1.0, kTol) << "Corner s0t1: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(3.0), kTol) << "Corner s0t1: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, x1)) << "Corner s0t1: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, x4)) << "Corner s0t1: wrong closest_2";
}

TEST(SegmentSegmentDistance, CornerS1T0){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(2.0, -1.0, 1.0);
    const Vec3 x4(2.0, -2.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Corner s=1,t=0 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Corner_s1t0) << "Corner s1t0: wrong region";
    EXPECT_NEAR(r.s, 1.0, kTol) << "Corner s1t0: wrong s";
    EXPECT_NEAR(r.t, 0.0, kTol) << "Corner s1t0: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(3.0), kTol) << "Corner s1t0: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, x2)) << "Corner s1t0: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, x3)) << "Corner s1t0: wrong closest_2";
}

TEST(SegmentSegmentDistance, CornerS1T1){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(3.0, -2.0, 1.0);
    const Vec3 x4(2.0, -1.0, 1.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Corner s=1,t=1 case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::Corner_s1t1) << "Corner s1t1: wrong region";
    EXPECT_NEAR(r.s, 1.0, kTol) << "Corner s1t1: wrong s";
    EXPECT_NEAR(r.t, 1.0, kTol) << "Corner s1t1: wrong t";
    EXPECT_NEAR(r.distance, std::sqrt(3.0), kTol) << "Corner s1t1: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, x2)) << "Corner s1t1: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, x4)) << "Corner s1t1: wrong closest_2";
}

TEST(SegmentSegmentDistance, Parallel){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(2.0, 0.0, 0.0);
    const Vec3 x3(0.5, 1.0, 0.0);
    const Vec3 x4(1.5, 1.0, 0.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Parallel case", r);

    EXPECT_EQ(r.region, SegmentSegmentRegion::ParallelSegments) << "Parallel case: wrong region";
    EXPECT_NEAR(r.distance, 1.0, kTol) << "Parallel case: wrong distance";
}

TEST(SegmentSegmentDistance, ParallelNonOverlapping){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(3.0, 1.0, 0.0);
    const Vec3 x4(4.0, 1.0, 0.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Parallel non-overlapping case", r);

    EXPECT_NEAR(r.distance, std::sqrt(5.0), kTol) << "Parallel non-overlap: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, x2)) << "Parallel non-overlap: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, x3)) << "Parallel non-overlap: wrong closest_2";
}

TEST(SegmentSegmentDistance, DegenerateSegment){
    const Vec3 x1(0.5, 0.0, 0.0);
    const Vec3 x2(0.5, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const Vec3 x4(1.0, 1.0, 0.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Degenerate segment case", r);

    EXPECT_NEAR(r.distance, 1.0, kTol) << "Degenerate segment: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0))) << "Degenerate segment: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(0.5, 1.0, 0.0))) << "Degenerate segment: wrong closest_2";
}

TEST(SegmentSegmentDistance, Symmetry){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.3, -0.5, 0.8);
    const Vec3 x4(0.7, 1.2, 0.8);

    const auto r1 = segment_segment_distance(x1, x2, x3, x4);
    const auto r2 = segment_segment_distance(x3, x4, x1, x2);

    print_result("Symmetry test (order 1)", r1);
    print_result("Symmetry test (order 2)", r2);

    EXPECT_NEAR(r1.distance, r2.distance, kTol) << "Symmetry: distances differ";
    EXPECT_TRUE(approx_vec(r1.closest_point_1, r2.closest_point_2)) << "Symmetry: closest points swapped";
    EXPECT_TRUE(approx_vec(r1.closest_point_2, r2.closest_point_1)) << "Symmetry: closest points swapped";
}

TEST(SegmentSegmentDistance, Touching){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.5, 0.0, 0.0);
    const Vec3 x4(0.5, 1.0, 0.0);

    const auto r = segment_segment_distance(x1, x2, x3, x4);
    print_result("Touching case", r);

    EXPECT_NEAR(r.distance, 0.0, kTol) << "Touching: wrong distance";
    EXPECT_TRUE(approx_vec(r.closest_point_1, Vec3(0.5, 0.0, 0.0))) << "Touching: wrong closest_1";
    EXPECT_TRUE(approx_vec(r.closest_point_2, Vec3(0.5, 0.0, 0.0))) << "Touching: wrong closest_2";
}

TEST(SegmentSegmentDistance, NearParallelStress){
    std::cout << "--- Near-parallel stress ---\n";
    const Vec3 x1(0,0,0), x2(1,0,0);
    const Vec3 x3(0.1, 0.0, 0.3), x4(0.9, 1e-10, 0.3);

    auto r = segment_segment_distance(x1, x2, x3, x4);
    std::cout << "  distance = " << r.distance << "  s=" << r.s << "  t=" << r.t
              << "  region=" << to_string(r.region) << "\n";
    EXPECT_TRUE(std::isfinite(r.distance)) << "near-parallel distance should be finite";
    EXPECT_GT(r.distance, 0.0) << "near-parallel distance should be positive";
    EXPECT_LT(r.distance, 0.31) << "near-parallel distance should be ~0.3";
    EXPECT_GE(r.s, 0.0) << "s should be >= 0";
    EXPECT_LE(r.s, 1.0) << "s should be <= 1";
    EXPECT_GE(r.t, 0.0) << "t should be >= 0";
    EXPECT_LE(r.t, 1.0) << "t should be <= 1";
}

TEST(SegmentSegmentDistance, LargeCoordinatesStress){
    std::cout << "--- Large coordinates stress ---\n";
    const double off = 1e6;
    const Vec3 x1(off, off, off), x2(off+1, off, off);
    const Vec3 x3(off+0.5, off-1, off+0.5), x4(off+0.5, off+1, off+0.5);

    auto r = segment_segment_distance(x1, x2, x3, x4);
    std::cout << "  distance = " << r.distance << "  region=" << to_string(r.region) << "\n";
    EXPECT_TRUE(std::isfinite(r.distance)) << "large-coord distance should be finite";
    EXPECT_EQ(r.region, SegmentSegmentRegion::Interior) << "should be interior region";
    EXPECT_NEAR(r.distance, 0.5, 1e-6) << "large-coord distance should be ~0.5";
}

TEST(SegmentSegmentDistance, VeryShortSegmentStress){
    std::cout << "--- Very short segment stress ---\n";
    const Vec3 x1(0,0,0), x2(1,0,0);
    const Vec3 x3(0.5, 0.5, 0.0), x4(0.5, 0.5, 1e-10);

    auto r = segment_segment_distance(x1, x2, x3, x4);
    std::cout << "  distance = " << r.distance << "  region=" << to_string(r.region) << "\n";
    EXPECT_TRUE(std::isfinite(r.distance)) << "short segment distance should be finite";
    EXPECT_GT(r.distance, 0.0) << "short segment distance should be positive";
}

TEST(SegmentSegmentDistance, SavedExample20ContactPreservesSubUlpSeparation) {
    // The former world-space closest-point subtraction returned exactly zero
    // for this saved contact. These reference values were evaluated with exact
    // rationals from the binary64 input coordinates, not from a distance clamp.
    const std::array<Vec3, 4> x{{
        Vec3(0.30643120172698846, 1.2537041974945615, 0.345215273017485),
        Vec3(0.2716303759805404, 1.2435156840614272, 0.3522099917587509),
        Vec3(0.3476899357030978, 1.254750762137517, 0.34482031348588066),
        Vec3(0.12785112730426101, 1.247125469633767, 0.34839148080369126)}};
    constexpr double exact_gap = 2.81637760742210603e-17;
    const Vec3 exact_separation(
        -1.96611440957398385e-19, 1.63932584825574502e-17,
        2.29002336892925505e-17);
    const auto dr = segment_segment_distance(x[0], x[1], x[2], x[3]);
    ASSERT_EQ(dr.region, SegmentSegmentRegion::Interior);
    EXPECT_TRUE(dr.robust);
    ASSERT_GT(dr.distance, 0.0);
    EXPECT_NEAR(dr.distance, exact_gap, exact_gap * 2.0e-14);
    EXPECT_NEAR(dr.s, 0.0428138387495429762, 3.0e-16);
    EXPECT_NEAR(dr.t, 0.194454706288313905, 3.0e-16);
    EXPECT_LE((dr.separation - exact_separation).norm(), exact_gap * 2.0e-14);
    const Vec3 normal = dr.separation / dr.distance;
    EXPECT_TRUE(normal.allFinite());
    EXPECT_NEAR(normal.norm(), 1.0, 2.0e-14);
    EXPECT_NEAR(normal.dot(x[1] - x[0]), 0.0, 2.0e-16);
    EXPECT_NEAR(normal.dot(x[3] - x[2]), 0.0, 2.0e-16);

    const auto swapped = segment_segment_distance(x[2], x[3], x[0], x[1]);
    const auto reversed = segment_segment_distance(x[1], x[0], x[3], x[2]);
    EXPECT_NEAR(swapped.distance, exact_gap, exact_gap * 2.0e-14);
    EXPECT_NEAR(reversed.distance, exact_gap, exact_gap * 2.0e-14);
    EXPECT_LE((swapped.separation + dr.separation).norm(), exact_gap * 2.0e-14);
    EXPECT_LE((reversed.separation - dr.separation).norm(), exact_gap * 2.0e-14);
    EXPECT_NEAR(swapped.s, dr.t, 3.0e-16);
    EXPECT_NEAR(swapped.t, dr.s, 3.0e-16);
    EXPECT_NEAR(reversed.s, 1.0 - dr.s, 3.0e-16);
    EXPECT_NEAR(reversed.t, 1.0 - dr.t, 3.0e-16);
    expect_batches_match_scalar({x});
}

TEST(SegmentSegmentDistance, NearParallelInteriorIsNotReplacedByAnEndpoint) {
    // A*C-B*B rounds to zero here, but the represented edges are not parallel.
    const Vec3 x0(0.0, 0.0, 0.0), x1(1.0, 0.0, 0.0);
    const Vec3 x2(0.0, -5.0e-9, 0.125), x3(1.0, 5.0e-9, 0.125);
    const auto dr = segment_segment_distance(x0, x1, x2, x3);
    EXPECT_TRUE(dr.robust);
    EXPECT_EQ(dr.region, SegmentSegmentRegion::Interior);
    EXPECT_NEAR(dr.s, 0.5, 1.0e-14);
    EXPECT_NEAR(dr.t, 0.5, 1.0e-14);
    EXPECT_DOUBLE_EQ(dr.distance, 0.125);
    EXPECT_LE((dr.separation - Vec3(0.0, 0.0, -0.125)).norm(), 1.0e-15);
}

TEST(SegmentSegmentDistance, DegenerateEndpointRetainsObliqueSubUlpGap) {
    const double next = std::nextafter(1.5, 2.0);
    const Vec3 point(1.5, next, 1.5);
    const auto dr = segment_segment_distance(
        point, point, Vec3(1.0, 1.0, 1.0), Vec3(2.0, 2.0, 2.0));
    const double ulp = next - 1.5;
    const Vec3 expected(-ulp / 3.0, 2.0 * ulp / 3.0, -ulp / 3.0);
    EXPECT_TRUE(dr.robust);
    ASSERT_GT(dr.distance, 0.0);
    EXPECT_NEAR(dr.distance, ulp * std::sqrt(2.0 / 3.0), ulp * 2.0e-14);
    EXPECT_LE((dr.separation - expected).norm(), ulp * 2.0e-14);
    EXPECT_NEAR(dr.t, 0.5, 3.0e-16);
}

TEST(SegmentSegmentDistance, TranslatedEndpointAndTrueContactAreNotClamped) {
    const double offset = std::ldexp(1.0, 40);
    const double gap = std::nextafter(offset, INFINITY) - offset;
    const Vec3 origin(offset, offset, offset);
    const auto endpoint = segment_segment_distance(
        origin, origin + Vec3(1.0, 0.0, 0.0),
        origin + Vec3(-1.0, gap, 0.0), origin + Vec3(0.0, gap, 0.0));
    EXPECT_DOUBLE_EQ(endpoint.distance, gap);
    EXPECT_DOUBLE_EQ(endpoint.s, 0.0);
    EXPECT_DOUBLE_EQ(endpoint.t, 1.0);
    EXPECT_LE((endpoint.separation - Vec3(0.0, -gap, 0.0)).norm(), gap * 1.0e-14);

    for (const Vec3& translation : {Vec3::Zero().eval(), origin}) {
        const auto crossing = segment_segment_distance(
            translation, translation + Vec3(1.0, 0.0, 0.0),
            translation + Vec3(0.5, -1.0, 0.0),
            translation + Vec3(0.5, 1.0, 0.0));
        EXPECT_DOUBLE_EQ(crossing.distance, 0.0);
        EXPECT_TRUE(crossing.separation.isZero(0.0));
        const auto touching = segment_segment_distance(
            translation, translation + Vec3(1.0, 0.0, 0.0),
            translation + Vec3(1.0, 0.0, 0.0),
            translation + Vec3(2.0, 1.0, 0.0));
        EXPECT_DOUBLE_EQ(touching.distance, 0.0);
        EXPECT_TRUE(touching.separation.isZero(0.0));
    }
}

TEST(SegmentSegmentDistance, RobustEndpointWeightsPrecedeParameterRounding) {
    const double near_one = std::nextafter(1.0, 0.0);
    const double deficit = 1.0 - near_one;
    const Vec3 point(near_one, 2.0, 0.0);
    const auto dr = segment_segment_distance(
        Vec3::Zero(), Vec3(1.0, 2.0, 0.0), point, point);
    ASSERT_TRUE(dr.robust);
    // Exact s is 1-deficit/5: s rounds to one, but 1-s must remain nonzero
    // when constructing the contact weights for gradient/Hessian assembly.
    EXPECT_DOUBLE_EQ(dr.s, 1.0);
    ASSERT_GT(dr.weights[0], 0.0);
    EXPECT_NEAR(dr.weights[0], deficit / 5.0, deficit * 1.0e-15);
    EXPECT_DOUBLE_EQ(dr.weights[1], 1.0);
    EXPECT_DOUBLE_EQ(dr.weights[2], -1.0);
    EXPECT_DOUBLE_EQ(dr.weights[3], 0.0);
    EXPECT_NEAR(dr.distance, deficit * 2.0 / std::sqrt(5.0), deficit * 1.0e-14);
    const Vec3 expected(deficit * 4.0 / 5.0, -deficit * 2.0 / 5.0, 0.0);
    EXPECT_LE((dr.separation - expected).norm(), deficit * 1.0e-14);

    std::vector<std::array<Vec3, 4>> weight_fixtures{{{
        Vec3::Zero(), Vec3(1.0,2.0,0.0), point, point}}};
    // Exact positive parameters can round below the smallest subnormal. Their
    // negative role weights must retain -0, while exact zero remains +0.
    // Reversing the long edge instead makes the exact complementary weight
    // tiny and negative although its rounded parameter is exactly one.
    constexpr std::array<std::array<int, 2>, 3> exponents{{
        {{600,-473}}, {{600,-475}}, {{700,-700}}}};
    for (const auto& exponent : exponents) {
        const double extent = std::ldexp(1.0, exponent[0]);
        const double axial = std::ldexp(1.0, exponent[1]);
        const Vec3 query(axial, 1.0, 0.0);
        for (unsigned permutation = 0; permutation < 4; ++permutation) {
            SCOPED_TRACE(::testing::Message() << exponent[0] << ':' << exponent[1] << ':' << permutation);
            std::array<Vec3, 4> points{{query, query, Vec3::Zero(), Vec3(extent,0,0)}};
            if (permutation & 1u) std::swap(points[2], points[3]);
            if (permutation & 2u) {
                std::swap(points[0], points[2]); std::swap(points[1], points[3]);
            }
            alignas(16) const auto scalar = segment_segment_distance(
                points[0], points[1], points[2], points[3]);
            alignas(16) const auto exhaustive = exhaustive_exact_distance(points);
            expect_same_exact_result(scalar, exhaustive);
            if (permutation == 0u) {
                EXPECT_TRUE(std::signbit(scalar.weights[3]));
                if (exponent[1] - exponent[0] <= -1075) {
                    EXPECT_DOUBLE_EQ(scalar.t, 0.0);
                    EXPECT_DOUBLE_EQ(scalar.weights[3], 0.0);
                }
            } else if (permutation == 1u) {
                EXPECT_DOUBLE_EQ(scalar.t, 1.0);
                EXPECT_TRUE(std::signbit(scalar.weights[2]));
            }
            weight_fixtures.push_back(points);
        }
    }
    expect_batches_match_scalar(weight_fixtures);
}

TEST(SegmentSegmentDistance, ExactBoundarySelectionMatchesExhaustiveReference) {
    // Force the exact path for a unit perpendicular edge pair. These nine
    // placements cover the interior, all four edges and all four corners.
    constexpr std::array<SegmentSegmentRegion, 9> regions{{
        SegmentSegmentRegion::Corner_s0t1, SegmentSegmentRegion::Edge_s0,
        SegmentSegmentRegion::Corner_s0t0, SegmentSegmentRegion::Edge_t1,
        SegmentSegmentRegion::Interior, SegmentSegmentRegion::Edge_t0,
        SegmentSegmentRegion::Corner_s1t1, SegmentSegmentRegion::Edge_s1,
        SegmentSegmentRegion::Corner_s1t0}};
    int fixture = 0;
    std::vector<std::array<Vec3, 4>> batch_fixtures;
    for (double horizontal : {-1.0, 0.5, 2.0}) {
        for (double vertical : {-2.0, -0.5, 1.0}) {
            SCOPED_TRACE(fixture);
            const std::array<Vec3, 4> x{{Vec3::Zero(), Vec3::UnitX(),
                Vec3(horizontal, vertical, 0.125),
                Vec3(horizontal, vertical + 1.0, 0.125)}};
            batch_fixtures.push_back(x);
            const auto expected = exhaustive_exact_distance(x);
            EXPECT_EQ(expected.active_region, regions[fixture++]);
            expect_same_exact_result(segment_segment_distance(
                x[0], x[1], x[2], x[3], 2.0, true), expected);
        }
    }

    // Integer-derived binary64 endpoints exercise oblique boundaries, tied
    // minima, intersections and degenerate segments with exact reproducibility.
    std::uint32_t seed = 0x6a09e667u;
    for (int sample = 0; sample < 96; ++sample) {
        SCOPED_TRACE(sample);
        std::array<Vec3, 4> x;
        for (Vec3& point : x) {
            for (int axis = 0; axis < 3; ++axis) {
                seed = 1664525u * seed + 1013904223u;
                point[axis] = (static_cast<int>((seed >> 16) % 17u) - 8) * 0.25;
            }
        }
        expect_same_exact_result(segment_segment_distance(
            x[0], x[1], x[2], x[3], 128.0, true), exhaustive_exact_distance(x));
        batch_fixtures.push_back(x);
    }
    // Invalid endpoints retain scalar fast-path handling and never enter a
    // rational cache, even alongside finite edges that do require fallback.
    batch_fixtures.push_back({{Vec3::Zero(), Vec3::UnitX(),
        Vec3(0, std::numeric_limits<double>::infinity(), 0), Vec3(1, 1, 0)}});
    batch_fixtures.push_back({{Vec3::Zero(), Vec3::UnitX(),
        Vec3(0, std::numeric_limits<double>::quiet_NaN(), 0), Vec3(1, 1, 0)}});
    expect_batches_match_scalar(batch_fixtures);
    // A nonempty batch with no exact fallbacks skips edge preparation too.
    expect_batches_match_scalar({{{Vec3::Zero(), Vec3::UnitX(),
        Vec3(0.5, -1, 1), Vec3(0.5, 1, 1)}}});

    // Preserve numerator-first feasibility and clipping at each boundary and
    // corner even when exact coefficients are huge or their floating Gram
    // determinant underflows. Negative scales also reverse both directions.
    std::vector<std::array<Vec3, 4>> scaled_boundaries;
    const double forced_eps = std::ldexp(1.0, 401);
    for (double scale : {std::ldexp(1.0, -520), -std::ldexp(1.0, -520),
                         std::ldexp(1.0, 200), -std::ldexp(1.0, 200)}) {
        for (double horizontal : {-1.0, 0.5, 2.0}) {
            for (double vertical : {-2.0, -0.5, 1.0}) {
                const std::array<Vec3, 4> points{{Vec3::Zero(), scale * Vec3::UnitX(),
                    scale * Vec3(horizontal, vertical, 0.125),
                    scale * Vec3(horizontal, vertical + 1.0, 0.125)}};
                alignas(16) const auto scalar = segment_segment_distance(
                    points[0], points[1], points[2], points[3], forced_eps, true);
                alignas(16) const auto exhaustive = exhaustive_exact_distance(points);
                expect_same_exact_result(scalar, exhaustive);
                scaled_boundaries.push_back(points);
            }
        }
    }
    expect_batches_match_scalar(scaled_boundaries, forced_eps);
}

TEST(SegmentSegmentDistance, ExactParallelTiesAndDegeneracyMatchExhaustiveReference) {
    const std::array<std::array<Vec3, 4>, 8> fixtures{{
        // Far parallel edges still enter the exact fallback.
        {{Vec3(0,0,0), Vec3(1,0,0), Vec3(0,8,0), Vec3(1,8,0)}},
        // Full overlap, partial overlap, containment and disjoint projections.
        {{Vec3(0,0,0), Vec3(1,0,0), Vec3(0,0.125,0), Vec3(1,0.125,0)}},
        {{Vec3(0,0,0), Vec3(2,0,0), Vec3(1,0.125,0), Vec3(3,0.125,0)}},
        {{Vec3(0,0,0), Vec3(3,0,0), Vec3(1,0.125,0), Vec3(2,0.125,0)}},
        {{Vec3(0,0,0), Vec3(1,0,0), Vec3(2,0.125,0), Vec3(3,0.125,0)}},
        // Either segment, or both segments, have zero length.
        {{Vec3(0.5,0,0), Vec3(0.5,0,0), Vec3(0,0.125,0), Vec3(1,0.125,0)}},
        {{Vec3(0,0,0), Vec3(1,0,0), Vec3(0.5,0.125,0), Vec3(0.5,0.125,0)}},
        {{Vec3(0,0,0), Vec3(0,0,0), Vec3(0,0.125,0), Vec3(0,0.125,0)}}
    }};
    std::vector<std::array<Vec3, 4>> batch_fixtures;
    for (std::size_t fixture = 0; fixture < fixtures.size(); ++fixture) {
        for (unsigned permutation = 0; permutation < 8; ++permutation) {
            SCOPED_TRACE(::testing::Message() << fixture << ':' << permutation);
            auto x = fixtures[fixture];
            if (permutation & 1u) std::swap(x[0], x[1]);
            if (permutation & 2u) std::swap(x[2], x[3]);
            if (permutation & 4u) {
                std::swap(x[0], x[2]); std::swap(x[1], x[3]);
            }
            expect_same_exact_result(segment_segment_distance(
                x[0], x[1], x[2], x[3]), exhaustive_exact_distance(x));
            for (int axis = 0; axis < 3; ++axis) {
                std::array<Vec3, 4> rotated;
                for (int node = 0; node < 4; ++node) {
                    rotated[node][axis] = x[node][0];
                    rotated[node][(axis + 1) % 3] = x[node][1];
                    rotated[node][(axis + 2) % 3] = x[node][2];
                }
                batch_fixtures.push_back(rotated);
            }
            // Non-axis parallel edges must use the same interval shortcut
            // after exact projection, including non-dyadic closest parameters.
            // A nonperpendicular offset shifts their projected intervals; a
            // zero offset covers collinear contact/overlap/touch ties.
            for (const Vec3& direction : {Vec3(1,1,0), Vec3(1,-2,3), Vec3(0,2,-1)}) {
                for (bool contact : {false, true}) {
                    std::array<Vec3, 4> oblique;
                    for (int node = 0; node < 4; ++node) {
                        oblique[node] = x[node][0] * direction;
                        if (!contact)
                            oblique[node] += x[node][1] * Vec3(0.5,1,-0.25);
                    }
                    alignas(16) const auto scalar = segment_segment_distance(
                        oblique[0], oblique[1], oblique[2], oblique[3]);
                    alignas(16) const auto exhaustive = exhaustive_exact_distance(oblique);
                    expect_same_exact_result(scalar, exhaustive);
                    batch_fixtures.push_back(oblique);
                }
            }
        }
    }
    // Unequal dyadic interval lengths exercise exact parameter cancellation
    // for either direction on every axis. Large axial offsets coexist with
    // tiny perpendicular gaps, which must survive rounded world witnesses.
    std::uint32_t seed = 0xbb67ae85u;
    for (int axis = 0; axis < 3; ++axis) {
        for (int sample = 0; sample < 48; ++sample) {
            SCOPED_TRACE(::testing::Message() << axis << ':' << sample);
            const auto next = [&]() {
                seed = 1664525u * seed + 1013904223u;
                return seed;
            };
            const double offset = sample % 3 == 0 ? std::ldexp(1.0, 40) : 0.0;
            const double start_a = offset + (static_cast<int>(next() % 65u) - 32) * 0.125;
            const double start_b = offset + (static_cast<int>(next() % 65u) - 32) * 0.125;
            double u = (1u + (next() >> 16) % 31u) * 0.125;
            double v = (1u + (next() >> 16) % 31u) * 0.0625;
            if (sample & 1) u = -u;
            if (sample & 2) v = -v;
            std::array<Vec3, 4> x{{Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()}};
            x[0][axis] = start_a; x[1][axis] = start_a + u;
            x[2][axis] = start_b; x[3][axis] = start_b + v;
            const double gap = sample % 3 == 0 ? std::ldexp(1.0, -80) : 0.125;
            x[2][(axis + 1) % 3] = x[3][(axis + 1) % 3] = gap;
            x[2][(axis + 2) % 3] = x[3][(axis + 2) % 3] = gap * 0.25;
            alignas(16) const auto scalar = segment_segment_distance(x[0], x[1], x[2], x[3]);
            alignas(16) const auto exhaustive = exhaustive_exact_distance(x);
            expect_same_exact_result(scalar, exhaustive);
            batch_fixtures.push_back(x);
        }
    }
    // A tied parallel overlap keeps the earliest original boundary witness.
    const auto tied = segment_segment_distance(fixtures[1][0], fixtures[1][1],
        fixtures[1][2], fixtures[1][3]);
    EXPECT_DOUBLE_EQ(tied.s, 0.0);
    EXPECT_DOUBLE_EQ(tied.t, 0.0);
    EXPECT_EQ(tied.active_region, SegmentSegmentRegion::Corner_s0t0);
    expect_batches_match_scalar(batch_fixtures);
}

TEST(SegmentSegmentDistance, ExactEarlySelectionRetainsNearParallelFallbackPolicy) {
    const std::array<Vec3, 4> x{{Vec3::Zero(), Vec3::UnitX(),
        Vec3(0.0, -5.0e-9, 0.125), Vec3(1.0, 5.0e-9, 0.125)}};
    const auto exact = segment_segment_distance(x[0], x[1], x[2], x[3], 1.0e-12, true);
    expect_same_exact_result(exact, exhaustive_exact_distance(x));
    EXPECT_EQ(exact.active_region, SegmentSegmentRegion::Interior);
    EXPECT_DOUBLE_EQ(exact.s, 0.5);
    EXPECT_DOUBLE_EQ(exact.t, 0.5);

    // Disabling fallback still uses the legacy floating parallel solve.
    const auto legacy = segment_segment_distance(x[0], x[1], x[2], x[3], 1.0e-12, false);
    EXPECT_FALSE(legacy.robust);
    EXPECT_EQ(legacy.region, SegmentSegmentRegion::ParallelSegments);
    EXPECT_DOUBLE_EQ(legacy.s, 0.0);
    EXPECT_LT(legacy.t, 1.0e-15);
    EXPECT_DOUBLE_EQ(legacy.distance, 0.125);
    std::vector<std::array<Vec3, 4>> batch_fixtures{x};
    // The floating Gram determinant can vanish although these oblique edges
    // are not exactly parallel. The new shortcut must require exact zero.
    const std::array<Vec3, 4> oblique{{Vec3::Zero(), Vec3(1,1,0),
        Vec3(0,-5.0e-9,0.125), Vec3(1,1.0+5.0e-9,0.125)}};
    for (unsigned permutation = 0; permutation < 8; ++permutation) {
        auto points = oblique;
        if (permutation & 1u) std::swap(points[0], points[1]);
        if (permutation & 2u) std::swap(points[2], points[3]);
        if (permutation & 4u) {
            std::swap(points[0], points[2]); std::swap(points[1], points[3]);
        }
        alignas(16) const auto scalar = segment_segment_distance(
            points[0], points[1], points[2], points[3], 128.0, true);
        alignas(16) const auto exhaustive = exhaustive_exact_distance(points);
        expect_same_exact_result(scalar, exhaustive);
        EXPECT_NE(scalar.region, SegmentSegmentRegion::ParallelSegments);
        batch_fixtures.push_back(points);
    }
    expect_batches_match_scalar(batch_fixtures);
}
