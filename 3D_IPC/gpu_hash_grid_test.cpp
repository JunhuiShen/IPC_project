// Correctness tests for gpu_hash_grid_build_pairs vs CPU BroadPhase.
//
// Compares the NT pair set produced by the GPU hash grid against the CPU
// reference produced by BroadPhase::initialize(blue_boxes, mesh, d_hat).
// The two sets must match exactly under canonicalization (pair vertex order
// normalized, lists sorted lexicographically).
//
// On non-CUDA builds the GPU stub forwards to the CPU broad phase, so these
// tests trivially pass — meaningful coverage requires a CUDA build.
#include <gtest/gtest.h>

#include "GPU_Elastic/gpu_hash_grid.h"
#include "broad_phase.h"
#include "example.h"
#include "make_shape.h"
#include "physics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <set>
#include <tuple>
#include <vector>

namespace {

// Canonical form of an NT pair: (node, sorted_tri_v0, sorted_tri_v1, sorted_tri_v2).
// Sorting the triangle indices makes pair identity invariant to corner ordering.
using NTKey = std::tuple<int, int, int, int>;

NTKey canonicalize(const NodeTrianglePair& p) {
    std::array<int, 3> tv = {p.tri_v[0], p.tri_v[1], p.tri_v[2]};
    std::sort(tv.begin(), tv.end());
    return std::make_tuple(p.node, tv[0], tv[1], tv[2]);
}

std::set<NTKey> to_set(const std::vector<NodeTrianglePair>& pairs) {
    std::set<NTKey> s;
    for (const auto& p : pairs) s.insert(canonicalize(p));
    return s;
}

// Canonical form of an SS pair: each edge endpoints sorted internally (min,max),
// then the two edges sorted as a pair. Identity invariant to edge orientation
// and which edge is "first" vs "second."
using SSKey = std::tuple<int, int, int, int>;

SSKey canonicalize(const SegmentSegmentPair& p) {
    int a0 = p.v[0], a1 = p.v[1];
    int b0 = p.v[2], b1 = p.v[3];
    if (a0 > a1) std::swap(a0, a1);
    if (b0 > b1) std::swap(b0, b1);
    if (std::make_pair(a0, a1) > std::make_pair(b0, b1)) {
        std::swap(a0, b0);
        std::swap(a1, b1);
    }
    return std::make_tuple(a0, a1, b0, b1);
}

std::set<SSKey> to_set(const std::vector<SegmentSegmentPair>& pairs) {
    std::set<SSKey> s;
    for (const auto& p : pairs) s.insert(canonicalize(p));
    return s;
}

void print_diff_nt(const std::set<NTKey>& cpu, const std::set<NTKey>& gpu) {
    std::vector<NTKey> cpu_only, gpu_only;
    std::set_difference(cpu.begin(), cpu.end(), gpu.begin(), gpu.end(),
                        std::back_inserter(cpu_only));
    std::set_difference(gpu.begin(), gpu.end(), cpu.begin(), cpu.end(),
                        std::back_inserter(gpu_only));

    std::printf("[diff NT] CPU pairs: %zu  GPU pairs: %zu  intersect: %zu\n",
                cpu.size(), gpu.size(), cpu.size() - cpu_only.size());
    std::printf("[diff NT] CPU-only (missed by GPU): %zu\n", cpu_only.size());
    const std::size_t kPrint = 10;
    for (std::size_t i = 0; i < std::min(kPrint, cpu_only.size()); ++i) {
        const auto& k = cpu_only[i];
        std::printf("          NT node=%d tri=(%d,%d,%d)\n",
                    std::get<0>(k), std::get<1>(k), std::get<2>(k), std::get<3>(k));
    }
    if (cpu_only.size() > kPrint)
        std::printf("          ... (+%zu more)\n", cpu_only.size() - kPrint);

    std::printf("[diff NT] GPU-only (false positives): %zu\n", gpu_only.size());
    for (std::size_t i = 0; i < std::min(kPrint, gpu_only.size()); ++i) {
        const auto& k = gpu_only[i];
        std::printf("          NT node=%d tri=(%d,%d,%d)\n",
                    std::get<0>(k), std::get<1>(k), std::get<2>(k), std::get<3>(k));
    }
    if (gpu_only.size() > kPrint)
        std::printf("          ... (+%zu more)\n", gpu_only.size() - kPrint);
}

void print_diff_ss(const std::set<SSKey>& cpu, const std::set<SSKey>& gpu) {
    std::vector<SSKey> cpu_only, gpu_only;
    std::set_difference(cpu.begin(), cpu.end(), gpu.begin(), gpu.end(),
                        std::back_inserter(cpu_only));
    std::set_difference(gpu.begin(), gpu.end(), cpu.begin(), cpu.end(),
                        std::back_inserter(gpu_only));

    std::printf("[diff SS] CPU pairs: %zu  GPU pairs: %zu  intersect: %zu\n",
                cpu.size(), gpu.size(), cpu.size() - cpu_only.size());
    std::printf("[diff SS] CPU-only (missed by GPU): %zu\n", cpu_only.size());
    const std::size_t kPrint = 10;
    for (std::size_t i = 0; i < std::min(kPrint, cpu_only.size()); ++i) {
        const auto& k = cpu_only[i];
        std::printf("          SS edges=(%d,%d)-(%d,%d)\n",
                    std::get<0>(k), std::get<1>(k), std::get<2>(k), std::get<3>(k));
    }
    if (cpu_only.size() > kPrint)
        std::printf("          ... (+%zu more)\n", cpu_only.size() - kPrint);

    std::printf("[diff SS] GPU-only (false positives): %zu\n", gpu_only.size());
    for (std::size_t i = 0; i < std::min(kPrint, gpu_only.size()); ++i) {
        const auto& k = gpu_only[i];
        std::printf("          SS edges=(%d,%d)-(%d,%d)\n",
                    std::get<0>(k), std::get<1>(k), std::get<2>(k), std::get<3>(k));
    }
    if (gpu_only.size() > kPrint)
        std::printf("          ... (+%zu more)\n", gpu_only.size() - kPrint);
}

struct CpuRefPairs {
    std::vector<NodeTrianglePair>   nt_pairs;
    std::vector<SegmentSegmentPair> ss_pairs;
};

CpuRefPairs cpu_reference(
    const std::vector<Vec3>& positions, const RefMesh& mesh,
    const std::vector<double>& per_vertex_radii, double d_hat)
{
    std::vector<AABB> blue_boxes(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const double r = per_vertex_radii[i];
        blue_boxes[i] = AABB(positions[i] - Vec3::Constant(r),
                             positions[i] + Vec3::Constant(r));
    }
    BroadPhase bp;
    bp.initialize(blue_boxes, mesh, d_hat);
    return CpuRefPairs{bp.cache().nt_pairs, bp.cache().ss_pairs};
}

// Convert GPU's device-resident packed key arrays into NT/SS sets for
// comparison with CPU. NT keys are (node << 32) | tri_idx — we look up the
// triangle's three vertices via mesh.tris. SS keys are (edge_a << 32) | edge_b
// — we look up endpoint vertices via the GPU's cached edge table.
std::set<NTKey> gpu_nt_set(const GpuBroadPhaseResult& g, const RefMesh& mesh) {
    std::set<NTKey> s;
    auto keys = gpu_broad_phase_nt_keys_to_host(g);
    for (auto k : keys) {
        const int node = static_cast<int>(k >> 32);
        const int t    = static_cast<int>(k & 0xFFFFFFFFu);
        NodeTrianglePair p;
        p.node     = node;
        p.tri_v[0] = mesh.tris[3*t + 0];
        p.tri_v[1] = mesh.tris[3*t + 1];
        p.tri_v[2] = mesh.tris[3*t + 2];
        s.insert(canonicalize(p));
    }
    return s;
}

std::set<SSKey> gpu_ss_set(const GpuBroadPhaseResult& g) {
    std::set<SSKey> s;
    auto keys = gpu_broad_phase_ss_keys_to_host(g);
    auto edges = gpu_broad_phase_edges_to_host(g);
    for (auto k : keys) {
        const int ea = static_cast<int>(k >> 32);
        const int eb = static_cast<int>(k & 0xFFFFFFFFu);
        SegmentSegmentPair p;
        p.v[0] = edges[2*ea + 0];
        p.v[1] = edges[2*ea + 1];
        p.v[2] = edges[2*eb + 0];
        p.v[3] = edges[2*eb + 1];
        s.insert(canonicalize(p));
    }
    return s;
}

void check_match(const std::vector<Vec3>& positions, const RefMesh& mesh,
                 double node_box_size, double d_hat) {
    std::vector<double> radii(positions.size(), node_box_size);
    const auto cpu_full = cpu_reference(positions, mesh, radii, d_hat);
    auto gpu_full = gpu_hash_grid_build_pairs(positions, mesh, radii, d_hat);

    const auto cpu_nt = to_set(cpu_full.nt_pairs);
    const auto gpu_nt = gpu_nt_set(gpu_full, mesh);
    if (cpu_nt != gpu_nt) print_diff_nt(cpu_nt, gpu_nt);
    EXPECT_EQ(cpu_nt, gpu_nt) << "NT pair sets differ";

    const auto cpu_ss = to_set(cpu_full.ss_pairs);
    const auto gpu_ss = gpu_ss_set(gpu_full);
    if (cpu_ss != gpu_ss) print_diff_ss(cpu_ss, gpu_ss);
    EXPECT_EQ(cpu_ss, gpu_ss) << "SS pair sets differ";
}

// ---------------------------------------------------------------------------
// Synthetic micro scene: two triangles in a plane, one vertex sandwiched
// between them. Exercises self-incidence skip + simple proximity registration.
// ---------------------------------------------------------------------------
TEST(GpuHashGrid, Synthetic_TwoTriangles) {
    // Vertices:
    //    0──1──2
    //    │ ╱│ ╱│
    //    │╱ │╱ │
    //    3──4──5
    // Two coplanar quads → 4 triangles. Bring vertex 6 close to a triangle.
    std::vector<Vec3> positions = {
        Vec3(0.0, 0.0, 0.0),  // 0
        Vec3(1.0, 0.0, 0.0),  // 1
        Vec3(2.0, 0.0, 0.0),  // 2
        Vec3(0.0, 1.0, 0.0),  // 3
        Vec3(1.0, 1.0, 0.0),  // 4
        Vec3(2.0, 1.0, 0.0),  // 5
        Vec3(0.5, 0.5, 0.05), // 6 — hovers above triangle (0,1,3) / (1,3,4)
    };
    RefMesh mesh;
    mesh.tris = {
        0, 1, 3,
        1, 4, 3,
        1, 2, 4,
        2, 5, 4,
        // triangle 4 contains vertex 6 — but vertex 6 is not a corner of
        // any triangle, so it's a candidate for NT pairs against everything
    };

    check_match(positions, mesh, /*node_box_size=*/0.1, /*d_hat=*/0.05);
}

// ---------------------------------------------------------------------------
// Cloth stack low res — example 2 from the simulator. Small, deterministic,
// has near-contact between layers at t=0.
// ---------------------------------------------------------------------------
TEST(GpuHashGrid, ClothStackLowRes) {
    RefMesh mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    build_cloth_stack_example_low_res(mesh, state, X, pins);

    check_match(state.deformed_positions, mesh,
                /*node_box_size=*/0.1, /*d_hat=*/0.01);
}

// Apply a twist about the y = axis_y axis: at x-position normalized to [0,1]
// across [axis_x_min, axis_x_max], rotate by `total_angle * t`. Pre-twist the
// cloth without running a simulation — gives non-trivial folded geometry for
// stressing the broad phase in the active regime.
void apply_twist(std::vector<Vec3>& positions, double total_angle,
                 double axis_x_min, double axis_x_max, double axis_y) {
    const double range = axis_x_max - axis_x_min;
    for (auto& p : positions) {
        const double t = (p(0) - axis_x_min) / range;
        const double a = total_angle * t;
        const double dy = p(1) - axis_y;
        const double dz = p(2);
        const double ca = std::cos(a), sa = std::sin(a);
        p(1) = axis_y + dy * ca - dz * sa;
        p(2) =         dy * sa + dz * ca;
    }
}

// ---------------------------------------------------------------------------
// Twisting cloth at rest config — example 5. Dense single-sheet mesh, no
// self-contact at t=0 but lots of triangles near each other through the
// blue-box margin. Stresses cell-walking and triangle multiplicity in cells.
// ---------------------------------------------------------------------------
TEST(GpuHashGrid, TwistingClothRest) {
    // Build a small twisting cloth scene by mirroring example 5's setup
    // without going through the IPCArgs3D path (avoids CLI plumbing in tests).
    RefMesh mesh;
    DeformedState state;
    std::vector<Vec2> X;
    std::vector<Pin> pins;
    build_square_mesh(mesh, state, X,
                      /*nx=*/15, /*ny=*/15,
                      /*width=*/1.0, /*height=*/1.0,
                      /*origin=*/Vec3(0.0, 0.0, 0.0));

    check_match(state.deformed_positions, mesh,
                /*node_box_size=*/0.1, /*d_hat=*/0.02);
}

// ---------------------------------------------------------------------------
// Half-twist (π radians end-to-end) — cloth folded over so opposite edges
// face each other. Creates real self-proximity that produces NT pairs the
// rest config never sees.
// ---------------------------------------------------------------------------
TEST(GpuHashGrid, ClothHalfTwist) {
    RefMesh mesh;
    DeformedState state;
    std::vector<Vec2> X;
    build_square_mesh(mesh, state, X,
                      /*nx=*/15, /*ny=*/15, /*width=*/1.0, /*height=*/1.0,
                      /*origin=*/Vec3(0.0, 0.0, 0.0));
    apply_twist(state.deformed_positions, /*total_angle=*/M_PI,
                /*x_min=*/0.0, /*x_max=*/1.0, /*axis_y=*/0.5);

    check_match(state.deformed_positions, mesh,
                /*node_box_size=*/0.1, /*d_hat=*/0.02);
}

// ---------------------------------------------------------------------------
// Full-twist (2π radians) — multi-fold geometry, dense self-proximity,
// many vertices clustered in shared cells. Stress test for the count→scan→
// emit pipeline.
// ---------------------------------------------------------------------------
TEST(GpuHashGrid, ClothFullTwist) {
    RefMesh mesh;
    DeformedState state;
    std::vector<Vec2> X;
    build_square_mesh(mesh, state, X,
                      /*nx=*/15, /*ny=*/15, /*width=*/1.0, /*height=*/1.0,
                      /*origin=*/Vec3(0.0, 0.0, 0.0));
    apply_twist(state.deformed_positions, /*total_angle=*/2.0 * M_PI,
                /*x_min=*/0.0, /*x_max=*/1.0, /*axis_y=*/0.5);

    check_match(state.deformed_positions, mesh,
                /*node_box_size=*/0.1, /*d_hat=*/0.02);
}

}  // namespace
