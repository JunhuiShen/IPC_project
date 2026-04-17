#include "GPU_Sim/gpu_ccd.h"
#include "solver.h"
#include "make_shape.h"

#include <gtest/gtest.h>
#include <cmath>

// ---------------------------------------------------------------------------
// Shared scene helpers
// ---------------------------------------------------------------------------
namespace {

void build_two_sheets(RefMesh& ref_mesh, DeformedState& state,
                      std::vector<Vec2>& X, std::vector<Pin>& pins) {
    clear_model(ref_mesh, state, X, pins);
    build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, 0.0, 0.0));
    build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, 2.0, 0.0));
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    ref_mesh.build_lumped_mass(1.0, 0.1);
}

void build_one_sheet(RefMesh& ref_mesh, DeformedState& state,
                     std::vector<Vec2>& X, std::vector<Pin>& pins) {
    clear_model(ref_mesh, state, X, pins);
    build_square_mesh(ref_mesh, state, X, 3, 3, 1.0, 1.0, Vec3(0.0, 0.0, 0.0));
    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());
    ref_mesh.build_lumped_mass(1.0, 0.1);
}

// Push two sheets toward each other along y.
std::vector<Vec3> make_collision_xhat(const std::vector<Vec3>& x) {
    std::vector<Vec3> xhat = x;
    const int nv   = static_cast<int>(xhat.size());
    const int half = nv / 2;
    for (int i = 0;    i < half; ++i) xhat[i].y() += 1.5;
    for (int i = half; i < nv;   ++i) xhat[i].y() -= 1.5;
    return xhat;
}

}  // namespace

// ---------------------------------------------------------------------------
// CollisionFreeAndAdvancedTowardXhat
// Two sheets pushed into each other: result should advance but be CCD-limited.
// ---------------------------------------------------------------------------
TEST(GPUCCDInitialGuess, CollisionFreeAndAdvancedTowardXhat) {
    RefMesh ref_mesh; DeformedState state; std::vector<Vec2> X; std::vector<Pin> pins;
    build_two_sheets(ref_mesh, state, X, pins);

    const std::vector<Vec3>& x    = state.deformed_positions;
    const std::vector<Vec3>  xhat = make_collision_xhat(x);
    const int nv = static_cast<int>(x.size());

    const std::vector<Vec3> xnew = gpu_ccd_initial_guess(x, xhat, ref_mesh);

    ASSERT_EQ(static_cast<int>(xnew.size()), nv);

    // Should advance toward xhat.
    double total_displacement = 0.0;
    for (int i = 0; i < nv; ++i)
        total_displacement += (xnew[i] - x[i]).norm();
    EXPECT_GT(total_displacement, 0.0) << "initial guess should advance toward xhat";

    // Should not reach xhat (collision limits the step).
    double total_remaining = 0.0;
    for (int i = 0; i < nv; ++i)
        total_remaining += (xnew[i] - xhat[i]).norm();
    EXPECT_GT(total_remaining, 0.0) << "initial guess should be limited by CCD";

    // All positions finite.
    for (int i = 0; i < nv; ++i) {
        EXPECT_TRUE(std::isfinite(xnew[i].x())) << "vertex " << i;
        EXPECT_TRUE(std::isfinite(xnew[i].y())) << "vertex " << i;
        EXPECT_TRUE(std::isfinite(xnew[i].z())) << "vertex " << i;
    }
}

// ---------------------------------------------------------------------------
// NoCollisionTakesFullStep
// Single sheet, no collision possible: should reach xhat exactly.
// ---------------------------------------------------------------------------
TEST(GPUCCDInitialGuess, NoCollisionTakesFullStep) {
    RefMesh ref_mesh; DeformedState state; std::vector<Vec2> X; std::vector<Pin> pins;
    build_one_sheet(ref_mesh, state, X, pins);

    const std::vector<Vec3>& x = state.deformed_positions;
    std::vector<Vec3> xhat = x;
    for (auto& v : xhat) v += Vec3(0.01, 0.0, 0.0);

    const std::vector<Vec3> xnew = gpu_ccd_initial_guess(x, xhat, ref_mesh);

    for (int i = 0; i < static_cast<int>(xnew.size()); ++i) {
        EXPECT_NEAR(xnew[i].x(), xhat[i].x(), 1e-12) << "vertex " << i;
        EXPECT_NEAR(xnew[i].y(), xhat[i].y(), 1e-12) << "vertex " << i;
        EXPECT_NEAR(xnew[i].z(), xhat[i].z(), 1e-12) << "vertex " << i;
    }
}

// ---------------------------------------------------------------------------
// MatchesCPUResult
// GPU stub and original CPU function should produce identical results.
// ---------------------------------------------------------------------------
TEST(GPUCCDInitialGuess, MatchesCPUResult) {
    RefMesh ref_mesh; DeformedState state; std::vector<Vec2> X; std::vector<Pin> pins;
    build_two_sheets(ref_mesh, state, X, pins);

    const std::vector<Vec3>& x    = state.deformed_positions;
    const std::vector<Vec3>  xhat = make_collision_xhat(x);

    const std::vector<Vec3> x_cpu = ccd_initial_guess(x, xhat, ref_mesh);
    const std::vector<Vec3> x_gpu = gpu_ccd_initial_guess(x, xhat, ref_mesh);

    ASSERT_EQ(x_cpu.size(), x_gpu.size());
    for (int i = 0; i < static_cast<int>(x_cpu.size()); ++i) {
        EXPECT_NEAR(x_cpu[i].x(), x_gpu[i].x(), 1e-12) << "vertex " << i;
        EXPECT_NEAR(x_cpu[i].y(), x_gpu[i].y(), 1e-12) << "vertex " << i;
        EXPECT_NEAR(x_cpu[i].z(), x_gpu[i].z(), 1e-12) << "vertex " << i;
    }
}
