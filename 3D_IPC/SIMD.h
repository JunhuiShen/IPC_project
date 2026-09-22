#pragma once

#include "IPC_math.h"

#include <array>
#include <cstddef>
#include <optional>

struct FrozenFrictionContact;
struct SDFEvaluation;

// Lanes hold independent vertices, triangles, hinges, or contacts.
// The caller owns accumulation and scheduling.
namespace ipc_simd {

// V2's organizer gathers AoS entries before calling these kernels. Each call
// transposes only a local tile and returns private AoS contributions. They do
// not read mesh indices or scatter into shared vertices. Elasticity and bending
// outputs include rest area/coefficient; dt^2 is applied during accumulation.
inline constexpr std::size_t tile_width = 8;
inline constexpr std::size_t contact_tile_width = 32;
const char* tile_backend_name();
void corotated_derivatives_tile(
    const Vec3* positions, const Mat22* dm_inverse, const double* areas,
    const Vec2* shape_gradients, std::size_t count, double mu, double lambda,
    Vec3* gradients, Mat33* hessians);
void bending_derivatives_tile(
    const Vec3* positions, const int* active_nodes, const double* coefficients,
    const double* rest_angles, std::size_t count, double stiffness,
    Vec3* gradients, Mat33* hessians);

struct PointInput {
    double mass;
    Vec3 position;
    Vec3 predicted_position;
    std::optional<Vec3> pin_target;
};

// Complete point contributions, in inertia/gravity/pin order. Gravity and pin
// terms include dt^2; each SIMD lane owns one independent vertex.
void point_derivatives_tile(const PointInput* inputs, std::size_t count,
    const Vec3& gravity, double kpin, double dt2,
    Vec3* gradients, Mat33* hessians);

struct MeshContactInput {
    std::array<Vec3, 4> positions;
    int role = 0;
    bool segment_segment = false;
    std::array<Vec3, 4> previous_positions;
};

struct MeshContactOutput {
    Vec3 gradient;
    Mat33 hessian;
    Vec3 friction_gradient;
    Mat33 friction_hessian;
};

// Contact tiles hold up to contact_tile_width private AoS records, with each
// feature evaluated in hardware-width packets. Normal outputs are unscaled;
// friction outputs include dt^2, matching the scalar assembly convention.
// Every field of the first count output records is overwritten. Optional
// derivative_active entries are zero only when all four output blocks are
// zero; one means consume normally. These flags carry no CCD information.
void mesh_contact_derivatives_tile(const MeshContactInput* inputs, std::size_t count,
    double d_hat, double k_barrier, double friction, double dt, double eps_v,
    MeshContactOutput* outputs, unsigned char* derivative_active = nullptr);
void sdf_derivatives_tile(const SDFEvaluation* evaluations, std::size_t count,
    double stiffness, double epsilon, Vec3* gradients, Mat33* hessians,
    bool include_curvature = false);
void friction_derivatives_tile(const FrozenFrictionContact* contacts, const int* roles,
    std::size_t count, double friction, double dt2, Vec3* gradients, Mat33* hessians);

const char* backend_name();

} // namespace ipc_simd
