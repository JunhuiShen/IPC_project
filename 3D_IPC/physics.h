#pragma once
#include "corotated_energy.h"
#include "barrier_energy.h"
#include <vector>

struct Tri { int v[3]; };

struct Pin {
    int vertex_index = -1;
    Vec3 target_position = Vec3::Zero();
};

struct SimParams {
    double dt{}, mu{}, lambda{}, density{}, thickness{}, kpin{}, tol_abs{}, step_weight{};
    double d_hat{0.0};          // barrier activation distance (0 = no barrier)
    Vec3 gravity = Vec3::Zero();
    int max_global_iters{};
};

struct DeformedState {
    std::vector<Vec3> deformed_positions;
    std::vector<Vec3> velocities;
};

struct RefMesh {
    std::vector<Vec2> ref_positions;
    std::vector<Tri> tris;
};

struct LumpedMass {
    std::vector<double> vertex_masses;
};

struct VertexAdjacency {
    std::vector<std::vector<int>> incident_triangle_indices;
};

double triangle_ref_area_2d(const RefMesh& ref_mesh, const Tri& tri);

LumpedMass build_lumped_mass(const RefMesh& ref_mesh, double density, double thickness);

VertexAdjacency build_vertex_adjacency(const RefMesh& ref_mesh);

// No-barrier versions: inertia + elastic + gravity + pinning only.
// The Gauss-Seidel solver will call these and add barrier contributions
// separately for AABB-filtered contact pairs.

double compute_incremental_potential_no_barrier(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);

Vec3 compute_local_gradient_no_barrier(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);

Mat33 compute_local_hessian_no_barrier(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x);

double compute_global_residual(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);