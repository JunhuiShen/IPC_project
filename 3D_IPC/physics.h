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
    std::vector<Mat22> Dm_inverse;
    std::vector<double> area;

    inline void compute_dm_inverse(){
        Dm_inverse.resize(tris.size());
        area.resize(tris.size());
        for(size_t t=0;t<tris.size();t++){
            const Vec2& X0 = ref_positions[tris[t].v[0]];
            const Vec2& X1 = ref_positions[tris[t].v[1]];
            const Vec2& X2 = ref_positions[tris[t].v[2]];

            Mat22 Dm_local;
            Dm_local.col(0) = X1 - X0;
            Dm_local.col(1) = X2 - X0;
            area[t]=0.5 * std::abs(Dm_local.determinant());
            Dm_inverse[t]=Dm_local.inverse();
        }
    }
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

double compute_incremental_potential_no_barrier(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);

std::pair<Vec3, Mat33> compute_local_gradient_and_hessian_no_barrier(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);

double compute_global_residual(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);