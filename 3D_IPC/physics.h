#pragma once
#include "corotated_energy.h"
#include "barrier_energy.h"
#include <unordered_map>
#include <vector>

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
    std::vector<int>  tris; // flat: every 3 ints = one triangle
    std::vector<Mat22> Dm_inverse;
    std::vector<double> area;

    inline void compute_dm_inverse(){
        int nt = static_cast<int>(tris.size()) / 3;
        Dm_inverse.resize(nt);
        area.resize(nt);
        for(int t = 0; t < nt; t++){
            const Vec2& X0 = ref_positions[tris[t*3+0]];
            const Vec2& X1 = ref_positions[tris[t*3+1]];
            const Vec2& X2 = ref_positions[tris[t*3+2]];

            Mat22 Dm_local;
            Dm_local.col(0) = X1 - X0;
            Dm_local.col(1) = X2 - X0;
            area[t] = 0.5 * std::abs(Dm_local.determinant());
            Dm_inverse[t] = Dm_local.inverse();
        }
    }
};

inline int tri_vertex(const RefMesh& ref_mesh, int tri_idx, int local) {
    return ref_mesh.tris[tri_idx * 3 + local];
}

inline int num_tris(const RefMesh& ref_mesh) {
    return static_cast<int>(ref_mesh.tris.size()) / 3;
}

struct LumpedMass {
    std::vector<double> vertex_masses;
};

// Maps each vertex index to the list of triangle indices it belongs to
using VertexTriangleMap = std::unordered_map<int, std::vector<int>>;

double triangle_ref_area_2d(const RefMesh& ref_mesh, int tri_idx);

LumpedMass build_lumped_mass(const RefMesh& ref_mesh, double density, double thickness);

double compute_incremental_potential_no_barrier(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);

std::pair<Vec3, Mat33> compute_local_gradient_and_hessian_no_barrier(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);

Vec3 compute_local_gradient_no_barrier(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);

double compute_global_residual(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);
