#pragma once
#include "corotated_energy.h"
#include "barrier_energy.h"
#include <unordered_map>
#include <vector>
#include <cassert>
#include <string>

struct Tri { int v[3]; };

struct Pin {
    int vertex_index = -1;
    Vec3 target_position = Vec3::Zero();
};

struct SimParams {
    double fps{30.0};
    int    substeps{1};
    double mu{}, lambda{}, density{}, thickness{}, kpin{}, tol_abs{}, step_weight{};
    double d_hat{0.0};
    Vec3 gravity = Vec3::Zero();
    int max_global_iters{};

    int    restart_frame{-1};  // -1 = no restart

    double dt()  const { return 1.0 / (fps * static_cast<double>(substeps)); }
    double dt2() const { double d = dt(); return d * d; }  // cached square
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
    std::vector<double> mass;
    size_t num_positions;

    inline void initialize(const std::vector<Vec2>& X){
        num_positions = X.size();
        compute_dm_inverse(X);
    }

    inline void compute_dm_inverse(const std::vector<Vec2>& X){
        int nt = static_cast<int>(tris.size()) / 3;
        Dm_inverse.resize(nt);
        area.resize(nt);
        for(int t = 0; t < nt; t++){
            const Vec2& X0 = X[tris[t*3+0]];
            const Vec2& X1 = X[tris[t*3+1]];
            const Vec2& X2 = X[tris[t*3+2]];
            Mat22 Dm_local;
            Dm_local.col(0) = X1 - X0;
            Dm_local.col(1) = X2 - X0;
            area[t] = 0.5 * std::abs(Dm_local.determinant());
            Dm_inverse[t] = Dm_local.inverse();
        }
    }

    inline void assert_valid() const {
        assert(area.size() == Dm_inverse.size());
        assert(mass.size() == num_positions);
    }

    inline void build_lumped_mass(double density, double thickness) {
        mass.assign(num_positions, 0.0);
        int nt = static_cast<int>(tris.size()) / 3;
        for (int t = 0; t < nt; ++t) {
            double m = density * area[t] * thickness;
            double mv = m / 3.0;
            for (int a = 0; a < 3; ++a) mass[tris[t * 3 + a]] += mv;
        }
    }
};

inline int tri_vertex(const RefMesh& ref_mesh, int tri_idx, int local) {
    return ref_mesh.tris[tri_idx * 3 + local];
}

inline int num_tris(const RefMesh& ref_mesh) {
    return static_cast<int>(ref_mesh.tris.size()) / 3;
}

// Maps each vertex to {triangle_index, local_node_index} pairs.
// Storing local_node_index eliminates the linear search in local_node() at call sites.
using VertexTriangleMap = std::unordered_map<int, std::vector<std::pair<int,int>>>;

//  Barrier contact pair types
struct NodeTrianglePair {
    int node;
    int tri_v[3];
};

struct SegmentSegmentPair {
    int v[4];
};

struct BarrierPairs {
    std::vector<NodeTrianglePair>   nt;
    std::vector<SegmentSegmentPair> ss;
};

BarrierPairs build_barrier_pairs(const RefMesh& ref_mesh);

//  Physics functions
double triangle_ref_area_2d(const RefMesh& ref_mesh, int tri_idx);

double compute_incremental_potential_no_barrier(const RefMesh& ref_mesh, const std::vector<Pin>& pins,
                                                const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);

std::pair<Vec3, Mat33> compute_local_gradient_and_hessian_no_barrier(int vi, const RefMesh& ref_mesh,
                                                                     const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                                                     const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);

Vec3 compute_local_gradient(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                            const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                            const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs);

double compute_global_residual(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                               const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                               const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs);

void serialize_state(const std::string& dir, int frame, const DeformedState& state);
bool deserialize_state(const std::string& dir, int frame, DeformedState& state);