#pragma once
#include "corotated_energy.h"
#include "bending_energy.h"
#include "barrier_energy.h"
#include <algorithm>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>

class BroadPhase;
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
    double tol_rel{0.0};  // relative tolerance (factor of initial residual); 0 disables
    double kB{0.0};  // bending (flexural) stiffness; 0 disables the bending term
    double d_hat{0.0};
    Vec3 gravity = Vec3::Zero();
    int max_global_iters{};

    int    restart_frame{-1};  // -1 = no restart
    bool   use_parallel{false};

    // Post-sweep CCD penetration check. Off by default.
    bool   ccd_check{false};

    // Use the trust-region narrow phase instead of CCD for step clamping.
    bool   use_trust_region{false};

    // Rebuild the broad-phase BVH per moved vertex during the GS sweep.
    bool   use_incremental_refresh{false};

    // Divide the per-vertex gradient by vertex mass when forming the global
    // residual. Makes the convergence criterion scale-invariant with mass so
    // the same tol_abs works for heavy ground and light cloth vertices.
    bool   mass_normalize_residual{false};

    // Route the per-substep Gauss-Seidel sweep through the GPU implementation
    // (gpu_gauss_seidel_solver).  On machines without CUDA the CPU stub is
    // used automatically.  Incompatible with use_parallel and
    // use_incremental_refresh (those are ignored when use_gpu is set).
    bool   use_gpu{false};

    double dt()  const {
        if (cached_dt_ < 0.0) cached_dt_ = 1.0 / (fps * static_cast<double>(substeps));
        return cached_dt_;
    }
    double dt2() const {
        if (cached_dt2_ < 0.0) { double d = dt(); cached_dt2_ = d * d; }
        return cached_dt2_;
    }
    // Call after mutating fps or substeps.
    void invalidate_dt_cache() const { cached_dt_ = -1.0; cached_dt2_ = -1.0; }

private:
    mutable double cached_dt_ = -1.0;
    mutable double cached_dt2_ = -1.0;
public:
};

struct DeformedState {
    std::vector<Vec3> deformed_positions;
    std::vector<Vec3> velocities;
};

// Discrete-shell hinge: two triangles sharing an edge.
// v[0..1] are the shared edge endpoints, v[2..3] the two apices. A/B
// orientation is fixed by build_hinges() so m_A, m_B agree when flat.
struct Hinge {
    int    v[4];
    double bar_theta;  // rest dihedral complement (0 for flat 2D rest)
    double c_e;        // |e|^2 / (A_A + A_B)
};

// vertex → {hinge_index, local_role ∈ {0..3}}
using VertexHingeMap = std::unordered_map<int, std::vector<std::pair<int,int>>>;

struct RefMesh {
    std::vector<Vec2> ref_positions;
    std::vector<int>  tris; // flat: every 3 ints = one triangle
    std::vector<Mat22> Dm_inverse;
    std::vector<double> area;
    std::vector<double> mass;
    std::vector<Hinge> hinges;
    VertexHingeMap hinge_adj;
    size_t num_positions;

    inline void initialize(const std::vector<Vec2>& X){
        num_positions = X.size();
        compute_dm_inverse(X);
        build_hinges(X);
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

    inline void build_hinges(const std::vector<Vec2>& X) {
        // dir = 0 means the triangle traverses this edge in v_min→v_max
        // order; dir = 1 means v_max→v_min. Pairing one of each yields a
        // consistently oriented hinge below.
        struct EdgeEntry { int tri; int dir; int apex; };
        std::map<std::pair<int,int>, std::vector<EdgeEntry>> edge_map;

        const int nt = static_cast<int>(tris.size()) / 3;
        for (int t = 0; t < nt; ++t) {
            const int v[3] = { tris[3*t+0], tris[3*t+1], tris[3*t+2] };
            for (int k = 0; k < 3; ++k) {
                const int va = v[k];
                const int vb = v[(k+1)%3];
                const int vc = v[(k+2)%3];
                const int vmin = std::min(va, vb);
                const int vmax = std::max(va, vb);
                const int dir  = (va == vmin) ? 0 : 1;
                edge_map[{vmin, vmax}].push_back({t, dir, vc});
            }
        }

        hinges.clear();
        hinge_adj.clear();
        for (const auto& [edge, entries] : edge_map) {
            if (entries.size() != 2) continue;  // boundary or non-manifold

            const EdgeEntry* triA = nullptr;
            const EdgeEntry* triB = nullptr;
            for (const auto& e : entries) {
                if (e.dir == 0) triA = &e;
                else            triB = &e;
            }
            if (triA == nullptr || triB == nullptr) continue;

            Hinge h;
            h.v[0] = edge.first;
            h.v[1] = edge.second;
            h.v[2] = triA->apex;
            h.v[3] = triB->apex;

            const Vec2& X0 = X[h.v[0]];
            const Vec2& X1 = X[h.v[1]];
            const Vec2& X2 = X[h.v[2]];
            const Vec2& X3 = X[h.v[3]];
            const Vec2 eVec = X1 - X0;
            const double edge_len2 = eVec.squaredNorm();
            const double areaA = 0.5 * std::abs(cross_product_in_2d(eVec, X2 - X0));
            const double areaB = 0.5 * std::abs(cross_product_in_2d(eVec, X3 - X0));
            const double area_sum = areaA + areaB;
            h.c_e = (area_sum > 0.0) ? (edge_len2 / area_sum) : 0.0;
            h.bar_theta = 0.0;  // flat 2D rest

            const int hidx = static_cast<int>(hinges.size());
            hinges.push_back(h);
            for (int k = 0; k < 4; ++k)
                hinge_adj[h.v[k]].emplace_back(hidx, k);
        }
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

// vertex → {triangle_index, local_corner ∈ {0,1,2}}
using VertexTriangleMap = std::unordered_map<int, std::vector<std::pair<int,int>>>;

struct NodeTrianglePair    { int node; int tri_v[3]; };
struct SegmentSegmentPair  { int v[4]; };

// vertex_index → pins[] index, or -1 if not pinned.
using PinMap = std::vector<int>;

inline PinMap build_pin_map(const std::vector<Pin>& pins, int nv) {
    PinMap m(nv, -1);
    for (int i = 0; i < static_cast<int>(pins.size()); ++i)
        m[pins[i].vertex_index] = i;
    return m;
}

double compute_incremental_potential_no_barrier(const RefMesh& ref_mesh, const std::vector<Pin>& pins,
                                                const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat);

std::pair<Vec3, Mat33> compute_local_gradient_and_hessian_no_barrier(int vi, const RefMesh& ref_mesh,
                                                                     const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                                                     const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                                                     const PinMap* pin_map = nullptr);

double compute_global_residual(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                               const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                               const BroadPhase& broad_phase, const PinMap* pin_map = nullptr);

void serialize_state(const std::string& dir, int frame, const DeformedState& state);

bool deserialize_state(const std::string& dir, int frame, DeformedState& state);