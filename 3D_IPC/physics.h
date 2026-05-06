#pragma once
#include "corotated_energy.h"
#include "bending_energy.h"
#include "barrier_energy.h"
#include "sdf_penalty_energy.h"
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

// Fields are left uninitialised on purpose. Construct via SimParams::zeros()
// (safe-zero seed) or IPCArgs3D::to_sim_params() (production CLI defaults).
// A bare `SimParams p;` is undefined for the fundamental-type members.
struct SimParams {
    double fps;
    int    substeps;
    double mu, lambda, density, thickness, kpin, tol_abs;
    double tol_rel;   // relative tolerance (factor of initial residual); 0 disables
    double kB;        // bending (flexural) stiffness; 0 disables the bending term
    double d_hat;     // barrier activation distance; 0 disables contact
    double k_sdf;     // SDF penalty stiffness; 0 disables the SDF term
    double eps_sdf;   // SDF soft-barrier range; cloth rest at phi=eps_sdf. 0 = hard quadratic.
    std::vector<PlaneSDF>    sdf_planes;
    std::vector<CylinderSDF> sdf_cylinders;
    std::vector<SphereSDF>   sdf_spheres;
    Vec3   gravity;
    int    max_global_iters;

    bool   use_parallel;
    bool   write_substeps;       // if true, export a frame file after every substep (not just every frame)
    bool   use_ccd;              // if true, run CCD step clamping in per_vertex_safe_step
    bool   use_ccd_guess;        // if true, use ccd_initial_guess as the substep start point
    bool   use_ogc;     // if true, use trust_region_initial_guess instead of CCD
    bool   use_ogc_solver;       // if true, route the substep through global_gauss_seidel_solver_ogc
    double ogc_box_pad;          // OGC node-box / tri-edge union pad used by the per-iter BVH rebuild
    bool   fixed_iters;          // if true, run exactly max_global_iters sweeps with no tolerance / convergence check

    // Route the per-substep Gauss-Seidel sweep through the GPU implementation
    // (gpu_gauss_seidel_solver). On machines without CUDA the CPU stub is
    // used automatically. When set, use_parallel is ignored.
    bool   use_gpu;
    double node_box_max;       // upper bound on node box half-extent used by the basic solver
    double node_box_min;       // lower bound on node box half-extent (floor when prev disp is near zero)
    double k_barrier;              // barrier stiffness multiplier
    bool   use_ticcd;              // true (default) -> Tight-Inclusion CCD library; false -> self-written linear CCD

    static SimParams zeros() {
        SimParams p;
        p.fps                       = 30.0;
        p.substeps                  = 1;
        p.mu                        = 0.0;
        p.lambda                    = 0.0;
        p.density                   = 0.0;
        p.thickness                 = 0.0;
        p.kpin                      = 0.0;
        p.tol_abs                   = 0.0;
        p.tol_rel                   = 0.0;
        p.kB                        = 0.0;
        p.d_hat                     = 0.0;
        p.k_sdf                     = 0.0;
        p.eps_sdf                   = 0.0;
        p.sdf_planes.clear();
        p.sdf_cylinders.clear();
        p.sdf_spheres.clear();
        p.gravity                   = Vec3::Zero();
        p.max_global_iters          = 0;
        p.use_parallel              = false;
        p.write_substeps            = false;
        p.use_ccd                   = false;
        p.use_ccd_guess             = true;
        p.use_ogc          = false;
        p.use_ogc_solver            = false;
        p.ogc_box_pad               = 0.0;
        p.fixed_iters               = false;
        p.use_gpu                   = false;
        p.node_box_max              = 0.0;
        p.node_box_min              = 0.0;
        p.k_barrier                     = 1.0;
        p.use_ticcd                     = true;
        p.cached_dt_                = -1.0;
        p.cached_dt2_               = -1.0;
        return p;
    }

    double dt()  const {
        if (cached_dt_ < 0.0) cached_dt_ = 1.0 / (fps * static_cast<double>(substeps));
        return cached_dt_;
    }
    double dt2() const {
        if (cached_dt2_ < 0.0) { double d = dt(); cached_dt2_ = d * d; }
        return cached_dt2_;
    }
    void invalidate_dt_cache() const { cached_dt_ = -1.0; cached_dt2_ = -1.0; }

private:
    mutable double cached_dt_;
    mutable double cached_dt2_;
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
    double bar_theta;  // rest dihedral complement, computed from the initial 3D configuration
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

    inline void initialize(const std::vector<Vec2>& X, const std::vector<Vec3>& x_rest){
        num_positions = X.size();
        compute_dm_inverse(X);
        build_hinges(X, x_rest);
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

    inline void build_hinges(const std::vector<Vec2>& X, const std::vector<Vec3>& x_rest) {
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

            HingeDef def;
            for (int k = 0; k < 4; ++k) def.x[k] = x_rest[h.v[k]];
            h.bar_theta = bending_theta(def);

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

// Per-triangle quantities shared across the triangle's three corners.
struct TriPrecompute {
    Mat32      P;
    Mat66      dPdF;
    ShapeGrads gradN;
    double     A            = 0.0;
    bool       has_hessian  = false;
};

// Per-hinge quantities shared across the 4 nodes of the bending stencil.
struct HingePrecompute {
    std::array<Vec3, 4> gtheta;             // d theta / d x_node for each role
    double              scale_grad = 0.0;   // 2 * k_B * c_e * (theta - bar_theta)
    double              scale_hess = 0.0;   // 2 * k_B * c_e
    bool                degenerate = true;  // true => gradient and PSD Hessian are zero
};

void build_elastic_precompute(const RefMesh& ref_mesh, const std::vector<Vec3>& x,
                              const SimParams& params, bool want_hessian,
                              std::vector<TriPrecompute>& out);

void build_bending_precompute(const RefMesh& ref_mesh, const std::vector<Vec3>& x,
                              const SimParams& params,
                              std::vector<HingePrecompute>& out);

std::pair<Vec3, Mat33> compute_local_gradient_and_hessian_no_barrier(int vi, const RefMesh& ref_mesh,
                                                                     const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                                                     const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                                                     const PinMap* pin_map = nullptr,
                                                                     const std::vector<TriPrecompute>* tri_cache = nullptr,
                                                                     const std::vector<HingePrecompute>* hinge_cache = nullptr);

double compute_global_residual(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                               const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                               const BroadPhase& broad_phase, const PinMap* pin_map = nullptr);

void serialize_state(const std::string& dir, int frame, const DeformedState& state);

bool deserialize_state(const std::string& dir, int frame, DeformedState& state);
