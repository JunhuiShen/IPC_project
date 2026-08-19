#pragma once
#include "corotated_energy.h"
#include "volumetric_corotated_energy.h"
#include "bending_energy.h"
#include "barrier_energy.h"
#include "sdf_penalty_energy.h"
#include <algorithm>
#include <array>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>
#include <stdexcept>

class BroadPhase;

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
    // Thickness-scaled shell material and mass parameters.
    double mu, lambda, density, thickness;
    // Unscaled volumetric-solid material and mass parameters.
    double solid_mu, solid_lambda, solid_density;
    // Density used when constructing rigid bodies from volumetric shapes.
    double rigid_density;
    double kpin, tol_abs;
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
    bool   use_verlet_guess;     // if true, start GS from xhat + dt²*gravity (Verlet predictor)
    bool   use_translation_guess;        // if true, start GS from x^n + C
    bool   use_ogc;     // if true, use trust_region_initial_guess instead of CCD
    bool   use_ogc_solver;       // if true, route the substep through global_gauss_seidel_solver_ogc
    double ogc_box_pad;          // OGC node-box / tri-edge union pad used by the per-iter BVH rebuild
    bool   fixed_iters;          // if true, run exactly max_global_iters sweeps with no tolerance / convergence check

    double node_box_max;       // upper bound on node box half-extent used by the basic solver
    double node_box_min;       // lower bound on node box half-extent (floor when prev disp is near zero)
    double theta_box_min;      // lower bound on the rigid orientation-box angular radius
    double theta_box_max;      // upper bound on the rigid orientation-box angular radius
    int    node_box_update_count;
    double k_barrier;              // barrier stiffness multiplier
    double damping;                // Newton-step damping used by deformable and rigid solvers
    bool   use_ticcd;              // true (default) -> Tight-Inclusion CCD library; false -> self-written linear CCD

    static SimParams zeros() {
        SimParams p;
        p.fps                       = 30.0;
        p.substeps                  = 1;
        p.mu                        = 0.0;
        p.lambda                    = 0.0;
        p.density                   = 0.0;
        p.thickness                 = 0.0;
        p.solid_mu                  = 0.0;
        p.solid_lambda              = 0.0;
        p.solid_density             = 0.0;
        p.rigid_density             = 0.0;
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
        p.use_translation_guess             = false;
        p.use_verlet_guess          = false;
        p.use_ogc          = false;
        p.use_ogc_solver            = false;
        p.ogc_box_pad               = 0.0;
        p.fixed_iters               = false;
        p.node_box_max              = 0.0;
        p.node_box_min              = 0.0;
        p.theta_box_min             = 0.0;
        p.theta_box_max             = 0.0;
        p.node_box_update_count     = 250;
        p.k_barrier                     = 1.0;
        p.damping                       = 1.0;
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

private:
    mutable double cached_dt_;
    mutable double cached_dt2_;
public:
};

struct DeformedState {
    std::vector<Vec3> deformed_positions;
    std::vector<Vec3> velocities;

    // Rigid Bodies
    std::vector<Vec3> x_coms; // center of mass position for each rb
    std::vector<Vec3> v_coms; // center of mass velocity for each rb
    std::vector<Vec4> orientations; // orientation stored as quaternion
    std::vector<Vec3> omega; // angular velocity
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
    // Shared particle and collision/render surface data
    // Collision/render surface. Every three entries form one triangle. For a
    // tetrahedral solid, this contains only the extracted boundary faces.
    std::vector<int> tris;
    std::vector<double> mass;
    size_t num_positions = 0;

    // Cloth
    std::vector<Mat22> Dm_inverse;
    std::vector<double> area;
    std::vector<Hinge> hinges;
    VertexHingeMap hinge_adj;

    // Deformable Solid
    std::vector<int> tets; // flat: every 4 ints = one tetrahedron
    std::vector<TetRestData> tet_rest_data;
    // Global vertex -> {(tet index, local role in 0..3)}
    std::vector<std::vector<std::pair<int, int>>> tet_adj;
    // Unique vertices referenced by tets, in first-connectivity-appearance
    // order. These are the volumetric solid degrees of freedom.
    std::vector<int> tet_nodes;
    // Unique vertices referenced by the extracted boundary tris, in
    // first-boundary-appearance order. These classify the collision/render
    // surface; interior tet nodes are deliberately excluded.
    std::vector<int> surface_nodes;

    // Rigid Bodies
    std::vector<std::vector<Vec3>> ref_positions; // body-space particle positions for each rb
    std::vector<double> total_mass; // total_mass for each rb
    std::vector<Mat33> I_hat; // IPC inertia tensor for each rb
    std::vector<std::vector<int>> rb_nodes; // global particle indices for each rb
    // One generalized-coordinate solver-update label per rigid body.
    std::vector<RigidBodyUpdateMode> rb_update_modes;
    std::vector<int> node_to_rb; // global particle index -> rb index (-1 if deformable)


    // Mixed Solver
    std::vector<int> deformable_nodes; // global indices of independently deformable nodes

    // Cloth
    inline void initialize(const std::vector<Vec2>& X, const std::vector<Vec3>& x_rest){
        num_positions = X.size();
        compute_dm_inverse(X);
        build_hinges(X, x_rest);
    }

    inline void initialize(const std::vector<Vec3>& x_rest){
        num_positions = x_rest.size();
        compute_dm_inverse(x_rest);
        build_hinges(x_rest);
    }

    inline void compute_dm_inverse(const std::vector<Vec3>& X){
        int nt = static_cast<int>(tris.size()) / 3;
        Dm_inverse.resize(nt);
        area.resize(nt);
        for(int t = 0; t < nt; t++){
            const Vec3& X0 = X[tris[t*3+0]];
            const Vec3& X1 = X[tris[t*3+1]];
            const Vec3& X2 = X[tris[t*3+2]];
            Mat32 Dm;
            Dm.col(0) = X1 - X0;
            Dm.col(1) = X2 - X0;
            // QR: Dm = Q * R, Q is 3x2 orthonormal, R is 2x2 upper triangular
            // area = 0.5 * |det(R)|  (since det(Dm^T Dm) = det(R)^2)
            // Dm_inverse = R^{-1}  (same role as the 2D case)
            Mat22 R = Dm.householderQr().matrixQR().topLeftCorner<2,2>()
                        .template triangularView<Eigen::Upper>();
            area[t] = 0.5 * std::abs(R.determinant());
            Dm_inverse[t] = R.inverse();
        }
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

    inline void build_hinges(const std::vector<Vec3>& x_rest) {
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

            const Vec3& X0 = x_rest[h.v[0]];
            const Vec3& X1 = x_rest[h.v[1]];
            const Vec3& X2 = x_rest[h.v[2]];
            const Vec3& X3 = x_rest[h.v[3]];
            const Vec3 eVec = X1 - X0;
            const double edge_len2 = eVec.squaredNorm();
            const double areaA = 0.5 * eVec.cross(X2 - X0).norm();
            const double areaB = 0.5 * eVec.cross(X3 - X0).norm();
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

    // Mixed Solver
    // Rebuilds shell masses in a mixed mesh. Tetrahedral masses assigned by
    // create_solid and rigid proxy masses assigned by create_rigid_body are
    // preserved. Shell triangles occupy the leading triangle prefix described
    // by Dm_inverse/area; later triangles are collision/render surfaces only.
    inline void build_deformable_lumped_mass(
        double density, double thickness) {
        if (node_to_rb.size() != num_positions) {
            throw std::invalid_argument(
                "build_deformable_lumped_mass: node ownership size mismatch");
        }
        if (Dm_inverse.size() != area.size()
            || area.size() > tris.size() / 3) {
            throw std::invalid_argument(
                "build_deformable_lumped_mass: cloth rest data is inconsistent");
        }

        mass.resize(num_positions, 0.0);
        std::vector<unsigned char> is_tet_node(num_positions, 0);
        for (const int node : tet_nodes) {
            if (node < 0 || static_cast<std::size_t>(node) >= num_positions) {
                throw std::out_of_range(
                    "build_deformable_lumped_mass: tet node is out of range");
            }
            is_tet_node[static_cast<std::size_t>(node)] = 1;
        }

        for (std::size_t node = 0; node < num_positions; ++node) {
            if (node_to_rb[node] < 0 && is_tet_node[node] == 0)
                mass[node] = 0.0;
        }

        const int shell_triangle_count = static_cast<int>(area.size());
        for (int triangle = 0; triangle < shell_triangle_count; ++triangle) {
            const int v0 = tris[3 * triangle + 0];
            const int v1 = tris[3 * triangle + 1];
            const int v2 = tris[3 * triangle + 2];
            const int vertices[3] = {v0, v1, v2};
            for (const int vertex : vertices) {
                if (vertex < 0
                    || static_cast<std::size_t>(vertex) >= num_positions) {
                    throw std::out_of_range(
                        "build_deformable_lumped_mass: cloth triangle vertex is out of range");
                }
                if (node_to_rb[static_cast<std::size_t>(vertex)] >= 0
                    || is_tet_node[static_cast<std::size_t>(vertex)] != 0) {
                    throw std::invalid_argument(
                        "build_deformable_lumped_mass: cloth rest prefix contains a non-cloth node");
                }
            }

            const double nodal_mass =
                density * area[triangle] * thickness / 3.0;
            mass[v0] += nodal_mass;
            mass[v1] += nodal_mass;
            mass[v2] += nodal_mass;
        }
    }

    inline void build_deformable_nodes() {
        deformable_nodes.clear();
        deformable_nodes.reserve(num_positions);
        for (std::size_t node = 0; node < num_positions; ++node) {
            if (node >= node_to_rb.size() || node_to_rb[node] < 0)
                deformable_nodes.push_back(static_cast<int>(node));
        }
    }

};

inline int tri_vertex(const RefMesh& ref_mesh, int tri_idx, int local) {
    return ref_mesh.tris[tri_idx * 3 + local];
}

inline int num_tris(const RefMesh& ref_mesh) {
    return static_cast<int>(ref_mesh.tris.size()) / 3;
}

inline int tet_vertex(const RefMesh& ref_mesh, int tet_idx, int local) {
    return ref_mesh.tets[tet_idx * 4 + local];
}

inline int num_tets(const RefMesh& ref_mesh) {
    return static_cast<int>(ref_mesh.tets.size()) / 4;
}

// vertex -> {triangle_index, local_corner in {0,1,2}}
using VertexTriangleMap = std::unordered_map<int, std::vector<std::pair<int,int>>>;
using IncidentTriangles = std::vector<std::pair<int,int>>;

struct NodeTrianglePair    { int node; int tri_v[3]; };
struct SegmentSegmentPair  { int v[4]; };

// Raw primitive gradients evaluated at one frozen position vector. Storage can
// be reused, but values must be rebuilt after x, material parameters, or the
// broad-phase cache changes. Residual assembly still traverses each node's
// original incidence rows, so this cache does not reorder floating-point sums.
// NT/SS gradient entries are defined only when the matching gradient_cached
// byte is nonzero; the validity arrays are cleared on every rebuild.
struct FrozenResidualWorkspace {
    const RefMesh* mesh = nullptr;
    const SimParams* params = nullptr;
    const std::vector<Vec3>* positions = nullptr;
    const BroadPhase* broad_phase = nullptr;
    std::vector<std::array<Vec3, 3>> cloth_triangle_gradients;
    std::vector<std::array<Vec3, 4>> hinge_gradients;
    std::vector<std::array<Vec3, 4>> tet_gradients;
    std::vector<std::array<Vec3, 4>> nt_gradients;
    std::vector<std::array<Vec3, 4>> ss_gradients;
    std::vector<unsigned char> nt_aabb_active;
    std::vector<unsigned char> ss_aabb_active;
    std::vector<unsigned char> nt_barrier_active;
    std::vector<unsigned char> ss_barrier_active;
    std::vector<unsigned char> nt_gradient_cached;
    std::vector<unsigned char> ss_gradient_cached;

    bool matches(const RefMesh& ref_mesh, const SimParams& sim_params, const std::vector<Vec3>& x, const BroadPhase& broad_phase_input) const { return mesh == &ref_mesh && params == &sim_params && positions == &x && broad_phase == &broad_phase_input; }
};

void build_frozen_residual_workspace(const RefMesh& ref_mesh, const SimParams& params, const std::vector<Vec3>& x, const BroadPhase& broad_phase, FrozenResidualWorkspace& workspace, const std::vector<ShapeGrads>* rest_shape_grads = nullptr);

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
                                                                     const PinMap* pin_map = nullptr,
                                                                     const IncidentTriangles* incident_triangles = nullptr,
                                                                     const std::vector<ShapeGrads>* rest_shape_grads = nullptr);

double compute_global_deformable_residual(const RefMesh& ref_mesh, const VertexTriangleMap& adj,
                                          const std::vector<Pin>& pins, const SimParams& params,
                                          const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                          const BroadPhase& broad_phase,
                                          const std::vector<int>& deformable_nodes,
                                          const PinMap* pin_map = nullptr,
                                          const std::vector<IncidentTriangles>* incident_triangles = nullptr,
                                          const std::vector<ShapeGrads>* rest_shape_grads = nullptr,
                                          const FrozenResidualWorkspace* frozen_workspace = nullptr);

double compute_global_residual(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                               const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                               const BroadPhase& broad_phase, const PinMap* pin_map = nullptr,
                               const std::vector<IncidentTriangles>* incident_triangles = nullptr,
                               const std::vector<ShapeGrads>* rest_shape_grads = nullptr);
