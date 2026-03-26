#include <Eigen/Dense>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <utility>

namespace fs = std::__fs::filesystem;

// ============================================================
// Basic types
// ============================================================

using Vec2  = Eigen::Vector2d;
using Vec3  = Eigen::Vector3d;
using Mat22 = Eigen::Matrix2d;
using Mat32 = Eigen::Matrix<double, 3, 2>;
using Mat33 = Eigen::Matrix3d;
using Mat66 = Eigen::Matrix<double, 6, 6>;
using Mat99 = Eigen::Matrix<double, 9, 9>;

// ============================================================
// Corotated model
// ============================================================

struct TriangleRef { std::array<Vec2, 3> X_ref; };
struct TriangleDeformed { std::array<Vec3, 3> x_deformed; };

inline int flatF(int r, int c) { return 2 * r + c; }

static Mat22 Dm(const TriangleRef& tri_ref) {
    Mat22 Dm_mat;
    Dm_mat.col(0) = tri_ref.X_ref[1] - tri_ref.X_ref[0];
    Dm_mat.col(1) = tri_ref.X_ref[2] - tri_ref.X_ref[0];
    return Dm_mat;
}

static Mat32 Ds(const TriangleDeformed& tri_def) {
    Mat32 Ds_mat;
    Ds_mat.col(0) = tri_def.x_deformed[1] - tri_def.x_deformed[0];
    Ds_mat.col(1) = tri_def.x_deformed[2] - tri_def.x_deformed[0];
    return Ds_mat;
}

static double ref_area(const TriangleRef& tri_ref) {
    return 0.5 * std::abs(Dm(tri_ref).determinant());
}

static Mat32 deformation_gradient(const TriangleRef& tri_ref, const TriangleDeformed& tri_def) {
    return Ds(tri_def) * Dm(tri_ref).inverse();
}

static std::array<Vec2, 3> shape_function_gradients_ref(const TriangleRef& tri_ref) {
    Mat22 Dm_inv = Dm(tri_ref).inverse();
    std::array<Vec2, 3> grads;
    grads[1] = Dm_inv.row(0).transpose();
    grads[2] = Dm_inv.row(1).transpose();
    grads[0] = -grads[1] - grads[2];
    return grads;
}

static double PsiCorotated32(const Mat32& F, double mu, double lambda) {
    Mat22 C = F.transpose() * F;
    Eigen::SelfAdjointEigenSolver<Mat22> es(C);
    if (es.info() != Eigen::Success) throw std::runtime_error("Eigen decomposition failed in PsiCorotated32.");

    Mat22 U = es.eigenvectors();
    Eigen::Vector2d evals = es.eigenvalues();
    evals(0) = std::max(evals(0), 1e-12);
    evals(1) = std::max(evals(1), 1e-12);

    Mat22 Sigma = Mat22::Zero();
    Sigma(0, 0) = std::sqrt(evals(0));
    Sigma(1, 1) = std::sqrt(evals(1));

    Mat22 S = U * Sigma * U.transpose();
    Mat32 R = F * S.inverse();
    double J = Sigma(0, 0) * Sigma(1, 1);

    return mu * (F - R).squaredNorm() + 0.5 * lambda * (J - 1.0) * (J - 1.0);
}

static Mat32 PCorotated32(const Mat32& F, double mu, double lambda) {
    Mat22 C = F.transpose() * F;
    Eigen::SelfAdjointEigenSolver<Mat22> es(C);
    if (es.info() != Eigen::Success) throw std::runtime_error("Eigen decomposition failed in PCorotated32.");

    Mat22 U = es.eigenvectors();
    Eigen::Vector2d evals = es.eigenvalues();
    evals(0) = std::max(evals(0), 1e-12);
    evals(1) = std::max(evals(1), 1e-12);

    Mat22 Sigma = Mat22::Zero();
    Sigma(0, 0) = std::sqrt(evals(0));
    Sigma(1, 1) = std::sqrt(evals(1));

    Mat22 S = U * Sigma * U.transpose();
    Mat32 R = F * S.inverse();
    double J = Sigma(0, 0) * Sigma(1, 1);
    Mat22 C_inv = C.inverse();

    return 2.0 * mu * (F - R) + lambda * (J - 1.0) * J * F * C_inv;
}

static void dPdFCorotated32(const Mat32& F, double mu, double lambda, Mat66& dPdF) {
    Mat22 FtF = F.transpose() * F;
    Mat22 FtF_inv = FtF.inverse();

    Eigen::SelfAdjointEigenSolver<Mat22> es(FtF);
    if (es.info() != Eigen::Success) throw std::runtime_error("Eigen decomposition failed in dPdFCorotated32.");

    Mat22 U = es.eigenvectors();
    Eigen::Vector2d evals = es.eigenvalues();
    evals(0) = std::max(evals(0), 1e-12);
    evals(1) = std::max(evals(1), 1e-12);

    Mat22 Sigma = Mat22::Zero();
    Sigma(0, 0) = std::sqrt(evals(0));
    Sigma(1, 1) = std::sqrt(evals(1));

    Mat22 S = U * Sigma * U.transpose();
    Mat32 R = F * S.inverse();

    double J = Sigma(0, 0) * Sigma(1, 1);
    Mat22 S_inv = S.inverse();
    Mat33 RRt = R * R.transpose();

    Mat32 R_perp;
    R_perp(0,0) =  R(0,1); R_perp(0,1) = -R(0,0);
    R_perp(1,0) =  R(1,1); R_perp(1,1) = -R(1,0);
    R_perp(2,0) =  R(2,1); R_perp(2,1) = -R(2,0);

    double trace_S = S.trace();

    Eigen::Matrix<double, 6, 1> dcdF;
    dcdF << -R(0,1) / trace_S,  R(0,0) / trace_S,
            -R(1,1) / trace_S,  R(1,0) / trace_S,
            -R(2,1) / trace_S,  R(2,0) / trace_S;

    Mat66 dRdF = Mat66::Zero();
    std::array<int, 12> ij = {0,0, 0,1, 1,0, 1,1, 2,0, 2,1};

    for (int a = 0; a < 6; ++a) {
        for (int b = 0; b < 6; ++b) {
            int m = ij[2 * a], n = ij[2 * a + 1];
            int i = ij[2 * b], j = ij[2 * b + 1];

            double value = 0.0;
            if (m == i) value += S_inv(j, n);
            value -= RRt(m, i) * S_inv(j, n);
            value -= dcdF(b) * R_perp(m, n);
            dRdF(a, b) = value;
        }
    }

    Mat32 F_FtF_inv = F * FtF_inv;
    Mat33 F_FtF_inv_Ft = F * FtF_inv * F.transpose();

    dPdF.setZero();

    for (int a = 0; a < 6; ++a) {
        for (int b = 0; b < 6; ++b) {
            int m = ij[2 * a], n = ij[2 * a + 1];
            int i = ij[2 * b], j = ij[2 * b + 1];

            double value = 0.0;
            if (m == i) value += lambda * (J - 1.0) * J * FtF_inv(j, n);

            value -= lambda * (J - 1.0) * J *
                     (F_FtF_inv(m, j) * F_FtF_inv(i, n)
                      + F_FtF_inv_Ft(m, i) * FtF_inv(j, n));

            value += lambda * (2.0 * J - 1.0) * J *
                     (F_FtF_inv(i, j) * F_FtF_inv(m, n));

            dPdF(a, b) = value;
        }
    }

    dPdF += 2.0 * mu * (Mat66::Identity() - dRdF);
}

static double corotated_energy(const TriangleRef& tri_ref, const TriangleDeformed& tri_def, double mu, double lambda) {
    return ref_area(tri_ref) * PsiCorotated32(deformation_gradient(tri_ref, tri_def), mu, lambda);
}

static std::array<Vec3, 3> corotated_node_gradient(const TriangleRef& tri_ref, const TriangleDeformed& tri_def, double mu, double lambda) {
    double A = ref_area(tri_ref);
    Mat32 F = deformation_gradient(tri_ref, tri_def);
    Mat32 P = PCorotated32(F, mu, lambda);
    auto grads_ref = shape_function_gradients_ref(tri_ref);

    std::array<Vec3, 3> g;
    for (int a = 0; a < 3; ++a) {
        g[a].setZero();
        for (int i = 0; i < 3; ++i) {
            double value = 0.0;
            for (int beta = 0; beta < 2; ++beta) value += P(i, beta) * grads_ref[a](beta);
            g[a](i) = A * value;
        }
    }
    return g;
}

static Mat99 corotated_node_hessian(const TriangleRef& tri_ref, const TriangleDeformed& tri_def, double mu, double lambda) {
    double A = ref_area(tri_ref);
    Mat32 F = deformation_gradient(tri_ref, tri_def);
    auto grads_ref = shape_function_gradients_ref(tri_ref);

    Mat66 dPdF;
    dPdFCorotated32(F, mu, lambda, dPdF);

    Mat99 H = Mat99::Zero();

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    double value = 0.0;
                    for (int beta = 0; beta < 2; ++beta) {
                        for (int eta = 0; eta < 2; ++eta) {
                            value += dPdF(flatF(i, beta), flatF(j, eta))
                                     * grads_ref[a](beta) * grads_ref[b](eta);
                        }
                    }
                    H(3 * a + i, 3 * b + j) = A * value;
                }
            }
        }
    }

    return H;
}

// ============================================================
// Mesh / constraints / simulation state
// ============================================================

struct Tri { int v[3]; };

struct Pin {
    int vertex_index = -1;
    Vec3 target_position = Vec3::Zero();
};

struct SimParams {
    double dt{}; // timestep
    double mu{}; // Shear modulus
    double lambda{}; // 1st Lame parameter
    double density{};   // volumetric density
    double thickness{};
    double kpin{};
    Vec3 gravity = Vec3::Zero();
    int max_global_iters{};
    double tol_abs{};
    double step_weight{};
};

struct DeformedState {
    std::vector<Vec3> deformed_positions;
    std::vector<Vec3> velocities;
};

struct RefMesh {
    std::vector<Vec2> ref_positions;
    std::vector<Tri> tris;
};

struct LumpedMass { std::vector<double> vertex_masses; };

struct VertexAdjacency { std::vector<std::vector<int>> incident_triangle_indices; };

struct SolverResult {
    double initial_residual = 0.0;
    double final_residual = 0.0;
    int iterations = 0;
};

// ============================================================
// Helpers for the absorbed 2D time-stepping pattern
// ============================================================

static void build_xhat(std::vector<Vec3>& xhat, const std::vector<Vec3>& x, const std::vector<Vec3>& v, double dt) {
    int n = static_cast<int>(x.size());
    xhat.resize(n);
    for (int i = 0; i < n; ++i) xhat[i] = x[i] + dt * v[i];
}

static void update_velocity(std::vector<Vec3>& v, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xold, double dt) {
    int n = static_cast<int>(xnew.size());
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = (xnew[i] - xold[i]) / dt;
}

// ============================================================
// Triangle extraction
// ============================================================

static TriangleRef make_ref_triangle(const RefMesh& ref_mesh, const Tri& tri) {
    TriangleRef tri_ref;
    tri_ref.X_ref[0] = ref_mesh.ref_positions[tri.v[0]];
    tri_ref.X_ref[1] = ref_mesh.ref_positions[tri.v[1]];
    tri_ref.X_ref[2] = ref_mesh.ref_positions[tri.v[2]];
    return tri_ref;
}

static TriangleDeformed make_deformed_triangle(const std::vector<Vec3>& x, const Tri& tri) {
    TriangleDeformed tri_def;
    tri_def.x_deformed[0] = x[tri.v[0]];
    tri_def.x_deformed[1] = x[tri.v[1]];
    tri_def.x_deformed[2] = x[tri.v[2]];
    return tri_def;
}

static double triangle_ref_area_2d(const RefMesh& ref_mesh, const Tri& tri) {
    const Vec2& X0 = ref_mesh.ref_positions[tri.v[0]];
    const Vec2& X1 = ref_mesh.ref_positions[tri.v[1]];
    const Vec2& X2 = ref_mesh.ref_positions[tri.v[2]];
    Mat22 Dm_local;
    Dm_local.col(0) = X1 - X0;
    Dm_local.col(1) = X2 - X0;
    return 0.5 * std::abs(Dm_local.determinant());
}

static LumpedMass build_lumped_mass(const RefMesh& ref_mesh, double density, double thickness) {
    int n = static_cast<int>(ref_mesh.ref_positions.size());
    LumpedMass M;
    M.vertex_masses.assign(n, 0.0);

    for (const Tri& tri : ref_mesh.tris) {
        double A = triangle_ref_area_2d(ref_mesh, tri);
        double m = density * A * thickness;
        double m_v = m / 3.0;
        for (int a : tri.v) M.vertex_masses[a] += m_v;
    }

    return M;
}

static VertexAdjacency build_vertex_adjacency(const RefMesh& ref_mesh) {
    VertexAdjacency adj;
    adj.incident_triangle_indices.resize(ref_mesh.ref_positions.size());

    for (int t = 0; t < static_cast<int>(ref_mesh.tris.size()); ++t) {
        const Tri& tri = ref_mesh.tris[t];
        for (int a : tri.v) adj.incident_triangle_indices[a].push_back(t);
    }

    return adj;
}

// ============================================================
// Energy / gradient / Hessian
// ============================================================

static double compute_incremental_potential(const RefMesh& ref_mesh, const LumpedMass& lumped_mass,
                                            const std::vector<Pin>& pins, const SimParams& params,
                                            const std::vector<Vec3>& x, const std::vector<Vec3>& xhat) {
    int n = static_cast<int>(x.size());
    double dt2 = params.dt * params.dt;
    double E = 0.0;
    double PE = 0.0;

    // inertial term: 0.5 * ||x - xhat||_M^2
    for (int i = 0; i < n; ++i) {
        Vec3 dx = x[i] - xhat[i];
        E += 0.5 * lumped_mass.vertex_masses[i] * dx.squaredNorm();
    }

    // gravity potential
    for (int i = 0; i < n; ++i) {
        PE += -lumped_mass.vertex_masses[i] * params.gravity.dot(x[i]);
    }

    // pin potential
    for (const Pin& pin : pins) {
        Vec3 dx = x[pin.vertex_index] - pin.target_position;
        PE += 0.5 * params.kpin * dx.squaredNorm();
    }

    // elastic potential
    for (const Tri& tri : ref_mesh.tris) {
        PE += corotated_energy(make_ref_triangle(ref_mesh, tri),
                               make_deformed_triangle(x, tri),
                               params.mu, params.lambda);
    }

    return E + dt2 * PE;
}

static Vec3 compute_local_gradient(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass,
                                   const VertexAdjacency& adj, const std::vector<Pin>& pins,
                                   const SimParams& params, const std::vector<Vec3>& x,
                                   const std::vector<Vec3>& xhat) {
    double dt2 = params.dt * params.dt;
    Vec3 g = Vec3::Zero();

    // inertial gradient
    g += lumped_mass.vertex_masses[vi] * (x[vi] - xhat[vi]);

    // gravity gradient
    g += dt2 * (-lumped_mass.vertex_masses[vi] * params.gravity);

    // pin gradient
    for (const Pin& pin : pins) {
        if (pin.vertex_index == vi) {
            g += dt2 * params.kpin * (x[vi] - pin.target_position);
        }
    }

    // elastic gradient
    for (int ti : adj.incident_triangle_indices[vi]) {
        const Tri& tri = ref_mesh.tris[ti];
        auto node_g = corotated_node_gradient(make_ref_triangle(ref_mesh, tri),
                                              make_deformed_triangle(x, tri),
                                              params.mu, params.lambda);

        for (int a = 0; a < 3; ++a) {
            if (tri.v[a] == vi) {
                g += dt2 * node_g[a];
            }
        }
    }

    return g;
}

static Mat33 compute_local_hessian(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass,
                                   const VertexAdjacency& adj, const std::vector<Pin>& pins,
                                   const SimParams& params, const std::vector<Vec3>& x) {
    double dt2 = params.dt * params.dt;
    Mat33 H = Mat33::Zero();

    // inertial Hessian
    H += lumped_mass.vertex_masses[vi] * Mat33::Identity();

    // pin Hessian
    for (const Pin& pin : pins) {
        if (pin.vertex_index == vi) {
            H += dt2 * params.kpin * Mat33::Identity();
        }
    }

    // elastic Hessian
    for (int ti : adj.incident_triangle_indices[vi]) {
        const Tri& tri = ref_mesh.tris[ti];
        Mat99 tri_H = corotated_node_hessian(make_ref_triangle(ref_mesh, tri),
                                             make_deformed_triangle(x, tri),
                                             params.mu, params.lambda);

        for (int a = 0; a < 3; ++a) {
            if (tri.v[a] == vi) {
                H += dt2 * tri_H.block<3, 3>(3 * a, 3 * a);
            }
        }
    }

    return H;
}

static double compute_global_residual(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj,
                                      const std::vector<Pin>& pins, const SimParams& params,
                                      const std::vector<Vec3>& x, const std::vector<Vec3>& xhat) {
    double r_inf = 0.0;
    int n = static_cast<int>(x.size());

    for (int i = 0; i < n; ++i) {
        Vec3 g = compute_local_gradient(i, ref_mesh, lumped_mass, adj, pins, params, x, xhat);
        r_inf = std::max(r_inf, std::abs(g.x()));
        r_inf = std::max(r_inf, std::abs(g.y()));
        r_inf = std::max(r_inf, std::abs(g.z()));
    }

    return r_inf;
}

// ============================================================
// One GS vertex update
// ============================================================

static Mat33 matrix3d_inverse(const Mat33& H) {
    double det = H(0,0)*(H(1,1)*H(2,2) - H(1,2)*H(2,1))
                 - H(0,1)*(H(1,0)*H(2,2) - H(1,2)*H(2,0))
                 + H(0,2)*(H(1,0)*H(2,1) - H(1,1)*H(2,0));

    if (std::abs(det) < 1e-12) throw std::runtime_error("Singular matrix in matrix3d_inverse().");

    double inv_det = 1.0 / det;
    Mat33 inv;

    inv(0,0) =  (H(1,1)*H(2,2) - H(1,2)*H(2,1));
    inv(0,1) = -(H(0,1)*H(2,2) - H(0,2)*H(2,1));
    inv(0,2) =  (H(0,1)*H(1,2) - H(0,2)*H(1,1));
    inv(1,0) = -(H(1,0)*H(2,2) - H(1,2)*H(2,0));
    inv(1,1) =  (H(0,0)*H(2,2) - H(0,2)*H(2,0));
    inv(1,2) = -(H(0,0)*H(1,2) - H(0,2)*H(1,0));
    inv(2,0) =  (H(1,0)*H(2,1) - H(1,1)*H(2,0));
    inv(2,1) = -(H(0,0)*H(2,1) - H(0,1)*H(2,0));
    inv(2,2) =  (H(0,0)*H(1,1) - H(0,1)*H(1,0));

    return inv * inv_det;
}

static void update_one_vertex(int vi, const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj,
                              const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& xhat, std::vector<Vec3>& x) {
    Vec3 g = compute_local_gradient(vi, ref_mesh, lumped_mass, adj, pins, params, x, xhat);
    Mat33 H = compute_local_hessian(vi, ref_mesh, lumped_mass, adj, pins, params, x);
    Vec3 dx = matrix3d_inverse(H) * g;
    x[vi] -= params.step_weight * dx;
}

// ============================================================
// Nonlinear Gauss-Seidel solver
// ============================================================

static SolverResult global_gauss_seidel_solver(const RefMesh& ref_mesh, const LumpedMass& lumped_mass, const VertexAdjacency& adj,
                                               const std::vector<Pin>& pins, const SimParams& params,
                                               std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                               std::vector<double>* residual_history = nullptr) {
    if (residual_history) residual_history->clear();

    auto eval_residual = [&]() {
        return compute_global_residual(ref_mesh, lumped_mass, adj, pins, params, xnew, xhat);
    };

    SolverResult result;
    result.initial_residual = eval_residual();
    result.final_residual = result.initial_residual;
    result.iterations = 0;

    if (residual_history) residual_history->push_back(result.initial_residual);

    if (result.initial_residual < params.tol_abs) return result;

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        for (int vi = 0; vi < static_cast<int>(xnew.size()); ++vi)
            update_one_vertex(vi, ref_mesh, lumped_mass, adj, pins, params, xhat, xnew);

        result.final_residual = eval_residual();
        result.iterations = iter;

        if (residual_history) residual_history->push_back(result.final_residual);

        if (result.final_residual < params.tol_abs) return result;
    }

    return result;
}

// ============================================================
// Visualization utilities
// ============================================================

static void export_obj(const std::string& filename, const std::vector<Vec3>& x, const std::vector<Tri>& tris) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: cannot write " << filename << "\n";
        return;
    }

    for (const auto& p : x) out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
    for (const auto& t : tris) out << "f " << (t.v[0] + 1) << " " << (t.v[1] + 1) << " " << (t.v[2] + 1) << "\n";
}

static void export_frame(const std::string& outdir, int frame, const std::vector<Vec3>& x, const std::vector<Tri>& tris) {
    std::ostringstream ss;
    ss << outdir << "/frame_" << std::setw(4) << std::setfill('0') << frame << ".obj";
    export_obj(ss.str(), x, tris);
}

// ============================================================
// Example triangle helpers
// ============================================================

static void clear_model(RefMesh& ref_mesh, DeformedState& state, std::vector<Pin>& pins) {
    ref_mesh.ref_positions.clear();
    ref_mesh.tris.clear();
    state.deformed_positions.clear();
    state.velocities.clear();
    pins.clear();
}

static int build_single_triangle(RefMesh& ref_mesh, DeformedState& state,
                                 const Vec2& X0, const Vec2& X1, const Vec2& X2,
                                 const Vec3& x0, const Vec3& x1, const Vec3& x2) {
    int base = static_cast<int>(state.deformed_positions.size());

    ref_mesh.ref_positions.push_back(X0);
    ref_mesh.ref_positions.push_back(X1);
    ref_mesh.ref_positions.push_back(X2);

    state.deformed_positions.push_back(x0);
    state.deformed_positions.push_back(x1);
    state.deformed_positions.push_back(x2);

    ref_mesh.tris.push_back(Tri{{base + 0, base + 1, base + 2}});
    return base;
}


static void append_pin(std::vector<Pin>& pins, int vertex_index, const std::vector<Vec3>& x) {
    pins.push_back(Pin{vertex_index, x[vertex_index]});
}

// ============================================================
// Main
// ============================================================

int main() {
    SimParams params;
    params.dt = 1.0 / 30.0;
    params.mu = 1.0;
    params.lambda = 1.0;
    params.density = 1.0;
    params.thickness = 0.1;
    params.kpin = 1e7;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 100;
    params.tol_abs = 1e-6;
    params.step_weight = 1.0;

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Pin> pins;

    clear_model(ref_mesh, state, pins);

    int tri0 = build_single_triangle(
            ref_mesh, state,
            Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(0.2, 1.0),
            Vec3(-1.5, 0.0, 0.2), Vec3(-0.5, 0.0, 0.4), Vec3(-1.3, 1.0, 0.3)
    );

    int tri1 = build_single_triangle(
            ref_mesh, state,
            Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(0.2, 1.0),
            Vec3( 0.5, 0.0, 1.0), Vec3( 1.5, 0.0, -0.5), Vec3( 0.7, 1.0, 0.0)
    );

    state.velocities.assign(state.deformed_positions.size(), Vec3::Zero());

    append_pin(pins, tri0 + 0, state.deformed_positions);
    append_pin(pins, tri1 + 0, state.deformed_positions);

    LumpedMass lumped_mass = build_lumped_mass(ref_mesh, params.density, params.thickness);
    VertexAdjacency adj = build_vertex_adjacency(ref_mesh);

    std::cout << "Vertices:  " << state.deformed_positions.size() << "\n";
    std::cout << "Triangles: " << ref_mesh.tris.size() << "\n";

    std::string outdir = "frames_sim3d";
    if (fs::exists(outdir)) fs::remove_all(outdir);
    fs::create_directories(outdir);

    export_frame(outdir, 0, state.deformed_positions, ref_mesh.tris);

    const int num_frames = 100;

    for (int frame_index = 1; frame_index <= num_frames; ++frame_index) {
        std::vector<Vec3> xhat;
        build_xhat(xhat, state.deformed_positions, state.velocities, params.dt);

        std::vector<Vec3> xnew = xhat;

        SolverResult result = global_gauss_seidel_solver(
                ref_mesh, lumped_mass, adj, pins, params, xnew, xhat
        );

        std::cout << "Frame " << std::setw(4) << frame_index
                  << " | initial_residual = " << std::scientific << result.initial_residual
                  << " | final_residual = "   << std::scientific << result.final_residual
                  << " | global_iters = "     << std::setw(3) << result.iterations
                  << "\n";

        update_velocity(state.velocities, xnew, state.deformed_positions, params.dt);
        state.deformed_positions = xnew;

        export_frame(outdir, frame_index, state.deformed_positions, ref_mesh.tris);
    }

    return 0;
}


