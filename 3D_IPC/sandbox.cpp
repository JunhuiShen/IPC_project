#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::__fs::filesystem;

// ============================================================
// Types
// ============================================================
using Vec2   = Eigen::Vector2d;
using Vec3   = Eigen::Vector3d;
using Vec6   = Eigen::Matrix<double, 6, 1>;
using Mat22  = Eigen::Matrix2d;
using Mat32  = Eigen::Matrix<double, 3, 2>;
using Mat33  = Eigen::Matrix3d;
using Mat39  = Eigen::Matrix<double, 3, 9>;
using Mat66  = Eigen::Matrix<double, 6, 6>;
using Mat312 = Eigen::Matrix<double, 3, 12>;

struct TriangleDef { Vec3 x[3]; };

struct Pin {
    int vertex_index{-1};
    Vec3 target_position{Vec3::Zero()};
};

struct SimParams {
    double fps{30.0};
    int    substeps{1};
    int    num_frames{60};

    double mu{10.0};
    double lambda{10.0};
    double density{1.0};
    double thickness{0.1};
    double kpin{1.0e7};
    Vec3   gravity{0.0, -9.81, 0.0};

    int    max_global_iters{80};
    double tol_abs{1.0e-6};
    double step_weight{0.25};
    double d_hat{0.0};

    double dt()  const { return 1.0 / (fps * static_cast<double>(substeps)); }
    double dt2() const { double h = dt(); return h * h; }
};

struct RefMesh {
    std::vector<Vec2> ref_positions;
    std::vector<int>  tris;      // flat triangle index buffer
    std::vector<Mat22> Dm_inv;
    std::vector<double> area;
    std::vector<double> mass;
    int num_positions{0};

    void initialize() {
        num_positions = static_cast<int>(ref_positions.size());
        const int nt = static_cast<int>(tris.size()) / 3;
        Dm_inv.resize(nt);
        area.resize(nt);
        for (int t = 0; t < nt; ++t) {
            const Vec2& X0 = ref_positions[tris[3 * t + 0]];
            const Vec2& X1 = ref_positions[tris[3 * t + 1]];
            const Vec2& X2 = ref_positions[tris[3 * t + 2]];
            Mat22 Dm;
            Dm.col(0) = X1 - X0;
            Dm.col(1) = X2 - X0;
            area[t] = 0.5 * std::abs(Dm.determinant());
            Dm_inv[t] = Dm.inverse();
        }
    }

    void build_lumped_mass(double density, double thickness) {
        mass.assign(num_positions, 0.0);
        const int nt = static_cast<int>(tris.size()) / 3;
        for (int t = 0; t < nt; ++t) {
            const double m = density * area[t] * thickness;
            const double mv = m / 3.0;
            for (int a = 0; a < 3; ++a) {
                mass[tris[3 * t + a]] += mv;
            }
        }
    }
};

struct DeformedState {
    std::vector<Vec3> x;
    std::vector<Vec3> v;
};

struct PatchInfo {
    int vertex_begin{0};
    int vertex_end{0};
    int tri_begin{0};
    int tri_end{0};
    int nx{0};
    int ny{0};
};

struct NodeTrianglePair {
    int node{-1};
    int tri_v[3]{-1, -1, -1};
};

struct SegmentSegmentPair {
    int v[4]{-1, -1, -1, -1};
};

using VertexTriangleMap = std::unordered_map<int, std::vector<std::pair<int, int>>>;

struct SolverResult {
    double initial_residual{0.0};
    double final_residual{0.0};
    int iterations{0};
};

// ============================================================
// Small helpers
// ============================================================
static inline int tri_vertex(const RefMesh& mesh, int tri, int local) {
    return mesh.tris[3 * tri + local];
}

static inline int num_tris(const RefMesh& mesh) {
    return static_cast<int>(mesh.tris.size()) / 3;
}

static inline double clamp_scalar(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
}

static Mat33 matrix3d_inverse(const Mat33& H) {
    const double det = H.determinant();
    if (std::abs(det) < 1e-12) {
        // Mild regularization instead of crashing.
        return (H + 1e-8 * Mat33::Identity()).inverse();
    }
    return H.inverse();
}

static TriangleDef make_def_triangle(const std::vector<Vec3>& x, const RefMesh& mesh, int tri_idx) {
    TriangleDef def;
    def.x[0] = x[tri_vertex(mesh, tri_idx, 0)];
    def.x[1] = x[tri_vertex(mesh, tri_idx, 1)];
    def.x[2] = x[tri_vertex(mesh, tri_idx, 2)];
    return def;
}

static void build_xhat(std::vector<Vec3>& xhat, const std::vector<Vec3>& x, const std::vector<Vec3>& v, double dt) {
    xhat.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) xhat[i] = x[i] + dt * v[i];
}

static void update_velocity(std::vector<Vec3>& v, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xold, double dt) {
    v.resize(xnew.size());
    for (size_t i = 0; i < xnew.size(); ++i) v[i] = (xnew[i] - xold[i]) / dt;
}

static Vec3 segment_closest_point(const Vec3& x, const Vec3& a, const Vec3& b, double& t) {
    const Vec3 ab = b - a;
    const double denom = ab.dot(ab);
    if (denom <= 0.0) {
        t = 0.0;
        return a;
    }
    t = clamp_scalar((x - a).dot(ab) / denom, 0.0, 1.0);
    return a + t * ab;
}

static std::array<double, 3> triangle_plane_barycentric_coordinates(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps = 1.0e-12) {
    const Vec3 e1 = x2 - x1;
    const Vec3 e2 = x3 - x1;
    const Vec3 r  = x  - x1;

    const double a11 = e1.dot(e1);
    const double a12 = e1.dot(e2);
    const double a22 = e2.dot(e2);
    const double b1  = r.dot(e1);
    const double b2  = r.dot(e2);
    const double det = a11 * a22 - a12 * a12;

    if (std::abs(det) <= eps) return {{0.0, 0.0, 0.0}};

    const double alpha = ( b1 * a22 - b2 * a12) / det;
    const double beta  = (-b1 * a12 + b2 * a11) / det;
    return {{1.0 - alpha - beta, alpha, beta}};
}

static double safe_distance(double d) {
    return std::max(d, 1.0e-8);
}

// ============================================================
// Distance queries
// ============================================================
enum class NodeTriangleRegion {
    FaceInterior, Edge12, Edge23, Edge31, Vertex1, Vertex2, Vertex3, DegenerateTriangle
};

struct NodeTriangleDistanceResult {
    Vec3 closest_point{Vec3::Zero()};
    Vec3 tilde_x{Vec3::Zero()};
    Vec3 normal{Vec3::Zero()};
    std::array<double, 3> barycentric_tilde_x{{0.0, 0.0, 0.0}};
    double phi{0.0};
    double distance{0.0};
    NodeTriangleRegion region{NodeTriangleRegion::DegenerateTriangle};
};

static NodeTriangleDistanceResult node_triangle_distance(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps = 1.0e-12) {
    NodeTriangleDistanceResult out;

    const Vec3 e12 = x2 - x1;
    const Vec3 e13 = x3 - x1;
    const Vec3 n_raw = e12.cross(e13);
    const double n_norm = n_raw.norm();

    if (n_norm <= eps) {
        double t12 = 0.0, t23 = 0.0, t31 = 0.0;
        const Vec3 q12 = segment_closest_point(x, x1, x2, t12);
        const Vec3 q23 = segment_closest_point(x, x2, x3, t23);
        const Vec3 q31 = segment_closest_point(x, x3, x1, t31);
        const double d12 = (x - q12).norm();
        const double d23 = (x - q23).norm();
        const double d31 = (x - q31).norm();
        if (d12 <= d23 && d12 <= d31) { out.closest_point = q12; out.distance = d12; }
        else if (d23 <= d12 && d23 <= d31) { out.closest_point = q23; out.distance = d23; }
        else { out.closest_point = q31; out.distance = d31; }
        out.tilde_x = out.closest_point;
        return out;
    }

    const Vec3 n = n_raw / n_norm;
    const double phi = n.dot(x - x1);
    const Vec3 tilde_x = x - phi * n;

    out.normal = n;
    out.phi = phi;
    out.tilde_x = tilde_x;
    out.barycentric_tilde_x = triangle_plane_barycentric_coordinates(tilde_x, x1, x2, x3, eps);

    const double l1 = out.barycentric_tilde_x[0];
    const double l2 = out.barycentric_tilde_x[1];
    const double l3 = out.barycentric_tilde_x[2];

    if (l1 >= 0.0 && l2 >= 0.0 && l3 >= 0.0) {
        out.closest_point = tilde_x;
        out.distance = std::abs(phi);
        out.region = NodeTriangleRegion::FaceInterior;
        return out;
    }
    if (l2 <= 0.0 && l3 <= 0.0) {
        out.closest_point = x1;
        out.distance = (x - x1).norm();
        out.region = NodeTriangleRegion::Vertex1;
        return out;
    }
    if (l3 <= 0.0 && l1 <= 0.0) {
        out.closest_point = x2;
        out.distance = (x - x2).norm();
        out.region = NodeTriangleRegion::Vertex2;
        return out;
    }
    if (l1 <= 0.0 && l2 <= 0.0) {
        out.closest_point = x3;
        out.distance = (x - x3).norm();
        out.region = NodeTriangleRegion::Vertex3;
        return out;
    }
    if (l3 < 0.0) {
        double t = 0.0;
        out.closest_point = segment_closest_point(tilde_x, x1, x2, t);
        out.distance = (x - out.closest_point).norm();
        out.region = NodeTriangleRegion::Edge12;
        return out;
    }
    if (l1 < 0.0) {
        double t = 0.0;
        out.closest_point = segment_closest_point(tilde_x, x2, x3, t);
        out.distance = (x - out.closest_point).norm();
        out.region = NodeTriangleRegion::Edge23;
        return out;
    }
    if (l2 < 0.0) {
        double t = 0.0;
        out.closest_point = segment_closest_point(tilde_x, x3, x1, t);
        out.distance = (x - out.closest_point).norm();
        out.region = NodeTriangleRegion::Edge31;
        return out;
    }

    return out;
}

enum class SegmentSegmentRegion {
    Interior, Edge_s0, Edge_s1, Edge_t0, Edge_t1,
    Corner_s0t0, Corner_s0t1, Corner_s1t0, Corner_s1t1, ParallelSegments
};

struct SegmentSegmentDistanceResult {
    Vec3 closest_point_1{Vec3::Zero()};
    Vec3 closest_point_2{Vec3::Zero()};
    double s{0.0};
    double t{0.0};
    double distance{0.0};
    SegmentSegmentRegion region{SegmentSegmentRegion::ParallelSegments};
};

static double optimal_t_for_fixed_s(const Vec3& x1, const Vec3& a,
                                    const Vec3& x3, const Vec3& b,
                                    double s, double C, double& t_out) {
    const Vec3 p = x1 + s * a;
    t_out = (C <= 0.0) ? 0.0 : clamp_scalar((p - x3).dot(b) / C, 0.0, 1.0);
    const Vec3 q = x3 + t_out * b;
    return (p - q).norm();
}

static double optimal_s_for_fixed_t(const Vec3& x1, const Vec3& a,
                                    const Vec3& x3, const Vec3& b,
                                    double t, double A, double& s_out) {
    const Vec3 q = x3 + t * b;
    s_out = (A <= 0.0) ? 0.0 : clamp_scalar((q - x1).dot(a) / A, 0.0, 1.0);
    const Vec3 p = x1 + s_out * a;
    return (p - q).norm();
}

static SegmentSegmentRegion classify_region(double s, double t, double tol = 1.0e-14) {
    const bool s0 = (s <= tol);
    const bool s1 = (s >= 1.0 - tol);
    const bool t0 = (t <= tol);
    const bool t1 = (t >= 1.0 - tol);

    if (s0 && t0) return SegmentSegmentRegion::Corner_s0t0;
    if (s0 && t1) return SegmentSegmentRegion::Corner_s0t1;
    if (s1 && t0) return SegmentSegmentRegion::Corner_s1t0;
    if (s1 && t1) return SegmentSegmentRegion::Corner_s1t1;
    if (s0) return SegmentSegmentRegion::Edge_s0;
    if (s1) return SegmentSegmentRegion::Edge_s1;
    if (t0) return SegmentSegmentRegion::Edge_t0;
    if (t1) return SegmentSegmentRegion::Edge_t1;
    return SegmentSegmentRegion::Interior;
}

static SegmentSegmentDistanceResult segment_segment_distance(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps = 1.0e-12) {
    SegmentSegmentDistanceResult out;

    const Vec3 a = x2 - x1;
    const Vec3 b = x4 - x3;
    const Vec3 c = x1 - x3;

    const double A = a.dot(a);
    const double B = a.dot(b);
    const double C = b.dot(b);
    const double D = a.dot(c);
    const double E = b.dot(c);
    const double Delta = A * C - B * B;

    if (Delta > eps * eps) {
        const double s_unc = (B * E - C * D) / Delta;
        const double t_unc = (A * E - B * D) / Delta;
        if (s_unc >= 0.0 && s_unc <= 1.0 && t_unc >= 0.0 && t_unc <= 1.0) {
            out.s = s_unc;
            out.t = t_unc;
            out.closest_point_1 = x1 + out.s * a;
            out.closest_point_2 = x3 + out.t * b;
            out.distance = (out.closest_point_1 - out.closest_point_2).norm();
            out.region = SegmentSegmentRegion::Interior;
            return out;
        }
    }

    double best_dist = 1.0e300;
    double best_s = 0.0, best_t = 0.0;

    {
        double t = 0.0;
        const double d = optimal_t_for_fixed_s(x1, a, x3, b, 0.0, C, t);
        if (d < best_dist) { best_dist = d; best_s = 0.0; best_t = t; }
    }
    {
        double t = 0.0;
        const double d = optimal_t_for_fixed_s(x1, a, x3, b, 1.0, C, t);
        if (d < best_dist) { best_dist = d; best_s = 1.0; best_t = t; }
    }
    {
        double s = 0.0;
        const double d = optimal_s_for_fixed_t(x1, a, x3, b, 0.0, A, s);
        if (d < best_dist) { best_dist = d; best_s = s; best_t = 0.0; }
    }
    {
        double s = 0.0;
        const double d = optimal_s_for_fixed_t(x1, a, x3, b, 1.0, A, s);
        if (d < best_dist) { best_dist = d; best_s = s; best_t = 1.0; }
    }

    out.s = best_s;
    out.t = best_t;
    out.closest_point_1 = x1 + out.s * a;
    out.closest_point_2 = x3 + out.t * b;
    out.distance = best_dist;
    out.region = (Delta > eps * eps) ? classify_region(best_s, best_t) : SegmentSegmentRegion::ParallelSegments;
    return out;
}

// ============================================================
// Corotated elasticity
// ============================================================
struct CorotatedCache32 {
    Mat22 S{Mat22::Zero()};
    Mat22 SInv{Mat22::Zero()};
    Mat32 R{Mat32::Zero()};
    Mat22 FTFinv{Mat22::Zero()};
    Mat32 FFTFInv{Mat32::Zero()};
    Eigen::Matrix3d FFTFInvFT{Eigen::Matrix3d::Zero()};
    double J{0.0};
    double traceS{0.0};
};

static inline int flatF(int a, int b) { return 2 * a + b; }

static Mat32 Ds(const TriangleDef& tri) {
    Mat32 out;
    out.col(0) = tri.x[1] - tri.x[0];
    out.col(1) = tri.x[2] - tri.x[0];
    return out;
}

static CorotatedCache32 buildCorotatedCache(const Mat32& F) {
    CorotatedCache32 c;
    const Mat22 C = F.transpose() * F;
    Eigen::SelfAdjointEigenSolver<Mat22> es(C);
    if (es.info() != Eigen::Success) throw std::runtime_error("Eigen decomposition failed.");

    Mat22 U = es.eigenvectors();
    Eigen::Vector2d evals = es.eigenvalues();
    evals(0) = std::max(evals(0), 1.0e-12);
    evals(1) = std::max(evals(1), 1.0e-12);

    Mat22 Sdiag = Mat22::Zero();
    Sdiag(0, 0) = std::sqrt(evals(0));
    Sdiag(1, 1) = std::sqrt(evals(1));

    c.S = U * Sdiag * U.transpose();
    c.SInv = c.S.inverse();
    c.R = F * c.SInv;
    c.J = Sdiag(0, 0) * Sdiag(1, 1);
    c.traceS = c.S.trace();
    c.FTFinv = C.inverse();
    c.FFTFInv = F * c.FTFinv;
    c.FFTFInvFT = c.FFTFInv * F.transpose();
    return c;
}

static double PsiCorotated32(const CorotatedCache32& c, const Mat32& F, double mu, double lambda) {
    return mu * (F - c.R).squaredNorm()
           + 0.5 * lambda * (c.J - 1.0) * (c.J - 1.0);
}

static Mat32 PCorotated32(const CorotatedCache32& c, const Mat32& F, double mu, double lambda) {
    return 2.0 * mu * (F - c.R)
           + lambda * (c.J - 1.0) * c.J * c.FFTFInv;
}

static void dPdFCorotated32(const CorotatedCache32& c, double mu, double lambda, Mat66& dPdF) {
    const Mat22& SInv      = c.SInv;
    const Mat32& R         = c.R;
    const Mat22& FTFinv    = c.FTFinv;
    const Mat32& FFTFInv   = c.FFTFInv;
    const Eigen::Matrix3d& FFTFInvFT = c.FFTFInvFT;
    const double J         = c.J;
    const double traceS    = c.traceS;

    const Eigen::Matrix3d RRT = R * R.transpose();

    Mat32 Re;
    Re(0,0) =  R(0,1); Re(0,1) = -R(0,0);
    Re(1,0) =  R(1,1); Re(1,1) = -R(1,0);
    Re(2,0) =  R(2,1); Re(2,1) = -R(2,0);

    Vec6 dcdF;
    dcdF << -R(0,1) / traceS,  R(0,0) / traceS,
            -R(1,1) / traceS,  R(1,0) / traceS,
            -R(2,1) / traceS,  R(2,0) / traceS;

    static constexpr int idx[12] = {0,0, 0,1, 1,0, 1,1, 2,0, 2,1};

    Mat66 dRdF = Mat66::Zero();
    for (int c1 = 0; c1 < 6; ++c1) {
        for (int c2 = 0; c2 < 6; ++c2) {
            const int m = idx[2 * c1], n = idx[2 * c1 + 1];
            const int i = idx[2 * c2], j = idx[2 * c2 + 1];
            double v = 0.0;
            if (m == i) v += SInv(j, n);
            v -= RRT(m, i) * SInv(j, n);
            v -= dcdF(c2) * Re(m, n);
            dRdF(c1, c2) = v;
        }
    }

    dPdF.setZero();
    for (int c1 = 0; c1 < 6; ++c1) {
        for (int c2 = 0; c2 < 6; ++c2) {
            const int m = idx[2 * c1], n = idx[2 * c1 + 1];
            const int i = idx[2 * c2], j = idx[2 * c2 + 1];
            double v = 0.0;
            if (m == i) v += lambda * (J - 1.0) * J * FTFinv(j, n);
            v -= lambda * (J - 1.0) * J *
                 (FFTFInv(m, j) * FFTFInv(i, n) + FFTFInvFT(m, i) * FTFinv(j, n));
            v += 0.5 * lambda * (2.0 * J - 1.0) * J *
                 (FFTFInv(i, j) * FFTFInv(m, n) + FFTFInv(i, j) * FFTFInv(m, n));
            dPdF(c1, c2) = v;
        }
    }

    dPdF += 2.0 * mu * (Mat66::Identity() - dRdF);
}

static double corotated_energy(double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda) {
    const Mat32 F = Ds(def) * Dm_inv;
    const CorotatedCache32 c = buildCorotatedCache(F);
    return ref_area * PsiCorotated32(c, F, mu, lambda);
}

using ShapeGrads = std::array<Vec2, 3>;

static ShapeGrads shape_function_gradients(const Mat22& Dm_inv) {
    ShapeGrads grads;
    grads[1] = Dm_inv.row(0).transpose();
    grads[2] = Dm_inv.row(1).transpose();
    grads[0] = -grads[1] - grads[2];
    return grads;
}

static Vec3 corotated_node_gradient(const Mat32& P, double ref_area, const ShapeGrads& gradN, int node) {
    Vec3 g = Vec3::Zero();
    for (int gamma = 0; gamma < 3; ++gamma) {
        double value = 0.0;
        for (int beta = 0; beta < 2; ++beta) value += P(gamma, beta) * gradN[node](beta);
        g(gamma) = ref_area * value;
    }
    return g;
}

static Mat39 corotated_node_hessian(const Mat66& dPdF, double ref_area, const ShapeGrads& gradN, int node) {
    Mat39 H = Mat39::Zero();
    for (int j = 0; j < 3; ++j) {
        for (int gamma = 0; gamma < 3; ++gamma) {
            for (int delta = 0; delta < 3; ++delta) {
                double value = 0.0;
                for (int beta = 0; beta < 2; ++beta) {
                    for (int eta = 0; eta < 2; ++eta) {
                        value += dPdF(flatF(gamma, beta), flatF(delta, eta))
                                 * gradN[node](beta) * gradN[j](eta);
                    }
                }
                H(gamma, 3 * j + delta) = ref_area * value;
            }
        }
    }
    return H;
}

// ============================================================
// Barrier
// ============================================================
static double scalar_barrier(double delta, double d_hat) {
    if (d_hat <= 0.0) return 0.0;
    const double d = safe_distance(delta);
    if (d >= d_hat) return 0.0;
    const double s = d - d_hat;
    return -(s * s) * std::log(d / d_hat);
}

static double scalar_barrier_gradient(double delta, double d_hat) {
    if (d_hat <= 0.0) return 0.0;
    const double d = safe_distance(delta);
    if (d >= d_hat) return 0.0;
    const double s = d - d_hat;
    return -2.0 * s * std::log(d / d_hat) - (s * s) / d;
}

static double scalar_barrier_hessian(double delta, double d_hat) {
    if (d_hat <= 0.0) return 0.0;
    const double d = safe_distance(delta);
    if (d >= d_hat) return 0.0;
    const double ratio = d_hat / d;
    return ratio * ratio + 2.0 * ratio - 3.0 - 2.0 * std::log(d / d_hat);
}

static double segment_parameter_from_closest_point(const Vec3& q, const Vec3& a, const Vec3& b) {
    double denom = 0.0, numer = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double ab = b(k) - a(k);
        denom += ab * ab;
        numer += (q(k) - a(k)) * ab;
    }
    if (denom <= 0.0) return 0.0;
    return clamp_scalar(numer / denom, 0.0, 1.0);
}

static double levi_civita(int i, int j, int k) {
    return 0.5 * (i - j) * (j - k) * (k - i);
}

static double node_triangle_barrier(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps = 1.0e-12) {
    const auto dr = node_triangle_distance(x, x1, x2, x3, eps);
    return scalar_barrier(dr.distance, d_hat);
}

static Vec3 node_triangle_barrier_gradient(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, int dof, double eps = 1.0e-12) {
    const auto dr = node_triangle_distance(x, x1, x2, x3, eps);
    const double delta = safe_distance(dr.distance);
    const double bp = scalar_barrier_gradient(delta, d_hat);

    Vec3 g = Vec3::Zero();
    if (bp == 0.0) return g;

    double u[3];
    for (int k = 0; k < 3; ++k) u[k] = (x(k) - dr.closest_point(k)) / delta;

    double coeff[4] = {0.0, 0.0, 0.0, 0.0};
    double face_n[3] = {0.0, 0.0, 0.0};
    bool use_normal = false;

    switch (dr.region) {
        case NodeTriangleRegion::FaceInterior: {
            const double phi = dr.phi;
            const double sphi = (phi > 0.0) ? 1.0 : (phi < 0.0) ? -1.0 : 0.0;
            coeff[0] =  bp * sphi;
            coeff[1] = -bp * sphi * dr.barycentric_tilde_x[0];
            coeff[2] = -bp * sphi * dr.barycentric_tilde_x[1];
            coeff[3] = -bp * sphi * dr.barycentric_tilde_x[2];
            for (int k = 0; k < 3; ++k) face_n[k] = dr.normal(k);
            use_normal = true;
            break;
        }
        case NodeTriangleRegion::Edge12: {
            const double t = segment_parameter_from_closest_point(dr.closest_point, x1, x2);
            coeff[0] =  bp; coeff[1] = -bp * (1.0 - t); coeff[2] = -bp * t; coeff[3] = 0.0;
            break;
        }
        case NodeTriangleRegion::Edge23: {
            const double t = segment_parameter_from_closest_point(dr.closest_point, x2, x3);
            coeff[0] =  bp; coeff[1] =  0.0; coeff[2] = -bp * (1.0 - t); coeff[3] = -bp * t;
            break;
        }
        case NodeTriangleRegion::Edge31: {
            const double t = segment_parameter_from_closest_point(dr.closest_point, x3, x1);
            coeff[0] =  bp; coeff[1] = -bp * t; coeff[2] =  0.0; coeff[3] = -bp * (1.0 - t);
            break;
        }
        case NodeTriangleRegion::Vertex1: coeff[0] =  bp; coeff[1] = -bp; coeff[2] =  0.0; coeff[3] =  0.0; break;
        case NodeTriangleRegion::Vertex2: coeff[0] =  bp; coeff[1] =  0.0; coeff[2] = -bp; coeff[3] =  0.0; break;
        case NodeTriangleRegion::Vertex3: coeff[0] =  bp; coeff[1] =  0.0; coeff[2] =  0.0; coeff[3] = -bp; break;
        case NodeTriangleRegion::DegenerateTriangle: {
            coeff[0] = bp;
            double d1 = (dr.closest_point - x1).norm();
            double d2 = (dr.closest_point - x2).norm();
            double d3 = (dr.closest_point - x3).norm();
            if (d1 <= d2 && d1 <= d3) coeff[1] = -bp;
            else if (d2 <= d3) coeff[2] = -bp;
            else coeff[3] = -bp;
            break;
        }
    }

    if (use_normal) {
        for (int k = 0; k < 3; ++k) g(k) = coeff[dof] * face_n[k];
    } else {
        for (int k = 0; k < 3; ++k) g(k) = coeff[dof] * u[k];
    }
    return g;
}

static Mat312 node_triangle_barrier_hessian(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, int dof, double eps = 1.0e-12) {
    Mat312 H_row = Mat312::Zero();

    const auto dr = node_triangle_distance(x, x1, x2, x3, eps);
    const double delta = safe_distance(dr.distance);
    const double bp  = scalar_barrier_gradient(delta, d_hat);
    const double bpp = scalar_barrier_hessian(delta, d_hat);

    if (bp == 0.0 && bpp == 0.0) return H_row;

    const Vec3* Y[4] = {&x, &x1, &x2, &x3};
    const int p = dof;

    switch (dr.region) {
        case NodeTriangleRegion::Vertex1:
        case NodeTriangleRegion::Vertex2:
        case NodeTriangleRegion::Vertex3: {
            const int a_idx = (dr.region == NodeTriangleRegion::Vertex1) ? 1 :
                              (dr.region == NodeTriangleRegion::Vertex2) ? 2 : 3;
            double sp[4] = {0.0, 0.0, 0.0, 0.0};
            sp[0] = 1.0;
            sp[a_idx] = -1.0;
            if (sp[p] == 0.0) break;

            double u[3];
            for (int k = 0; k < 3; ++k) u[k] = (x(k) - (*Y[a_idx])(k)) / delta;

            const double c1 = bpp;
            const double c2 = bp / delta;
            for (int q = 0; q < 4; ++q) {
                if (sp[q] == 0.0) continue;
                const double sq = sp[p] * sp[q];
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const double dkl = (k == l) ? 1.0 : 0.0;
                        H_row(k, 3 * q + l) = sq * (c1 * u[k] * u[l] + c2 * (dkl - u[k] * u[l]));
                    }
                }
            }
            break;
        }

        case NodeTriangleRegion::Edge12:
        case NodeTriangleRegion::Edge23:
        case NodeTriangleRegion::Edge31: {
            int a_idx, b_idx;
            if      (dr.region == NodeTriangleRegion::Edge12) { a_idx = 1; b_idx = 2; }
            else if (dr.region == NodeTriangleRegion::Edge23) { a_idx = 2; b_idx = 3; }
            else                                              { a_idx = 3; b_idx = 1; }

            double omega[4]   = {0.0, 0.0, 0.0, 0.0};
            double epsilon[4] = {0.0, 0.0, 0.0, 0.0};
            omega[0] = 1.0;
            omega[a_idx] = -1.0;
            epsilon[a_idx] = -1.0;
            epsilon[b_idx] = 1.0;

            const Vec3& xa = *Y[a_idx];
            const Vec3& xb = *Y[b_idx];

            double e[3], w[3];
            for (int i = 0; i < 3; ++i) {
                e[i] = xb(i) - xa(i);
                w[i] = x(i)  - xa(i);
            }

            double alpha = 0.0, beta = 0.0;
            for (int i = 0; i < 3; ++i) {
                alpha += w[i] * e[i];
                beta  += e[i] * e[i];
            }
            const double t = alpha / beta;

            double r[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = x(i) - (xa(i) + t * e[i]);
                u[i] = r[i] / delta;
            }

            double t_d[4][3];
            double r_d[4][3][3];
            double q_d[4][3][3];

            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    const double alpha_pk = omega[pp] * e[k] + epsilon[pp] * w[k];
                    const double beta_pk  = 2.0 * epsilon[pp] * e[k];
                    t_d[pp][k] = alpha_pk / beta - alpha * beta_pk / (beta * beta);
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dpa = (pp == a_idx) ? 1.0 : 0.0;
                        const double dpx = (pp == 0) ? 1.0 : 0.0;
                        q_d[pp][k][i] = dpa * dik + t_d[pp][k] * e[i] + t * epsilon[pp] * dik;
                        r_d[pp][k][i] = dpx * dik - q_d[pp][k][i];
                    }
                }
            }

            for (int q = 0; q < 4; ++q) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const double dkl = (k == l) ? 1.0 : 0.0;

                        const double alpha_pk = omega[p] * e[k] + epsilon[p] * w[k];
                        const double alpha_ql = omega[q] * e[l] + epsilon[q] * w[l];
                        const double alpha_pkql = (omega[p] * epsilon[q] + omega[q] * epsilon[p]) * dkl;
                        const double beta_pk   = 2.0 * epsilon[p] * e[k];
                        const double beta_ql   = 2.0 * epsilon[q] * e[l];
                        const double beta_pkql = 2.0 * epsilon[p] * epsilon[q] * dkl;

                        const double t_pkql = alpha_pkql / beta
                                              - (alpha_pk * beta_ql + alpha_ql * beta_pk + alpha * beta_pkql) / (beta * beta)
                                              + 2.0 * alpha * beta_pk * beta_ql / (beta * beta * beta);

                        double ddelta_pk = 0.0, ddelta_ql = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            ddelta_pk += u[i] * r_d[p][k][i];
                            ddelta_ql += u[i] * r_d[q][l][i];
                        }

                        double proj_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                const double dij = (i == j) ? 1.0 : 0.0;
                                proj_term += (dij - u[i] * u[j]) * r_d[p][k][i] * r_d[q][l][j];
                            }
                        }
                        proj_term /= delta;

                        double uq_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            const double dik = (i == k) ? 1.0 : 0.0;
                            const double dil = (i == l) ? 1.0 : 0.0;
                            const double q_ipkql = t_pkql * e[i]
                                                   + t_d[p][k] * epsilon[q] * dil
                                                   + t_d[q][l] * epsilon[p] * dik;
                            uq_term += u[i] * q_ipkql;
                        }

                        const double d2delta = proj_term - uq_term;
                        H_row(k, 3 * q + l) = bpp * ddelta_pk * ddelta_ql + bp * d2delta;
                    }
                }
            }
            break;
        }

        case NodeTriangleRegion::FaceInterior: {
            double sig_a[4] = { 0.0, -1.0,  1.0,  0.0};
            double sig_b[4] = { 0.0, -1.0,  0.0,  1.0};
            double sig_w[4] = { 1.0, -1.0,  0.0,  0.0};

            double a[3], b[3], w[3];
            for (int i = 0; i < 3; ++i) {
                a[i] = x2(i) - x1(i);
                b[i] = x3(i) - x1(i);
                w[i] = x(i)  - x1(i);
            }

            double N[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < 3; ++i)
                for (int m = 0; m < 3; ++m)
                    for (int n = 0; n < 3; ++n)
                        N[i] += levi_civita(i, m, n) * a[m] * b[n];

            double eta = 0.0;
            for (int i = 0; i < 3; ++i) eta += N[i] * N[i];
            eta = std::sqrt(eta);

            double n[3];
            for (int i = 0; i < 3; ++i) n[i] = N[i] / eta;

            double psi = 0.0;
            for (int i = 0; i < 3; ++i) psi += N[i] * w[i];
            const double phi = psi / eta;
            const double s_sign = (phi > 0.0) ? 1.0 : (phi < 0.0) ? -1.0 : 0.0;

            double Nd[4][3][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    for (int i = 0; i < 3; ++i) {
                        double val = 0.0;
                        for (int nn = 0; nn < 3; ++nn) val += sig_a[pp] * levi_civita(i, k, nn) * b[nn];
                        for (int m = 0; m < 3; ++m)   val += sig_b[pp] * levi_civita(i, m, k) * a[m];
                        Nd[pp][k][i] = val;
                    }
                }
            }

            double eta_d[4][3], psi_d[4][3], phi_d[4][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    double eta_pk = 0.0;
                    for (int i = 0; i < 3; ++i) eta_pk += n[i] * Nd[pp][k][i];
                    eta_d[pp][k] = eta_pk;

                    double psi_pk = 0.0;
                    for (int i = 0; i < 3; ++i) psi_pk += Nd[pp][k][i] * w[i];
                    psi_pk += sig_w[pp] * N[k];
                    psi_d[pp][k] = psi_pk;

                    phi_d[pp][k] = psi_pk / eta - psi * eta_pk / (eta * eta);
                }
            }

            for (int q = 0; q < 4; ++q) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const double coeff_N2 = sig_a[p] * sig_b[q] - sig_a[q] * sig_b[p];

                        double nN2 = 0.0;
                        for (int i = 0; i < 3; ++i) nN2 += n[i] * coeff_N2 * levi_civita(i, k, l);

                        double proj_NN = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                const double dij = (i == j) ? 1.0 : 0.0;
                                proj_NN += (dij - n[i] * n[j]) * Nd[p][k][i] * Nd[q][l][j];
                            }
                        }

                        const double eta_pkql = nN2 + proj_NN / eta;

                        double psi_pkql = 0.0;
                        for (int i = 0; i < 3; ++i) psi_pkql += coeff_N2 * levi_civita(i, k, l) * w[i];
                        psi_pkql += sig_w[q] * Nd[p][k][l];
                        psi_pkql += sig_w[p] * Nd[q][l][k];

                        const double phi_pkql =
                                psi_pkql / eta
                                - (psi_d[p][k] * eta_d[q][l] + psi_d[q][l] * eta_d[p][k] + psi * eta_pkql) / (eta * eta)
                                + 2.0 * psi * eta_d[p][k] * eta_d[q][l] / (eta * eta * eta);

                        H_row(k, 3 * q + l) = bpp * phi_d[p][k] * phi_d[q][l] + s_sign * bp * phi_pkql;
                    }
                }
            }
            break;
        }

        case NodeTriangleRegion::DegenerateTriangle:
            break;
    }

    return H_row;
}

static double segment_segment_barrier(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, double eps = 1.0e-12) {
    const auto dr = segment_segment_distance(x1, x2, x3, x4, eps);
    return scalar_barrier(dr.distance, d_hat);
}

static Vec3 segment_segment_barrier_gradient(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, int dof, double eps = 1.0e-12) {
    const auto dr = segment_segment_distance(x1, x2, x3, x4, eps);
    const double delta = safe_distance(dr.distance);
    const double bp = scalar_barrier_gradient(delta, d_hat);

    Vec3 g = Vec3::Zero();
    if (bp == 0.0) return g;

    const Vec3 r = dr.closest_point_1 - dr.closest_point_2;
    double u[3];
    for (int k = 0; k < 3; ++k) u[k] = r(k) / delta;

    const double s = dr.s;
    const double t = dr.t;
    double mu[4] = {0.0, 0.0, 0.0, 0.0};

    switch (dr.region) {
        case SegmentSegmentRegion::Interior:     mu[0] =  bp * (1.0 - s); mu[1] =  bp * s;         mu[2] = -bp * (1.0 - t); mu[3] = -bp * t;         break;
        case SegmentSegmentRegion::Edge_s0:      mu[0] =  bp;             mu[1] =  0.0;            mu[2] = -bp * (1.0 - t); mu[3] = -bp * t;         break;
        case SegmentSegmentRegion::Edge_s1:      mu[0] =  0.0;            mu[1] =  bp;             mu[2] = -bp * (1.0 - t); mu[3] = -bp * t;         break;
        case SegmentSegmentRegion::Edge_t0:      mu[0] =  bp * (1.0 - s); mu[1] =  bp * s;         mu[2] = -bp;             mu[3] =  0.0;            break;
        case SegmentSegmentRegion::Edge_t1:      mu[0] =  bp * (1.0 - s); mu[1] =  bp * s;         mu[2] =  0.0;            mu[3] = -bp;             break;
        case SegmentSegmentRegion::Corner_s0t0:  mu[0] =  bp;             mu[1] =  0.0;            mu[2] = -bp;             mu[3] =  0.0;            break;
        case SegmentSegmentRegion::Corner_s0t1:  mu[0] =  bp;             mu[1] =  0.0;            mu[2] =  0.0;            mu[3] = -bp;             break;
        case SegmentSegmentRegion::Corner_s1t0:  mu[0] =  0.0;            mu[1] =  bp;             mu[2] = -bp;             mu[3] =  0.0;            break;
        case SegmentSegmentRegion::Corner_s1t1:  mu[0] =  0.0;            mu[1] =  bp;             mu[2] =  0.0;            mu[3] = -bp;             break;
        case SegmentSegmentRegion::ParallelSegments: {
            const double fallback[4] = {1.0 - s, s, -(1.0 - t), -t};
            for (int k = 0; k < 3; ++k) g(k) = bp * fallback[dof] * u[k];
            return g;
        }
    }

    for (int k = 0; k < 3; ++k) g(k) = mu[dof] * u[k];
    return g;
}

static Mat312 segment_segment_barrier_hessian(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, int dof, double eps = 1.0e-12) {
    Mat312 H_row = Mat312::Zero();

    const auto dr = segment_segment_distance(x1, x2, x3, x4, eps);
    const double delta = safe_distance(dr.distance);
    const double bp  = scalar_barrier_gradient(delta, d_hat);
    const double bpp = scalar_barrier_hessian(delta, d_hat);
    if (bp == 0.0 && bpp == 0.0) return H_row;

    const Vec3* Y[4] = {&x1, &x2, &x3, &x4};
    const int p = dof;

    switch (dr.region) {
        case SegmentSegmentRegion::Corner_s0t0:
        case SegmentSegmentRegion::Corner_s0t1:
        case SegmentSegmentRegion::Corner_s1t0:
        case SegmentSegmentRegion::Corner_s1t1: {
            int a_idx, b_idx;
            if      (dr.region == SegmentSegmentRegion::Corner_s0t0) { a_idx = 0; b_idx = 2; }
            else if (dr.region == SegmentSegmentRegion::Corner_s0t1) { a_idx = 0; b_idx = 3; }
            else if (dr.region == SegmentSegmentRegion::Corner_s1t0) { a_idx = 1; b_idx = 2; }
            else                                                     { a_idx = 1; b_idx = 3; }

            double sp[4] = {0.0, 0.0, 0.0, 0.0};
            sp[a_idx] = 1.0;
            sp[b_idx] = -1.0;
            if (sp[p] == 0.0) break;

            double u[3];
            for (int k = 0; k < 3; ++k) u[k] = ((*Y[a_idx])(k) - (*Y[b_idx])(k)) / delta;

            const double c1 = bpp;
            const double c2 = bp / delta;
            for (int q = 0; q < 4; ++q) {
                if (sp[q] == 0.0) continue;
                const double sq = sp[p] * sp[q];
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const double dkl = (k == l) ? 1.0 : 0.0;
                        H_row(k, 3 * q + l) = sq * (c1 * u[k] * u[l] + c2 * (dkl - u[k] * u[l]));
                    }
                }
            }
            break;
        }

        case SegmentSegmentRegion::Edge_s0:
        case SegmentSegmentRegion::Edge_s1:
        case SegmentSegmentRegion::Edge_t0:
        case SegmentSegmentRegion::Edge_t1: {
            int query_idx, ea_idx, eb_idx;
            if      (dr.region == SegmentSegmentRegion::Edge_s0) { query_idx = 0; ea_idx = 2; eb_idx = 3; }
            else if (dr.region == SegmentSegmentRegion::Edge_s1) { query_idx = 1; ea_idx = 2; eb_idx = 3; }
            else if (dr.region == SegmentSegmentRegion::Edge_t0) { query_idx = 2; ea_idx = 0; eb_idx = 1; }
            else                                                 { query_idx = 3; ea_idx = 0; eb_idx = 1; }

            const Vec3& xq  = *Y[query_idx];
            const Vec3& xea = *Y[ea_idx];
            const Vec3& xeb = *Y[eb_idx];

            double omega[4]   = {0.0, 0.0, 0.0, 0.0};
            double epsilon[4] = {0.0, 0.0, 0.0, 0.0};
            omega[query_idx] = 1.0;
            omega[ea_idx] = -1.0;
            epsilon[ea_idx] = -1.0;
            epsilon[eb_idx] = 1.0;

            double e[3], w[3];
            for (int i = 0; i < 3; ++i) {
                e[i] = xeb(i) - xea(i);
                w[i] = xq(i)  - xea(i);
            }

            double alpha = 0.0, beta = 0.0;
            for (int i = 0; i < 3; ++i) {
                alpha += w[i] * e[i];
                beta  += e[i] * e[i];
            }
            const double t_param = alpha / beta;

            double r[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = xq(i) - (xea(i) + t_param * e[i]);
                u[i] = r[i] / delta;
            }

            double t_d[4][3];
            double r_d[4][3][3];
            double q_d[4][3][3];

            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    const double alpha_pk = omega[pp] * e[k] + epsilon[pp] * w[k];
                    const double beta_pk  = 2.0 * epsilon[pp] * e[k];
                    t_d[pp][k] = alpha_pk / beta - alpha * beta_pk / (beta * beta);
                    for (int i = 0; i < 3; ++i) {
                        const double dik   = (i == k) ? 1.0 : 0.0;
                        const double dp_ea = (pp == ea_idx) ? 1.0 : 0.0;
                        const double dp_q  = (pp == query_idx) ? 1.0 : 0.0;
                        q_d[pp][k][i] = dp_ea * dik + t_d[pp][k] * e[i] + t_param * epsilon[pp] * dik;
                        r_d[pp][k][i] = dp_q * dik - q_d[pp][k][i];
                    }
                }
            }

            for (int q = 0; q < 4; ++q) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const double dkl = (k == l) ? 1.0 : 0.0;

                        const double alpha_pk = omega[p] * e[k] + epsilon[p] * w[k];
                        const double alpha_ql = omega[q] * e[l] + epsilon[q] * w[l];
                        const double alpha_pkql = (omega[p] * epsilon[q] + omega[q] * epsilon[p]) * dkl;
                        const double beta_pk = 2.0 * epsilon[p] * e[k];
                        const double beta_ql = 2.0 * epsilon[q] * e[l];
                        const double beta_pkql = 2.0 * epsilon[p] * epsilon[q] * dkl;

                        const double t_pkql = alpha_pkql / beta
                                              - (alpha_pk * beta_ql + alpha_ql * beta_pk + alpha * beta_pkql) / (beta * beta)
                                              + 2.0 * alpha * beta_pk * beta_ql / (beta * beta * beta);

                        double ddelta_pk = 0.0, ddelta_ql = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            ddelta_pk += u[i] * r_d[p][k][i];
                            ddelta_ql += u[i] * r_d[q][l][i];
                        }

                        double proj_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                const double dij = (i == j) ? 1.0 : 0.0;
                                proj_term += (dij - u[i] * u[j]) * r_d[p][k][i] * r_d[q][l][j];
                            }
                        }
                        proj_term /= delta;

                        double uq_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            const double dik = (i == k) ? 1.0 : 0.0;
                            const double dil = (i == l) ? 1.0 : 0.0;
                            const double q_ipkql = t_pkql * e[i]
                                                   + t_d[p][k] * epsilon[q] * dil
                                                   + t_d[q][l] * epsilon[p] * dik;
                            uq_term += u[i] * q_ipkql;
                        }

                        H_row(k, 3 * q + l) = bpp * ddelta_pk * ddelta_ql + bp * (proj_term - uq_term);
                    }
                }
            }
            break;
        }

        case SegmentSegmentRegion::Interior: {
            double sig_a[4] = {-1.0,  1.0,  0.0,  0.0};
            double sig_b[4] = { 0.0,  0.0, -1.0,  1.0};
            double sig_c[4] = { 1.0,  0.0, -1.0,  0.0};

            double a[3], b[3], c[3];
            for (int i = 0; i < 3; ++i) {
                a[i] = x2(i) - x1(i);
                b[i] = x4(i) - x3(i);
                c[i] = x1(i) - x3(i);
            }

            double A = 0.0, B = 0.0, C = 0.0, D = 0.0, E = 0.0;
            for (int i = 0; i < 3; ++i) {
                A += a[i] * a[i];
                B += a[i] * b[i];
                C += b[i] * b[i];
                D += a[i] * c[i];
                E += b[i] * c[i];
            }

            const double Delta = A * C - B * B;
            const double nu    = B * E - C * D;
            const double zeta  = A * E - B * D;
            const double s_val = nu / Delta;
            const double t_val = zeta / Delta;

            double Ad[4][3], Bd[4][3], Cd[4][3], Dd[4][3], Ed[4][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    Ad[pp][k] = 2.0 * sig_a[pp] * a[k];
                    Bd[pp][k] = sig_a[pp] * b[k] + sig_b[pp] * a[k];
                    Cd[pp][k] = 2.0 * sig_b[pp] * b[k];
                    Dd[pp][k] = sig_a[pp] * c[k] + sig_c[pp] * a[k];
                    Ed[pp][k] = sig_b[pp] * c[k] + sig_c[pp] * b[k];
                }
            }

            double nu_d[4][3], zeta_d[4][3], Delta_d[4][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    nu_d[pp][k]    = Bd[pp][k] * E + B * Ed[pp][k] - Cd[pp][k] * D - C * Dd[pp][k];
                    zeta_d[pp][k]  = Ad[pp][k] * E + A * Ed[pp][k] - Bd[pp][k] * D - B * Dd[pp][k];
                    Delta_d[pp][k] = Ad[pp][k] * C + A * Cd[pp][k] - 2.0 * B * Bd[pp][k];
                }
            }

            double s_d[4][3], t_d[4][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    s_d[pp][k] = nu_d[pp][k] / Delta   - nu   * Delta_d[pp][k] / (Delta * Delta);
                    t_d[pp][k] = zeta_d[pp][k] / Delta - zeta * Delta_d[pp][k] / (Delta * Delta);
                }
            }

            double r_vec[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r_vec[i] = (x1(i) + s_val * a[i]) - (x3(i) + t_val * b[i]);
                u[i] = r_vec[i] / delta;
            }

            double p_d[4][3][3], q_d_arr[4][3][3], r_d[4][3][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dp0 = (pp == 0) ? 1.0 : 0.0;
                        const double dp2 = (pp == 2) ? 1.0 : 0.0;
                        p_d[pp][k][i]     = dp0 * dik + s_d[pp][k] * a[i] + s_val * sig_a[pp] * dik;
                        q_d_arr[pp][k][i] = dp2 * dik + t_d[pp][k] * b[i] + t_val * sig_b[pp] * dik;
                        r_d[pp][k][i]     = p_d[pp][k][i] - q_d_arr[pp][k][i];
                    }
                }
            }

            for (int q = 0; q < 4; ++q) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const double dkl = (k == l) ? 1.0 : 0.0;

                        const double A_pkql = 2.0 * sig_a[p] * sig_a[q] * dkl;
                        const double B_pkql = (sig_a[p] * sig_b[q] + sig_a[q] * sig_b[p]) * dkl;
                        const double C_pkql = 2.0 * sig_b[p] * sig_b[q] * dkl;
                        const double D_pkql = (sig_a[p] * sig_c[q] + sig_a[q] * sig_c[p]) * dkl;
                        const double E_pkql = (sig_b[p] * sig_c[q] + sig_b[q] * sig_c[p]) * dkl;

                        const double nu_pkql = B_pkql * E + Bd[p][k] * Ed[q][l] + Bd[q][l] * Ed[p][k] + B * E_pkql
                                               - C_pkql * D - Cd[p][k] * Dd[q][l] - Cd[q][l] * Dd[p][k] - C * D_pkql;

                        const double Delta_pkql = A_pkql * C + Ad[p][k] * Cd[q][l] + Ad[q][l] * Cd[p][k] + A * C_pkql
                                                  - 2.0 * (Bd[p][k] * Bd[q][l] + B * B_pkql);

                        const double zeta_pkql = A_pkql * E + Ad[p][k] * Ed[q][l] + Ad[q][l] * Ed[p][k] + A * E_pkql
                                                 - B_pkql * D - Bd[p][k] * Dd[q][l] - Bd[q][l] * Dd[p][k] - B * D_pkql;

                        const double s_pkql = nu_pkql / Delta
                                              - (nu_d[p][k] * Delta_d[q][l] + nu_d[q][l] * Delta_d[p][k] + nu * Delta_pkql) / (Delta * Delta)
                                              + 2.0 * nu * Delta_d[p][k] * Delta_d[q][l] / (Delta * Delta * Delta);

                        const double t_pkql = zeta_pkql / Delta
                                              - (zeta_d[p][k] * Delta_d[q][l] + zeta_d[q][l] * Delta_d[p][k] + zeta * Delta_pkql) / (Delta * Delta)
                                              + 2.0 * zeta * Delta_d[p][k] * Delta_d[q][l] / (Delta * Delta * Delta);

                        double ddelta_pk = 0.0, ddelta_ql = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            ddelta_pk += u[i] * r_d[p][k][i];
                            ddelta_ql += u[i] * r_d[q][l][i];
                        }

                        double proj_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                const double dij = (i == j) ? 1.0 : 0.0;
                                proj_term += (dij - u[i] * u[j]) * r_d[p][k][i] * r_d[q][l][j];
                            }
                        }
                        proj_term /= delta;

                        double ur_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            const double dik = (i == k) ? 1.0 : 0.0;
                            const double dil = (i == l) ? 1.0 : 0.0;
                            const double p_ipkql = s_pkql * a[i] + s_d[p][k] * sig_a[q] * dil + s_d[q][l] * sig_a[p] * dik;
                            const double q_ipkql = t_pkql * b[i] + t_d[p][k] * sig_b[q] * dil + t_d[q][l] * sig_b[p] * dik;
                            ur_term += u[i] * (p_ipkql - q_ipkql);
                        }

                        H_row(k, 3 * q + l) = bpp * ddelta_pk * ddelta_ql + bp * (proj_term + ur_term);
                    }
                }
            }
            break;
        }

        case SegmentSegmentRegion::ParallelSegments:
            break;
    }

    return H_row;
}

static std::pair<Vec3, Mat312> node_triangle_barrier_gradient_and_hessian(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, int dof, double eps = 1.0e-12) {
    return {
            node_triangle_barrier_gradient(x, x1, x2, x3, d_hat, dof, eps),
            node_triangle_barrier_hessian(x, x1, x2, x3, d_hat, dof, eps)
    };
}

static std::pair<Vec3, Mat312> segment_segment_barrier_gradient_and_hessian(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, int dof, double eps = 1.0e-12) {
    return {
            segment_segment_barrier_gradient(x1, x2, x3, x4, d_hat, dof, eps),
            segment_segment_barrier_hessian(x1, x2, x3, x4, d_hat, dof, eps)
    };
}

// ============================================================
// Scene construction
// ============================================================
static void clear_scene(RefMesh& mesh, DeformedState& state, std::vector<Pin>& pins) {
    mesh.ref_positions.clear();
    mesh.tris.clear();
    mesh.Dm_inv.clear();
    mesh.area.clear();
    mesh.mass.clear();
    mesh.num_positions = 0;
    state.x.clear();
    state.v.clear();
    pins.clear();
}

static PatchInfo build_square_patch(
        RefMesh& mesh, DeformedState& state,
        int nx, int ny, double width, double height, const Vec3& origin) {
    PatchInfo patch;
    patch.vertex_begin = static_cast<int>(state.x.size());
    patch.tri_begin = num_tris(mesh);
    patch.nx = nx;
    patch.ny = ny;

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            const double u = static_cast<double>(i) / nx;
            const double v = static_cast<double>(j) / ny;
            const double xr = u * width;
            const double yr = v * height;
            mesh.ref_positions.push_back(Vec2(xr, yr));
            state.x.push_back(origin + Vec3(xr, 0.0, yr));
        }
    }

    auto vertex_index = [base = patch.vertex_begin, nx](int i, int j) {
        return base + j * (nx + 1) + i;
    };

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int v00 = vertex_index(i, j);
            const int v10 = vertex_index(i + 1, j);
            const int v01 = vertex_index(i, j + 1);
            const int v11 = vertex_index(i + 1, j + 1);

            mesh.tris.push_back(v00); mesh.tris.push_back(v10); mesh.tris.push_back(v11);
            mesh.tris.push_back(v00); mesh.tris.push_back(v11); mesh.tris.push_back(v01);
        }
    }

    patch.vertex_end = static_cast<int>(state.x.size());
    patch.tri_end = num_tris(mesh);
    return patch;
}

static void append_pin(std::vector<Pin>& pins, int vi, const std::vector<Vec3>& x) {
    pins.push_back(Pin{vi, x[vi]});
}

static int patch_corner_index(const PatchInfo& patch, int i, int j) {
    return patch.vertex_begin + j * (patch.nx + 1) + i;
}

static VertexTriangleMap build_incident_triangle_map(const std::vector<int>& tris) {
    VertexTriangleMap map;
    for (int idx = 0; idx < static_cast<int>(tris.size()); ++idx) {
        const int v = tris[idx];
        map[v].push_back({idx / 3, idx % 3});
    }
    return map;
}

static std::vector<std::array<int, 2>> extract_patch_edges(const RefMesh& mesh, const PatchInfo& patch) {
    std::set<std::pair<int, int>> edges;
    for (int t = patch.tri_begin; t < patch.tri_end; ++t) {
        for (int e = 0; e < 3; ++e) {
            int a = tri_vertex(mesh, t, e);
            int b = tri_vertex(mesh, t, (e + 1) % 3);
            if (a > b) std::swap(a, b);
            edges.insert({a, b});
        }
    }
    std::vector<std::array<int, 2>> out;
    out.reserve(edges.size());
    for (const auto& [a, b] : edges) out.push_back({a, b});
    return out;
}

static std::vector<NodeTrianglePair> build_inter_patch_node_triangle_pairs(
        const RefMesh& mesh, const PatchInfo& a, const PatchInfo& b) {
    std::vector<NodeTrianglePair> pairs;

    for (int node = a.vertex_begin; node < a.vertex_end; ++node) {
        for (int t = b.tri_begin; t < b.tri_end; ++t) {
            pairs.push_back(NodeTrianglePair{node,
                                             {tri_vertex(mesh, t, 0), tri_vertex(mesh, t, 1), tri_vertex(mesh, t, 2)}});
        }
    }
    for (int node = b.vertex_begin; node < b.vertex_end; ++node) {
        for (int t = a.tri_begin; t < a.tri_end; ++t) {
            pairs.push_back(NodeTrianglePair{node,
                                             {tri_vertex(mesh, t, 0), tri_vertex(mesh, t, 1), tri_vertex(mesh, t, 2)}});
        }
    }
    return pairs;
}

static std::vector<SegmentSegmentPair> build_inter_patch_segment_segment_pairs(
        const RefMesh& mesh, const PatchInfo& a, const PatchInfo& b) {
    const auto edges_a = extract_patch_edges(mesh, a);
    const auto edges_b = extract_patch_edges(mesh, b);

    std::vector<SegmentSegmentPair> pairs;
    pairs.reserve(edges_a.size() * edges_b.size());

    for (const auto& ea : edges_a) {
        for (const auto& eb : edges_b) {
            pairs.push_back(SegmentSegmentPair{{ea[0], ea[1], eb[0], eb[1]}});
        }
    }
    return pairs;
}

// ============================================================
// Solver / energy assembly
// ============================================================
static std::pair<Vec3, Mat33> compute_local_gradient_and_hessian_no_barrier(
        int vi, const RefMesh& mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
        const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat) {
    const double dt2 = params.dt2();
    Vec3 g = Vec3::Zero();
    Mat33 H = Mat33::Zero();

    g += mesh.mass[vi] * (x[vi] - xhat[vi]);
    g += dt2 * (-mesh.mass[vi] * params.gravity);
    H += mesh.mass[vi] * Mat33::Identity();

    for (const auto& pin : pins) {
        if (pin.vertex_index == vi) {
            g += dt2 * params.kpin * (x[vi] - pin.target_position);
            H += dt2 * params.kpin * Mat33::Identity();
        }
    }

    const auto it = adj.find(vi);
    if (it != adj.end()) {
        for (const auto& [tri_idx, local_node] : it->second) {
            const TriangleDef def = make_def_triangle(x, mesh, tri_idx);
            const Mat32 F = Ds(def) * mesh.Dm_inv[tri_idx];
            const double A = mesh.area[tri_idx];

            const CorotatedCache32 cache = buildCorotatedCache(F);
            const ShapeGrads gradN = shape_function_gradients(mesh.Dm_inv[tri_idx]);
            const Mat32 P = PCorotated32(cache, F, params.mu, params.lambda);
            Mat66 dPdF;
            dPdFCorotated32(cache, params.mu, params.lambda, dPdF);

            g += dt2 * corotated_node_gradient(P, A, gradN, local_node);
            H += dt2 * corotated_node_hessian(dPdF, A, gradN, local_node).template block<3, 3>(0, 3 * local_node);
        }
    }

    return {g, H};
}

static Vec3 compute_local_gradient(
        int vi, const RefMesh& mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
        const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
        const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs) {
    Vec3 g = compute_local_gradient_and_hessian_no_barrier(vi, mesh, adj, pins, params, x, xhat).first;
    if (params.d_hat <= 0.0) return g;
    const double dt2 = params.dt2();
    for (const auto& p : nt_pairs) {
        int dof = -1;
        if      (vi == p.node)      dof = 0;
        else if (vi == p.tri_v[0])  dof = 1;
        else if (vi == p.tri_v[1])  dof = 2;
        else if (vi == p.tri_v[2])  dof = 3;
        if (dof >= 0) g += dt2 * node_triangle_barrier_gradient(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, dof);
    }
    for (const auto& p : ss_pairs) {
        int dof = -1;
        if      (vi == p.v[0]) dof = 0;
        else if (vi == p.v[1]) dof = 1;
        else if (vi == p.v[2]) dof = 2;
        else if (vi == p.v[3]) dof = 3;
        if (dof >= 0) g += dt2 * segment_segment_barrier_gradient(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, dof);
    }
    return g;
}

static double compute_global_residual(
        const RefMesh& mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
        const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
        const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs) {
    double r_inf = 0.0;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        const Vec3 g = compute_local_gradient(i, mesh, adj, pins, params, x, xhat, nt_pairs, ss_pairs);
        r_inf = std::max(r_inf, g.cwiseAbs().maxCoeff());
    }
    return r_inf;
}

static void update_one_vertex(
        int vi, const RefMesh& mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
        const SimParams& params, const std::vector<Vec3>& xhat, std::vector<Vec3>& x,
        const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs) {
    static bool printed = false;
    if (!printed) {
        std::cout << "update_one_vertex: barrier "
                  << (params.d_hat > 0.0 ? "ENABLED" : "DISABLED")
                  << ", d_hat = " << params.d_hat << "\n";
        printed = true;
    }

    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, mesh, adj, pins, params, x, xhat);

    if (params.d_hat > 0.0) {
        const double dt2 = params.dt2();
        for (const auto& p : nt_pairs) {
            int dof = -1;
            if      (vi == p.node)      dof = 0;
            else if (vi == p.tri_v[0])  dof = 1;
            else if (vi == p.tri_v[1])  dof = 2;
            else if (vi == p.tri_v[2])  dof = 3;
            if (dof < 0) continue;
            auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(
                    x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, dof);
            g += dt2 * bg;
            H += dt2 * bH.block<3, 3>(0, 3 * dof);
        }
        for (const auto& p : ss_pairs) {
            int dof = -1;
            if      (vi == p.v[0]) dof = 0;
            else if (vi == p.v[1]) dof = 1;
            else if (vi == p.v[2]) dof = 2;
            else if (vi == p.v[3]) dof = 3;
            if (dof < 0) continue;
            auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(
                    x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, dof);
            g += dt2 * bg;
            H += dt2 * bH.block<3, 3>(0, 3 * dof);
        }
    }

    // Mild diagonal regularization improves robustness when barrier terms stiffen up.
    H += 1.0e-8 * Mat33::Identity();
    const Vec3 dx = matrix3d_inverse(H) * g;
    x[vi] -= params.step_weight * dx;
}

static SolverResult global_gauss_seidel_solver(
        const RefMesh& mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
        const SimParams& params, std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
        const std::vector<NodeTrianglePair>& nt_pairs, const std::vector<SegmentSegmentPair>& ss_pairs) {
    SolverResult result;
    result.initial_residual = compute_global_residual(mesh, adj, pins, params, xnew, xhat, nt_pairs, ss_pairs);
    result.final_residual = result.initial_residual;
    if (result.initial_residual < params.tol_abs) return result;

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        for (int vi = 0; vi < static_cast<int>(xnew.size()); ++vi) {
            update_one_vertex(vi, mesh, adj, pins, params, xhat, xnew, nt_pairs, ss_pairs);
        }
        result.final_residual = compute_global_residual(mesh, adj, pins, params, xnew, xhat, nt_pairs, ss_pairs);
        result.iterations = iter;
        if (result.final_residual < params.tol_abs) return result;
    }
    return result;
}

// ============================================================
// Output
// ============================================================
static void export_obj(const std::string& filename, const std::vector<Vec3>& x, const std::vector<int>& tris) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: cannot write " << filename << "\n";
        return;
    }
    for (const auto& p : x) out << "v " << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    for (int t = 0; t < static_cast<int>(tris.size()); t += 3) {
        out << "f " << (tris[t] + 1) << ' ' << (tris[t + 1] + 1) << ' ' << (tris[t + 2] + 1) << '\n';
    }
}

static void export_frame(const std::string& outdir, int frame, const std::vector<Vec3>& x, const std::vector<int>& tris) {
    std::ostringstream ss;
    ss << outdir << "/frame_" << std::setw(4) << std::setfill('0') << frame << ".obj";
    export_obj(ss.str(), x, tris);
}

// ============================================================
// Main experiment: two sheets, inter-sheet pairs only
// ============================================================
int main() {
    SimParams params;
    params.fps = 30.0;
    params.substeps = 1;
    params.num_frames = 60;
    params.mu = 10.0;
    params.lambda = 10.0;
    params.density = 1.0;
    params.thickness = 0.1;
    params.kpin = 1.0e7;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.max_global_iters = 500;
    params.tol_abs = 1.0e-6;
    params.step_weight = 1.0;

    // Toggle this to compare no-barrier vs barrier runs.
    params.d_hat = 0.2;

    std::cout << "num_frames = " << params.num_frames << "\n";
    std::cout << "d_hat = " << params.d_hat
              << (params.d_hat > 0.0 ? "  (barrier ON)" : "  (barrier OFF)")
              << "\n";

    RefMesh mesh;
    DeformedState state;
    std::vector<Pin> pins;
    clear_scene(mesh, state, pins);

    // Two separate sheets, side-by-side in x, with a small z offset.
    const PatchInfo left = build_square_patch(
            mesh, state, 2, 2, 0.4, 0.4,
            Vec3(-0.45, 0.20, 0.00)
    );

    const PatchInfo right = build_square_patch(
            mesh, state, 2, 2, 0.4, 0.4,
            Vec3(0.05, 0.20, 0.02)
    );

    state.v.assign(state.x.size(), Vec3::Zero());

    // Tiny asymmetry so they do not evolve perfectly symmetrically.
    state.x[right.vertex_begin + 0] += Vec3(-0.02, 0.00, -0.01);
    state.x[right.vertex_begin + 2] += Vec3(-0.02, 0.00, -0.01);

    // Pin the inner-side top and bottom corners.
    append_pin(pins, patch_corner_index(left,  left.nx, left.ny), state.x);   // left top-right
    append_pin(pins, patch_corner_index(left,  left.nx, 0),       state.x);   // left bottom-right

    append_pin(pins, patch_corner_index(right, 0,        right.ny), state.x); // right top-left
    append_pin(pins, patch_corner_index(right, 0,        0),        state.x); // right bottom-left

    mesh.initialize();
    mesh.build_lumped_mass(params.density, params.thickness);
    const VertexTriangleMap adj = build_incident_triangle_map(mesh.tris);

    // Only cross-sheet contact pairs.
    const auto nt_pairs = build_inter_patch_node_triangle_pairs(mesh, left, right);
    const auto ss_pairs = build_inter_patch_segment_segment_pairs(mesh, left, right);

    std::cout << "Vertices:  " << state.x.size() << "\n";
    std::cout << "Triangles: " << num_tris(mesh) << "\n";
    std::cout << "NT pairs:  " << nt_pairs.size() << "\n";
    std::cout << "SS pairs:  " << ss_pairs.size() << "\n";

    const std::string outdir = (params.d_hat > 0.0) ? "frames_clean_barrier_on" : "frames_clean_barrier_off";
    if (fs::exists(outdir)) fs::remove_all(outdir);
    fs::create_directories(outdir);
    export_frame(outdir, 0, state.x, mesh.tris);

    using Clock = std::chrono::steady_clock;
    const auto sim_start = Clock::now();
    double total_solver_ms = 0.0;

    for (int frame = 1; frame <= params.num_frames; ++frame) {
        const auto solver_start = Clock::now();
        SolverResult result;

        for (int sub = 0; sub < params.substeps; ++sub) {
            std::vector<Vec3> xhat;
            build_xhat(xhat, state.x, state.v, params.dt());

            std::vector<Vec3> xnew = state.x;
            result = global_gauss_seidel_solver(mesh, adj, pins, params, xnew, xhat, nt_pairs, ss_pairs);
            update_velocity(state.v, xnew, state.x, params.dt());
            state.x = xnew;
        }

        const auto solver_end = Clock::now();
        const double solver_ms = std::chrono::duration<double, std::milli>(solver_end - solver_start).count();
        total_solver_ms += solver_ms;

        std::cout << "Frame " << std::setw(4) << frame
                  << " | initial_residual = " << std::scientific << result.initial_residual
                  << " | final_residual = "   << std::scientific << result.final_residual
                  << " | global_iters = "     << std::setw(3)    << result.iterations
                  << " | solver_time = "      << std::fixed << std::setprecision(3)
                  << solver_ms << " ms\n";

        export_frame(outdir, frame, state.x, mesh.tris);
    }

    const auto sim_end = Clock::now();
    const double total_sim_ms = std::chrono::duration<double, std::milli>(sim_end - sim_start).count();

    std::cout << "\nSimulation finished.\n";
    std::cout << "Total simulation time: " << std::fixed << std::setprecision(3) << total_sim_ms << " ms\n";
    std::cout << "Total solver time:     " << total_solver_ms << " ms\n";
    std::cout << "Average solver time:   " << (total_solver_ms / params.num_frames) << " ms/frame\n";
    std::cout << "Frames written to:     " << outdir << "\n";
    return 0;
}