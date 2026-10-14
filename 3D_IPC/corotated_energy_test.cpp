#include "IPC_math.h"
#include "corotated_energy.h"
#include "SIMD.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ===========================================================================
//  Utilities
// ===========================================================================

namespace {

bool check_convergence(const std::string& label, double analytic, const std::vector<double>& hs, const std::vector<double>& errors, double noise_scale = 1e-10, bool verbose = true){
    const double noise_floor = noise_scale * (1.0 + std::abs(analytic));
    bool saw_reliable_slope = false;
    bool all_below_noise = true;
    bool passed = true;

    if (verbose) {
        std::cout << "  " << label
                  << "  analytic=" << std::scientific << std::setprecision(8)
                  << analytic << "\n";
    }

    for (std::size_t i = 1; i < hs.size(); ++i) {
        if (errors[i] < noise_floor || errors[i-1] < noise_floor) {
            if (verbose)
                std::cout << "    h=" << hs[i] << "  err=" << errors[i]
                          << "  (round-off regime, skipped)\n";
            continue;
        }
        all_below_noise = false;
        double slope = std::log(errors[i-1] / errors[i]) / std::log(hs[i-1] / hs[i]);
        if (verbose)
            std::cout << "    h=" << hs[i] << "  err=" << errors[i]
                      << "  slope=" << std::fixed << std::setprecision(6) << slope << "\n";
        saw_reliable_slope = true;
        if (slope < 1.99 || slope > 2.01) {
            std::cerr << "  FAIL: slope " << slope
                      << " outside [1.99, 2.01] for " << label << "\n";
            passed = false;
        }
    }

    if (all_below_noise) {
        if (verbose)
            std::cout << "    (all errors below noise floor -- exact match)\n";
        return true;
    }

    if (!saw_reliable_slope) {
        std::cerr << "  FAIL: no reliable slope data for " << label << "\n";
        passed = false;
    }
    return passed;
}

// ===========================================================================
//  Helper: assemble full 9x9 Hessian analytically from dPdF + shape grads
// ===========================================================================

using Mat99 = Eigen::Matrix<double, 9, 9>;

Mat99 assemble_hessian(const CorotatedCache32& cache, const Mat32& F,
                       double ref_area, const Mat22& Dm_inv,
                       double mu, double lambda) {
    Mat66 dPdF;
    dPdFCorotated32(cache, mu, lambda, dPdF);
    const ShapeGrads gradN = shape_function_gradients(Dm_inv);
    Mat99 H = Mat99::Zero();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int gamma = 0; gamma < 3; ++gamma) {
                for (int delta = 0; delta < 3; ++delta) {
                    double value = 0.0;
                    for (int beta = 0; beta < 2; ++beta) {
                        for (int eta = 0; eta < 2; ++eta) {
                            value += dPdF(2 * gamma + beta, 2 * delta + eta)
                                     * gradN[i](beta) * gradN[j](eta);
                        }
                    }
                    H(3 * i + gamma, 3 * j + delta) = ref_area * value;
                }
            }
        }
    }
    return H;
}

// ===========================================================================
//  Test configurations
// ===========================================================================

constexpr double kMu = 2.0;
constexpr double kLambda = 5.0;

struct TestTriangle {
    double ref_area;
    Mat22 Dm_inv;
};

TestTriangle MakeTestTriangle(){
    Vec2 X0(0.0, 0.0), X1(1.2, 0.1), X2(0.2, 0.9);
    Mat22 Dm_local;
    Dm_local.col(0) = X1 - X0;
    Dm_local.col(1) = X2 - X0;
    TestTriangle t;
    t.ref_area = 0.5 * std::abs(Dm_local.determinant());
    t.Dm_inv = Dm_local.inverse();
    return t;
}

TriangleDef EmbedRestTriangle(){
    TriangleDef def;
    def.x[0] = Vec3(0.0, 0.0, 0.0);
    def.x[1] = Vec3(1.2, 0.1, 0.0);
    def.x[2] = Vec3(0.2, 0.9, 0.0);
    return def;
}

TriangleDef MakeDeformedTriangle(){
    TriangleDef def;
    def.x[0] = Vec3(0.1, -0.2, 0.3);
    def.x[1] = Vec3(1.4, 0.2, -0.1);
    def.x[2] = Vec3(0.0, 1.0, 0.4);
    return def;
}

Mat32 compute_F(const Mat22& Dm_inv, const TriangleDef& def){
    Mat32 Ds;
    Ds.col(0) = def.x[1] - def.x[0];
    Ds.col(1) = def.x[2] - def.x[0];
    return Ds * Dm_inv;
}

}  // anonymous namespace

// ===========================================================================
//  Rest state has zero energy and zero gradient
// ===========================================================================

TEST(CorotatedEnergy, RestStateZeroEnergyAndGradient){
    const auto tri = MakeTestTriangle();
    const auto def = EmbedRestTriangle();
    const double E = corotated_energy(tri.ref_area, tri.Dm_inv, def, kMu, kLambda);

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);

    EXPECT_NEAR(E, 0.0, 1e-10) << "rest energy should be zero";
    {
        const Mat32 P = PCorotated32(cache, F, kMu, kLambda);
        const ShapeGrads gradN = shape_function_gradients(tri.Dm_inv);
        for (int i = 0; i < 3; ++i) {
            Vec3 g = corotated_node_gradient(P, tri.ref_area, gradN, i);
            EXPECT_LT(g.norm(), 1e-10) << "rest gradient should be zero (node " << i << ")";
        }
    }
}

// ===========================================================================
//  Rotation invariance -- rotated rest state has zero energy
// ===========================================================================

TEST(CorotatedEnergy, RotationInvariance){
    const auto tri = MakeTestTriangle();
    const auto def0 = EmbedRestTriangle();

    Eigen::AngleAxisd aa(0.7, Vec3(1.0, 2.0, -1.0).normalized());
    const Mat33 R = aa.toRotationMatrix();
    TriangleDef def1;
    for (int i = 0; i < 3; ++i) def1.x[i] = R * def0.x[i];

    const double E = corotated_energy(tri.ref_area, tri.Dm_inv, def1, kMu, kLambda);

    const Mat32 F = compute_F(tri.Dm_inv, def1);
    const CorotatedCache32 cache = buildCorotatedCache(F);

    EXPECT_NEAR(E, 0.0, 1e-10) << "rotated rest energy should be zero";
    {
        const Mat32 P = PCorotated32(cache, F, kMu, kLambda);
        const ShapeGrads gradN = shape_function_gradients(tri.Dm_inv);
        for (int i = 0; i < 3; ++i) {
            Vec3 g = corotated_node_gradient(P, tri.ref_area, gradN, i);
            EXPECT_LT(g.norm(), 1e-10) << "rotated rest gradient should be zero (node " << i << ")";
        }
    }
}

// ===========================================================================
//  Translation invariance
// ===========================================================================

TEST(CorotatedEnergy, TranslationInvariance){
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    TriangleDef shifted = def;
    Vec3 t(2.5, -1.25, 0.75);
    for (auto& x : shifted.x) x += t;

    const double E0 = corotated_energy(tri.ref_area, tri.Dm_inv, def, kMu, kLambda);
    const double E1 = corotated_energy(tri.ref_area, tri.Dm_inv, shifted, kMu, kLambda);

    EXPECT_NEAR(E0, E1, 1e-10) << "energy should be translation invariant";
}

// ===========================================================================
//  Nodal gradients sum to zero
// ===========================================================================

TEST(CorotatedEnergy, GradientSumZero){
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);

    Vec3 total = Vec3::Zero();
    {
        const Mat32 P = PCorotated32(cache, F, kMu, kLambda);
        const ShapeGrads gradN = shape_function_gradients(tri.Dm_inv);
        for (int i = 0; i < 3; ++i)
            total += corotated_node_gradient(P, tri.ref_area, gradN, i);
    }

    EXPECT_LT(total.norm(), 1e-10) << "nodal gradients should sum to zero";
}

// ===========================================================================
//  Hessian symmetry
// ===========================================================================

TEST(CorotatedEnergy, HessianSymmetry){
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const Mat99 H = assemble_hessian(cache, F, tri.ref_area, tri.Dm_inv, kMu, kLambda);

    double max_asym = (H - H.transpose()).cwiseAbs().maxCoeff();
    EXPECT_LT(max_asym, 1e-8) << "Hessian should be symmetric";
}

TEST(CorotatedEnergy, SelfBlockMatchesAnalytic){
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    Mat66 dPdF;
    dPdFCorotated32(cache, kMu, kLambda, dPdF);
    const ShapeGrads gradN = shape_function_gradients(tri.Dm_inv);

    for (int node = 0; node < 3; ++node) {
        Mat33 H_ref = Mat33::Zero();
        for (int gamma = 0; gamma < 3; ++gamma) {
            for (int delta = 0; delta < 3; ++delta) {
                double value = 0.0;
                for (int beta = 0; beta < 2; ++beta) {
                    for (int eta = 0; eta < 2; ++eta) {
                        value += dPdF(2 * gamma + beta, 2 * delta + eta)
                                 * gradN[node](beta) * gradN[node](eta);
                    }
                }
                H_ref(gamma, delta) = tri.ref_area * value;
            }
        }
        const Mat33 H_self = corotated_node_hessian(dPdF, tri.ref_area, gradN, node);
        const double err = (H_ref - H_self).cwiseAbs().maxCoeff();
        EXPECT_LT(err, 1e-12) << "corotated self-block mismatch at node " << node;
    }
}

// ===========================================================================
//  Hessian has translation null modes
// ===========================================================================

TEST(CorotatedEnergy, HessianTranslationNull){
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const Mat99 H = assemble_hessian(cache, F, tri.ref_area, tri.Dm_inv, kMu, kLambda);

    for (int axis = 0; axis < 3; ++axis) {
        Vec9 t = Vec9::Zero();
        for (int node = 0; node < 3; ++node)
            t(3 * node + axis) = 1.0;
        double res = (H * t).cwiseAbs().maxCoeff();
        EXPECT_LT(res, 1e-8) << "Hessian should have translation null mode (axis " << axis << ")";
    }
}

// ===========================================================================
//  Gradient convergence (slope = 2)
// ===========================================================================

TEST(CorotatedEnergy, GradientConvergence){
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);

    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};
    bool all_passed = true;

    const Mat32 P_base = PCorotated32(cache, F, kMu, kLambda);
    const ShapeGrads gradN_base = shape_function_gradients(tri.Dm_inv);

    for (int i = 0; i < 3; ++i) {
        Vec3 g = corotated_node_gradient(P_base, tri.ref_area, gradN_base, i);
        for (int c = 0; c < 3; ++c) {
            double analytic = g(c);

            if (std::abs(analytic) < 1e-14) continue;

            std::vector<double> errors;
            for (auto h : hs) {
                TriangleDef dp = def, dm = def;
                set_dof(dp, i, c, get_dof(dp, i, c) + h);
                set_dof(dm, i, c, get_dof(dm, i, c) - h);
                double fd = (corotated_energy(tri.ref_area, tri.Dm_inv, dp, kMu, kLambda)
                             - corotated_energy(tri.ref_area, tri.Dm_inv, dm, kMu, kLambda)) / (2.0 * h);
                errors.push_back(std::abs(fd - analytic));
            }

            std::string label = "g[" + std::to_string(i) + "](" + std::to_string(c) + ")";
            if (!check_convergence(label, analytic, hs, errors)) all_passed = false;
        }
    }

    EXPECT_TRUE(all_passed) << "gradient convergence failed";
}

// ===========================================================================
//  Hessian convergence (slope = 2)
// ===========================================================================

TEST(CorotatedEnergy, HessianConvergence){
    const auto tri = MakeTestTriangle();
    const auto def = MakeDeformedTriangle();

    const Mat32 F = compute_F(tri.Dm_inv, def);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const Mat99 H = assemble_hessian(cache, F, tri.ref_area, tri.Dm_inv, kMu, kLambda);

    const std::vector<double> hs = {1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4};
    bool all_passed = true;
    int tested = 0, skipped = 0;

    for (int j = 0; j < 3; ++j) {
        for (int d = 0; d < 3; ++d) {
            // Perturb node j component d, collect per-node gradients at each h
            std::vector<std::array<Vec3, 3>> g_plus_list, g_minus_list;

            for (auto h : hs) {
                TriangleDef dp = def, dm = def;
                set_dof(dp, j, d, get_dof(dp, j, d) + h);
                set_dof(dm, j, d, get_dof(dm, j, d) - h);

                Mat32 Fp = compute_F(tri.Dm_inv, dp);
                CorotatedCache32 cp = buildCorotatedCache(Fp);
                const Mat32 Pp = PCorotated32(cp, Fp, kMu, kLambda);
                const ShapeGrads gradNp = shape_function_gradients(tri.Dm_inv);
                std::array<Vec3, 3> gp, gm;
                for (int node = 0; node < 3; ++node)
                    gp[node] = corotated_node_gradient(Pp, tri.ref_area, gradNp, node);
                g_plus_list.push_back(gp);

                Mat32 Fm = compute_F(tri.Dm_inv, dm);
                CorotatedCache32 cm = buildCorotatedCache(Fm);
                const Mat32 Pm = PCorotated32(cm, Fm, kMu, kLambda);
                const ShapeGrads gradNm = shape_function_gradients(tri.Dm_inv);
                for (int node = 0; node < 3; ++node)
                    gm[node] = corotated_node_gradient(Pm, tri.ref_area, gradNm, node);
                g_minus_list.push_back(gm);
            }

            for (int i = 0; i < 3; ++i) {
                for (int c = 0; c < 3; ++c) {
                    double analytic = H(3*i+c, 3*j+d);

                    if (std::abs(analytic) < 1e-14) {
                        double fd_fine = (g_plus_list.back()[i](c) - g_minus_list.back()[i](c)) / (2.0 * hs.back());
                        if (std::abs(fd_fine) > 1e-6) {
                            std::cerr << "  FAIL: H(" << i << c << "," << j << d << ") analytic=0 but fd=" << fd_fine << "\n";
                            all_passed = false;
                        }
                        skipped++;
                        continue;
                    }

                    std::vector<double> errors;
                    for (std::size_t hi = 0; hi < hs.size(); ++hi) {
                        double fd = (g_plus_list[hi][i](c) - g_minus_list[hi][i](c)) / (2.0 * hs[hi]);
                        errors.push_back(std::abs(fd - analytic));
                    }

                    std::string label = "H(" + std::to_string(3*i+c) + "," + std::to_string(3*j+d) + ")";
                    if (!check_convergence(label, analytic, hs, errors, 1e-9, false)) {
                        check_convergence(label, analytic, hs, errors, 1e-9, true);
                        all_passed = false;
                    }
                    tested++;
                }
            }
        }
    }

    std::cout << "  Tested " << tested << " entries, skipped " << skipped << " zero entries\n";
    EXPECT_TRUE(all_passed) << "Hessian convergence failed";
}

// ===========================================================================
//  Stress: near-degenerate triangle (area ~ 0)
// ===========================================================================

TEST(CorotatedEnergy, NearDegenerateTriangle){
    // Nearly collapsed: vertices almost collinear, area ~ 1e-10
    Vec2 X0(0.0, 0.0), X1(1.0, 0.0), X2(0.5, 1e-10);
    Mat22 Dm_local;
    Dm_local.col(0) = X1 - X0;
    Dm_local.col(1) = X2 - X0;
    double ref_area = 0.5 * std::abs(Dm_local.determinant());
    ASSERT_TRUE(ref_area > 0.0) << "area should be nonzero";

    Mat22 Dm_inv = Dm_local.inverse();
    ASSERT_TRUE(std::isfinite(Dm_inv.norm())) << "Dm_inv should be finite";

    TriangleDef def;
    def.x[0] = Vec3(0.0, 0.0, 0.0);
    def.x[1] = Vec3(1.0, 0.0, 0.0);
    def.x[2] = Vec3(0.5, 1e-10, 0.0);

    double E = corotated_energy(ref_area, Dm_inv, def, kMu, kLambda);
    ASSERT_TRUE(std::isfinite(E)) << "energy should be finite for near-degenerate triangle";

    // Deform slightly and check gradient is finite
    TriangleDef deformed = def;
    deformed.x[2] = Vec3(0.5, 1e-10, 0.1);
    double E2 = corotated_energy(ref_area, Dm_inv, deformed, kMu, kLambda);
    ASSERT_TRUE(std::isfinite(E2)) << "deformed energy should be finite";

    const Mat32 F = compute_F(Dm_inv, deformed);
    const CorotatedCache32 cache = buildCorotatedCache(F);
    const Mat32 P = PCorotated32(cache, F, kMu, kLambda);
    const ShapeGrads gradN = shape_function_gradients(Dm_inv);
    for (int i = 0; i < 3; ++i) {
        Vec3 g = corotated_node_gradient(P, ref_area, gradN, i);
        ASSERT_TRUE(std::isfinite(g.norm())) << "gradient should be finite for near-degenerate triangle (node " << i << ")";
    }
}

// ===========================================================================
//  Stress: large coordinate values (coordinates ~ 1e6)
// ===========================================================================

TEST(CorotatedEnergy, LargeCoordinates){
    const auto tri = MakeTestTriangle();

    const double offset = 1e6;
    TriangleDef def;
    def.x[0] = Vec3(offset + 0.0, offset + 0.0, offset + 0.0);
    def.x[1] = Vec3(offset + 1.2, offset + 0.1, offset + 0.0);
    def.x[2] = Vec3(offset + 0.2, offset + 0.9, offset + 0.0);

    // This is the rest triangle shifted by a large offset -- energy should still be ~0
    double E = corotated_energy(tri.ref_area, tri.Dm_inv, def, kMu, kLambda);
    EXPECT_NEAR(E, 0.0, 1e-4) << "shifted rest energy should be near zero";

    // Deform and check
    TriangleDef deformed;
    deformed.x[0] = Vec3(offset + 0.1, offset - 0.2, offset + 0.3);
    deformed.x[1] = Vec3(offset + 1.4, offset + 0.2, offset - 0.1);
    deformed.x[2] = Vec3(offset + 0.0, offset + 1.0, offset + 0.4);

    double E2 = corotated_energy(tri.ref_area, tri.Dm_inv, deformed, kMu, kLambda);
    ASSERT_TRUE(std::isfinite(E2)) << "deformed energy at large coordinates should be finite";

    // Compare with the same deformation at origin
    const auto def_origin = MakeDeformedTriangle();
    double E_origin = corotated_energy(tri.ref_area, tri.Dm_inv, def_origin, kMu, kLambda);
    EXPECT_NEAR(E2, E_origin, 1e-4) << "energy should match regardless of coordinate magnitude";
}

namespace {

// A second-order scalar jet differentiates the smooth invariant energy, without
// using the polar cache, material tensor, or compact contraction being checked.
struct SmoothEnergyJet {
    long double value = 0;
    std::array<long double,3> gradient{};
    std::array<long double,9> hessian{};
    SmoothEnergyJet(long double v = 0) : value(v) {}
    friend SmoothEnergyJet operator+(const SmoothEnergyJet& a, const SmoothEnergyJet& b) {
        SmoothEnergyJet c(a.value+b.value);
        for (int i=0;i<3;++i) c.gradient[i]=a.gradient[i]+b.gradient[i];
        for (int i=0;i<9;++i) c.hessian[i]=a.hessian[i]+b.hessian[i];
        return c;
    }
    friend SmoothEnergyJet operator-(const SmoothEnergyJet& a, const SmoothEnergyJet& b) {
        SmoothEnergyJet c(a.value-b.value);
        for (int i=0;i<3;++i) c.gradient[i]=a.gradient[i]-b.gradient[i];
        for (int i=0;i<9;++i) c.hessian[i]=a.hessian[i]-b.hessian[i];
        return c;
    }
    friend SmoothEnergyJet operator*(const SmoothEnergyJet& a, const SmoothEnergyJet& b) {
        SmoothEnergyJet c(a.value*b.value);
        for (int i=0;i<3;++i) c.gradient[i]=a.gradient[i]*b.value+a.value*b.gradient[i];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j)
            c.hessian[3*i+j]=a.hessian[3*i+j]*b.value+a.value*b.hessian[3*i+j]
                +a.gradient[i]*b.gradient[j]+b.gradient[i]*a.gradient[j];
        return c;
    }
    friend SmoothEnergyJet sqrt(const SmoothEnergyJet& a) {
        SmoothEnergyJet c(std::sqrt(a.value));
        const long double first=.5L/c.value, second=-.25L/(a.value*c.value);
        for (int i=0;i<3;++i) c.gradient[i]=first*a.gradient[i];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j)
            c.hessian[3*i+j]=first*a.hessian[3*i+j]+second*a.gradient[i]*a.gradient[j];
        return c;
    }
};

SmoothEnergyJet independently_differentiated_membrane(const Mat32& F, const Vec2& shape,
    double area, double mu, double lambda) {
    SmoothEnergyJet c00, c01, c11;
    for (int axis=0;axis<3;++axis) {
        SmoothEnergyJet a(F(axis,0)), b(F(axis,1));
        a.gradient[axis]=shape[0]; b.gradient[axis]=shape[1];
        c00=c00+a*a; c01=c01+a*b; c11=c11+b*b;
    }
    const auto J=sqrt(c00*c11-c01*c01);
    const auto trace_s=sqrt(c00+c11+SmoothEnergyJet(2)*J);
    return SmoothEnergyJet(area)*(SmoothEnergyJet(mu)*(c00+c11-SmoothEnergyJet(2)*trace_s+SmoothEnergyJet(2))
        +SmoothEnergyJet(.5L*lambda)*(J-SmoothEnergyJet(1))*(J-SmoothEnergyJet(1)));
}

// Preserve the old basic SIMD operation graph as a fallback regression oracle.
// The independent energy jet above, rather than this graph, validates the math.
double old_basic_fma(double a,double b,double c) {
#if defined(__AVX512F__) || (defined(__AVX2__) && defined(__FMA__)) || defined(__aarch64__) || defined(_M_ARM64)
    return std::fma(a,b,c);
#else
    volatile double product=a*b;
    return product+c;
#endif
}

Mat33 old_basic_membrane_block(const Mat32& F, const Vec2& shape, double area, double mu, double lambda) {
    const auto cache=buildCorotatedCache(F);
    const double volumetric=lambda*(cache.J-1)*cache.J;
    const double positive=(.5*lambda)*(2*cache.J-1)*cache.J;
    Mat33 H;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        const double rrt=old_basic_fma(cache.R(i,1),cache.R(j,1),cache.R(i,0)*cache.R(j,0));
        const double Q=old_basic_fma(cache.FFTFInv(i,1),F(j,1),cache.FFTFInv(i,0)*F(j,0));
        double sum=0;
        for (int beta=0;beta<2;++beta) for (int eta=0;eta<2;++eta) {
            const double si=cache.SInv(eta,beta),ci=cache.FTFinv(eta,beta);
            const double dcd=(eta==0?0.0-cache.R(j,1):cache.R(j,0))/cache.traceS;
            const double re=beta==0?cache.R(i,1):0.0-cache.R(i,0);
            double dr=i==j?si:0;
            dr=old_basic_fma(-1.0*rrt,si,dr);
            dr=old_basic_fma(-1.0*dcd,re,dr);
            double dp=i==j?volumetric*ci:0;
            dp=old_basic_fma(-1.0*volumetric,old_basic_fma(cache.FFTFInv(i,eta),cache.FFTFInv(j,beta),Q*ci),dp);
            const double product=cache.FFTFInv(j,eta)*cache.FFTFInv(i,beta);
            dp=old_basic_fma(positive,product+product,dp);
            dp=old_basic_fma(2*mu,(i==j && beta==eta?1.0:0.0)-dr,dp);
            sum=old_basic_fma(dp*shape[beta],shape[eta],sum);
        }
        H(i,j)=area*sum;
    }
    return H;
}

} // namespace

TEST(CorotatedEnergy, BasicSIMDCompactBlockMatchesIndependentSmoothEnergy) {
    const auto tri=MakeTestTriangle();
    const auto shapes=shape_function_gradients(tri.Dm_inv);
    for (int sample=0;sample<18;++sample) for (int role=0;role<3;++role) {
        const Mat33 rotation=Eigen::AngleAxisd(.13*sample,Vec3(1,-2,3).normalized()).toRotationMatrix();
        Mat32 F=rotation.leftCols<2>();
        F.col(0)*=.5+.07*sample;
        F.col(1)=1.1*F.col(1)+.12*F.col(0);
        std::array<Vec3,3> x{Vec3(.1,-.2,.3),Vec3::Zero(),Vec3::Zero()};
        const Mat22 dm=tri.Dm_inv.inverse();
        x[1]=x[0]+F*dm.col(0);x[2]=x[0]+F*dm.col(1);
        Mat32 ds;ds.col(0)=x[1]-x[0];ds.col(1)=x[2]-x[0];F=ds*tri.Dm_inv;
        const auto reference=independently_differentiated_membrane(F,shapes[role],tri.ref_area,kMu,kLambda);
        Vec3 g;Mat33 H;
        ipc_simd::corotated_derivatives_tile(x.data(),&tri.Dm_inv,&tri.ref_area,&shapes[role],1,kMu,kLambda,&g,&H);
        Vec3 expected_g;Mat33 expected_H;
        for (int i=0;i<3;++i) {
            expected_g[i]=static_cast<double>(reference.gradient[i]);
            for (int j=0;j<3;++j) expected_H(i,j)=static_cast<double>(reference.hessian[3*i+j]);
        }
        EXPECT_LE((g-expected_g).norm(),2e-12*(1+expected_g.norm()));
        EXPECT_LE((H-expected_H).norm(),2e-12*(1+expected_H.norm()));
    }
}

TEST(CorotatedEnergy, BasicSIMDFallbackRetainsClampedAndExtremeOperationGraph) {
    const Mat22 identity=Mat22::Identity();
    const Vec2 shape(-.75,.625);const double area=.5;
    for (double second_scale : {0x1p-150,0x1p-22,0x1p-20,0.01,0x1p40}) {
        Mat32 F=Mat32::Zero();F(0,0)=second_scale>1?second_scale:1;F(1,1)=second_scale;
        const std::array<Vec3,3> x{Vec3::Zero(),F.col(0),F.col(1)};
        Vec3 g;Mat33 H;
        ipc_simd::corotated_derivatives_tile(x.data(),&identity,&area,&shape,1,kMu,kLambda,&g,&H);
        const Mat33 old=old_basic_membrane_block(F,shape,area,kMu,kLambda);
        EXPECT_EQ(std::memcmp(H.data(),old.data(),9*sizeof(double)),0)
            << "fallback changed at scale=" << second_scale;
    }
    for (const auto& values : std::array<std::array<double,3>,3>{{
            {{.125,.125,kLambda}}, {{3.0,3.0,kLambda}}, {{1.0,1.0,64.0}}}}) {
        Mat32 F=Mat32::Zero();F(0,0)=values[0];F(1,1)=values[1];
        const std::array<Vec3,3> x{Vec3::Zero(),F.col(0),F.col(1)};
        Vec3 g;Mat33 H;
        ipc_simd::corotated_derivatives_tile(x.data(),&identity,&area,&shape,1,kMu,values[2],&g,&H);
        const Mat33 old=old_basic_membrane_block(F,shape,area,kMu,values[2]);
        EXPECT_EQ(std::memcmp(H.data(),old.data(),9*sizeof(double)),0)
            << "fallback changed for compressed/stretched or high-bulk material";
    }
}
