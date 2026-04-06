#include "barrier_energy.h"

#include <cmath>
#include <stdexcept>

//  t = ((q - a) · (b - a)) / ||b - a||^2
double segment_parameter_from_closest_point(const Vec3& q, const Vec3& a, const Vec3& b){
    double denom = 0.0;
    double numer = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double ab_k = b(k) - a(k);
        denom += ab_k * ab_k;
        numer += (q(k) - a(k)) * ab_k;
    }
    if (denom <= 0.0) return 0.0;
    return clamp_scalar(numer / denom, 0.0, 1.0);
}

//  eps_{ijk} = (i-j)(j-k)(k-i) / 2
double levi_civita(int i, int j, int k){
    return 0.5 * (i - j) * (j - k) * (k - i);
}

//  Scalar barrier
double scalar_barrier(double delta, double d_hat){
    if (d_hat <= 0.0) throw std::runtime_error("scalar_barrier: d_hat must be positive.");
    if (delta >= d_hat) return 0.0;
    if (delta <= 0.0) throw std::runtime_error("scalar_barrier: delta must be positive.");
    const double s = delta - d_hat;
    return -(s * s) * std::log(delta / d_hat);
}

double scalar_barrier_gradient(double delta, double d_hat){
    if (d_hat <= 0.0) throw std::runtime_error("scalar_barrier_gradient: d_hat must be positive.");
    if (delta >= d_hat) return 0.0;
    if (delta <= 0.0) throw std::runtime_error("scalar_barrier_gradient: delta must be positive.");
    const double s = delta - d_hat;
    return -2.0 * s * std::log(delta / d_hat) - (s * s) / delta;
}

double scalar_barrier_hessian(double delta, double d_hat){
    if (d_hat <= 0.0) throw std::runtime_error("scalar_barrier_hessian: d_hat must be positive.");
    if (delta >= d_hat) return 0.0;
    if (delta <= 0.0) throw std::runtime_error("scalar_barrier_hessian: delta must be positive.");
    const double ratio = d_hat / delta;
    return ratio * ratio + 2.0 * ratio - 3.0 - 2.0 * std::log(delta / d_hat);
}

//  Node--triangle barrier
double node_triangle_barrier(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                             double d_hat, double eps){
    const NodeTriangleDistanceResult dr = node_triangle_distance(x, x1, x2, x3, eps);
    return scalar_barrier(dr.distance, d_hat);
}

// Single-DOF gradient: returns the 3-vector dE/d(y_dof) with dof: 0=x, 1=x1, 2=x2, 3=x3
Vec3 node_triangle_barrier_gradient(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                    double d_hat, int dof, double eps){
    const NodeTriangleDistanceResult dr = node_triangle_distance(x, x1, x2, x3, eps);
    const double delta = dr.distance;
    const double bp    = scalar_barrier_gradient(delta, d_hat);

    Vec3 g = Vec3::Zero();
    if (bp == 0.0) return g;
    if (delta <= 0.0) throw std::runtime_error("node_triangle_barrier_gradient: distance must be positive.");

    double u[3];
    for (int k = 0; k < 3; ++k) u[k] = (x(k) - dr.closest_point(k)) / delta;

    // Coefficients s[p] such that grad_p = s[p] * (something).
    // We compute all four then pick dof.
    double coeff[4] = {0.0, 0.0, 0.0, 0.0};  // scalar multiplier on u[k] per DOF
    double face_n[3] = {0.0, 0.0, 0.0};       // for FaceInterior: grad_p = coeff[p] * n_k
    bool use_normal = false;

    switch (dr.region) {

        case NodeTriangleRegion::FaceInterior:
        {
            const double phi = dr.phi;
            const double sphi = (phi > 0.0) ? 1.0 : (phi < 0.0) ? -1.0 : 0.0;
            const double l1 = dr.barycentric_tilde_x[0];
            const double l2 = dr.barycentric_tilde_x[1];
            const double l3 = dr.barycentric_tilde_x[2];
            coeff[0] =  bp * sphi;
            coeff[1] = -bp * sphi * l1;
            coeff[2] = -bp * sphi * l2;
            coeff[3] = -bp * sphi * l3;
            for (int k = 0; k < 3; ++k) face_n[k] = dr.normal(k);
            use_normal = true;
            break;
        }

        case NodeTriangleRegion::Edge12:
        {
            const double t = segment_parameter_from_closest_point(dr.closest_point, x1, x2);
            coeff[0] =  bp;
            coeff[1] = -bp * (1.0 - t);
            coeff[2] = -bp * t;
            coeff[3] =  0.0;
            break;
        }

        case NodeTriangleRegion::Edge23:
        {
            const double t = segment_parameter_from_closest_point(dr.closest_point, x2, x3);
            coeff[0] =  bp;
            coeff[1] =  0.0;
            coeff[2] = -bp * (1.0 - t);
            coeff[3] = -bp * t;
            break;
        }

        case NodeTriangleRegion::Edge31:
        {
            const double t = segment_parameter_from_closest_point(dr.closest_point, x3, x1);
            coeff[0] =  bp;
            coeff[1] = -bp * t;
            coeff[2] =  0.0;
            coeff[3] = -bp * (1.0 - t);
            break;
        }

        case NodeTriangleRegion::Vertex1:
            coeff[0] =  bp; coeff[1] = -bp; coeff[2] = 0.0; coeff[3] = 0.0;
            break;

        case NodeTriangleRegion::Vertex2:
            coeff[0] =  bp; coeff[1] = 0.0; coeff[2] = -bp; coeff[3] = 0.0;
            break;

        case NodeTriangleRegion::Vertex3:
            coeff[0] =  bp; coeff[1] = 0.0; coeff[2] = 0.0; coeff[3] = -bp;
            break;

        case NodeTriangleRegion::DegenerateTriangle:
        {
            coeff[0] = bp;
            double d1 = 0.0, d2 = 0.0, d3 = 0.0;
            for (int k = 0; k < 3; ++k) {
                double v1 = dr.closest_point(k) - x1(k); d1 += v1*v1;
                double v2 = dr.closest_point(k) - x2(k); d2 += v2*v2;
                double v3 = dr.closest_point(k) - x3(k); d3 += v3*v3;
            }
            if (std::sqrt(d1) <= std::sqrt(d2) && std::sqrt(d1) <= std::sqrt(d3))
                coeff[1] = -bp;
            else if (std::sqrt(d2) <= std::sqrt(d3))
                coeff[2] = -bp;
            else
                coeff[3] = -bp;
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

// Single-DOF hessian row: returns 3x12 block d^2E / (d(y_dof) d(y_all)) with dof: 0=x, 1=x1, 2=x2, 3=x3
static Eigen::Matrix<double, 3, 12> node_triangle_barrier_hessian_row(const Vec3& x, const Vec3& x1,
                                                                       const Vec3& x2, const Vec3& x3,
                                                                       double d_hat, int dof, double eps){
    Eigen::Matrix<double, 3, 12> H_row = Eigen::Matrix<double, 3, 12>::Zero();

    const NodeTriangleDistanceResult dr = node_triangle_distance(x, x1, x2, x3, eps);
    const double delta = dr.distance;
    const double bp    = scalar_barrier_gradient(delta, d_hat);
    const double bpp   = scalar_barrier_hessian(delta, d_hat);

    if (bp == 0.0 && bpp == 0.0) return H_row;
    if (delta <= 0.0) throw std::runtime_error("node_triangle_barrier_hessian: distance must be positive.");

    const Vec3* Y[4] = {&x, &x1, &x2, &x3};
    const int p = dof;  // fixed outer DOF

    switch (dr.region) {

        // ---------------------------------------------------------------
        //  Vertex cases: H_{pk,ql} = s_p*s_q*(b''*u_k*u_l + (b'/delta)*(dkl - u_k*u_l))
        // ---------------------------------------------------------------
        case NodeTriangleRegion::Vertex1:
        case NodeTriangleRegion::Vertex2:
        case NodeTriangleRegion::Vertex3:
        {
            int a_idx = (dr.region == NodeTriangleRegion::Vertex1) ? 1
                                                                   : (dr.region == NodeTriangleRegion::Vertex2) ? 2 : 3;

            double sp[4] = {0.0, 0.0, 0.0, 0.0};
            sp[0] = 1.0; sp[a_idx] = -1.0;

            double u[3];
            for (int k = 0; k < 3; ++k) u[k] = (x(k) - (*Y[a_idx])(k)) / delta;

            const double c1 = bpp;
            const double c2 = bp / delta;

            if (sp[p] == 0.0) break;  // entire row is zero

            for (int q = 0; q < 4; ++q) {
                if (sp[q] == 0.0) continue;
                const double sq = sp[p] * sp[q];
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        double dkl = (k == l) ? 1.0 : 0.0;
                        H_row(k, 3*q+l) = sq * (c1 * u[k] * u[l] + c2 * (dkl - u[k] * u[l]));
                    }
                }
            }
            break;
        }

            // ---------------------------------------------------------------
            //  Edge cases
            // ---------------------------------------------------------------
        case NodeTriangleRegion::Edge12:
        case NodeTriangleRegion::Edge23:
        case NodeTriangleRegion::Edge31:
        {
            int a_idx, b_idx, c_idx;
            if      (dr.region == NodeTriangleRegion::Edge12) { a_idx = 1; b_idx = 2; c_idx = 3; }
            else if (dr.region == NodeTriangleRegion::Edge23) { a_idx = 2; b_idx = 3; c_idx = 1; }
            else                                              { a_idx = 3; b_idx = 1; c_idx = 2; }

            double omega[4]   = {0.0, 0.0, 0.0, 0.0};
            double epsilon[4] = {0.0, 0.0, 0.0, 0.0};
            omega[0] = 1.0;          omega[a_idx] = -1.0;
            epsilon[a_idx] = -1.0;   epsilon[b_idx] = 1.0;

            const Vec3& xa = *Y[a_idx];
            const Vec3& xb = *Y[b_idx];

            double e[3], w[3];
            for (int i = 0; i < 3; ++i) { e[i] = xb(i) - xa(i); w[i] = x(i) - xa(i); }

            double alpha = 0.0, beta = 0.0;
            for (int i = 0; i < 3; ++i) { alpha += w[i]*e[i]; beta += e[i]*e[i]; }
            double t = alpha / beta;

            double r[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = x(i) - (xa(i) + t * e[i]);
                u[i] = r[i] / delta;
            }

            // Precompute t_{,pk} and r_{i,pk} for all (p,k) — needed for both rows
            double t_d[4][3];
            double r_d[4][3][3];
            double q_d[4][3][3];

            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    double alpha_pk = omega[pp]*e[k] + epsilon[pp]*w[k];
                    double beta_pk  = 2.0*epsilon[pp]*e[k];
                    t_d[pp][k] = alpha_pk/beta - alpha*beta_pk/(beta*beta);
                    for (int i = 0; i < 3; ++i) {
                        double dik  = (i == k) ? 1.0 : 0.0;
                        double dpa  = (pp == a_idx) ? 1.0 : 0.0;
                        double dpx  = (pp == 0) ? 1.0 : 0.0;
                        q_d[pp][k][i] = dpa*dik + t_d[pp][k]*e[i] + t*epsilon[pp]*dik;
                        r_d[pp][k][i] = dpx*dik - q_d[pp][k][i];
                    }
                }
            }

            // Fixed outer row = p
            for (int q = 0; q < 4; ++q) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        double dkl = (k == l) ? 1.0 : 0.0;

                        double alpha_pk = omega[p]*e[k] + epsilon[p]*w[k];
                        double alpha_ql = omega[q]*e[l] + epsilon[q]*w[l];
                        double alpha_pkql = (omega[p]*epsilon[q] + omega[q]*epsilon[p])*dkl;
                        double beta_pk   = 2.0*epsilon[p]*e[k];
                        double beta_ql   = 2.0*epsilon[q]*e[l];
                        double beta_pkql = 2.0*epsilon[p]*epsilon[q]*dkl;

                        double t_pkql = alpha_pkql/beta
                                        - (alpha_pk*beta_ql + alpha_ql*beta_pk + alpha*beta_pkql)/(beta*beta)
                                        + 2.0*alpha*beta_pk*beta_ql/(beta*beta*beta);

                        double ddelta_pk = 0.0, ddelta_ql = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            ddelta_pk += u[i]*r_d[p][k][i];
                            ddelta_ql += u[i]*r_d[q][l][i];
                        }

                        double proj_term = 0.0;
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j) {
                                double dij = (i == j) ? 1.0 : 0.0;
                                proj_term += (dij - u[i]*u[j])*r_d[p][k][i]*r_d[q][l][j];
                            }
                        proj_term /= delta;

                        double uq_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            double dik = (i == k) ? 1.0 : 0.0;
                            double dil = (i == l) ? 1.0 : 0.0;
                            double q_ipkql = t_pkql*e[i]
                                             + t_d[p][k]*epsilon[q]*dil
                                             + t_d[q][l]*epsilon[p]*dik;
                            uq_term += u[i]*q_ipkql;
                        }

                        double d2delta = proj_term - uq_term;
                        H_row(k, 3*q+l) = bpp*ddelta_pk*ddelta_ql + bp*d2delta;
                    }
                }
            }
            break;
        }

            // ---------------------------------------------------------------
            //  Face interior case
            // ---------------------------------------------------------------
        case NodeTriangleRegion::FaceInterior:
        {
            double sig_a[4] = { 0.0, -1.0,  1.0,  0.0};
            double sig_b[4] = { 0.0, -1.0,  0.0,  1.0};
            double sig_w[4] = { 1.0, -1.0,  0.0,  0.0};

            double a[3], b[3], w[3];
            for (int i = 0; i < 3; ++i) {
                a[i] = x1(i) - x(i);   // Note: a=x2-x1, b=x3-x1 in original; kept same local vars
                // Actually use original convention: a=x2-x1, b=x3-x1, w=x-x1
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
            for (int i = 0; i < 3; ++i) eta += N[i]*N[i];
            eta = std::sqrt(eta);

            double n[3];
            for (int i = 0; i < 3; ++i) n[i] = N[i]/eta;

            double psi = 0.0;
            for (int i = 0; i < 3; ++i) psi += N[i]*w[i];
            double phi = psi/eta;

            double s_sign = (phi > 0.0) ? 1.0 : (phi < 0.0) ? -1.0 : 0.0;

            // Nd[pp][k][i] = N_{i,pk}
            double Nd[4][3][3];
            for (int pp = 0; pp < 4; ++pp)
                for (int k = 0; k < 3; ++k)
                    for (int i = 0; i < 3; ++i) {
                        double val = 0.0;
                        for (int nn = 0; nn < 3; ++nn)
                            val += sig_a[pp]*levi_civita(i,k,nn)*b[nn];
                        for (int m = 0; m < 3; ++m)
                            val += sig_b[pp]*levi_civita(i,m,k)*a[m];
                        Nd[pp][k][i] = val;
                    }

            double eta_d[4][3], psi_d[4][3], phi_d[4][3];
            for (int pp = 0; pp < 4; ++pp)
                for (int k = 0; k < 3; ++k) {
                    double eta_pk = 0.0;
                    for (int i = 0; i < 3; ++i) eta_pk += n[i]*Nd[pp][k][i];
                    eta_d[pp][k] = eta_pk;

                    double psi_pk = 0.0;
                    for (int i = 0; i < 3; ++i) psi_pk += Nd[pp][k][i]*w[i];
                    psi_pk += sig_w[pp]*N[k];
                    psi_d[pp][k] = psi_pk;

                    phi_d[pp][k] = psi_pk/eta - psi*eta_pk/(eta*eta);
                }

            // Fixed outer row = p
            for (int q = 0; q < 4; ++q) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        double coeff_N2 = sig_a[p]*sig_b[q] - sig_a[q]*sig_b[p];

                        double nN2 = 0.0;
                        for (int i = 0; i < 3; ++i)
                            nN2 += n[i]*coeff_N2*levi_civita(i,k,l);

                        double proj_NN = 0.0;
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j) {
                                double dij = (i==j) ? 1.0 : 0.0;
                                proj_NN += (dij - n[i]*n[j])*Nd[p][k][i]*Nd[q][l][j];
                            }

                        double eta_pkql = nN2 + proj_NN/eta;

                        double psi_pkql = 0.0;
                        for (int i = 0; i < 3; ++i)
                            psi_pkql += coeff_N2*levi_civita(i,k,l)*w[i];
                        psi_pkql += sig_w[q]*Nd[p][k][l];
                        psi_pkql += sig_w[p]*Nd[q][l][k];

                        double phi_pkql = psi_pkql/eta
                                          - (psi_d[p][k]*eta_d[q][l] + psi_d[q][l]*eta_d[p][k]
                                             + psi*eta_pkql)/(eta*eta)
                                          + 2.0*psi*eta_d[p][k]*eta_d[q][l]/(eta*eta*eta);

                        H_row(k, 3*q+l) = bpp*phi_d[p][k]*phi_d[q][l] + s_sign*bp*phi_pkql;
                    }
                }
            }
            break;
        }

        case NodeTriangleRegion::DegenerateTriangle:
            break;  // leave as zero
    }

    return H_row;
}

Mat33 node_triangle_barrier_hessian(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
                                    double d_hat, int dof, double eps){
    Mat33 H = Mat33::Zero();

    const NodeTriangleDistanceResult dr = node_triangle_distance(x, x1, x2, x3, eps);
    const double delta = dr.distance;
    const double bp    = scalar_barrier_gradient(delta, d_hat);
    const double bpp   = scalar_barrier_hessian(delta, d_hat);

    if (bp == 0.0 && bpp == 0.0) return H;
    if (delta <= 0.0) throw std::runtime_error("node_triangle_barrier_hessian: distance must be positive.");

    const Vec3* Y[4] = {&x, &x1, &x2, &x3};
    const int p = dof;

    switch (dr.region) {
        case NodeTriangleRegion::Vertex1:
        case NodeTriangleRegion::Vertex2:
        case NodeTriangleRegion::Vertex3:
        {
            const int a_idx = (dr.region == NodeTriangleRegion::Vertex1) ? 1
                            : (dr.region == NodeTriangleRegion::Vertex2) ? 2 : 3;

            double sp[4] = {0.0, 0.0, 0.0, 0.0};
            sp[0] = 1.0;
            sp[a_idx] = -1.0;
            if (sp[p] == 0.0) break;

            double u[3];
            for (int k = 0; k < 3; ++k) u[k] = (x(k) - (*Y[a_idx])(k)) / delta;

            const double c1 = bpp;
            const double c2 = bp / delta;
            const double sq = sp[p] * sp[p];

            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const double dkl = (k == l) ? 1.0 : 0.0;
                    H(k, l) = sq * (c1 * u[k] * u[l] + c2 * (dkl - u[k] * u[l]));
                }
            }
            break;
        }

        case NodeTriangleRegion::Edge12:
        case NodeTriangleRegion::Edge23:
        case NodeTriangleRegion::Edge31:
        {
            int a_idx, b_idx;
            if      (dr.region == NodeTriangleRegion::Edge12) { a_idx = 1; b_idx = 2; }
            else if (dr.region == NodeTriangleRegion::Edge23) { a_idx = 2; b_idx = 3; }
            else                                              { a_idx = 3; b_idx = 1; }

            double omega[4]   = {0.0, 0.0, 0.0, 0.0};
            double epsilon[4] = {0.0, 0.0, 0.0, 0.0};
            omega[0] = 1.0;          omega[a_idx] = -1.0;
            epsilon[a_idx] = -1.0;   epsilon[b_idx] = 1.0;

            const Vec3& xa = *Y[a_idx];
            const Vec3& xb = *Y[b_idx];

            double e[3], w[3];
            for (int i = 0; i < 3; ++i) { e[i] = xb(i) - xa(i); w[i] = x(i) - xa(i); }

            double alpha = 0.0, beta = 0.0;
            for (int i = 0; i < 3; ++i) { alpha += w[i] * e[i]; beta += e[i] * e[i]; }
            const double t = alpha / beta;

            double r[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = x(i) - (xa(i) + t * e[i]);
                u[i] = r[i] / delta;
            }

            double t_d[4][3];
            double r_d[4][3][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    const double alpha_pk = omega[pp] * e[k] + epsilon[pp] * w[k];
                    const double beta_pk  = 2.0 * epsilon[pp] * e[k];
                    t_d[pp][k] = alpha_pk / beta - alpha * beta_pk / (beta * beta);
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dpa = (pp == a_idx) ? 1.0 : 0.0;
                        const double dpx = (pp == 0) ? 1.0 : 0.0;
                        const double q_d = dpa * dik + t_d[pp][k] * e[i] + t * epsilon[pp] * dik;
                        r_d[pp][k][i] = dpx * dik - q_d;
                    }
                }
            }

            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const double dkl = (k == l) ? 1.0 : 0.0;

                    const double alpha_pk = omega[p] * e[k] + epsilon[p] * w[k];
                    const double alpha_ql = omega[p] * e[l] + epsilon[p] * w[l];
                    const double alpha_pkql = 2.0 * omega[p] * epsilon[p] * dkl;
                    const double beta_pk   = 2.0 * epsilon[p] * e[k];
                    const double beta_ql   = 2.0 * epsilon[p] * e[l];
                    const double beta_pkql = 2.0 * epsilon[p] * epsilon[p] * dkl;

                    const double t_pkql = alpha_pkql / beta
                                         - (alpha_pk * beta_ql + alpha_ql * beta_pk + alpha * beta_pkql) / (beta * beta)
                                         + 2.0 * alpha * beta_pk * beta_ql / (beta * beta * beta);

                    double ddelta_pk = 0.0, ddelta_ql = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        ddelta_pk += u[i] * r_d[p][k][i];
                        ddelta_ql += u[i] * r_d[p][l][i];
                    }

                    double proj_term = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            const double dij = (i == j) ? 1.0 : 0.0;
                            proj_term += (dij - u[i] * u[j]) * r_d[p][k][i] * r_d[p][l][j];
                        }
                    }
                    proj_term /= delta;

                    double uq_term = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dil = (i == l) ? 1.0 : 0.0;
                        const double q_ipkql = t_pkql * e[i]
                                             + t_d[p][k] * epsilon[p] * dil
                                             + t_d[p][l] * epsilon[p] * dik;
                        uq_term += u[i] * q_ipkql;
                    }

                    const double d2delta = proj_term - uq_term;
                    H(k, l) = bpp * ddelta_pk * ddelta_ql + bp * d2delta;
                }
            }
            break;
        }

        case NodeTriangleRegion::FaceInterior:
        {
            const double sig_a[4] = {0.0, -1.0, 1.0, 0.0};
            const double sig_b[4] = {0.0, -1.0, 0.0, 1.0};
            const double sig_w[4] = {1.0, -1.0, 0.0, 0.0};

            double a[3], b[3], w[3];
            for (int i = 0; i < 3; ++i) {
                a[i] = x2(i) - x1(i);
                b[i] = x3(i) - x1(i);
                w[i] = x(i)  - x1(i);
            }

            double N[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < 3; ++i) {
                for (int m = 0; m < 3; ++m) {
                    for (int n = 0; n < 3; ++n) {
                        N[i] += levi_civita(i, m, n) * a[m] * b[n];
                    }
                }
            }

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
                        for (int m = 0; m < 3; ++m)    val += sig_b[pp] * levi_civita(i, m, k) * a[m];
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

            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const double coeff_N2 = sig_a[p] * sig_b[p] - sig_a[p] * sig_b[p];

                    double nN2 = 0.0;
                    for (int i = 0; i < 3; ++i) nN2 += n[i] * coeff_N2 * levi_civita(i, k, l);

                    double proj_NN = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            const double dij = (i == j) ? 1.0 : 0.0;
                            proj_NN += (dij - n[i] * n[j]) * Nd[p][k][i] * Nd[p][l][j];
                        }
                    }

                    const double eta_pkql = nN2 + proj_NN / eta;

                    double psi_pkql = 0.0;
                    for (int i = 0; i < 3; ++i) psi_pkql += coeff_N2 * levi_civita(i, k, l) * w[i];
                    psi_pkql += sig_w[p] * Nd[p][k][l];
                    psi_pkql += sig_w[p] * Nd[p][l][k];

                    const double phi_pkql = psi_pkql / eta
                                          - (psi_d[p][k] * eta_d[p][l] + psi_d[p][l] * eta_d[p][k] + psi * eta_pkql) / (eta * eta)
                                          + 2.0 * psi * eta_d[p][k] * eta_d[p][l] / (eta * eta * eta);

                    H(k, l) = bpp * phi_d[p][k] * phi_d[p][l] + s_sign * bp * phi_pkql;
                }
            }
            break;
        }

        case NodeTriangleRegion::DegenerateTriangle:
            break;
    }

    return H;
}

// ====================================================================
//  Segment--segment barrier
// ====================================================================

double segment_segment_barrier(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
                               double d_hat, double eps){
    const auto dr = segment_segment_distance(x1, x2, x3, x4, eps);
    return scalar_barrier(dr.distance, d_hat);
}

// Single-DOF gradient for segment-segment barrier.
// dof: 0=x1, 1=x2, 2=x3, 3=x4
Vec3 segment_segment_barrier_gradient(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
                                      double d_hat, int dof, double eps){
    const SegmentSegmentDistanceResult dr = segment_segment_distance(x1, x2, x3, x4, eps);
    const double delta = dr.distance;
    const double bp    = scalar_barrier_gradient(delta, d_hat);

    Vec3 g = Vec3::Zero();
    if (bp == 0.0) return g;
    if (delta <= 0.0) throw std::runtime_error("segment_segment_barrier_gradient: distance must be positive.");

    const Vec3 r = dr.closest_point_1 - dr.closest_point_2;
    double u[3];
    for (int k = 0; k < 3; ++k) u[k] = r(k) / delta;

    const double s = dr.s;
    const double t = dr.t;

    // Per-DOF scalar coefficients on u[k]
    // mu[p] such that grad_p = mu[p] * u[k]
    double mu[4] = {0.0, 0.0, 0.0, 0.0};

    switch (dr.region) {

        case SegmentSegmentRegion::Interior:
            mu[0] = bp*(1.0-s); mu[1] = bp*s; mu[2] = -bp*(1.0-t); mu[3] = -bp*t;
            break;

        case SegmentSegmentRegion::Edge_s0:  // x1 vs (x3,x4)
            mu[0] =  bp; mu[1] = 0.0; mu[2] = -bp*(1.0-t); mu[3] = -bp*t;
            break;

        case SegmentSegmentRegion::Edge_s1:  // x2 vs (x3,x4)
            mu[0] = 0.0; mu[1] =  bp; mu[2] = -bp*(1.0-t); mu[3] = -bp*t;
            break;

        case SegmentSegmentRegion::Edge_t0:  // x3 vs (x1,x2)
            mu[0] =  bp*(1.0-s); mu[1] = bp*s; mu[2] = -bp; mu[3] = 0.0;
            break;

        case SegmentSegmentRegion::Edge_t1:  // x4 vs (x1,x2)
            mu[0] =  bp*(1.0-s); mu[1] = bp*s; mu[2] = 0.0; mu[3] = -bp;
            break;

        case SegmentSegmentRegion::Corner_s0t0:
            mu[0] =  bp; mu[1] = 0.0; mu[2] = -bp; mu[3] = 0.0;
            break;

        case SegmentSegmentRegion::Corner_s0t1:
            mu[0] =  bp; mu[1] = 0.0; mu[2] = 0.0; mu[3] = -bp;
            break;

        case SegmentSegmentRegion::Corner_s1t0:
            mu[0] = 0.0; mu[1] =  bp; mu[2] = -bp; mu[3] = 0.0;
            break;

        case SegmentSegmentRegion::Corner_s1t1:
            mu[0] = 0.0; mu[1] =  bp; mu[2] = 0.0; mu[3] = -bp;
            break;

        case SegmentSegmentRegion::ParallelSegments:
        {
            // Fallback: use mu weights from the resolved (s,t)
            const double muf[4] = {1.0-s, s, -(1.0-t), -t};
            for (int k = 0; k < 3; ++k) g(k) = bp * muf[dof] * u[k];
            return g;
        }
    }

    for (int k = 0; k < 3; ++k) g(k) = mu[dof] * u[k];
    return g;
}

// Single-DOF hessian row for segment-segment barrier.
// Returns 3x12 block d^2E / (d(y_dof) d(y_all)).
// dof: 0=x1, 1=x2, 2=x3, 3=x4
static Eigen::Matrix<double, 3, 12> segment_segment_barrier_hessian_row(const Vec3& x1, const Vec3& x2,
                                                                         const Vec3& x3, const Vec3& x4,
                                                                         double d_hat, int dof, double eps){
    Eigen::Matrix<double, 3, 12> H_row = Eigen::Matrix<double, 3, 12>::Zero();

    const SegmentSegmentDistanceResult dr = segment_segment_distance(x1, x2, x3, x4, eps);
    const double delta = dr.distance;
    const double bp    = scalar_barrier_gradient(delta, d_hat);
    const double bpp   = scalar_barrier_hessian(delta, d_hat);

    if (bp == 0.0 && bpp == 0.0) return H_row;
    if (delta <= 0.0) throw std::runtime_error("segment_segment_barrier_hessian: distance must be positive.");

    const Vec3* Y[4] = {&x1, &x2, &x3, &x4};
    const int p = dof;

    switch (dr.region) {

        // ---------------------------------------------------------------
        //  Vertex-vertex corners
        // ---------------------------------------------------------------
        case SegmentSegmentRegion::Corner_s0t0:
        case SegmentSegmentRegion::Corner_s0t1:
        case SegmentSegmentRegion::Corner_s1t0:
        case SegmentSegmentRegion::Corner_s1t1:
        {
            int a_idx, b_idx;
            if      (dr.region == SegmentSegmentRegion::Corner_s0t0) { a_idx = 0; b_idx = 2; }
            else if (dr.region == SegmentSegmentRegion::Corner_s0t1) { a_idx = 0; b_idx = 3; }
            else if (dr.region == SegmentSegmentRegion::Corner_s1t0) { a_idx = 1; b_idx = 2; }
            else                                                      { a_idx = 1; b_idx = 3; }

            double sp[4] = {0.0, 0.0, 0.0, 0.0};
            sp[a_idx] = 1.0; sp[b_idx] = -1.0;

            if (sp[p] == 0.0) break;

            double u[3];
            for (int k = 0; k < 3; ++k) u[k] = ((*Y[a_idx])(k) - (*Y[b_idx])(k)) / delta;

            const double c1 = bpp;
            const double c2 = bp / delta;

            for (int q = 0; q < 4; ++q) {
                if (sp[q] == 0.0) continue;
                const double sq = sp[p]*sp[q];
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l) {
                        double dkl = (k==l) ? 1.0 : 0.0;
                        H_row(k, 3*q+l) = sq*(c1*u[k]*u[l] + c2*(dkl - u[k]*u[l]));
                    }
            }
            break;
        }

            // ---------------------------------------------------------------
            //  Point-segment edges
            // ---------------------------------------------------------------
        case SegmentSegmentRegion::Edge_s0:
        case SegmentSegmentRegion::Edge_s1:
        case SegmentSegmentRegion::Edge_t0:
        case SegmentSegmentRegion::Edge_t1:
        {
            int query_idx, ea_idx, eb_idx;
            if      (dr.region == SegmentSegmentRegion::Edge_s0) { query_idx = 0; ea_idx = 2; eb_idx = 3; }
            else if (dr.region == SegmentSegmentRegion::Edge_s1) { query_idx = 1; ea_idx = 2; eb_idx = 3; }
            else if (dr.region == SegmentSegmentRegion::Edge_t0) { query_idx = 2; ea_idx = 0; eb_idx = 1; }
            else                                                  { query_idx = 3; ea_idx = 0; eb_idx = 1; }

            const Vec3& xq  = *Y[query_idx];
            const Vec3& xea = *Y[ea_idx];
            const Vec3& xeb = *Y[eb_idx];

            double omega[4]   = {0.0, 0.0, 0.0, 0.0};
            double epsilon[4] = {0.0, 0.0, 0.0, 0.0};
            omega[query_idx] = 1.0; omega[ea_idx] = -1.0;
            epsilon[ea_idx] = -1.0; epsilon[eb_idx] = 1.0;

            double e[3], w[3];
            for (int i = 0; i < 3; ++i) { e[i] = xeb(i)-xea(i); w[i] = xq(i)-xea(i); }

            double alpha = 0.0, beta = 0.0;
            for (int i = 0; i < 3; ++i) { alpha += w[i]*e[i]; beta += e[i]*e[i]; }
            double t_param = alpha/beta;

            double r[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = xq(i) - (xea(i) + t_param*e[i]);
                u[i] = r[i]/delta;
            }

            double t_d[4][3];
            double r_d[4][3][3];
            double q_d[4][3][3];

            for (int pp = 0; pp < 4; ++pp)
                for (int k = 0; k < 3; ++k) {
                    double alpha_pk = omega[pp]*e[k] + epsilon[pp]*w[k];
                    double beta_pk  = 2.0*epsilon[pp]*e[k];
                    t_d[pp][k] = alpha_pk/beta - alpha*beta_pk/(beta*beta);
                    for (int i = 0; i < 3; ++i) {
                        double dik   = (i==k) ? 1.0 : 0.0;
                        double dp_ea = (pp==ea_idx) ? 1.0 : 0.0;
                        double dp_q  = (pp==query_idx) ? 1.0 : 0.0;
                        q_d[pp][k][i] = dp_ea*dik + t_d[pp][k]*e[i] + t_param*epsilon[pp]*dik;
                        r_d[pp][k][i] = dp_q*dik - q_d[pp][k][i];
                    }
                }

            for (int q = 0; q < 4; ++q)
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l) {
                        double dkl = (k==l) ? 1.0 : 0.0;

                        double alpha_pk   = omega[p]*e[k]  + epsilon[p]*w[k];
                        double alpha_ql   = omega[q]*e[l]  + epsilon[q]*w[l];
                        double alpha_pkql = (omega[p]*epsilon[q] + omega[q]*epsilon[p])*dkl;
                        double beta_pk    = 2.0*epsilon[p]*e[k];
                        double beta_ql    = 2.0*epsilon[q]*e[l];
                        double beta_pkql  = 2.0*epsilon[p]*epsilon[q]*dkl;

                        double t_pkql = alpha_pkql/beta
                                        - (alpha_pk*beta_ql + alpha_ql*beta_pk + alpha*beta_pkql)/(beta*beta)
                                        + 2.0*alpha*beta_pk*beta_ql/(beta*beta*beta);

                        double ddelta_pk = 0.0, ddelta_ql = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            ddelta_pk += u[i]*r_d[p][k][i];
                            ddelta_ql += u[i]*r_d[q][l][i];
                        }

                        double proj_term = 0.0;
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j) {
                                double dij = (i==j) ? 1.0 : 0.0;
                                proj_term += (dij-u[i]*u[j])*r_d[p][k][i]*r_d[q][l][j];
                            }
                        proj_term /= delta;

                        double uq_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            double dik = (i==k) ? 1.0 : 0.0;
                            double dil = (i==l) ? 1.0 : 0.0;
                            double q_ipkql = t_pkql*e[i]
                                             + t_d[p][k]*epsilon[q]*dil
                                             + t_d[q][l]*epsilon[p]*dik;
                            uq_term += u[i]*q_ipkql;
                        }

                        H_row(k, 3*q+l) = bpp*ddelta_pk*ddelta_ql + bp*(proj_term - uq_term);
                    }
            break;
        }

            // ---------------------------------------------------------------
            //  Interior
            // ---------------------------------------------------------------
        case SegmentSegmentRegion::Interior:
        {
            double sig_a[4] = {-1.0,  1.0,  0.0,  0.0};
            double sig_b[4] = { 0.0,  0.0, -1.0,  1.0};
            double sig_c[4] = { 1.0,  0.0, -1.0,  0.0};

            double a[3], b[3], c[3];
            for (int i = 0; i < 3; ++i) {
                a[i] = x2(i)-x1(i); b[i] = x4(i)-x3(i); c[i] = x1(i)-x3(i);
            }

            double A=0, B=0, C=0, D=0, E=0;
            for (int i = 0; i < 3; ++i) {
                A += a[i]*a[i]; B += a[i]*b[i]; C += b[i]*b[i];
                D += a[i]*c[i]; E += b[i]*c[i];
            }

            double Delta = A*C - B*B;
            double nu    = B*E - C*D;
            double zeta  = A*E - B*D;
            double s_val = nu/Delta;
            double t_val = zeta/Delta;

            double Ad[4][3], Bd[4][3], Cd[4][3], Dd[4][3], Ed[4][3];
            for (int pp = 0; pp < 4; ++pp)
                for (int k = 0; k < 3; ++k) {
                    Ad[pp][k] = 2.0*sig_a[pp]*a[k];
                    Bd[pp][k] = sig_a[pp]*b[k] + sig_b[pp]*a[k];
                    Cd[pp][k] = 2.0*sig_b[pp]*b[k];
                    Dd[pp][k] = sig_a[pp]*c[k] + sig_c[pp]*a[k];
                    Ed[pp][k] = sig_b[pp]*c[k] + sig_c[pp]*b[k];
                }

            double nu_d[4][3], zeta_d[4][3], Delta_d[4][3];
            for (int pp = 0; pp < 4; ++pp)
                for (int k = 0; k < 3; ++k) {
                    nu_d[pp][k]    = Bd[pp][k]*E + B*Ed[pp][k] - Cd[pp][k]*D - C*Dd[pp][k];
                    zeta_d[pp][k]  = Ad[pp][k]*E + A*Ed[pp][k] - Bd[pp][k]*D - B*Dd[pp][k];
                    Delta_d[pp][k] = Ad[pp][k]*C + A*Cd[pp][k] - 2.0*B*Bd[pp][k];
                }

            double s_d[4][3], t_d[4][3];
            for (int pp = 0; pp < 4; ++pp)
                for (int k = 0; k < 3; ++k) {
                    s_d[pp][k] = nu_d[pp][k]/Delta   - nu*Delta_d[pp][k]/(Delta*Delta);
                    t_d[pp][k] = zeta_d[pp][k]/Delta - zeta*Delta_d[pp][k]/(Delta*Delta);
                }

            double r_vec[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r_vec[i] = (x1(i)+s_val*a[i]) - (x3(i)+t_val*b[i]);
                u[i] = r_vec[i]/delta;
            }

            double p_d[4][3][3], q_d_arr[4][3][3], r_d[4][3][3];
            for (int pp = 0; pp < 4; ++pp)
                for (int k = 0; k < 3; ++k)
                    for (int i = 0; i < 3; ++i) {
                        double dik  = (i==k) ? 1.0 : 0.0;
                        double dp0  = (pp==0) ? 1.0 : 0.0;
                        double dp2  = (pp==2) ? 1.0 : 0.0;
                        p_d[pp][k][i]     = dp0*dik + s_d[pp][k]*a[i] + s_val*sig_a[pp]*dik;
                        q_d_arr[pp][k][i] = dp2*dik + t_d[pp][k]*b[i] + t_val*sig_b[pp]*dik;
                        r_d[pp][k][i]     = p_d[pp][k][i] - q_d_arr[pp][k][i];
                    }

            for (int q = 0; q < 4; ++q)
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l) {
                        double dkl = (k==l) ? 1.0 : 0.0;

                        double A_pkql = 2.0*sig_a[p]*sig_a[q]*dkl;
                        double B_pkql = (sig_a[p]*sig_b[q] + sig_a[q]*sig_b[p])*dkl;
                        double C_pkql = 2.0*sig_b[p]*sig_b[q]*dkl;
                        double D_pkql = (sig_a[p]*sig_c[q] + sig_a[q]*sig_c[p])*dkl;
                        double E_pkql = (sig_b[p]*sig_c[q] + sig_b[q]*sig_c[p])*dkl;

                        double nu_pkql = B_pkql*E + Bd[p][k]*Ed[q][l] + Bd[q][l]*Ed[p][k] + B*E_pkql
                                         - C_pkql*D - Cd[p][k]*Dd[q][l] - Cd[q][l]*Dd[p][k] - C*D_pkql;
                        double Delta_pkql = A_pkql*C + Ad[p][k]*Cd[q][l] + Ad[q][l]*Cd[p][k] + A*C_pkql
                                            - 2.0*(Bd[p][k]*Bd[q][l] + B*B_pkql);
                        double zeta_pkql = A_pkql*E + Ad[p][k]*Ed[q][l] + Ad[q][l]*Ed[p][k] + A*E_pkql
                                           - B_pkql*D - Bd[p][k]*Dd[q][l] - Bd[q][l]*Dd[p][k] - B*D_pkql;

                        double s_pkql = nu_pkql/Delta
                                        - (nu_d[p][k]*Delta_d[q][l] + nu_d[q][l]*Delta_d[p][k]
                                           + nu*Delta_pkql)/(Delta*Delta)
                                        + 2.0*nu*Delta_d[p][k]*Delta_d[q][l]/(Delta*Delta*Delta);
                        double t_pkql = zeta_pkql/Delta
                                        - (zeta_d[p][k]*Delta_d[q][l] + zeta_d[q][l]*Delta_d[p][k]
                                           + zeta*Delta_pkql)/(Delta*Delta)
                                        + 2.0*zeta*Delta_d[p][k]*Delta_d[q][l]/(Delta*Delta*Delta);

                        double ddelta_pk = 0.0, ddelta_ql = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            ddelta_pk += u[i]*r_d[p][k][i];
                            ddelta_ql += u[i]*r_d[q][l][i];
                        }

                        double proj_term = 0.0;
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j) {
                                double dij = (i==j) ? 1.0 : 0.0;
                                proj_term += (dij-u[i]*u[j])*r_d[p][k][i]*r_d[q][l][j];
                            }
                        proj_term /= delta;

                        double ur_term = 0.0;
                        for (int i = 0; i < 3; ++i) {
                            double dik = (i==k) ? 1.0 : 0.0;
                            double dil = (i==l) ? 1.0 : 0.0;
                            double p_ipkql = s_pkql*a[i] + s_d[p][k]*sig_a[q]*dil + s_d[q][l]*sig_a[p]*dik;
                            double q_ipkql = t_pkql*b[i] + t_d[p][k]*sig_b[q]*dil + t_d[q][l]*sig_b[p]*dik;
                            ur_term += u[i]*(p_ipkql - q_ipkql);
                        }

                        H_row(k, 3*q+l) = bpp*ddelta_pk*ddelta_ql + bp*(proj_term + ur_term);
                    }
            break;
        }

        case SegmentSegmentRegion::ParallelSegments:
            break;  // leave as zero
    }

    return H_row;
}

Mat33 segment_segment_barrier_hessian(const Vec3& x1, const Vec3& x2,
                                      const Vec3& x3, const Vec3& x4,
                                      double d_hat, int dof, double eps){
    Mat33 H = Mat33::Zero();

    const SegmentSegmentDistanceResult dr = segment_segment_distance(x1, x2, x3, x4, eps);
    const double delta = dr.distance;
    const double bp    = scalar_barrier_gradient(delta, d_hat);
    const double bpp   = scalar_barrier_hessian(delta, d_hat);

    if (bp == 0.0 && bpp == 0.0) return H;
    if (delta <= 0.0) throw std::runtime_error("segment_segment_barrier_hessian: distance must be positive.");

    const Vec3* Y[4] = {&x1, &x2, &x3, &x4};
    const int p = dof;

    switch (dr.region) {
        case SegmentSegmentRegion::Corner_s0t0:
        case SegmentSegmentRegion::Corner_s0t1:
        case SegmentSegmentRegion::Corner_s1t0:
        case SegmentSegmentRegion::Corner_s1t1:
        {
            int a_idx, b_idx;
            if      (dr.region == SegmentSegmentRegion::Corner_s0t0) { a_idx = 0; b_idx = 2; }
            else if (dr.region == SegmentSegmentRegion::Corner_s0t1) { a_idx = 0; b_idx = 3; }
            else if (dr.region == SegmentSegmentRegion::Corner_s1t0) { a_idx = 1; b_idx = 2; }
            else                                                      { a_idx = 1; b_idx = 3; }

            double sp[4] = {0.0, 0.0, 0.0, 0.0};
            sp[a_idx] = 1.0;
            sp[b_idx] = -1.0;
            if (sp[p] == 0.0) break;

            double u[3];
            for (int k = 0; k < 3; ++k) u[k] = ((*Y[a_idx])(k) - (*Y[b_idx])(k)) / delta;

            const double c1 = bpp;
            const double c2 = bp / delta;
            const double sq = sp[p] * sp[p];
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const double dkl = (k == l) ? 1.0 : 0.0;
                    H(k, l) = sq * (c1 * u[k] * u[l] + c2 * (dkl - u[k] * u[l]));
                }
            }
            break;
        }

        case SegmentSegmentRegion::Edge_s0:
        case SegmentSegmentRegion::Edge_s1:
        case SegmentSegmentRegion::Edge_t0:
        case SegmentSegmentRegion::Edge_t1:
        {
            int query_idx, ea_idx, eb_idx;
            if      (dr.region == SegmentSegmentRegion::Edge_s0) { query_idx = 0; ea_idx = 2; eb_idx = 3; }
            else if (dr.region == SegmentSegmentRegion::Edge_s1) { query_idx = 1; ea_idx = 2; eb_idx = 3; }
            else if (dr.region == SegmentSegmentRegion::Edge_t0) { query_idx = 2; ea_idx = 0; eb_idx = 1; }
            else                                                  { query_idx = 3; ea_idx = 0; eb_idx = 1; }

            const Vec3& xq  = *Y[query_idx];
            const Vec3& xea = *Y[ea_idx];
            const Vec3& xeb = *Y[eb_idx];

            double omega[4]   = {0.0, 0.0, 0.0, 0.0};
            double epsilon[4] = {0.0, 0.0, 0.0, 0.0};
            omega[query_idx] = 1.0; omega[ea_idx] = -1.0;
            epsilon[ea_idx] = -1.0; epsilon[eb_idx] = 1.0;

            double e[3], w[3];
            for (int i = 0; i < 3; ++i) { e[i] = xeb(i) - xea(i); w[i] = xq(i) - xea(i); }

            double alpha = 0.0, beta = 0.0;
            for (int i = 0; i < 3; ++i) { alpha += w[i] * e[i]; beta += e[i] * e[i]; }
            const double t_param = alpha / beta;

            double r[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = xq(i) - (xea(i) + t_param * e[i]);
                u[i] = r[i] / delta;
            }

            double t_d[4][3];
            double r_d[4][3][3];
            for (int pp = 0; pp < 4; ++pp) {
                for (int k = 0; k < 3; ++k) {
                    const double alpha_pk = omega[pp] * e[k] + epsilon[pp] * w[k];
                    const double beta_pk  = 2.0 * epsilon[pp] * e[k];
                    t_d[pp][k] = alpha_pk / beta - alpha * beta_pk / (beta * beta);
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dp_ea = (pp == ea_idx) ? 1.0 : 0.0;
                        const double dp_q  = (pp == query_idx) ? 1.0 : 0.0;
                        const double q_d = dp_ea * dik + t_d[pp][k] * e[i] + t_param * epsilon[pp] * dik;
                        r_d[pp][k][i] = dp_q * dik - q_d;
                    }
                }
            }

            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const double dkl = (k == l) ? 1.0 : 0.0;

                    const double alpha_pk   = omega[p] * e[k] + epsilon[p] * w[k];
                    const double alpha_ql   = omega[p] * e[l] + epsilon[p] * w[l];
                    const double alpha_pkql = 2.0 * omega[p] * epsilon[p] * dkl;
                    const double beta_pk    = 2.0 * epsilon[p] * e[k];
                    const double beta_ql    = 2.0 * epsilon[p] * e[l];
                    const double beta_pkql  = 2.0 * epsilon[p] * epsilon[p] * dkl;

                    const double t_pkql = alpha_pkql / beta
                                        - (alpha_pk * beta_ql + alpha_ql * beta_pk + alpha * beta_pkql) / (beta * beta)
                                        + 2.0 * alpha * beta_pk * beta_ql / (beta * beta * beta);

                    double ddelta_pk = 0.0, ddelta_ql = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        ddelta_pk += u[i] * r_d[p][k][i];
                        ddelta_ql += u[i] * r_d[p][l][i];
                    }

                    double proj_term = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            const double dij = (i == j) ? 1.0 : 0.0;
                            proj_term += (dij - u[i] * u[j]) * r_d[p][k][i] * r_d[p][l][j];
                        }
                    }
                    proj_term /= delta;

                    double uq_term = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dil = (i == l) ? 1.0 : 0.0;
                        const double q_ipkql = t_pkql * e[i]
                                             + t_d[p][k] * epsilon[p] * dil
                                             + t_d[p][l] * epsilon[p] * dik;
                        uq_term += u[i] * q_ipkql;
                    }

                    H(k, l) = bpp * ddelta_pk * ddelta_ql + bp * (proj_term - uq_term);
                }
            }
            break;
        }

        case SegmentSegmentRegion::Interior:
        {
            const double sig_a[4] = {-1.0,  1.0,  0.0,  0.0};
            const double sig_b[4] = { 0.0,  0.0, -1.0,  1.0};
            const double sig_c[4] = { 1.0,  0.0, -1.0,  0.0};

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

            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const double dkl = (k == l) ? 1.0 : 0.0;

                    const double A_pkql = 2.0 * sig_a[p] * sig_a[p] * dkl;
                    const double B_pkql = (sig_a[p] * sig_b[p] + sig_a[p] * sig_b[p]) * dkl;
                    const double C_pkql = 2.0 * sig_b[p] * sig_b[p] * dkl;
                    const double D_pkql = (sig_a[p] * sig_c[p] + sig_a[p] * sig_c[p]) * dkl;
                    const double E_pkql = (sig_b[p] * sig_c[p] + sig_b[p] * sig_c[p]) * dkl;

                    const double nu_pkql = B_pkql * E + Bd[p][k] * Ed[p][l] + Bd[p][l] * Ed[p][k] + B * E_pkql
                                         - C_pkql * D - Cd[p][k] * Dd[p][l] - Cd[p][l] * Dd[p][k] - C * D_pkql;
                    const double Delta_pkql = A_pkql * C + Ad[p][k] * Cd[p][l] + Ad[p][l] * Cd[p][k] + A * C_pkql
                                            - 2.0 * (Bd[p][k] * Bd[p][l] + B * B_pkql);
                    const double zeta_pkql = A_pkql * E + Ad[p][k] * Ed[p][l] + Ad[p][l] * Ed[p][k] + A * E_pkql
                                           - B_pkql * D - Bd[p][k] * Dd[p][l] - Bd[p][l] * Dd[p][k] - B * D_pkql;

                    const double s_pkql = nu_pkql / Delta
                                        - (nu_d[p][k] * Delta_d[p][l] + nu_d[p][l] * Delta_d[p][k] + nu * Delta_pkql) / (Delta * Delta)
                                        + 2.0 * nu * Delta_d[p][k] * Delta_d[p][l] / (Delta * Delta * Delta);
                    const double t_pkql = zeta_pkql / Delta
                                        - (zeta_d[p][k] * Delta_d[p][l] + zeta_d[p][l] * Delta_d[p][k] + zeta * Delta_pkql) / (Delta * Delta)
                                        + 2.0 * zeta * Delta_d[p][k] * Delta_d[p][l] / (Delta * Delta * Delta);

                    double ddelta_pk = 0.0, ddelta_ql = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        ddelta_pk += u[i] * r_d[p][k][i];
                        ddelta_ql += u[i] * r_d[p][l][i];
                    }

                    double proj_term = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            const double dij = (i == j) ? 1.0 : 0.0;
                            proj_term += (dij - u[i] * u[j]) * r_d[p][k][i] * r_d[p][l][j];
                        }
                    }
                    proj_term /= delta;

                    double ur_term = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        const double dik = (i == k) ? 1.0 : 0.0;
                        const double dil = (i == l) ? 1.0 : 0.0;
                        const double p_ipkql = s_pkql * a[i] + s_d[p][k] * sig_a[p] * dil + s_d[p][l] * sig_a[p] * dik;
                        const double q_ipkql = t_pkql * b[i] + t_d[p][k] * sig_b[p] * dil + t_d[p][l] * sig_b[p] * dik;
                        ur_term += u[i] * (p_ipkql - q_ipkql);
                    }

                    H(k, l) = bpp * ddelta_pk * ddelta_ql + bp * (proj_term + ur_term);
                }
            }
            break;
        }

        case SegmentSegmentRegion::ParallelSegments:
            break;
    }

    return H;
}

std::pair<Vec3, Mat33> node_triangle_barrier_gradient_and_hessian(
        const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3,
        double d_hat, int dof, double eps) {
    const Vec3 g = node_triangle_barrier_gradient(x, x1, x2, x3, d_hat, dof, eps);
    const Mat33 H = node_triangle_barrier_hessian(x, x1, x2, x3, d_hat, dof, eps);
    return {g, H};
}

std::pair<Vec3, Mat33> segment_segment_barrier_gradient_and_hessian(
        const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4,
        double d_hat, int dof, double eps) {
    const Vec3 g = segment_segment_barrier_gradient(x1, x2, x3, x4, d_hat, dof, eps);
    const Mat33 H = segment_segment_barrier_hessian(x1, x2, x3, x4, d_hat, dof, eps);
    return {g, H};
}
