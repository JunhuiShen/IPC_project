#include "barrier_energy.h"

#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
//  Helper: recover edge parameter t from closest point q on segment [a, b]
// ---------------------------------------------------------------------------

double segment_parameter_from_closest_point(const Vec3& q, const Vec3& a, const Vec3& b){
    double denom = 0.0;
    double numer = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double ab_k = b(k) - a(k);
        denom += ab_k * ab_k;
        numer += (q(k) - a(k)) * ab_k;
    }

    if (denom <= 0.0) {
        return 0.0;
    }

    return clamp_scalar(numer / denom, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
//  Levi-Civita helper:  eps_{ijk}
// ---------------------------------------------------------------------------

double levi_civita(int i, int j, int k){
    // eps_{ijk} = (i-j)(j-k)(k-i) / 2  for i,j,k in {0,1,2}
    return 0.5 * (i - j) * (j - k) * (k - i);
}

// ---------------------------------------------------------------------------
//  Scalar barrier
// ---------------------------------------------------------------------------

double scalar_barrier(double delta, double d_hat){
    if (d_hat <= 0.0) throw std::runtime_error("scalar_barrier: d_hat must be positive.");
    if (delta >= d_hat) return 0.0;
    if (delta <= 0.0) throw std::runtime_error("scalar_barrier: delta must be positive.");

    const double s = delta - d_hat;
    return -(s * s) * std::log(delta / d_hat);
}

// ---------------------------------------------------------------------------
//  Scalar barrier gradient  db/d(delta)
// ---------------------------------------------------------------------------

double scalar_barrier_gradient(double delta, double d_hat){
    if (d_hat <= 0.0) throw std::runtime_error("scalar_barrier_gradient: d_hat must be positive.");
    if (delta >= d_hat) return 0.0;
    if (delta <= 0.0) throw std::runtime_error("scalar_barrier_gradient: delta must be positive.");

    const double s = delta - d_hat;
    return -2.0 * s * std::log(delta / d_hat) - (s * s) / delta;
}

// ---------------------------------------------------------------------------
//  Scalar barrier hessian  d2b/d(delta)2
// ---------------------------------------------------------------------------

double scalar_barrier_hessian(double delta, double d_hat){
    if (d_hat <= 0.0) throw std::runtime_error("scalar_barrier_hessian: d_hat must be positive.");
    if (delta >= d_hat) return 0.0;
    if (delta <= 0.0) throw std::runtime_error("scalar_barrier_hessian: delta must be positive.");

    // b''(delta; d_hat) = d_hat^2/delta^2 + 2*d_hat/delta - 3 - 2*ln(delta/d_hat)
    const double ratio = d_hat / delta;
    return ratio * ratio + 2.0 * ratio - 3.0 - 2.0 * std::log(delta / d_hat);
}

// ---------------------------------------------------------------------------
//  Barrier energy with node-triangle distance
// ---------------------------------------------------------------------------

double node_triangle_barrier(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps){
    const NodeTriangleDistanceResult dr = node_triangle_distance(x, x1, x2, x3, eps);
    return scalar_barrier(dr.distance, d_hat);
}

// ---------------------------------------------------------------------------
//  Barrier energy gradient with node-triangle distance (index notation)
// ---------------------------------------------------------------------------

NodeTriangleBarrierResult node_triangle_barrier_gradient(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps){
    NodeTriangleBarrierResult out;
    out.distance_result = node_triangle_distance(x, x1, x2, x3, eps);
    out.distance = out.distance_result.distance;
    out.energy = scalar_barrier(out.distance, d_hat);
    out.barrier_derivative = scalar_barrier_gradient(out.distance, d_hat);

    if (out.barrier_derivative == 0.0) return out;

    const NodeTriangleDistanceResult& dr = out.distance_result;
    const double delta = dr.distance;
    const double bp = out.barrier_derivative;

    if (delta <= 0.0) throw std::runtime_error("node_triangle_barrier_gradient: distance must be positive.");

    double u[3];
    for (int k = 0; k < 3; ++k) u[k] = (x(k) - dr.closest_point(k)) / delta;

    switch (dr.region) {

        case NodeTriangleRegion::FaceInterior:
        {
            const double phi = dr.phi;
            double sphi;
            if      (phi > 0.0) sphi =  1.0;
            else if (phi < 0.0) sphi = -1.0;
            else                sphi =  0.0;

            const double l1 = dr.barycentric_tilde_x[0];
            const double l2 = dr.barycentric_tilde_x[1];
            const double l3 = dr.barycentric_tilde_x[2];

            for (int k = 0; k < 3; ++k) {
                const double n_k = dr.normal(k);
                out.grad_x(k)  =  bp * sphi * n_k;
                out.grad_x1(k) = -bp * sphi * l1 * n_k;
                out.grad_x2(k) = -bp * sphi * l2 * n_k;
                out.grad_x3(k) = -bp * sphi * l3 * n_k;
            }
            break;
        }

        case NodeTriangleRegion::Edge12:
        {
            const double t = segment_parameter_from_closest_point(dr.closest_point, x1, x2);
            for (int k = 0; k < 3; ++k) {
                out.grad_x(k)  =  bp * u[k];
                out.grad_x1(k) = -bp * (1.0 - t) * u[k];
                out.grad_x2(k) = -bp * t * u[k];
                out.grad_x3(k) =  0.0;
            }
            break;
        }

        case NodeTriangleRegion::Edge23:
        {
            const double t = segment_parameter_from_closest_point(dr.closest_point, x2, x3);
            for (int k = 0; k < 3; ++k) {
                out.grad_x(k)  =  bp * u[k];
                out.grad_x1(k) =  0.0;
                out.grad_x2(k) = -bp * (1.0 - t) * u[k];
                out.grad_x3(k) = -bp * t * u[k];
            }
            break;
        }

        case NodeTriangleRegion::Edge31:
        {
            const double t = segment_parameter_from_closest_point(dr.closest_point, x3, x1);
            for (int k = 0; k < 3; ++k) {
                out.grad_x(k)  =  bp * u[k];
                out.grad_x1(k) = -bp * t * u[k];
                out.grad_x2(k) =  0.0;
                out.grad_x3(k) = -bp * (1.0 - t) * u[k];
            }
            break;
        }

        case NodeTriangleRegion::Vertex1:
        {
            for (int k = 0; k < 3; ++k) {
                out.grad_x(k) = bp * u[k]; out.grad_x1(k) = -bp * u[k];
                out.grad_x2(k) = 0.0;      out.grad_x3(k) = 0.0;
            }
            break;
        }

        case NodeTriangleRegion::Vertex2:
        {
            for (int k = 0; k < 3; ++k) {
                out.grad_x(k) = bp * u[k]; out.grad_x1(k) = 0.0;
                out.grad_x2(k) = -bp * u[k]; out.grad_x3(k) = 0.0;
            }
            break;
        }

        case NodeTriangleRegion::Vertex3:
        {
            for (int k = 0; k < 3; ++k) {
                out.grad_x(k) = bp * u[k]; out.grad_x1(k) = 0.0;
                out.grad_x2(k) = 0.0; out.grad_x3(k) = -bp * u[k];
            }
            break;
        }

        case NodeTriangleRegion::DegenerateTriangle:
        {
            for (int k = 0; k < 3; ++k) out.grad_x(k) = bp * u[k];
            // simplified fallback: assign to closest vertex
            double d1 = 0.0, d2 = 0.0, d3 = 0.0;
            for (int k = 0; k < 3; ++k) {
                double v1 = dr.closest_point(k) - x1(k); d1 += v1*v1;
                double v2 = dr.closest_point(k) - x2(k); d2 += v2*v2;
                double v3 = dr.closest_point(k) - x3(k); d3 += v3*v3;
            }
            if (std::sqrt(d1) <= std::sqrt(d2) && std::sqrt(d1) <= std::sqrt(d3))
                for (int k = 0; k < 3; ++k) out.grad_x1(k) = -bp * u[k];
            else if (std::sqrt(d2) <= std::sqrt(d3))
                for (int k = 0; k < 3; ++k) out.grad_x2(k) = -bp * u[k];
            else
                for (int k = 0; k < 3; ++k) out.grad_x3(k) = -bp * u[k];
            break;
        }
    }

    return out;
}

// ===========================================================================
//  Barrier energy hessian with node-triangle distance (index notation)
//
//  H_{pk,ql} = b''(delta) * (d delta/dy_{pk}) * (d delta/dy_{ql})
//            + b'(delta)  * (d^2 delta / dy_{pk} dy_{ql})
//
//  DOF ordering: p = 0..3 for (x, x1, x2, x3), k,l = 0..2 for spatial.
//  Row/col index in 12x12:  3*p + k.
// ===========================================================================

NodeTriangleBarrierHessianResult node_triangle_barrier_hessian(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps){
    NodeTriangleBarrierHessianResult out;

    // --- energy and gradient via existing function ---
    const auto gr = node_triangle_barrier_gradient(x, x1, x2, x3, d_hat, eps);
    out.distance_result = gr.distance_result;
    out.distance = gr.distance;
    out.energy = gr.energy;
    out.grad_x  = gr.grad_x;
    out.grad_x1 = gr.grad_x1;
    out.grad_x2 = gr.grad_x2;
    out.grad_x3 = gr.grad_x3;

    const double delta = gr.distance;
    const double bp  = scalar_barrier_gradient(delta, d_hat);
    const double bpp = scalar_barrier_hessian(delta, d_hat);

    if (bp == 0.0 && bpp == 0.0) return out;

    const NodeTriangleDistanceResult& dr = gr.distance_result;
    if (delta <= 0.0) throw std::runtime_error("node_triangle_barrier_hessian: distance must be positive.");

    // ---------------------------------------------------------------
    //  Node positions packed into an array for index access:
    //    Y[0] = x,  Y[1] = x1,  Y[2] = x2,  Y[3] = x3
    // ---------------------------------------------------------------
    const Vec3* Y[4] = {&x, &x1, &x2, &x3};

    switch (dr.region) {

        // ===============================================================
        //  Cases 5-7: vertex
        //
        //  q = x_a,  r = x - x_a,  delta = ||r||,  u_i = r_i / delta
        //
        //  s_p = { +1 if p == x,  -1 if p == a,  0 otherwise }
        //
        //  H_{pk,ql} = s_p * s_q * [ b'' * u_k * u_l
        //                           + (b'/delta) * (delta_{kl} - u_k * u_l) ]
        // ===============================================================
        case NodeTriangleRegion::Vertex1:
        case NodeTriangleRegion::Vertex2:
        case NodeTriangleRegion::Vertex3:
        {
            // Determine which vertex a is
            int a_idx; // index in Y: 1=x1, 2=x2, 3=x3
            if      (dr.region == NodeTriangleRegion::Vertex1) a_idx = 1;
            else if (dr.region == NodeTriangleRegion::Vertex2) a_idx = 2;
            else                                               a_idx = 3;

            // s_p for each of the 4 node groups
            double sp[4] = {0.0, 0.0, 0.0, 0.0};
            sp[0] = 1.0;         // p = x
            sp[a_idx] = -1.0;    // p = a

            // u_k
            double u[3];
            for (int k = 0; k < 3; ++k)
                u[k] = (x(k) - (*Y[a_idx])(k)) / delta;

            // Fill 12x12 Hessian
            const double c1 = bpp;
            const double c2 = bp / delta;

            for (int p = 0; p < 4; ++p) {
                for (int q = 0; q < 4; ++q) {
                    const double sq = sp[p] * sp[q];
                    if (sq == 0.0) continue;
                    for (int k = 0; k < 3; ++k) {
                        for (int l = 0; l < 3; ++l) {
                            double dkl = (k == l) ? 1.0 : 0.0;
                            out.hessian(3*p+k, 3*q+l) =
                                    sq * (c1 * u[k] * u[l] + c2 * (dkl - u[k] * u[l]));
                        }
                    }
                }
            }
            break;
        }

            // ===============================================================
            //  Cases 2-4: edge interior
            //
            //  Edge (x_a, x_b), inactive vertex x_c.
            //  e_i = (x_b)_i - (x_a)_i
            //  w_i = x_i - (x_a)_i
            //  alpha = w_i e_i,  beta = e_i e_i,  t = alpha / beta
            //  q_i = (x_a)_i + t * e_i
            //  r_i = x_i - q_i,  delta = ||r||,  u_i = r_i / delta
            //
            //  omega_p: dw_i/dy_{pk} = omega_p * delta_{ik}
            //    omega = { +1 for x, -1 for a, 0 for b,c }
            //
            //  epsilon_p: de_i/dy_{pk} = epsilon_p * delta_{ik}
            //    epsilon = { -1 for a, +1 for b, 0 for x,c }
            // ===============================================================
        case NodeTriangleRegion::Edge12:
        case NodeTriangleRegion::Edge23:
        case NodeTriangleRegion::Edge31:
        {
            int a_idx, b_idx, c_idx;
            if      (dr.region == NodeTriangleRegion::Edge12) { a_idx = 1; b_idx = 2; c_idx = 3; }
            else if (dr.region == NodeTriangleRegion::Edge23) { a_idx = 2; b_idx = 3; c_idx = 1; }
            else                                              { a_idx = 3; b_idx = 1; c_idx = 2; }

            // omega_p and epsilon_p for p = 0(x), 1(x1), 2(x2), 3(x3)
            double omega[4]   = {0.0, 0.0, 0.0, 0.0};
            double epsilon[4] = {0.0, 0.0, 0.0, 0.0};
            omega[0] = 1.0;     omega[a_idx] = -1.0;
            epsilon[a_idx] = -1.0;  epsilon[b_idx] = 1.0;

            const Vec3& xa = *Y[a_idx];
            const Vec3& xb = *Y[b_idx];

            // e_i, w_i
            double e[3], w[3];
            for (int i = 0; i < 3; ++i) {
                e[i] = xb(i) - xa(i);
                w[i] = x(i)  - xa(i);
            }

            // alpha = w_i e_i, beta = e_i e_i
            double alpha = 0.0, beta = 0.0;
            for (int i = 0; i < 3; ++i) { alpha += w[i]*e[i]; beta += e[i]*e[i]; }

            double t = alpha / beta;

            // r_i, u_i
            double r[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = x(i) - (xa(i) + t * e[i]);
                u[i] = r[i] / delta;
            }

            // --- first derivatives ---
            // alpha_{,pk} = omega_p * e_k + epsilon_p * w_k
            // beta_{,pk}  = 2 * epsilon_p * e_k
            // t_{,pk}     = alpha_{,pk}/beta - alpha*beta_{,pk}/beta^2

            // q_{i,pk} = delta_{p,a}*delta_{ik} + t_{,pk}*e_i + t*epsilon_p*delta_{ik}
            // r_{i,pk} = delta_{p,x}*delta_{ik} - q_{i,pk}

            // Precompute t_{,pk} and r_{i,pk} for all (p,k)
            double t_d[4][3];       // t_{,pk}
            double r_d[4][3][3];    // r_{i,pk}
            double q_d[4][3][3];    // q_{i,pk}

            for (int p = 0; p < 4; ++p) {
                for (int k = 0; k < 3; ++k) {
                    double alpha_pk = omega[p] * e[k] + epsilon[p] * w[k];
                    double beta_pk  = 2.0 * epsilon[p] * e[k];
                    t_d[p][k] = alpha_pk / beta - alpha * beta_pk / (beta * beta);

                    for (int i = 0; i < 3; ++i) {
                        double dik = (i == k) ? 1.0 : 0.0;
                        double dpa = (p == a_idx) ? 1.0 : 0.0;
                        double dpx = (p == 0) ? 1.0 : 0.0;
                        q_d[p][k][i] = dpa * dik + t_d[p][k] * e[i] + t * epsilon[p] * dik;
                        r_d[p][k][i] = dpx * dik - q_d[p][k][i];
                    }
                }
            }

            // --- second derivatives ---
            // alpha_{,pk,ql} = omega_p*epsilon_q*delta_{kl} + omega_q*epsilon_p*delta_{kl}
            // beta_{,pk,ql}  = 2*epsilon_p*epsilon_q*delta_{kl}

            // t_{,pk,ql} = alpha_{,pk,ql}/beta
            //            - (alpha_{,pk}*beta_{,ql} + alpha_{,ql}*beta_{,pk} + alpha*beta_{,pk,ql}) / beta^2
            //            + 2*alpha*beta_{,pk}*beta_{,ql} / beta^3

            // q_{i,pk,ql} = t_{,pk,ql}*e_i + t_{,pk}*epsilon_q*delta_{il} + t_{,ql}*epsilon_p*delta_{ik}
            // r_{i,pk,ql} = -q_{i,pk,ql}

            // d^2 delta / dy_{pk} dy_{ql}
            //   = (1/delta) * (delta_{ij} - u_i*u_j) * r_{i,pk} * r_{j,ql}
            //     - u_i * q_{i,pk,ql}

            // H_{pk,ql} = b'' * (u_i r_{i,pk}) * (u_j r_{j,ql})
            //           + b' * d^2 delta / dy_{pk} dy_{ql}

            for (int p = 0; p < 4; ++p) {
                for (int k = 0; k < 3; ++k) {
                    for (int q = 0; q < 4; ++q) {
                        for (int l = 0; l < 3; ++l) {

                            double dkl = (k == l) ? 1.0 : 0.0;

                            // alpha second derivatives
                            double alpha_pk = omega[p]*e[k] + epsilon[p]*w[k];
                            double alpha_ql = omega[q]*e[l] + epsilon[q]*w[l];
                            double alpha_pkql = (omega[p]*epsilon[q] + omega[q]*epsilon[p]) * dkl;

                            double beta_pk = 2.0*epsilon[p]*e[k];
                            double beta_ql = 2.0*epsilon[q]*e[l];
                            double beta_pkql = 2.0*epsilon[p]*epsilon[q]*dkl;

                            // t second derivative
                            double t_pkql = alpha_pkql / beta
                                            - (alpha_pk*beta_ql + alpha_ql*beta_pk + alpha*beta_pkql) / (beta*beta)
                                            + 2.0*alpha*beta_pk*beta_ql / (beta*beta*beta);

                            // q_{i,pk,ql} and sum for Hessian
                            double ddelta_pk = 0.0;  // u_i * r_{i,pk}
                            double ddelta_ql = 0.0;  // u_j * r_{j,ql}
                            for (int i = 0; i < 3; ++i) {
                                ddelta_pk += u[i] * r_d[p][k][i];
                                ddelta_ql += u[i] * r_d[q][l][i];
                            }

                            // (1/delta)*(delta_{ij}-u_i*u_j)*r_{i,pk}*r_{j,ql}
                            double proj_term = 0.0;
                            for (int i = 0; i < 3; ++i) {
                                for (int j = 0; j < 3; ++j) {
                                    double dij = (i == j) ? 1.0 : 0.0;
                                    proj_term += (dij - u[i]*u[j]) * r_d[p][k][i] * r_d[q][l][j];
                                }
                            }
                            proj_term /= delta;

                            // -u_i * q_{i,pk,ql}
                            double uq_term = 0.0;
                            for (int i = 0; i < 3; ++i) {
                                double dik = (i == k) ? 1.0 : 0.0;
                                double dil = (i == l) ? 1.0 : 0.0;
                                double q_ipkql = t_pkql * e[i]
                                                 + t_d[p][k] * epsilon[q] * dil
                                                 + t_d[q][l] * epsilon[p] * dik;
                                uq_term += u[i] * q_ipkql;
                            }

                            double d2delta = proj_term - uq_term;

                            out.hessian(3*p+k, 3*q+l) = bpp * ddelta_pk * ddelta_ql + bp * d2delta;
                        }
                    }
                }
            }
            break;
        }

            // ===============================================================
            //  Case 1: face interior
            //
            //  a_i = (x2)_i - (x1)_i,   b_i = (x3)_i - (x1)_i
            //  N_i = eps_{imn} a_m b_n,  eta = ||N||, n_i = N_i / eta
            //  w_i = x_i - (x1)_i
            //  psi = N_i w_i,  phi = psi / eta,  delta = |phi|,  s = sign(phi)
            //
            //  sigma^(a)_p: da_i/dy_{pk} = sigma^(a)_p * delta_{ik}
            //    { -1 for x1, +1 for x2, 0 for x,x3 }
            //
            //  sigma^(b)_p: db_i/dy_{pk} = sigma^(b)_p * delta_{ik}
            //    { -1 for x1, +1 for x3, 0 for x,x2 }
            //
            //  sigma^(w)_p: dw_i/dy_{pk} = sigma^(w)_p * delta_{ik}
            //    { +1 for x, -1 for x1, 0 for x2,x3 }
            //
            //  H_{pk,ql} = b'' * phi_{,pk} * phi_{,ql} + s * b' * phi_{,pk,ql}
            // ===============================================================
        case NodeTriangleRegion::FaceInterior:
        {
            // sigma coefficients for p = 0(x), 1(x1), 2(x2), 3(x3)
            double sig_a[4] = { 0.0, -1.0,  1.0,  0.0};  // da/dy_p
            double sig_b[4] = { 0.0, -1.0,  0.0,  1.0};  // db/dy_p
            double sig_w[4] = { 1.0, -1.0,  0.0,  0.0};  // dw/dy_p

            // a_i, b_i, w_i
            double a[3], b[3], w[3];
            for (int i = 0; i < 3; ++i) {
                a[i] = x2(i) - x1(i);
                b[i] = x3(i) - x1(i);
                w[i] = x(i)  - x1(i);
            }

            // N_i = eps_{imn} a_m b_n
            double N[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < 3; ++i)
                for (int m = 0; m < 3; ++m)
                    for (int n = 0; n < 3; ++n)
                        N[i] += levi_civita(i, m, n) * a[m] * b[n];

            // eta = ||N||
            double eta = 0.0;
            for (int i = 0; i < 3; ++i) eta += N[i] * N[i];
            eta = std::sqrt(eta);

            // n_i = N_i / eta
            double n[3];
            for (int i = 0; i < 3; ++i) n[i] = N[i] / eta;

            // psi = N_i w_i
            double psi = 0.0;
            for (int i = 0; i < 3; ++i) psi += N[i] * w[i];

            // phi = psi / eta
            double phi = psi / eta;

            // s = sign(phi)
            double s;
            if      (phi > 0.0) s =  1.0;
            else if (phi < 0.0) s = -1.0;
            else                s =  0.0;

            // --- Precompute N_{i,pk} for all (p,k) ---
            // N_{i,pk} = eps_{imn} (sig_a_p * delta_{mk} * b_n + a_m * sig_b_p * delta_{nk})
            //          = eps_{imn} * sig_a_p * delta_{mk} * b_n + eps_{imn} * a_m * sig_b_p * delta_{nk}
            //          = sig_a_p * eps_{ikn} * b_n + sig_b_p * eps_{imk} * a_m
            double Nd[4][3][3];  // Nd[p][k][i] = N_{i,pk}
            for (int p = 0; p < 4; ++p) {
                for (int k = 0; k < 3; ++k) {
                    for (int i = 0; i < 3; ++i) {
                        double val = 0.0;
                        for (int n = 0; n < 3; ++n)
                            val += sig_a[p] * levi_civita(i, k, n) * b[n];
                        for (int m = 0; m < 3; ++m)
                            val += sig_b[p] * levi_civita(i, m, k) * a[m];
                        Nd[p][k][i] = val;
                    }
                }
            }

            // --- Precompute eta_{,pk}, psi_{,pk}, phi_{,pk} ---
            double eta_d[4][3];
            double psi_d[4][3];
            double phi_d[4][3];

            for (int p = 0; p < 4; ++p) {
                for (int k = 0; k < 3; ++k) {
                    // eta_{,pk} = n_i * N_{i,pk}
                    double eta_pk = 0.0;
                    for (int i = 0; i < 3; ++i) eta_pk += n[i] * Nd[p][k][i];
                    eta_d[p][k] = eta_pk;

                    // psi_{,pk} = N_{i,pk} * w_i + N_i * sig_w_p * delta_{ik}
                    //           = N_{i,pk} * w_i + sig_w_p * N_k
                    double psi_pk = 0.0;
                    for (int i = 0; i < 3; ++i) psi_pk += Nd[p][k][i] * w[i];
                    psi_pk += sig_w[p] * N[k];
                    psi_d[p][k] = psi_pk;

                    // phi_{,pk} = psi_{,pk}/eta - psi*eta_{,pk}/eta^2
                    phi_d[p][k] = psi_pk / eta - psi * eta_pk / (eta * eta);
                }
            }

            // --- Compute H_{pk,ql} = b'' * phi_{,pk} * phi_{,ql} + s * b' * phi_{,pk,ql} ---
            for (int p = 0; p < 4; ++p) {
                for (int k = 0; k < 3; ++k) {
                    for (int q = 0; q < 4; ++q) {
                        for (int l = 0; l < 3; ++l) {

                            double dkl = (k == l) ? 1.0 : 0.0;

                            // N_{i,pk,ql} = eps_{imn} (sig_a_p * delta_{mk} * sig_b_q * delta_{nl}
                            //             + sig_a_q * delta_{ml} * sig_b_p * delta_{nk})
                            //            = sig_a_p * sig_b_q * eps_{ikl} + sig_a_q * sig_b_p * eps_{ilk}
                            // Since eps_{ilk} = -eps_{ikl}:
                            //            = (sig_a_p * sig_b_q - sig_a_q * sig_b_p) * eps_{ikl}
                            double coeff_N2 = sig_a[p] * sig_b[q] - sig_a[q] * sig_b[p];

                            // eta_{,pk,ql} = n_i * N_{i,pk,ql}
                            //              + (1/eta) * (delta_{ij} - n_i*n_j) * N_{i,pk} * N_{j,ql}
                            double nN2 = 0.0;  // n_i * N_{i,pk,ql}
                            for (int i = 0; i < 3; ++i)
                                nN2 += n[i] * coeff_N2 * levi_civita(i, k, l);

                            double proj_NN = 0.0;  // (delta_{ij}-n_i*n_j)*N_{i,pk}*N_{j,ql}
                            for (int i = 0; i < 3; ++i)
                                for (int j = 0; j < 3; ++j) {
                                    double dij = (i == j) ? 1.0 : 0.0;
                                    proj_NN += (dij - n[i]*n[j]) * Nd[p][k][i] * Nd[q][l][j];
                                }

                            double eta_pkql = nN2 + proj_NN / eta;

                            // psi_{,pk,ql} = N_{i,pk,ql} * w_i
                            //              + N_{i,pk} * sig_w_q * delta_{il}
                            //              + N_{i,ql} * sig_w_p * delta_{ik}
                            double psi_pkql = 0.0;
                            for (int i = 0; i < 3; ++i)
                                psi_pkql += coeff_N2 * levi_civita(i, k, l) * w[i];
                            psi_pkql += sig_w[q] * Nd[p][k][l];  // N_{l,pk} * sig_w_q
                            psi_pkql += sig_w[p] * Nd[q][l][k];  // N_{k,ql} * sig_w_p

                            // phi_{,pk,ql} = psi_{,pk,ql}/eta
                            //   - (psi_{,pk}*eta_{,ql} + psi_{,ql}*eta_{,pk} + psi*eta_{,pk,ql}) / eta^2
                            //   + 2*psi*eta_{,pk}*eta_{,ql} / eta^3
                            double phi_pkql =
                                    psi_pkql / eta
                                    - (psi_d[p][k]*eta_d[q][l] + psi_d[q][l]*eta_d[p][k] + psi*eta_pkql) / (eta*eta)
                                    + 2.0*psi*eta_d[p][k]*eta_d[q][l] / (eta*eta*eta);

                            out.hessian(3*p+k, 3*q+l) = bpp * phi_d[p][k] * phi_d[q][l]
                                                        + s * bp * phi_pkql;
                        }
                    }
                }
            }
            break;
        }

        case NodeTriangleRegion::DegenerateTriangle:
            // No Hessian for degenerate case — leave as zero
            break;
    }

    return out;
}