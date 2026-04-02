#include "barrier_energy.h"

#include <cmath>
#include <stdexcept>


//  t = ((q - a) · (b - a)) / ||b - a||^2
//  q = a + t (b - a)
//  t in [0, 1]
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

//  eps_{ijk} = (i-j)(j-k)(k-i) / 2
double levi_civita(int i, int j, int k){
    return 0.5 * (i - j) * (j - k) * (k - i);
}

//  b(delta) = -(delta - d_hat)^2 log(delta / d_hat)
//  b(delta) = 0  for delta >= d_hat
double scalar_barrier(double delta, double d_hat){
    if (d_hat <= 0.0) throw std::runtime_error("scalar_barrier: d_hat must be positive.");
    if (delta >= d_hat) return 0.0;
    if (delta <= 0.0) throw std::runtime_error("scalar_barrier: delta must be positive.");

    const double s = delta - d_hat;
    return -(s * s) * std::log(delta / d_hat);
}

//  b'(delta) = -2 (delta - d_hat) log(delta / d_hat) - (delta - d_hat)^2 / delta
double scalar_barrier_gradient(double delta, double d_hat){
    if (d_hat <= 0.0) throw std::runtime_error("scalar_barrier_gradient: d_hat must be positive.");
    if (delta >= d_hat) return 0.0;
    if (delta <= 0.0) throw std::runtime_error("scalar_barrier_gradient: delta must be positive.");

    const double s = delta - d_hat;
    return -2.0 * s * std::log(delta / d_hat) - (s * s) / delta;
}

// b''(delta) = d_hat^2 / delta^2 + 2 d_hat / delta - 3 - 2 log(delta / d_hat)
double scalar_barrier_hessian(double delta, double d_hat){
    if (d_hat <= 0.0) throw std::runtime_error("scalar_barrier_hessian: d_hat must be positive.");
    if (delta >= d_hat) return 0.0;
    if (delta <= 0.0) throw std::runtime_error("scalar_barrier_hessian: delta must be positive.");

    const double ratio = d_hat / delta;
    return ratio * ratio + 2.0 * ratio - 3.0 - 2.0 * std::log(delta / d_hat);
}

// E = b(delta), delta = node_triangle_distance(x, x1, x2, x3)
double node_triangle_barrier(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps){
    const NodeTriangleDistanceResult dr = node_triangle_distance(x, x1, x2, x3, eps);
    return scalar_barrier(dr.distance, d_hat);
}

// E(y) = b(delta(y)), dE/dy = b'(delta) * ddelta/dy
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
            // simplified fallback: assign to the closest vertex
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

//  H_{pk,ql} = b''(delta) * (d delta/dy_{pk}) * (d delta/dy_{ql}) + b'(delta)  * (d^2 delta / dy_{pk} dy_{ql})
//  DOF ordering: p = 0..3 for (x, x1, x2, x3), k,l = 0..2 for spatial.
//  Row/col index in 12x12:  3*p + k.

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
                            double phi_pkql = psi_pkql / eta - (psi_d[p][k]*eta_d[q][l] + psi_d[q][l]*eta_d[p][k] + psi*eta_pkql) / (eta*eta)
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
            // No Hessian for degenerate case and leave as zero
            break;
    }

    return out;
}

// =====================================================================
//  Segment--segment barrier
// =====================================================================

// E = b(delta), delta = segment_segment_distance(x1, x2, x3, x4)
double segment_segment_barrier(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, double eps){
    const auto dr = segment_segment_distance(x1, x2, x3, x4, eps);
    return scalar_barrier(dr.distance, d_hat);
}

// E(y) = b(delta(y)), dE/dy_{pk} = b'(delta) * ddelta/dy_{pk}
// DOF ordering: p = 0(x1), 1(x2), 2(x3), 3(x4), k = 0..2
SegmentSegmentBarrierResult segment_segment_barrier_gradient(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, double eps){
    SegmentSegmentBarrierResult out;
    out.distance_result = segment_segment_distance(x1, x2, x3, x4, eps);
    out.distance = out.distance_result.distance;
    out.energy = scalar_barrier(out.distance, d_hat);
    out.barrier_derivative = scalar_barrier_gradient(out.distance, d_hat);

    if (out.barrier_derivative == 0.0) return out;

    const auto& dr = out.distance_result;
    const double delta = dr.distance;
    const double bp = out.barrier_derivative;

    if (delta <= 0.0) throw std::runtime_error("segment_segment_barrier_gradient: distance must be positive.");

    const Vec3 r = dr.closest_point_1 - dr.closest_point_2;
    double u[3];
    for (int k = 0; k < 3; ++k) u[k] = r(k) / delta;

    const double s = dr.s;
    const double t = dr.t;

    switch (dr.region) {

        // Case 1: interior
        // mu_1 = 1-s, mu_2 = s, mu_3 = -(1-t), mu_4 = -t
        case SegmentSegmentRegion::Interior:
        {
            const double mu[4] = {1.0-s, s, -(1.0-t), -t};
            for (int k = 0; k < 3; ++k) {
                out.grad_x1(k) = bp * mu[0] * u[k];
                out.grad_x2(k) = bp * mu[1] * u[k];
                out.grad_x3(k) = bp * mu[2] * u[k];
                out.grad_x4(k) = bp * mu[3] * u[k];
            }
            break;
        }

            // Cases 2-5: point-segment (one parameter clamped)
            // These reduce to the node-triangle edge-interior gradient structure.
        case SegmentSegmentRegion::Edge_s0:  // x1 vs (x3,x4), x2 inactive
        {
            for (int k = 0; k < 3; ++k) {
                out.grad_x1(k) =  bp * u[k];
                out.grad_x2(k) =  0.0;
                out.grad_x3(k) = -bp * (1.0 - t) * u[k];
                out.grad_x4(k) = -bp * t * u[k];
            }
            break;
        }

        case SegmentSegmentRegion::Edge_s1:  // x2 vs (x3,x4), x1 inactive
        {
            for (int k = 0; k < 3; ++k) {
                out.grad_x1(k) =  0.0;
                out.grad_x2(k) =  bp * u[k];
                out.grad_x3(k) = -bp * (1.0 - t) * u[k];
                out.grad_x4(k) = -bp * t * u[k];
            }
            break;
        }

        case SegmentSegmentRegion::Edge_t0:  // x3 vs (x1,x2), x4 inactive
        {
            // r = x3_closest - edge_closest = closest_point_2 - closest_point_1 = -u direction
            for (int k = 0; k < 3; ++k) {
                out.grad_x1(k) =  bp * (1.0 - s) * u[k];
                out.grad_x2(k) =  bp * s * u[k];
                out.grad_x3(k) = -bp * u[k];
                out.grad_x4(k) =  0.0;
            }
            break;
        }

        case SegmentSegmentRegion::Edge_t1:  // x4 vs (x1,x2), x3 inactive
        {
            // r = x4_closest - edge_closest = closest_point_2 - closest_point_1 = -u direction
            for (int k = 0; k < 3; ++k) {
                out.grad_x1(k) =  bp * (1.0 - s) * u[k];
                out.grad_x2(k) =  bp * s * u[k];
                out.grad_x3(k) =  0.0;
                out.grad_x4(k) = -bp * u[k];
            }
            break;
        }

            // Cases 6-9: vertex-vertex
        case SegmentSegmentRegion::Corner_s0t0:  // x1 vs x3
        {
            for (int k = 0; k < 3; ++k) {
                out.grad_x1(k) =  bp * u[k]; out.grad_x2(k) = 0.0;
                out.grad_x3(k) = -bp * u[k]; out.grad_x4(k) = 0.0;
            }
            break;
        }

        case SegmentSegmentRegion::Corner_s0t1:  // x1 vs x4
        {
            for (int k = 0; k < 3; ++k) {
                out.grad_x1(k) =  bp * u[k]; out.grad_x2(k) = 0.0;
                out.grad_x3(k) = 0.0;        out.grad_x4(k) = -bp * u[k];
            }
            break;
        }

        case SegmentSegmentRegion::Corner_s1t0:  // x2 vs x3
        {
            for (int k = 0; k < 3; ++k) {
                out.grad_x1(k) = 0.0;        out.grad_x2(k) =  bp * u[k];
                out.grad_x3(k) = -bp * u[k]; out.grad_x4(k) = 0.0;
            }
            break;
        }

        case SegmentSegmentRegion::Corner_s1t1:  // x2 vs x4
        {
            for (int k = 0; k < 3; ++k) {
                out.grad_x1(k) = 0.0;        out.grad_x2(k) =  bp * u[k];
                out.grad_x3(k) = 0.0;        out.grad_x4(k) = -bp * u[k];
            }
            break;
        }

        case SegmentSegmentRegion::ParallelSegments:
        {
            // Fallback: use mu weights from the resolved (s,t)
            const double mu[4] = {1.0-s, s, -(1.0-t), -t};
            for (int k = 0; k < 3; ++k) {
                out.grad_x1(k) = bp * mu[0] * u[k];
                out.grad_x2(k) = bp * mu[1] * u[k];
                out.grad_x3(k) = bp * mu[2] * u[k];
                out.grad_x4(k) = bp * mu[3] * u[k];
            }
            break;
        }
    }

    return out;
}

// H_{pk,ql} = b'' * ddelta/dy_{pk} * ddelta/dy_{ql} + b' * d2delta/dy_{pk}dy_{ql}
// DOF ordering: p = 0(x1), 1(x2), 2(x3), 3(x4), k,l = 0..2
SegmentSegmentBarrierHessianResult segment_segment_barrier_hessian(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double d_hat, double eps){
    SegmentSegmentBarrierHessianResult out;

    const auto gr = segment_segment_barrier_gradient(x1, x2, x3, x4, d_hat, eps);
    out.distance_result = gr.distance_result;
    out.distance = gr.distance;
    out.energy = gr.energy;
    out.grad_x1 = gr.grad_x1;
    out.grad_x2 = gr.grad_x2;
    out.grad_x3 = gr.grad_x3;
    out.grad_x4 = gr.grad_x4;

    const double delta = gr.distance;
    const double bp  = scalar_barrier_gradient(delta, d_hat);
    const double bpp = scalar_barrier_hessian(delta, d_hat);

    if (bp == 0.0 && bpp == 0.0) return out;

    const auto& dr = gr.distance_result;
    if (delta <= 0.0) throw std::runtime_error("segment_segment_barrier_hessian: distance must be positive.");

    const Vec3* Y[4] = {&x1, &x2, &x3, &x4};

    switch (dr.region) {

        // ===============================================================
        //  Cases 6-9: vertex-vertex
        //  Same structure as node-triangle vertex cases.
        // ===============================================================
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

            double u[3];
            for (int k = 0; k < 3; ++k) u[k] = ((*Y[a_idx])(k) - (*Y[b_idx])(k)) / delta;

            const double c1 = bpp;
            const double c2 = bp / delta;

            for (int p = 0; p < 4; ++p) {
                for (int q = 0; q < 4; ++q) {
                    const double sq = sp[p] * sp[q];
                    if (sq == 0.0) continue;
                    for (int k = 0; k < 3; ++k) {
                        for (int l = 0; l < 3; ++l) {
                            double dkl = (k == l) ? 1.0 : 0.0;
                            out.hessian(3*p+k, 3*q+l) = sq * (c1 * u[k] * u[l] + c2 * (dkl - u[k] * u[l]));
                        }
                    }
                }
            }
            break;
        }

            // ===============================================================
            //  Cases 2-5: point-segment
            //  One parameter clamped. Reduces to node-triangle edge-interior Hessian.
            // ===============================================================
        case SegmentSegmentRegion::Edge_s0:
        case SegmentSegmentRegion::Edge_s1:
        case SegmentSegmentRegion::Edge_t0:
        case SegmentSegmentRegion::Edge_t1:
        {
            // Identify the query point, edge endpoints, and inactive vertex
            int query_idx, ea_idx, eb_idx, inactive_idx;
            if      (dr.region == SegmentSegmentRegion::Edge_s0) { query_idx = 0; ea_idx = 2; eb_idx = 3; inactive_idx = 1; }
            else if (dr.region == SegmentSegmentRegion::Edge_s1) { query_idx = 1; ea_idx = 2; eb_idx = 3; inactive_idx = 0; }
            else if (dr.region == SegmentSegmentRegion::Edge_t0) { query_idx = 2; ea_idx = 0; eb_idx = 1; inactive_idx = 3; }
            else                                                  { query_idx = 3; ea_idx = 0; eb_idx = 1; inactive_idx = 2; }

            const Vec3& xq  = *Y[query_idx];
            const Vec3& xea = *Y[ea_idx];
            const Vec3& xeb = *Y[eb_idx];

            // omega_p and epsilon_p
            double omega[4]   = {0.0, 0.0, 0.0, 0.0};
            double epsilon[4] = {0.0, 0.0, 0.0, 0.0};
            omega[query_idx] = 1.0; omega[ea_idx] = -1.0;
            epsilon[ea_idx] = -1.0; epsilon[eb_idx] = 1.0;

            double e[3], w[3];
            for (int i = 0; i < 3; ++i) {
                e[i] = xeb(i) - xea(i);
                w[i] = xq(i)  - xea(i);
            }

            double alpha = 0.0, beta = 0.0;
            for (int i = 0; i < 3; ++i) { alpha += w[i]*e[i]; beta += e[i]*e[i]; }

            double t = alpha / beta;

            double r[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r[i] = xq(i) - (xea(i) + t * e[i]);
                u[i] = r[i] / delta;
            }

            // Precompute t_{,pk} and r_{i,pk}, q_{i,pk}
            double t_d[4][3];
            double r_d[4][3][3];
            double q_d[4][3][3];

            for (int p = 0; p < 4; ++p) {
                for (int k = 0; k < 3; ++k) {
                    double alpha_pk = omega[p] * e[k] + epsilon[p] * w[k];
                    double beta_pk  = 2.0 * epsilon[p] * e[k];
                    t_d[p][k] = alpha_pk / beta - alpha * beta_pk / (beta * beta);

                    for (int i = 0; i < 3; ++i) {
                        double dik = (i == k) ? 1.0 : 0.0;
                        double dp_ea = (p == ea_idx) ? 1.0 : 0.0;
                        double dp_q  = (p == query_idx) ? 1.0 : 0.0;
                        q_d[p][k][i] = dp_ea * dik + t_d[p][k] * e[i] + t * epsilon[p] * dik;
                        r_d[p][k][i] = dp_q * dik - q_d[p][k][i];
                    }
                }
            }

            // Second derivatives and Hessian assembly
            for (int p = 0; p < 4; ++p) {
                for (int k = 0; k < 3; ++k) {
                    for (int q = 0; q < 4; ++q) {
                        for (int l = 0; l < 3; ++l) {
                            double dkl = (k == l) ? 1.0 : 0.0;

                            double alpha_pk = omega[p]*e[k] + epsilon[p]*w[k];
                            double alpha_ql = omega[q]*e[l] + epsilon[q]*w[l];
                            double alpha_pkql = (omega[p]*epsilon[q] + omega[q]*epsilon[p]) * dkl;

                            double beta_pk = 2.0*epsilon[p]*e[k];
                            double beta_ql = 2.0*epsilon[q]*e[l];
                            double beta_pkql = 2.0*epsilon[p]*epsilon[q]*dkl;

                            double t_pkql = alpha_pkql / beta
                                            - (alpha_pk*beta_ql + alpha_ql*beta_pk + alpha*beta_pkql) / (beta*beta)
                                            + 2.0*alpha*beta_pk*beta_ql / (beta*beta*beta);

                            double ddelta_pk = 0.0, ddelta_ql = 0.0;
                            for (int i = 0; i < 3; ++i) {
                                ddelta_pk += u[i] * r_d[p][k][i];
                                ddelta_ql += u[i] * r_d[q][l][i];
                            }

                            double proj_term = 0.0;
                            for (int i = 0; i < 3; ++i) {
                                for (int j = 0; j < 3; ++j) {
                                    double dij = (i == j) ? 1.0 : 0.0;
                                    proj_term += (dij - u[i]*u[j]) * r_d[p][k][i] * r_d[q][l][j];
                                }
                            }
                            proj_term /= delta;

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
            //  Case 1: interior
            //  Two free parameters s, t. Full differentiation through the
            //  quotient rule for s = nu/Delta, t = zeta/Delta.
            // ===============================================================
        case SegmentSegmentRegion::Interior:
        {
            // sigma coefficients for p = 0(x1), 1(x2), 2(x3), 3(x4)
            double sig_a[4] = {-1.0,  1.0,  0.0,  0.0};  // da/dy_p, a = x2-x1
            double sig_b[4] = { 0.0,  0.0, -1.0,  1.0};  // db/dy_p, b = x4-x3
            double sig_c[4] = { 1.0,  0.0, -1.0,  0.0};  // dc/dy_p, c = x1-x3

            double a[3], b[3], c[3];
            for (int i = 0; i < 3; ++i) {
                a[i] = x2(i) - x1(i);
                b[i] = x4(i) - x3(i);
                c[i] = x1(i) - x3(i);
            }

            double A = 0.0, B = 0.0, C = 0.0, D = 0.0, E = 0.0;
            for (int i = 0; i < 3; ++i) {
                A += a[i]*a[i]; B += a[i]*b[i]; C += b[i]*b[i];
                D += a[i]*c[i]; E += b[i]*c[i];
            }

            double Delta = A*C - B*B;
            double nu    = B*E - C*D;   // numerator for s
            double zeta  = A*E - B*D;   // numerator for t
            double s = nu / Delta;
            double t = zeta / Delta;

            // First derivatives of A,B,C,D,E
            double Ad[4][3], Bd[4][3], Cd[4][3], Dd[4][3], Ed[4][3];
            for (int p = 0; p < 4; ++p) {
                for (int k = 0; k < 3; ++k) {
                    Ad[p][k] = 2.0 * sig_a[p] * a[k];
                    Bd[p][k] = sig_a[p] * b[k] + sig_b[p] * a[k];
                    Cd[p][k] = 2.0 * sig_b[p] * b[k];
                    Dd[p][k] = sig_a[p] * c[k] + sig_c[p] * a[k];
                    Ed[p][k] = sig_b[p] * c[k] + sig_c[p] * b[k];
                }
            }

            // First derivatives of nu, zeta, Delta
            double nu_d[4][3], zeta_d[4][3], Delta_d[4][3];
            for (int p = 0; p < 4; ++p) {
                for (int k = 0; k < 3; ++k) {
                    nu_d[p][k]    = Bd[p][k]*E + B*Ed[p][k] - Cd[p][k]*D - C*Dd[p][k];
                    zeta_d[p][k]  = Ad[p][k]*E + A*Ed[p][k] - Bd[p][k]*D - B*Dd[p][k];
                    Delta_d[p][k] = Ad[p][k]*C + A*Cd[p][k] - 2.0*B*Bd[p][k];
                }
            }

            // s_{,pk} and t_{,pk}
            double s_d[4][3], t_d_arr[4][3];
            for (int p = 0; p < 4; ++p) {
                for (int k = 0; k < 3; ++k) {
                    s_d[p][k]     = nu_d[p][k]/Delta   - nu*Delta_d[p][k]/(Delta*Delta);
                    t_d_arr[p][k] = zeta_d[p][k]/Delta - zeta*Delta_d[p][k]/(Delta*Delta);
                }
            }

            // p_i = (x1)_i + s*a_i,  q_i = (x3)_i + t*b_i
            // p_{i,pk} = delta_{p,0}*delta_{ik} + s_{,pk}*a_i + s*sig_a_p*delta_{ik}
            // q_{i,pk} = delta_{p,2}*delta_{ik} + t_{,pk}*b_i + t*sig_b_p*delta_{ik}
            // r_{i,pk} = p_{i,pk} - q_{i,pk}
            double r_d[4][3][3];
            double p_d[4][3][3], q_d[4][3][3];

            for (int p = 0; p < 4; ++p) {
                for (int k = 0; k < 3; ++k) {
                    for (int i = 0; i < 3; ++i) {
                        double dik = (i == k) ? 1.0 : 0.0;
                        double dp0 = (p == 0) ? 1.0 : 0.0;  // x1
                        double dp2 = (p == 2) ? 1.0 : 0.0;  // x3
                        p_d[p][k][i] = dp0 * dik + s_d[p][k] * a[i] + s * sig_a[p] * dik;
                        q_d[p][k][i] = dp2 * dik + t_d_arr[p][k] * b[i] + t * sig_b[p] * dik;
                        r_d[p][k][i] = p_d[p][k][i] - q_d[p][k][i];
                    }
                }
            }

            double r_vec[3], u[3];
            for (int i = 0; i < 3; ++i) {
                r_vec[i] = (x1(i) + s*a[i]) - (x3(i) + t*b[i]);
                u[i] = r_vec[i] / delta;
            }

            // Second derivatives and Hessian
            for (int p = 0; p < 4; ++p) {
                for (int k = 0; k < 3; ++k) {
                    for (int q = 0; q < 4; ++q) {
                        for (int l = 0; l < 3; ++l) {
                            double dkl = (k == l) ? 1.0 : 0.0;

                            // Second derivatives of A,B,C,D,E
                            double A_pkql = 2.0 * sig_a[p] * sig_a[q] * dkl;
                            double B_pkql = (sig_a[p]*sig_b[q] + sig_a[q]*sig_b[p]) * dkl;
                            double C_pkql = 2.0 * sig_b[p] * sig_b[q] * dkl;
                            double D_pkql = (sig_a[p]*sig_c[q] + sig_a[q]*sig_c[p]) * dkl;
                            double E_pkql = (sig_b[p]*sig_c[q] + sig_b[q]*sig_c[p]) * dkl;

                            // nu second derivative
                            double nu_pkql = B_pkql*E + Bd[p][k]*Ed[q][l] + Bd[q][l]*Ed[p][k] + B*E_pkql
                                             - C_pkql*D - Cd[p][k]*Dd[q][l] - Cd[q][l]*Dd[p][k] - C*D_pkql;

                            // Delta second derivative
                            double Delta_pkql = A_pkql*C + Ad[p][k]*Cd[q][l] + Ad[q][l]*Cd[p][k] + A*C_pkql
                                                - 2.0*(Bd[p][k]*Bd[q][l] + B*B_pkql);

                            // zeta second derivative
                            double zeta_pkql = A_pkql*E + Ad[p][k]*Ed[q][l] + Ad[q][l]*Ed[p][k] + A*E_pkql
                                               - B_pkql*D - Bd[p][k]*Dd[q][l] - Bd[q][l]*Dd[p][k] - B*D_pkql;

                            // s_{,pk,ql}
                            double s_pkql = nu_pkql/Delta
                                            - (nu_d[p][k]*Delta_d[q][l] + nu_d[q][l]*Delta_d[p][k] + nu*Delta_pkql)/(Delta*Delta)
                                            + 2.0*nu*Delta_d[p][k]*Delta_d[q][l]/(Delta*Delta*Delta);

                            // t_{,pk,ql}
                            double t_pkql = zeta_pkql/Delta
                                            - (zeta_d[p][k]*Delta_d[q][l] + zeta_d[q][l]*Delta_d[p][k] + zeta*Delta_pkql)/(Delta*Delta)
                                            + 2.0*zeta*Delta_d[p][k]*Delta_d[q][l]/(Delta*Delta*Delta);

                            // p_{i,pk,ql} = s_{,pk,ql}*a_i + s_{,pk}*sig_a_q*delta_{il} + s_{,ql}*sig_a_p*delta_{ik}
                            // q_{i,pk,ql} = t_{,pk,ql}*b_i + t_{,pk}*sig_b_q*delta_{il} + t_{,ql}*sig_b_p*delta_{ik}
                            // r_{i,pk,ql} = p_{i,pk,ql} - q_{i,pk,ql}

                            // delta_{,pk} = u_i * r_{i,pk}
                            double ddelta_pk = 0.0, ddelta_ql = 0.0;
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

                            // u_i * r_{i,pk,ql}
                            double ur_term = 0.0;
                            for (int i = 0; i < 3; ++i) {
                                double dik = (i == k) ? 1.0 : 0.0;
                                double dil = (i == l) ? 1.0 : 0.0;
                                double p_ipkql = s_pkql * a[i] + s_d[p][k] * sig_a[q] * dil + s_d[q][l] * sig_a[p] * dik;
                                double q_ipkql = t_pkql * b[i] + t_d_arr[p][k] * sig_b[q] * dil + t_d_arr[q][l] * sig_b[p] * dik;
                                ur_term += u[i] * (p_ipkql - q_ipkql);
                            }

                            double d2delta = proj_term + ur_term;

                            out.hessian(3*p+k, 3*q+l) = bpp * ddelta_pk * ddelta_ql + bp * d2delta;
                        }
                    }
                }
            }
            break;
        }

        case SegmentSegmentRegion::ParallelSegments:
            // No Hessian for parallel case, leave as zero
            break;
    }

    return out;
}