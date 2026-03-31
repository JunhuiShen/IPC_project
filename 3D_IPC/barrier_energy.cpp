#include "barrier_energy.h"

#include <cmath>
#include <stdexcept>

// Find the edge parameter t from the closest point q on segment [a, b]
// ab_k = b_k - a_k,  denom = ab_k * ab_k (sum over k), and t = clamp( (q_k - a_k) * ab_k / denom, 0, 1 )
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

//  Scalar barrier: b(delta; d_hat) = -(delta - d_hat)^2 * ln(delta / d_hat)
double scalar_barrier(double delta, double d_hat)
{
    if (d_hat <= 0.0) {
        throw std::runtime_error("scalar_barrier: d_hat must be positive.");
    }
    if (delta >= d_hat) {
        return 0.0;
    }
    if (delta <= 0.0) {
        throw std::runtime_error("scalar_barrier: delta must be positive.");
    }

    const double s = delta - d_hat;
    return -(s * s) * std::log(delta / d_hat);
}

//  Scalar barrier gradient: db/d(delta) = -2*(delta - d_hat)*ln(delta/d_hat) - (delta - d_hat)^2 / delta
double scalar_barrier_gradient(double delta, double d_hat){
    if (d_hat <= 0.0) {
        throw std::runtime_error("scalar_barrier_gradient: d_hat must be positive.");
    }
    if (delta >= d_hat) {
        return 0.0;
    }
    if (delta <= 0.0) {
        throw std::runtime_error("scalar_barrier_gradient: delta must be positive.");
    }

    const double s = delta - d_hat;
    return -2.0 * s * std::log(delta / d_hat) - (s * s) / delta;
}

// Barrier energy with node-triangle distance
double node_triangle_barrier(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps){
    const NodeTriangleDistanceResult dr = node_triangle_distance(x, x1, x2, x3, eps);
    return scalar_barrier(dr.distance, d_hat);
}

//  Barrier energy gradient with node-triangle distance
//  In general, we have dPE / dy_alpha = b'(delta) * (r_k / delta) * (dx_k/dy_alpha - dq_k/dy_alpha) where  r_k = x_k - q_k,  delta = sqrt(r_k * r_k)
//  Each case below expands this for every degree of freedom over k = 0, 1 , 2
NodeTriangleBarrierResult node_triangle_barrier_gradient(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double d_hat, double eps){
    NodeTriangleBarrierResult out;
    out.distance_result = node_triangle_distance(x, x1, x2, x3, eps);
    out.distance = out.distance_result.distance;
    out.energy = scalar_barrier(out.distance, d_hat);
    out.barrier_derivative = scalar_barrier_gradient(out.distance, d_hat);

    // If outside the activation zone, everything is zero
    if (out.barrier_derivative == 0.0) {
        return out;
    }

    const NodeTriangleDistanceResult& dr = out.distance_result;
    const double delta = dr.distance;
    const double bp = out.barrier_derivative; // b'(delta; d_hat)

    if (delta <= 0.0) {
        throw std::runtime_error("node_triangle_barrier_gradient: distance must be positive.");
    }

    // u_k = r_k / delta where r_k = x_k - q_k
    double r[3], u[3];
    for (int k = 0; k < 3; ++k) {
        r[k] = x(k) - dr.closest_point(k);
        u[k] = r[k] / delta;
    }

    switch (dr.region) {

        // Case 1: face interior
        // Let tilde_x be the projection onto the triangle plane
        // We have q = tilde_x, delta = |phi|, where phi is the signed distance to the triangle plane
        // dPE / dx_k =  b' * sign(phi) * n_k and dPE / d(x_a)_k = -b' * sign(phi) * lambda_a * n_k
        case NodeTriangleRegion::FaceInterior:{
            const double phi = dr.phi;
            double sphi; // sign of phi
            if  (phi > 0.0) sphi =  1.0;
            else if (phi < 0.0) sphi = -1.0;
            else sphi =  0.0;

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

        // Case 2: edge (x1, x2)
        // We have q = (1-t)*x1 + t*x2
        // Let  u = || x- q|| /delta. dPE / dx_k = b' * u_k. dPE / d(x1)_k = -b' * (1-t) * u_k.
        // dPE / d(x2)_k  = -b' * t * u_k.  dPE / d(x3)_k =  0.
        case NodeTriangleRegion::Edge12:{
            const double t = segment_parameter_from_closest_point(dr.closest_point, x1, x2);

            for (int k = 0; k < 3; ++k) {
                out.grad_x(k)  =  bp * u[k];
                out.grad_x1(k) = -bp * (1.0 - t) * u[k];
                out.grad_x2(k) = -bp * t * u[k];
                out.grad_x3(k) =  0.0;
            }
            break;
        }

        // Case 3: edge (x2, x3)
        // We have q = (1-t)*x2 + t*x3
        // Let  u = || x- q|| /delta. dPE / dx_k = b' * u_k. dPE / d(x1)_k = -b' * (1-t) * u_k.
        // dPE / d(x2)_k = -b' * (1-t) * u_k. dPE / d(x3)_k = -b' * t * u_k.
        case NodeTriangleRegion::Edge23:{
            const double t = segment_parameter_from_closest_point(dr.closest_point, x2, x3);

            for (int k = 0; k < 3; ++k) {
                out.grad_x(k)  =  bp * u[k];
                out.grad_x1(k) =  0.0;
                out.grad_x2(k) = -bp * (1.0 - t) * u[k];
                out.grad_x3(k) = -bp * t * u[k];
            }
            break;
        }

        // Case 4: edge (x3, x1)
        // We have q = (1-t)*x3 + t*x1
        // Let  u = || x- q|| /delta. dPE / dx_k = b' * u_k. dPE / d(x1)_k = -b' * t * u_k.
        // dPE / d(x2)_k =  0. dPE / d(x3)_k = -b' * (1-t) * u_k.
        case NodeTriangleRegion::Edge31:{
            const double t = segment_parameter_from_closest_point(dr.closest_point, x3, x1);

            for (int k = 0; k < 3; ++k) {
                out.grad_x(k)  =  bp * u[k];
                out.grad_x1(k) = -bp * t * u[k];
                out.grad_x2(k) =  0.0;
                out.grad_x3(k) = -bp * (1.0 - t) * u[k];
            }
            break;
        }

        // Case 5: vertex x1
        // We have q = x1
        // Let  u = || x- q|| /delta. dPE / dx_k = b' * u_k. dPE / d(x1)_k = -b' * u_k.
        // dPE / d(x2)_k =  0. dPE / d(x3)_k = 0.
        case NodeTriangleRegion::Vertex1:{
            for (int k = 0; k < 3; ++k) {
                out.grad_x(k)  =  bp * u[k];
                out.grad_x1(k) = -bp * u[k];
                out.grad_x2(k) =  0.0;
                out.grad_x3(k) =  0.0;
            }
            break;
        }

        // Case 6: vertex x2
        // We have q = x2.
        // Let  u = || x - q|| /delta. dPE / dx_k =  b' * u_k.dPE / d(x1)_k =  0.
        // dPE / d(x2)_k = -b' * u_k. dPE / d(x3)_k =  0.
        case NodeTriangleRegion::Vertex2:{
            for (int k = 0; k < 3; ++k) {
                out.grad_x(k)  =  bp * u[k];
                out.grad_x1(k) =  0.0;
                out.grad_x2(k) = -bp * u[k];
                out.grad_x3(k) =  0.0;
            }
            break;
        }

        // Case 7: vertex x3
        // We have q = x3.
        // Let  u = || x - q|| /delta. dPE / dx_k =  b' * u_k.dPE / d(x1)_k =  0.
        // dPE / d(x2)_k = 0. dPE / d(x3)_k = -b' * u_k.
        case NodeTriangleRegion::Vertex3:{
            for (int k = 0; k < 3; ++k) {
                out.grad_x(k)  =  bp * u[k];
                out.grad_x1(k) =  0.0;
                out.grad_x2(k) =  0.0;
                out.grad_x3(k) = -bp * u[k];
            }
            break;
        }

        // Degenerate case
        case NodeTriangleRegion::DegenerateTriangle:{
            for (int k = 0; k < 3; ++k) {
                out.grad_x(k) = bp * u[k];
            }

            // Assign reaction to the closest feature
            double d1 = 0.0, d2 = 0.0, d3 = 0.0;
            for (int k = 0; k < 3; ++k) {
                double v1 = dr.closest_point(k) - x1(k);
                double v2 = dr.closest_point(k) - x2(k);
                double v3 = dr.closest_point(k) - x3(k);
                d1 += v1 * v1;
                d2 += v2 * v2;
                d3 += v3 * v3;
            }
            d1 = std::sqrt(d1);
            d2 = std::sqrt(d2);
            d3 = std::sqrt(d3);

            if (d1 <= eps) {
                for (int k = 0; k < 3; ++k) {
                    out.grad_x1(k) = -bp * u[k];
                }
            } else if (d2 <= eps) {
                for (int k = 0; k < 3; ++k) {
                    out.grad_x2(k) = -bp * u[k];
                }
            } else if (d3 <= eps) {
                for (int k = 0; k < 3; ++k) {
                    out.grad_x3(k) = -bp * u[k];
                }
            } else {
                // Closest point on an edge of the degenerate triangle.
                // Try each edge; pick the one whose cross product vanishes.
                Vec3 qa = dr.closest_point;

                // Edge (x1, x2)
                double cross12_sq = 0.0;
                {
                    double a[3], b[3];
                    for (int k = 0; k < 3; ++k) {
                        a[k] = qa(k) - x1(k);
                        b[k] = x2(k) - x1(k);
                    }
                    double c[3];
                    c[0] = a[1]*b[2] - a[2]*b[1];
                    c[1] = a[2]*b[0] - a[0]*b[2];
                    c[2] = a[0]*b[1] - a[1]*b[0];
                    for (int k = 0; k < 3; ++k) cross12_sq += c[k]*c[k];
                }

                if (std::sqrt(cross12_sq) <= eps) {
                    double t = segment_parameter_from_closest_point(qa, x1, x2);
                    for (int k = 0; k < 3; ++k) {
                        out.grad_x1(k) = -bp * (1.0 - t) * u[k];
                        out.grad_x2(k) = -bp * t * u[k];
                    }
                } else {
                    // Edge (x2, x3)
                    double cross23_sq = 0.0;
                    {
                        double a[3], b[3];
                        for (int k = 0; k < 3; ++k) {
                            a[k] = qa(k) - x2(k);
                            b[k] = x3(k) - x2(k);
                        }
                        double c[3];
                        c[0] = a[1]*b[2] - a[2]*b[1];
                        c[1] = a[2]*b[0] - a[0]*b[2];
                        c[2] = a[0]*b[1] - a[1]*b[0];
                        for (int k = 0; k < 3; ++k) cross23_sq += c[k]*c[k];
                    }

                    if (std::sqrt(cross23_sq) <= eps) {
                        double t = segment_parameter_from_closest_point(qa, x2, x3);
                        for (int k = 0; k < 3; ++k) {
                            out.grad_x2(k) = -bp * (1.0 - t) * u[k];
                            out.grad_x3(k) = -bp * t * u[k];
                        }
                    } else {
                        // Must be edge (x3, x1)
                        double t = segment_parameter_from_closest_point(qa, x3, x1);
                        for (int k = 0; k < 3; ++k) {
                            out.grad_x3(k) = -bp * (1.0 - t) * u[k];
                            out.grad_x1(k) = -bp * t * u[k];
                        }
                    }
                }
            }
            break;
        }
    }

    return out;
}