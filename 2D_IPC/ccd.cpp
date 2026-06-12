#include "ccd.h"
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>

namespace step_filter::ccd {

    using namespace math;

    bool point_segment_2d(const Vec2& x1, const Vec2& dx1,
                          const Vec2& x2, const Vec2& dx2,
                          const Vec2& x3, const Vec2& dx3,
                          double& t_out, double eps) {
        Vec2 x21  = sub(x1, x2),  x32  = sub(x3, x2);
        Vec2 dx21 = sub(dx1, dx2), dx32 = sub(dx3, dx2);

        double a = cross(dx32, dx21);
        double b = cross(dx32, x21) + cross(x32, dx21);
        double c = cross(x32, x21);

        double t_candidates[2];
        int num_roots = 0;

        if (std::fabs(a) < eps) {
            if (std::fabs(b) < eps) return false;
            double t = -c / b;
            if (t >= 0.0 && t <= 1.0) t_candidates[num_roots++] = t;
        } else {
            double D = b * b - 4.0 * a * c;
            if (D < 0.0) return false;

            double sqrtD = std::sqrt(std::max(D, 0.0));
            double s = (b >= 0.0) ? 1.0 : -1.0;
            double q = -0.5 * (b + s * sqrtD);

            double t1 = q / a, t2 = c / q;
            if (t1 >= 0.0 && t1 <= 1.0) t_candidates[num_roots++] = t1;
            if (t2 >= 0.0 && t2 <= 1.0) t_candidates[num_roots++] = t2;
        }

        if (num_roots == 0) return false;

        std::sort(t_candidates, t_candidates + num_roots);

        constexpr double segment_tol = 1e-9;
        for (int r = 0; r < num_roots; ++r) {
            double t_star = t_candidates[r];

            Vec2 x1t = add(x1, scale(dx1, t_star));
            Vec2 x2t = add(x2, scale(dx2, t_star));
            Vec2 x3t = add(x3, scale(dx3, t_star));

            Vec2 seg = sub(x3t, x2t);
            Vec2 rel = sub(x1t, x2t);

            double seg_len2 = norm2(seg);
            if (seg_len2 < eps) continue;

            double s_param = dot(rel, seg) / seg_len2;
            if (s_param < -segment_tol || s_param > 1.0 + segment_tol)
                continue;

            t_out = t_star;
            return true;
        }

        return false;
    }

    bool point_segment_2d_rb_rotation(const Eigen::Vector2d& x, const Eigen::Vector2d& x_com, const double& theta_n, const double& theta_new, const Eigen::Vector2d& x0, const Eigen::Vector2d& x1, double& step) {
        // We assume counterclockwise rotation from theta_n to theta_new.
        
        step = 0.0;

        Eigen::Vector2d dx = x - x_com;
        double cos_n = std::cos(theta_n);
        double sin_n = std::sin(theta_n);

        Eigen::Vector2d r{cos_n * dx.x() + sin_n * dx.y(), -sin_n * dx.x() + cos_n * dx.y()};

        Eigen::Vector2d d = x1 - x0;
        double seg_len = d.norm();
        if (seg_len < 1e-14) {
            std::cerr << "Warning: degenerate segment in point_segment_2d_rotation(): seg_len = " << seg_len
                      << ", threshold = 1e-14\n";
            return false;
        }

        Eigen::Vector2d d_hat = d / seg_len;

        double A = r.x() * d_hat.y() - r.y() * d_hat.x();
        double B = -r.x() * d_hat.x() - r.y() * d_hat.y();
        double C = (x_com.x() - x0.x()) * d_hat.y()
                 - (x_com.y() - x0.y()) * d_hat.x();

        double amplitude = std::sqrt(A*A + B*B);
        if (std::abs(C) > amplitude + 1e-14) return false;

        double phi = std::atan2(B, A);
        double arccos_val = std::acos(std::clamp(-C / amplitude, -1.0, 1.0)); // clamp to prevent numerical errors.

        double theta_candidates[2] = { phi + arccos_val, phi - arccos_val };
        
        double best_s = std::numeric_limits<double>::infinity();
        double dtheta = theta_new - theta_n;
        if (std::abs(dtheta) < 1e-14){ 
            std::cerr << "Warning: theta_new - theta_n < threshold\n";
            return false;
        }

        for (double theta_star : theta_candidates) {
            double s = (theta_star - theta_n) / dtheta;
            if (s < 0.0 || s > 1.0) continue;

            double theta_s = theta_n + s * dtheta;
            Eigen::Vector2d x_s{
                std::cos(theta_s) * r.x() - std::sin(theta_s) * r.y(),
                std::sin(theta_s) * r.x() + std::cos(theta_s) * r.y()
            };
            x_s += x_com;

            double t_star = (x_s - x0).dot(d) / (seg_len * seg_len);
            if (t_star < 0.0 || t_star > 1.0) continue;

            if (s < best_s) best_s = s;
        }

        if (best_s == std::numeric_limits<double>::infinity()) return false;
        step = best_s;
        return true;
    }

    double safe_step_rb_rotation(const Eigen::Vector2d& x, const Eigen::Vector2d& x_com, const double& theta_n, const double& theta_new, const Eigen::Vector2d& x0, const Eigen::Vector2d& x1, const double eta) {
        double step;
        if (!point_segment_2d_rb_rotation(x, x_com, theta_n, theta_new, x0, x1, step)) {
            return 1.0;
        }
        return (step <= 1e-12) ? 0.0 : eta * step;
    }


    double safe_step(const Vec2& x1, const Vec2& dx1,
                     const Vec2& x2, const Vec2& dx2,
                     const Vec2& x3, const Vec2& dx3,
                     double eta) {
        double t_hit;
        if (!point_segment_2d(x1, dx1, x2, dx2, x3, dx3, t_hit))
            return 1.0;

        return (t_hit <= 1e-12) ? 0.0 : eta * t_hit;
    }

} // namespace step_filter::ccd
