#include "IPC_math.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

Mat33 matrix3d_inverse(const Mat33& H) {

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

TriangleDef ZeroTriangleDef() {
    TriangleDef out;
    out.x[0].setZero();
    out.x[1].setZero();
    out.x[2].setZero();
    return out;
}

TriangleDef add_scale(const TriangleDef& a, const TriangleDef& b, double s) {
    TriangleDef out;
    for (int i = 0; i < 3; ++i) out.x[i] = a.x[i] + s * b.x[i];
    return out;
}

Vec9 flatten_def(const TriangleDef& def) {
    Vec9 out;
    for (int i = 0; i < 3; ++i) out.segment<3>(3 * i) = def.x[i];
    return out;
}

double get_dof(const TriangleDef& def, int node, int comp) {
    return def.x[node](comp);
}

void set_dof(TriangleDef& def, int node, int comp, double value) {
    def.x[node](comp) = value;
}

double clamp_scalar(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
}

Vec3 segment_closest_point(const Vec3& x, const Vec3& a, const Vec3& b, double& t) {
    const Vec3 ab = b - a;
    const double ab2 = ab.dot(ab);

    if (ab2 <= 0.0) {
        t = 0.0;
        return a;
    }

    t = (x - a).dot(ab) / ab2;
    t = clamp_scalar(t, 0.0, 1.0);
    return a + t * ab;
}

std::array<double, 3> triangle_plane_barycentric_coordinates(const Vec3& x, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps){
    const Vec3 e1 = x2 - x1;
    const Vec3 e2 = x3 - x1;
    const Vec3 r  = x - x1;

    const double a11 = e1.dot(e1);
    const double a12 = e1.dot(e2);
    const double a22 = e2.dot(e2);

    const double b1 = r.dot(e1);
    const double b2 = r.dot(e2);

    const double det = a11 * a22 - a12 * a12;

    if (std::abs(det) <= eps) {
        return {{0.0, 0.0, 0.0}};
    }

    const double alpha = ( b1 * a22 - b2 * a12) / det;
    const double beta  = (-b1 * a12 + b2 * a11) / det;

    const double lambda1 = 1.0 - alpha - beta;
    const double lambda2 = alpha;
    const double lambda3 = beta;

    return {{lambda1, lambda2, lambda3}};
}