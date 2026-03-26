#include "IPC_math.h"
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

Vec9 flatten_gradient(const std::array<Vec3, 3>& g) {
    Vec9 out;
    for (int i = 0; i < 3; ++i) out.segment<3>(3 * i) = g[i];
    return out;
}

double get_dof(const TriangleDef& def, int node, int comp) {
    return def.x[node](comp);
}

void set_dof(TriangleDef& def, int node, int comp, double value) {
    def.x[node](comp) = value;
}
