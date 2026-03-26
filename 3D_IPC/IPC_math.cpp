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
