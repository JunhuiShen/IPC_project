#pragma once

#include "ipc_math.h"
#include "mesh.h"

#include <vector>

namespace physics {

    Vec2 local_grad_no_barrier(int i, const Vec &x, const Vec &xhat, const Vec &xpin,
                               const std::vector<double> &mass,
                               const RefMesh& ref_mesh,
                               const std::vector<char> &is_pinned,
                               double dt, double k_spring, const Vec2 &g_accel);

    Mat2 local_hess_no_barrier(int i, const Vec &x,
                               const std::vector<double> &mass,
                               const RefMesh& ref_mesh,
                               const std::vector<char> &is_pinned,
                               double dt, double k_spring);

} // namespace physics
