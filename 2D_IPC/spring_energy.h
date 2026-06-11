#pragma once

#include "ipc_math.h"
#include <vector>

namespace physics {

Vec2 local_spring_grad(int i, const Vec& x, double k,
                       const std::vector<double>& rest_lengths, int rest_offset);

Mat2 local_spring_hess(int i, const Vec& x, double k,
                       const std::vector<double>& rest_lengths, int rest_offset);

}
