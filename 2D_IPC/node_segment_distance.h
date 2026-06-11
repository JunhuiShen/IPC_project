#pragma once

#include "ipc_math.h"

namespace physics {

double node_segment_distance(const Vec2& xi, const Vec2& xj, const Vec2& xjp1,
                             double& t, Vec2& p, Vec2& r);

}
