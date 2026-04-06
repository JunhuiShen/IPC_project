#pragma once

#include "IPC_math.h"

#include <limits>

struct CCDResult {
    bool has_candidate_time = false;
    bool collision = false;
    bool coplanar_entire_step = false;
    bool parallel_or_no_crossing = false;
    double t = std::numeric_limits<double>::quiet_NaN();
};

CCDResult node_triangle_only_one_node_moves(const Vec3& x, const Vec3& dx, const Vec3& x1, const Vec3& x2, const Vec3& x3, double eps = 1.0e-12);

CCDResult segment_segment_only_one_node_moves(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps = 1.0e-12);
