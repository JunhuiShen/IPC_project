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

// Linear NT CCD (one moving node)
CCDResult node_triangle_only_one_node_moves(
        const Vec3& x,  const Vec3& dx,
        const Vec3& x1, const Vec3& dx1,
        const Vec3& x2, const Vec3& dx2,
        const Vec3& x3, const Vec3& dx3,
        double eps = 1.0e-12);

// Linear SS CCD (one moving node)
CCDResult segment_segment_only_one_node_moves(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& x3, const Vec3& x4, double eps = 1.0e-12);

// General NT/SS CCD: all vertices may move, finds exact earliest TOI and
// returns 1.0 when no collision occurs in [0,1].
double node_triangle_general_ccd(const Vec3& x, const Vec3& dx, const Vec3& x1, const Vec3& dx1,
                                 const Vec3& x2, const Vec3& dx2, const Vec3& x3, const Vec3& dx3,
                                 double eps1 = 1e-12, double eps2 = 1e-10);

double segment_segment_general_ccd(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& dx2,
                                   const Vec3& x3, const Vec3& dx3, const Vec3& x4, const Vec3& dx4,
                                   double eps1 = 1e-12, double eps2 = 1e-10);
