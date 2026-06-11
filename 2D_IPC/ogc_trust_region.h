#pragma once

#include "ipc_math.h"

struct TrustRegionResult2D {
    double omega = 1.0;
    double d0 = 0.0;
    double M = 0.0;
};

TrustRegionResult2D trust_region_node_segment_gauss_seidel(
        const Vec2& xi, const Vec2& dxi,
        const Vec2& xj, const Vec2& dxj,
        const Vec2& xk, const Vec2& dxk,
        double eta = 0.4);
