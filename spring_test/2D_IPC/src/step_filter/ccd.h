#pragma once

#include "step_filter.h"
#include <cstddef>

// ======================================================
// CCD step filter
//
// Solves the quadratic colinearity condition for
// point-segment 2D CCD, returns eta * t_hit as
// the safe fraction of the Newton step.
// ======================================================

namespace step_filter::ccd {

    // Global CCD statistics
    inline std::size_t total_tests = 0;
    inline std::size_t total_collisions = 0;

    inline void reset_stats() {
        total_tests = 0;
        total_collisions = 0;
    }

    // Low-level free functions
    bool point_segment_2d(const Vec2& x1, const Vec2& dx1,
                          const Vec2& x2, const Vec2& dx2,
                          const Vec2& x3, const Vec2& dx3,
                          double& t_out, double eps = 1e-12);

    double safe_step(const Vec2& x1, const Vec2& dx1,
                     const Vec2& x2, const Vec2& dx2,
                     const Vec2& x3, const Vec2& dx3,
                     double eta = 0.9);

} // namespace step_filter::ccd

class CCDFilter : public StepFilter {
public:
    double compute_safe_step(int who_global, const Vec2& dx,
                             const Vec& x_global,
                             const std::vector<physics::NodeSegmentPair>& candidates,
                             double eta) override;
};
