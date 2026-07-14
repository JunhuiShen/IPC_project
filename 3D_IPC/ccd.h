#pragma once

#include "IPC_math.h"
#include "algebra/algebra.h"

#include <limits>

struct CCDResult {
    bool collision = false;
    double t = std::numeric_limits<double>::quiet_NaN();
};

// One-moving-node NT CCD. Dispatches based on `use_ticcd`:
//   true  (default) -> Tight-Inclusion CCD library (conservative, robust)
//   false           -> self-written closed-form linear CCD (faster, exact
//                       when only one of the four vertices moves).
CCDResult node_triangle_only_one_node_moves(
        const Vec3& x,  const Vec3& dx,
        const Vec3& x1, const Vec3& dx1,
        const Vec3& x2, const Vec3& dx2,
        const Vec3& x3, const Vec3& dx3,
        double eps = 1.0e-12,
        bool use_ticcd = true);

// One-moving-node SS CCD. Same dispatch semantics as above.
CCDResult segment_segment_only_one_node_moves(
        const Vec3& x1, const Vec3& dx1,
        const Vec3& x2, const Vec3& x3, const Vec3& x4,
        double eps = 1.0e-12,
        bool use_ticcd = true);

// General NT/SS CCD: all vertices may move. Backed by Tight-Inclusion CCD.
// Returns the earliest time of impact in [0, 1], or 1.0 when no collision
// occurs over the step.
double node_triangle_general_ccd(const Vec3& x, const Vec3& dx, const Vec3& x1, const Vec3& dx1,
                                 const Vec3& x2, const Vec3& dx2, const Vec3& x3, const Vec3& dx3);

double segment_segment_general_ccd(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& dx2,
                                   const Vec3& x3, const Vec3& dx3, const Vec3& x4, const Vec3& dx4);

///////////////////// Rigid Body CCD ////////////////////////////

// Segment [x0, x1] rotating rigidly about x_com from orientation q_n to q_new
// against the fixed segment [x2, x3]. Returns the earliest time of impact `s`
// in [0, 1], or false if none.
bool segment_segment_rb_rotation_ccd(
        const Vec3& x0, const Vec3& x1,
        const Vec3& x_com,
        const Vec4& q_new, const Vec4& q_n,
        const Vec3& x2, const Vec3& x3,
        double& s);

// Particle x rotating rigidly about x_com from orientation q_n to q_new
// against the fixed triangle (x2, x3, x4). Returns the earliest time of
// impact `s` in [0, 1], or false if none.
bool point_triangle_rb_rotation_ccd(
        const Vec3& x,
        const Vec3& x_com,
        const Vec4& q_new, const Vec4& q_n,
        const Vec3& x2, const Vec3& x3, const Vec3& x4,
        double& s);
