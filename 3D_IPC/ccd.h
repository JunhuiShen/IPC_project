#pragma once

#include "IPC_math.h"

#include <array>
#include <limits>
#include <memory>

struct CCDResult {
    bool collision = false;
    double t = std::numeric_limits<double>::quiet_NaN();
};

// Versioned linear kernels. Tight-Inclusion dispatch below is independent.
CCDResult node_triangle_linear_ccd_v1(
    const Vec3&, const Vec3&, const Vec3&, const Vec3&,
    const Vec3&, const Vec3&, const Vec3&, const Vec3&, double eps = 1e-12);
CCDResult node_triangle_linear_ccd_v2(
    const Vec3&, const Vec3&, const Vec3&, const Vec3&,
    const Vec3&, const Vec3&, const Vec3&, const Vec3&, double eps = 1e-12);
CCDResult segment_segment_linear_ccd_v1(
    const Vec3&, const Vec3&, const Vec3&, const Vec3&, const Vec3&, double eps = 1e-12);

// Cold-path checks for a represented NT (point, a, b, c; vertex_face=true) or
// SS (a, b, c, d; false) configuration. Original binary64 coordinates are
// converted before arithmetic; no positive gap is clamped to a tolerance.
// Returns -1/0/+1 according to gap < / == / > distance. distance must be finite
// and nonnegative; comparison against zero distinguishes true intersections.
int compare_contact_distance_exact(const std::array<Vec3, 4>& positions,
                                   bool vertex_face, double distance);

// Check the actual rounded endpoint against gap_end >= min(gap_start, floor).
// A true initial intersection returns false: this is not an overlap-repair or
// contact-side inference policy. floor must be finite and strictly positive.
// This endpoint test alone does not establish a collision-free path.
bool contact_preserves_separation_exact(const std::array<Vec3, 4>& start,
                                        const std::array<Vec3, 4>& end,
                                        bool vertex_face, double floor);

// CCD for the segment between two ACTUAL represented endpoint configurations,
// using exact subtraction rather than rounded end-start displacements. At most
// one vertex may differ. True initial intersections still return t=0; ordinary
// linear-CCD future-event semantics below apply. All inputs must be finite.
CCDResult single_vertex_ccd_between_exact(const std::array<Vec3, 4>& start,
                                          const std::array<Vec3, 4>& end,
                                          bool vertex_face);

enum class ExactContactStepResult { Safe, Unsafe, InitialContact };

// Combined cold-path check: distinguish initial contact, preserve the endpoint
// gap >= min(initial gap, floor), then validate the actual represented path.
// Uses exact integer predicates, with rational fallback for coplanar paths.
// All coordinates must be finite, floor finite and positive, and at most one
// vertex may change; invalid inputs throw std::invalid_argument. InitialContact
// is not permission to separate without a known contact side/history.
ExactContactStepResult exact_contact_step_result(const std::array<Vec3, 4>& start,
    const std::array<Vec3, 4>& end, bool vertex_face, double floor);

// Immutable exact start snapshot for repeated backtracking trials of ONE
// vertex. Reuse only while all four start positions and the floor stay fixed;
// each test receives the actual represented position of moving_dof in [0,3].
// Owns its snapshot, so later caller-array changes do not update this context.
// Constructor/test validation and results match exact_contact_step_result.
class PreparedExactContactStep {
public:
    PreparedExactContactStep(const std::array<Vec3, 4>& start, bool vertex_face,
                             double floor, int moving_dof);
    ~PreparedExactContactStep();
    PreparedExactContactStep(PreparedExactContactStep&&) noexcept;
    PreparedExactContactStep& operator=(PreparedExactContactStep&&) noexcept;
    PreparedExactContactStep(const PreparedExactContactStep&) = delete;
    PreparedExactContactStep& operator=(const PreparedExactContactStep&) = delete;

    // Calling test on a moved-from context throws std::logic_error.
    ExactContactStepResult test(const Vec3& endpoint) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// One-moving-node NT CCD. Dispatches based on `use_ticcd`:
//   true  (default) -> Tight-Inclusion CCD library (conservative, robust)
//   false           -> closed-form linear CCD in a scaled local frame;
//                       ambiguous/degenerate cases use exact rational arithmetic.
// Linear mode never calls TICCD.
// At most one of the four displacements may be nonzero in linear mode.
// eps sets dimensionless time/membership ambiguity tolerances, not a distance pad.
// A true initial intersection returns t=0. Positive initial gaps, even below
// 1e-10, are not initial contact. Far-start future events retain a fixed 1e-10
// world-space boundary tolerance; starts within that band use exact membership
// throughout the motion so separating near-contact queries can escape.
CCDResult node_triangle_only_one_node_moves(
        const Vec3& x,  const Vec3& dx,
        const Vec3& x1, const Vec3& dx1,
        const Vec3& x2, const Vec3& dx2,
        const Vec3& x3, const Vec3& dx3,
        double eps = 1.0e-12,
        bool use_ticcd = true, bool original = false);

// One-moving-node SS CCD. Same dispatch semantics as above.
CCDResult segment_segment_only_one_node_moves(const Vec3& x1, const Vec3& dx1,
        const Vec3& x2, const Vec3& x3, const Vec3& x4,
        double eps = 1.0e-12, bool use_ticcd = true, bool original = false);

// Translating-edge SS CCD. Both endpoints of [x1, x2] must have the same
// displacement (`dx1 == dx2`), while [x3, x4] remains fixed.
// Uses the same independent local-frame and exact predicates as above.
CCDResult segment_segment_same_displacement_linear_ccd(const Vec3& x1, const Vec3& dx1,
        const Vec3& x2, const Vec3& dx2,
        const Vec3& x3, const Vec3& x4,
        double eps = 1.0e-12, bool original = false);

// General NT/SS CCD: all vertices may move. Backed by Tight-Inclusion CCD.
// Returns the earliest time of impact in [0, 1], or 1.0 when no collision
// occurs over the step.
double node_triangle_general_ccd(const Vec3& x, const Vec3& dx, const Vec3& x1, const Vec3& dx1,
                                 const Vec3& x2, const Vec3& dx2, const Vec3& x3, const Vec3& dx3);

double segment_segment_general_ccd(const Vec3& x1, const Vec3& dx1, const Vec3& x2, const Vec3& dx2,
                                   const Vec3& x3, const Vec3& dx3, const Vec3& x4, const Vec3& dx4);

///////////////////// Rigid Body CCD ////////////////////////////

namespace ccd_detail {
inline constexpr double rigid_rotation_epsilon = 1.0e-10;
// The common first rejection in both rotation CCD kernels, evaluated using
// their raw quaternion product (without an additional normalization).
bool negligible_rigid_rotation(const Vec4& q_new, const Vec4& q_n);
}

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
