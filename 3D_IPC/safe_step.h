#pragma once

#include "broad_phase.h"
#include "ccd.h"

#include <vector>

struct RefMesh;

namespace safe_step_detail {
// AABB-distance rejections from the current vertex's barrier assembly, in NT
// then SS incidence order. Pair positions and incidence lists must remain fixed
// until safe-step evaluation; supporting-plane rejections do not qualify.
struct VertexAabbRejections {
    std::vector<unsigned char> clear;
    double distance = 0.0;
};
} // namespace safe_step_detail

// Returns gamma_p times the minimum incident NT/SS distance.
double compute_trust_region_bound_for_vertex(int vi, const std::vector<Vec3>& x, const BroadPhase& broad_phase, double gamma_p);

// Cooperative calls share contact checks on the existing OpenMP team and join
// before applying any position update. Serial callers can keep the default.
// Applies the box/CCD-clipped update to one vertex and returns its safe weight.
double per_vertex_safe_step(
    const BroadPhase& broad_phase, std::vector<Vec3>& x, int vi,
    const Vec3& raw_proposed_position, double safety = 0.9, bool clip_ccd = true,
    bool use_ticcd = true, bool use_ogc = false, bool cooperative = false,
    const safe_step_detail::VertexAabbRejections* rejections = nullptr);

// Returns a safe alpha for translating rigid body rb by alpha * dx using linear CCD.
double per_rigid_body_translation_safe_step(const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<Vec3>& x, int rb, const Vec3& dx, double safety = 0.9, bool cooperative = false);

// Bounds the raw-sign q_current-to-q_target arc by theta_bound about q_box_anchor.
Vec4 bound_quaternion(const Vec4& q_box_anchor, const Vec4& q_current, const Vec4& q_target, double theta_bound);

// Returns a safe alpha for rotating rigid body rb from q_current to q_target.
double per_rigid_body_rotation_safe_step(const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<Vec3>& x, int rb, const Vec3& x_com, const Vec4& q_current, const Vec4& q_target, double safety = 0.9, bool cooperative = false);

namespace safe_step_detail {
// Read-only single-contact tests, including the original swept-AABB rejection.
// The caller must hold all four pair positions fixed until the test finishes.
CCDResult node_triangle_vertex_ccd(const NodeTrianglePair& pair, int dof, int vertex,
    const std::vector<Vec3>& positions, const Vec3& displacement, bool use_ticcd);
CCDResult segment_segment_vertex_ccd(const SegmentSegmentPair& pair, int dof, int vertex,
    const std::vector<Vec3>& positions, const Vec3& displacement, bool use_ticcd);
} // namespace safe_step_detail
