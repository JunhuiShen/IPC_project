#pragma once

#include "broad_phase.h"

#include <functional>
#include <vector>

struct RefMesh;

// Returns gamma_p times the minimum incident NT/SS distance.
double compute_trust_region_bound_for_vertex(int vi, const std::vector<Vec3>& x, const BroadPhase& broad_phase, double gamma_p);

// Clips targets to cached boxes, then limits x += alpha*dx by CCD or the OGC bound.
void per_vertex_safe_step(const BroadPhase& broad_phase, std::vector<Vec3>& x, const std::function<Vec3(int)>& x_new_fn, double safety = 0.9, bool clip_ccd = true, bool use_ticcd = true, bool use_ogc = false, const std::vector<std::vector<int>>* color_groups = nullptr);

// Mixed-solver-only variant that applies the same box/CCD-clipped update to
// one independently deformable vertex. Keeping this separate leaves the
// original deformable-only per_vertex_safe_step path unchanged.
void mixed_deformable_vertex_safe_step(const BroadPhase& broad_phase, std::vector<Vec3>& x, int vi, const Vec3& raw_proposed_position, double safety = 0.9, bool clip_ccd = true, bool use_ticcd = true, bool use_ogc = false);

// Returns a safe alpha for translating rigid body rb by alpha * dx using linear CCD.
double per_rigid_body_translation_safe_step(const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<Vec3>& x, int rb, const Vec3& dx, double safety = 0.9);

// Bounds the raw-sign q_current-to-q_target arc by theta_bound about q_box_anchor.
Vec4 bound_quaternion(const Vec4& q_box_anchor, const Vec4& q_current, const Vec4& q_target, double theta_bound);

// Returns a safe alpha for rotating rigid body rb from q_current to q_target.
double per_rigid_body_rotation_safe_step(const RefMesh& ref_mesh, const BroadPhase::Cache& bp_cache, const std::vector<int>& nt_pair_indices, const std::vector<int>& ss_pair_indices, const std::vector<Vec3>& x, int rb, const Vec3& x_com, const Vec4& q_current, const Vec4& q_target, double safety = 0.9);
