#pragma once

#include "IPC_math.h"

#include <array>
#include <atomic>
#include <functional>
#include <vector>

class BroadPhase;
struct RefMesh;

// Returns 1 when the rotating segment does not collide, 0 for contact at the
// start of the step, and eta times the first time of impact otherwise.
double segment_segment_rb_rotation_safe_step(
        const Vec3& x0, const Vec3& x1,
        const Vec3& x_com,
        const Vec4& q_new, const Vec4& q_n,
        const Vec3& x2, const Vec3& x3,
        double eta = 0.9);

// Returns 1 when the rotating point does not collide, 0 for contact at the
// start of the step, and eta times the first time of impact otherwise.
double point_triangle_rb_rotation_safe_step(
        const Vec3& x,
        const Vec3& x_com,
        const Vec4& q_new, const Vec4& q_n,
        const Vec3& x2, const Vec3& x3, const Vec3& x4,
        double eta = 0.9);

// Returns gamma_p times the minimum incident NT/SS distance.
double compute_trust_region_bound_for_vertex(int vi, const std::vector<Vec3>& x, const BroadPhase& broad_phase, double gamma_p);

// Clips targets to cached boxes, then limits x += alpha*dx by CCD or the OGC bound.
void per_vertex_safe_step(const BroadPhase& broad_phase, std::vector<Vec3>& x, const std::function<Vec3(int)>& x_new_fn, double safety = 0.9, bool clip_ccd = true, bool use_ticcd = true, bool use_ogc = false, const std::vector<std::vector<int>>* color_groups = nullptr, std::atomic<int>* clip_count = nullptr);

// Returns a safe alpha for translating rigid body rb by alpha * dx using linear CCD.
double per_rigid_body_translation_safe_step(const RefMesh& ref_mesh, const std::vector<std::array<int, 2>>& edges, const std::vector<Vec3>& x, int rb, const Vec3& dx, double safety = 0.9);

// Returns a safe alpha for rotating rigid body rb from q_current to q_new.
double per_rigid_body_omega_safe_step(
    const RefMesh& ref_mesh,
    const std::vector<std::array<int, 2>>& edges,
    const std::vector<Vec3>& x,
    int rb,
    const Vec3& x_com,
    const Vec4& q_n,
    const Vec3& omega_current,
    const Vec3& delta_omega,
    double dt,
    double safety = 0.9);
