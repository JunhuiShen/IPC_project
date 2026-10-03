#pragma once

#include "physics.h"

#include <vector>

class BroadPhase;

std::vector<Vec3> ccd_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, BroadPhase* scratch_broad_phase = nullptr);

// Collision-only, colored Gauss-Seidel CCD sweeps for deformable vertex meshes.
// The second argument is a displacement, not an absolute target: targets remain
// x + intended_displacement for every sweep. Neither input is modified.
//
// Build padded swept node boxes, primitive BVHs, candidate pairs, and their
// contact-only coloring once. Each sweep processes colors in order and vertices
// within a color in parallel, using one-moving-vertex linear CCD with safety 0.9.
// params.use_parallel=false keeps the CCD sweep team serial.
// Green primitive boxes use params.d_hat padding, as in the solver broad phase.
// No elasticity/bending edges are added. Zero sweeps return x unchanged.
// This filters mesh NT/SS contacts only; it does not enforce SDF or element-
// inversion constraints, or repair initial overlap.
std::vector<Vec3> collision_colored_ccd_initial_guess(
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& intended_displacement,
    const RefMesh& ref_mesh, const SimParams& params, int ccd_iterations);

std::vector<Vec3> verlet_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, const SimParams& params, BroadPhase* scratch_broad_phase = nullptr);

std::vector<Vec3> translation_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, const std::vector<Pin>& pins, const SimParams& params);
