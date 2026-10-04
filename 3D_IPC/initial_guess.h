#pragma once

#include "physics.h"

#include <vector>

class BroadPhase;

// Move toward xhat using a global mesh-CCD step with a 0.9 TOI safety factor.
std::vector<Vec3> ccd_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, BroadPhase* scratch_broad_phase = nullptr);

// Move deformable nodes toward x + intended_displacement using colored linear CCD.
// Rigid proxies stay fixed; mesh contacts preserve a 1e-8 gap or an existing smaller gap.
// Rebuild swept contacts while reusing optional broad-phase and thread-local storage.
std::vector<Vec3> collision_colored_ccd_initial_guess(
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& intended_displacement,
    const RefMesh& ref_mesh, const SimParams& params, int ccd_iterations,
    BroadPhase* scratch_broad_phase = nullptr);

// Add dt^2 * gravity to xhat, then apply the global CCD guess.
std::vector<Vec3> verlet_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, const SimParams& params, BroadPhase* scratch_broad_phase = nullptr);

// Apply a uniform translation driven by inertia, gravity, pins, and SDF penalties.
std::vector<Vec3> translation_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, const std::vector<Pin>& pins, const SimParams& params);
