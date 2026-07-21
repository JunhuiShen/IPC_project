#pragma once

#include "physics.h"

#include <vector>

class BroadPhase;

std::vector<Vec3> ccd_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, BroadPhase* scratch_broad_phase = nullptr);

std::vector<Vec3> verlet_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, const SimParams& params, BroadPhase* scratch_broad_phase = nullptr);

std::vector<Vec3> translation_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, const std::vector<Pin>& pins, const SimParams& params);
