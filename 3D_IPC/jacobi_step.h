#pragma once

#include "physics.h"

#include <vector>

// The mixed solver may hold rigid proxies fixed during particle assembly;
// their generalized-coordinate proposals are assembled separately.
std::vector<Vec3> jacobi_step(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat, const std::vector<Vec3>& v, const std::vector<Vec3>* previous_positions = nullptr, bool hold_rigid_proxies = false);
