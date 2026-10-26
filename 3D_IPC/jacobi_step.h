#pragma once

#include "physics.h"

#include <vector>

std::vector<Vec3> jacobi_step(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat, const std::vector<Vec3>& v, const std::vector<Vec3>* previous_positions = nullptr);
