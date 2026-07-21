#pragma once

#include "IPC_math.h"

#include <vector>

void build_xhat(std::vector<Vec3>& xhat, const std::vector<Vec3>& x, const std::vector<Vec3>& v, double dt);

void update_velocity(std::vector<Vec3>& v, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xold, double dt);
