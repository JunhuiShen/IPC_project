#pragma once

#include "ipc_math.h"
#include "mesh.h"

Vec2 local_spring_grad(int i, const Vec& x, double k_spring,
                       const RefMesh& ref_mesh);

Mat2 local_spring_hess(int i, const Vec& x, double k_spring,
                       const RefMesh& ref_mesh);
