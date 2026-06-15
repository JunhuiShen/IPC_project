#pragma once

#include "ipc_math.h"
#include "physics.h"

double spring_energy(int edge_index, const Vec& x, double k_spring,
                     const RefMesh& ref_mesh);

Vec2 local_spring_grad(int i, const Vec& x, double k_spring,
                       const RefMesh& ref_mesh);

Mat2 local_spring_hess(int i, const Vec& x, double k_spring,
                       const RefMesh& ref_mesh);
