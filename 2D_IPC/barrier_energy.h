#pragma once

#include "ipc_math.h"

namespace physics {

double barrier_energy(double d, double dhat);
double barrier_grad(double d, double dhat);
double barrier_hess(double d, double dhat);

Vec2 local_barrier_grad(int who, const Vec& x, int node, int seg0, int seg1, double dhat);
Mat2 local_barrier_hess(int who, const Vec& x, int node, int seg0, int seg1, double dhat);

}
