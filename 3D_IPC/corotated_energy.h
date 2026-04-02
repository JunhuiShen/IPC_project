#pragma once

#include "IPC_math.h"

double corotated_energy(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda);
double corotated_energy(double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);

std::array<Vec3, 3> corotated_node_gradient(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda);
std::array<Vec3, 3> corotated_node_gradient(double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);

Mat99 corotated_node_hessian(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda);
Mat99 corotated_node_hessian(double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);