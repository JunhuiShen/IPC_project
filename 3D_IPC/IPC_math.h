#pragma once
#include "corotated_energy.h"
#include <array>

Mat33 matrix3d_inverse(const Mat33& H);

TriangleDef ZeroTriangleDef();

TriangleDef add_scale(const TriangleDef& a, const TriangleDef& b, double s);

Vec9 flatten_def(const TriangleDef& def);

Vec9 flatten_gradient(const std::array<Vec3, 3>& g);

double get_dof(const TriangleDef& def, int node, int comp);

void set_dof(TriangleDef& def, int node, int comp, double value);