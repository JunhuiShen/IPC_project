#pragma once

#include "IPC_math.h"

// Local quaternion wrappers
Vec4 quaternion_multiply(const Vec4& a, const Vec4& b);
Vec4 quaternion_conjugate(const Vec4& quat);
Vec4 quaternion_inverse(const Vec4& quat);
Vec4 quaternion_normalize(const Vec4& quat);
Vec4 quaternion_align_sign(const Vec4& quat, const Vec4& reference);

// These rotation helpers expect a unit quaternion
Vec3 quaternion_rotate(const Vec4& quat, const Vec3& vector);
Vec3 quaternion_inverse_rotate(const Vec4& quat, const Vec3& vector);

// Quaternion product tensor: (a * b)_alpha = sum_{beta,gamma} Q(alpha,beta,gamma) a_beta b_gamma
int quaternion_product_tensor(int alpha, int beta, int gamma);
