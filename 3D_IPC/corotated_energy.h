#pragma once

#include "IPC_math.h"

// Corotated cache
struct CorotatedCache32 {
    Mat22 S;
    Mat22 SInv;
    Mat32 R;
    Mat22 FTFinv;
    Mat32 FFTFInv;
    Mat33 FFTFInvFT;
    double J;
    double traceS;
};

CorotatedCache32 buildCorotatedCache(const Mat32& F);

double PsiCorotated32(const CorotatedCache32& cache, const Mat32& F, double mu, double lambda);
Mat32  PCorotated32(const CorotatedCache32& cache, const Mat32& F, double mu, double lambda);
void   dPdFCorotated32(const CorotatedCache32& cache, double mu, double lambda, Mat66& dPdF);

double corotated_energy(double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);

// Shape-function gradients
using ShapeGrads = std::array<Vec2, 3>;
ShapeGrads shape_function_gradients(const Mat22& Dm_inv);

// Single-node gradient: returns the 3-vector force on node only.
Vec3 corotated_node_gradient(const Mat32& P, double ref_area, const ShapeGrads& gradN, int node);

// Single-node self hessian: returns d(g_node)/d(x_node), i.e. the 3x3 diagonal block.
Mat33 corotated_node_hessian(const Mat66& dPdF, double ref_area, const ShapeGrads& gradN, int node);
