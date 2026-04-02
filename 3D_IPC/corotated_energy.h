#pragma once

#include "IPC_math.h"

struct CorotatedCache32 {
    Mat22 S;
    Mat22 SInv;
    Mat32 R;
    Mat22 FTFinv;
    double J;
    double traceS;
};

CorotatedCache32 buildCorotatedCache(const Mat32& F);

double PsiCorotated32(const CorotatedCache32& cache, const Mat32& F, double mu, double lambda);
Mat32  PCorotated32(const CorotatedCache32& cache, const Mat32& F, double mu, double lambda);
void   dPdFCorotated32(const CorotatedCache32& cache, const Mat32& F, double mu, double lambda, Mat66& dPdF);

double   corotated_energy  (double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);
std::array<Vec3, 3> corotated_node_gradient(const CorotatedCache32& cache, double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);
Mat99 corotated_node_hessian (const CorotatedCache32& cache, double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);

