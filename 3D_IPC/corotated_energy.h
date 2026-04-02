#pragma once

#include "IPC_math.h"

// Cache for the shared eigendecomposition of F^T * F where F = RS, and J = det(S)
struct CorotatedCache32 {
    Mat22 S;
    Mat22 SInv;
    Mat32 R;
    Mat22 FTFinv;
    double J;
    double traceS;
};

// Build the cache from a 3x2 deformation gradient.
CorotatedCache32 buildCorotatedCache(const Mat32& F);

// ---------- Low-level (work on F directly) ----------------------------------

double       PsiCorotated32 (const Mat32& F, double mu, double lambda);
Mat32        PCorotated32 (const Mat32& F, double mu, double lambda);
void         dPdFCorotated32 (const Mat32& F, double mu, double lambda, Mat66& dPdF);

// Cached variants with the same results but no redundant eigendecomposition.
double       PsiCorotated32  (const CorotatedCache32& cache, const Mat32& F, double mu, double lambda);
Mat32        PCorotated32    (const CorotatedCache32& cache, const Mat32& F, double mu, double lambda);
void         dPdFCorotated32 (const CorotatedCache32& cache, const Mat32& F, double mu, double lambda, Mat66& dPdF);

// ---------- High-level  ----------------------------------

double corotated_energy(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda);
double corotated_energy(double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);;

std::array<Vec3,3>  corotated_node_gradient(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda);
std::array<Vec3,3>  corotated_node_gradient(double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);

Mat99 corotated_node_hessian(const TriangleRest& rest, const TriangleDef& def, double mu, double lambda);
Mat99 corotated_node_hessian(double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);

// Cached high-level overloads as we pass a pre-built cache to avoid redundant work
double  corotated_energy (const CorotatedCache32& cache, double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);
std::array<Vec3,3>  corotated_node_gradient(const CorotatedCache32& cache, double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);
Mat99 corotated_node_hessian (const CorotatedCache32& cache, double ref_area, const Mat22& Dm_inv, const TriangleDef& def, double mu, double lambda);