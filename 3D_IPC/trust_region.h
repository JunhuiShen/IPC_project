#pragma once

#include "IPC_math.h"

// Trust-region narrow phase

struct TrustRegionResult {
    double omega = 1.0;   // safe scale in [0, 1]
    double d0    = 0.0;   // initial separation distance
    double M     = 0.0;   // total displacement magnitude
};

// Vertex-triangle pair where every vertex may move.
TrustRegionResult trust_region_vertex_triangle(
        const Vec3& x,  const Vec3& dx,
        const Vec3& x1, const Vec3& dx1,
        const Vec3& x2, const Vec3& dx2,
        const Vec3& x3, const Vec3& dx3,
        double eta = 0.4);

// Edge-edge pair where every endpoint may move.
TrustRegionResult trust_region_edge_edge(
        const Vec3& a1, const Vec3& da1,
        const Vec3& a2, const Vec3& da2,
        const Vec3& b1, const Vec3& db1,
        const Vec3& b2, const Vec3& db2,
        double eta = 0.4);

// Gauss-Seidel substep: only one vertex of the pair moves, with
// displacement delta. Caller is responsible for tracking which vertex the delta belongs to.
TrustRegionResult trust_region_vertex_triangle_gauss_seidel(
        const Vec3& x,  const Vec3& x1, const Vec3& x2, const Vec3& x3,
        const Vec3& delta,
        double eta = 0.4);

TrustRegionResult trust_region_edge_edge_gauss_seidel(
        const Vec3& a1, const Vec3& a2, const Vec3& b1, const Vec3& b2,
        const Vec3& delta,
        double eta = 0.4);
