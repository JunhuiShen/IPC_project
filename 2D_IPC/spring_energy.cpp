#include "spring_energy.h"
#include <cmath>

namespace physics {
using namespace math;

Vec2 local_spring_grad(int i, const Vec& x, double k_spring,
                       const RefMesh& ref_mesh) {
    Vec2 g_i{0.0, 0.0};

    for (int edge_index : ref_mesh.incident_edges[i]) {
        const auto [a, b] = ref_mesh.edges[edge_index];
        const double Lref = ref_mesh.rest_lengths[edge_index];
        Vec2 xa = get_xi(x, a), xb = get_xi(x, b);
        double dx = xb.x - xa.x, dy = xb.y - xa.y;
        double ell = std::sqrt(dx * dx + dy * dy);
        if (ell < 1e-12) continue;

        double coeff = k_spring / Lref * (ell - Lref) / ell;
        double sgn = (i == b) ? +1.0 : -1.0;

        g_i.x += sgn * coeff * dx;
        g_i.y += sgn * coeff * dy;
    }

    return g_i;
}

Mat2 local_spring_hess(int i, const Vec& x, double k_spring,
                       const RefMesh& ref_mesh) {
    Mat2 H_ii{0.0, 0.0, 0.0, 0.0};

    for (int edge_index : ref_mesh.incident_edges[i]) {
        const auto [a, b] = ref_mesh.edges[edge_index];
        const double Lref = ref_mesh.rest_lengths[edge_index];
        Vec2 xa = get_xi(x, a), xb = get_xi(x, b);
        double dx = xb.x - xa.x, dy = xb.y - xa.y;
        double ell = std::sqrt(dx * dx + dy * dy);
        if (ell < 1e-12) continue;

        double coeff1 = k_spring / Lref * (ell - Lref) / ell;
        double coeff2 = k_spring / Lref * Lref / (ell * ell * ell);

        double Kxx = coeff1 + coeff2 * dx * dx;
        double Kyy = coeff1 + coeff2 * dy * dy;
        double Kxy = coeff2 * dx * dy;

        H_ii.a11 += Kxx;
        H_ii.a12 += Kxy;
        H_ii.a21 += Kxy;
        H_ii.a22 += Kyy;
    }

    return H_ii;
}

}
