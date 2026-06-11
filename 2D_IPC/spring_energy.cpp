#include "spring_energy.h"
#include <cmath>
#include <functional>

namespace physics {
using namespace math;

Vec2 local_spring_grad(int i, const Vec& x, double k,
                       const std::vector<double>& L, int rest_offset) {
    Vec2 g_i{0.0, 0.0};

    std::function<void(int, int, double)> contrib = [&](int a, int b, double Lref) {
        Vec2 xa = get_xi(x, a), xb = get_xi(x, b);
        double dx = xb.x - xa.x, dy = xb.y - xa.y;
        double ell = std::sqrt(dx * dx + dy * dy);
        if (ell < 1e-12) return;

        double coeff = k / Lref * (ell - Lref) / ell;
        double sgn = (i == b) ? +1.0 : -1.0;

        g_i.x += sgn * coeff * dx;
        g_i.y += sgn * coeff * dy;
    };

    int N = static_cast<int>(x.size() / 2);
    if (i - 1 >= 0)     contrib(i - 1, i,     L[rest_offset + i - 1]);
    if (i + 1 <= N - 1) contrib(i,     i + 1, L[rest_offset + i]);

    return g_i;
}

Mat2 local_spring_hess(int i, const Vec& x, double k,
                       const std::vector<double>& L, int rest_offset) {
    Mat2 H_ii{0.0, 0.0, 0.0, 0.0};

    std::function<void(int, int, double)> contrib = [&](int a, int b, double Lref) {
        Vec2 xa = get_xi(x, a), xb = get_xi(x, b);
        double dx = xb.x - xa.x, dy = xb.y - xa.y;
        double ell = std::sqrt(dx * dx + dy * dy);
        if (ell < 1e-12) return;

        double coeff1 = k / Lref * (ell - Lref) / ell;
        double coeff2 = k / Lref * Lref / (ell * ell * ell);

        double Kxx = coeff1 + coeff2 * dx * dx;
        double Kyy = coeff1 + coeff2 * dy * dy;
        double Kxy = coeff2 * dx * dy;

        H_ii.a11 += Kxx;
        H_ii.a12 += Kxy;
        H_ii.a21 += Kxy;
        H_ii.a22 += Kyy;
    };

    int N = static_cast<int>(x.size() / 2);
    if (i - 1 >= 0)     contrib(i - 1, i,     L[rest_offset + i - 1]);
    if (i + 1 <= N - 1) contrib(i,     i + 1, L[rest_offset + i]);

    return H_ii;
}

}
