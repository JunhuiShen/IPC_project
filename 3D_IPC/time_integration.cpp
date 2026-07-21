#include "time_integration.h"

void build_xhat(std::vector<Vec3>& xhat, const std::vector<Vec3>& x, const std::vector<Vec3>& v, double dt) {
    int n = static_cast<int>(x.size());
    xhat.resize(n);
    for (int i = 0; i < n; ++i) xhat[i] = x[i] + dt * v[i];
}

void update_velocity(std::vector<Vec3>& v, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xold, double dt) {
    int n = static_cast<int>(xnew.size());
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = (xnew[i] - xold[i]) / dt;
}
