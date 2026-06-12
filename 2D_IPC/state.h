#pragma once

#include "ipc_math.h"
#include <vector>

struct State2D {
    Vec x;
    Vec v;
    Vec xhat;
    Vec xpin;
    std::vector<double> mass;
    std::vector<char> is_pinned;

    int size() const { return static_cast<int>(mass.size()); }
};

void build_xhat(Vec& xhat, const Vec& x, const Vec& v, double dt);
void update_velocity(Vec& v, const Vec& xnew, const Vec& xold, double dt);
