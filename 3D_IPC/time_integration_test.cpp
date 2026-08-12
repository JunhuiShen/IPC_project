#include "time_integration.h"

#include <gtest/gtest.h>

#include <vector>

TEST(TimeIntegration, UpdatesVelocityFromPositionDifference) {
    const std::vector<Vec3> xold = {Vec3(1.0, 2.0, 3.0), Vec3(-1.0, 0.5, 4.0)};
    const std::vector<Vec3> xnew = {Vec3(1.5, 1.75, 3.25), Vec3(-1.0, 1.25, 3.5)};
    std::vector<Vec3> v;

    update_velocity(v, xnew, xold, 0.25);

    ASSERT_EQ(v.size(), xnew.size());
    EXPECT_NEAR((v[0] - (xnew[0] - xold[0]) / 0.25).norm(), 0.0, 1e-15);
    EXPECT_NEAR((v[1] - (xnew[1] - xold[1]) / 0.25).norm(), 0.0, 1e-15);
}

TEST(TimeIntegration, UpdatesLargeVelocityArray) {
    constexpr int n = 2048;
    constexpr double dt = 0.125;
    std::vector<Vec3> xold(n);
    std::vector<Vec3> xnew(n);
    for (int i = 0; i < n; ++i) {
        xold[i] = Vec3(i, -2.0 * i, 0.5 * i);
        xnew[i] = xold[i] + dt * Vec3(1.0 + i, 2.0 - i, -3.0);
    }

    std::vector<Vec3> v;
    update_velocity(v, xnew, xold, dt);

    ASSERT_EQ(v.size(), xnew.size());
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR((v[i] - Vec3(1.0 + i, 2.0 - i, -3.0)).norm(), 0.0, 1e-12);
}
