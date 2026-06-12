#include "state.h"
#include <gtest/gtest.h>

TEST(State, BuildsPredictedPositions) {
    Vec x = {1.0, 2.0, -1.0, 0.5};
    Vec v = {0.5, -1.0, 2.0, 3.0};
    Vec xhat;

    build_xhat(xhat, x, v, 0.25);

    EXPECT_DOUBLE_EQ(xhat[0], 1.125);
    EXPECT_DOUBLE_EQ(xhat[1], 1.75);
    EXPECT_DOUBLE_EQ(xhat[2], -0.5);
    EXPECT_DOUBLE_EQ(xhat[3], 1.25);
}

TEST(State, UpdatesVelocityFromPositionChange) {
    Vec xold = {1.0, 2.0, -1.0, 0.5};
    Vec xnew = {1.2, 1.8, -0.4, 1.4};
    Vec v;

    update_velocity(v, xnew, xold, 0.2);

    EXPECT_NEAR(v[0], 1.0, 1e-12);
    EXPECT_NEAR(v[1], -1.0, 1e-12);
    EXPECT_NEAR(v[2], 3.0, 1e-12);
    EXPECT_NEAR(v[3], 4.5, 1e-12);
}
