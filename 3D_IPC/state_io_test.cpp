#include "state_io.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

TEST(StateIO, RoundTrip) {
    DeformedState original;
    original.deformed_positions = {Vec3(1,2,3), Vec3(4,5,6), Vec3(7,8,9)};
    original.velocities = {Vec3(0.1,0.2,0.3), Vec3(0.4,0.5,0.6), Vec3(0.7,0.8,0.9)};
    original.x_coms = {Vec3(-1.0, 2.0, 3.5), Vec3(4.0, -5.0, 6.0)};
    original.v_coms = {Vec3(0.3, -0.2, 0.1), Vec3(-0.6, 0.5, -0.4)};
    original.orientations = {
        Vec4(1.0, 0.0, 0.0, 0.0),
        Vec4(0.5, 0.5, 0.5, 0.5),
    };
    original.omega = {Vec3(1.0, 2.0, 3.0), Vec3(-3.0, -2.0, -1.0)};

    std::string dir = "/tmp/ipc_serialize_test";
    std::filesystem::create_directories(dir);
    serialize_state(dir, 42, original);

    DeformedState loaded;
    ASSERT_TRUE(deserialize_state(dir, 42, loaded));

    ASSERT_EQ(loaded.deformed_positions.size(), original.deformed_positions.size());
    ASSERT_EQ(loaded.velocities.size(), original.velocities.size());
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR((loaded.deformed_positions[i] - original.deformed_positions[i]).norm(), 0.0, 1e-15);
        EXPECT_NEAR((loaded.velocities[i] - original.velocities[i]).norm(), 0.0, 1e-15);
    }
    ASSERT_EQ(loaded.x_coms.size(), original.x_coms.size());
    ASSERT_EQ(loaded.v_coms.size(), original.v_coms.size());
    ASSERT_EQ(loaded.orientations.size(), original.orientations.size());
    ASSERT_EQ(loaded.omega.size(), original.omega.size());
    for (int rb = 0; rb < static_cast<int>(original.x_coms.size()); ++rb) {
        EXPECT_TRUE(loaded.x_coms[rb].isApprox(original.x_coms[rb], 0.0));
        EXPECT_TRUE(loaded.v_coms[rb].isApprox(original.v_coms[rb], 0.0));
        EXPECT_TRUE(loaded.orientations[rb].isApprox(
            original.orientations[rb], 0.0));
        EXPECT_TRUE(loaded.omega[rb].isApprox(original.omega[rb], 0.0));
    }
    std::filesystem::remove_all(dir);
}
