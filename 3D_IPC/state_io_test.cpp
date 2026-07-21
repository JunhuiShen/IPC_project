#include "state_io.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

TEST(StateIO, RoundTrip) {
    DeformedState original;
    original.deformed_positions = {Vec3(1,2,3), Vec3(4,5,6), Vec3(7,8,9)};
    original.velocities = {Vec3(0.1,0.2,0.3), Vec3(0.4,0.5,0.6), Vec3(0.7,0.8,0.9)};

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
    std::filesystem::remove_all(dir);
}
