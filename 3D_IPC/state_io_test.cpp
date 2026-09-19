#include "state_io.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

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


namespace {

struct StateIODirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path()
        / ("ipc_state_io_compat_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    StateIODirectory() { EXPECT_TRUE(std::filesystem::create_directory(path)); }
    ~StateIODirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

template <class Scalar>
void append_native_scalar(std::string& bytes, const Scalar& value) {
    bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

// Independent encoder for the checkpoint layout predating bulk writes. Only
// coordinate values are encoded, never sizeof(Vec3/Vec4), Eigen object padding,
// or assumptions about adjacent vector allocations.
template <class Vector>
void append_original_array(std::string& bytes, const std::vector<Vector>& values) {
    append_native_scalar(bytes, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) {
        for (int component = 0; component < Vector::SizeAtCompileTime; ++component) {
            const double coordinate = value[component];
            append_native_scalar(bytes, coordinate);
        }
    }
}

std::string original_checkpoint_bytes(const DeformedState& state, bool rigid = true) {
    std::string bytes;
    append_original_array(bytes, state.deformed_positions);
    append_original_array(bytes, state.velocities);
    if (rigid) {
        bytes += "RBSTATE1";
        append_original_array(bytes, state.x_coms);
        append_original_array(bytes, state.v_coms);
        append_original_array(bytes, state.orientations);
        append_original_array(bytes, state.omega);
    }
    return bytes;
}

std::string read_checkpoint_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    EXPECT_TRUE(input.is_open());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_checkpoint_bytes(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(output.good());
}

double checkpoint_coordinate(std::size_t index) {
    // Values that arithmetic comparisons cannot check faithfully: both zero
    // signs, infinities, subnormals, and quiet NaNs with distinct payloads.
    constexpr std::array<std::uint64_t, 12> patterns = {{
        0x0000000000000000ULL, 0x8000000000000000ULL,
        0x3ff0000000000000ULL, 0xbff0000000000000ULL,
        0x0000000000000001ULL, 0x8000000000000001ULL,
        0x7fefffffffffffffULL, 0xffefffffffffffffULL,
        0x7ff0000000000000ULL, 0xfff0000000000000ULL,
        0x7ff8123456789abcULL, 0xfff8abcdef012345ULL,
    }};
    if (index % 17 < patterns.size()) {
        double value;
        const auto bits = patterns[index % 17];
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    return static_cast<double>(index) * .03125 - 17.25;
}

template <class Vector>
void fill_checkpoint_array(std::vector<Vector>& values, std::size_t count,
    std::size_t offset) {
    values.resize(count);
    for (std::size_t i = 0; i < count; ++i)
        for (int component = 0; component < Vector::SizeAtCompileTime; ++component)
            values[i][component] = checkpoint_coordinate(
                offset + i * Vector::SizeAtCompileTime + component);
}

DeformedState checkpoint_fixture(std::size_t particles, std::size_t bodies) {
    DeformedState state;
    fill_checkpoint_array(state.deformed_positions, particles, 0);
    // Unequal particle counts are legal in the stored layout; test that the
    // lengths are encoded separately rather than inferred from positions.
    fill_checkpoint_array(state.velocities, particles ? particles + 1 : 0, 11);
    fill_checkpoint_array(state.x_coms, bodies, 23);
    fill_checkpoint_array(state.v_coms, bodies, 37);
    fill_checkpoint_array(state.orientations, bodies, 51);
    fill_checkpoint_array(state.omega, bodies, 69);
    return state;
}

template <class Vector>
void expect_checkpoint_array_bits(const std::vector<Vector>& actual,
    const std::vector<Vector>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        for (int component = 0; component < Vector::SizeAtCompileTime; ++component) {
            const double actual_value = actual[i][component];
            const double expected_value = expected[i][component];
            EXPECT_EQ(std::memcmp(&actual_value, &expected_value, sizeof(double)), 0)
                << "record=" << i << " component=" << component;
        }
}

void expect_checkpoint_bits(const DeformedState& actual, const DeformedState& expected) {
    expect_checkpoint_array_bits(actual.deformed_positions, expected.deformed_positions);
    expect_checkpoint_array_bits(actual.velocities, expected.velocities);
    expect_checkpoint_array_bits(actual.x_coms, expected.x_coms);
    expect_checkpoint_array_bits(actual.v_coms, expected.v_coms);
    expect_checkpoint_array_bits(actual.orientations, expected.orientations);
    expect_checkpoint_array_bits(actual.omega, expected.omega);
}

} // namespace

TEST(StateIO, SerializationMatchesOriginalScalarLayoutAtBufferBoundaries) {
    StateIODirectory directory;
    // The 8192-double buffer holds 2730 Vec3 or 2048 Vec4 records. Include
    // both sides of each boundary and several full buffers with a final tail.
    for (std::size_t count : {0u, 1u, 2047u, 2048u, 2049u,
                              2729u, 2730u, 2731u, 8193u}) {
        SCOPED_TRACE(count);
        const auto original = checkpoint_fixture(count, count);
        serialize_state(directory.path.string(), 7, original);
        const auto expected = original_checkpoint_bytes(original);
        const auto actual = read_checkpoint_bytes(directory.path / "state_0007.bin");
        ASSERT_EQ(actual.size(), expected.size());
        EXPECT_TRUE(actual == expected) << "First differing byte: "
            << std::distance(actual.begin(),
                std::mismatch(actual.begin(), actual.end(), expected.begin()).first);
        const auto coordinate_count = 3 * (original.deformed_positions.size()
            + original.velocities.size() + original.x_coms.size()
            + original.v_coms.size() + original.omega.size())
            + 4 * original.orientations.size();
        EXPECT_EQ(actual.size(), 6 * sizeof(std::uint64_t) + 8
            + coordinate_count * sizeof(double));

        DeformedState loaded;
        ASSERT_TRUE(deserialize_state(directory.path.string(), 7, loaded));
        expect_checkpoint_bits(loaded, original);
    }
}

TEST(StateIO, ReadsIndependentOriginalLayoutIncludingEmptyArrayBoundaries) {
    StateIODirectory directory;
    for (const auto& counts : {std::array<std::size_t, 2>{0, 0},
                              std::array<std::size_t, 2>{0, 3},
                              std::array<std::size_t, 2>{4, 0},
                              std::array<std::size_t, 2>{2731, 2049}}) {
        SCOPED_TRACE(::testing::Message() << "particles=" << counts[0]
            << " bodies=" << counts[1]);
        const auto original = checkpoint_fixture(counts[0], counts[1]);
        write_checkpoint_bytes(directory.path / "state_0011.bin",
            original_checkpoint_bytes(original));
        auto loaded = checkpoint_fixture(2, 2);
        ASSERT_TRUE(deserialize_state(directory.path.string(), 11, loaded));
        expect_checkpoint_bits(loaded, original);
    }
}

TEST(StateIO, LegacyTwoArrayLayoutPreservesSceneRigidState) {
    StateIODirectory directory;
    const auto particle_state = checkpoint_fixture(2731, 0);
    write_checkpoint_bytes(directory.path / "state_0003.bin",
        original_checkpoint_bytes(particle_state, false));
    auto loaded = checkpoint_fixture(1, 3);
    const auto scene_state = loaded;
    ASSERT_TRUE(deserialize_state(directory.path.string(), 3, loaded));
    expect_checkpoint_array_bits(loaded.deformed_positions, particle_state.deformed_positions);
    expect_checkpoint_array_bits(loaded.velocities, particle_state.velocities);
    expect_checkpoint_array_bits(loaded.x_coms, scene_state.x_coms);
    expect_checkpoint_array_bits(loaded.v_coms, scene_state.v_coms);
    expect_checkpoint_array_bits(loaded.orientations, scene_state.orientations);
    expect_checkpoint_array_bits(loaded.omega, scene_state.omega);
}
