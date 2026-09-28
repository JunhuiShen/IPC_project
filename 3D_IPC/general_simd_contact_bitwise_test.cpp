#include "general_simd_contact.h"

#include "barrier_energy.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <iomanip>
#include <vector>

namespace {

struct ContactFixture {
    const char* name;
    bool segment;
    std::array<Vec3, 4> positions;
};

std::vector<ContactFixture> fixtures() {
    std::vector<ContactFixture> result;
    const auto nt = [&](const char* name, const Vec3& point) {
        result.push_back({name, false,
            {point, Vec3::Zero(), Vec3(1, 0, 0), Vec3(0, 1, 0)}});
    };
    nt("nt-face", Vec3(0.2, 0.3, 0.17));
    nt("nt-face-negative", Vec3(0.2, 0.3, -0.17));
    nt("nt-edge12", Vec3(0.4, -0.12, 0.17));
    nt("nt-edge23", Vec3(0.6, 0.6, 0.17));
    nt("nt-edge31", Vec3(-0.12, 0.4, 0.17));
    nt("nt-vertex1", Vec3(-0.12, -0.11, 0.17));
    nt("nt-vertex2", Vec3(1.12, -0.11, 0.17));
    nt("nt-vertex3", Vec3(-0.12, 1.11, 0.17));
    result.push_back({"nt-degenerate-edge", false,
        {Vec3(0.4, 0.12, 0.17), Vec3::Zero(), Vec3(1, 0, 0), Vec3(0.6, 0, 0)}});
    result.push_back({"nt-degenerate-point", false,
        {Vec3(0.1, 0.12, 0.17), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()}});
    const auto ss = [&](const char* name, const Vec3& a, const Vec3& b) {
        result.push_back({name, true,
            {Vec3(-0.7, 0, 0), Vec3(0.7, 0, 0), a, b}});
    };
    ss("ss-interior", Vec3(0, -0.6, 0.17), Vec3(0, 0.6, 0.17));
    ss("ss-edge-s0", Vec3(-0.9, -0.6, 0.17), Vec3(-0.9, 0.6, 0.17));
    ss("ss-edge-s1", Vec3(0.9, -0.6, 0.17), Vec3(0.9, 0.6, 0.17));
    ss("ss-edge-t0", Vec3(0, 0.12, 0.17), Vec3(0, 0.9, 0.17));
    ss("ss-edge-t1", Vec3(0, -0.9, 0.17), Vec3(0, -0.12, 0.17));
    ss("ss-corner-s0t0", Vec3(-0.9, 0.12, 0.17), Vec3(-0.9, 0.9, 0.17));
    ss("ss-corner-s0t1", Vec3(-0.9, -0.9, 0.17), Vec3(-0.9, -0.12, 0.17));
    ss("ss-corner-s1t0", Vec3(0.9, 0.12, 0.17), Vec3(0.9, 0.9, 0.17));
    ss("ss-corner-s1t1", Vec3(0.9, -0.9, 0.17), Vec3(0.9, -0.12, 0.17));
    ss("ss-parallel", Vec3(-0.4, 0.12, 0.17), Vec3(0.5, 0.12, 0.17));
    result.push_back({"ss-degenerate-first", true,
        {Vec3::Zero(), Vec3::Zero(), Vec3(0, -0.6, 0.17), Vec3(0, 0.6, 0.17)}});
    result.push_back({"ss-two-points", true,
        {Vec3::Zero(), Vec3::Zero(), Vec3(0.1, 0.12, 0.17), Vec3(0.1, 0.12, 0.17)}});
    return result;
}

template <class Value>
void expect_bits(const Value& actual, const Value& expected, const char* field) {
    for (Eigen::Index i = 0; i < actual.size(); ++i) {
        if (std::memcmp(actual.data() + i, expected.data() + i, sizeof(double)) == 0)
            continue;
        ADD_FAILURE() << field << '[' << i << "] actual=" << std::hexfloat
            << actual.data()[i] << " expected=" << expected.data()[i];
        return;
    }
}

} // namespace

TEST(GeneralSimdContactBitwise, AllFeaturesRolesTransformsAndPacketTails) {
    constexpr double d_hat = 0.55;
    const auto scenes = fixtures();
    std::array<ipc_simd::MeshContactInput, ipc_simd::contact_tile_width> inputs;
    std::array<ipc_simd::MeshContactOutput, ipc_simd::contact_tile_width> outputs;
    for (int sample = 0; sample < 6; ++sample) {
        const Mat33 rotation = Eigen::AngleAxisd(0.173 * sample,
            Vec3(0.31, -0.23, 0.71).normalized()).toRotationMatrix();
        const Vec3 translation(0.031 * sample, -0.017 * sample, 0.011 * sample);
        for (const auto& scene : scenes) {
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                inputs[i].segment_segment = scene.segment;
                inputs[i].role = static_cast<int>(i % 4);
                for (int corner = 0; corner < 4; ++corner)
                    inputs[i].positions[corner] = rotation * scene.positions[corner] + translation;
            }
            for (std::size_t count : {std::size_t(1), std::size_t(2), std::size_t(3),
                     std::size_t(7), std::size_t(17), inputs.size()}) {
                ipc_simd::general_mesh_contact_derivatives_tile(inputs.data(), count,
                    d_hat, 370.0, 0.0, 0.031, 0.1, outputs.data());
                // Roles repeat within a tile; compare each once, including the
                // final lane to exercise actual packet tails and compaction.
                for (std::size_t i = 0; i < count; ++i) {
                    if (i >= 4 && i + 1 != count) continue;
                    SCOPED_TRACE(::testing::Message() << "feature=" << scene.name
                        << " sample=" << sample << " count=" << count << " entry=" << i);
                    const auto& input = inputs[i];
                    const auto& x = input.positions;
                    const auto expected = input.segment_segment
                        ? segment_segment_barrier_self_gradient_and_hessian(
                            x[0], x[1], x[2], x[3], d_hat, input.role)
                        : node_triangle_barrier_self_gradient_and_hessian(
                            x[0], x[1], x[2], x[3], d_hat, input.role);
                    expect_bits(outputs[i].gradient, expected.first, "gradient");
                    expect_bits(outputs[i].hessian, expected.second, "hessian");
                }
            }
        }
    }
}
