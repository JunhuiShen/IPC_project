#include "general_simd_contact.h"

#include "barrier_energy.h"
#include "segment_segment_distance.h"

#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <cmath>
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

namespace {

struct CapturedInteriorContact {
    int pair, role;
    std::array<Vec3, 4> positions;
    Vec3 gradient;
    std::array<double, 9> hessian_column_major;
};

// Original general-solver captures from example 20, frame 28, substep 1,
// sweep 1: vertex 3255 (first three pairs), then vertex 5976. Hex literals
// preserve the actual input/output bits from the GCC 11.4 native server build.
std::array<CapturedInteriorContact, 4> captured_interior_contacts() {
    return {{
        {84528, 0,
            {Vec3(0x1.655f258fa1c5ap-8, 0x1.3068ebe350b17p+0, 0x1.3a870669a4232p-5),
             Vec3(0x1.3290effb08550p-6, 0x1.32436eab88c7ap+0, 0x1.a990780162676p-5),
             Vec3(0x1.c8d1a27c2e932p-8, 0x1.0235af24e0796p+0, 0x1.4c09c655584b4p-5),
             Vec3(0x1.c7602ddf6028dp-7, 0x1.3a72880e78a39p+0, 0x1.9b356b3edeab8p-5)},
            Vec3(-0x1.281ae1a4359c5p-17, -0x1.be726a945cc02p-24, 0x1.238029af2488cp-17),
            {0x1.c8b49fc6f7ee4p-6, 0x1.770d09590badfp-12, -0x1.cc870a7f003b8p-6,
             0x1.770d09590bae0p-12, 0x1.31ec94013dcabp-18, -0x1.797432fc921a4p-12,
             -0x1.cc870a7f003b8p-6, -0x1.797432fc921a4p-12, 0x1.d01ec273432ffp-6}},
        {84557, 0,
            {Vec3(0x1.655f258fa1c5ap-8, 0x1.3068ebe350b17p+0, 0x1.3a870669a4232p-5),
             Vec3(0x1.8b20f2ae1942cp-6, 0x1.2e49d70d2126ep+0, 0x1.01ed4f294a996p-4),
             Vec3(0x1.c8d1a27c2e932p-8, 0x1.0235af24e0796p+0, 0x1.4c09c655584b4p-5),
             Vec3(0x1.c7602ddf6028dp-7, 0x1.3a72880e78a39p+0, 0x1.9b356b3edeab8p-5)},
            Vec3(-0x1.d4e5599ad48eep-8, -0x1.bcaa84446ed13p-17, 0x1.63ccafe95665fp-8),
            {0x1.5cdd0aa3379c1p+3, 0x1.0cce1fe06c3afp-5, -0x1.11e6068f8ebfap+3,
             0x1.0cce1fe06c3afp-5, 0x1.60f5a278d4bc4p-14, -0x1.a0a557788d12ap-6,
             -0x1.11e6068f8ebfbp+3, -0x1.a0a557788d12bp-6, 0x1.ad99fef02314fp+2}},
        {118505, 1,
            {Vec3(0x1.11c60df884ca5p-6, 0x1.2a77033211af7p+0, 0x1.a6d2bb3ec1511p-5),
             Vec3(0x1.655f258fa1c5ap-8, 0x1.3068ebe350b17p+0, 0x1.3a870669a4232p-5),
             Vec3(0x1.c8d1a27c2e932p-8, 0x1.0235af24e0796p+0, 0x1.4c09c655584b4p-5),
             Vec3(0x1.c7602ddf6028dp-7, 0x1.3a72880e78a39p+0, 0x1.9b356b3edeab8p-5)},
            Vec3(-0x1.8f74ee3180619p-11, -0x1.197e896138298p-18, 0x1.5047abe7ca1f2p-11),
            {0x1.c3df89fd605bbp-2, 0x1.39765d892eaf7p-8, -0x1.b3309995bf53bp-2,
             0x1.39765d892eaf7p-8, 0x1.49974dfd22938p-15, -0x1.1b305cfe27413p-8,
             -0x1.b3309995bf53bp-2, -0x1.1b305cfe27413p-8, 0x1.9c7b0f1db400cp-2}},
        {196351, 1,
            {Vec3(-0x1.2c18c8c66e40fp-2, 0x1.1ec2150716936p+0, 0x1.197b3257a866ap-1),
             Vec3(-0x1.21292bdd29bdfp-2, 0x1.1c13333da5f52p+0, 0x1.1195624306c5cp-1),
             Vec3(-0x1.2db25f8c021c8p-2, 0x1.0500699cb96eep+0, 0x1.186a03f1729ddp-1),
             Vec3(-0x1.1d3796fab0645p-2, 0x1.3d0408c2ec085p+0, 0x1.10134540eba28p-1)},
            Vec3(-0x1.30636fda43117p-9, 0x1.983d46ef76965p-15, -0x1.ae2803eac5adap-10),
            {0x1.5c1223d715301p+1, -0x1.151aaf7251500p-4, 0x1.c7339dca641d8p+0,
             -0x1.151aaf7251500p-4, 0x1.ae3e9ed54d285p-10, -0x1.6eff6204829c2p-5,
             0x1.c7339dca641d8p+0, -0x1.6eff6204829c3p-5, 0x1.27b82f89539b8p+0}}
    }};
}

} // namespace

TEST(GeneralSimdContactBitwise, CapturedSolidInteriorContactsAndPacketTails) {
    constexpr double d_hat = 0x1.11ab7f760b865p-9;
    const auto scenes = captured_interior_contacts();
    std::array<ipc_simd::MeshContactInput, ipc_simd::contact_tile_width> inputs;
    std::array<ipc_simd::MeshContactOutput, ipc_simd::contact_tile_width> outputs;
    std::array<unsigned char, ipc_simd::contact_tile_width> active;
    for (const auto& scene : scenes) {
        const auto& x = scene.positions;
        ASSERT_EQ(segment_segment_distance(x[0], x[1], x[2], x[3]).region,
            SegmentSegmentRegion::Interior) << "pair=" << scene.pair;
    }
    // Rotate the records so every capture is exercised as a singleton, at
    // every packet lane, and at a partially filled final packet.
    for (std::size_t offset = 0; offset < scenes.size(); ++offset) {
        for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
            const auto& scene = scenes[(lane + offset) % scenes.size()];
            inputs[lane].positions = scene.positions;
            inputs[lane].role = scene.role;
            inputs[lane].segment_segment = true;
        }
        for (std::size_t count = 1; count <= inputs.size(); ++count) {
            for (auto& output : outputs) {
                output.gradient.setConstant(1234567.0);
                output.hessian.setConstant(1234567.0);
            }
            active.fill(255);
            ipc_simd::general_mesh_contact_derivatives_tile(inputs.data(), count,
                d_hat, 1000.0, 0.0, 1.0 / 600.0, 0.1, outputs.data(), active.data());
            for (std::size_t lane = 0; lane < count; ++lane) {
                const auto& scene = scenes[(lane + offset) % scenes.size()];
                const auto& x = scene.positions;
                SCOPED_TRACE(::testing::Message() << "pair=" << scene.pair
                    << " count=" << count << " lane=" << lane);
                const auto reference = segment_segment_barrier_self_gradient_and_hessian(
                    x[0], x[1], x[2], x[3], d_hat, scene.role);
                const Mat33 captured_hessian = Eigen::Map<const Mat33>(scene.hessian_column_major.data());
#if defined(__AVX512F__) && defined(__FMA__) && defined(__GNUC__) && !defined(__clang__)
                // Native GCC is the diagnosed contraction configuration.
                // Compare both the current scalar kernel and the original
                // captured bits so a common-mode compiler change cannot pass.
                expect_bits(outputs[lane].gradient, reference.first, "scalar_gradient");
                expect_bits(outputs[lane].hessian, reference.second, "scalar_hessian");
                expect_bits(outputs[lane].gradient, scene.gradient, "captured_gradient");
                expect_bits(outputs[lane].hessian, captured_hessian, "captured_hessian");
#else
                // Other compilers/ISAs retain numerical agreement, without
                // imposing GCC's captured contraction order on their scalar code.
                EXPECT_LE((outputs[lane].gradient - reference.first).norm(),
                    2e-11 * std::max(1e-12, reference.first.norm()));
                EXPECT_LE((outputs[lane].hessian - reference.second).norm(),
                    2e-11 * std::max(1e-12, reference.second.norm()));
                EXPECT_LE((outputs[lane].gradient - scene.gradient).norm(),
                    2e-11 * scene.gradient.norm());
                EXPECT_LE((outputs[lane].hessian - captured_hessian).norm(),
                    2e-11 * captured_hessian.norm());
#endif
                EXPECT_EQ(active[lane], 1);
                EXPECT_TRUE(outputs[lane].friction_gradient.isZero());
                EXPECT_TRUE(outputs[lane].friction_hessian.isZero());
            }
            for (std::size_t lane = count; lane < outputs.size(); ++lane) {
                EXPECT_TRUE((outputs[lane].gradient.array() == 1234567.0).all());
                EXPECT_TRUE((outputs[lane].hessian.array() == 1234567.0).all());
                EXPECT_EQ(active[lane], 255);
            }
        }
    }
}
