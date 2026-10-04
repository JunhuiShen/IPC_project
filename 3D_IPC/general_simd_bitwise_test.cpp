#include "general_simd_assembly.h"
#include "general_simd_cloth.h"
#include "bending_energy.h"
#include "solid_ipc.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

namespace {

std::uint64_t bits(double value) {
    std::uint64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

template<class Actual, class Expected>
void expect_bitwise(const Actual& actual, const Expected& expected,
    const char* quantity) {
    int mismatches = 0;
    std::ostringstream first;
    for (int i = 0; i < expected.size(); ++i) {
        if (bits(actual.data()[i]) == bits(expected.data()[i])) continue;
        if (mismatches++ == 0) {
            first << quantity << '[' << i << "] actual=" << std::setprecision(17)
                << actual.data()[i] << " reference=" << expected.data()[i]
                << " actual_bits=0x" << std::hex << bits(actual.data()[i])
                << " reference_bits=0x" << bits(expected.data()[i]);
        }
    }
    EXPECT_EQ(mismatches, 0) << first.str();
}

template<class Actual, class Expected>
void expect_numerically_equal(const Actual& actual, const Expected& expected,
    const char* quantity) {
    const double scale = std::max(1e-12, expected.norm());
    for (int component = 0; component < expected.size(); ++component) {
        SCOPED_TRACE(::testing::Message() << quantity << '[' << component << ']');
        if (std::isnan(expected.data()[component]))
            EXPECT_TRUE(std::isnan(actual.data()[component]));
        else if (std::isinf(expected.data()[component]))
            EXPECT_EQ(actual.data()[component], expected.data()[component]);
        else
            EXPECT_NEAR(actual.data()[component], expected.data()[component], 2e-11 * scale);
    }
}

struct ClothFixture {
    RefMesh mesh;
    VertexTriangleMap adjacency;
    std::vector<IncidentTriangles> incident;
    std::vector<ShapeGrads> shapes;
    std::vector<Vec3> positions, predicted;
    std::vector<Pin> pins;
    PinMap pin_map;
    std::vector<unsigned char> solid, surface;
    SimParams params = SimParams::zeros();

    ClothFixture() {
        const std::array<Vec3, 4> local = {Vec3(.13, -.2, .03),
            Vec3(1.23, -.13, .06), Vec3(.43, .9, .15), Vec3(.53, -1.1, -.2)};
        for (int object = 0; object < 2; ++object)
            for (const auto& value : local)
                positions.push_back(value + Vec3(3.1 * object, .01 * object, 0));
        mesh.num_positions = positions.size();
        mesh.tris = {0, 1, 2, 1, 0, 3, 4, 5, 6, 5, 4, 7};
        mesh.compute_dm_inverse(positions);
        mesh.mass.resize(8);
        for (int node = 0; node < 8; ++node) mesh.mass[node] = .13 + .027 * node;
        mesh.node_to_rb.assign(8, -1);
        mesh.hinges = {Hinge{{0, 1, 2, 3}, -.2, .7}, Hinge{{4, 5, 6, 7}, .13, 1.1}};
        for (int hinge = 0; hinge < 2; ++hinge)
            for (int role = 0; role < 4; ++role)
                mesh.hinge_adj[mesh.hinges[hinge].v[role]].emplace_back(hinge, role);
        incident.resize(8);
        for (int triangle = 0; triangle < 4; ++triangle) {
            shapes.push_back(shape_function_gradients(mesh.Dm_inverse[triangle]));
            for (int role = 0; role < 3; ++role)
                incident[mesh.tris[3 * triangle + role]].emplace_back(triangle, role);
        }
        for (int node = 0; node < 8; ++node) adjacency[node] = incident[node];
        predicted = positions;
        for (int node = 0; node < 8; ++node) {
            positions[node] += Vec3(.013 * (node + 1), -.017 * (node % 3), .031 * (node % 2));
            predicted[node] += Vec3(.021, -.012, .037);
        }
        pins.push_back(Pin{0, Vec3(.15, -.22, .01)});
        pins.push_back(Pin{5, Vec3(4.2, -.15, .02)});
        pin_map.assign(8, -1); pin_map[0] = 0; pin_map[5] = 1;
        solid.assign(8, 0); surface.assign(8, 1);
        params.fps = 30; params.substeps = 20;
        params.gravity = Vec3(.03, -9.81, -.017);
        params.mu = 115000 / 2.5; params.lambda = 115000 * .25 / (1.25 * .5);
        params.kB = .009; params.kpin = 1e9;
        params.use_basic_experimental = false;
        params.use_basic_experimental_v2 = false;
        params.use_simd = false;
    }

    void disable_point() {
        mesh.mass.assign(8, 0.0);
        pins.clear(); pin_map.assign(8, -1);
    }

    void disable_membrane() {
        for (int node = 0; node < 8; ++node) {
            incident[node].clear(); adjacency[node].clear();
        }
    }

    std::pair<Vec3, Mat33> reference_with_shared_bending(int node) const {
        SimParams without_bending = params;
        without_bending.kB = 0.0;
        auto result = compute_local_gradient_and_hessian_no_barrier(
            node, mesh, adjacency, pins, without_bending, positions, predicted,
            &pin_map, &incident[node], &shapes, nullptr);
        if (!(params.kB > 0.0)) return result;
        const auto found = mesh.hinge_adj.find(node);
        if (found == mesh.hinge_adj.end()) return result;
        // Point/membrane keep their exact scalar-reference contract. Bending
        // must instead match the same SIMD entry used by basic experimental v2.
        for (const auto& [hinge, role] : found->second) {
            const auto& value = mesh.hinges[hinge];
            std::array<Vec3, 4> corners;
            for (int corner = 0; corner < 4; ++corner)
                corners[corner] = positions[value.v[corner]];
            Vec3 gradient;
            Mat33 hessian;
            ipc_simd::bending_derivatives_tile(corners.data(), &role, &value.c_e,
                &value.bar_theta, 1, params.kB, &gradient, &hessian);
            result.first += params.dt2() * gradient;
            result.second += params.dt2() * hessian;
        }
        return result;
    }

    void compare(bool cached = true) {
        const int nodes[] = {0, 1, 2, 3, 4, 5, 6, 7};
        solver_detail::GeneralSimdMaterials materials;
        materials.prepare(mesh, incident, 8, params.kB > 0.0);
        for (std::size_t count = 1; count <= 8; ++count) {
            std::array<solver_detail::GeneralSimdVertexSystem, 8> output;
            solver_detail::prepare_general_simd_batch(nodes, count, mesh, incident,
                shapes, pins, pin_map, params, positions, predicted, nullptr,
                solid, surface, output.data(), cached ? &materials : nullptr);
            for (std::size_t entry = 0; entry < count; ++entry) {
                SCOPED_TRACE(::testing::Message() << "batch_width=" << count << " node=" << entry);
                const auto reference = compute_local_gradient_and_hessian_no_barrier(
                    nodes[entry], mesh, adjacency, pins, params, positions, predicted,
                    &pin_map, &incident[entry], &shapes, nullptr);
                if (params.kB > 0.0) {
                    const auto shared_reference = reference_with_shared_bending(nodes[entry]);
                    expect_bitwise(output[entry].gradient, shared_reference.first, "shared_bending_gradient");
                    expect_bitwise(output[entry].hessian, shared_reference.second, "shared_bending_hessian");
                    expect_numerically_equal(output[entry].gradient, reference.first, "scalar_gradient");
                    expect_numerically_equal(output[entry].hessian, reference.second, "scalar_hessian");
                } else {
                    expect_bitwise(output[entry].gradient, reference.first, "gradient");
                    expect_bitwise(output[entry].hessian, reference.second, "hessian");
                }
            }
        }
    }
};

} // namespace

TEST(GeneralSimdBitwise, PointMatchesOriginalAssembly) {
    ClothFixture fixture;
    fixture.disable_membrane(); fixture.params.kB = 0;
    fixture.compare();
}

TEST(GeneralSimdBitwise, MixedSolidClothPointRoundingPinsAndBatchTails) {
    constexpr std::size_t width = ipc_simd::tile_width;
    static_assert(width == 8);
    RefMesh mesh;
    mesh.num_positions = width;
    mesh.node_to_rb.assign(width, -1);
    mesh.tet_adj.resize(width);
    mesh.mass.resize(width);
    std::vector<Vec3> positions(width), predicted(width);
    std::vector<IncidentTriangles> incident(width);
    const std::vector<ShapeGrads> shapes;
    std::vector<Pin> pins;
    PinMap pin_map(width, -1);
    std::vector<unsigned char> solid(width), surface(width, 1);
    SimParams params = SimParams::zeros();
    params.fps = 30; params.substeps = 20;
    params.gravity = Vec3(0.0, -9.81, 0.0);
    params.kpin = 100000.0;
    for (std::size_t node = 0; node < width; ++node) {
        // Node zero retains the exact frame-28/substep-1/sweep-1 inputs
        // captured for solid vertex 1134 on the GCC native server build.
        mesh.mass[node] = 0x1.d865de9b0ba24p-8 + .001 * node;
        positions[node] = Vec3(-0x1.9c2762463df2bp-2,
            0x1.35198b5adc71cp+0, -0x1.fd7319a438896p-2);
        predicted[node] = Vec3(-0x1.9bf14a748594fp-2,
            0x1.357131181ab89p+0, -0x1.fd71b59147924p-2);
        predicted[node] += Vec3(.00001 * node, -.00003 * node, .00007 * node);
        if (node % 4 >= 2) {
            pin_map[node] = static_cast<int>(pins.size());
            pins.push_back(Pin{static_cast<int>(node),
                positions[node] + Vec3(.00013, -.00027, .00019)});
        }
    }
    // Permutation makes it impossible to accidentally use the batch lane as
    // the global node ID when selecting the cloth/solid arithmetic.
    const std::array<int, width> nodes{7, 0, 5, 2, 3, 4, 1, 6};
    for (int pattern = 0; pattern < 3; ++pattern) {
        mesh.tet_nodes.clear();
        for (std::size_t node = 0; node < width; ++node) {
            solid[node] = pattern == 1 || (pattern == 2 && node % 2 == 0);
            if (solid[node]) mesh.tet_nodes.push_back(static_cast<int>(node));
        }
        solver_detail::GeneralSimdMaterials materials;
        materials.prepare(mesh, incident, width, false);
        for (bool cached : {false, true}) {
            for (std::size_t count = 1; count <= width; ++count) {
                std::array<solver_detail::GeneralSimdVertexSystem, width> output;
                for (auto& value : output) {
                    value.gradient.setConstant(1234567.0);
                    value.hessian.setConstant(1234567.0);
                }
                solver_detail::prepare_general_simd_batch(nodes.data(), count,
                    mesh, incident, shapes, pins, pin_map, params, positions,
                    predicted, nullptr, solid, surface, output.data(),
                    cached ? &materials : nullptr);
                for (std::size_t lane = 0; lane < count; ++lane) {
                    const int node = nodes[lane];
                    SCOPED_TRACE(::testing::Message() << "pattern=" << pattern
                        << " cached=" << cached << " count=" << count
                        << " lane=" << lane << " node=" << node);
                    if (solid[node]) {
                        const auto expected = compute_solid_local_gradient_and_pbgs_block_no_barrier(
                            node, mesh, pins, params, positions, predicted, &surface, &pin_map);
#if defined(__AVX512F__) && defined(__FMA__) && defined(__GNUC__) && !defined(__clang__)
                        // Exactness is a contract for the diagnosed GCC native
                        // build, not a claim of cross-compiler reproducibility.
                        expect_bitwise(output[lane].gradient, expected.first, "solid_gradient");
                        expect_bitwise(output[lane].hessian, expected.second, "solid_hessian");
                        if (node == 0)
                            EXPECT_EQ(bits(output[lane].gradient.y()), UINT64_C(0xbee3ce1a61699710));
#else
                        expect_numerically_equal(output[lane].gradient, expected.first, "solid_gradient");
                        expect_numerically_equal(output[lane].hessian, expected.second, "solid_hessian");
#endif
                    } else {
                        ipc_simd::PointInput input;
                        input.mass = mesh.mass[node]; input.position = positions[node];
                        input.predicted_position = predicted[node];
                        if (pin_map[node] >= 0) input.pin_target = pins[pin_map[node]].target_position;
                        Vec3 gradient; Mat33 hessian;
                        ipc_simd::point_derivatives_tile(&input, 1, params.gravity,
                            params.kpin, params.dt2(), &gradient, &hessian);
                        // The solid fix must not change basic-v2 cloth arithmetic.
                        expect_bitwise(output[lane].gradient, gradient, "cloth_gradient");
                        expect_bitwise(output[lane].hessian, hessian, "cloth_hessian");
                    }
                }
                for (std::size_t lane = count; lane < width; ++lane) {
                    EXPECT_TRUE((output[lane].gradient.array() == 1234567.0).all());
                    EXPECT_TRUE((output[lane].hessian.array() == 1234567.0).all());
                }
            }
        }
    }
}

TEST(GeneralSimdBitwise, BendingMatchesSharedKernelAssembly) {
    ClothFixture fixture;
    fixture.disable_point(); fixture.disable_membrane();
    fixture.compare();
}

namespace {

void check_membrane_kernel(const ClothFixture& fixture, std::size_t count) {
    std::array<Vec3, 24> positions;
    std::array<Mat22, 8> inverse;
    std::array<Vec2, 8> shape;
    std::array<double, 8> areas;
    std::array<Vec3, 8> gradient;
    std::array<Mat33, 8> hessian;
    for (int e = 0; e < 8; ++e) {
        const int triangle = e / 2;
        for (int role = 0; role < 3; ++role)
            positions[3 * e + role] = fixture.positions[fixture.mesh.tris[3 * triangle + role]];
        inverse[e] = fixture.mesh.Dm_inverse[triangle];
        shape[e] = fixture.shapes[triangle][e % 3];
        areas[e] = fixture.mesh.area[triangle];
    }
    ipc_simd::general_corotated_derivatives_tile(positions.data(), inverse.data(), areas.data(),
        shape.data(), count, fixture.params.mu, fixture.params.lambda, gradient.data(), hessian.data());
    for (std::size_t e = 0; e < count; ++e) {
        SCOPED_TRACE(::testing::Message() << "entry=" << e);
        Mat32 ds;
        ds.col(0) = positions[3 * e + 1] - positions[3 * e];
        ds.col(1) = positions[3 * e + 2] - positions[3 * e];
        const Mat32 F = ds * inverse[e];
        const auto cache = buildCorotatedCache(F);
        const auto P = PCorotated32(cache, F, fixture.params.mu, fixture.params.lambda);
        Mat66 derivative;
        dPdFCorotated32(cache, fixture.params.mu, fixture.params.lambda, derivative);
        ShapeGrads shapes;
        shapes[0] = shape[e];
        expect_bitwise(gradient[e], corotated_node_gradient(P, areas[e], shapes, 0), "gradient");
        expect_bitwise(hessian[e], corotated_node_hessian(derivative, areas[e], shapes, 0), "hessian");
    }
}

void check_bending_kernel(const ClothFixture& fixture, std::size_t count) {
    std::array<Vec3, 32> positions;
    std::array<int, 8> roles;
    std::array<double, 8> coefficients, angles;
    std::array<Vec3, 8> gradient;
    std::array<Mat33, 8> hessian;
    for (int e = 0; e < 8; ++e) {
        const auto& hinge = fixture.mesh.hinges[e / 4];
        for (int role = 0; role < 4; ++role)
            positions[4 * e + role] = fixture.positions[hinge.v[role]];
        roles[e] = e % 4; coefficients[e] = hinge.c_e; angles[e] = hinge.bar_theta;
    }
    ipc_simd::bending_derivatives_tile(positions.data(), roles.data(), coefficients.data(),
        angles.data(), count, fixture.params.kB, gradient.data(), hessian.data());
    for (std::size_t e = 0; e < count; ++e) {
        SCOPED_TRACE(::testing::Message() << "entry=" << e);
        HingeDef def;
        for (int role = 0; role < 4; ++role) def.x[role] = positions[4 * e + role];
        const auto reference = bending_node_gradient_hessian_psd(
            def, fixture.params.kB, coefficients[e], angles[e], roles[e]);
        expect_numerically_equal(gradient[e], reference.first, "scalar_gradient");
        expect_numerically_equal(hessian[e], reference.second, "scalar_hessian");
        Vec3 single_gradient;
        Mat33 single_hessian;
        ipc_simd::bending_derivatives_tile(positions.data() + 4 * e,
            roles.data() + e, coefficients.data() + e, angles.data() + e,
            1, fixture.params.kB, &single_gradient, &single_hessian);
        expect_bitwise(gradient[e], single_gradient, "shared_kernel_gradient");
        expect_bitwise(hessian[e], single_hessian, "shared_kernel_hessian");
    }
}

void deform_fixture(ClothFixture& fixture, int sample) {
    for (std::size_t node = 0; node < fixture.positions.size(); ++node) {
        const double phase = static_cast<double>(sample * (node + 1));
        fixture.positions[node] += Vec3(.071 * std::sin(phase),
            .043 * std::sin(.7 * phase), .083 * std::sin(1.3 * phase));
    }
}

} // namespace

TEST(GeneralSimdBitwise, SharedBendingKernelRetainsScalarNumericalAgreement) {
    for (int sample = 0; sample < 16; ++sample) {
        ClothFixture fixture;
        deform_fixture(fixture, sample);
        for (std::size_t count = 1; count <= 8; ++count) {
            SCOPED_TRACE(::testing::Message() << "sample=" << sample << " count=" << count);
            check_bending_kernel(fixture, count);
        }
    }
}

TEST(GeneralSimdBitwise, BendingDegeneracyAndSignedZerosMatchReference) {
    ClothFixture fixture;
    const std::array<double, 4> stiffnesses = {0.0, -0.0, .009,
        std::numeric_limits<double>::quiet_NaN()};
    for (double stiffness : stiffnesses) {
        for (int role = 0; role < 4; ++role) {
            HingeDef def;
            for (int corner = 0; corner < 4; ++corner)
                def.x[corner] = fixture.positions[corner];
            def.x[1] = def.x[0];
            const double coefficient = std::numeric_limits<double>::quiet_NaN();
            const double rest = std::numeric_limits<double>::quiet_NaN();
            Vec3 gradient;
            Mat33 hessian;
            ipc_simd::bending_derivatives_tile(def.x, &role, &coefficient,
                &rest, 1, stiffness, &gradient, &hessian);
            const auto reference = bending_node_gradient_hessian_psd(
                def, stiffness, coefficient, rest, role);
            expect_bitwise(gradient, reference.first, "degenerate_gradient");
            expect_bitwise(hessian, reference.second, "degenerate_hessian");
        }
    }
    for (double stiffness : {0.0, -0.0}) {
        fixture.params.kB = stiffness;
        check_bending_kernel(fixture, 8);
    }
    // Empty tiles are valid even with null pointers.
    ipc_simd::bending_derivatives_tile(nullptr, nullptr, nullptr,
        nullptr, 0, .009, nullptr, nullptr);
    ipc_simd::general_corotated_derivatives_tile(nullptr, nullptr, nullptr,
        nullptr, 0, 2.0, 3.0, nullptr, nullptr);
}

TEST(GeneralSimdBitwise, BendingNonfiniteGeometryRetainsReferenceClassification) {
    ClothFixture fixture;
    for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity()}) {
        HingeDef def;
        for (int corner = 0; corner < 4; ++corner)
            def.x[corner] = fixture.positions[corner];
        def.x[2][1] = invalid;
        const double coefficient = .7, rest = -.2;
        for (int role = 0; role < 4; ++role) {
            Vec3 gradient;
            Mat33 hessian;
            ipc_simd::bending_derivatives_tile(def.x, &role, &coefficient,
                &rest, 1, .009, &gradient, &hessian);
            const auto reference = bending_node_gradient_hessian_psd(
                def, .009, coefficient, rest, role);
            const auto compare = [](const auto& actual, const auto& expected) {
                for (int component = 0; component < expected.size(); ++component) {
                    if (std::isnan(expected.data()[component]))
                        EXPECT_TRUE(std::isnan(actual.data()[component]));
                    else
                        EXPECT_EQ(bits(actual.data()[component]), bits(expected.data()[component]));
                }
            };
            compare(gradient, reference.first);
            compare(hessian, reference.second);
        }
    }
}
