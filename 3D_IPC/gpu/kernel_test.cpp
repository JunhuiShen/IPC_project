#include "kernels.h"
#include "SIMD.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// These are cross-device numerical comparisons, not bitwise requirements.
// Thin hinges amplify the differences in CPU/device dot and cross products.
constexpr double absolute_tolerance = 2e-9;
constexpr double relative_tolerance = 2e-9;
constexpr double thin_absolute_tolerance = 2e-8;
constexpr double thin_relative_tolerance = 2e-7;
constexpr double canary = 1234567.25;

struct Fixture {
    std::vector<ipc_gpu::Point> points;
    std::vector<ipc_gpu::Triangle> triangles;
    std::vector<ipc_gpu::Hinge> hinges;
    std::vector<bool> thin_triangle, thin_hinge;
};

struct CheckState {
    std::uint64_t components = 0;
    double maximum_tolerance_fraction = 0;
    ipc_gpu::Statistics expected;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void pack_vector(const Vec3& value, double* output) {
    for (int axis = 0; axis < 3; ++axis) output[axis] = value[axis];
}

Vec3 unpack_vector(const double* value) {
    return Vec3(value[0], value[1], value[2]);
}

Fixture make_fixture(std::size_t count) {
    Fixture fixture;
    fixture.points.resize(count);
    fixture.triangles.resize(count);
    fixture.hinges.resize(count);
    fixture.thin_triangle.resize(count);
    fixture.thin_hinge.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double t = static_cast<double>(i);
        auto& point = fixture.points[i];
        const Vec3 position(.03 * t - .2, .17 * std::sin(.3 * t), -.04 * t);
        pack_vector(position, point.position);
        pack_vector(position + Vec3(.031, -.013 - .0001 * t, .017), point.predicted);
        pack_vector(position + Vec3(-.023, .041, .019), point.target);
        point.mass = i % 11 == 0 ? 0 : .07 + .013 * (i % 19);
        point.pinned = i % 3 != 0;
        point.solid = i % 2;

        const Mat33 rotation = Eigen::AngleAxisd(.11 * t,
            Vec3(.3, -.2, 1).normalized()).toRotationMatrix();
        auto& triangle = fixture.triangles[i];
        Mat22 inverse;
        inverse << 1.1 + .01 * (i % 5), .13, -.07, .9 + .02 * (i % 7);
        const int triangle_case = static_cast<int>((i / 3) % 11);
        fixture.thin_triangle[i] = triangle_case == 4 || triangle_case == 5;
        std::array<Vec3, 3> corners;
        if (fixture.thin_triangle[i]) {
            // Diagonal deformation exercises eigenvalue clamping without
            // cancellation obscuring whether the reference itself is valid.
            inverse.setIdentity();
            const double thickness = triangle_case == 4 ? 1e-5 : 1e-7;
            corners = {Vec3::Zero(), Vec3(1.1, 0, 0), Vec3(0, thickness, 0)};
        } else {
            corners = {position,
                position + rotation * Vec3(.8 + .01 * (i % 13), .09, .03),
                position + rotation * Vec3(-.13, 1.1 + .02 * (i % 9), .14)};
        }
        for (int vertex = 0; vertex < 3; ++vertex)
            pack_vector(corners[vertex], triangle.positions + 3 * vertex);
        for (int row = 0; row < 2; ++row)
            for (int column = 0; column < 2; ++column)
                triangle.dm_inverse[2 * row + column] = inverse(row, column);
        const int role = static_cast<int>(i % 3);
        const Vec2 shape = role == 0
            ? Vec2(-inverse(0, 0) - inverse(1, 0), -inverse(0, 1) - inverse(1, 1))
            : Vec2(inverse(role - 1, 0), inverse(role - 1, 1));
        triangle.shape[0] = shape[0]; triangle.shape[1] = shape[1];
        triangle.area = .2 + .017 * (i % 17);

        auto& hinge = fixture.hinges[i];
        const int hinge_case = static_cast<int>((i / 4) % 9);
        fixture.thin_hinge[i] = hinge_case == 3;
        const double height = fixture.thin_hinge[i] ? 1e-4 : .7 + .03 * (i % 5);
        std::array<Vec3, 4> hinge_corners = {Vec3::Zero(), Vec3(1.2, 0, 0),
            Vec3(.23, height, .12 * height), Vec3(.81, -.8 * height, .6 * height)};
        if (hinge_case == 5) hinge_corners[3] = hinge_corners[0];
        if (hinge_case == 7) hinge_corners[1] = hinge_corners[0];
        for (int vertex = 0; vertex < 4; ++vertex)
            pack_vector(rotation * hinge_corners[vertex] + position,
                hinge.positions + 3 * vertex);
        hinge.coefficient = .4 + .031 * (i % 23);
        hinge.rest_angle = -.3 + .017 * (i % 29);
        hinge.active_node = static_cast<int>(i % 4);
    }
    return fixture;
}

std::vector<ipc_gpu::Derivatives> cpu_reference(const Fixture& fixture,
    const ipc_gpu::Parameters& parameters, std::size_t count, int kind) {
    std::vector<ipc_gpu::Derivatives> reference(count);
    constexpr std::size_t width = ipc_simd::tile_width;
    for (std::size_t begin = 0; begin < count; begin += width) {
        const std::size_t active = std::min(width, count - begin);
        std::array<Vec3, width> gradients;
        std::array<Mat33, width> hessians;
        if (kind == 0) {
            std::array<ipc_simd::PointInput, width> inputs;
            for (std::size_t lane = 0; lane < active; ++lane) {
                const auto& point = fixture.points[begin + lane];
                auto& input = inputs[lane];
                input.mass = point.mass;
                input.position = unpack_vector(point.position);
                input.predicted_position = unpack_vector(point.predicted);
                if (point.pinned) input.pin_target = unpack_vector(point.target);
            }
            for (std::size_t lane=0; lane<active; ++lane) {
                const auto kernel = fixture.points[begin+lane].solid
                    ? ipc_simd::solid_point_derivatives_tile : ipc_simd::point_derivatives_tile;
                kernel(inputs.data()+lane,1,unpack_vector(parameters.gravity),
                    parameters.kpin,parameters.dt2,gradients.data()+lane,hessians.data()+lane);
            }
        } else if (kind == 1) {
            std::array<Vec3, 3 * width> positions;
            std::array<Mat22, width> inverses;
            std::array<Vec2, width> shapes;
            std::array<double, width> areas;
            for (std::size_t lane = 0; lane < active; ++lane) {
                const auto& triangle = fixture.triangles[begin + lane];
                for (int vertex = 0; vertex < 3; ++vertex)
                    positions[3 * lane + vertex] = unpack_vector(triangle.positions + 3 * vertex);
                for (int row = 0; row < 2; ++row)
                    for (int column = 0; column < 2; ++column)
                        inverses[lane](row, column) = triangle.dm_inverse[2 * row + column];
                shapes[lane] = Vec2(triangle.shape[0], triangle.shape[1]);
                areas[lane] = triangle.area;
            }
            ipc_simd::corotated_derivatives_tile(positions.data(), inverses.data(),
                areas.data(), shapes.data(), active, parameters.mu, parameters.lambda,
                gradients.data(), hessians.data());
        } else {
            std::array<Vec3, 4 * width> positions;
            std::array<int, width> roles;
            std::array<double, width> coefficients, rest_angles;
            for (std::size_t lane = 0; lane < active; ++lane) {
                const auto& hinge = fixture.hinges[begin + lane];
                for (int vertex = 0; vertex < 4; ++vertex)
                    positions[4 * lane + vertex] = unpack_vector(hinge.positions + 3 * vertex);
                roles[lane] = hinge.active_node;
                coefficients[lane] = hinge.coefficient;
                rest_angles[lane] = hinge.rest_angle;
            }
            ipc_simd::bending_derivatives_tile(positions.data(), roles.data(),
                coefficients.data(), rest_angles.data(), active, parameters.bending_stiffness,
                gradients.data(), hessians.data());
        }
        for (std::size_t lane = 0; lane < active; ++lane) {
            pack_vector(gradients[lane], reference[begin + lane].gradient);
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    reference[begin + lane].hessian[3 * row + column] = hessians[lane](row, column);
        }
    }
    return reference;
}

void compare_component(double actual, double expected, bool thin,
    const std::string& context, CheckState& state) {
    const double absolute = thin ? thin_absolute_tolerance : absolute_tolerance;
    const double relative = thin ? thin_relative_tolerance : relative_tolerance;
    const double limit = absolute + relative * std::max(std::abs(actual), std::abs(expected));
    const double error = std::abs(actual - expected);
    if (!std::isfinite(actual) || !std::isfinite(expected) || error > limit) {
        std::ostringstream message;
        message << std::setprecision(17) << context << ": actual=" << actual
                << " reference=" << expected << " absolute_error=" << error
                << " allowed=" << limit;
        throw std::runtime_error(message.str());
    }
    state.maximum_tolerance_fraction = std::max(state.maximum_tolerance_fraction, error / limit);
    ++state.components;
}

void check_statistics(const CheckState& state) {
    const auto actual = ipc_gpu::statistics();
    require(actual.calls == state.expected.calls, "Unexpected successful compute call count");
    require(actual.launches == state.expected.launches, "Unexpected CUDA launch count");
    require(actual.points == state.expected.points, "Unexpected processed point count");
    require(actual.triangles == state.expected.triangles, "Unexpected processed triangle count");
    require(actual.hinges == state.expected.hinges, "Unexpected processed hinge count");
    for (double elapsed : {actual.host_to_device_ms, actual.kernel_ms, actual.device_to_host_ms})
        require(std::isfinite(elapsed) && elapsed >= 0, "Invalid CUDA timing statistic");
}

void run_case(const Fixture& fixture, const ipc_gpu::Parameters& parameters,
    const std::array<std::size_t, 3>& counts, CheckState& state) {
    std::array<std::vector<ipc_gpu::Derivatives>, 3> outputs;
    for (int kind = 0; kind < 3; ++kind) {
        outputs[kind].resize(counts[kind] + 1);
        for (auto& record : outputs[kind]) {
            std::fill(std::begin(record.gradient), std::end(record.gradient), canary);
            std::fill(std::begin(record.hessian), std::end(record.hessian), canary);
        }
    }
    ipc_gpu::compute(counts[0] ? fixture.points.data() : nullptr, counts[0],
        counts[1] ? fixture.triangles.data() : nullptr, counts[1],
        counts[2] ? fixture.hinges.data() : nullptr, counts[2], parameters,
        counts[0] ? outputs[0].data() : nullptr, counts[1] ? outputs[1].data() : nullptr,
        counts[2] ? outputs[2].data() : nullptr);
    if (counts[0] || counts[1] || counts[2]) ++state.expected.calls;
    for (std::size_t count : counts) if (count) ++state.expected.launches;
    state.expected.points += counts[0];
    state.expected.triangles += counts[1];
    state.expected.hinges += counts[2];
    static constexpr const char* names[] = {"point", "triangle", "hinge"};
    for (int kind = 0; kind < 3; ++kind) {
        const auto reference = cpu_reference(fixture, parameters, counts[kind], kind);
        for (std::size_t entry = 0; entry < counts[kind]; ++entry) {
            const bool thin = kind == 1 ? fixture.thin_triangle[entry]
                            : kind == 2 ? fixture.thin_hinge[entry] : false;
            const std::string context = std::string(names[kind]) + " count="
                + std::to_string(counts[kind]) + " entry=" + std::to_string(entry);
            for (int axis = 0; axis < 3; ++axis)
                compare_component(outputs[kind][entry].gradient[axis], reference[entry].gradient[axis],
                    thin, context + " gradient[" + std::to_string(axis) + "]", state);
            for (int component = 0; component < 9; ++component)
                compare_component(outputs[kind][entry].hessian[component], reference[entry].hessian[component],
                    thin, context + " hessian[" + std::to_string(component) + "]", state);
        }
        for (double value : outputs[kind].back().gradient)
            require(value == canary, std::string(names[kind]) + " gradient tail overwritten");
        for (double value : outputs[kind].back().hessian)
            require(value == canary, std::string(names[kind]) + " hessian tail overwritten");
    }
    check_statistics(state);
}

void check_nonfinite_failure(const Fixture& fixture, const ipc_gpu::Parameters& parameters,
    CheckState& state) {
    for (int kind = 0; kind < 3; ++kind) {
        Fixture invalid = fixture;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (kind == 0) invalid.points[0].position[0] = nan;
        if (kind == 1) invalid.triangles[0].dm_inverse[0] = nan;
        if (kind == 2) invalid.hinges[0].coefficient = nan;
        ipc_gpu::Derivatives output{};
        bool threw = false;
        try {
            ipc_gpu::compute(kind == 0 ? invalid.points.data() : nullptr, kind == 0 ? 1 : 0,
                kind == 1 ? invalid.triangles.data() : nullptr, kind == 1 ? 1 : 0,
                kind == 2 ? invalid.hinges.data() : nullptr, kind == 2 ? 1 : 0,
                parameters, kind == 0 ? &output : nullptr,
                kind == 1 ? &output : nullptr, kind == 2 ? &output : nullptr);
        } catch (const std::exception&) {
            threw = true;
        }
        require(threw, "Nonfinite output did not throw for kernel kind " + std::to_string(kind));
        check_statistics(state);
        // Error flags and reusable buffers must permit the next valid call.
        run_case(fixture, parameters, {1, 1, 1}, state);
    }
}

} // namespace

int main() {
    try {
        std::cout << "CUDA kernel reference device: " << ipc_gpu::device_name() << '\n'
                  << "CPU reference backend: " << ipc_simd::tile_backend_name() << '\n'
                  << "Tolerance: |GPU-CPU| <= " << absolute_tolerance << " + "
                  << relative_tolerance << " * max(|GPU|,|CPU|); thin records: "
                  << thin_absolute_tolerance << " + " << thin_relative_tolerance
                  << " * max(|GPU|,|CPU|). Bitwise equality is not required.\n";
        const Fixture fixture = make_fixture(129);
        const ipc_gpu::Parameters base{{.13, -9.81, -.07}, 1300, .0009, 2.3, 5.7, .009};
        const std::array<std::size_t, 8> counts{0, 1, 31, 32, 33, 127, 128, 129};
        CheckState state;
        ipc_gpu::reset_statistics();
        check_statistics(state);
        ipc_gpu::compute(nullptr, 0, nullptr, 0, nullptr, 0, base, nullptr, nullptr, nullptr);
        check_statistics(state);
        for (int material = 0; material < 4; ++material) {
            ipc_gpu::Parameters parameters = base;
            if (material & 1) parameters.mu = 0;
            if (material & 2) parameters.lambda = 0;
            if (material == 3) parameters.kpin = parameters.bending_stiffness = 0;
            for (std::size_t count : counts)
                run_case(fixture, parameters, {count, count, count}, state);
        }
        run_case(fixture, base, {129, 0, 0}, state);
        run_case(fixture, base, {0, 129, 0}, state);
        run_case(fixture, base, {0, 0, 129}, state);
        run_case(fixture, base, {1, 33, 127}, state);
        check_nonfinite_failure(fixture, base, state);
        std::cout << "PASS: " << state.components << " derivative components; "
                  << state.expected.calls << " successful calls, " << state.expected.launches
                  << " launches; records=" << state.expected.points << '/' << state.expected.triangles
                  << '/' << state.expected.hinges << " (point/triangle/hinge); max tolerance fraction="
                  << state.maximum_tolerance_fraction << ".\n";
        ipc_gpu::reset_statistics();
        check_statistics(CheckState{});
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
