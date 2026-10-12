#include "solid_types.h"
#include "volumetric_corotated_energy.h"
#if defined(IPC_GPU_HOST_ARITHMETIC_TEST)
#include "solid_math.cuh"
#endif

#include <Eigen/Eigenvalues>
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
#include <type_traits>
#include <vector>

namespace {

constexpr double absolute_tolerance = 2e-9;
constexpr double relative_tolerance = 2e-9;
constexpr double canary = 1234567.25;

struct Record {
    ipc_gpu::Solid input;
    TetRestData rest;
    int role;
};
struct CheckState {
    std::uint64_t components = 0;
    double maximum_tolerance_fraction = 0;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<Record> make_fixture(std::size_t count) {
    std::vector<Record> records(count);
    for (std::size_t entry = 0; entry < count; ++entry) {
        const int sample = static_cast<int>(entry / 4), kind = sample % 16;
        const double angle = .071 * sample;
        const Mat33 rotation = Eigen::AngleAxisd(angle,
            Vec3(.3, -.2, 1).normalized()).toRotationMatrix();
        Mat33 deformation;
        deformation << 1.07 + .013 * sample, -.13, .07,
            .03, .89 - .002 * sample, -.11, -.06, .17, 1.13;
        deformation = rotation * deformation;
        if (kind == 1) deformation = Mat33::Identity();
        if (kind == 2) deformation = rotation;
        if (kind == 3) deformation = rotation * Vec3(-.8, 1.1, .7).asDiagonal();
        if (kind == 4) deformation = Vec3(0, .9, 1.2).asDiagonal();
        if (kind == 5) deformation.setZero();
        if (kind == 6) deformation = rotation * Vec3(1e-10, .8, 1.3).asDiagonal();
        if (kind == 7) deformation = Vec3(0, 0, 1).asDiagonal();
        if (kind == 8) deformation = rotation * Vec3(1e-13, 2e-13, -1e-13).asDiagonal();
        if (kind == 9) deformation = rotation * Vec3(1e8, 2e8, -1e8).asDiagonal();
        if (kind == 10) deformation = Vec3(1, 0, 0).asDiagonal();
        if (kind == 11) deformation = Vec3(1, 1, 0).asDiagonal();
        if (kind == 12) deformation << 0, .5, 0, 0, 1, .2, 0, 0, 1.3;
        if (kind == 13) deformation << 1, .5, 0, 0, 0, .2, 0, 0, 1.3;
        if (kind == 14) deformation << 1, .5, 0, 0, 1, .2, 0, 0, 0;
        if (kind == 15) deformation = Vec3(-1.2, -.8, -.3).asDiagonal();

        Mat33 dm;
        dm << 1.3, .1, -.2, -.1, 1.1, .2, .04, -.13, .9;
        dm *= .3 + .07 * (sample % 11);
        // These records exercise exact zero branches of QR-SVD. Identity rest
        // avoids injecting an arbitrary roundoff perturbation at rank changes.
        if ((kind >= 4 && kind <= 8) || (kind >= 10 && kind <= 14)) dm.setIdentity();
        const Vec3 origin = (kind >= 4 && kind <= 8) || (kind >= 10 && kind <= 14)
            ? Vec3::Zero() : Vec3(.31, -.19, .23);
        const std::vector<Vec3> rest_positions = {
            origin, origin + dm.col(0), origin + dm.col(1), origin + dm.col(2)};
        auto& record = records[entry];
        record.role = static_cast<int>(entry % 4);
        record.rest = EFEMInitializeElasticMaterialState(rest_positions, {0, 1, 2, 3})[0];
        for (int vertex = 0; vertex < 4; ++vertex) {
            Vec3 position = origin;
            if (vertex) position += deformation * dm.col(vertex - 1);
            for (int axis = 0; axis < 3; ++axis) record.input.positions[3 * vertex + axis] = position[axis];
        }
        for (int row = 0; row < 3; ++row) {
            record.input.shape[row] = record.rest.grad_N[record.role][row];
            for (int column = 0; column < 3; ++column)
                record.input.dm_inverse[3 * row + column] = record.rest.Dm_inverse(row, column);
        }
        record.input.measure = record.rest.measure;
    }
    return records;
}

ipc_gpu::Derivatives reference(const Record& record, const ipc_gpu::SolidParameters& parameters) {
    std::vector<Vec3> positions(4);
    for (int vertex = 0; vertex < 4; ++vertex)
        for (int axis = 0; axis < 3; ++axis)
            positions[vertex][axis] = record.input.positions[3 * vertex + axis];
    const Mat33 f = ElementF(0, positions, {0, 1, 2, 3}, {record.rest});
    CorotatedCache cache;
    cache.UpdateCache(f, CorotatedCacheMode::Lean);
    const auto expected = EFEMElementNodeGradientAndPBGSBlock(cache, f, record.rest,
        parameters.mu, parameters.lambda, record.role);
    ipc_gpu::Derivatives result;
    for (int row = 0; row < 3; ++row) {
        result.gradient[row] = expected.first[row];
        for (int column = 0; column < 3; ++column)
            result.hessian[3 * row + column] = expected.second(row, column);
    }
    return result;
}

void compute(const ipc_gpu::Solid* records, std::size_t count,
    const ipc_gpu::SolidParameters& parameters, ipc_gpu::Derivatives* outputs) {
#if defined(IPC_GPU_HOST_ARITHMETIC_TEST)
    for (std::size_t entry = 0; entry < count; ++entry) {
        outputs[entry] = ipc_gpu::evaluate(records[entry], parameters);
        for (double value : outputs[entry].gradient)
            require(std::isfinite(value), "nonfinite host arithmetic gradient");
        for (double value : outputs[entry].hessian)
            require(std::isfinite(value), "nonfinite host arithmetic block");
    }
#else
    ipc_gpu::compute_solids(records, count, parameters, outputs);
#endif
}

void compare(double actual, double expected, const std::string& context, CheckState& state) {
    require(std::isfinite(actual) && std::isfinite(expected), context + " is nonfinite");
    const double tolerance = absolute_tolerance + relative_tolerance * std::max(std::abs(actual), std::abs(expected));
    const double fraction = std::abs(actual - expected) / tolerance;
    state.maximum_tolerance_fraction = std::max(state.maximum_tolerance_fraction, fraction);
    ++state.components;
    if (fraction > 1) {
        std::ostringstream message;
        message << std::setprecision(17) << context << ": actual=" << actual
                << " expected=" << expected << " tolerance=" << tolerance;
        throw std::runtime_error(message.str());
    }
}

void run_case(const std::vector<Record>& fixture, std::size_t count,
    const ipc_gpu::SolidParameters& parameters, CheckState& state) {
    std::vector<ipc_gpu::Solid> inputs;
    for (std::size_t entry = 0; entry < count; ++entry) inputs.push_back(fixture[entry].input);
    std::vector<ipc_gpu::Derivatives> outputs(count + 1);
    for (auto& output : outputs) {
        std::fill(std::begin(output.gradient), std::end(output.gradient), canary);
        std::fill(std::begin(output.hessian), std::end(output.hessian), canary);
    }
    compute(count ? inputs.data() : nullptr, count, parameters, outputs.data());
    for (std::size_t entry = 0; entry < count; ++entry) {
        const auto expected = reference(fixture[entry], parameters);
        const std::string context = "count=" + std::to_string(count) + " entry=" + std::to_string(entry)
            + " role=" + std::to_string(fixture[entry].role);
        Mat33 block;
        for (int axis = 0; axis < 3; ++axis)
            compare(outputs[entry].gradient[axis], expected.gradient[axis],
                context + " gradient[" + std::to_string(axis) + "]", state);
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column) {
                const int component = 3 * row + column;
                compare(outputs[entry].hessian[component], expected.hessian[component],
                    context + " block[" + std::to_string(component) + "]", state);
                block(row, column) = outputs[entry].hessian[component];
            }
        const double block_tolerance = 1e-12 * (1 + block.norm());
        require((block - block.transpose()).norm() <= block_tolerance, context + " block is asymmetric");
        Eigen::SelfAdjointEigenSolver<Mat33> eigen(block);
        require(eigen.eigenvalues().minCoeff() >= -block_tolerance, context + " block is not PSD");
    }
    for (double value : outputs.back().gradient) require(value == canary, "gradient tail overwritten");
    for (double value : outputs.back().hessian) require(value == canary, "block tail overwritten");
}

void check_nonfinite_failure(const Record& record, const ipc_gpu::SolidParameters& parameters) {
    for (int kind = 0; kind < 4; ++kind) {
        auto invalid = record.input;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (kind == 0) invalid.positions[0] = nan;
        if (kind == 1) invalid.dm_inverse[0] = nan;
        if (kind == 2) invalid.shape[0] = nan;
        if (kind == 3) invalid.measure = nan;
        bool threw = false;
        ipc_gpu::Derivatives output;
        try { compute(&invalid, 1, parameters, &output); }
        catch (const std::exception&) { threw = true; }
        require(threw, "nonfinite solid record did not report an error");
        compute(&record.input, 1, parameters, &output);
    }
}

} // namespace

int main() {
    static_assert(std::is_trivially_copyable_v<ipc_gpu::Solid>);
    static_assert(std::is_standard_layout_v<ipc_gpu::Solid>);
    try {
#if defined(IPC_GPU_HOST_ARITHMETIC_TEST)
        std::cout << "Explicit HOST arithmetic test; this does not validate CUDA execution.\n";
#else
        std::cout << "CUDA solid reference device: " << ipc_gpu::device_name() << '\n';
#endif
        std::cout << "Signed TGSL polar/PSD PBGS comparison: |actual-reference| <= "
                  << absolute_tolerance << " + " << relative_tolerance << " * max(|actual|,|reference|).\n";
        const auto fixture = make_fixture(257);
        const std::array<std::size_t, 12> counts{0, 1, 7, 8, 9, 31, 32, 33, 127, 128, 129, 257};
        const std::array<ipc_gpu::SolidParameters, 4> materials{{{2.3, 5.7}, {0, 5.7}, {2.3, 0}, {0, 0}}};
        CheckState state;
        compute(nullptr, 0, materials[0], nullptr);
        for (const auto& material : materials)
            for (std::size_t count : counts) run_case(fixture, count, material, state);
        check_nonfinite_failure(fixture[0], materials[0]);
        std::cout << "PASS: " << state.components << " solid derivative components; all four roles, "
                  << "inverted, rank-deficient, zero, near-singular, large-scale records and tails; "
                  << "maximum tolerance fraction=" << state.maximum_tolerance_fraction << ".\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
