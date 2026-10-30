// Device-side original-mode smoke: exercises the qualified original contact
// and original CCD components through real CUDA launches. This is not a
// substitute for the full CPU oracle suites.
#include "original_contact.cuh"
#include "original_ccd.cuh"
#include "kernels.h"
#include "linear_pipeline.h"
#include "contact_pipeline.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

__global__ void contact_probe(const ipc_gpu::solver::ContactInput* input,
                              ipc_gpu::solver::ContactOutput* output) {
    *output = ipc_gpu::solver::evaluate_contact_original(*input);
}

__global__ void ccd_probe(const ipc_gpu::original_ccd::Query* query,
                          ipc_gpu::original_ccd::Result* result) {
    *result = ipc_gpu::original_ccd::evaluate_fast(*query);
}

}  // namespace

int gpu_original_mode_device_check() {
    using ipc_gpu::solver::ContactInput;
    using ipc_gpu::solver::ContactOutput;
    using ipc_gpu::solver::ContactKind;
    using ipc_gpu::solver::ContactStatus;
    using ipc_gpu::original_ccd::Query;
    using ipc_gpu::original_ccd::Result;
    using ipc_gpu::original_ccd::Status;

    ContactInput input{};
    input.kind = ContactKind::PointTriangle;
    input.d_hat = 0.005;
    input.k_barrier = 100.0;
    input.dt = 0.01;
    input.eps_v = 0.001;
    input.eps = 1e-12;
    input.hessian_blocks = 1u;
    const double point[3] = {0.2, 0.2, 0.004};
    const double triangle[3][3] = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    for (int k = 0; k < 3; ++k) {
        input.positions[0][k] = point[k];
        input.previous[0][k] = point[k];
    }
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) {
            input.positions[i + 1][k] = triangle[i][k];
            input.previous[i + 1][k] = triangle[i][k];
        }

    Query query{};
    query.vertex_face = true;
    query.eps = 1e-12;
    for (int k = 0; k < 3; ++k) {
        query.positions[0][k] = (k == 2) ? 0.001 : point[k];
        query.motion[0][k] = (k == 2) ? -0.01 : 0.0;
    }
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) {
            query.positions[i + 1][k] = triangle[i][k];
            query.motion[i + 1][k] = 0.0;
        }

    ContactInput* device_input = nullptr;
    ContactOutput* device_output = nullptr;
    Query* device_query = nullptr;
    Result* device_result = nullptr;
    if (cudaMalloc(&device_input, sizeof(ContactInput)) != cudaSuccess
        || cudaMalloc(&device_output, sizeof(ContactOutput)) != cudaSuccess
        || cudaMalloc(&device_query, sizeof(Query)) != cudaSuccess
        || cudaMalloc(&device_result, sizeof(Result)) != cudaSuccess) {
        if (device_input) cudaFree(device_input);
        if (device_output) cudaFree(device_output);
        if (device_query) cudaFree(device_query);
        if (device_result) cudaFree(device_result);
        return 1;
    }
    cudaMemcpy(device_input, &input, sizeof(input), cudaMemcpyHostToDevice);
    cudaMemcpy(device_query, &query, sizeof(query), cudaMemcpyHostToDevice);
    contact_probe<<<1, 1>>>(device_input, device_output);
    ccd_probe<<<1, 1>>>(device_query, device_result);
    ContactOutput output{};
    Result result{};
    cudaMemcpy(&output, device_output, sizeof(output), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost);
    const cudaError_t launch = cudaGetLastError();
    cudaFree(device_result);
    cudaFree(device_query);
    cudaFree(device_output);
    cudaFree(device_input);
    if (launch != cudaSuccess) return 1;
    if (output.status != ContactStatus::Ok || output.active == 0) return 1;
    if (result.status == Status::InvalidInput) return 1;
    if (result.status == Status::Ok && !result.collision) return 1;
    return 0;
}

// Actual GpuLinearClipper/GpuContactPipeline original-mode routing controls.
// These are pipeline integration checks, not a substitute for the protected
// CPU-false oracle corpus.
namespace {
namespace gpu = ipc_gpu::solver;
using V = std::array<double, 3>;

void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <class T> struct TestBuffer {
    T* pointer = nullptr;
    std::size_t count = 0;
    explicit TestBuffer(const std::vector<T>& values) : count(values.size()) {
        if (count) {
            cuda_check(cudaMalloc(&pointer, count * sizeof(T)));
            cuda_check(cudaMemcpy(pointer, values.data(), count * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }
    }
    ~TestBuffer() { if (pointer) cudaFree(pointer); }
    std::vector<T> read() const {
        std::vector<T> result(count);
        if (count)
            cuda_check(cudaMemcpy(result.data(), pointer, count * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        return result;
    }
};

double (*xyz(V* p))[3] { return reinterpret_cast<double(*)[3]>(p); }

struct ClipOutcome {
    int status = 0;
    std::array<V, 4> accepted{};
};

// One explicit GpuLinearClipper call. `mask` is the solver-owned {0,1}
// cooperative node mask; an empty vector passes the documented nullptr.
ClipOutcome run_clip_full(const std::array<V, 4>& points, const V& target,
                          gpu::LinearClipMode mode,
                          gpu::OriginalUpdateConvention convention, bool use_ccd,
                          const std::vector<unsigned char>& mask) {
    std::vector<V> positions(points.begin(), points.end());
    std::vector<V> proposed = positions;
    proposed[0] = target;
    TestBuffer<V> x(positions), raw(proposed), accepted(std::vector<V>(4, V{{-91, -92, -93}}));
    TestBuffer<unsigned char> mask_buffer(mask);
    TestBuffer<unsigned char> moving(std::vector<unsigned char>(4, 1));
    TestBuffer<int> blocks(std::vector<int>{0, 1, 2, 3});
    TestBuffer<int> colors(std::vector<int>{0, 1, 1, 1});
    TestBuffer<int> indices(std::vector<int>{0}), error(std::vector<int>{0});
    TestBuffer<std::size_t> offsets(std::vector<std::size_t>{0, 1, 1, 1, 1});
    TestBuffer<gpu::BpContact> contacts(std::vector<gpu::BpContact>{{{0, 1, 2, 3}, 0}});
    std::vector<gpu::BpAabb> bounds(4);
    for (int i = 0; i < 4; ++i) {
        bounds[i].lo[0] = -100; bounds[i].lo[1] = -100; bounds[i].lo[2] = -100;
        bounds[i].hi[0] = 100; bounds[i].hi[1] = 100; bounds[i].hi[2] = 100;
    }
    bounds[0].lo[0] = -1; bounds[0].lo[1] = -2; bounds[0].lo[2] = -2;
    bounds[0].hi[0] = 1; bounds[0].hi[1] = 2; bounds[0].hi[2] = 2;
    TestBuffer<gpu::BpAabb> boxes(bounds);
    gpu::DeformableScene scene{};
    scene.nodes = 4; scene.deformable = moving.pointer; scene.node_blocks = blocks.pointer;
    gpu::DeformableState state{};
    state.positions = xyz(x.pointer);
    gpu::ElasticWorkspace work{};
    work.proposed = xyz(raw.pointer); work.error = error.pointer;
    gpu::BroadPhaseView bp{};
    bp.contacts = contacts.pointer; bp.contact_count = 1;
    gpu::GpuLinearClipper clipper;
    clipper.clip_deformable_color(scene, state, work, bp, offsets.pointer, indices.pointer,
        boxes.pointer, colors.pointer, 0, xyz(accepted.pointer), error.pointer, use_ccd,
        nullptr, {}, false, mode, convention,
        mask.empty() ? nullptr : mask_buffer.pointer);
    cuda_check(cudaDeviceSynchronize());
    ClipOutcome outcome;
    outcome.status = error.read()[0];
    const auto accepted_values = accepted.read();
    if (accepted_values.size() != outcome.accepted.size())
        throw std::runtime_error("clipper accepted coordinate count mismatch");
    std::copy(accepted_values.begin(), accepted_values.end(), outcome.accepted.begin());
    const auto physical = x.read();
    if (physical != positions)
        throw std::runtime_error("clipper published physical state");
    return outcome;
}

std::array<double, 3> run_clip(const std::array<V, 4>& points, const V& target,
                               gpu::LinearClipMode mode, bool expected_error) {
    const auto outcome = run_clip_full(points, target, mode,
        gpu::OriginalUpdateConvention::StoredNoop, true, {});
    const auto status = outcome.status;
    if ((status != 0) != expected_error)
        throw std::runtime_error("original clipper error policy mismatch");
    return outcome.accepted[0];
}

int clipper_route_check() {
    const std::array<V, 4> points{{V{{0, .25, 1}}, V{{-10, -10, 0}},
                                   V{{10, -10, 0}}, V{{0, 10, 0}}}};
    // Ordinary OriginalSolver has no positive floor: a target below 1e-8 but
    // still a healthy positive gap is accepted unchanged from the CPU TOI.
    auto near = points;
    near[0] = V{{0, .25, 5e-9}};
    const V approaching{{0, .25, 2.5e-9}};
    const auto ordinary = run_clip(near, approaching, gpu::LinearClipMode::OriginalSolver, false);
    const auto guess = run_clip(near, approaching, gpu::LinearClipMode::OriginalColoredGuess, false);
    if (!(std::fabs(ordinary[2] - 2.5e-9) <= 1e-22))
        throw std::runtime_error("ordinary OriginalSolver imposed a predictor floor");
    if (!(std::fabs(guess[2] - 5e-9) <= 1e-22))
        throw std::runtime_error("OriginalColoredGuess missed min(startgap,1e-8)");
    // Positive-band t0 stays a healthy unchanged endpoint.
    auto band = points;
    band[0] = V{{0, .25, 1e-12}};
    const auto t0 = run_clip(band, V{{0, .25, -1}}, gpu::LinearClipMode::OriginalSolver, false);
    if (t0[2] != 1e-12)
        throw std::runtime_error("positive-band original t0 was not an unchanged no-op");
    // A true zero initial intersection maps to the explicit initial-contact
    // error rather than a backtrack/freeze.
    auto zero = points;
    zero[0] = V{{0, .25, 0}};
    run_clip(zero, V{{0, .25, 1}}, gpu::LinearClipMode::OriginalSolver, true);
    return 0;
}

// CPU selected-vertex dispatch matrix for the 1e-28 squared cutoff, exercised
// through the production clipper (not the isolated helper). The CPU freezes a
// selected cooperative vertex only while `squaredNorm(steps) < 1e-28`; exactly
// 1e-28 and one ULP above are stored moves. The same rule applies to original
// mode with CCD on or off and to exact mode with CCD off; unselected
// (batch/baseline) vertices keep the small move in every case.
int selected_cutoff_check() {
    // CCD off: the stored raw displacement is committed verbatim from origin.
    const std::array<V, 4> flat{{V{{0, 0, 0}}, V{{0, .25, 1}},
                                 V{{-10, -10, 0}}, V{{10, -10, 0}}}};
    // CCD on: a healthy positive gap, so the ORIGINAL TOI is a miss.
    const std::array<V, 4> above_triangle{{V{{0, .25, 1}}, V{{-10, -10, 0}},
                                           V{{10, -10, 0}}, V{{0, 10, 0}}}};
    // z*z == 1e-28 exactly; the neighbours are one ULP below/above.
    const V below{{0, 0, 0x1.6849b86a12b9ap-47}};
    const V equal{{0, 0, 0x1.6849b86a12b9bp-47}};
    const V above{{0, 0, 0x1.6849b86a12b9cp-47}};
    const std::vector<unsigned char> selected{1, 0, 0, 0};
    const std::vector<unsigned char> unselected{0, 0, 0, 0};
    for (const bool ccd : {false, true}) {
        const auto& points = ccd ? above_triangle : flat;
        const double start_z = points[0][2];
        // run_clip_full consumes an ABSOLUTE target, while these controls
        // specify a displacement. Keep x/y at the original point and add the
        // delta componentwise, including the CCD-on nonzero starting point.
        const auto target = [&](const V& delta) {
            V end = points[0];
            for (int axis = 0; axis < 3; ++axis) end[axis] += delta[axis];
            return end;
        };
        // propose_nodes stores raw = (clipped - start) with the same plain
        // subtraction, so mirror it here instead of assuming exactness.
        const auto raw = [&](const V& value) { return (start_z + value[2]) - start_z; };
        const auto frozen = [&](const V& value) {
            const double r = raw(value);
            return r * r < 1e-28;
        };
        const auto endpoint = [&](const V& value) {
            return frozen(value) ? start_z : start_z + raw(value);
        };
        const auto check = [&](const V& value, const std::vector<unsigned char>& mask,
                               bool expect_frozen, const char* message) {
            const auto run = run_clip_full(points, target(value), gpu::LinearClipMode::OriginalSolver,
                gpu::OriginalUpdateConvention::CooperativeSquaredNorm, ccd, mask);
            if (run.status != 0)
                throw std::runtime_error("original selected-cutoff run failed");
            const double wanted = expect_frozen ? start_z
                : (mask[0] ? endpoint(value) : start_z + raw(value));
            if (!(run.accepted[0][2] == wanted))
                throw std::runtime_error(message);
        };
        check(below, selected, frozen(below), "selected sub-1e-28 step was not frozen");
        check(equal, selected, frozen(equal), "selected step at 1e-28 was discarded");
        check(above, selected, frozen(above), "selected step above 1e-28 was discarded");
        check(below, unselected, false, "unselected batch vertex lost its small move");
        // Robust exact mode with CCD disabled applies the same selected rule;
        // without a mask (default direct callers) nothing is filtered.
        const auto robust_low = run_clip_full(points, target(below), gpu::LinearClipMode::Robust,
            gpu::OriginalUpdateConvention::CooperativeSquaredNorm, ccd, selected);
        if (robust_low.status != 0)
            throw std::runtime_error("robust selected cutoff run failed");
        if (ccd) {
            if (!(robust_low.accepted[0][2] == start_z + raw(below)))
                throw std::runtime_error("exact+CCD-on discarded a representable small move");
        } else if (!(robust_low.accepted[0][2] == start_z)) {
            throw std::runtime_error("exact+CCD-off selected sub-1e-28 step was not frozen");
        }
        const auto robust_default = run_clip_full(points, target(below), gpu::LinearClipMode::Robust,
            gpu::OriginalUpdateConvention::StoredNoop, ccd, {});
        if (robust_default.status != 0
            || !(robust_default.accepted[0][2] == start_z + raw(below)))
            throw std::runtime_error("default robust caller lost its small move");
    }
    // A mask value outside {0,1} fails closed before any publish.
    const std::vector<unsigned char> invalid{1, 2, 0, 0};
    const auto bad = run_clip_full(flat, equal, gpu::LinearClipMode::OriginalSolver,
        gpu::OriginalUpdateConvention::CooperativeSquaredNorm, false, invalid);
    if (bad.status == 0)
        throw std::runtime_error("invalid cooperative mask was accepted");
    return 0;
}

// Bounded-parallel original resolve: several same-color nodes with DISJOINT
// incident contacts must each commit their own raw displacement from their own
// worker, without sharing an arena or a scratch result.
int parallel_worker_check() {
    const int n = 9;
    std::vector<V> positions(n);
    for (int i = 0; i < 6; ++i)
        positions[i] = V{{0.2 + 0.05 * i, 0.2, 1.0}};
    positions[6] = V{{0, 0, 0}};
    positions[7] = V{{1, 0, 0}};
    positions[8] = V{{0, 1, 0}};
    std::vector<V> proposed = positions;
    for (int i = 0; i < 6; ++i)
        proposed[i] = V{{positions[i][0], positions[i][1],
                         positions[i][2] - 1e-3 * (i + 1)}};
    TestBuffer<V> x(positions), raw(proposed),
        accepted(std::vector<V>(n, V{{-91, -92, -93}}));
    TestBuffer<unsigned char> deformable(std::vector<unsigned char>(n, 1));
    std::vector<int> blocks(n), colors(n, 1);
    for (int i = 0; i < n; ++i) blocks[i] = i;
    for (int i = 0; i < 6; ++i) colors[i] = 0;
    TestBuffer<int> block_buffer(blocks), color_buffer(colors);
    std::vector<gpu::BpContact> contact_values(6);
    for (int i = 0; i < 6; ++i) contact_values[i] = gpu::BpContact{{i, 6, 7, 8}, 0};
    TestBuffer<gpu::BpContact> contacts(contact_values);
    TestBuffer<std::size_t> offsets(std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 6, 6, 6});
    TestBuffer<int> indices(std::vector<int>{0, 1, 2, 3, 4, 5}), error(std::vector<int>{0});
    std::vector<gpu::BpAabb> bounds(n);
    for (int i = 0; i < n; ++i) {
        bounds[i].lo[0] = -100; bounds[i].lo[1] = -100; bounds[i].lo[2] = -100;
        bounds[i].hi[0] = 100; bounds[i].hi[1] = 100; bounds[i].hi[2] = 100;
    }
    TestBuffer<gpu::BpAabb> boxes(bounds);
    gpu::DeformableScene scene{};
    scene.nodes = n; scene.deformable = deformable.pointer;
    scene.node_blocks = block_buffer.pointer;
    gpu::DeformableState state{};
    state.positions = xyz(x.pointer);
    gpu::ElasticWorkspace work{};
    work.proposed = xyz(raw.pointer); work.error = error.pointer;
    gpu::BroadPhaseView bp{};
    bp.contacts = contacts.pointer; bp.contact_count = 6;
    gpu::GpuLinearClipper clipper;
    clipper.clip_deformable_color(scene, state, work, bp, offsets.pointer,
        indices.pointer, boxes.pointer, color_buffer.pointer, 0,
        xyz(accepted.pointer), error.pointer, true, nullptr, {}, false,
        gpu::LinearClipMode::OriginalSolver,
        gpu::OriginalUpdateConvention::StoredNoop);
    cuda_check(cudaDeviceSynchronize());
    if (error.read()[0] != 0)
        throw std::runtime_error("parallel original resolve failed");
    const auto out = accepted.read();
    for (int i = 0; i < 6; ++i) {
        // Mirror propose_nodes: clipped proposal, then unfused raw and endpoint.
        const double clipped = proposed[i][2];
        const double displacement = clipped - positions[i][2];
        const double endpoint = positions[i][2] + displacement;
        if (!(out[i][2] == endpoint))
            throw std::runtime_error("parallel worker published another node's endpoint");
        if (!(out[i][0] == positions[i][0] && out[i][1] == positions[i][1]))
            throw std::runtime_error("parallel worker moved an untouched axis");
    }
    return 0;
}

// Device-callable CPU FALSE predictor box helper: per-axis padding uses the raw
// intended displacement, not target-start. The frozen 0.97 control shows a
// 3-ULP difference that shrinks a target-only box.
int original_box_check() {
    const double start = std::strtod("-0x1.a461117f5232ap-1", nullptr);
    const double xhat = std::strtod("-0x1.6d4f72c869f3ap-1", nullptr);
    const double raw = (xhat - start) * 0.97;
    const double target = start + raw;
    std::vector<double> start_array{start, 0.0, 0.0};
    std::vector<double> target_array{target, 0.0, 0.0};
    std::vector<double> raw_array{raw, 0.0, 0.0};
    TestBuffer<double> device_start(start_array), device_target(target_array),
        device_raw(raw_array);
    TestBuffer<gpu::BpAabb> boxes(std::vector<gpu::BpAabb>(1));
    gpu::make_original_guess_boxes(device_start.pointer, device_target.pointer,
        device_raw.pointer, 1, boxes.pointer);
    cuda_check(cudaDeviceSynchronize());
    const auto box = boxes.read()[0];
    const double pad = std::max(1.0e-8,
        std::max(0.1 * std::fabs(raw), 64.0 * DBL_EPSILON
            * std::max(std::fabs(start), std::fabs(target))));
    const double target_only_pad = std::max(1.0e-8,
        std::max(0.1 * std::fabs(target - start), 64.0 * DBL_EPSILON
            * std::max(std::fabs(start), std::fabs(target))));
    if (box.lo[0] != std::min(start, target) - pad)
        throw std::runtime_error("raw-padded original guess box lower bound changed");
    if (box.hi[0] != std::max(start, target) + pad)
        throw std::runtime_error("raw-padded original guess box upper bound changed");
    if (box.lo[0] == std::min(start, target) - target_only_pad)
        throw std::runtime_error("target-only padding was silently accepted");
    if (pad == target_only_pad)
        throw std::runtime_error("frozen 0.97 raw/target padding control degenerated");
    // Stationary axes get the 1e-8 floor only.
    if (!(box.lo[1] == -1.0e-8 && box.hi[1] == 1.0e-8))
        throw std::runtime_error("stationary-axis original guess padding changed");
    // No target-only fallback: the raw array is required.
    bool rejected = false;
    try {
        gpu::make_original_guess_boxes(device_start.pointer, device_target.pointer,
            nullptr, 1, boxes.pointer);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("original guess boxes accepted a null raw array");
    return 0;
}

int pipeline_route_check() {
    const double s = std::ldexp(1.0, -22);
    const std::array<V, 4> points{{V{{0, 0, 0}}, V{{0, s, 0}},
                                   V{{-.5 * s, .5 * s, .125 * s}},
                                   V{{.5 * s, .5 * s, .125 * s}}}};
    std::vector<V> positions(points.begin(), points.end());
    TestBuffer<V> x(positions);
    TestBuffer<unsigned char> deformable(std::vector<unsigned char>(4, 1));
    TestBuffer<unsigned char> solid(std::vector<unsigned char>(4, 0));
    TestBuffer<int> ids(std::vector<int>{0, 1, 2, 3}), error(std::vector<int>{0});
    TestBuffer<gpu::BpContact> contacts(std::vector<gpu::BpContact>{{{0, 1, 2, 3}, 1}});
    TestBuffer<ipc_gpu::Derivatives> systems(std::vector<ipc_gpu::Derivatives>(4));
    gpu::BroadPhaseView bp{};
    bp.contacts = contacts.pointer; bp.contact_count = 1;
    bp.block_colors = ids.pointer; bp.color_count = 4;
    gpu::DeformableScene scene{};
    scene.nodes = 4; scene.node_blocks = ids.pointer; scene.deformable = deformable.pointer;
    scene.solid = solid.pointer;
    gpu::DeformableState state{};
    state.positions = state.previous = xyz(x.pointer);
    gpu::ElasticWorkspace work{};
    work.systems = systems.pointer; work.error = error.pointer;
    gpu::ContactParameters p{.8 * s, 37, 1.0 / 32.0, 0, .125, 1e-12};
    gpu::GpuContactPipeline pipe;
    pipe.rebuild(bp, 4, ids.pointer);
    std::vector<ipc_gpu::Derivatives> robust_first, original, robust_last;
    for (int pass = 0; pass < 3; ++pass) {
        cuda_check(cudaMemset(systems.pointer, 0, 4 * sizeof(ipc_gpu::Derivatives)));
        cuda_check(cudaMemset(error.pointer, 0, sizeof(int)));
        work.systems = systems.pointer; work.error = error.pointer;
        const auto policy = pass == 1 ? gpu::ContactGeometryPolicy::Original
                                      : gpu::ContactGeometryPolicy::Robust;
        pipe.assemble_color(scene, state, work, {}, {}, {}, {}, p, 0,
            gpu::ContactAssemblyPhase::Deformable, gpu::ContactAssemblyMode::ColorSweep,
            {}, policy);
        cuda_check(cudaDeviceSynchronize());
        if (error.read()[0] != 0)
            throw std::runtime_error("mode-routed contact assembly failed");
        const auto out = systems.read();
        if (pass == 0) robust_first = out;
        else if (pass == 1) original = out;
        else robust_last = out;
    }
    bool robust_preserved = true;
    for (int k = 0; k < 3; ++k)
        robust_preserved = robust_preserved
            && robust_first[0].gradient[k] == robust_last[0].gradient[k];
    for (int k = 0; k < 9; ++k)
        robust_preserved = robust_preserved
            && robust_first[0].hessian[k] == robust_last[0].hessian[k];
    if (!robust_preserved)
        throw std::runtime_error("returning to robust contact geometry was not preserved");
    double difference = 0.0;
    for (int k = 0; k < 3; ++k)
        difference = std::max(difference,
            std::fabs(robust_first[0].gradient[k] - original[0].gradient[k]));
    for (int k = 0; k < 9; ++k)
        difference = std::max(difference,
            std::fabs(robust_first[0].hessian[k] - original[0].hessian[k]));
    if (!(difference > 0.0))
        throw std::runtime_error("original contact geometry was not actually routed");
    return 0;
}
}  // namespace

int gpu_original_integration_check() {
    if (clipper_route_check() != 0) return 1;
    if (parallel_worker_check() != 0) return 1;
    if (selected_cutoff_check() != 0) return 1;
    if (original_box_check() != 0) return 1;
    if (pipeline_route_check() != 0) return 1;
    return 0;
}
