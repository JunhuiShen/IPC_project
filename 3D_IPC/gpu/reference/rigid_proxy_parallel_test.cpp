// Native-ready acceptance tests for the isolated rigid proxy-loop prototype.
// NATIVE_UNQUALIFIED until compiled/executed on CUDA hardware. This file does
// not replace or relax any existing rigid-pipeline fixture or tolerance.
#include "rigid_pipeline.h"
#include "ccd.h"
#include "rigid_body_ipc.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu = ipc_gpu::solver;
using D3 = std::array<double, 3>;
using D4 = std::array<double, 4>;
using D9 = std::array<double, 9>;

namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
void cuda_check(cudaError_t value) {
    if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value));
}
template<class T> struct Device {
    T* data = nullptr;
    std::size_t count = 0;
    explicit Device(const std::vector<T>& host) : count(host.size()) {
        if (!count) return;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
        cuda_check(cudaMemcpy(data, host.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Device() { if (data) cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    std::vector<T> get() const {
        std::vector<T> host(count);
        if (count) cuda_check(cudaMemcpy(host.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
};
template<std::size_t N> auto coordinates(Device<std::array<double, N>>& device) -> double (*)[N] {
    return reinterpret_cast<double (*)[N]>(device.data);
}
template<class T> void same(const std::vector<T>& a, const std::vector<T>& b,
    const std::string& label) {
    require(a.size() == b.size() && (a.empty()
        || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0), label);
}
void close(double actual, double expected, const std::string& label) {
    require(std::isfinite(actual) && std::isfinite(expected)
        && std::abs(actual - expected) <= 5e-12 * (1 + std::abs(expected)), label);
}
Vec3 eigen(const D3& value) { return Vec3(value[0], value[1], value[2]); }
Vec4 eigen(const D4& value) { return Vec4(value[0], value[1], value[2], value[3]); }
D3 array3(const Vec3& value) { return {value[0], value[1], value[2]}; }
D4 array4(const Vec4& value) { return {value[0], value[1], value[2], value[3]}; }

class Environment {
    std::string name_, original_;
    bool existed_ = false;
public:
    explicit Environment(const char* name) : name_(name) {
        if (const char* value = std::getenv(name)) { existed_ = true; original_ = value; }
    }
    ~Environment() {
        if (existed_) ::setenv(name_.c_str(), original_.c_str(), 1);
        else ::unsetenv(name_.c_str());
    }
    void assign(const char* value) {
        require(::setenv(name_.c_str(), value, 1) == 0, "cannot set test environment");
    }
};

struct Host {
    double dt = .125;
    std::vector<int> blocks, owners, nodes, colors;
    std::vector<std::size_t> offsets;
    std::vector<unsigned char> translate, rotate;
    std::vector<double> mass;
    std::vector<D9> moment;
    std::vector<D3> reference, positions, proxy_velocity, com, velocity, omega;
    std::vector<D3> previous_com, previous_velocity, previous_omega;
    std::vector<D4> orientation, previous_orientation;
    std::vector<D3> accepted_proxy, accepted_com, accepted_omega, proposed_com, proposed_omega;
    std::vector<D4> accepted_orientation, proposed_orientation;
    std::size_t bodies() const { return blocks.size(); }
};

Host fixture(const std::vector<std::size_t>& counts) {
    Host h;
    const auto bodies = counts.size();
    h.offsets.push_back(0);
    for (const auto count : counts) h.offsets.push_back(h.offsets.back() + count);
    const auto slots = h.offsets.back(), nodes = 2 * slots + 11;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    h.reference.assign(nodes, D3{nan, nan, nan});
    h.positions.assign(nodes, D3{8.25, -3.5, .75});
    h.proxy_velocity.assign(nodes, D3{-7.25, 6.5, -5.75});
    h.accepted_proxy.assign(nodes, D3{nan, nan, nan});
    h.owners.assign(nodes, -1);
    h.nodes.resize(slots);
    h.colors.assign(3 * bodies + 3, 99);
    h.mass.assign(bodies, 2);
    h.moment.assign(bodies, D9{2, .125, 0, .125, 3, .25, 0, .25, 4});
    for (std::size_t body = 0; body < bodies; ++body) {
        const auto block = static_cast<int>(3 * body + 2);
        h.blocks.push_back(block);
        h.colors[block] = body % 5 == 4 ? 11 : 7;
        h.translate.push_back(body % 4 < 2);
        h.rotate.push_back(body % 4 == 0 || body % 4 == 2);
        h.com.push_back(D3{2.0 * body, .25 + .125 * body, -.5});
        h.velocity.push_back(D3{-.375, .625, -.875});
        h.omega.push_back(D3{.03125, -.015625, .0234375});
        const double angle = .01 * (body + 1);
        h.orientation.push_back(D4{std::cos(angle / 2), 0, 0, std::sin(angle / 2)});
        h.previous_orientation.push_back(h.orientation.back());
        h.previous_com.push_back(array3(eigen(h.com.back()) - Vec3(.03125, -.015625, .0078125)));
        h.previous_velocity.push_back(D3{.0625, -.125, .1875});
        h.previous_omega.push_back(D3{-.09375, .15625, -.21875});
        h.accepted_com.push_back(array3(eigen(h.com.back()) + Vec3(.0625, -.03125, .015625)));
        h.accepted_omega.push_back(D3{.15625, -.09375, .046875});
        h.accepted_orientation.push_back(D4{std::cos(.035), 0, std::sin(.035), 0});
        h.proposed_com.push_back(h.accepted_com.back());
        h.proposed_omega.push_back(D3{.125, -.25, .375});
        h.proposed_orientation.push_back(h.orientation.back());
        for (auto slot = h.offsets[body]; slot < h.offsets[body + 1]; ++slot) {
            // Authoritative CSR order is DESCENDING odd global vertex IDs.
            const auto node = 5 + 2 * (slots - 1 - slot);
            h.nodes[slot] = static_cast<int>(node);
            h.owners[node] = static_cast<int>(body);
            const int local = static_cast<int>(slot - h.offsets[body]);
            h.reference[node] = D3{double(local % 31 - 15) / 32,
                double(local % 23 - 11) / 64, double(local % 19 - 9) / 128};
            h.positions[node] = array3(world_space_position(eigen(h.reference[node]),
                eigen(h.com[body]), eigen(h.orientation[body])));
            h.accepted_proxy[node] = array3(eigen(h.positions[node]) + Vec3(.0625, -.03125, .015625));
            h.accepted_proxy[node][0] = std::nextafter(h.accepted_proxy[node][0],
                std::numeric_limits<double>::infinity());
            if (slot % 17 == 0) h.accepted_proxy[node][1] = -0.0;
        }
    }
    return h;
}

struct Image {
    int error = 0;
    std::vector<D3> positions, proxy_velocity, com, velocity, omega;
    std::vector<D4> orientation;
    std::vector<D3> previous_com, previous_velocity, previous_omega;
    std::vector<D4> previous_orientation;
    std::vector<D3> accepted_proxy, accepted_com, accepted_omega;
    std::vector<D4> accepted_orientation;
};
void physical_same(const Image& a, const Image& b, const std::string& label) {
    same(a.positions, b.positions, label + " positions");
    same(a.proxy_velocity, b.proxy_velocity, label + " proxy velocity");
    same(a.com, b.com, label + " COM");
    same(a.velocity, b.velocity, label + " body velocity");
    same(a.omega, b.omega, label + " omega");
    same(a.orientation, b.orientation, label + " quaternion");
    same(a.previous_com, b.previous_com, label + " previous COM");
    same(a.previous_velocity, b.previous_velocity, label + " previous velocity");
    same(a.previous_omega, b.previous_omega, label + " previous omega");
    same(a.previous_orientation, b.previous_orientation, label + " previous quaternion");
}
void image_same(const Image& a, const Image& b, const std::string& label) {
    require(a.error == b.error, label + " error class differs");
    physical_same(a, b, label);
    same(a.accepted_proxy, b.accepted_proxy, label + " accepted proxy scratch");
    same(a.accepted_com, b.accepted_com, label + " accepted COM scratch");
    same(a.accepted_omega, b.accepted_omega, label + " accepted omega scratch");
    same(a.accepted_orientation, b.accepted_orientation, label + " accepted quaternion scratch");
}

struct DeviceFixture {
    Device<int> blocks, owners, nodes, colors, error;
    Device<std::size_t> offsets;
    Device<unsigned char> translate, rotate;
    Device<double> mass;
    Device<D9> moment;
    Device<D3> reference, positions, proxy_velocity, previous_positions, predicted;
    Device<D3> com, velocity, omega, previous_com, previous_velocity, previous_omega;
    Device<D4> orientation, previous_orientation;
    Device<D3> accepted_proxy, accepted_com, accepted_omega, proposed_com, proposed_omega;
    Device<D4> accepted_orientation, proposed_orientation;
    Device<ipc_gpu::Derivatives> systems;
    gpu::RigidScene scene{};
    gpu::RigidState state{};
    gpu::RigidWorkspace work{};
    gpu::DeformableState particles{};
    explicit DeviceFixture(const Host& h, int prior_error = 0)
        : blocks(h.blocks), owners(h.owners), nodes(h.nodes), colors(h.colors),
          error(std::vector<int>{prior_error}), offsets(h.offsets),
          translate(h.translate), rotate(h.rotate), mass(h.mass), moment(h.moment),
          reference(h.reference), positions(h.positions), proxy_velocity(h.proxy_velocity),
          previous_positions(h.positions), predicted(h.positions), com(h.com), velocity(h.velocity),
          omega(h.omega), previous_com(h.previous_com), previous_velocity(h.previous_velocity),
          previous_omega(h.previous_omega), orientation(h.orientation),
          previous_orientation(h.previous_orientation), accepted_proxy(h.accepted_proxy),
          accepted_com(h.accepted_com), accepted_omega(h.accepted_omega),
          proposed_com(h.proposed_com), proposed_omega(h.proposed_omega),
          accepted_orientation(h.accepted_orientation), proposed_orientation(h.proposed_orientation),
          systems(std::vector<ipc_gpu::Derivatives>(h.bodies())) {
        scene = {h.bodies(), blocks.data, owners.data, coordinates(reference), offsets.data,
            nodes.data, mass.data, coordinates(moment), translate.data, rotate.data};
        state = {coordinates(com), coordinates(velocity), coordinates(orientation), coordinates(omega),
            coordinates(previous_com), coordinates(previous_velocity), coordinates(previous_orientation),
            coordinates(previous_omega)};
        work = {systems.data, coordinates(proposed_com), coordinates(proposed_omega),
            coordinates(proposed_orientation), coordinates(accepted_com),
            coordinates(accepted_orientation), coordinates(accepted_omega)};
        particles = {coordinates(positions), coordinates(proxy_velocity),
            coordinates(previous_positions), coordinates(predicted)};
    }
    Image get() const {
        cuda_check(cudaDeviceSynchronize());
        return {error.get()[0], positions.get(), proxy_velocity.get(), com.get(), velocity.get(),
            omega.get(), orientation.get(), previous_com.get(), previous_velocity.get(),
            previous_omega.get(), previous_orientation.get(), accepted_proxy.get(),
            accepted_com.get(), accepted_omega.get(), accepted_orientation.get()};
    }
};

enum class Operation { None, Translation, RotationCommit, Finish, RotationClip };
Image run(const Host& h, Operation operation, int color = 7, int prior_error = 0,
    gpu::GpuRigidProxyPipeline* owner = nullptr, gpu::GpuRigidRotationClipper* clipper = nullptr,
    bool publish_rotation = false, const std::vector<gpu::BpContact>& pairs = {},
    bool use_ccd = false, cudaStream_t stream = nullptr) {
    DeviceFixture f(h, prior_error);
    // Pageable H2D uploads belong to the default stream. Join them before
    // launching this test's nonblocking stream; production paths are untouched.
    if(stream)cuda_check(cudaDeviceSynchronize());
    switch (operation) {
    case Operation::None: break;
    case Operation::Translation:
        if (owner) owner->commit_translation_color(f.scene, f.state, f.work, f.particles,
            f.colors.data, color, coordinates(f.accepted_proxy), f.error.data, stream);
        else gpu::commit_rigid_translation_color(f.scene, f.state, f.work, f.particles,
            f.colors.data, color, coordinates(f.accepted_proxy), f.error.data, stream);
        break;
    case Operation::RotationCommit:
        if (owner) owner->commit_rotation_color(f.scene, f.state, f.work, f.particles,
            f.colors.data, color, coordinates(f.accepted_proxy), f.error.data, stream);
        else gpu::commit_rigid_rotation_color(f.scene, f.state, f.work, f.particles,
            f.colors.data, color, coordinates(f.accepted_proxy), f.error.data, stream);
        break;
    case Operation::Finish:
        if (owner) owner->finish_substep(f.scene, f.state, f.particles, h.dt, f.error.data, stream);
        else gpu::finish_rigid_substep(f.scene, f.state, f.particles, h.dt, f.error.data, stream);
        break;
    case Operation::RotationClip: {
        Device<gpu::BpContact> contacts(pairs);
        gpu::BroadPhaseView broadphase{};broadphase.contacts=contacts.data;
        broadphase.contact_count=pairs.size();
        if (clipper) clipper->clip_color(f.scene, f.state, f.work, f.particles, broadphase,
            f.colors.data, color, h.dt, .9, use_ccd, coordinates(f.accepted_proxy), f.error.data, stream);
        else gpu::clip_rigid_rotation_color(f.scene, f.state, f.work, f.particles, broadphase,
            f.colors.data, color, h.dt, .9, use_ccd, coordinates(f.accepted_proxy), f.error.data, stream);
        if (publish_rotation) {
            if (owner) owner->commit_rotation_color(f.scene, f.state, f.work, f.particles,
                f.colors.data, color, coordinates(f.accepted_proxy), f.error.data, stream);
            else gpu::commit_rigid_rotation_color(f.scene, f.state, f.work, f.particles,
                f.colors.data, color, coordinates(f.accepted_proxy), f.error.data, stream);
        }
        break;
    }
    }
    return f.get();
}

void commit_oracle(const Host& h, const Image& actual, Operation operation, int color) {
    auto expected = run(h, Operation::None);
    for (std::size_t body = 0; body < h.bodies(); ++body) {
        const bool selected = h.colors[h.blocks[body]] == color
            && (operation == Operation::Translation ? h.translate[body] : h.rotate[body]);
        if (!selected) continue;
        if (operation == Operation::Translation) expected.com[body] = h.accepted_com[body];
        else {
            expected.orientation[body] = h.accepted_orientation[body];
            expected.omega[body] = h.accepted_omega[body];
        }
        for (auto slot = h.offsets[body]; slot < h.offsets[body + 1]; ++slot)
            expected.positions[h.nodes[slot]] = h.accepted_proxy[h.nodes[slot]];
    }
    image_same(actual, expected, "CPU accepted-copy/mode/color oracle");
}

void finish_oracle(const Host& h, const Image& actual) {
    require(actual.error == 0, "healthy finish oracle received error");
    same(actual.positions, h.positions, "finish changed positions");
    same(actual.orientation, h.orientation, "finish changed orientation");
    same(actual.previous_com, h.previous_com, "finish changed previous COM");
    for (std::size_t body = 0; body < h.bodies(); ++body) {
        Vec3 velocity = Vec3::Zero(), omega = Vec3::Zero();
        if (h.translate[body]) velocity = (eigen(h.com[body]) - eigen(h.previous_com[body])) / h.dt;
        if (h.rotate[body]) omega = eigen(h.omega[body]);
        for (int k = 0; k < 3; ++k) {
            close(actual.velocity[body][k], velocity[k], "CPU COM velocity formula");
            close(actual.omega[body][k], omega[k], "CPU constrained omega formula");
        }
        for (auto slot = h.offsets[body]; slot < h.offsets[body + 1]; ++slot) {
            const int node = h.nodes[slot];
            const Vec3 offset = world_space_position(eigen(h.reference[node]), Vec3::Zero(),
                eigen(h.orientation[body]));
            const Vec3 expected = velocity + omega.cross(offset);
            for (int k = 0; k < 3; ++k)
                close(actual.proxy_velocity[node][k], expected[k], "CPU world/cross proxy velocity formula");
        }
    }
    for (std::size_t node = 0; node < h.owners.size(); ++node)
        if (h.owners[node] < 0)
            require(std::memcmp(actual.proxy_velocity[node].data(), h.proxy_velocity[node].data(),
                sizeof(D3)) == 0, "finish touched deformable padding");
}

Host rotation_fixture(const std::vector<std::size_t>& counts) {
    auto h = fixture(counts);
    for (std::size_t body = 0; body < h.bodies(); ++body) {
        h.translate[body] = h.rotate[body] = 1;
        h.colors[h.blocks[body]] = 7;
        h.omega[body] = D3{0, 0, 0};
        h.proposed_orientation[body] = array4(quaternion_from_angular_velocity(
            eigen(h.previous_orientation[body]), Vec3(.25, -.125, .1875), h.dt));
    }
    return h;
}

void rotation_oracle(const Host& h, const Image& actual) {
    require(actual.error == 0, "healthy rotation oracle received error");
    for (std::size_t body = 0; body < h.bodies(); ++body) {
        const Vec4 accepted = interpolate_orientation_full_arc(eigen(h.orientation[body]),
            eigen(h.proposed_orientation[body]), 1.0);
        const Vec3 omega = angular_velocity_from_orientation_full_arc(accepted,
            eigen(h.previous_orientation[body]), h.dt);
        for (int k = 0; k < 4; ++k)
            close(actual.accepted_orientation[body][k], accepted[k], "CPU accepted quaternion formula");
        for (int k = 0; k < 3; ++k)
            close(actual.accepted_omega[body][k], omega[k], "CPU accepted angular velocity formula");
        for (auto slot = h.offsets[body]; slot < h.offsets[body + 1]; ++slot) {
            const int node = h.nodes[slot];
            const Vec3 expected = world_space_position(eigen(h.reference[node]),
                eigen(h.com[body]), accepted);
            for (int k = 0; k < 3; ++k)
                close(actual.accepted_proxy[node][k], expected[k], "CPU accepted world-position formula");
        }
    }
}

void test_healthy(gpu::GpuRigidProxyPipeline& off, gpu::GpuRigidProxyPipeline& on) {
    for (const std::size_t count : {127u, 128u, 129u, 16383u, 16384u, 16385u, 32769u}) {
        const auto h = fixture({count, 129, 0, 128, 127});
        for (const auto operation : {Operation::Translation, Operation::RotationCommit, Operation::Finish}) {
            const auto reference = run(h, operation);
            image_same(run(h, operation, 7, 0, &off), reference, "OFF/public old healthy");
            const auto actual = run(h, operation, 7, 0, &on);
            image_same(actual, reference, "ON/public old healthy");
            if (operation == Operation::Finish) finish_oracle(h, actual);
            else commit_oracle(h, actual, operation, 7);
        }
    }
    for (const auto bodies : {std::size_t(0), std::size_t(1), std::size_t(64), std::size_t(65)}) {
        std::vector<std::size_t> counts(bodies, 0);
        for (std::size_t b = 0; b < bodies; ++b) counts[b] = b % 3;
        if (bodies) counts[0] = 129;
        const auto h = fixture(counts);
        for (const auto operation : {Operation::Translation, Operation::RotationCommit, Operation::Finish})
            image_same(run(h, operation, 7, 0, &on), run(h, operation), "empty/64/65 owner reuse");
        if (bodies) {
            const auto before = run(h, Operation::None);
            physical_same(run(h, Operation::Translation, 999, 0, &on), before, "inactive translation");
            physical_same(run(h, Operation::RotationCommit, 999, 0, &on), before, "inactive rotation");
        }
    }
}

void test_errors(gpu::GpuRigidProxyPipeline& off, gpu::GpuRigidProxyPipeline& on) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const int fault : {0, 1}) {
        auto h = fixture({32769, 129, 0, 128});
        if (fault == 0) h.accepted_com[0][1] = nan;
        else h.accepted_proxy[h.nodes[h.offsets[1] - 1]][2] = nan;
        const auto reference = run(h, Operation::Translation);
        require(reference.error == gpu::RigidInvalidState, "translation finite fault error class");
        image_same(run(h, Operation::Translation, 7, 0, &off), reference, "translation OFF fault");
        const auto actual = run(h, Operation::Translation, 7, 0, &on);
        image_same(actual, reference, "translation ON fault");
        physical_same(actual, run(h, Operation::None), "translation error published physical state");
    }
    for (const int fixed : {0, 1}) {
        // ONE body makes the scalar prefix deterministic even when it fails.
        auto h = fixture({32769});
        h.translate[0] = h.rotate[0] = fixed ? 0 : 1;
        h.reference[h.nodes[16384]][0] = std::numeric_limits<double>::infinity();
        const auto reference = run(h, Operation::Finish);
        require(reference.error == gpu::RigidInvalidState, "late reference/fixed zero-times-Inf class");
        image_same(run(h, Operation::Finish, 7, 0, &off), reference, "finish OFF partial prefix");
        image_same(run(h, Operation::Finish, 7, 0, &on), reference, "finish ON partial prefix");
        require(reference.proxy_velocity[h.nodes[32768]] == h.proxy_velocity[h.nodes[32768]],
            "scalar finish failed to leave late proxy sentinel untouched");
    }
    auto overflow = fixture({129});
    overflow.com[0][0] = std::numeric_limits<double>::max();
    overflow.previous_com[0][0] = -std::numeric_limits<double>::max();
    const auto reference = run(overflow, Operation::Finish);
    require(reference.error == gpu::RigidInvalidState, "finish finite body arithmetic overflow class");
    image_same(run(overflow, Operation::Finish, 7, 0, &on), reference, "finish overflow fallback");
    auto world_overflow = fixture({129});
    const double maximum = std::numeric_limits<double>::max();
    world_overflow.reference[world_overflow.nodes[64]] = D3{maximum, maximum, maximum};
    const auto world_reference = run(world_overflow, Operation::Finish);
    require(world_reference.error == gpu::RigidInvalidState,
        "finite reference world/cross arithmetic overflow class");
    image_same(run(world_overflow, Operation::Finish, 7, 0, &on), world_reference,
        "finite reference overflow ordered fallback");
    auto gated = fixture({129, 1, 0, 1, 127});
    gated.accepted_com[2][0] = gated.accepted_com[3][0] = nan;
    gated.accepted_proxy[gated.nodes[gated.offsets[4]]][0] = nan;
    for (const auto operation : {Operation::Translation, Operation::RotationCommit}) {
        const auto expected = run(gated, operation);
        require(expected.error == 0, "inactive/disabled accepted-data guard control failed");
        image_same(run(gated, operation, 7, 0, &on), expected,
            "inactive/disabled accepted data was consumed");
    }
    const auto h = fixture({129, 0, 1, 1});
    for (const auto operation : {Operation::Translation, Operation::RotationCommit, Operation::Finish}) {
        const auto expected = run(h, Operation::None, 7, gpu::RigidInvalidDerivative);
        image_same(run(h, operation, 7, gpu::RigidInvalidDerivative, &on), expected,
            "upstream error allowed physical/private writes");
    }
    // Reuse the same owner after fault scratch has been set.
    const auto healthy = fixture({16385});
    image_same(run(healthy, Operation::Finish, 7, 0, &on), run(healthy, Operation::Finish),
        "finish fault flag survived into a later healthy call");
}

void test_rotations(gpu::GpuRigidProxyPipeline& off, gpu::GpuRigidProxyPipeline& on,
    gpu::GpuRigidRotationClipper& clip_off, gpu::GpuRigidRotationClipper& clip_on) {
    for (const auto& counts : {std::vector<std::size_t>{32769},
         std::vector<std::size_t>{129, 127}, std::vector<std::size_t>{0}}) {
        const auto h = rotation_fixture(counts);
        const auto reference = run(h, Operation::RotationClip);
        require(reference.error == 0, "healthy rotation scalar error");
        image_same(run(h, Operation::RotationClip, 7, 0, &off, &clip_off), reference,
            "rotation OFF batch/scalar");
        const auto actual = run(h, Operation::RotationClip, 7, 0, &on, &clip_on);
        image_same(actual, reference, "rotation ON batch/multibody scalar fallback");
        rotation_oracle(h, actual);
        image_same(run(h, Operation::RotationClip, 7, 0, &on, &clip_on, true),
            run(h, Operation::RotationClip, 7, 0, nullptr, nullptr, true), "rotation clip+publish");
    }
    auto bad = rotation_fixture({32769});
    bad.reference[bad.nodes[16384]][1] = std::numeric_limits<double>::infinity();
    for (const bool ambiguous : {false, true}) {
        auto h = bad;
        if (ambiguous) for (double& x : h.proposed_orientation[0]) x = -x;
        // Force the exact ambiguous full-turn relation, independent of target construction.
        if (ambiguous) for (int k = 0; k < 4; ++k)
            h.proposed_orientation[0][k] = -h.orientation[0][k];
        const auto reference = run(h, Operation::RotationClip, 7, 0, nullptr, nullptr, true);
        require(reference.error == (ambiguous ? gpu::RigidRotationFailure : gpu::RigidInvalidState),
            "rotation invalid reference / earlier quaternion error class");
        image_same(run(h, Operation::RotationClip, 7, 0, &off, &clip_off, true), reference,
            "rotation OFF private scratch/error prefix");
        const auto actual = run(h, Operation::RotationClip, 7, 0, &on, &clip_on, true);
        image_same(actual, reference, "rotation ON private scratch/error prefix");
        physical_same(actual, run(h, Operation::None), "rotation error published physical state");
        require(actual.accepted_proxy[h.nodes[32768]] == h.accepted_proxy[h.nodes[32768]],
            "rotation error wrote beyond the ordered failing prefix");
    }
    const auto h = rotation_fixture({129});
    image_same(run(h, Operation::RotationClip, 999, 0, &on, &clip_on),
        run(h, Operation::None), "inactive batch rotation touched scratch");
    image_same(run(h, Operation::RotationClip, 7, gpu::RigidInvalidDerivative, &on, &clip_on),
        run(h, Operation::None, 7, gpu::RigidInvalidDerivative), "rotation upstream error");
    image_same(run(h, Operation::RotationClip, 7, 0, &on, &clip_on),
        run(h, Operation::RotationClip), "rotation owner failed healthy reuse after error");
    auto other_color = rotation_fixture({129, 129});
    other_color.colors[other_color.blocks[1]] = 11;
    other_color.reference[other_color.nodes[other_color.offsets[1]]][0]
        = std::numeric_limits<double>::infinity();
    const auto guarded_reference = run(other_color, Operation::RotationClip);
    require(guarded_reference.error == 0, "inactive reference guard control failed");
    image_same(run(other_color, Operation::RotationClip, 7, 0, &on, &clip_on),
        guarded_reference, "batch rotation consumed another color's invalid reference");
}


void test_rotation_contacts(gpu::GpuRigidProxyPipeline& off, gpu::GpuRigidProxyPipeline& on,
    gpu::GpuRigidRotationClipper& clip_off, gpu::GpuRigidRotationClipper& clip_on,
    Environment& threads) {
    // One moving proxy follows (cos(theta),sin(theta),0), theta0->pi,
    // through a fixed triangle at y=.5. Its endpoint alone is healthy; only
    // curved-path CCD exposes the first interior hit. Remaining owned proxies
    // have XY radius<.5 and cannot hit this plane during the complete arc.
    auto healthy=rotation_fixture({32769});healthy.com[0]=D3{0,0,0};
    healthy.previous_com[0]=healthy.com[0];
    healthy.orientation[0]=healthy.previous_orientation[0]=D4{1,0,0,0};
    healthy.proposed_orientation[0]=D4{0,0,0,1};
    const int mover=healthy.nodes.front();healthy.reference[mover]=D3{1,0,0};
    for(const int node:healthy.nodes) {
        healthy.positions[node]=healthy.reference[node];
        if(node!=mover)require(std::hypot(healthy.reference[node][0],healthy.reference[node][1])<.5,
            "fractional-CCD fixture acquired an unlisted plane crossing");
    }
    healthy.positions[0]=D3{0,.5,-2};healthy.positions[2]=D3{2,.5,-2};
    healthy.positions[4]=D3{1,.5,2};
    require(healthy.owners[0]<0&&healthy.owners[2]<0&&healthy.owners[4]<0,
        "fractional-CCD obstacle is not fixed outside the body CSR");
    const std::vector<gpu::BpContact> contact{{{mover,0,2,4},0}};
    auto malformed=contact;malformed.push_back({{mover,healthy.nodes[1],2,4},0});
    auto bad_reference=healthy;
    bad_reference.reference[bad_reference.nodes[16384]][1]=std::numeric_limits<double>::infinity();
    threads.assign("1");
    const auto reference=run(healthy,Operation::RotationClip,7,0,nullptr,nullptr,true,contact,true);
    require(reference.error==0,"fractional curved-CCD scalar reference failed");
    require(reference.accepted_proxy[mover][0]>0.&&reference.accepted_proxy[mover][1]>0.
        &&reference.accepted_proxy[mover][1]<.5,
        "curved-CCD fixture neither moved nor clipped before the interior plane hit");
    require(std::memcmp(reference.accepted_orientation[0].data(),healthy.proposed_orientation[0].data(),sizeof(D4))!=0,
        "curved-CCD reference accepted its collision-crossing full arc");
    for(unsigned width:{1u,32u,64u,128u}) {
        threads.assign(std::to_string(width).c_str());
        for(int fault=0;fault<4;++fault) {
            const auto& h=fault>=2?bad_reference:healthy;
            const auto& pairs=fault==1||fault==3?malformed:contact;
            const auto expected=run(h,Operation::RotationClip,7,0,nullptr,nullptr,true,pairs,true);
            const int code=fault==0?0:(fault==2?gpu::RigidInvalidState:gpu::RigidContactOwnership);
            require(expected.error==code,"fractional-CCD/pair/reference fault changed scalar error precedence");
            const auto a=run(h,Operation::RotationClip,7,0,&off,&clip_off,true,pairs,true);
            const auto b=run(h,Operation::RotationClip,7,0,&on,&clip_on,true,pairs,true);
            image_same(a,expected,"CCDtrue proxy OFF/reference full image");
            image_same(b,expected,"CCDtrue proxy ON/reference full image");
            if(fault)physical_same(b,run(h,Operation::None),"CCDtrue failure published state");
            else image_same(b,reference,"fractional-CCD pair width changed accepted state");
        }
        const auto upstream=run(healthy,Operation::RotationClip,7,gpu::RigidInvalidDerivative,
            &on,&clip_on,true,contact,true);
        image_same(upstream,run(healthy,Operation::None,7,gpu::RigidInvalidDerivative),
            "CCDtrue upstream error allowed publication or scratch writes");
    }
    threads.assign("1");
    image_same(run(healthy,Operation::RotationClip,7,0,&on,&clip_on,true,contact,true),reference,
        "CCDtrue owner retained a fault flag after pair/reference failures");
    std::cout<<"Rigid proxy CCDtrue: fractional interior hit, 32769 proxies, widths1/32/64/128, "
        <<"pair-before-reference faults, full physical/private bytes and no-publication/reuse PASS\n";
}


Image finish_alias_image(Host h,gpu::GpuRigidProxyPipeline* owner=nullptr) {
    DeviceFixture f(h);
    auto state=f.state;
    // Deliberately overlap body and proxy velocity STORAGE, while retaining
    // unique/in-range node CSR. Only raw/default false-gated owners may use it.
    state.velocity=coordinates(f.proxy_velocity);
    require(state.velocity==f.particles.velocities,"raw alias fixture is not aliased");
    if(owner)owner->finish_substep(f.scene,state,f.particles,h.dt,f.error.data);
    else gpu::finish_rigid_substep(f.scene,state,f.particles,h.dt,f.error.data);
    auto result=f.get();
    result.velocity.assign(result.proxy_velocity.begin(),result.proxy_velocity.begin()+h.bodies());
    return result;
}
void test_nondefault_stream(gpu::GpuRigidProxyPipeline& off,gpu::GpuRigidProxyPipeline& on,
    gpu::GpuRigidRotationClipper& clip_off,gpu::GpuRigidRotationClipper& clip_on) {
    struct Stream {
        cudaStream_t value=nullptr;
        Stream(){cuda_check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
        ~Stream(){if(value)cudaStreamDestroy(value);}
    } stream;
    const auto h=rotation_fixture({16385});
    for(auto operation:{Operation::Translation,Operation::RotationCommit,Operation::Finish,Operation::RotationClip}) {
        const auto expected=run(h,operation,7,0,nullptr,nullptr,false,{},false,stream.value);
        image_same(run(h,operation,7,0,&off,&clip_off,false,{},false,stream.value),expected,
            "nondefault stream OFF/reference");
        image_same(run(h,operation,7,0,&on,&clip_on,false,{},false,stream.value),expected,
            "nondefault stream ON/reference");
    }
    std::cout<<"Rigid proxy nondefault nonblocking stream: four operations, OFF/ON full images PASS\n";
}

void test_unvalidated(Environment& environment) {
    // A raw direct API may contain duplicate slots. The false/default topology
    // assertion must select the old path, and must not even parse the new flag.
    environment.assign("deliberately-invalid");
    gpu::GpuRigidProxyPipeline raw(false), default_raw;
    gpu::GpuRigidRotationClipper raw_rotation(false), default_rotation;
    bool proxy_rejected = false, rotation_rejected = false;
    try { gpu::GpuRigidProxyPipeline invalid(true); } catch (const std::invalid_argument&) { proxy_rejected = true; }
    try { gpu::GpuRigidRotationClipper invalid(true); } catch (const std::invalid_argument&) { rotation_rejected = true; }
    require(proxy_rejected && rotation_rejected, "validated constructors accepted malformed flag");
    auto h = rotation_fixture({129});
    h.nodes[1] = h.nodes[0];
    for (const auto operation : {Operation::Translation, Operation::RotationCommit, Operation::Finish}) {
        const auto reference = run(h, operation);
        image_same(run(h, operation, 7, 0, &raw), reference, "unvalidated duplicate CSR explicit fallback");
        image_same(run(h, operation, 7, 0, &default_raw), reference, "unvalidated duplicate CSR default fallback");
    }
    auto aliased=fixture({129});
    const int replaced=aliased.nodes.front();
    aliased.nodes.front()=0;aliased.owners[replaced]=-1;aliased.owners[0]=0;
    aliased.reference[0]=aliased.reference[replaced];aliased.positions[0]=aliased.positions[replaced];
    aliased.accepted_proxy[0]=aliased.accepted_proxy[replaced];
    const auto alias_reference=finish_alias_image(aliased);
    require(alias_reference.error==0,"raw alias reference unexpectedly failed");
    image_same(finish_alias_image(aliased,&raw),alias_reference,"false-gated aliased finish route");
    image_same(finish_alias_image(aliased,&default_raw),alias_reference,"default aliased finish route");
    const auto reference = run(h, Operation::RotationClip);
    image_same(run(h, Operation::RotationClip, 7, 0, &raw, &raw_rotation), reference,
        "unvalidated duplicate CSR rotation fallback");
    image_same(run(h, Operation::RotationClip, 7, 0, &default_raw, &default_rotation), reference,
        "default rotation constructor compatibility");
}
} // namespace

int main() {
    try {
        Environment parallel("IPC_GPU_PARALLEL_RIGID_PROXIES");
        Environment threads("IPC_GPU_RIGID_ROTATION_PAIR_THREADS");
        threads.assign("1");
        parallel.assign("0");
        gpu::GpuRigidProxyPipeline off(true);
        gpu::GpuRigidRotationClipper clip_off(true);
        parallel.assign("1");
        gpu::GpuRigidProxyPipeline on(true);
        gpu::GpuRigidRotationClipper clip_on(true);
        // Both owners must retain their captured settings across environment
        // changes, including a value that a freshly validated owner rejects.
        parallel.assign("invalid-after-construction");
        test_healthy(off, on);
        test_errors(off, on);
        test_rotations(off, on, clip_off, clip_on);
        test_rotation_contacts(off, on, clip_off, clip_on, threads);
        test_nondefault_stream(off, on, clip_off, clip_on);
        test_unvalidated(parallel);
        std::cout << "Rigid proxy parallel native acceptance PASS: bitwise OFF/ON/public oracle, "
            "large/tail/scrambled/empty/multibody rows, CPU formulas, exact endpoint copies, "
            "ordered error fallbacks, captured flags and unvalidated topology fallback\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Rigid proxy parallel native acceptance failed: " << error.what() << '\n';
        return 1;
    }
}
