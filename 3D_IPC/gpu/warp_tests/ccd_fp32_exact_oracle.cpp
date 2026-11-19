// Test-only independent exact-dyadic oracle. Production CCD stays FP32.
#include "ccd_fp32.cuh"
#include "exact.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

using ipc_gpu::ccd32::LinearQuery;
using F32Status = ipc_gpu::ccd32::Status;
using ExactStatus = ipc_gpu::exact::Status;

int queries = 0, accepted = 0, exact_unsafe = 0, clipped = 0;

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << " (query " << queries << ")\n";
        std::exit(1);
    }
}

ExactStatus exact(const LinearQuery& q) {
    // Every FP32 input is represented exactly by binary64. The oracle then
    // converts these bits to arbitrary-size dyadic integers; no input rounding
    // or floating-point geometric acceptance takes place in this check.
    double a[4][3], b[4][3];
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        a[i][k] = static_cast<double>(q.start[i][k]);
        b[i][k] = static_cast<double>(q.end[i][k]);
    }
    auto status = ipc_gpu::exact::validate_collision_step<256>(a, b, q.vertex_face);
    if (status == ExactStatus::CapacityExceeded)
        status = ipc_gpu::exact::validate_collision_step<2048>(a, b, q.vertex_face);
    require(status != ExactStatus::CapacityExceeded && status != ExactStatus::Invalid,
            "Exact oracle could not decide the one-moving-vertex query");
    return status;
}

void check(const LinearQuery& q) {
    ++queries;
    int moving = 0;
    for (int i = 0; i < 4; ++i) {
        bool changed = false;
        for (int k = 0; k < 3; ++k) changed = changed || q.start[i][k] != q.end[i][k];
        moving += changed ? 1 : 0;
    }
    require(moving == 1, "Fixture must exercise a nonstationary single-vertex trajectory");
    const auto status = ipc_gpu::ccd32::validate_linear(q);
    const auto oracle = exact(q);
    if (oracle == ExactStatus::Unsafe || oracle == ExactStatus::InitialContact) ++exact_unsafe;
    if (status == F32Status::Safe) {
        ++accepted;
        require(oracle == ExactStatus::Safe, "FP32 certified a path rejected by exact geometry");
    }
    const auto result = ipc_gpu::ccd32::clip_linear(q);
    require(result.status == F32Status::Safe, "Finite fixture failed to produce a safe clipped result");
    if (result.fraction > 0.0f && !result.full_step) {
        auto represented = q;
        for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
            represented.end[i][k] = result.endpoint[i][k];
        require(exact(represented) == ExactStatus::Safe,
                "A positive clipped FP32 step was rejected by exact geometry");
        ++clipped;
    }
}

LinearQuery rotated(std::mt19937& random, bool face, int index) {
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    const double angle = unit(random) * 3.141592653589793;
    const double tilt = unit(random) * 3.141592653589793;
    const double c = std::cos(angle), s = std::sin(angle);
    const double u = std::cos(tilt), v = std::sin(tilt);
    const double rotation[3][3] = {{c, -s*u, s*v}, {s, c*u, -c*v}, {0.0, v, u}};
    constexpr int exponents[] = {-10, -5, 0, 10, 20};
    const double scale = std::ldexp(1.0, exponents[index % 5]);
    const double translation[3] = {unit(random)*scale, unit(random)*scale, unit(random)*scale};
    const double vf[4][3] = {{.1,.1,.25},{-1,-1,0},{1,-1,0},{0,1,0}};
    const double ee[4][3] = {{-1,0,.25},{1,0,.25},{0,-1,0},{0,1,0}};
    const auto& base = face ? vf : ee;
    LinearQuery q; q.vertex_face = face;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        double value = translation[k];
        for (int j = 0; j < 3; ++j) value += rotation[k][j]*base[i][j]*scale;
        q.start[i][k] = q.end[i][k] = static_cast<float>(value);
    }
    // Exercise every moving-corner role, with occasional definite crossings
    // of the point or first edge endpoint. Only one represented vertex moves.
    const bool crossing = index % 4 == 0;
    const int mover = crossing ? 0 : index % 4;
    const double displacement[3] = {crossing ? 0.0 : unit(random)*.12,
                                    crossing ? 0.0 : unit(random)*.12,
                                    crossing ? -1.0 : unit(random)*.12};
    for (int k = 0; k < 3; ++k) {
        double delta = 0.0;
        for (int j = 0; j < 3; ++j) delta += rotation[k][j]*displacement[j]*scale;
        q.end[mover][k] = static_cast<float>(static_cast<double>(q.start[mover][k]) + delta);
    }
    return q;
}

int main() {
    std::mt19937 random(908127);
    for (bool face : {true, false}) {
        for (int i = 0; i < 64; ++i) check(rotated(random, face, i));
        for (int direction : {-1, 1}) {
            LinearQuery q; q.vertex_face = face;
            // Triangle collapses to one point; EE consists of two segments,
            // one collapsed. The moving endpoint can separate or intersect.
            q.start[0][0] = -1.0f; q.end[0][0] = static_cast<float>(direction)*2.0f;
            if (!face) q.start[1][0] = q.end[1][0] = -1.0f;
            check(q);
        }
        for (int i = 0; i < 8; ++i) {
            auto q = rotated(random, face, i);
            constexpr int exponents[] = {-10, -5, 0, 10, 20};
            const float magnify = std::ldexp(1.0f, 12 - exponents[i % 5]);
            // Use large offsets while keeping enough represented extent and
            // motion to avoid converting these fixtures into stationary paths.
            for (int corner = 0; corner < 4; ++corner) for (int k = 0; k < 3; ++k) {
                const float offset = k == 0 ? 1073741824.0f : (k == 1 ? -536870912.0f : 268435456.0f);
                // Relative scale is not material to the oracle, and every
                // transformed endpoint remains an exact FP32 test input.
                q.start[corner][k] = offset + q.start[corner][k] * magnify;
                q.end[corner][k] = offset + q.end[corner][k] * magnify;
            }
            check(q);
        }
    }
    require(accepted >= 80, "Too few nontrivial FP32 full-step accepts exercised the exact oracle");
    require(exact_unsafe >= 20, "The independent oracle did not see enough actual collisions");
    require(clipped >= 20, "Too few positive shortened endpoints exercised the exact oracle");
    std::cout << "{\"queries\":" << queries << ",\"accepted\":" << accepted
              << ",\"exact_unsafe\":" << exact_unsafe << ",\"clipped\":" << clipped
              << ",\"all_passed\":true}\n";
}
