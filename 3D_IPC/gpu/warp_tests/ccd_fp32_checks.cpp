// Standalone behavioral checks for the production FP32 CCD certificate.
// The planar crossing oracle is analytic and independent of CCD's predicates.
#include "ccd_fp32.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>

using ipc_gpu::ccd32::LinearQuery;
using ipc_gpu::ccd32::Status;
using ipc_gpu::ccd32::validate_linear;

static_assert(std::is_same<decltype(LinearQuery{}.start[0][0]), float&&>::value,
              "CCD coordinates must be FP32");
static_assert(std::is_same<decltype(LinearQuery{}.gap_floor), float>::value,
              "CCD gap must be FP32");
static_assert(std::is_same<decltype(ipc_gpu::ccd32::ClipResult{}.fraction), float>::value,
              "CCD clipping fractions must be FP32");
static_assert(std::is_same<decltype(ipc_gpu::ccd32::detail::Interval{}.lo), float>::value,
              "CCD interval endpoints must be FP32");

int checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

LinearQuery planar(bool vertex_face = true, float height = 1.0f,
                   float plane = 0.0f) {
    LinearQuery q;
    q.vertex_face = vertex_face;
    q.gap_floor = 1.0e-8f;
    if (vertex_face) {
        q.start[0][0] = 0.25f; q.start[0][1] = 0.25f;
        q.start[2][0] = 1.0f; q.start[3][1] = 1.0f;
    } else {
        q.start[0][0] = -1.0f; q.start[1][0] = 1.0f;
        q.start[2][1] = -1.0f; q.start[3][1] = 1.0f;
    }
    const int split = vertex_face ? 1 : 2;
    for (int i = 0; i < 4; ++i) {
        q.start[i][2] = plane + (i < split ? height : 0.0f);
        for (int k = 0; k < 3; ++k) q.end[i][k] = q.start[i][k];
    }
    return q;
}

int main() {
    int crossings = 0, safe = 0, degeneracies = 0, invalid = 0;
    // Powers of two avoid turning the reference crossing decision into a
    // floating-point approximation: the two groups straddle at t = 1/2.
    for (bool vt : {true, false}) {
        for (int exponent = -18; exponent <= 18; ++exponent) {
            const float h = std::ldexp(1.0f, exponent);
            for (int depth : {0, 2, 6}) {
                auto q = planar(vt, h);
                const int split = vt ? 1 : 2;
                for (int i = 0; i < split; ++i) q.end[i][2] = -h;
                require(validate_linear(q, depth) != Status::Safe,
                        "A vertex-face or edge-edge crossing was certified safe");
                ++crossings;
                const auto clipped = ipc_gpu::ccd32::clip_linear(q, 64, depth);
                require(clipped.status == Status::Safe && !clipped.full_step
                        && clipped.fraction > 0.0f && clipped.fraction < 0.5f,
                        "Clipping did not stop strictly before analytic impact");
                for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
                    require(clipped.endpoint[i][k] == ipc_gpu::ccd32::detail::interpolate(
                                q.start[i][k], q.end[i][k], clipped.fraction),
                            "Returned fraction did not reproduce the certified endpoint");
                const auto exhausted = ipc_gpu::ccd32::clip_linear(q, 1, depth);
                require(exhausted.status == Status::Safe && !exhausted.full_step
                        && exhausted.fraction == 0.0f,
                        "Exhausted backtracking budget did not fall back to no motion");
                for (int i = 0; i < split; ++i) q.end[i][2] = 2.0f * h;
                require(validate_linear(q, depth) == Status::Safe,
                        "A well-separated departing trajectory was rejected");
                ++safe;
            }
        }
        auto q = planar(vt);
        const int split = vt ? 1 : 2;
        for (int i = 0; i < split; ++i) q.end[i][2] = 0.0f;
        require(validate_linear(q) != Status::Safe,
                "An endpoint contact was certified safe"); ++crossings;
        for (int i = 0; i < split; ++i) q.start[i][2] = 0.0f;
        for (int i = 0; i < split; ++i) q.end[i][2] = -1.0f;
        require(validate_linear(q) != Status::Safe,
                "Motion through an initially touching pair was certified safe"); ++crossings;

        // FP32 can round a clipped proposal back onto the obstacle. Certify
        // the represented endpoint, not the unrounded interpolation fraction.
        const float plane = 100000.0f;
        const float above = std::nextafter(plane, INFINITY);
        const float below = std::nextafter(plane, -INFINITY);
        q = planar(vt, above - plane, plane);
        for (int i = 0; i < split; ++i) q.end[i][2] = below;
        require(validate_linear(q) != Status::Safe,
                "Large-coordinate one-ULP crossing was certified safe"); ++crossings;
        const float rounded = q.start[0][2] + 0.5f * (below - q.start[0][2]);
        require(rounded == plane, "Endpoint-rounding fixture does not reach the plane");
        for (int i = 0; i < split; ++i) q.end[i][2] = rounded;
        require(validate_linear(q) != Status::Safe,
                "Rounded-to-contact endpoint was certified safe"); ++crossings;
        for (int i = 0; i < split; ++i) q.end[i][2] = std::nextafter(above, INFINITY);
        require(validate_linear(q) == Status::Safe,
                "Representable outward motion at large coordinates was rejected"); ++safe;

        // Degenerate primitives still describe point/segment sets. Crossing
        // coincident sets must not become safe just because their normals vanish.
        q = LinearQuery{}; q.vertex_face = vt;
        for (int i = 0; i < split; ++i) {
            q.start[i][0] = -1.0f; q.end[i][0] = 1.0f;
        }
        require(validate_linear(q) != Status::Safe,
                "Degenerate point/segment crossing was certified safe"); ++degeneracies;
        for (int i = 0; i < split; ++i) q.end[i][0] = -2.0f;
        require(validate_linear(q) == Status::Safe,
                "Separated degenerate point/segment departure was rejected"); ++degeneracies;

        q = planar(vt, 1.0e-12f); q.gap_floor = 1.0e-15f;
        for (int i = 0; i < split; ++i) q.end[i][2] = -1.0e-12f;
        require(validate_linear(q) != Status::Safe,
                "Near-coplanar crossing was certified safe"); ++crossings;

        q = planar(vt, std::numeric_limits<float>::denorm_min());
        q.gap_floor = std::numeric_limits<float>::denorm_min();
        for (int i = 0; i < split; ++i) q.end[i][2] = -q.start[i][2];
        require(validate_linear(q) != Status::Safe,
                "Subnormal-scale crossing was certified safe"); ++crossings;

        q = planar(vt, std::numeric_limits<float>::max());
        for (int i = 0; i < split; ++i) q.end[i][2] = -q.start[i][2];
        require(validate_linear(q) != Status::Safe,
                "Overflowing trajectory difference was certified safe"); ++crossings;
    }
    for (float bad : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
        auto q = planar(); q.start[0][0] = bad;
        require(validate_linear(q) == Status::Invalid, "Nonfinite start was not invalid"); ++invalid;
        q = planar(); q.end[3][1] = bad;
        require(validate_linear(q) == Status::Invalid, "Nonfinite endpoint was not invalid"); ++invalid;
        q = planar(); q.gap_floor = bad;
        require(validate_linear(q) == Status::Invalid, "Nonfinite gap was not invalid"); ++invalid;
    }
    for (float gap : {0.0f, -1.0f}) {
        auto q = planar(); q.gap_floor = gap;
        require(validate_linear(q) == Status::Invalid, "Nonpositive gap was not invalid"); ++invalid;
    }
    require(validate_linear(planar(), -1) == Status::Invalid, "Negative depth was accepted"); ++invalid;
    require(validate_linear(planar(), 11) == Status::Invalid, "Excessive depth was accepted"); ++invalid;
    std::cout << "{\"checks\":" << checks << ",\"crossings\":" << crossings
              << ",\"safe\":" << safe << ",\"degeneracies\":" << degeneracies
              << ",\"invalid\":" << invalid << ",\"all_passed\":true}\n";
}
