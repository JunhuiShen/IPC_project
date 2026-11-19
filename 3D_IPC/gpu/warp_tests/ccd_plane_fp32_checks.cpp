// Independent behavioral checks for bounded-work FP32 plane clipping.
// Exact dyadic arithmetic is test-only: every accepted represented trajectory
// is checked by the existing exact geometric collision oracle.
#include "ccd_plane_fp32.cuh"
#include "exact.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>

using ipc_gpu::ccd32::LinearQuery;
using ipc_gpu::ccd32::Status;
namespace plane = ipc_gpu::ccd32plane;
using ExactStatus = ipc_gpu::exact::Status;

static_assert(std::is_same<decltype(ipc_gpu::ccd32::ClipResult{}.fraction), float>::value,
              "Plane clipping must retain FP32 fractions");

int checks = 0, queries = 0, crossings = 0, accepted = 0, clipped = 0;
int exact_unsafe = 0, invalid = 0, rounding = 0, degeneracies = 0;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        std::cerr << message << " (query " << queries << ")\n";
        std::exit(1);
    }
}

ExactStatus exact(const LinearQuery& q) {
    // Converting binary32 to binary64 is exact. The oracle uses dyadic integer
    // predicates after that conversion, independently of plane certificates.
    double start[4][3], end[4][3];
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        start[i][k] = static_cast<double>(q.start[i][k]);
        end[i][k] = static_cast<double>(q.end[i][k]);
    }
    auto status = ipc_gpu::exact::validate_collision_step<256>(start, end, q.vertex_face);
    if (status == ExactStatus::CapacityExceeded)
        status = ipc_gpu::exact::validate_collision_step<2048>(start, end, q.vertex_face);
    require(status != ExactStatus::Invalid && status != ExactStatus::CapacityExceeded,
            "Exact oracle could not decide a finite one-mover query");
    return status;
}

LinearQuery at_fraction(const LinearQuery& q, float fraction) {
    auto result = q;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        result.end[i][k] = ipc_gpu::ccd32::detail::interpolate(
            q.start[i][k], q.end[i][k], fraction);
    return result;
}

bool unchanged(const LinearQuery& q) {
    return ipc_gpu::ccd32::detail::unchanged(q);
}

ipc_gpu::ccd32::ClipResult check(const LinearQuery& q, bool use_oracle = true) {
    ++queries;
    const auto status = plane::validate_linear(q);
    require(status != Status::Invalid, "A finite one-mover fixture was invalid");
    if (use_oracle && !unchanged(q)) {
        const auto oracle = exact(q);
        if (oracle == ExactStatus::Unsafe || oracle == ExactStatus::InitialContact)
            ++exact_unsafe;
        if (status == Status::Safe) {
            require(oracle == ExactStatus::Safe,
                    "Plane certificate accepted an exact geometric collision");
            ++accepted;
        }
    }
    const auto result = plane::clip_linear(q);
    require(result.status == Status::Safe, "Valid query did not yield a safe clipped result");
    require(std::isfinite(result.fraction) && result.fraction >= 0.0f
            && result.fraction <= 1.0f, "Invalid clipping fraction");
    require(result.full_step == (result.fraction == 1.0f),
            "Full-step flag and fraction disagree");
    auto represented = q;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        const float replay = ipc_gpu::ccd32::detail::interpolate(
            q.start[i][k], q.end[i][k], result.fraction);
        require(ipc_gpu::ccd32::detail::same_value(result.endpoint[i][k], replay),
                "Returned endpoint does not replay its fraction exactly");
        represented.end[i][k] = result.endpoint[i][k];
    }
    require(plane::validate_linear(represented) == Status::Safe,
            "Returned represented endpoint does not revalidate");
    if (use_oracle && !unchanged(represented))
        require(exact(represented) == ExactStatus::Safe,
                "A positive represented clipped path collided under exact geometry");
    if (result.fraction > 0.0f && result.fraction < 1.0f) ++clipped;

    // A vertex can receive a smaller fraction from an independent pair. Replay
    // and revalidate that actual stored endpoint before treating it as safe.
    for (float factor : {0.125f, 0.25f, 0.5f, 0.875f}) {
        const auto reduced = at_fraction(q, result.fraction * factor);
        if (use_oracle && !unchanged(reduced) && plane::validate_linear(reduced) == Status::Safe)
            require(exact(reduced) == ExactStatus::Safe,
                    "A revalidated independently reduced fraction collided");
    }
    return result;
}

LinearQuery crossing(bool face, int mover, float height = 1.0f,
                     float plane = 0.0f) {
    LinearQuery q;
    q.vertex_face = face;
    if (face) {
        q.start[0][0] = 0.25f; q.start[0][1] = 0.25f;
        q.start[2][0] = 1.0f; q.start[3][1] = 1.0f;
        for (int i = 0; i < 4; ++i)
            q.start[i][2] = plane + (i == 0 ? height : 0.0f);
    } else {
        q.start[0][0] = -1.0f; q.start[1][0] = 1.0f;
        q.start[2][1] = -1.0f; q.start[3][1] = 1.0f;
        for (int i = 0; i < 4; ++i)
            q.start[i][2] = plane + (i < 2 ? height : 0.0f);
    }
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k)
        q.end[i][k] = q.start[i][k];
    // Every case contacts at t=1/2. PT triangle-corner weights at the fixed
    // point are 1/2, 1/4, 1/4; EE midpoint weights are 1/2 on either edge.
    q.end[mover][2] = face
        ? (mover == 0 ? plane - height : plane + (mover == 1 ? 4.0f : 8.0f) * height)
        : (mover < 2 ? plane - 3.0f * height : plane + 4.0f * height);
    return q;
}

LinearQuery rotated(std::mt19937& random, bool face, int mover, int index) {
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    const double angle = unit(random) * 3.141592653589793;
    const double tilt = unit(random) * 3.141592653589793;
    const double c = std::cos(angle), s = std::sin(angle);
    const double u = std::cos(tilt), v = std::sin(tilt);
    const double rotation[3][3] = {{c,-s*u,s*v}, {s,c*u,-c*v}, {0.0,v,u}};
    constexpr int powers[] = {-10, -4, 0, 8, 16};
    const double scale = std::ldexp(1.0, powers[index % 5]);
    const double offset[3] = {unit(random)*scale, unit(random)*scale, unit(random)*scale};
    auto base = crossing(face, mover);
    if (index % 3 != 0) {
        for (int k = 0; k < 3; ++k)
            base.end[mover][k] = base.start[mover][k] + static_cast<float>(unit(random)*0.15);
    }
    LinearQuery q; q.vertex_face = face;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        double start = offset[k], end = offset[k];
        for (int j = 0; j < 3; ++j) {
            start += rotation[k][j] * static_cast<double>(base.start[i][j]) * scale;
            end += rotation[k][j] * static_cast<double>(base.end[i][j]) * scale;
        }
        q.start[i][k] = static_cast<float>(start);
        q.end[i][k] = static_cast<float>(end);
    }
    return q;
}

void expect_invalid(const LinearQuery& q) {
    require(plane::validate_linear(q) == Status::Invalid, "Invalid query was not rejected");
    require(plane::clip_linear(q).status == Status::Invalid, "Invalid query was clipped as valid");
    ++invalid;
}

int main() {
    std::mt19937 random(1837291);
    for (bool face : {true, false}) for (int mover = 0; mover < 4; ++mover) {
        for (int exponent : {-18, -8, 0, 8, 18}) {
            auto q = crossing(face, mover, std::ldexp(1.0f, exponent));
            require(plane::validate_linear(q) != Status::Safe, "Analytic crossing was certified safe");
            const auto result = check(q);
            require(result.fraction > 0.0f && result.fraction < 0.5f,
                    "Ordinary crossing did not make positive safe progress before impact");
            ++crossings;
            // Move that same corner away, retaining the one-mover contract.
            q.end[mover][2] = q.start[mover][2]
                - 0.125f * (q.end[mover][2] - q.start[mover][2]);
            require(check(q).full_step, "Well-separated departing motion was truncated");
        }
        for (int index = 0; index < 24; ++index) check(rotated(random, face, mover, index));
    }

    // A closest point transitions from the triangle interior to an edge and
    // then a vertex while a fixed z separator continues to certify the path.
    auto q = crossing(true, 0);
    q.end[0][0] = 3.0f; q.end[0][1] = -2.0f; q.end[0][2] = 1.0f;
    require(check(q).full_step, "Changing closest features broke a valid fixed separator");
    // The EE closest pair likewise changes while one endpoint moves laterally.
    q = crossing(false, 0);
    q.end[0][0] = 4.0f; q.end[0][1] = 3.0f; q.end[0][2] = 1.0f;
    require(check(q).full_step, "Changing EE closest features broke a fixed separator");

    // Analytic gap-floor check on accepted point/triangle motion.
    q = crossing(true, 0, 0.125f); q.gap_floor = 0.01f;
    const auto floor_result = check(q);
    require(floor_result.fraction > 0.0f && floor_result.endpoint[0][2] > q.gap_floor,
            "Clipping failed to preserve the provable initial gap floor");
    // One axis proves the initial floor and another separates the endpoint.
    // Accepting only positive starting clearance on the latter axis would let
    // this path pass inside the floor without ever reaching actual contact.
    q = LinearQuery{}; q.gap_floor = 1.0f;
    q.start[0][0] = 2.0f; q.start[0][1] = 0.1f;
    q.end[0][1] = 1.01f;
    require(plane::validate_linear(q) != Status::Safe,
            "Changing the separating axis allowed a provable initial floor to be lost");
    const auto preserved = check(q);
    if (preserved.fraction > 0.0f) {
        // Independent analytic point/point minimum along the stored segment.
        const double ax = static_cast<double>(q.start[0][0]);
        const double ay = static_cast<double>(q.start[0][1]);
        const double dx = static_cast<double>(preserved.endpoint[0][0]) - ax;
        const double dy = static_cast<double>(preserved.endpoint[0][1]) - ay;
        const double parameter = std::fmax(0.0, std::fmin(1.0, -(ax*dx+ay*dy)/(dx*dx+dy*dy)));
        const double x = ax + parameter*dx, y = ay + parameter*dy;
        require(x*x+y*y > 1.0, "Clipped represented segment lost the initial gap floor");
    }
    q = crossing(true, 0, 1.0e-10f);
    require(plane::validate_linear(q) != Status::Safe, "Near-zero crossing was accepted");
    check(q);
    q = crossing(true, 0);
    q.start[0][0] = -1.0f; q.end[0][0] = 1.0f;
    q.start[0][1] = q.end[0][1] = 0.0f;
    require(plane::validate_linear(q) != Status::Safe, "Vertex grazing contact was accepted");
    check(q); ++degeneracies;
    q = crossing(true, 0);
    q.start[0][0] = -1.0f; q.end[0][0] = 1.0f;
    q.start[0][2] = q.end[0][2] = 0.0f;
    require(plane::validate_linear(q) != Status::Safe, "Coplanar entry into triangle was accepted");
    check(q); ++degeneracies;

    for (bool face : {true, false}) {
        // Degenerate primitive sets and fully coplanar paths remain collision
        // queries. Conservative zero progress is permitted for these fixtures.
        q = LinearQuery{}; q.vertex_face = face;
        q.start[0][0] = -1.0f; q.end[0][0] = 1.0f;
        if (!face) q.start[1][0] = q.end[1][0] = -1.0f;
        require(plane::validate_linear(q) != Status::Safe, "Degenerate crossing was accepted");
        check(q); ++degeneracies;
        q = crossing(face, 0);
        for (int i = 0; i < 4; ++i) q.start[i][2] = q.end[i][2] = 0.0f;
        q.end[0][0] += 1.0f;
        require(plane::validate_linear(q) != Status::Safe, "Initially touching coplanar motion was accepted");
        check(q); ++degeneracies;
        q = LinearQuery{}; q.vertex_face = face;
        require(check(q).full_step, "Unchanged contact should remain safe as no motion");

        // Subnormals, tiny clearances, and overflowing motion differences may
        // be rejected conservatively; they must never authorize a collision.
        for (float height : {std::numeric_limits<float>::denorm_min(),
                             std::numeric_limits<float>::min(), 1.0e-30f}) {
            q = crossing(face, 0, height);
            require(plane::validate_linear(q) != Status::Safe, "Tiny crossing was accepted");
            check(q); ++degeneracies;
        }
        q = crossing(face, 0);
        q.start[0][2] = std::numeric_limits<float>::max();
        q.end[0][2] = -std::numeric_limits<float>::max();
        require(plane::validate_linear(q) != Status::Safe, "Overflowing crossing was accepted");
        check(q); ++degeneracies;
    }

    // Even a fraction strictly below the exact TOI can round onto contact.
    // This models applying a fraction chosen by an independent constraint.
    const float plane = 100000.0f;
    q = crossing(true, 0, std::nextafter(plane, INFINITY) - plane, plane);
    q.end[0][2] = std::nextafter(plane, -INFINITY);
    const auto rounded = at_fraction(q, 0.25f);
    require(rounded.end[0][2] == plane, "Rounding fixture did not land on contact");
    require(plane::validate_linear(rounded) != Status::Safe,
            "Mathematically safe fraction hid an unsafe represented endpoint");
    check(q); check(rounded); rounding += 2;
    for (float fraction : {0.125f, 0.25f, 0.375f, 0.5f, 0.75f}) {
        check(at_fraction(q, fraction)); ++rounding;
    }

    // Inputs are finite and exactly one vertex moves unless explicitly invalid.
    for (float bad : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
        q = crossing(true, 0); q.start[0][0] = bad; expect_invalid(q);
        q = crossing(false, 0); q.end[3][1] = bad; expect_invalid(q);
        q = crossing(true, 0); q.gap_floor = bad; expect_invalid(q);
    }
    for (float gap : {0.0f, -1.0f}) {
        q = crossing(true, 0); q.gap_floor = gap; expect_invalid(q);
    }
    for (bool face : {true, false}) for (int second = 1; second < 4; ++second) {
        q = crossing(face, 0); q.end[second][0] += 0.125f; expect_invalid(q);
    }
    require(accepted >= 100 && clipped >= 40 && exact_unsafe >= 40,
            "Safety checks did not exercise enough accepts, clips, and collisions");
    std::cout << "{\"checks\":" << checks << ",\"queries\":" << queries
              << ",\"crossings\":" << crossings << ",\"accepted\":" << accepted
              << ",\"clipped\":" << clipped << ",\"exact_unsafe\":" << exact_unsafe
              << ",\"rounding\":" << rounding << ",\"degeneracies\":" << degeneracies
              << ",\"invalid\":" << invalid << ",\"all_passed\":true}\n";
}
