// Independent behavioral checks for closest-feature FP32 direct clipping.
// Exact dyadic arithmetic is test-only: every accepted represented trajectory
// is checked by the existing exact geometric collision oracle.
#include "ccd_direct_fp32.cuh"
#include "ccd_direct_multi_axis.cuh"
#include "exact.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>

using ipc_gpu::ccd32::LinearQuery;
using ipc_gpu::ccd32::Status;
namespace direct = ipc_gpu::ccd32direct;
namespace multi_axis = ipc_gpu::ccd32direct_multi;
using ExactStatus = ipc_gpu::exact::Status;

static_assert(std::is_same<decltype(ipc_gpu::ccd32::ClipResult{}.fraction), float>::value,
              "Direct clipping must retain FP32 fractions");

int checks = 0, queries = 0, crossings = 0, accepted = 0, clipped = 0;
int exact_unsafe = 0, invalid = 0, rounding = 0, degeneracies = 0;
int gap_checks = 0, analytic_gap_checks = 0, subfloor = 0, thin_features = 0;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        std::cerr << message << " (query " << queries << ")\n";
        std::exit(1);
    }
}

ExactStatus exact(const LinearQuery& q) {
    // Converting binary32 to binary64 is exact. The oracle uses dyadic integer
    // predicates after that conversion, independently of direct certificates.
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

// This is an independent endpoint-distance oracle, not a whole-path floor
// oracle. Whole-path floor behavior is also exercised below on analytically
// solvable point/point and point/frozen-triangle motions.
void exact_endpoint_gap(const LinearQuery& q) {
    double start[4][3], end[4][3];
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        start[i][k] = static_cast<double>(q.start[i][k]);
        end[i][k] = static_cast<double>(q.end[i][k]);
    }
    auto status = ipc_gpu::exact::validate_endpoint_gap<256>(
        start, end, q.vertex_face, static_cast<double>(q.gap_floor));
    if (status == ExactStatus::CapacityExceeded)
        status = ipc_gpu::exact::validate_endpoint_gap<2048>(
            start, end, q.vertex_face, static_cast<double>(q.gap_floor));
    require(status == ExactStatus::Safe, "Accepted endpoint lost the independently computed gap floor");
    ++gap_checks;
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
    const auto status = direct::validate_linear(q);
    require(status != Status::Invalid, "A finite one-mover fixture was invalid");
    if (use_oracle && !unchanged(q)) {
        const auto oracle = exact(q);
        if (oracle == ExactStatus::Unsafe || oracle == ExactStatus::InitialContact)
            ++exact_unsafe;
        if (status == Status::Safe) {
            require(oracle == ExactStatus::Safe,
                    "Direct certificate accepted an exact geometric collision");
            exact_endpoint_gap(q);
            ++accepted;
        }
    }
    const float proposal = direct::propose_fraction(q);
    require(std::isfinite(proposal) && proposal >= 0.0f && proposal <= 1.0f,
            "Valid query produced an invalid direct fraction");
    const float retried = multi_axis::propose_fraction(q);
    require(std::isfinite(retried) && retried >= 0.0f && retried <= 1.0f,
            "Invalid multi-axis direct proposal");
    if (proposal > 0.0f)
        require(retried == proposal, "Multi-axis retry changed a positive direct fraction");
    // Exercise the same acceptance rule as VBD, including fractions shortened
    // by another incident pair. Check the actual represented segment independently.
    for (float fraction : {retried, 0.375f * retried}) {
        const auto represented = at_fraction(q, fraction);
        if (!use_oracle || unchanged(represented)) continue;
        if (fraction == 1.0f || multi_axis::validate_linear(represented) == Status::Safe) {
            require(exact(represented) == ExactStatus::Safe, "Multi-axis path crossed a primitive");
            exact_endpoint_gap(represented);
        }
    }
    const auto result = direct::clip_linear(q);
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
    require(direct::validate_linear(represented) == Status::Safe,
            "Returned represented endpoint does not revalidate");
    if (use_oracle && !unchanged(represented)) {
        require(exact(represented) == ExactStatus::Safe,
                "A positive represented clipped path collided under exact geometry");
        exact_endpoint_gap(represented);
    }
    if (result.fraction > 0.0f && result.fraction < 1.0f) ++clipped;

    // A vertex can receive a smaller fraction from an independent pair. Replay
    // and revalidate that actual stored endpoint before treating it as safe.
    for (float factor : {0.125f, 0.25f, 0.5f, 0.875f}) {
        const auto reduced = at_fraction(q, result.fraction * factor);
        if (use_oracle && !unchanged(reduced) && direct::validate_linear(reduced) == Status::Safe) {
            require(exact(reduced) == ExactStatus::Safe,
                    "A revalidated independently reduced fraction collided");
            exact_endpoint_gap(reduced);
        }
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
    require(direct::validate_linear(q) == Status::Invalid, "Invalid query was not rejected");
    require(direct::clip_linear(q).status == Status::Invalid, "Invalid query was clipped as valid");
    require(direct::propose_fraction(q) < 0.0f, "Invalid query yielded a nonnegative proposal");
    require(multi_axis::propose_fraction(q) < 0.0f, "Multi-axis retry accepted an invalid query");
    ++invalid;
}

// Exact-oracle-only replay of actual stored CPU/CUDA endpoints. Input uses
// round-trip binary32 decimal values: count, then kind (0=VF), floor, 24 coords.
// This mode deliberately does not call the implementation being tested.
int replay_oracle() {
    int count = 0;
    require(bool(std::cin >> count) && count > 0, "Missing replay query count");
    int moved = 0;
    for (int index = 0; index < count; ++index) {
        LinearQuery q;
        int kind = -1;
        double value = 0.0;
        require(bool(std::cin >> kind >> value), "Malformed replay query header");
        q.gap_floor = static_cast<float>(value);
        require(kind == 0 || kind == 1, "Invalid replay query kind");
        q.vertex_face = kind == 0;
        // Parse through double because some standard libraries set failbit
        // when extracting representable binary32 subnormals directly.
        for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
            require(bool(std::cin >> value), "Malformed replay start");
            q.start[i][k] = static_cast<float>(value);
        }
        for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
            require(bool(std::cin >> value), "Malformed replay endpoint");
            q.end[i][k] = static_cast<float>(value);
        }
        require(ipc_gpu::ccd32::detail::valid(q), "Invalid represented replay query");
        ++queries;
        if (!unchanged(q)) {
            require(exact(q) == ExactStatus::Safe, "Actual CPU/CUDA committed trajectory collided");
            exact_endpoint_gap(q);
            ++moved;
        }
    }
    std::cout << "{\"queries\":" << count << ",\"moved\":" << moved
              << ",\"gap_checks\":" << gap_checks << ",\"all_passed\":true}\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--oracle") return replay_oracle();
    std::mt19937 random(1837291);
    for (bool face : {true, false}) for (int mover = 0; mover < 4; ++mover) {
        for (int exponent : {-18, -8, 0, 8, 18}) {
            auto q = crossing(face, mover, std::ldexp(1.0f, exponent));
            require(direct::validate_linear(q) != Status::Safe, "Analytic crossing was certified safe");
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
    require(direct::validate_linear(q) != Status::Safe,
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
    require(direct::validate_linear(q) != Status::Safe, "Near-zero crossing was accepted");
    check(q);
    q = crossing(true, 0);
    q.start[0][0] = -1.0f; q.end[0][0] = 1.0f;
    q.start[0][1] = q.end[0][1] = 0.0f;
    require(direct::validate_linear(q) != Status::Safe, "Vertex grazing contact was accepted");
    check(q); ++degeneracies;
    q = crossing(true, 0);
    q.start[0][0] = -1.0f; q.end[0][0] = 1.0f;
    q.start[0][2] = q.end[0][2] = 0.0f;
    require(direct::validate_linear(q) != Status::Safe, "Coplanar entry into triangle was accepted");
    check(q); ++degeneracies;

    for (bool face : {true, false}) {
        // Degenerate primitive sets and fully coplanar paths remain collision
        // queries. Conservative zero progress is permitted for these fixtures.
        q = LinearQuery{}; q.vertex_face = face;
        q.start[0][0] = -1.0f; q.end[0][0] = 1.0f;
        if (!face) q.start[1][0] = q.end[1][0] = -1.0f;
        require(direct::validate_linear(q) != Status::Safe, "Degenerate crossing was accepted");
        check(q); ++degeneracies;
        q = crossing(face, 0);
        for (int i = 0; i < 4; ++i) q.start[i][2] = q.end[i][2] = 0.0f;
        q.end[0][0] += 1.0f;
        require(direct::validate_linear(q) != Status::Safe, "Initially touching coplanar motion was accepted");
        check(q); ++degeneracies;
        q = LinearQuery{}; q.vertex_face = face;
        require(check(q).full_step, "Unchanged contact should remain safe as no motion");

        // Subnormals, tiny clearances, and overflowing motion differences may
        // be rejected conservatively; they must never authorize a collision.
        for (float height : {std::numeric_limits<float>::denorm_min(),
                             std::numeric_limits<float>::min(), 1.0e-30f}) {
            q = crossing(face, 0, height);
            require(direct::validate_linear(q) != Status::Safe, "Tiny crossing was accepted");
            check(q); ++degeneracies;
        }
        q = crossing(face, 0);
        q.start[0][2] = std::numeric_limits<float>::max();
        q.end[0][2] = -std::numeric_limits<float>::max();
        require(direct::validate_linear(q) != Status::Safe, "Overflowing crossing was accepted");
        check(q); ++degeneracies;
    }

    // Even a fraction strictly below the exact TOI can round onto contact.
    // This models applying a fraction chosen by an independent constraint.
    const float plane = 100000.0f;
    q = crossing(true, 0, std::nextafter(plane, INFINITY) - plane, plane);
    q.end[0][2] = std::nextafter(plane, -INFINITY);
    const auto rounded = at_fraction(q, 0.25f);
    require(rounded.end[0][2] == plane, "Rounding fixture did not land on contact");
    require(direct::validate_linear(rounded) != Status::Safe,
            "Mathematically safe fraction hid an unsafe represented endpoint");
    check(q); check(rounded); rounding += 2;
    for (float fraction : {0.125f, 0.25f, 0.375f, 0.5f, 0.75f}) {
        check(at_fraction(q, fraction)); ++rounding;
    }

    // The full move can pass while a smaller stored prefix hits an oblique
    // face. Different coordinate ULPs let the two rounding errors add.
    const float x = 150000.0f, y = 100000.0f;
    const float ulp = std::nextafter(y, INFINITY) - y;
    q = {{{x + 2.0f * ulp, y, .09f}, {x-1.0f, y+1.0f+ulp, -1.0f},
          {x+1.0f, y-1.0f+ulp, -1.0f}, {x, y+ulp, 1.0f}},
         {{x, y + 2.0f * ulp, -.06f}, {x-1.0f, y+1.0f+ulp, -1.0f},
          {x+1.0f, y-1.0f+ulp, -1.0f}, {x, y+ulp, 1.0f}}};
    require(direct::validate_linear(q) == Status::Safe && exact(q) == ExactStatus::Safe,
            "Full endpoint should preserve separation for the oblique fixture");
    require(exact(at_fraction(q, .6f)) == ExactStatus::Unsafe,
            "Oblique fixture did not produce a colliding rounded prefix");
    require(direct::propose_fraction(q) < .6f,
            "Direct proposal allowed an unsafe fraction selected by another pair");
    check(q); ++rounding;

    // The rounded displacement Dhat can differ from the exact subtraction.
    // Reconstructing with FMA must remain safe even when that lost bit changes
    // the represented endpoint near the limiting fraction.
    for (bool face : {true, false}) {
        q = crossing(face, 0);
        q.end[0][2] = -16777216.0f;
        volatile float displacement = q.end[0][2] - q.start[0][2];
        require(static_cast<double>(displacement) !=
                    static_cast<double>(q.end[0][2]) - static_cast<double>(q.start[0][2]),
                "Displacement fixture did not exercise inexact FP32 subtraction");
        const auto result = check(q);
        require(result.fraction > 0.0f && result.endpoint[0][2] > q.gap_floor,
                "Inexact displacement lost the strict projected gap floor");
        ++rounding;
    }

    // The experimental contract is strict floor, unlike the CPU min(start,
    // floor) policy. Departing positive but sub-floor starts must also block.
    for (bool face : {true, false}) {
        for (float height : {0.0f, 0.005f, std::nextafter(0.01f, 0.0f), 0.01f}) {
            q = crossing(face, 0, height); q.gap_floor = 0.01f;
            q.end[0][2] = 0.25f;
            require(direct::validate_linear(q) != Status::Safe,
                    "Strict floor policy accepted a sub-floor or equal-floor moving start");
            require(check(q).fraction == 0.0f, "Sub-floor start made new represented motion");
            require(multi_axis::propose_fraction(q) == 0.0f,
                    "Multi-axis retry bypassed a touching, equal-floor or sub-floor start");
            ++subfloor;
        }
    }

    // Closest-feature arithmetic may be inaccurate for slender/near-parallel
    // primitives. It proposes an axis only; every accepted path must still
    // pass independent exact geometry. No positive-progress promise here.
    for (int exponent : {-4, -16, -28, -60}) {
        const float width = std::ldexp(1.0f, exponent);
        for (bool face : {true, false}) for (int mover = 0; mover < 4; ++mover) {
            q = crossing(face, mover);
            if (face) {
                q.start[0][1] = q.end[0][1] = 0.25f * width;
                q.start[3][1] = q.end[3][1] = width;
            } else {
                const float xy[4][2] = {{-1.0f,0.0f},{1.0f,width},{-1.0f,width},{1.0f,0.0f}};
                for (int i = 0; i < 4; ++i) for (int k = 0; k < 2; ++k)
                    q.start[i][k] = q.end[i][k] = xy[i][k];
            }
            check(q); ++thin_features;
        }
    }

    // Analytic minimum distance for a point moving past a collapsed frozen
    // triangle. This checks the ENTIRE represented segment against the floor;
    // separated endpoints alone would miss an interior floor violation.
    for (float lateral : {0.0f, 0.1f, 0.9f, 1.1f}) {
        q = LinearQuery{}; q.gap_floor = 0.25f;
        q.start[0][0] = 1.0f; q.start[0][1] = 0.125f;
        q.end[0][0] = -1.0f; q.end[0][1] = lateral;
        const auto result = check(q);
        if (result.fraction > 0.0f) {
            long double a2 = 0.0L, ad = 0.0L, d2 = 0.0L;
            for (int k = 0; k < 3; ++k) {
                const long double a = q.start[0][k];
                const long double d = static_cast<long double>(result.endpoint[0][k]) - a;
                a2 += a*a; ad += a*d; d2 += d*d;
            }
            const long double t = d2 > 0.0L ? std::fmax(0.0L, std::fmin(1.0L, -ad/d2)) : 0.0L;
            const long double minimum = a2 + t*(2.0L*ad + t*d2);
            const long double floor = q.gap_floor;
            require(minimum > floor*floor, "Analytic whole-path point/point gap floor was lost");
            ++analytic_gap_checks;
        }
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
              << ",\"invalid\":" << invalid << ",\"gap_checks\":" << gap_checks
              << ",\"analytic_gap_checks\":" << analytic_gap_checks
              << ",\"subfloor\":" << subfloor << ",\"thin_features\":" << thin_features
              << ",\"all_passed\":true}\n";
}
