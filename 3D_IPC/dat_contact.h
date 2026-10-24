#pragma once

#include "ogc_contact.h"
#include "segment_segment_distance.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

// Contact geometry used by DAT with OGC's corrected point-triangle distances.
namespace dat_detail {

using WideVec3 = Eigen::Matrix<long double, 3, 1>;

struct PairGeometry {
    Vec3 normal = Vec3::Zero();
    // Conservative projections relative to x[0]: positive roles are rounded
    // down and negative roles up before their supporting extrema are formed.
    std::array<double, 4> projections{{0.0, 0.0, 0.0, 0.0}};
    double negative_support = 0.0;
    double positive_support = 0.0;
    double distance = 0.0;
    int positive_count = 0;
    bool valid_plane = false;
};

inline PairGeometry make_pair_geometry(const std::array<Vec3, 4>& positions, const Vec3& separation, double distance, int positive_count) {
    PairGeometry result;
    result.positive_count = positive_count;
    result.distance = std::isfinite(distance) && distance > 0.0 ? distance : 0.0;
    const double length = separation.stableNorm();
    if (result.distance == 0.0 || !separation.allFinite() || !std::isfinite(length) || length <= 0.0) return result;
    result.normal = separation / length;
    const WideVec3 origin = positions[0].cast<long double>();
    const WideVec3 normal = result.normal.cast<long double>();
    long double coordinate_scale = 0.0L;
    std::array<long double, 4> projections{};
    for (int role = 0; role < 4; ++role) {
        if (!positions[role].allFinite()) return result;
        const WideVec3 relative = positions[role].cast<long double>() - origin;
        projections[role] = normal.dot(relative);
        coordinate_scale = std::max(coordinate_scale, relative.cwiseAbs().maxCoeff());
    }
    const long double margin = 32.0L * std::numeric_limits<double>::epsilon() * coordinate_scale;
    result.positive_support = std::numeric_limits<double>::infinity();
    result.negative_support = -std::numeric_limits<double>::infinity();
    for (int role = 0; role < 4; ++role) {
        const bool positive = role < positive_count;
        const double projection = static_cast<double>(projections[role] + (positive ? -margin : margin));
        if (!std::isfinite(projection)) return result;
        result.projections[role] = std::nextafter(projection, positive ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity());
        if (positive)
            result.positive_support = std::min(result.positive_support, result.projections[role]);
        else
            result.negative_support = std::max(result.negative_support, result.projections[role]);
    }
    const double gap = result.positive_support - result.negative_support;
    result.valid_plane = std::isfinite(gap) && gap > 0.0;
    return result;
}

inline PairGeometry make_vertex_triangle_geometry(const std::array<Vec3, 4>& positions, const NodeTriangleDistanceResult& distance) {
    Vec3 separation = positions[0] - distance.closest_point;
    if (distance.region == NodeTriangleRegion::FaceInterior) {
        separation = (distance.phi < 0.0 ? -1.0 : 1.0) * distance.normal;
    }
    return make_pair_geometry(positions, separation, distance.distance, 1);
}

inline PairGeometry make_edge_edge_geometry(const std::array<Vec3, 4>& positions, const SegmentSegmentDistanceResult& distance) {
    Vec3 separation = distance.separation;
    if (!separation.allFinite() || separation.stableNorm() == 0.0)
        separation = distance.closest_point_1 - distance.closest_point_2;
    return make_pair_geometry(positions, separation, distance.distance, 2);
}

inline double division_plane_offset(const PairGeometry& geometry, const std::array<Vec3, 4>& displacements) {
    if (!geometry.valid_plane) return 0.0;
    double positive_approach = 0.0, negative_approach = 0.0;
    for (int role = 0; role < 4; ++role) {
        const double projection = geometry.normal.dot(displacements[role]);
        if (!displacements[role].allFinite() || !std::isfinite(projection)) continue;
        const bool positive = role < geometry.positive_count;
        const double approach = std::max(0.0, positive ? -projection : projection);
        if (positive) positive_approach = std::max(positive_approach, approach);
        else negative_approach = std::max(negative_approach, approach);
    }
    double lambda = 0.5;
    const double largest_approach = std::max(positive_approach, negative_approach);
    if (largest_approach > 0.0) {
        const double positive = positive_approach / largest_approach;
        const double negative = negative_approach / largest_approach;
        lambda = negative / (positive + negative);
    }
    constexpr double minimum_fraction = 1.0e-8;
    lambda = std::clamp(lambda, minimum_fraction, 1.0 - minimum_fraction);
    const double gap = geometry.positive_support - geometry.negative_support;
    return geometry.positive_support - lambda * gap;
}

inline void accumulate_pair_scales(const PairGeometry& geometry, const std::array<Vec3, 4>& displacements, double gamma, std::array<double, 4>& scales) {
    if (!std::isfinite(gamma)) gamma = 0.0;
    gamma = std::clamp(gamma, 0.0, 1.0);
    if (!geometry.valid_plane) {
        const double radius = 0.5 * gamma * geometry.distance;
        for (int role = 0; role < 4; ++role) {
            const double length = displacements[role].stableNorm();
            if (!std::isfinite(length)) scales[role] = 0.0;
            else if (length > 0.0)
                scales[role] = std::min(scales[role], radius / length);
        }
        return;
    }

    const double plane = division_plane_offset(geometry, displacements);
    for (int role = 0; role < 4; ++role) {
        const double projection = geometry.normal.dot(displacements[role]);
        if (!displacements[role].allFinite() || !std::isfinite(projection)) {
            scales[role] = 0.0;
            continue;
        }
        const double approach = std::max(0.0, role < geometry.positive_count ? -projection : projection);
        if (approach <= 0.0) continue;
        const double clearance = std::max(0.0, role < geometry.positive_count ? geometry.projections[role] - plane : plane - geometry.projections[role]);
        scales[role] = std::min(scales[role], gamma * clearance / approach);
    }
}

inline bool accepts_rounded_position(const PairGeometry& geometry, int role, double plane, const Vec3& base, const Vec3& candidate, double gamma) {
    if (role < 0 || role >= 4 || !base.allFinite() || !candidate.allFinite() || !std::isfinite(gamma)) return false;
    const WideVec3 realized = candidate.cast<long double>() - base.cast<long double>();
    if (!realized.allFinite()) return false;
    if (!geometry.valid_plane) {
        if (!std::isfinite(geometry.distance) || geometry.distance < 0.0) return false;
        const long double radius = 0.5L * std::clamp(gamma, 0.0, 1.0) * static_cast<long double>(geometry.distance);
        const long double length = realized.stableNorm();
        return std::isfinite(length) && length <= radius;
    }
    if (!geometry.normal.allFinite() || !std::isfinite(plane) || !std::isfinite(geometry.projections[role])) return false;
    const bool positive = role < geometry.positive_count;
    const long double initial = positive ? static_cast<long double>(geometry.projections[role]) - plane : static_cast<long double>(plane) - geometry.projections[role];
    const long double shift = geometry.normal.cast<long double>().dot(realized);
    const long double final = initial + (positive ? shift : -shift);
    if (!std::isfinite(initial) || !std::isfinite(final)) return false;
    // A scalar within its analytic bound can still round to the opposing
    // primitive at large world coordinates. Check the stored endpoint itself.
    if (initial > 0.0L) return final > 0.0L;
    return initial == 0.0L && final >= 0.0L;
}

} // namespace dat_detail
