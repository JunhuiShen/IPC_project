#include "mesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

using FaceKey = std::array<int, 3>;
using TetKey = std::array<int, 4>;
using FacePair = std::array<int, 2>;

template <std::size_t N>
struct IndexArrayHash {
    std::size_t operator()(const std::array<int, N>& indices) const noexcept {
        // A small hash-combine implementation.  The keys are sorted before
        // hashing, so permutations of a face/tet map to the same entry.
        std::size_t seed = 0;
        for (const int index : indices) {
            const std::size_t value = std::hash<int>{}(index);
            seed ^= value + static_cast<std::size_t>(0x9e3779b9U)
                    + (seed << 6U) + (seed >> 2U);
        }
        return seed;
    }
};

FaceKey canonical_face(int a, int b, int c) {
    FaceKey key{a, b, c};
    std::sort(key.begin(), key.end());
    return key;
}

TetKey canonical_tet(int a, int b, int c, int d) {
    TetKey key{a, b, c, d};
    std::sort(key.begin(), key.end());
    return key;
}

std::string tet_label(std::size_t tet_index) {
    std::ostringstream message;
    message << "tetrahedron " << tet_index;
    return message.str();
}

void validate_tolerance(double relative_degeneracy_tolerance) {
    if (!std::isfinite(relative_degeneracy_tolerance)
        || relative_degeneracy_tolerance < 0.0) {
        throw std::invalid_argument(
            "tet degeneracy tolerance must be finite and non-negative");
    }
}

void validate_rest_positions(const std::vector<Vec3>& rest_positions) {
    if (rest_positions.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "tet mesh has more vertices than its int indices can address");
    }

    for (std::size_t i = 0; i < rest_positions.size(); ++i) {
        if (!rest_positions[i].allFinite()) {
            std::ostringstream message;
            message << "rest position " << i << " is not finite";
            throw std::invalid_argument(message.str());
        }
    }
}

void validate_tet_geometry(
    const std::array<int, 4>& tet,
    std::size_t tet_index,
    const std::vector<Vec3>& rest_positions,
    double relative_degeneracy_tolerance) {
    const Vec3* p[4] = {
        &rest_positions[static_cast<std::size_t>(tet[0])],
        &rest_positions[static_cast<std::size_t>(tet[1])],
        &rest_positions[static_cast<std::size_t>(tet[2])],
        &rest_positions[static_cast<std::size_t>(tet[3])],
    };

    double max_edge_squared = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            max_edge_squared =
                std::max(max_edge_squared, (*p[j] - *p[i]).squaredNorm());
        }
    }

    const double edge_scale =
        max_edge_squared * std::sqrt(max_edge_squared);
    const double signed_six_volume =
        (*p[1] - *p[0]).dot((*p[2] - *p[0]).cross(*p[3] - *p[0]));

    if (!std::isfinite(edge_scale) || !std::isfinite(signed_six_volume)) {
        throw std::invalid_argument(
            tet_label(tet_index) + " has numerically invalid geometry");
    }
    if (edge_scale == 0.0
        || signed_six_volume
               <= relative_degeneracy_tolerance * edge_scale) {
        throw std::invalid_argument(
            tet_label(tet_index)
            + " must have positive, non-degenerate rest measure");
    }
}

void validate_boundary_tet_topology(const std::vector<int>& tet_mesh) {
    if (tet_mesh.size() % 4 != 0) {
        throw std::invalid_argument(
            "tet connectivity must contain exactly four indices per tetrahedron");
    }

    std::unordered_set<TetKey, IndexArrayHash<4>> unique_tets;
    unique_tets.reserve(tet_mesh.size() / 4);
    std::unordered_map<FaceKey, std::size_t, IndexArrayHash<3>>
        face_incident_counts;
    face_incident_counts.reserve(tet_mesh.size());

    for (std::size_t element = 0; element < tet_mesh.size() / 4; ++element) {
        TetKey key{
            tet_mesh[4 * element], tet_mesh[4 * element + 1],
            tet_mesh[4 * element + 2], tet_mesh[4 * element + 3]};
        for (const int vertex : key) {
            if (vertex < 0)
                throw std::out_of_range("tet connectivity has a negative index");
        }
        std::sort(key.begin(), key.end());
        if (std::adjacent_find(key.begin(), key.end()) != key.end()) {
            throw std::invalid_argument(
                "tetrahedron contains a repeated vertex index");
        }
        if (!unique_tets.insert(key).second)
            throw std::invalid_argument("duplicate tetrahedron");

        for (std::size_t local = 0; local < 4; ++local) {
            const std::array<int, 3> local_face_indices = TetFace(local);
            const FaceKey face = canonical_face(
                tet_mesh[4 * element + local_face_indices[0]],
                tet_mesh[4 * element + local_face_indices[1]],
                tet_mesh[4 * element + local_face_indices[2]]);
            const std::size_t incident_count = ++face_incident_counts[face];
            if (incident_count > 2) {
                std::ostringstream message;
                message << "non-manifold face (" << face[0] << ", "
                        << face[1] << ", " << face[2]
                        << ") is incident on more than two tetrahedra";
                throw std::invalid_argument(message.str());
            }
        }
    }
}

// std::vector/std::array port of TGSL::MESH::ComputeTetMeshFaces. A face is
// represented by its flat tet-connectivity index n = 4 * element + local.
std::vector<FacePair> ComputeTetMeshFaces(
    const std::vector<int>& tet_mesh) {
    if (tet_mesh.empty())
        return {};

    const auto FaceGreaterThan = [](FaceKey& i1, FaceKey& i2) {
        bool greater_than = false;
        std::sort(i1.begin(), i1.end());
        std::sort(i2.begin(), i2.end());
        if (i1[0] > i2[0])
            greater_than = true;
        else if (i1[0] == i2[0] && i1[1] > i2[1])
            greater_than = true;
        else if (i1[0] == i2[0] && i1[1] == i2[1] && i1[2] > i2[2])
            greater_than = true;
        return greater_than;
    };

    const auto FaceEqual = [](FaceKey& i1, FaceKey& i2) {
        std::sort(i1.begin(), i1.end());
        std::sort(i2.begin(), i2.end());
        return i1[0] == i2[0] && i1[1] == i2[1] && i1[2] == i2[2];
    };

    const auto GreaterThan = [&FaceGreaterThan](
                                 const std::vector<int>& mesh,
                                 std::size_t e1,
                                 std::size_t e2) {
        const std::size_t local_1 = e1 % 4;
        const std::size_t element_1 = e1 / 4;
        const FaceKey local_face_1 = TetFace(local_1);
        FaceKey edge1 = {
            mesh[4 * element_1 + local_face_1[0]],
            mesh[4 * element_1 + local_face_1[1]],
            mesh[4 * element_1 + local_face_1[2]]};

        const std::size_t local_2 = e2 % 4;
        const std::size_t element_2 = e2 / 4;
        const FaceKey local_face_2 = TetFace(local_2);
        FaceKey edge2 = {
            mesh[4 * element_2 + local_face_2[0]],
            mesh[4 * element_2 + local_face_2[1]],
            mesh[4 * element_2 + local_face_2[2]]};
        return FaceGreaterThan(edge1, edge2);
    };

    const auto Equal = [&FaceEqual](
                           const std::vector<int>& mesh,
                           const std::vector<int>& all_faces,
                           std::size_t i1,
                           std::size_t i2) {
        const std::size_t e1 = static_cast<std::size_t>(all_faces[i1]);
        const std::size_t e2 = static_cast<std::size_t>(all_faces[i2]);

        const std::size_t local_1 = e1 % 4;
        const std::size_t element_1 = e1 / 4;
        const FaceKey local_face_1 = TetFace(local_1);
        FaceKey edge1 = {
            mesh[4 * element_1 + local_face_1[0]],
            mesh[4 * element_1 + local_face_1[1]],
            mesh[4 * element_1 + local_face_1[2]]};

        const std::size_t local_2 = e2 % 4;
        const std::size_t element_2 = e2 / 4;
        const FaceKey local_face_2 = TetFace(local_2);
        FaceKey edge2 = {
            mesh[4 * element_2 + local_face_2[0]],
            mesh[4 * element_2 + local_face_2[1]],
            mesh[4 * element_2 + local_face_2[2]]};
        return FaceEqual(edge1, edge2);
    };

    const std::size_t face_number = tet_mesh.size();
    std::vector<int> all_faces(face_number);
    std::vector<int> ranges(face_number + 1);

    for (std::size_t f = 0; f < face_number; ++f) {
        all_faces[f] = static_cast<int>(f);
        ranges[f] = static_cast<int>(f);
    }

    ranges[ranges.size() - 1] = static_cast<int>(ranges.size() - 1);

    std::sort(
        all_faces.begin(), all_faces.end(),
        [&GreaterThan, &tet_mesh](int face_a, int face_b) {
            return GreaterThan(
                tet_mesh, static_cast<std::size_t>(face_a),
                static_cast<std::size_t>(face_b));
        });

    const auto last = std::unique(
        ranges.begin(), ranges.begin() + static_cast<std::ptrdiff_t>(face_number),
        [&Equal, &tet_mesh, &all_faces](int face_a, int face_b) {
            return Equal(
                tet_mesh, all_faces, static_cast<std::size_t>(face_a),
                static_cast<std::size_t>(face_b));
        });

    const std::size_t total_faces =
        static_cast<std::size_t>(last - ranges.begin());
    ranges[total_faces] = static_cast<int>(face_number);
    ranges.resize(total_faces + 1);

    std::vector<FacePair> faces(total_faces);
    for (std::size_t f = 0; f < faces.size(); ++f) {
        faces[f][0] = all_faces[static_cast<std::size_t>(ranges[f])];
        if (ranges[f + 1] - ranges[f] == 1)
            faces[f][1] = -1;
        else
            faces[f][1] =
                all_faces[static_cast<std::size_t>(ranges[f] + 1)];
    }
    return faces;
}

}  // namespace

std::array<int, 3> TetFace(std::size_t f) {
    switch (f) {
    case 0:
        return {1, 2, 3};
    case 1:
        return {0, 3, 2};
    case 2:
        return {0, 1, 3};
    case 3:
        return {0, 2, 1};
    default:
        return {-1, -1, -1};
    }
}

void validate_tet_mesh(
    const std::vector<int>& tets,
    const std::vector<Vec3>& rest_positions,
    double relative_degeneracy_tolerance) {
    validate_tolerance(relative_degeneracy_tolerance);
    validate_rest_positions(rest_positions);

    if (tets.size() % 4 != 0) {
        throw std::invalid_argument(
            "tet connectivity must contain exactly four indices per tetrahedron");
    }

    std::unordered_set<TetKey, IndexArrayHash<4>> unique_tets;
    unique_tets.reserve(tets.size() / 4);

    for (std::size_t t = 0; t < tets.size() / 4; ++t) {
        const std::array<int, 4> tet{
            tets[4 * t], tets[4 * t + 1], tets[4 * t + 2], tets[4 * t + 3]};

        for (const int vertex : tet) {
            if (vertex < 0
                || static_cast<std::size_t>(vertex) >= rest_positions.size()) {
                std::ostringstream message;
                message << tet_label(t) << " has out-of-range vertex index "
                        << vertex;
                throw std::out_of_range(message.str());
            }
        }

        const TetKey key = canonical_tet(tet[0], tet[1], tet[2], tet[3]);
        if (std::adjacent_find(key.begin(), key.end()) != key.end()) {
            throw std::invalid_argument(
                tet_label(t) + " contains a repeated vertex index");
        }
        if (!unique_tets.insert(key).second) {
            throw std::invalid_argument(
                tet_label(t) + " duplicates an earlier tetrahedron");
        }

        validate_tet_geometry(
            tet, t, rest_positions, relative_degeneracy_tolerance);
    }
}

std::vector<int> compute_boundary_tri_mesh(
    const std::vector<int>& tet_mesh) {
    validate_boundary_tet_topology(tet_mesh);

    std::vector<int> tri_mesh;
    const std::vector<FacePair> faces = ComputeTetMeshFaces(tet_mesh);

    for (std::size_t f = 0; f < faces.size(); ++f) {
        if (faces[f][1] == -1) {
            const int n = faces[f][0];
            const std::size_t local = static_cast<std::size_t>(n) % 4;
            const std::size_t element = static_cast<std::size_t>(n) / 4;
            const std::array<int, 3> local_face_indices = TetFace(local);
            tri_mesh.push_back(
                tet_mesh[4 * element + local_face_indices[0]]);
            tri_mesh.push_back(
                tet_mesh[4 * element + local_face_indices[1]]);
            tri_mesh.push_back(
                tet_mesh[4 * element + local_face_indices[2]]);
        }
    }
    return tri_mesh;
}

std::vector<int> compute_boundary_tri_mesh_nodes(
    const std::vector<int>& boundary_mesh) {
    if (boundary_mesh.size() % 3 != 0) {
        throw std::invalid_argument(
            "boundary triangle connectivity must contain three indices per triangle");
    }
    for (const int node : boundary_mesh) {
        if (node < 0) {
            throw std::out_of_range(
                "boundary triangle connectivity has a negative index");
        }
    }

    std::vector<int> boundary_vertices;
    std::unordered_set<int> node_set;
    for (std::size_t i = 0; i < boundary_mesh.size(); ++i) {
        const int node = boundary_mesh[i];

        if (node_set.find(node) == node_set.end()) {
            node_set.insert(node);
            boundary_vertices.push_back(node);
        }
    }
    return boundary_vertices;
}
