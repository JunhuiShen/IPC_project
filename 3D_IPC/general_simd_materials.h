#pragma once

#include "physics.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace solver_detail {

template <std::size_t Corners, class Matrix, class Shape>
struct GeneralSimdElementMaterials {
    std::vector<std::size_t> node_offsets;
    std::vector<std::array<int, Corners>> nodes;
    std::vector<Matrix> dm_inverse;
    std::vector<double> measures;
    std::vector<Shape> shape_gradients;
};

struct GeneralSimdHingeMaterials {
    std::vector<std::size_t> node_offsets;
    std::vector<std::array<int, 4>> nodes;
    std::vector<int> active_nodes;
    std::vector<double> coefficients, rest_angles;
};

// Static AoS material records in original per-node incident order. Prepare
// once before the worker team starts; positions, mass, pins and constitutive
// coefficients remain dynamic inputs and are never cached here. Value
// snapshots detect in-place topology, adjacency and rest-property edits.
class GeneralSimdMaterials {
public:
    GeneralSimdElementMaterials<3, Mat22, Vec2> triangles;
    GeneralSimdElementMaterials<4, Mat33, Vec3> tets;
    GeneralSimdHingeMaterials hinges;

    bool matches(const RefMesh& mesh, const std::vector<IncidentTriangles>& incident,
        std::size_t node_count, bool with_bending) const {
        return valid_ && mesh_ == &mesh && node_count_ == node_count
            && with_bending_ == with_bending && incident_ == incident
            && tet_adjacency_ == mesh.tet_adj && hinge_adjacency_ == mesh.hinge_adj
            && triangle_nodes_ == mesh.tris && tet_nodes_ == mesh.tets
            && solid_nodes_ == mesh.tet_nodes && rigid_owner_ == mesh.node_to_rb
            && rigid_nodes_ == mesh.rb_nodes
            && same_matrix_values(dm_inverse_, mesh.Dm_inverse)
            && same_scalar_values(areas_, mesh.area)
            && same_hinges(hinges_source_, mesh.hinges)
            && same_tet_rest(tet_rest_, mesh.tet_rest_data);
    }

    // Returns true only when records were rebuilt. Partial rebuilds are never
    // marked reusable, so a failed allocation/invalid index cannot poison the
    // next solver call's cache.
    bool prepare(const RefMesh& mesh, const std::vector<IncidentTriangles>& incident,
        std::size_t node_count, bool with_bending) {
        if (matches(mesh, incident, node_count, with_bending)) return false;
        valid_ = false;
        clear_records(triangles); clear_records(tets);
        hinges.nodes.clear(); hinges.active_nodes.clear();
        hinges.coefficients.clear(); hinges.rest_angles.clear();
        triangles.node_offsets.resize(node_count + 1);
        tets.node_offsets.resize(node_count + 1);
        hinges.node_offsets.resize(node_count + 1);
        std::vector<unsigned char> solid(node_count, 0), rigid(node_count, 0);
        for (int node : mesh.tet_nodes) solid.at(static_cast<std::size_t>(node)) = 1;
        for (const auto& body : mesh.rb_nodes)
            for (int node : body) rigid.at(static_cast<std::size_t>(node)) = 1;
        for (std::size_t node = 0; node < std::min(node_count, mesh.node_to_rb.size()); ++node)
            if (mesh.node_to_rb[node] >= 0) rigid[node] = 1;

        for (std::size_t node = 0; node < node_count; ++node) {
            triangles.node_offsets[node] = triangles.nodes.size();
            tets.node_offsets[node] = tets.nodes.size();
            hinges.node_offsets[node] = hinges.nodes.size();
            if (solid[node]) {
                for (const auto& [element, role] : mesh.tet_adj.at(node)) {
                    const auto& rest = mesh.tet_rest_data.at(element);
                    std::array<int, 4> corners;
                    for (int j = 0; j < 4; ++j) corners[j] = mesh.tets.at(4 * static_cast<std::size_t>(element) + j);
                    tets.nodes.push_back(corners);
                    tets.dm_inverse.push_back(rest.Dm_inverse);
                    tets.measures.push_back(rest.measure);
                    tets.shape_gradients.push_back(rest.grad_N.at(role));
                }
            } else if (!rigid[node]) {
                if (node < incident.size())
                    for (const auto& [element, role] : incident[node]) {
                        // Match the assembly's treatment of surface-only faces.
                        if (element < 0 || static_cast<std::size_t>(element) >= mesh.Dm_inverse.size()) continue;
                        std::array<int, 3> corners;
                        for (int j = 0; j < 3; ++j) corners[j] = mesh.tris.at(3 * static_cast<std::size_t>(element) + j);
                        const Mat22& inverse = mesh.Dm_inverse[element];
                        triangles.nodes.push_back(corners);
                        triangles.dm_inverse.push_back(inverse);
                        triangles.measures.push_back(mesh.area.at(element));
                        triangles.shape_gradients.push_back(shape_function_gradients(inverse).at(role));
                    }
                const auto found = with_bending ? mesh.hinge_adj.find(static_cast<int>(node)) : mesh.hinge_adj.end();
                if (found != mesh.hinge_adj.end())
                    for (const auto& [element, role] : found->second) {
                        const auto& hinge = mesh.hinges.at(element);
                        hinges.nodes.push_back({hinge.v[0], hinge.v[1], hinge.v[2], hinge.v[3]});
                        hinges.active_nodes.push_back(role);
                        hinges.coefficients.push_back(hinge.c_e);
                        hinges.rest_angles.push_back(hinge.bar_theta);
                    }
            }
        }
        triangles.node_offsets[node_count] = triangles.nodes.size();
        tets.node_offsets[node_count] = tets.nodes.size();
        hinges.node_offsets[node_count] = hinges.nodes.size();
        incident_ = incident;
        tet_adjacency_ = mesh.tet_adj;
        hinge_adjacency_ = mesh.hinge_adj;
        triangle_nodes_ = mesh.tris; tet_nodes_ = mesh.tets;
        solid_nodes_ = mesh.tet_nodes; rigid_owner_ = mesh.node_to_rb;
        rigid_nodes_ = mesh.rb_nodes;
        dm_inverse_ = mesh.Dm_inverse; areas_ = mesh.area;
        hinges_source_ = mesh.hinges; tet_rest_ = mesh.tet_rest_data;
        mesh_ = &mesh; node_count_ = node_count; with_bending_ = with_bending;
        valid_ = true;
        return true;
    }

private:
    template <std::size_t Corners, class Matrix, class Shape>
    static void clear_records(GeneralSimdElementMaterials<Corners, Matrix, Shape>& records) {
        records.nodes.clear(); records.dm_inverse.clear();
        records.measures.clear(); records.shape_gradients.clear();
    }
    static bool same_scalar_values(const std::vector<double>& a, const std::vector<double>& b) {
        return a.size() == b.size() && (a.empty()
            || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
    }
    template <class Matrix>
    static bool same_matrix_values(const std::vector<Matrix>& a, const std::vector<Matrix>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (std::memcmp(a[i].data(), b[i].data(), sizeof(double) * a[i].size()) != 0) return false;
        return true;
    }
    static bool same_hinges(const std::vector<Hinge>& a, const std::vector<Hinge>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (!std::equal(a[i].v, a[i].v + 4, b[i].v)
                || std::memcmp(&a[i].bar_theta, &b[i].bar_theta, sizeof(double)) != 0
                || std::memcmp(&a[i].c_e, &b[i].c_e, sizeof(double)) != 0) return false;
        }
        return true;
    }
    static bool same_tet_rest(const std::vector<TetRestData>& a, const std::vector<TetRestData>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (std::memcmp(a[i].Dm_inverse.data(), b[i].Dm_inverse.data(), 9 * sizeof(double)) != 0
                || std::memcmp(&a[i].measure, &b[i].measure, sizeof(double)) != 0) return false;
            for (int role = 0; role < 4; ++role)
                if (std::memcmp(a[i].grad_N[role].data(), b[i].grad_N[role].data(), 3 * sizeof(double)) != 0) return false;
        }
        return true;
    }
    bool valid_ = false, with_bending_ = false;
    const RefMesh* mesh_ = nullptr;
    std::size_t node_count_ = 0;
    std::vector<IncidentTriangles> incident_, tet_adjacency_;
    VertexHingeMap hinge_adjacency_;
    std::vector<int> triangle_nodes_, tet_nodes_, solid_nodes_, rigid_owner_;
    std::vector<std::vector<int>> rigid_nodes_;
    std::vector<Mat22> dm_inverse_;
    std::vector<double> areas_;
    std::vector<Hinge> hinges_source_;
    std::vector<TetRestData> tet_rest_;
};

} // namespace solver_detail
