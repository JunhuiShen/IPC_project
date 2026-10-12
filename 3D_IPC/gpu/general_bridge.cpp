#include "general_bridge.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ipc_gpu {
namespace {

void assign(const Derivatives& value, solver_detail::GeneralSimdVertexSystem& output) {
    output = solver_detail::GeneralSimdVertexSystem{};
    for (int row = 0; row < 3; ++row) {
        output.gradient[row] = value.gradient[row];
        for (int column = 0; column < 3; ++column)
            output.hessian(row, column) = value.hessian[3 * row + column];
    }
}

void accumulate(const Derivatives& value, double dt2,
    solver_detail::GeneralSimdVertexSystem& output) {
    Vec3 gradient;
    Mat33 hessian;
    for (int row = 0; row < 3; ++row) {
        gradient[row] = value.gradient[row];
        for (int column = 0; column < 3; ++column)
            hessian(row, column) = value.hessian[3 * row + column];
    }
    output.gradient += dt2 * gradient;
    output.hessian += dt2 * hessian;
}

} // namespace

void GeneralColorBridge::prepare_color(const std::vector<int>& blocks,
    const std::vector<int>& cloth_nodes, const std::vector<int>& solid_nodes,
    int solid_begin, int rigid_begin, const RefMesh& mesh,
    const solver_detail::GeneralSimdMaterials& materials,
    const std::vector<Pin>& pins, const PinMap& pin_map,
    const SimParams& params, const std::vector<Vec3>& positions,
    const std::vector<Vec3>& predicted) {
    systems_.resize(positions.size());
    node_generation_.resize(positions.size(), 0);
    if (++generation_ == 0) {
        std::fill(node_generation_.begin(), node_generation_.end(), 0);
        generation_ = 1;
    }
    nodes_.clear();
    point_solid_.clear();
    for (int block : blocks)
        if (block < rigid_begin) {
            nodes_.push_back(block < solid_begin ? cloth_nodes.at(block)
                : solid_nodes.at(block - solid_begin));
            point_solid_.push_back(block >= solid_begin);
        }
    const auto count = nodes_.size();
    triangle_offsets_.assign(count + 1, 0);
    solid_offsets_.assign(count + 1, 0);
    hinge_offsets_.assign(count + 1, 0);
    for (std::size_t i = 0; i < count; ++i) {
        const int node = nodes_[i];
        triangle_offsets_[i + 1] = triangle_offsets_[i]
            + materials.triangles.node_offsets[node + 1] - materials.triangles.node_offsets[node];
        solid_offsets_[i + 1] = solid_offsets_[i]
            + materials.tets.node_offsets[node + 1] - materials.tets.node_offsets[node];
        hinge_offsets_[i + 1] = hinge_offsets_[i] + (params.kB > 0.0
            ? materials.hinges.node_offsets[node + 1] - materials.hinges.node_offsets[node] : 0);
    }
    points_.resize(count);
    triangles_.resize(triangle_offsets_.back());
    solids_.resize(solid_offsets_.back());
    hinges_.resize(hinge_offsets_.back());
    point_outputs_.resize(count);
    triangle_outputs_.resize(triangles_.size());
    solid_outputs_.resize(solids_.size());
    hinge_outputs_.resize(hinges_.size());
    #pragma omp parallel for schedule(static) if(params.use_parallel && count >= 128)
    for (std::size_t i = 0; i < count; ++i) {
        const int node = nodes_[i];
        auto& point = points_[i];
        point.mass = mesh.mass[node];
        point.pinned = pin_map[node] >= 0;
        point.solid = point_solid_[i];
        for (int axis = 0; axis < 3; ++axis) {
            point.position[axis] = positions[node][axis];
            point.predicted[axis] = predicted[node][axis];
            point.target[axis] = point.pinned ? pins[pin_map[node]].target_position[axis] : positions[node][axis];
        }
        auto output = triangle_offsets_[i];
        const auto& triangles = materials.triangles;
        for (auto entry = triangles.node_offsets[node]; entry < triangles.node_offsets[node + 1]; ++entry, ++output) {
            auto& triangle = triangles_[output];
            for (int corner = 0; corner < 3; ++corner)
                for (int axis = 0; axis < 3; ++axis)
                    triangle.positions[3 * corner + axis] = positions[triangles.nodes[entry][corner]][axis];
            for (int row = 0; row < 2; ++row) {
                triangle.shape[row] = triangles.shape_gradients[entry][row];
                for (int column = 0; column < 2; ++column)
                    triangle.dm_inverse[2 * row + column] = triangles.dm_inverse[entry](row, column);
            }
            triangle.area = triangles.measures[entry];
        }
        output = solid_offsets_[i];
        const auto& tets = materials.tets;
        for (auto entry = tets.node_offsets[node]; entry < tets.node_offsets[node + 1]; ++entry, ++output) {
            auto& solid = solids_[output];
            for (int corner = 0; corner < 4; ++corner)
                for (int axis = 0; axis < 3; ++axis)
                    solid.positions[3 * corner + axis] = positions[tets.nodes[entry][corner]][axis];
            for (int row = 0; row < 3; ++row) {
                solid.shape[row] = tets.shape_gradients[entry][row];
                for (int column = 0; column < 3; ++column)
                    solid.dm_inverse[3 * row + column] = tets.dm_inverse[entry](row, column);
            }
            solid.measure = tets.measures[entry];
        }
        if (params.kB > 0.0) {
            output = hinge_offsets_[i];
            const auto& hinges = materials.hinges;
            for (auto entry = hinges.node_offsets[node]; entry < hinges.node_offsets[node + 1]; ++entry, ++output) {
                auto& hinge = hinges_[output];
                for (int corner = 0; corner < 4; ++corner)
                    for (int axis = 0; axis < 3; ++axis)
                        hinge.positions[3 * corner + axis] = positions[hinges.nodes[entry][corner]][axis];
                hinge.active_node = hinges.active_nodes[entry];
                hinge.coefficient = hinges.coefficients[entry];
                hinge.rest_angle = hinges.rest_angles[entry];
            }
        }
    }
    Parameters parameters{};
    for (int axis = 0; axis < 3; ++axis) parameters.gravity[axis] = params.gravity[axis];
    parameters.kpin = params.kpin;
    parameters.dt2 = params.dt2();
    parameters.mu = params.mu;
    parameters.lambda = params.lambda;
    parameters.bending_stiffness = params.kB;
    if (count)
        compute(points_.data(), points_.size(), triangles_.data(), triangles_.size(),
            hinges_.data(), hinges_.size(), parameters, point_outputs_.data(),
            triangle_outputs_.data(), hinge_outputs_.data());
    if (!solids_.empty())
        compute_solids(solids_.data(), solids_.size(),
            SolidParameters{params.solid_mu, params.solid_lambda}, solid_outputs_.data());
    #pragma omp parallel for schedule(static) if(params.use_parallel && count >= 128)
    for (std::size_t i = 0; i < count; ++i) {
        const int node = nodes_[i];
        auto& system = systems_[node];
        assign(point_outputs_[i], system);
        for (auto entry = triangle_offsets_[i]; entry < triangle_offsets_[i + 1]; ++entry)
            accumulate(triangle_outputs_[entry], parameters.dt2, system);
        for (auto entry = solid_offsets_[i]; entry < solid_offsets_[i + 1]; ++entry)
            accumulate(solid_outputs_[entry], parameters.dt2, system);
        for (auto entry = hinge_offsets_[i]; entry < hinge_offsets_[i + 1]; ++entry)
            accumulate(hinge_outputs_[entry], parameters.dt2, system);
        node_generation_[node] = generation_;
    }
}

void GeneralColorBridge::prepare_batch(const int* nodes, std::size_t count,
    const RefMesh&, const std::vector<IncidentTriangles>&,
    const std::vector<ShapeGrads>&, const std::vector<Pin>&, const PinMap&,
    const SimParams& params, const std::vector<Vec3>& positions,
    const std::vector<Vec3>&, const std::vector<Vec3>* previous,
    const std::vector<unsigned char>& solid_mask,
    const std::vector<unsigned char>& surface_mask,
    solver_detail::GeneralSimdVertexSystem* outputs,
    const solver_detail::GeneralSimdMaterials*) const {
    constexpr std::size_t width = ipc_simd::tile_width;
    if (count > width) throw std::invalid_argument("GPU general batch exceeds SIMD tile width");
    for (std::size_t i = 0; i < count; ++i) {
        const auto node = static_cast<std::size_t>(nodes[i]);
        if (node >= systems_.size() || node_generation_[node] != generation_)
            throw std::logic_error("GPU general batch was not prepared for the current color");
        outputs[i] = systems_[node];
    }
    if (!(params.k_sdf > 0.0) || (params.sdf_planes.empty()
        && params.sdf_cylinders.empty() && params.sdf_spheres.empty())) return;
    std::array<Vec3, width> current, prior;
    std::array<std::size_t, width> owners;
    std::array<physics_detail::SdfDerivatives, width> values;
    std::size_t active = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const int node = nodes[i];
        if (solid_mask[node] && !surface_mask[node]) continue;
        owners[active] = i;
        current[active] = positions[node];
        if (params.friction_coefficient > 0.0) prior[active] = (*previous)[node];
        ++active;
    }
    physics_detail::compute_sdf_derivatives_tile(params, current.data(),
        params.friction_coefficient > 0.0 ? prior.data() : nullptr, active, values.data());
    for (std::size_t entry = 0; entry < active; ++entry) {
        auto& output = outputs[owners[entry]];
        output.gradient += params.dt2() * values[entry].gradient;
        output.hessian += params.dt2() * values[entry].hessian;
        if (solid_mask[nodes[owners[entry]]]) {
            output.sdf_friction_gradient = values[entry].friction_gradient;
            output.sdf_friction_hessian = values[entry].friction_hessian;
        } else if (params.friction_coefficient > 0.0) {
            output.gradient += values[entry].friction_gradient;
            output.hessian += values[entry].friction_hessian;
        }
    }
}

} // namespace ipc_gpu
