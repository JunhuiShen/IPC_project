#pragma once

#include "general_simd_assembly.h"
#include "kernels.h"
#include "solid_types.h"

#include <vector>

namespace ipc_gpu {

// Gather a complete independent mixed color before any of its particle or
// rigid blocks update positions. CUDA outputs are private element records;
// accumulation retains the CPU solver's per-node incident-element order.
class GeneralColorBridge {
public:
    void prepare_color(const std::vector<int>& blocks,
        const std::vector<int>& cloth_nodes, const std::vector<int>& solid_nodes,
        int solid_begin, int rigid_begin, const RefMesh& mesh,
        const solver_detail::GeneralSimdMaterials& materials,
        const std::vector<Pin>& pins, const PinMap& pin_map,
        const SimParams& params, const std::vector<Vec3>& positions,
        const std::vector<Vec3>& predicted);

    // Matches the CPU assembly interface so the checked solver overlay changes
    // only the call target. SDF evaluation stays on the CPU, at its original
    // batch boundary; contacts consume these systems through unchanged code.
    void prepare_batch(const int* nodes, std::size_t count, const RefMesh& mesh,
        const std::vector<IncidentTriangles>& incident,
        const std::vector<ShapeGrads>& shape_gradients,
        const std::vector<Pin>& pins, const PinMap& pin_map,
        const SimParams& params, const std::vector<Vec3>& positions,
        const std::vector<Vec3>& predicted, const std::vector<Vec3>* previous,
        const std::vector<unsigned char>& solid_mask,
        const std::vector<unsigned char>& surface_mask,
        solver_detail::GeneralSimdVertexSystem* outputs,
        const solver_detail::GeneralSimdMaterials* materials) const;

private:
    std::vector<int> nodes_;
    std::vector<unsigned char> point_solid_;
    std::vector<Point> points_;
    std::vector<Triangle> triangles_;
    std::vector<Solid> solids_;
    std::vector<Hinge> hinges_;
    std::vector<std::size_t> triangle_offsets_, solid_offsets_, hinge_offsets_;
    std::vector<Derivatives> point_outputs_, triangle_outputs_, solid_outputs_, hinge_outputs_;
    std::vector<solver_detail::GeneralSimdVertexSystem> systems_;
    std::vector<std::size_t> node_generation_;
    std::size_t generation_ = 0;
};

} // namespace ipc_gpu
