#pragma once

#include "SIMD.h"
#include "broad_phase.h"
#include "general_simd_materials.h"
#include "physics.h"
#include "safe_step.h"

#include <functional>

namespace solver_detail {

// Private outputs for complete independent vertex ranges, just as in cloth
// experimental_v2. No kernel scatters into shared mesh positions.
struct GeneralSimdVertexSystem {
    Vec3 gradient = Vec3::Zero();
    Mat33 hessian = Mat33::Zero();
    Vec3 sdf_friction_gradient = Vec3::Zero();
    Mat33 sdf_friction_hessian = Mat33::Zero();
};

void prepare_general_simd_batch(
    const int* nodes, std::size_t count, const RefMesh& mesh,
    const std::vector<IncidentTriangles>& incident,
    const std::vector<ShapeGrads>& shape_gradients,
    const std::vector<Pin>& pins, const PinMap& pin_map,
    const SimParams& params, const std::vector<Vec3>& positions,
    const std::vector<Vec3>& predicted,
    const std::vector<Vec3>* previous,
    const std::vector<unsigned char>& solid_mask,
    const std::vector<unsigned char>& surface_mask,
    GeneralSimdVertexSystem* outputs,
    const GeneralSimdMaterials* materials = nullptr);

// Retains cloth's per-pair accumulation and solid's separate normal/friction
// totals. Tet-interior filtering is identical to the solid scalar path.
void accumulate_general_simd_contacts(
    int node, bool solid, const BroadPhase::Cache& cache,
    const SimParams& params, const std::vector<Vec3>& positions,
    const std::vector<Vec3>* previous,
    const std::vector<unsigned char>& solid_mask,
    const std::vector<unsigned char>& surface_mask, bool cooperative,
    GeneralSimdVertexSystem& output,
    safe_step_detail::VertexAabbRejections* rejections = nullptr,
    const std::function<void()>* leader_work = nullptr);

} // namespace solver_detail
