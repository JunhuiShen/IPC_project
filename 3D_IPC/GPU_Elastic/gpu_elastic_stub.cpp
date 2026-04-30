// CPU-stub implementation of gpu_elastic.h. Used in builds without CUDA.
// Returns false from run_substep to signal "not available" so the bridge
// falls back to the CPU parallel solver.

#include "gpu_elastic.h"

void gpu_elastic_init(const RefMesh&, const VertexTriangleMap&,
                      const std::vector<Pin>&, const SimParams&) {}
void gpu_elastic_shutdown() {}
void gpu_elastic_set_pin_targets(const std::vector<Pin>&) {}
bool gpu_elastic_run_substep(const std::vector<Vec3>&, const std::vector<Vec3>&,
                             int, std::vector<Vec3>&) { return false; }
void gpu_elastic_begin_frame(const std::vector<Vec3>&, const std::vector<Vec3>&) {}
bool gpu_elastic_run_substep_device(int) { return false; }
void gpu_elastic_peek_positions(std::vector<Vec3>&) {}
void gpu_elastic_end_frame(std::vector<Vec3>&, std::vector<Vec3>&) {}
double gpu_elastic_substep_residual(int) { return -1.0; }
double gpu_elastic_last_residual() { return -1.0; }
