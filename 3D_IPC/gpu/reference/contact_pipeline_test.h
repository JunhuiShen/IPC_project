#pragma once
#include "contact.cuh"
#include "sdf.cuh"
#include "kernels.h"

struct GpuPipelineTestInput {
    double positions[8][3]{},previous[8][3]{},references[8][3]{};
    double shift[2][3]{};
    double previous_orientation[2][4]{{1,0,0,0},{1,0,0,0}};
    double omega[2][3]{},omega_after[2][3]{};
    ipc_gpu::solver::SdfCollider collider;
    ipc_gpu::solver::SdfParameters sdf_parameters;
    int kind=0,initial_error=0;
};
struct GpuPipelineTestOutput {
    ipc_gpu::Derivatives deformable[8],rigid_com[2],rigid_omega[2],rigid_omega_after[2];
    int error=0,incident_counts[10]{};
};
void gpu_contact_pipeline_test_gpu(const GpuPipelineTestInput&,GpuPipelineTestOutput&);
