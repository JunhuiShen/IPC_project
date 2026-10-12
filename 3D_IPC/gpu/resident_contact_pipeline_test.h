#pragma once
#include "resident_contact.cuh"
#include "resident_sdf.cuh"
#include "kernels.h"

struct ResidentPipelineTestInput {
    double positions[8][3]{},previous[8][3]{},references[8][3]{};
    double shift[2][3]{};
    double previous_orientation[2][4]{{1,0,0,0},{1,0,0,0}};
    double omega[2][3]{},omega_after[2][3]{};
    ipc_gpu::resident::SdfCollider collider;
    ipc_gpu::resident::SdfParameters sdf_parameters;
    int kind=0,initial_error=0;
};
struct ResidentPipelineTestOutput {
    ipc_gpu::Derivatives deformable[8],rigid_com[2],rigid_omega[2],rigid_omega_after[2];
    int error=0,incident_counts[10]{};
};
void resident_contact_pipeline_test_gpu(const ResidentPipelineTestInput&,ResidentPipelineTestOutput&);
