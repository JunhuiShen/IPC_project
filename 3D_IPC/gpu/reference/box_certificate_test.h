#pragma once
#include "box_certificate.cuh"
#include "contact.cuh"
#include <cstddef>

struct GpuBoxCertificateInput {
    ipc_gpu::solver::BpAabb boxes[4];
    double positions[4][3];
    double d_hat;
    bool vertex_face;
};
struct GpuBoxCertificateResult {
    unsigned certified;
    ipc_gpu::solver::MovementBoxCertificateWitness witness;
};
void gpu_box_certificate_gpu(const GpuBoxCertificateInput* input,
    GpuBoxCertificateResult* output, std::size_t count);
