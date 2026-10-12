#pragma once
#include "resident_box_certificate.cuh"
#include "resident_contact.cuh"
#include <cstddef>

struct ResidentBoxCertificateInput {
    ipc_gpu::resident::BpAabb boxes[4];
    double positions[4][3];
    double d_hat;
    bool vertex_face;
};
struct ResidentBoxCertificateResult {
    unsigned certified;
    ipc_gpu::resident::MovementBoxCertificateWitness witness;
};
void resident_box_certificate_gpu(const ResidentBoxCertificateInput* input,
    ResidentBoxCertificateResult* output, std::size_t count);
