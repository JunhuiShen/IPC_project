#pragma once
#include "resident_contact.cuh"
#include <cstddef>

struct ResidentContactRejectionResult {
    unsigned certified=0;
    ipc_gpu::resident::ContactOutput output;
};

void resident_contact_rejection_gpu(const ipc_gpu::resident::ContactInput* input,
    ResidentContactRejectionResult* output,std::size_t count);
