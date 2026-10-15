#pragma once
#include "resident_contact.cuh"

#ifdef __CUDACC__
#define IPC_INPUT_PUBLICATION_HD __host__ __device__
#else
#define IPC_INPUT_PUBLICATION_HD
#endif

namespace ipc_gpu::resident {

IPC_INPUT_PUBLICATION_HD inline bool contact_status_queues_exact(ContactStatus status) {
    return status == ContactStatus::NeedsExactGeometry
        || status == ContactStatus::ZeroDistance
        || status == ContactStatus::NonFiniteDerivative;
}

// Exact kernels are the only consumers of the persistent ContactInput array.
// In the ordinary mode, gather writes that array directly, as before. In the
// opt-in mode, gather evaluates an identical local input and publishes it only
// when its status requires the later exact kernel. Publication must precede
// queue insertion; completion of the gather kernel joins every publication
// before the sequential exact/cooperative/retry kernels can consume the queue.
template<bool Deferred> struct ContactInputPublication;

template<> struct ContactInputPublication<false> {
    ContactInput& input;
    IPC_INPUT_PUBLICATION_HD explicit ContactInputPublication(ContactInput* destination)
        : input(*destination) {}
    IPC_INPUT_PUBLICATION_HD bool publish_if_queued(ContactStatus status) const {
        return contact_status_queues_exact(status);
    }
};

template<> struct ContactInputPublication<true> {
    ContactInput input;
    ContactInput* destination;
    IPC_INPUT_PUBLICATION_HD explicit ContactInputPublication(ContactInput* target)
        : destination(target) {}
    IPC_INPUT_PUBLICATION_HD bool publish_if_queued(ContactStatus status) const {
        if (!contact_status_queues_exact(status)) return false;
        *destination = input;
        return true;
    }
};

} // namespace ipc_gpu::resident
#undef IPC_INPUT_PUBLICATION_HD
