#pragma once
#include "resident_elastic.h"
#include "resident_broadphase.h"
#include <memory>

namespace ipc_gpu::resident {

// CUDA port of the CPU standard initial guess, not a colored/single-moving-
// vertex sweep. All four vertices of every PT/EE query may move. Uses the
// Tight-Inclusion BFS defaults from ccd.cpp: separation 1e-10, tolerance 1e-6,
// t_max 1, max_itr 1000000, no_zero_toi false. One global minimum TOI scales
// every deformable displacement, with omega = toi >= 1 ? 1 : .9 * toi.
// All numerical data and the global minimum remain resident. Host reads only
// queue-overflow counts to grow scratch and resubmit unresolved GPU queries.
class ResidentGlobalGuess {
public:
    ResidentGlobalGuess();
    ~ResidentGlobalGuess();
    ResidentGlobalGuess(const ResidentGlobalGuess&) = delete;
    ResidentGlobalGuess& operator=(const ResidentGlobalGuess&) = delete;

    void apply(const DeformableScene&, const DeformableState&,
               const BroadPhaseTopology&, int* device_error);

    // Diagnostic/query API. All pointers are device pointers; endpoints are
    // exactly the represented endpoints (already reconstructed by apply).
    // Results are CPU-compatible clamped TOIs: 1 means no clipping. No caller
    // may consume output when device_error != 0. Status 50 invalid input,
    // 51 arithmetic overflow, 52 exhausted queue, 53 dyadic interval overflow.
    void query(const double (*start)[3], const double (*end)[3], int nodes,
               const BpContact* contacts, std::size_t count, double* tois,
               int* device_error);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ipc_gpu::resident
