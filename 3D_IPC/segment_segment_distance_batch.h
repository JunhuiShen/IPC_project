#pragma once

#include "segment_segment_distance.h"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

// Reuse exact edge geometry within one frozen batch, without exposing Boost
// types to solver/GPU headers. A workspace supports sequential batches only.
// Exact norms use legacy OGC's 16-byte-aligned return convention: Eigen's
// stableNorm can round differently for scalar callers with other alignments.
class SegmentSegmentDistanceBatchWorkspace {
public:
    SegmentSegmentDistanceBatchWorkspace();
    ~SegmentSegmentDistanceBatchWorkspace();
    SegmentSegmentDistanceBatchWorkspace(SegmentSegmentDistanceBatchWorkspace&&) noexcept;
    SegmentSegmentDistanceBatchWorkspace& operator=(SegmentSegmentDistanceBatchWorkspace&&) noexcept;

    // Call outside the worker team. Bind arrays and allocate scalar scratch;
    // positions are not read until compute_collective(), and may be filled by
    // the caller's preceding workshare. All indices must be valid. Each pair
    // uses the endpoint order of its two referenced edges, exactly as the
    // scalar call (edge0.start, edge0.end, edge1.start, edge1.end).
    void prepare(const std::vector<Vec3>& positions,
        const std::vector<std::array<int, 2>>& edges,
        const std::vector<std::array<int, 2>>& pair_edges,
        double eps, bool exact_computation_fallback,
        std::vector<SegmentSegmentDistanceResult>& outputs);

    // Every worker in the caller's OpenMP team must enter. Workshares join
    // classification, used-edge preparation and exact solves. The inputs must
    // stay frozen throughout. Exceptions are captured while barriers drain.
    void compute_collective();
    bool failed() const noexcept;
    void rethrow_if_failed() const;

    std::size_t fallback_pair_count() const noexcept;
    std::size_t prepared_edge_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
