#pragma once

#include "broad_phase.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>

namespace solver_detail {

struct GridCell {
    std::array<std::int64_t, 3> index{};
    AABB bounds;
    std::vector<int> vertices;
    int color_id = -1;
    int batch_id = -1;
};

// Sparse, world-origin-aligned checkerboard for the basic cloth solver. Cells
// have eight parity colors in 3D (four in one XY layer). Long elastic/contact
// dependencies can split one parity color into several execution batches.
struct ClothGridSchedule {
    double dx = 0.0;
    // Inclusive grid-index bounds of all node boxes, including empty cells.
    // An empty schedule has min_index={0,0,0}, max_index={-1,-1,-1}.
    std::array<std::int64_t, 3> min_index{{0, 0, 0}};
    std::array<std::int64_t, 3> max_index{{-1, -1, -1}};
    std::vector<GridCell> cells;
    std::vector<std::vector<int>> batches; // Cell indices, in execution order.
    std::vector<std::vector<int>> vertex_color_groups; // Same batches, flattened.

    // Positions anchor cell ownership until the next build. Every node box
    // must contain its anchor and extend strictly less than dx/2 from it in
    // every direction. Thus boxes in distinct cells of one parity cannot
    // touch. Actual solver read/write dependencies are additionally projected
    // to cells and made undirected before greedy batch coloring.
    // Empty dependencies means an empty graph; otherwise one row per vertex
    // is required. Invalid input throws without changing this schedule.
    void build(const std::vector<Vec3>& positions,
               const std::vector<AABB>& node_boxes,
               const std::vector<std::vector<int>>& dependencies,
               double cell_dx);

    // Start expensive cells first within each conflict-free batch. Costs sum
    // over the vertices in a cell; equal sums retain ascending cell indices.
    // Cell ownership, batch membership, and serial vertex order are unchanged.
    // Requires one cost per vertex. Invalid input or a sum overflow throws
    // without changing this schedule.
    void prioritize_cells(const std::vector<std::size_t>& vertex_costs);

    // Parallelize whole cells, keeping the vertex order within a cell serial.
    // Batch barriers preserve dependencies. The scalar path uses exactly the
    // same batch/cell/vertex ordering and is a deterministic reference.
    // On a callback exception the current batch joins, no later batch starts,
    // and the first observed exception is rethrown on the calling thread.
    template <class Process>
    void run(bool parallel, const Process& process) const {
        if (batches.empty()) return;
        if (!parallel) {
            for (const auto& batch : batches)
                for (int cell : batch)
                    for (int vertex : cells[cell].vertices)
                        process(vertex);
            return;
        }
        // Separate slots keep a fast worker's next-batch exception from
        // racing a slower worker still checking the preceding batch.
        std::vector<std::exception_ptr> errors(batches.size());
        #pragma omp parallel shared(errors)
        {
            for (std::size_t b = 0; b < batches.size(); ++b) {
                const auto& batch = batches[b];
                #pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < static_cast<int>(batch.size()); ++i) {
                    try {
                        for (int vertex : cells[batch[i]].vertices)
                            process(vertex);
                    } catch (...) {
                        #pragma omp critical(ipc_cloth_grid_callback_error)
                        { if (!errors[b]) errors[b] = std::current_exception(); }
                    }
                }
                if (errors[b]) break;
            }
        }
        for (const auto& error : errors)
            if (error) std::rethrow_exception(error);
    }
};

} // namespace solver_detail
