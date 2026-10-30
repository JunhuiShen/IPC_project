#pragma once

namespace ipc_gpu::solver {

struct DatParameters {
    double gamma = 0.9;
    double query_radius = 0.0075;
    double node_box_padding = 1.2;
};

// New device status values; an existing nonzero status is never cleared.
enum DatError : int {
    DatInvalidInput = 70,
    DatUnresolvedGeometry = 71,
    DatZeroGap = 72,
    DatExactCapacity = 73,
    DatNonfiniteOutput = 74
};

} // namespace ipc_gpu::solver
