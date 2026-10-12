#pragma once

#include <cstddef>
#include <cstdint>

// Plain double-precision AoS records: no Eigen, host pointers, SIMD alignment,
// mesh lookups, or shared-vertex writes enter the CUDA compute kernels.
namespace ipc_gpu {
struct Derivatives {
    double gradient[3];
    double hessian[9]; // row-major
};
struct Point {
    double position[3], predicted[3], target[3], mass;
    int pinned;
    int solid = 0; // solid point terms retain the CPU solid FMA order
};
struct Triangle {
    double positions[9]; // vertex-major xyz
    double dm_inverse[4]; // row-major
    double shape[2], area;
};
struct Hinge {
    double positions[12];
    double coefficient, rest_angle;
    int active_node;
};
struct Parameters {
    double gravity[3], kpin, dt2, mu, lambda, bending_stiffness;
};
struct Statistics {
    std::uint64_t calls = 0, launches = 0;
    std::uint64_t points = 0, triangles = 0, hinges = 0;
    std::uint64_t solids = 0, rigid_contacts = 0, rigid_inertia = 0;
    double host_to_device_ms = 0, kernel_ms = 0, device_to_host_ms = 0;
};

// One call covers a whole conflict-graph color, not a CPU tile of 8 entries.
// Results remain private per record; the organizer accumulates in CPU order.
// Throws on CUDA errors; never silently falls back to a CPU compute kernel.
void compute(const Point*, std::size_t, const Triangle*, std::size_t,
             const Hinge*, std::size_t, const Parameters&,
             Derivatives*, Derivatives*, Derivatives*);
const char* device_name();
Statistics statistics();
void reset_statistics();
} // namespace ipc_gpu
