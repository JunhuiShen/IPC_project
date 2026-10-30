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
} // namespace ipc_gpu
