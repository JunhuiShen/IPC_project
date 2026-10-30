#pragma once

#include "rigid_types.h"
#include <cstddef>
#include <type_traits>

namespace ipc_gpu::solver {

// Soft spring at a body-local material point, not a prescribed body pose.
// Material points need not coincide with any surface vertex.
struct RigidPinConstraint {
    int body = -1;
    double local_position[3]{};
    double initial_target[3]{};
    double rotation_center[3]{};
    double angular_velocity_z = 0;
};

// Device pointers. Pins are stably grouped by body; [offsets[b],offsets[b+1])
// preserves the CPU parameter-list order of pins belonging to that body.
struct RigidPinScene {
    std::size_t bodies = 0, count = 0;
    const RigidPinConstraint* pins = nullptr;
    const std::size_t* offsets = nullptr;
};

enum RigidPinError : int {
    RigidPinInvalidInput = 46,
    RigidPinInvalidDerivative = 47,
};

static_assert(std::is_standard_layout<RigidPinConstraint>::value);
static_assert(std::is_trivially_copyable<RigidPinConstraint>::value);

} // namespace ipc_gpu::solver
