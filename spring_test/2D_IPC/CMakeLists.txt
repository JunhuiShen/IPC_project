cmake_minimum_required(VERSION 3.16)
project(2D_IPC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

# ── Core library ──────────────────────────────────────────────────────────────
add_library(ipc_core
    src/physics.cpp
    src/chain.cpp
    src/visualization.cpp
    src/broad_phase/bvh.cpp
    src/step_filter/ccd.cpp
    src/step_filter/trust_region.cpp
    src/initial_guess/trivial.cpp
    src/initial_guess/affine.cpp
    src/initial_guess/ccd.cpp
    src/initial_guess/trust_region.cpp
    src/solver.cpp
    src/example.cpp
    src/initial_guess/initial_guess.cpp
)

target_include_directories(ipc_core PUBLIC src)

# ── Simulation executable ─────────────────────────────────────────────────────
add_executable(simulation src/simulation.cpp)
target_link_libraries(simulation PRIVATE ipc_core)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()
