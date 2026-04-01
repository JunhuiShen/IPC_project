cmake_minimum_required(VERSION 3.10)

project(CorotatedEnergy LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)

add_library(corotated_energy_lib
    corotated_energy.cpp
    IPC_math.cpp
    node_triangle_distance.cpp
    segment_segment_distance.cpp
    barrier_energy.cpp
)

target_link_libraries(corotated_energy_lib
    PUBLIC Eigen3::Eigen
)

target_include_directories(corotated_energy_lib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(corotated_energy_test
    corotated_energy_test.cpp
)

target_link_libraries(corotated_energy_test
    PRIVATE
    corotated_energy_lib
)

add_executable(node_triangle_distance_test
    node_triangle_distance_test.cpp
)

target_link_libraries(node_triangle_distance_test
    PRIVATE
    corotated_energy_lib
)

add_executable(segment_segment_distance_test
    segment_segment_distance_test.cpp
)

target_link_libraries(segment_segment_distance_test
    PRIVATE
    corotated_energy_lib
)

add_executable(barrier_energy_test
    barrier_energy_test.cpp
)

target_link_libraries(barrier_energy_test
    PRIVATE
    corotated_energy_lib
)

add_executable(3D_sim
    make_shape.cpp
    physics.cpp
    solver.cpp
    visualization.cpp
    simulation.cpp
)

target_link_libraries(3D_sim
    PRIVATE
    corotated_energy_lib
)

add_executable(make_shape_test
    make_shape_test.cpp
    make_shape.cpp
    physics.cpp
)

target_link_libraries(make_shape_test
    PRIVATE
    corotated_energy_lib
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(make_shape_test)

add_executable(total_energy_test
    total_energy_test.cpp
    make_shape.cpp
    physics.cpp
    solver.cpp
    visualization.cpp
)

target_link_libraries(total_energy_test
    PRIVATE
    corotated_energy_lib
)