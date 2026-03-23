cmake_minimum_required(VERSION 3.10)

project(CorotatedEnergy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)

add_executable(corated_test
    corated_test.cpp
    corotated_energy.cpp
)

target_link_libraries(corated_test
    PRIVATE
    Eigen3::Eigen
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(corated_test)
