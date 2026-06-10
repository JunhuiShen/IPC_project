find_package(GTest REQUIRED)

add_executable(rigid_body_ccd_test rigid_body_ccd_test.cpp)
target_link_libraries(rigid_body_ccd_test PRIVATE ipc_core GTest::gtest GTest::gtest_main)
target_include_directories(rigid_body_ccd_test PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(GoogleTest)
gtest_discover_tests(rigid_body_ccd_test)
