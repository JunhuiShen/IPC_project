# Syntax/anchor generation check without a CUDA toolkit. The output is only
# generated solver/driver source; this does not compile or execute CUDA.
# Usage: cmake -P gpu/check_overlay.cmake
get_filename_component(CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(IPC_CPU_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(CMAKE_CURRENT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build-overlay-check")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" build_script)
string(FIND "${build_script}" "function(replace_once" begin)
string(FIND "${build_script}" "get_target_property(host_sources" end)
if(begin LESS 0 OR end LESS begin)
    message(FATAL_ERROR "Could not locate checked overlay generator")
endif()
math(EXPR length "${end}-${begin}")
string(SUBSTRING "${build_script}" ${begin} ${length} overlay)
cmake_language(EVAL CODE "${overlay}")
message(STATUS "Generated checked overlay: ${CMAKE_CURRENT_BINARY_DIR}/solver_gpu.cpp")

# Check the resident dispatch independently too: upstream changes to the CPU
# driver's rigid-only API must never leave a CPU frame call in the GPU driver.
string(FIND "${build_script}" "# Independent resident driver:" begin)
string(FIND "${build_script}" "add_executable(3D_sim_gpu_resident" end)
if(begin LESS 0 OR end LESS begin)
    message(FATAL_ERROR "Could not locate resident driver generator")
endif()
math(EXPR length "${end}-${begin}")
string(SUBSTRING "${build_script}" ${begin} ${length} resident_overlay)
cmake_language(EVAL CODE "${resident_overlay}")
if(resident_driver MATCHES "advance_one_frame(_general|_rb)?[ \t\r\n]*\\(")
    message(FATAL_ERROR "Resident driver still contains a CPU frame dispatch")
endif()
message(STATUS "Generated checked resident driver: ${CMAKE_CURRENT_BINARY_DIR}/simulation_resident.cpp")
