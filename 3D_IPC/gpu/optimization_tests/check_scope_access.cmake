cmake_minimum_required(VERSION 3.24)
foreach(_required CXX GPU_SOURCE_DIR TEST_SOURCE_DIR OUTPUT_ROOT)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "Missing ${_required} for compile-only scope check")
    endif()
endforeach()
if(NOT EXISTS "${CXX}")
    message(FATAL_ERROR "Selected C++ compiler does not exist: ${CXX}")
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef _nonce)
set(_run "${OUTPUT_ROOT}/${_nonce}")
file(MAKE_DIRECTORY "${_run}")
set(_flags -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Werror -pthread
    "-I${GPU_SOURCE_DIR}" "-I${TEST_SOURCE_DIR}")
foreach(_cuda_include IN LISTS CUDA_INCLUDE_DIRS)
    list(APPEND _flags -isystem "${_cuda_include}")
endforeach()
execute_process(
    COMMAND "${CXX}" ${_flags} -c "${TEST_SOURCE_DIR}/scope_access_positive.cpp"
        -o "${_run}/positive.cpp.o"
    RESULT_VARIABLE _positive
    OUTPUT_FILE "${_run}/positive.stdout.log"
    ERROR_FILE "${_run}/positive.stderr.log")
if(NOT "${_positive}" STREQUAL "0")
    message(FATAL_ERROR "Positive actual-class scope compile failed; see ${_run}/positive.stderr.log")
endif()
execute_process(
    COMMAND "${CXX}" ${_flags} -c "${TEST_SOURCE_DIR}/scope_access_untrusted_rejected.cpp"
        -o "${_run}/untrusted.cpp.o"
    RESULT_VARIABLE _negative
    OUTPUT_FILE "${_run}/untrusted.stdout.log"
    ERROR_FILE "${_run}/untrusted.stderr.log")
file(WRITE "${_run}/result.txt"
    "compile_only=true\npositive_return=${_positive}\nuntrusted_return=${_negative}\nGPU_execution=false\n")
if("${_negative}" STREQUAL "0")
    message(FATAL_ERROR "Untrusted code constructed a positive scope; evidence ${_run}")
endif()
if(NOT "${_negative}" MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "Negative compile did not exit normally with a compiler rejection; see ${_run}")
endif()
file(READ "${_run}/untrusted.stderr.log" _diagnostic)
if(NOT _diagnostic MATCHES "private")
    message(FATAL_ERROR "Negative compile failed without the expected private-access diagnostic; see ${_run}")
endif()
message(STATUS "Actual-class positive scope and untrusted rejection passed (compile-only): ${_run}")
