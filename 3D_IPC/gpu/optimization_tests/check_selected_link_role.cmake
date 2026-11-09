# Test-only post-link proof. Never runs CUDA or changes a solver binary.
if(NOT DEFINED EXECUTABLE OR NOT EXISTS "${EXECUTABLE}" OR
   NOT DEFINED LINK_MAP OR NOT EXISTS "${LINK_MAP}" OR
   NOT DEFINED NM OR NOT DEFINED ROLE)
    message(FATAL_ERROR "selected-test link-role audit requires actual executable/map/nm/role")
endif()
execute_process(COMMAND "${NM}" -C "${EXECUTABLE}"
    RESULT_VARIABLE _nm_status OUTPUT_VARIABLE _symbols ERROR_VARIABLE _nm_error)
if(NOT _nm_status EQUAL 0)
    message(FATAL_ERROR "selected-test nm failed: ${_nm_error}")
endif()
file(READ "${LINK_MAP}" _map)
set(_solver_member "(^|[ /\\(])solver\\.cu\\.o([) \n]|$)")
if(ROLE STREQUAL "PUBLIC")
    if(NOT _map MATCHES "${_solver_member}" OR
       NOT _symbols MATCHES "ipc_gpu::solver::GpuSolver::advance_substep")
        message(FATAL_ERROR "public Solver regression did not link production Solver")
    endif()
elseif(ROLE STREQUAL "FACADE" OR ROLE STREQUAL "COMPONENT" OR ROLE STREQUAL "STANDALONE")
    # A stand-in GpuSolver::bind may exist in a private-authority facade. The
    # real production object/constructors/destructor/step/snapshot may not.
    if(_map MATCHES "${_solver_member}" OR
       _symbols MATCHES "ipc_gpu::solver::GpuSolver::(GpuSolver|~GpuSolver|advance_substep|snapshot|Impl)")
        message(FATAL_ERROR "component/facade test accidentally linked production Solver")
    endif()
    if(ROLE STREQUAL "STANDALONE" AND _map MATCHES "(^|[ /\\(])contact_pipeline\\.cu\\.o([) \n]|$)")
        message(FATAL_ERROR "standalone evaluator oracle linked production ContactPipeline")
    endif()
else()
    message(FATAL_ERROR "unknown selected-test link role: ${ROLE}")
endif()
foreach(_api cudaMalloc cudaFree cudaMemcpy cudaLaunchKernel cudaFuncGetName
             cudaGraphLaunch cudaGraphInstantiateWithFlags cudaGraphExecKernelNodeSetParams)
    # GNU --wrap bodies are allowed. Unmangled CUDA API bodies in the test
    # image are not: native tests must use the real shared runtime.
    foreach(_name "${_api}" "__real_${_api}")
        if(_symbols MATCHES "[ \t][TtWwDdBb][ \t]+${_name}(@[^ \n]*)?([ \n]|$)")
            message(FATAL_ERROR "native test contains a CUDA API body/mock: ${_name}")
        endif()
    endforeach()
endforeach()
foreach(_api cudaMalloc cudaMemcpy)
    if(NOT _symbols MATCHES "[ \t]U[ \t]+${_api}(@[^ \n]*)?([ \n]|$)")
        message(FATAL_ERROR "native test lacks real dynamic CUDA import: ${_api}")
    endif()
endforeach()
find_program(_ldd ldd REQUIRED)
execute_process(COMMAND "${_ldd}" "${EXECUTABLE}"
    RESULT_VARIABLE _ldd_status OUTPUT_VARIABLE _dynamic ERROR_VARIABLE _ldd_error)
if(NOT _ldd_status EQUAL 0 OR NOT _dynamic MATCHES "libcudart\\.so" OR _dynamic MATCHES "/stubs/")
    message(FATAL_ERROR "native test lacks real shared CUDART: ${_ldd_error} ${_dynamic}")
endif()
message(STATUS "Selected-test link role ${ROLE} verified: ${EXECUTABLE}")
