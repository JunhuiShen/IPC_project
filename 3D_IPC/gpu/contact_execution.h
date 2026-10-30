#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ipc_gpu::solver {
// Execution-only diagnostic switches. They never alter candidate membership,
// coloring, arithmetic precision, step limits, or contact tolerances.
inline bool contact_execution_option(const char* name,bool default_value=true) {
    const char* value=std::getenv(name);
    if(!value||!*value)return default_value;
    if(std::strcmp(value,"0")==0)return false;
    if(std::strcmp(value,"1")==0)return true;
    throw std::invalid_argument(std::string(name)+" must be 0 or 1");
}
}
