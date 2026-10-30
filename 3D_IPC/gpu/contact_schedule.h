#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ipc_gpu::solver {

// Host-only launch configuration. This changes how many independent exact
// queries can run concurrently, not their arithmetic, capacity, ordering of
// per-node accumulation, or the second-pass retry policy. Construct once per
// GpuContactPipeline so an environment change cannot resize a live pool.
struct ExactContactSchedule {
    int small_workers=1024,small_threads=1;
    int large_workers=256,large_threads=4;

    struct ArenaSizes {std::size_t words,free_slots;};

    static int environment_integer(const char* name,int fallback,int maximum) {
        const char* value=std::getenv(name);
        if(!value)return fallback;
        if(!*value)throw std::invalid_argument(std::string(name)+" must be a positive integer");
        int result=0;
        for(const char* digit=value;*digit;++digit) {
            if(*digit<'0'||*digit>'9'||result>(maximum-(*digit-'0'))/10)
                throw std::invalid_argument(std::string(name)+" must be a positive integer <= "+std::to_string(maximum));
            result=result*10+(*digit-'0');
        }
        if(result<=0||result>maximum)
            throw std::invalid_argument(std::string(name)+" must be a positive integer <= "+std::to_string(maximum));
        return result;
    }

    void validate()const {
        const auto threads_valid=[](int count) {return count>=1&&count<=32&&(count&(count-1))==0;};
        if(small_workers<1||small_workers>4096||large_workers<1||large_workers>1024)
            throw std::invalid_argument("exact contact workers must be in [1,4096] (small) or [1,1024] (large)");
        if(!threads_valid(small_threads)||!threads_valid(large_threads))
            throw std::invalid_argument("exact contact threads must be one of 1,2,4,8,16,32");
        // Exact divisibility means the existing kernel needs no additional
        // worker guard and cannot access an arena outside the reserved range.
        if(small_workers%small_threads||large_workers%large_threads)
            throw std::invalid_argument("exact contact workers must be divisible by their threads per block");
    }

    static ExactContactSchedule from_environment() {
        ExactContactSchedule result;
        result.small_workers=environment_integer("IPC_GPU_CONTACT_EXACT_SMALL_WORKERS",1024,4096);
        result.small_threads=environment_integer("IPC_GPU_CONTACT_EXACT_SMALL_THREADS",1,32);
        result.large_workers=environment_integer("IPC_GPU_CONTACT_EXACT_LARGE_WORKERS",256,1024);
        result.large_threads=environment_integer("IPC_GPU_CONTACT_EXACT_LARGE_THREADS",4,32);
        result.validate();return result;
    }

    ArenaSizes arena_sizes(std::size_t candidates,int small_limbs,int large_limbs,int slots)const {
        const auto small=std::min(candidates,std::size_t(small_workers));
        const auto large=std::min(candidates,std::size_t(large_workers));
        return {std::max(small*small_limbs,large*large_limbs)*slots,
                std::max(small,large)*slots};
    }
};

} // namespace ipc_gpu::solver
