#pragma once

__global__ void pbgs_ordered_greedy_warp_shared_marks_exact(const std::size_t* offsets,
    const int* adjacency,int n,const int* order,int* colors,int* global_forbidden,int* count) {
    constexpr unsigned full=0xffffffffu;
    const int lane=int(threadIdx.x);
    extern __shared__ int forbidden[];
    for(int c=lane;c<n;c+=32)forbidden[c]=global_forbidden[c];
    __syncwarp(full);
    int largest=-1;
    for(int position=0;position<n;++position) {
        const int v=order[position];
        for(std::size_t k=offsets[v]+std::size_t(lane);k<offsets[v+1];k+=32) {
            const int c=colors[adjacency[k]];
            // Duplicate neighbors can write the same mark. Atomic exchange
            // makes those writes race-free without changing the exact value.
            if(c>=0)atomicExch(forbidden+c,v);
        }
        __syncwarp(full);
        int chosen=n; // Exact original sentinel, including a partial last batch.
        for(std::size_t base=0;base<static_cast<std::size_t>(n);base+=32) {
            const std::size_t c=base+static_cast<std::size_t>(lane);
            const unsigned available=__ballot_sync(full,c<static_cast<std::size_t>(n)&&forbidden[c]!=v);
            // An available lane is strictly below n, so this conversion is
            // representable even when the final palette batch is partial.
            if(available) {chosen=static_cast<int>(base+std::size_t(__ffs(available)-1));break;}
        }
        if(lane==0) {
            colors[v]=chosen;
            largest=max(largest,chosen);
        }
        // The next vertex observes exactly the preceding committed label.
        __syncwarp(full);
    }
    for(int c=lane;c<n;c+=32)global_forbidden[c]=forbidden[c];
    if(lane==0)*count=largest+1;
}

// Include the global exact warp header before this shared-memory route.
// The original global kernel remains the fallback above the default 48 KiB
// per-block dynamic shared-memory budget. No device-limit override is used.
inline bool pbgs_shared_marks_eligible(int n) {
    return n>0 && n<=12288;
}
inline void pbgs_launch_warp_shared_marks_exact(const std::size_t* offsets,
    const int* adjacency,int n,const int* order,int* colors,int* forbidden,int* count) {
    if(pbgs_shared_marks_eligible(n))
        pbgs_ordered_greedy_warp_shared_marks_exact<<<1,32,std::size_t(n)*sizeof(int)>>>(offsets,adjacency,n,order,colors,forbidden,count);
    else
        pbgs_ordered_greedy_warp_exact<<<1,32>>>(offsets,adjacency,n,order,colors,forbidden,count);
}

// Shared marks are the default for ordered Warp coloring within its budget.
inline bool pbgs_warp_shared_marks_enabled() {
    const char* value=std::getenv("IPC_GPU_WARP_SHARED_MARKS");
    if (!value) return true;
    if (std::string(value)=="0") return false;
    if (std::string(value)=="1") return true;
    throw std::invalid_argument("IPC_GPU_WARP_SHARED_MARKS must be 0 or 1");
}
