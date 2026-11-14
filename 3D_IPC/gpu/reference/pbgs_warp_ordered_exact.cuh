// Private alternative to the original serial ordered GPU kernel. A full warp
// processes the SAME vertex sequence; vertices never execute concurrently.
inline bool pbgs_warp_ordered_enabled() {
    const char* value=std::getenv("IPC_GPU_WARP_ORDERED_COLOR");
    if (!value) return true;
    if (std::string(value)=="0") return false;
    if (std::string(value)=="1") return true;
    throw std::invalid_argument("IPC_GPU_WARP_ORDERED_COLOR must be 0 or 1");
}

__global__ void pbgs_ordered_greedy_warp_exact(const std::size_t* offsets,
    const int* adjacency,int n,const int* order,int* colors,int* forbidden,int* count) {
    constexpr unsigned full=0xffffffffu;
    const int lane=int(threadIdx.x);
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
    if(lane==0)*count=largest+1;
}
