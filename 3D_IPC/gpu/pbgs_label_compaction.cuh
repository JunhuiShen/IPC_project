// Normalize occupied vendor graph-color labels; validate full CSR before publication.
// Include beside pbgs_coloring.cuh inside broadphase.cu's anonymous
// namespace, after Buffer and CUDA/Thrust sort/unique headers are available.
//
// Required call contract:
//   1. Initialize all library output labels to -1 BEFORE csrcolor.
//   2. Call csrcolor; require its CUDA/library status to succeed.
//   3. Call pbgs_compact_occupied_labels(...), treating the returned number of
//      occupied labels as colors. Raw vendor ncolors is diagnostic only.
//   4. Run the existing full-CSR pbgs_validate and blocking require_status
//      BEFORE any histogram/worklist indexing or solver view publication.
//
// Storage is O(n), independent of raw vendor label maximum/ncolors. The mapping
// is the rank of each raw label in sorted unique labels, hence is injective and
// order-preserving over occupied labels. It cannot merge two distinct colors
// or repair an invalid same-color edge, and never changes graph dependencies.

__global__ void pbgs_remap_occupied_labels(const int* occupied,int count,
                                         int* labels,int n,int* failed) {
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<n;v+=blockDim.x*gridDim.x) {
        const int label=labels[v];
        if(label<0) { atomicCAS(failed,0,6);continue; }
        int lo=0,hi=count;
        while(lo<hi) {
            const int mid=lo+(hi-lo)/2;
            if(occupied[mid]<label)lo=mid+1;else hi=mid;
        }
        if(lo==count||occupied[lo]!=label) {atomicCAS(failed,0,6);continue;}
        labels[v]=lo;
    }
}

inline int pbgs_compact_occupied_labels(Buffer<int>& occupied,int* labels,int n,
                                        int* failed) {
    if(n<0)throw std::invalid_argument("negative PBGS label count");
    if(!n)return 0;
    occupied.reserve(n);
    check(cudaMemcpy(occupied.data,labels,std::size_t(n)*sizeof(int),cudaMemcpyDeviceToDevice),
          "copy raw vendor color labels");
    auto first=thrust::device_pointer_cast(occupied.data);
    thrust::sort(thrust::device,first,first+n);
    const auto end=thrust::unique(thrust::device,first,first+n);
    const int count=static_cast<int>(end-first); // <= n by construction.
    if(count<1||count>n)throw std::runtime_error("invalid compact PBGS color count");
    pbgs_remap_occupied_labels<<<blocks(n),threads>>>(occupied.data,count,labels,n,failed);
    launched("compact occupied vendor color labels");
    return count;
}
