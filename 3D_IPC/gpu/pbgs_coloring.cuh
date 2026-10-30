// Included inside broadphase.cu's anonymous namespace after Buffer.
// Alternative color construction only. The full original conflict graph,
// sequential color execution, numerical kernels and safety semantics remain.
inline int pbgs_coloring_mode() {
    const char* s=std::getenv("IPC_GPU_COLORING");
    if(!s||std::string(s)=="greedy")return 0;
    if(std::string(s)=="cusparse")return 1;
    if(std::string(s)=="degree")return 2;
    if(std::string(s)=="spatial")return 3;
    throw std::invalid_argument("IPC_GPU_COLORING must be greedy, cusparse, degree or spatial");
}
inline void sparse_check(cusparseStatus_t status) {
    if(status!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error("cuSPARSE coloring failed: "+std::to_string(int(status)));
}
struct PbgsColoringStorage {
    Buffer<int> offsets,order,anchors,occupied;
    Buffer<float> values;
    Buffer<std::uint64_t> keys;
    cusparseHandle_t handle=nullptr;
    cusparseMatDescr_t descriptor=nullptr;
    cusparseColorInfo_t info=nullptr;
    std::ofstream diagnostics;
    ~PbgsColoringStorage() {
        if(info)cusparseDestroyColorInfo(info);
        if(descriptor)cusparseDestroyMatDescr(descriptor);
        if(handle)cusparseDestroy(handle);
    }
    void vendor_init() {
        if(!handle)sparse_check(cusparseCreate(&handle));
        if(!descriptor)sparse_check(cusparseCreateMatDescr(&descriptor));
        if(!info)sparse_check(cusparseCreateColorInfo(&info));
        sparse_check(cusparseSetMatType(descriptor,CUSPARSE_MATRIX_TYPE_GENERAL));
        sparse_check(cusparseSetMatIndexBase(descriptor,CUSPARSE_INDEX_BASE_ZERO));
        sparse_check(cusparseSetPointerMode(handle,CUSPARSE_POINTER_MODE_HOST));
        sparse_check(cusparseSetStream(handle,nullptr));
    }
};
__global__ void pbgs_convert_offsets(const std::size_t* in,int* out,int n) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<=std::size_t(n);i+=std::size_t(blockDim.x)*gridDim.x)out[i]=int(in[i]);
}
__global__ void pbgs_anchors(const int* owners,int n,int* anchors) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)atomicMin(anchors+owners[i],i);
}
__global__ void pbgs_priority(const std::size_t* offsets,int n,std::uint64_t* keys,
    const int* anchors,const BpAabb* boxes,int mode,double dx,int* failed) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
    unsigned parity=0;
    if(mode==3&&anchors[i]!=INT_MAX) {
        for(int a=0;a<3;++a) {
            const auto& b=boxes[anchors[i]];
            const double q=floor((.5*b.lo[a]+.5*b.hi[a])/dx);
            if(!isfinite(q)||fabs(q)>=4503599627370496.){atomicCAS(failed,0,6);return;}
            parity|=unsigned(static_cast<unsigned long long>(static_cast<long long>(q))&1ull)<<a;
        }
    }
    keys[i]=(std::uint64_t(parity)<<32)|(0xffffffffu-unsigned(offsets[i+1]-offsets[i]));
    }
}
__global__ void pbgs_ordered_greedy(const std::size_t* offsets,const int* adjacency,int n,
    const int* order,int* colors,int* forbidden,int* count) {
    int largest=-1;
    for(int position=0;position<n;++position) {
        const int v=order[position];
        for(std::size_t k=offsets[v];k<offsets[v+1];++k) {
            const int c=colors[adjacency[k]];if(c>=0)forbidden[c]=v;
        }
        int c=0;while(c<n&&forbidden[c]==v)++c;
        colors[v]=c;largest=max(largest,c);
    }
    *count=largest+1;
}
// Independently checks every CSR edge before colors may index a histogram.
// Covers elastic cliques, inactive candidate contacts and nonlocal dependencies.
__global__ void pbgs_validate(const std::size_t* offsets,const int* adjacency,int n,
    const int* colors,int count,int* failed) {
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<n;v+=blockDim.x*gridDim.x) {
    const int c=colors[v];
    if(c<0||c>=count){atomicCAS(failed,0,6);continue;}
    for(std::size_t k=offsets[v];k<offsets[v+1];++k)
        if(colors[adjacency[k]]==c){atomicCAS(failed,0,6);break;}
    }
}
