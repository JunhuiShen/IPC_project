// TEST ONLY: separately compiled full and constant-k0 probe kernels. The full
// kernel publishes ALL1416 bytes, keeping its Hessian work observable.
#define main independent_host_oracle_not_executed
#include "host_sparse_probe_test.cpp"
#undef main
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>
#include <cfloat>
#include <utility>

namespace {
struct ProbeRecord {gpu::original_contact_storage::SelfZeroOutput out;unsigned discard;};
static_assert(sizeof(gpu::ContactOutput)==1416,"full observable publication ABI");
__global__ void sparse_oracle_full(const gpu::ContactInput* inputs,std::size_t n,gpu::ContactOutput* outputs) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)outputs[i]=gpu::evaluate_contact_original(inputs[i]);
}
__global__ void sparse_oracle_probe(const gpu::ContactInput* inputs,std::size_t n,ProbeRecord* outputs) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n)return;
    auto copy=inputs[i];copy.k_barrier=0.0;copy.hessian_blocks=1u;copy.friction=0.0;
    const auto out=gpu::evaluate_contact_original_self_zero(copy,0);
    outputs[i].out=out;outputs[i].discard=policy::discard_force(inputs[i],out);
}
__global__ void sparse_oracle_boundaries(const gpu::ContactInput* base,
    const gpu::ContactOutput* full,std::size_t n,gpu::ContactInput* derived,unsigned char* valid) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n)return;
    const double distance=full[i].distance;
    const bool usable=gpu::contact_detail::finite(distance)&&distance>0.0;
    for(unsigned which=0;which<3;++which) {
        auto input=base[i];
        if(usable)input.d_hat=which==0?nextafter(distance,0.0):which==1?distance:nextafter(distance,CUDART_INF);
        derived[3*i+which]=input;valid[3*i+which]=usable;
    }
}
__device__ bool same_double_bits(double a,double b){return __double_as_longlong(a)==__double_as_longlong(b);}
__global__ void sparse_oracle_compare(const gpu::ContactInput* inputs,const gpu::ContactOutput* full,
    const ProbeRecord* probe,std::size_t n,const unsigned char* boundary,
    const gpu::ContactOutput* base_full,unsigned long long* counts,unsigned long long* first_bad,
    std::size_t index_offset) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n)return;
    atomicAdd(counts,1ull);const auto& old=full[i];const auto& candidate=probe[i];
    unsigned failure=0;
    if(candidate.discard) {
        atomicAdd(counts+(inputs[i].kind==gpu::ContactKind::PointTriangle?1:2),1ull);
        if(old.status!=gpu::ContactStatus::Ok||old.active!=0)failure|=1;
        else {
            for(unsigned role=0;role<4;++role)for(unsigned axis=0;axis<3;++axis)
                if(old.barrier_gradient[role][axis]!=0.0)failure|=64;
            for(unsigned a=0;a<4;++a)for(unsigned b=0;b<4;++b)for(unsigned k=0;k<9;++k)
                if(old.barrier_hessian[a][b][k]!=0.0)failure|=64;
            for(unsigned k=0;k<3;++k)if(old.friction_relative_gradient[k]!=0.0)failure|=64;
            for(unsigned k=0;k<9;++k)if(old.friction_relative_hessian[k]!=0.0)failure|=64;
        }
    }
    if(candidate.out.status==gpu::ContactStatus::Ok&&candidate.out.active!=0)failure|=32;
    if(old.status!=gpu::ContactStatus::Ok) {
        atomicAdd(counts+3,1ull);if(candidate.discard)failure|=2;
    }
    if(old.status==gpu::ContactStatus::Ok&&candidate.out.status==gpu::ContactStatus::Ok) {
        bool equal=same_double_bits(old.distance,candidate.out.distance);
        for(unsigned k=0;k<4;++k)equal=equal&&same_double_bits(old.weights[k],candidate.out.weights[k]);
        for(unsigned k=0;k<3;++k)equal=equal&&same_double_bits(old.normal[k],candidate.out.normal[k]);
        atomicAdd(counts+4,1ull);if(!equal)failure|=4;
    }
    if(boundary&&boundary[i]) {
        atomicAdd(counts+7,1ull);
        if(i%3==2) {
            if(candidate.discard)failure|=8;
            else atomicAdd(counts+6,1ull);
        }
        // Supported exact/below-GPU-distance packets must be old OK/inactive
        // and discarded. Unsupported params/masks remain a full-eval path.
        if(i%3!=2&&inputs[i].d_hat>0.0&&base_full[i/3].status==gpu::ContactStatus::Ok
            &&inputs[i].k_barrier>=0.0&&inputs[i].k_barrier<=DBL_MAX
            &&inputs[i].friction==0.0&&(inputs[i].hessian_blocks&~0xffffu)==0) {
            if(!candidate.discard||old.status!=gpu::ContactStatus::Ok||old.active!=0)failure|=16;
            else if(i%3==1) {
                atomicAdd(counts+5,1ull);
                atomicAdd(counts+(inputs[i].kind==gpu::ContactKind::PointTriangle?8:9),1ull);
            }
        }
    }
    if(failure)atomicMin(first_bad,(static_cast<unsigned long long>(index_offset+i)<<8)|failure);
}
void cuda_checked(cudaError_t e,const char* stage) {
    if(e!=cudaSuccess)throw std::runtime_error(std::string(stage)+": "+cudaGetErrorString(e));
}
template<class T>struct DeviceBuffer {
    T* p=nullptr;std::size_t n;
    explicit DeviceBuffer(std::size_t count):n(count) {
        require(n&&n<=std::numeric_limits<std::size_t>::max()/sizeof(T),"oracle allocation overflow");
        cuda_checked(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"oracle allocation");
    }
    ~DeviceBuffer(){if(p)cudaFree(p);}
    DeviceBuffer(const DeviceBuffer&)=delete;DeviceBuffer& operator=(const DeviceBuffer&)=delete;
};
std::vector<gpu::ContactInput> device_inputs() {
    auto result=geometry();const auto seeds=result;
    const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
    for(const auto& original:seeds) {
        for(double dhat:{.001,.1,1.,10.})for(double k:{0.,-0.,1.,DBL_MAX})
            for(unsigned mask:{0u,1u,1u<<5,0xffffu,0x10000u}) {
                auto in=original;in.d_hat=dhat;in.k_barrier=k;in.hessian_blocks=mask;result.push_back(in);
            }
        for(unsigned field=0;field<7;++field)for(double value:{-1.,-0.,0.,nan,inf,-inf,
            std::numeric_limits<double>::denorm_min(),DBL_MAX}) {
            auto in=original;
            if(field==0)in.dt=value;else if(field==1)in.eps=value;else if(field==2)in.eps_v=value;
            else if(field==3)in.d_hat=value;else if(field==4)in.k_barrier=value;
            else if(field==5)in.friction=value;else in.kind=static_cast<gpu::ContactKind>(2);
            result.push_back(in);
        }
        for(bool previous:{false,true})for(unsigned role=0;role<4;++role)
            for(unsigned axis=0;axis<3;++axis)for(double value:{nan,inf,-inf}) {
                auto in=original;(previous?in.previous:in.positions)[role][axis]=value;result.push_back(in);
            }
        for(const auto& dt:std::vector<std::pair<double,double>>{{1e200,1e-200},{1e-200,1e-200},{1e200,1e200}}) {
            auto in=original;in.dt=dt.first;in.eps_v=dt.second;result.push_back(in);
        }
    }
    std::mt19937_64 random(0x8921617deULL);
    for(unsigned sample=0;sample<256;++sample)for(int exponent:{-500,-300,-100,-30,0,30,100,300,500,900}) {
        std::array<Point,4> points{};const double scale=std::ldexp(1.,exponent);
        for(auto& point:points)for(auto& value:point)value=(int(random()%2049)-1024)/512.*scale;
        auto in=make(sample%2?gpu::ContactKind::PointTriangle:gpu::ContactKind::EdgeEdge,points);
        in.hessian_blocks=sample%3?0xffffu:1u;result.push_back(in);
    }
    return result;
}
}
int main(){try {
    const auto inputs=device_inputs();const auto n=inputs.size();
    require(n<=UINT_MAX/3,"device oracle grid/index bound");
    DeviceBuffer<gpu::ContactInput> base(n),derived(3*n);
    DeviceBuffer<gpu::ContactOutput> full(n),boundary_full(3*n);
    DeviceBuffer<ProbeRecord> probe(n),boundary_probe(3*n);
    DeviceBuffer<unsigned char> valid(3*n);
    DeviceBuffer<unsigned long long> counts(10),first_bad(1);
    cuda_checked(cudaMemcpy(base.p,inputs.data(),n*sizeof(inputs[0]),cudaMemcpyHostToDevice),"oracle inputs");
    cuda_checked(cudaMemset(counts.p,0,10*sizeof(unsigned long long)),"oracle counters");
    cuda_checked(cudaMemset(first_bad.p,0xff,sizeof(unsigned long long)),"oracle first failure");
    const auto grid=static_cast<unsigned>((n+127)/128),boundary_grid=static_cast<unsigned>((3*n+127)/128);
    sparse_oracle_full<<<grid,128>>>(base.p,n,full.p);
    sparse_oracle_probe<<<grid,128>>>(base.p,n,probe.p);
    sparse_oracle_compare<<<grid,128>>>(base.p,full.p,probe.p,n,nullptr,nullptr,counts.p,first_bad.p,0);
    sparse_oracle_boundaries<<<grid,128>>>(base.p,full.p,n,derived.p,valid.p);
    sparse_oracle_full<<<boundary_grid,128>>>(derived.p,3*n,boundary_full.p);
    sparse_oracle_probe<<<boundary_grid,128>>>(derived.p,3*n,boundary_probe.p);
    sparse_oracle_compare<<<boundary_grid,128>>>(derived.p,boundary_full.p,boundary_probe.p,3*n,
        valid.p,full.p,counts.p,first_bad.p,n);
    cuda_checked(cudaGetLastError(),"oracle launches");
    std::array<unsigned long long,10> report{};unsigned long long failure=0;
    cuda_checked(cudaMemcpy(report.data(),counts.p,sizeof(report),cudaMemcpyDeviceToHost),"oracle integer report");
    cuda_checked(cudaMemcpy(&failure,first_bad.p,sizeof(failure),cudaMemcpyDeviceToHost),"oracle integer failure");
    if(failure!=~0ull)throw std::runtime_error("GPU boundary/discard oracle failed at index="
        +std::to_string(failure>>8)+" reason_bits="+std::to_string(failure&255));
    require(report[0]==4*n&&report[1]&&report[2]&&report[3]&&report[4]&&report[5]&&report[6]&&report[7]&&report[8]&&report[9],
        "GPU oracle lacks required discard/error/geometry/exact-boundary witnesses");
    std::cout<<"SPARSE_FORCE_DISCARD_GPU_ORACLE_PASS base="<<n<<" tested="<<report[0]
        <<" discardPT="<<report[1]<<" discardEE="<<report[2]<<" oldErrorsKept="<<report[3]
        <<" geometryBitChecks="<<report[4]<<" GPUexactDhatDiscard="<<report[5]
        <<" GPUexactDhatDiscardPT="<<report[8]<<" GPUexactDhatDiscardEE="<<report[9]
        <<" GPUaboveDhatKept="<<report[6]<<" GPUderivedBoundary="<<report[7]
        <<" fullPublicationBytes=1416 separateFullProbeKernels=1; pipeline/trajectory gates separate\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
