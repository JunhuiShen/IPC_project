// Standalone old/new exact-ratio differential test. This reference deliberately
// retains the pre-optimization implementation; production never calls it.
// Host: c++ -std=c++17 -O2 -ffp-contract=off -I gpu \
//       gpu/resident_ratio_differential_test.cpp -o /tmp/resident_ratio_diff
// CUDA: nvcc -std=c++17 -O2 --fmad=false -I gpu \
//       gpu/resident_ratio_differential_test.cu -o /tmp/resident_ratio_diff_cuda
#include "resident_exact.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define RATIO_TEST_HD __host__ __device__
#else
#define RATIO_TEST_HD
#endif

namespace exact=ipc_gpu::resident_exact;
namespace ratio_differential {

template<int C> RATIO_TEST_HD double original_ratio_to_double(
    const exact::Integer<C>& numerator,const exact::Integer<C>& denominator,
    int scale_exponent,bool* arithmetic_ok) {
    using namespace exact;
    if(!numerator.valid||!denominator.valid||!denominator.sign) {
        if(arithmetic_ok) *arithmetic_ok=false; return NAN;
    }
    if(!numerator.sign) return 0.0;
    const int sign=numerator.sign*denominator.sign;
    Integer<C> n=numerator,d=denominator; n.sign=1; d.sign=1;
    int exponent=bit_count(n)-bit_count(d);
    Integer<C> shifted=exponent>=0?shift_left(d,exponent):shift_left(n,-exponent);
    if(!shifted.valid) { if(arithmetic_ok) *arithmetic_ok=false; return NAN; }
    const int order=exponent>=0?compare_magnitude(n,shifted):compare_magnitude(shifted,d);
    if(order<0) --exponent;
    const long long final_exponent=static_cast<long long>(exponent)+scale_exponent;
    if(final_exponent>1023) return sign>0?INFINITY:-INFINITY;
    if(final_exponent < -1075) return sign>0?0.0:-0.0;
    const int output_exponent=final_exponent>=-1022?int(final_exponent)-52:-1074;
    const long long shift=static_cast<long long>(scale_exponent)-output_exponent;
    if(shift>32LL*C || shift < -32LL*C) {
        if(arithmetic_ok) *arithmetic_ok=false; return NAN;
    }
    if(shift>=0) n=shift_left(n,int(shift)); else d=shift_left(d,int(-shift));
    if(!n.valid||!d.valid) { if(arithmetic_ok) *arithmetic_ok=false; return NAN; }
    std::uint64_t quotient=0;
    const int quotient_bits=bit_count(n)-bit_count(d);
    if(quotient_bits>=64) { if(arithmetic_ok) *arithmetic_ok=false; return NAN; }
    for(int bit=quotient_bits;bit>=0;--bit) {
        const auto trial=shift_left(d,bit);
        if(!trial.valid) { if(arithmetic_ok) *arithmetic_ok=false; return NAN; }
        if(compare_magnitude(n,trial)>=0) {
            n=subtract(n,trial); quotient|=UINT64_C(1)<<bit;
        }
    }
    const auto twice=shift_left(n,1);
    if(!twice.valid) { if(arithmetic_ok) *arithmetic_ok=false; return NAN; }
    const int halfway=compare_magnitude(twice,d);
    if(halfway>0 || (halfway==0 && (quotient&1))) ++quotient;
    const double magnitude=::ldexp(static_cast<double>(quotient),output_exponent);
    return sign>0?magnitude:-magnitude;
}

constexpr int max_pool_slots=32;
constexpr unsigned invalid_numerator=1u,invalid_denominator=2u,null_ok=4u,
    initially_false=8u,invalid_pool=16u;
struct Case {
    std::size_t numerator_offset=0,denominator_offset=0;
    int numerator_words=0,denominator_words=0,numerator_sign=0,denominator_sign=0;
    int exponent=0,pool_slots=max_pool_slots,pool_width=0;
    unsigned flags=0;
};
struct Batch {std::vector<Case> cases;std::vector<std::uint32_t> limbs;};
struct Result {
    std::uint64_t bits=0;
    int ok=0,pool_failed=0,available=0,peak=0;
};
RATIO_TEST_HD std::uint64_t bits_of(double value) {
#if defined(__CUDA_ARCH__)
    return static_cast<std::uint64_t>(__double_as_longlong(value));
#else
    std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));return bits;
#endif
}
template<int C> RATIO_TEST_HD void fill(exact::Integer<C>& value,
    const std::uint32_t* limbs,std::size_t offset,int words,int sign,bool invalid) {
    if(!value.valid)return;
    value.valid=!invalid;
    if(invalid)return;
    value.used=words;value.sign=sign;
    for(int i=0;i<words;++i)value.limb[i]=limbs[offset+std::size_t(i)];
    value.normalize();
}
template<int C,bool Optimized> RATIO_TEST_HD Result evaluate(
    const Case& input,const std::uint32_t* limbs) {
    exact::Integer<C> numerator,denominator;
    fill(numerator,limbs,input.numerator_offset,input.numerator_words,
         input.numerator_sign,bool(input.flags&invalid_numerator));
    fill(denominator,limbs,input.denominator_offset,input.denominator_words,
         input.denominator_sign,bool(input.flags&invalid_denominator));
    bool ok=!(input.flags&initially_false);
    bool* ok_pointer=(input.flags&null_ok)?nullptr:&ok;
    const double value=Optimized
        ?exact::ratio_to_double(numerator,denominator,input.exponent,ok_pointer)
        :original_ratio_to_double(numerator,denominator,input.exponent,ok_pointer);
    Result result;result.bits=bits_of(value);result.ok=ok;
    return result;
}

void append(Batch& batch,std::vector<std::uint32_t> numerator,
    std::vector<std::uint32_t> denominator,int ns,int ds,int exponent,
    int width,unsigned flags=0,int slots=max_pool_slots) {
    while(!numerator.empty()&&!numerator.back())numerator.pop_back();
    while(!denominator.empty()&&!denominator.back())denominator.pop_back();
    Case input;
    input.numerator_offset=batch.limbs.size();input.numerator_words=int(numerator.size());
    input.numerator_sign=numerator.empty()?0:ns;
    batch.limbs.insert(batch.limbs.end(),numerator.begin(),numerator.end());
    input.denominator_offset=batch.limbs.size();input.denominator_words=int(denominator.size());
    input.denominator_sign=denominator.empty()?0:ds;
    batch.limbs.insert(batch.limbs.end(),denominator.begin(),denominator.end());
    input.exponent=exponent;input.pool_width=width;input.flags=flags;input.pool_slots=slots;
    batch.cases.push_back(input);
}
std::vector<std::uint32_t> power(int bit,bool below=false) {
    std::vector<std::uint32_t> value(std::size_t(bit/32)+1,below?UINT32_MAX:0);
    value.back()=below?(bit%32?(UINT32_C(1)<<(bit%32))-1:0):UINT32_C(1)<<(bit%32);
    return value;
}
template<int C> Batch make_cases() {
    Batch batch;
    // Signed zero, negative signs, exact ties, just-off ties, subnormal/normal
    // boundaries, infinity, and exponent underflow. Compare all output bits.
    for(int exponent:{-1200,-1076,-1075,-1074,-1023,-1022,-54,-53,0,970,1023,1024})
        for(std::uint32_t n:{0u,1u,2u,3u,5u,7u,UINT32_MAX})
            for(std::uint32_t d:{1u,2u,3u,4u,7u})for(int ns:{-1,1})for(int ds:{-1,1})
                append(batch,{n},{d},ns,ds,exponent,C);
    // Mantissa ties with either quotient parity, the transition to infinity,
    // and limb carries at exact 32-bit boundaries.
    for(int bit:{31,32,52,53,54,63,32*C-2,32*C-1})if(bit>=0&&bit<32*C) {
        auto p=power(bit);auto before=power(bit,true);auto after=p;++after[0];
        for(const auto& value:{before,p,after})for(int exponent:{-1074,-bit-1,-bit,0,1023-bit}) {
            append(batch,value,{1},1,1,exponent,C);
            append(batch,{1},value,-1,1,exponent,C);
            append(batch,value,value,1,-1,exponent,C);
        }
    }
    // The exact halfway point between maximum finite binary64 and infinity,
    // plus its immediate integer neighbors before the common 2^970 scaling.
    auto overflow_half=power(54,true);
    auto below_overflow_half=overflow_half;--below_overflow_half[0];
    auto above_overflow_half=power(54);
    for(const auto& value:{below_overflow_half,overflow_half,above_overflow_half})
        for(int sign:{-1,1})append(batch,value,{1},sign,1,970,C);
    // Every capacity limit is exercised from both sides, including cases in
    // which the initial exponent shift fits but later significand/rounding
    // shifts do not. Old and new must both preserve arithmetic_ok failures.
    for(int words:{1,2,std::max(1,C-2),C-1,C}) {
        std::vector<std::uint32_t> full(std::size_t(words),UINT32_MAX);
        auto sparse=power(32*words-1);
        for(const auto& a:{full,sparse})for(const auto& b:{full,sparse})
            for(int exponent:{-32*C-1,-32*C,-1075,-53,-1,0,1,53,1024,32*C,32*C+1})
                append(batch,a,b,1,1,exponent,C);
    }
    std::mt19937_64 random(UINT64_C(0x781947) + C);
    for(int i=0;i<1500;++i) {
        const int limit=i%4?std::min(C,12):C;
        std::vector<std::uint32_t> n(1+random()%limit),d(1+random()%limit);
        for(auto& limb:n)limb=std::uint32_t(random());
        for(auto& limb:d)limb=std::uint32_t(random());
        append(batch,n,d,random()&1?1:-1,random()&1?1:-1,
            int(random()%7001)-3500,C,i%7==0?initially_false:0);
    }
    // Invalid operands, null arithmetic_ok, and a preexisting false flag must
    // retain exactly the old return bits and flag behavior.
    for(unsigned flags:{0u,initially_false,null_ok,initially_false|null_ok,
        invalid_numerator,invalid_denominator,invalid_numerator|invalid_denominator}) {
        append(batch,{3},{7},1,-1,-1074,C,flags);
        append(batch,{0},{1},1,1,0,C,flags);
        append(batch,{1},{0},1,1,0,C,flags);
    }
    // GPU pool exhaustion, too-narrow installed storage, and failed install.
    // The host runs these same arithmetic cases without a device allocator.
    for(int slots=0;slots<=16;++slots)for(int width:{C-1,C})
        for(int exponent:{-1075,-1074,-53,0,1000})for(unsigned flags:{0u,initially_false,null_ok,invalid_pool}) {
            append(batch,{3,1},{7,1},-1,1,exponent,width,flags,slots);
            append(batch,{1},{1},1,1,exponent,width,flags,slots);
        }
    return batch;
}

void known_rounding_checks() {
    struct Known {std::uint32_t numerator;int sign,exponent;std::uint64_t bits;};
    for(const Known example:std::vector<Known>{
        {1,1,-1075,UINT64_C(0)}, {1,-1,-1075,UINT64_C(0x8000000000000000)},
        {3,1,-1075,UINT64_C(2)}, {5,1,-1075,UINT64_C(2)}, {7,1,-1075,UINT64_C(4)},
        {1,1,-1074,UINT64_C(1)}, {1,-1,-1074,UINT64_C(0x8000000000000001)},
        {1,1,-1022,UINT64_C(0x0010000000000000)},
        {1,1,0,UINT64_C(0x3ff0000000000000)},
        {1,1,1024,UINT64_C(0x7ff0000000000000)},
        {1,-1,1024,UINT64_C(0xfff0000000000000)}}) {
        Batch batch;append(batch,{example.numerator},{1},example.sign,1,example.exponent,64);
        const auto old_result=evaluate<64,false>(batch.cases[0],batch.limbs.data());
        const auto new_result=evaluate<64,true>(batch.cases[0],batch.limbs.data());
        if(old_result.bits!=example.bits||new_result.bits!=example.bits||!old_result.ok||!new_result.ok)
            throw std::runtime_error("Known signed-zero/subnormal/tie/overflow result is incorrect");
    }
}

void require_equal(const Result& original,const Result& optimized,const Case& input,
                   int capacity,std::size_t index,bool device) {
    if(original.bits==optimized.bits&&original.ok==optimized.ok
        &&(!device||(original.pool_failed==optimized.pool_failed
            &&original.available==optimized.available&&original.peak==optimized.peak)))return;
    std::ostringstream message;
    message<<(device?"CUDA":"host")<<" ratio mismatch C="<<capacity<<" case="<<index
        <<" exponent="<<input.exponent<<" nwords="<<input.numerator_words
        <<" dwords="<<input.denominator_words<<" flags="<<input.flags
        <<" slots="<<input.pool_slots<<" pool_width="<<input.pool_width<<std::hex
        <<" bits old="<<original.bits<<" new="<<optimized.bits<<std::dec
        <<" ok="<<original.ok<<'/'<<optimized.ok
        <<" pool_failed="<<original.pool_failed<<'/'<<optimized.pool_failed
        <<" available="<<original.available<<'/'<<optimized.available
        <<" peak="<<original.peak<<'/'<<optimized.peak;
    throw std::runtime_error(message.str());
}

#if defined(__CUDACC__)
void cuda_check(cudaError_t error) {
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct Device {
    T* pointer=nullptr;std::size_t count=0;
    explicit Device(std::size_t n):count(n) {
        if(n)cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer),n*sizeof(T)));
    }
    explicit Device(const std::vector<T>& values):Device(values.size()) {
        if(count)cuda_check(cudaMemcpy(pointer,values.data(),count*sizeof(T),cudaMemcpyHostToDevice));
    }
    ~Device(){cudaFree(pointer);}
    std::vector<T> read()const {
        std::vector<T> values(count);
        if(count)cuda_check(cudaMemcpy(values.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost));
        return values;
    }
};
template<int C,bool Optimized> __device__ Result pooled_evaluate(
    const Case& input,const std::uint32_t* limbs,std::uint32_t* words,int* slots) {
    exact::DevicePool pool;
    exact::initialize_pool(pool,input.flags&invalid_pool?nullptr:words,slots,
                           input.pool_width,input.pool_slots);
    Result result=evaluate<C,Optimized>(input,limbs);
    result.pool_failed=pool.failed;result.available=pool.available;result.peak=pool.peak;
    exact::release_pool();
    return result;
}
template<int C> __global__ void compare_kernel(const Case* cases,const std::uint32_t* limbs,
    std::size_t count,std::uint32_t* scratch,int* slots,Result* original,Result* optimized) {
    const unsigned worker=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned workers=gridDim.x*blockDim.x;
    auto* words=scratch+std::size_t(worker)*C*max_pool_slots;
    auto* free_slots=slots+std::size_t(worker)*max_pool_slots;
    for(std::size_t i=worker;i<count;i+=workers) {
        original[i]=pooled_evaluate<C,false>(cases[i],limbs,words,free_slots);
        optimized[i]=pooled_evaluate<C,true>(cases[i],limbs,words,free_slots);
    }
}
template<int C> void gpu_check(const Batch& batch) {
    constexpr unsigned workers=64;
    Device<Case> cases(batch.cases);Device<std::uint32_t> limbs(batch.limbs);
    Device<std::uint32_t> scratch(std::size_t(workers)*C*max_pool_slots);
    Device<int> slots(std::size_t(workers)*max_pool_slots);
    Device<Result> old_result(batch.cases.size()),new_result(batch.cases.size());
    compare_kernel<C><<<workers/8,8>>>(cases.pointer,limbs.pointer,batch.cases.size(),
        scratch.pointer,slots.pointer,old_result.pointer,new_result.pointer);
    cuda_check(cudaGetLastError());cuda_check(cudaDeviceSynchronize());
    const auto original=old_result.read(),optimized=new_result.read();
    for(std::size_t i=0;i<batch.cases.size();++i) {
        require_equal(original[i],optimized[i],batch.cases[i],C,i,true);
        if(batch.cases[i].pool_slots==max_pool_slots&&batch.cases[i].pool_width==C
            &&!(batch.cases[i].flags&invalid_pool)) {
            const auto host=evaluate<C,false>(batch.cases[i],batch.limbs.data());
            require_equal(host,original[i],batch.cases[i],C,i,false);
            if(original[i].available!=max_pool_slots)
                throw std::runtime_error("CUDA exact ratio leaked an arena slot");
        }
    }
}
#endif

template<int C> std::size_t run() {
    const auto batch=make_cases<C>();
    for(std::size_t i=0;i<batch.cases.size();++i) {
        const auto original=evaluate<C,false>(batch.cases[i],batch.limbs.data());
        const auto optimized=evaluate<C,true>(batch.cases[i],batch.limbs.data());
        require_equal(original,optimized,batch.cases[i],C,i,false);
    }
#if defined(__CUDACC__)
    gpu_check<C>(batch);
    std::cout<<"Host/CUDA";
#else
    std::cout<<"Host";
#endif
    std::cout<<" exact ratio C="<<C<<": "<<batch.cases.size()<<" cases PASS\n";
    return batch.cases.size();
}
} // namespace ratio_differential

int main() {
    try {
        using namespace ratio_differential;
        known_rounding_checks();
        const auto total=run<2>()+run<4>()+run<8>()+run<64>()+run<128>()+run<256>()+run<1600>();
        std::cout<<"PASS: "<<total<<" old/new exact-ratio cases; output bits and arithmetic_ok match";
#if defined(__CUDACC__)
        std::cout<<"; device allocator failures, peaks, and restored slot counts match";
#else
        std::cout<<" (host arithmetic only; CUDA allocator tests require the .cu target)";
#endif
        std::cout<<'\n';return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
#undef RATIO_TEST_HD
