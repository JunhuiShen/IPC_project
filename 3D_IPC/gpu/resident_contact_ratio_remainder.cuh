#pragma once
#include "resident_exact.cuh"
#if defined(__CUDACC__)
#define IPC_CONTACT_RATIO_HD __host__ __device__
#else
#define IPC_CONTACT_RATIO_HD
#endif
namespace ipc_gpu::resident::contact_ratio_detail {
// Contact-only specialization. Exact predicates, coordinate/feature selection,
// limb capacities and rounding guards remain in the original implementations.
// The only changed computation is positive remainder subtraction after the
// original n>=trial comparison. Insufficient arena headroom uses the original.
template<int C> IPC_CONTACT_RATIO_HD inline double ratio_to_double(
    const resident_exact::Integer<C>& numerator,const resident_exact::Integer<C>& denominator,int scale_exponent,
    bool* arithmetic_ok) {
    using namespace resident_exact;
#if defined(__CUDA_ARCH__)
    // The original conversion needs fewer than16 additional live Integer
    // slots, even with non-elided return copies. Preserve its failure path
    // verbatim when the installed arena could make allocation observable.
    DevicePool* pool=current_pool();
    if(!pool||pool->failed||pool->words_per_slot<C||pool->available<16)
        return resident_exact::ratio_to_double(numerator,denominator,scale_exponent,arithmetic_ok);
#endif
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
    if(quotient_bits>=0) {
        // Materialize the largest trial once. Each subsequent trial is an
        // exact one-bit right shift: d*2^bit is divisible by two while bit>0.
        // The largest shift has the strongest limb-capacity requirement, and
        // one trial occupies the same arena slot count as the original loop.
        // Keep its lifetime inside this block, before constructing `twice`.
        auto trial=shift_left(d,quotient_bits);
        if(!trial.valid) { if(arithmetic_ok) *arithmetic_ok=false; return NAN; }
        for(int bit=quotient_bits;bit>=0;--bit) {
            if(compare_magnitude(n,trial)>=0) {
                // Both magnitudes are nonnegative and n>=trial was just
                // established. The exact remainder cannot grow in limb count.
                // Avoid negate(trial), an extra magnitude comparison, arena
                // temporaries, and copying the result back into n.
                std::uint64_t borrow=0;
                for(int word=0;word<n.used;++word) {
                    const std::uint64_t lhs=n.limb[word];
                    const std::uint64_t rhs=(word<trial.used?std::uint64_t(trial.limb[word]):0)+borrow;
                    n.limb[word]=std::uint32_t(lhs-rhs);borrow=lhs<rhs;
                }
                if(borrow){if(arithmetic_ok)*arithmetic_ok=false;return NAN;}
                n.normalize();quotient|=UINT64_C(1)<<bit;
            }
            if(bit>0) {
                std::uint32_t carry=0;
                for(int word=trial.used;word-- >0;) {
                    const std::uint32_t original=trial.limb[word];
                    trial.limb[word]=(original>>1)|carry;
                    carry=original<<31;
                }
                trial.normalize();
            }
        }
    }
    const auto twice=shift_left(n,1);
    if(!twice.valid) { if(arithmetic_ok) *arithmetic_ok=false; return NAN; }
    const int halfway=compare_magnitude(twice,d);
    if(halfway>0 || (halfway==0 && (quotient&1))) ++quotient;
    const double magnitude=::ldexp(static_cast<double>(quotient),output_exponent);
    return sign>0?magnitude:-magnitude;
}

} // namespace ipc_gpu::resident::contact_ratio_detail
#undef IPC_CONTACT_RATIO_HD
