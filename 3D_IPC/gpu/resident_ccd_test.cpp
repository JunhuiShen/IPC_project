#include "resident_ccd.cuh"
#include "resident_exact.cuh"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#if defined(IPC_RESIDENT_PRODUCTION_REFERENCE)
#include "ccd.h"
#endif

#if defined(IPC_RESIDENT_CUDA_TEST)
#include "resident_ccd_cuda_test.h"
#endif

namespace cert=ipc_gpu::resident_ccd;
namespace exact=ipc_gpu::resident_exact;
using Big=boost::multiprecision::cpp_int;
using Rat=boost::multiprecision::cpp_rational;
using P=std::array<Rat,3>;
using Ps=std::array<P,4>;

void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
double round_reference(const Rat& value) {
    // Boost's rational convert_to<double> can double-round at subnormal
    // boundaries. Verify its candidate against exact rational midpoints.
    double best=value.convert_to<double>();
    if(std::isinf(best)) {
        const double largest=std::numeric_limits<double>::max();
        const Rat threshold=Rat(largest)+(Rat(largest)-Rat(std::nextafter(largest,0.0)))/2;
        if(abs(value)>=threshold) return best;
        best=value<0?-largest:largest;
    }
    for(int step=0;step<4;++step) {
        double next_best=best; Rat error=abs(value-Rat(best));
        for(double candidate:{std::nextafter(best,-INFINITY),std::nextafter(best,INFINITY)}) {
            if(!std::isfinite(candidate)) continue;
            const Rat candidate_error=abs(value-Rat(candidate));
            std::uint64_t bits;std::memcpy(&bits,&candidate,sizeof(bits));
            if(candidate_error<error||(candidate_error==error&&!(bits&1))) {
                next_best=candidate;error=candidate_error;
            }
        }
        if(next_best==best) return next_best;
        best=next_best;
    }
    return best;
}
template<int C> Big reference(const exact::Integer<C>& a) {
    require(a.valid,"invalid Integer in reference conversion"); Big r=0;
    for(int i=a.used-1;i>=0;--i) { r<<=32; r+=a.limb[i]; }
    return a.sign<0?-r:r;
}
template<int C> exact::Integer<C> random_integer(std::mt19937_64& rng,int words) {
    exact::Integer<C> r; r.used=words; r.sign=(rng()&1)?1:-1;
    for(int i=0;i<words;++i) r.limb[i]=std::uint32_t(rng());
    r.normalize(); return r;
}

P sub(const P&a,const P&b) { return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}}; }
Rat dot(const P&a,const P&b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
P cross(const P&a,const P&b) { return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}}; }
Rat point_segment(const P&p,const P&a,const P&b) {
    const auto e=sub(b,a),d=sub(p,a); const Rat l=dot(e,e),t=dot(d,e);
    if(l==0||t<=0) return dot(d,d);
    if(t>=l) { const auto v=sub(p,b); return dot(v,v); }
    return dot(d,d)-t*t/l;
}
Rat pt(const Ps&p) {
    const auto a=sub(p[2],p[1]),b=sub(p[3],p[1]),d=sub(p[0],p[1]),n=cross(a,b);
    const Rat area=dot(n,n);
    if(area!=0) {
        const Rat aa=dot(a,a),ab=dot(a,b),bb=dot(b,b),da=dot(d,a),db=dot(d,b);
        const Rat u=bb*da-ab*db,v=aa*db-ab*da;
        if(u>=0&&v>=0&&u+v<=area) { const Rat h=dot(d,n); return h*h/area; }
    }
    return std::min({point_segment(p[0],p[1],p[2]),point_segment(p[0],p[2],p[3]),point_segment(p[0],p[3],p[1])});
}
Rat ee(const Ps&p) {
    const auto a=sub(p[1],p[0]),b=sub(p[3],p[2]),d=sub(p[0],p[2]);
    const Rat aa=dot(a,a),ab=dot(a,b),bb=dot(b,b),denom=aa*bb-ab*ab;
    if(denom!=0) {
        const Rat ar=dot(a,d),br=dot(b,d),s=ab*br-bb*ar,t=aa*br-ab*ar;
        if(s>=0&&s<=denom&&t>=0&&t<=denom) { const Rat h=dot(d,cross(a,b)); return h*h/denom; }
    }
    return std::min({point_segment(p[0],p[2],p[3]),point_segment(p[1],p[2],p[3]),point_segment(p[2],p[0],p[1]),point_segment(p[3],p[0],p[1])});
}
Rat distance(const Ps&p,bool face) { return face?pt(p):ee(p); }
Rat determinant(const Ps&p) { return dot(cross(sub(p[1],p[0]),sub(p[2],p[0])),sub(p[3],p[0])); }
exact::Status reference_step(const cert::LinearQuery&q) {
    Ps x,y,dx; bool changed=false;
    for(int i=0;i<4;++i) for(int k=0;k<3;++k) {
        x[i][k]=Rat(q.start[i][k]); y[i][k]=Rat(q.end[i][k]); dx[i][k]=y[i][k]-x[i][k]; changed=changed||dx[i][k]!=0;
    }
    const Rat start=distance(x,q.vertex_face),end=distance(y,q.vertex_face),floor=Rat(q.gap_floor)*Rat(q.gap_floor);
    if(start==0) return exact::Status::InitialContact;
    if(!changed) return exact::Status::Safe;
    if(end<std::min(start,floor)) return exact::Status::Unsafe;
    const Rat validation=Rat(1e-10)*Rat(1e-10),threshold=start<=validation?Rat(0):validation;
    const auto hit=[&](const Rat&t) {
        Ps p;
        for(int i=0;i<4;++i) for(int k=0;k<3;++k) p[i][k]=x[i][k]+t*dx[i][k];
        return distance(p,q.vertex_face)<=threshold;
    };
    const Rat intercept=determinant(x),slope=determinant(y)-intercept;
    if(slope!=0) { const Rat t=-intercept/slope; return t>0&&t<=1&&hit(t)?exact::Status::Unsafe:exact::Status::Safe; }
    if(intercept!=0) return exact::Status::Safe;
    std::vector<Rat> roots{Rat(1)};
    const auto root=[&](const Rat&v,const Rat&dv) { if(dv!=0) { const Rat t=-v/dv; if(t>0&&t<=1) roots.push_back(t); } };
    for(int i=0;i<4;++i) for(int j=i+1;j<4;++j) {
        const auto e=sub(x[j],x[i]),v=sub(dx[j],dx[i]);
        for(int a=0;a<3;++a) root(e[a],v[a]);
        for(int k=j+1;k<4;++k) {
            const auto o=sub(x[k],x[i]),ov=sub(dx[k],dx[i]),value=cross(e,o),first=cross(v,o),second=cross(e,ov);
            for(int a=0;a<3;++a) root(value[a],first[a]+second[a]);
        }
    }
    std::sort(roots.begin(),roots.end()); roots.erase(std::unique(roots.begin(),roots.end()),roots.end());
    for(const auto&t:roots) if(hit(t)) return exact::Status::Unsafe;
    return exact::Status::Safe;
}

cert::LinearQuery triangle_query(double gap,double end_gap) {
    cert::LinearQuery q;
    const double p[4][3]={{.25,.25,gap},{0,0,0},{1,0,0},{0,1,0}};
    std::memcpy(q.start,p,sizeof(p)); std::memcpy(q.end,p,sizeof(p)); q.end[0][2]=end_gap;
    return q;
}
std::vector<cert::LinearQuery> fixtures() {
    std::vector<cert::LinearQuery> result;
    for(auto pair:{std::pair<double,double>{1.,.5},{1.,-1.},{1e-12,2e-12},{1e-12,.5e-12},{0.,1.},{1.,0.}})
        result.push_back(triangle_query(pair.first,pair.second));
    auto q=triangle_query(0.,0.); q.start[0][0]=-1.;q.end[0][0]=2.; result.push_back(q); // coplanar through triangle
    q.end[0][0]=-2.;result.push_back(q); // coplanar receding
    q=triangle_query(1e-12,1e-12); q.end[0][0]=.4; result.push_back(q); // tiny tangential
    q=triangle_query(1.,1.); q.start[3][0]=q.end[3][0]=1.;q.start[3][1]=q.end[3][1]=0.; result.push_back(q); // collapsed triangle
    for(int exponent:{-1000,-400,400,900}) {
        q=triangle_query(1.,.5);
        for(int i=0;i<4;++i) for(int k=0;k<3;++k) { q.start[i][k]=std::ldexp(q.start[i][k],exponent);q.end[i][k]=std::ldexp(q.end[i][k],exponent); }
        q.gap_floor=std::ldexp(1e-8,exponent); result.push_back(q);
    }
    q=triangle_query(1.,.5);q.start[1][0]=q.end[1][0]=-std::numeric_limits<double>::max();
    q.start[2][0]=q.end[2][0]=std::numeric_limits<double>::max();
    q.start[3][1]=q.end[3][1]=std::numeric_limits<double>::denorm_min();result.push_back(q);
    std::mt19937_64 rng(107);
    std::uniform_real_distribution<double> random(-2,2);
    for(int n=0;n<700;++n) {
        q={};q.vertex_face=n%2==0;
        for(int i=0;i<4;++i) for(int k=0;k<3;++k) q.start[i][k]=q.end[i][k]=(n%3==0&&k==2)?0.:random(rng);
        const int moved=n%4; for(int k=0;k<3;++k) q.end[moved][k]=(n%3==0&&k==2)?0.:random(rng);
        if(n%11==0) for(int k=0;k<3;++k) { q.start[3][k]=q.start[2][k]; q.end[3][k]=q.start[3][k]; if(moved==2) q.end[2][k]=q.start[2][k]; }
        result.push_back(q);
    }
    for(int n=0;n<48;++n) {
        q={};q.vertex_face=n%2==0;
        for(int i=0;i<4;++i) for(int k=0;k<3;++k) {
            // Finite binary64 bit patterns cover the complete exponent range,
            // not only ordinary scene-scale coordinates.
            std::uint64_t bits=rng(); bits&=~(UINT64_C(0x7ff)<<52);
            bits|=(rng()%2047)<<52;
            std::memcpy(&q.start[i][k],&bits,sizeof(bits));q.end[i][k]=q.start[i][k];
        }
        const int moved=n%4;
        for(int k=0;k<3;++k) q.end[moved][k]=std::ldexp(random(rng),int(rng()%2000)-1000);
        result.push_back(q);
    }
    return result;
}

void arithmetic_tests() {
    std::mt19937_64 rng(51123);
    for(int sample=0;sample<3000;++sample) {
        const auto a=random_integer<64>(rng,1+int(rng()%12)),b=random_integer<64>(rng,1+int(rng()%12));
        const Big ar=reference(a),br=reference(b);
        require(reference(exact::add(a,b))==ar+br,"integer add");
        require(reference(exact::subtract(a,b))==ar-br,"integer subtract");
        require(reference(exact::multiply(a,b))==ar*br,"integer multiply");
        const int shift=int(rng()%300);
        require(reference(exact::shift_left(a,shift))==(ar<<shift),"integer shift");
        bool ok=true;const int exponent=int(rng()%2300)-1150;
        const double got=exact::ratio_to_double(a,b,exponent,&ok);
        Rat expected=Rat(ar)/Rat(br);
        if(exponent>=0) expected*=Big(1)<<exponent; else expected/=Big(1)<<-exponent;
        const double want=round_reference(expected);
        require(ok,"ratio capacity");
        if(got!=want && !(std::isnan(got)&&std::isnan(want))) {
            std::cerr<<"ratio sample "<<sample<<" got="<<std::hexfloat<<got<<" want="<<want<<std::defaultfloat<<" exponent="<<exponent<<'\n';
            throw std::runtime_error("ratio rounding");
        }
    }
    // Exactly-halfway conversions at normal/subnormal boundaries.
    using I=exact::Integer<64>;
    for(int shift:{-1075,-1074,-1023,-54,-53,0,970}) for(int numerator:{1,3,5,7}) {
        bool ok=true; Rat value=Rat(numerator); if(shift<0)value/=Big(1)<<-shift;else value*=Big(1)<<shift;
        require(exact::ratio_to_double(I(numerator),I(1),shift,&ok)==round_reference(value),"ratio halfway");
        require(ok,"ratio halfway capacity");
    }
    auto huge=exact::shift_left(exact::Integer<2>(1),63);
    require(!exact::add(huge,huge).valid,"add overflow must be explicit");
    require(!exact::multiply(huge,huge).valid,"multiply overflow must be explicit");
    require(!exact::shift_left(huge,1).valid,"shift overflow must be explicit");
}

#if defined(IPC_RESIDENT_CUDA_TEST)
void gpu_tests(const std::vector<cert::LinearQuery>& queries,const std::vector<exact::Status>& expected) {
    std::vector<cert::Status> certs(queries.size());std::vector<exact::Status> exacts(queries.size());
    resident_ccd_cuda_validate(queries.data(),queries.size(),certs.data(),exacts.data());
    for(std::size_t i=0;i<queries.size();++i) {
        if(certs[i]==cert::Status::Safe) require(expected[i]==exact::Status::Safe,"GPU unsafe interval acceptance");
        require(exacts[i]!=exact::Status::CapacityExceeded,"GPU full-range retry capacity exhausted");
        require(exacts[i]==expected[i],"GPU exact predicate mismatch");
    }
    std::cout<<"CUDA exact/certificate checks: "<<queries.size()<<"; full-range retries completed on device\n";
}
#endif

int main() {
    try {
        arithmetic_tests();const auto queries=fixtures();std::vector<exact::Status> expected;
        int safe_certificates=0,retries=0;
        for(std::size_t i=0;i<queries.size();++i) {
            const auto&q=queries[i]; const auto want=reference_step(q);expected.push_back(want);
#if defined(IPC_RESIDENT_PRODUCTION_REFERENCE)
            std::array<Vec3,4> start,end;
            for(int v=0;v<4;++v) for(int k=0;k<3;++k) { start[v][k]=q.start[v][k];end[v][k]=q.end[v][k]; }
            const auto production=exact_contact_step_result(start,end,q.vertex_face,q.gap_floor);
            const auto mapped=production==ExactContactStepResult::Safe?exact::Status::Safe
                :production==ExactContactStepResult::Unsafe?exact::Status::Unsafe:exact::Status::InitialContact;
            require(mapped==want,"CPU production versus independent rational reference mismatch");
#endif
            auto got=exact::validate_step<128>(q.start,q.end,q.vertex_face,q.gap_floor);
            if(got==exact::Status::CapacityExceeded) { ++retries; got=exact::validate_step<2048>(q.start,q.end,q.vertex_face,q.gap_floor); }
            if(got!=want) { std::cerr<<"fixture "<<i<<" got="<<int(got)<<" expected="<<int(want)<<'\n';throw std::runtime_error("exact step mismatch"); }
            if(cert::validate_linear(q)==cert::Status::Safe) { ++safe_certificates;require(want==exact::Status::Safe,"unsafe interval acceptance"); }
        }
        const auto crossing=triangle_query(1.,-1.);const auto clipped=cert::clip_linear(crossing);
        require(clipped.status==cert::Status::Safe&&!clipped.full_step&&clipped.fraction>0.&&clipped.fraction<.5,"conservative crossing clip");
        const auto exact_clip=exact::clip_step<128>(crossing.start,crossing.end,true,1e-8);
        require(exact_clip.status==exact::Status::Safe&&!exact_clip.full_step,"exact crossing clip");
        const auto receding=triangle_query(1e-12,2e-12);
        const auto exact_recede=exact::clip_step<128>(receding.start,receding.end,true,1e-8);
        require(exact_recede.status==exact::Status::Safe&&exact_recede.full_step,"tiny-gap receding must not lock");
        const auto extreme=triangle_query(1e-12,-1e200);
        const auto exact_stationary=exact::clip_step<2048>(extreme.start,extreme.end,true,1e-8);
        require(exact_stationary.status==exact::Status::Safe&&exact_stationary.fraction==0.
            &&exact_stationary.endpoint[0][2]==extreme.start[0][2],"64 attempts require certified stationary fallback");
        // A full path certificate is not an endpoint-floor certificate for
        // every prefix. This is why per-contact caches must be invalidated
        // when another pair forces the shared node to backtrack.
        cert::LinearQuery cache_query;
        const double cache_start[4][3]={{-1,0,1e-9},{-.1,-1,0},{.1,-1,0},{0,1,0}};
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)
            cache_query.start[i][k]=cache_query.end[i][k]=cache_start[i][k];
        cache_query.end[0][0]=1.;
        require(cert::validate_linear(cache_query)==cert::Status::Safe,"full cache fixture should be certified");
        cache_query.end[0][0]=0.;
        require(cert::validate_linear(cache_query)!=cert::Status::Safe
            &&exact::validate_step<128>(cache_query.start,cache_query.end,true,1e-8)==exact::Status::Unsafe,
            "half-prefix must not reuse a full-step endpoint certificate");
        cache_query.end[0][0]=-.5;
        require(exact::validate_step<128>(cache_query.start,cache_query.end,true,1e-8)==exact::Status::Safe,
            "quarter-prefix cache fixture should remain safe");
        auto invalid=triangle_query(1.,.5);invalid.end[0][0]=INFINITY;
        require(cert::validate_linear(invalid)==cert::Status::Invalid,"nonfinite interval input");
        require(exact::validate_step<128>(invalid.start,invalid.end,true,1e-8)==exact::Status::Invalid,"nonfinite exact input");
#if defined(IPC_RESIDENT_CUDA_TEST)
        gpu_tests(queries,expected);
#endif
        std::cout<<"Passed 3,000 integer/ratio cases and "<<queries.size()<<" exact CCD cases; interval certificates="<<safe_certificates<<"; wide retries="<<retries<<'\n';
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
