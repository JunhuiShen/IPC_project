#include "resident_box_certificate_test.h"
#include "resident_contact_exact.cuh"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu::resident;
using Rational = boost::multiprecision::cpp_rational;
using BigInt = boost::multiprecision::cpp_int;

namespace {
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
Rational exact(double x) {
    require(std::isfinite(x), "nonfinite exact input");
    if (x == 0.0) return 0;
    int exponent;
    const double fraction = std::frexp(std::abs(x), &exponent);
    BigInt numerator = static_cast<std::uint64_t>(std::ldexp(fraction, 53)), denominator = 1;
    exponent -= 53;
    if (exponent >= 0) numerator <<= exponent; else denominator <<= -exponent;
    if (x < 0.0) numerator = -numerator;
    return Rational(numerator)/Rational(denominator);
}
struct Fixture {
    ResidentBoxCertificateInput input;
    int expected = -1; // Unknown may stay unknown; only explicit fixtures demand success/failure.
    bool sample = true;
};
using Positions = std::array<std::array<double,3>,4>;
Fixture make(bool face, const Positions& x, double radius, double d_hat, int expected = -1) {
    Fixture f{}; f.expected = expected; f.input.vertex_face = face; f.input.d_hat = d_hat;
    for (int i=0;i<4;++i) for (int k=0;k<3;++k) {
        f.input.positions[i][k] = x[i][k];
        f.input.boxes[i].lo[k] = x[i][k]-radius;
        f.input.boxes[i].hi[k] = x[i][k]+radius;
    }
    return f;
}
Fixture parallel(double gap, double radius, double d_hat, int expected = -1) {
    return make(false, {{{{0,0,0}},{{1,0,0}},{{0,gap,0}},{{1,gap,0}}}}, radius, d_hat, expected);
}
std::vector<Fixture> fixtures() {
    std::vector<Fixture> f;
    f.push_back(parallel(.25, .01, .1, 1));
    f.push_back(parallel(.12, .01, .1, 0));
    f.push_back(parallel(.1, 0, .1, 0));
    f.push_back(parallel(std::nextafter(.1, 0.), 0, .1, 0));
    f.push_back(parallel(std::nextafter(.1, INFINITY), 0, .1));
    f.push_back(parallel(0, .01, .1, 0));
    f.push_back(parallel(2e-8, 0, 0, 1));
    f.push_back(parallel(1e-8, 0, 0, 0));
    f.push_back(parallel(5e-9, 0, 1e-12, 0));
    f.push_back(parallel(8*std::numeric_limits<double>::denorm_min(), 0,
        std::numeric_limits<double>::denorm_min(), 0));
    // Current primitives are separated, but permitted movement can collide.
    f.push_back(parallel(.25, .2, .1, 0));
    f.push_back(make(true, {{{{.25,.25,.25}},{{0,0,0}},{{1,0,0}},{{0,1,0}}}}, .01, .1, 1));
    f.push_back(make(true, {{{{.25,.25,0}},{{0,0,0}},{{1,0,0}},{{0,1,0}}}}, .01, .1, 0));
    // AABBs overlap; a non-coordinate projection must certify separation.
    f.push_back(make(false, {{{{0,0,0}},{{1,1,0}},{{0,1,0}},{{1,2,0}}}}, .01, .2, 1));
    f.push_back(make(true, {{{{.8,.8,0}},{{0,0,0}},{{1,0,0}},{{0,1,0}}}}, .01, .1, 1));
    // Degenerate triangles/edges are allowed: convex primitive bounds remain valid.
    f.push_back(make(true, {{{{.5,.3,0}},{{0,0,0}},{{1,0,0}},{{2,0,0}}}}, .01, .1, 1));
    f.push_back(make(false, {{{{0,0,0}},{{0,0,0}},{{0,.3,0}},{{0,.3,0}}}}, .01, .1, 1));
    // Scale-dependent gap floor dominates d_hat, including at future positions.
    for (double gap : {128., 512.}) {
        auto large=parallel(gap, 0, .001, gap==128.?0:1);
        for(int i=0;i<4;++i)large.input.positions[i][0]=large.input.boxes[i].lo[0]=large.input.boxes[i].hi[0]=1e16;
        large.sample=false; f.push_back(large);
    }
    auto future_scale=parallel(1e-4, 0, 1e-6, 0);
    for(int i=0;i<4;++i)future_scale.input.boxes[i].hi[0]=1e16;
    future_scale.sample=false;f.push_back(future_scale);
    // Overflow cannot turn uncertainty into separation.
    auto huge=parallel(1,0,1,0);
    for(int i=0;i<4;++i)huge.input.positions[i][0]=huge.input.boxes[i].lo[0]=huge.input.boxes[i].hi[0]=i<2?-DBL_MAX:DBL_MAX;
    huge.sample=false; f.push_back(huge);
    for(int exponent : {-1000,-500,-50,50,500,900}) {
        auto scaled=parallel(.25,.01,.1);
        scaled.input.d_hat=std::ldexp(scaled.input.d_hat,exponent);
        for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
            scaled.input.positions[i][k]=std::ldexp(scaled.input.positions[i][k],exponent);
            scaled.input.boxes[i].lo[k]=std::ldexp(scaled.input.boxes[i].lo[k],exponent);
            scaled.input.boxes[i].hi[k]=std::ldexp(scaled.input.boxes[i].hi[k],exponent);
        }
        scaled.sample=false; f.push_back(scaled);
    }
    const Fixture good=parallel(.25,.01,.1,1);
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        for(double bad : {NAN, INFINITY, -INFINITY}) {
            auto a=good;a.expected=0;a.sample=false;a.input.positions[i][k]=bad;f.push_back(a);
            auto b=good;b.expected=0;b.sample=false;b.input.boxes[i].lo[k]=bad;f.push_back(b);
            auto c=good;c.expected=0;c.sample=false;c.input.boxes[i].hi[k]=bad;f.push_back(c);
        }
        auto outside=good;outside.expected=0;outside.sample=false;
        outside.input.positions[i][k]=std::nextafter(outside.input.boxes[i].hi[k],INFINITY);f.push_back(outside);
        auto reversed=good;reversed.expected=0;reversed.sample=false;
        reversed.input.boxes[i].lo[k]=std::nextafter(reversed.input.boxes[i].hi[k],INFINITY);f.push_back(reversed);
    }
    for(double bad : {double(NAN), double(INFINITY), -1., -DBL_MAX}) {
        auto a=good;a.expected=0;a.sample=false;a.input.d_hat=bad;f.push_back(a);
    }
    std::mt19937_64 random(0x624236d3ULL);
    for(int n=0;n<1200;++n) {
        Positions x{};
        for(auto& p:x)for(double& v:p)v=(int(random()%65)-32)/16.;
        const double radius=(random()%5)/128.;
        auto a=make(n%2,x,radius,(1+random()%16)/128.);
        if(n%9==0)for(int k=0;k<3;++k) {
            a.input.positions[1][k]=a.input.positions[0][k];
            a.input.boxes[1]=a.input.boxes[0];
        }
        f.push_back(a);
    }
    return f;
}

void prove_witness(const ResidentBoxCertificateInput& in, const MovementBoxCertificateWitness& w) {
    Rational lower[2],upper[2]; bool initialized[2]={false,false};
    Rational norm2=0; double scale=0;
    for(int k=0;k<3;++k)norm2+=exact(w.axis[k])*exact(w.axis[k]);
    require(norm2>0,"zero witness axis");
    const int split=in.vertex_face?1:2;
    for(int i=0;i<4;++i) {
        Rational lo=0,hi=0;
        for(int k=0;k<3;++k) {
            const Rational axis=exact(w.axis[k]);
            const Rational a=axis*exact(in.boxes[i].lo[k]),b=axis*exact(in.boxes[i].hi[k]);
            lo+=a<b?a:b;hi+=a>b?a:b;
            scale=std::max(scale,std::max(std::abs(in.boxes[i].lo[k]),std::abs(in.boxes[i].hi[k])));
        }
        const int side=i<split?0:1;
        if(!initialized[side]){lower[side]=lo;upper[side]=hi;initialized[side]=true;}
        else {if(lo<lower[side])lower[side]=lo;if(hi>upper[side])upper[side]=hi;}
    }
    const Rational gap1=lower[0]-upper[1],gap2=lower[1]-upper[0],gap=gap1>gap2?gap1:gap2;
    const Rational threshold=exact(w.required_gap);
    require(threshold>=exact(in.d_hat)&&threshold>=exact(1e-8)&&threshold>=exact(1e-10),"threshold weakened");
    require(threshold>=exact(128*DBL_EPSILON)*exact(scale),"future-coordinate gap floor underestimated");
    require(gap>0&&gap*gap>threshold*threshold*norm2,"false whole-box certificate (exact projection oracle)");
}

std::size_t sample_baseline(const ResidentBoxCertificateInput& input,std::uint64_t seed,std::size_t& exact_checks) {
    std::mt19937_64 random(seed);std::size_t comparisons=0;
    // Every moving-vertex role, endpoints at box corners and in their interior.
    for(int sample=0;sample<16;++sample) {
        ContactInput in;in.kind=input.vertex_face?ContactKind::PointTriangle:ContactKind::EdgeEdge;
        in.d_hat=std::max(input.d_hat,1e-12);in.hessian_blocks=1u<<(5*(sample%4));
        ipc_gpu::resident_ccd::LinearQuery q;q.vertex_face=input.vertex_face;
        double scale=0;
        for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
            const double t=sample<8?double((sample>>k)&1):double(random()%1025)/1024.;
            const auto& b=input.boxes[i];
            const double position=std::max(b.lo[k],std::min(b.hi[k],(1-t)*b.lo[k]+t*b.hi[k]));
            in.positions[i][k]=in.previous[i][k]=q.start[i][k]=q.end[i][k]=position;
            scale=std::max(scale,std::abs(position));
            if(i==sample%4)q.end[i][k]=t<.5?b.hi[k]:b.lo[k];
        }
        q.gap_floor=std::max(1e-8,128*DBL_EPSILON*scale);
        auto out=evaluate_contact(in);
        if(out.status==ContactStatus::NeedsExactGeometry||out.status==ContactStatus::ZeroDistance||out.status==ContactStatus::NonFiniteDerivative)
            out=evaluate_contact_exact<256>(in);
        require(out.status==ContactStatus::Ok&&!out.active,"certified box produced active/failing baseline contact");
        auto safe=ipc_gpu::resident_ccd::validate_linear(q);
        require(safe!=ipc_gpu::resident_ccd::Status::Invalid,"certified box produced invalid baseline CCD");
        // Independently exercise exact validation for every moving role even
        // when the ordinary interval implementation already accepts the step.
        if(sample<4||safe!=ipc_gpu::resident_ccd::Status::Safe) {
            require(ipc_gpu::resident_exact::validate_step<256>(q.start,q.end,q.vertex_face,q.gap_floor)==ipc_gpu::resident_exact::Status::Safe,
                "certified box produced unsafe/failing exact CCD");
            ++exact_checks;
        }
        ++comparisons;
    }
    return comparisons;
}
}

int main() {
    try {
        const auto cases=fixtures();
        std::vector<ResidentBoxCertificateInput> input;input.reserve(cases.size());
        for(const auto& f:cases)input.push_back(f.input);
        std::vector<ResidentBoxCertificateResult> output(input.size());
#ifdef IPC_RESIDENT_BOX_CERTIFICATE_CUDA_TEST
        resident_box_certificate_gpu(input.data(),output.data(),input.size());
#else
        for(std::size_t i=0;i<input.size();++i)output[i].certified=certify_movement_box_contact(
            input[i].boxes,input[i].positions,input[i].vertex_face,input[i].d_hat,&output[i].witness);
#endif
        std::size_t accepted=0,sampled=0,exact_checks=0;
        for(std::size_t i=0;i<cases.size();++i) {
            if(cases[i].expected>=0)require(output[i].certified==unsigned(cases[i].expected),"fixture expectation "+std::to_string(i));
            if(!output[i].certified)continue;
            prove_witness(input[i],output[i].witness);++accepted;
            if(cases[i].sample)sampled+=sample_baseline(input[i],0x28c0deULL+i,exact_checks);
        }
        require(accepted>500,"insufficient nonvacuous certificate coverage");
        require(!certify_movement_box_contact(nullptr,input[0].positions,true,.1),"null boxes accepted");
        require(!certify_movement_box_contact(input[0].boxes,nullptr,true,.1),"null positions accepted");
        std::cout<<"PASS: "<<cases.size()<<" fixtures; "<<accepted<<" exact whole-box witness proofs; "
                 <<sampled<<" baseline contact/CCD sampled-state comparisons; "<<exact_checks<<" exact CCD verdicts\n";
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
