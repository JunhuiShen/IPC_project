#include "resident_contact_rejection_test.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu::resident;
using R=boost::multiprecision::cpp_rational;
using I=boost::multiprecision::cpp_int;
using V=std::array<R,3>;

namespace {
void require(bool ok,const std::string& message) {if(!ok)throw std::runtime_error(message);}
R rational(double x) {
    require(std::isfinite(x),"nonfinite rational input");
    if(x==0)return 0;
    int exponent=0;
    const double fraction=std::frexp(std::abs(x),&exponent);
    const auto significand=static_cast<std::uint64_t>(std::ldexp(fraction,53));
    I n=significand,d=1;
    exponent-=53;
    if(exponent>=0)n<<=exponent;else d<<=-exponent;
    if(x<0)n=-n;
    return R(n)/R(d);
}
V sub(const V& a,const V& b) {return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}};}
R dot(const V& a,const V& b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
R clamp(R t) {if(t<0)return 0;if(t>1)return 1;return t;}
R point_segment(const V& p,const V& a,const V& b) {
    const V e=sub(b,a),w=sub(p,a);
    const R length=dot(e,e),t=length>0?clamp(R(dot(w,e)/length)):R(0);
    V r{};for(int k=0;k<3;++k)r[k]=w[k]-t*e[k];
    return dot(r,r);
}
R exact_squared_distance(const ContactInput& in) {
    V x[4];for(int i=0;i<4;++i)for(int k=0;k<3;++k)x[i][k]=rational(in.positions[i][k]);
    if(in.kind==ContactKind::PointTriangle) {
        const V a=sub(x[2],x[1]),b=sub(x[3],x[1]),w=sub(x[0],x[1]);
        const R A=dot(a,a),B=dot(a,b),C=dot(b,b),D=dot(a,w),E=dot(b,w),det=A*C-B*B;
        if(det>0) {
            const R u=(D*C-E*B)/det,v=(E*A-D*B)/det;
            if(u>=0&&v>=0&&u+v<=1) {
                V r{};for(int k=0;k<3;++k)r[k]=w[k]-u*a[k]-v*b[k];
                return dot(r,r);
            }
        }
        R best=point_segment(x[0],x[1],x[2]);
        for(int i=2;i<=3;++i) {const R d=point_segment(x[0],x[i],x[i==3?1:3]);if(d<best)best=d;}
        return best;
    }
    const V a=sub(x[1],x[0]),b=sub(x[3],x[2]),w=sub(x[0],x[2]);
    const R A=dot(a,a),B=dot(a,b),C=dot(b,b),D=dot(a,w),E=dot(b,w),det=A*C-B*B;
    R best=point_segment(x[0],x[2],x[3]);
    const R boundary[3]={point_segment(x[1],x[2],x[3]),point_segment(x[2],x[0],x[1]),point_segment(x[3],x[0],x[1])};
    for(const R& d:boundary)if(d<best)best=d;
    if(det>0) {
        const R s=(B*E-C*D)/det,t=(A*E-B*D)/det;
        if(s>=0&&s<=1&&t>=0&&t<=1) {
            V r{};for(int k=0;k<3;++k)r[k]=w[k]+s*a[k]-t*b[k];
            const R d=dot(r,r);if(d<best)best=d;
        }
    }
    return best;
}
ContactInput make(ContactKind kind,std::array<std::array<double,3>,4> x,double d_hat) {
    ContactInput in;in.kind=kind;in.d_hat=d_hat;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)in.previous[i][k]=in.positions[i][k]=x[i][k];
    return in;
}
ContactInput parallel(double gap,double d_hat) {
    return make(ContactKind::EdgeEdge,{{{{0,0,0}},{{1,0,0}},{{0,gap,0}},{{1,gap,0}}}},d_hat);
}
struct Fixture {ContactInput in;bool must_certify=false;};
std::vector<Fixture> fixtures() {
    std::vector<Fixture> f;
    f.push_back({parallel(.2,.1),true});
    // Coordinate AABBs overlap; only an in-plane perpendicular separates.
    const auto diagonal=make(ContactKind::EdgeEdge,{{{{0,0,0}},{{1,1,0}},{{0,1,0}},{{1,2,0}}}},.25);
    f.push_back({diagonal,true});
    f.push_back({make(ContactKind::PointTriangle,{{{{-.1,.1,0}},{{0,0,0}},{{2,0,0}},{{3,1,0}}}},.05),true});
    f.push_back({make(ContactKind::PointTriangle,{{{{.5,.2,0}},{{0,0,0}},{{1,0,0}},{{2,0,0}}}},.1),true});
    // Small triangle: ordinary Gram determinant requests exact arithmetic;
    // the point lies inside its AABB but beyond the diagonal edge.
    f.push_back({make(ContactKind::PointTriangle,{{{{.00008,.00008,0}},{{0,0,0}},{{.0001,0,0}},{{0,.0001,0}}}},.00001),true});
    f.push_back({parallel(1,1)});
    f.push_back({parallel(std::nextafter(1.,0.),1)});
    f.push_back({parallel(std::nextafter(1.,INFINITY),1)});
    f.push_back({parallel(std::nextafter(std::nextafter(1.,INFINITY),INFINITY),1),true});
    f.push_back({parallel(0,1)});
    for(int exponent:{-1000,-500,-100,100,500,900}) {
        auto scaled=diagonal;
        scaled.d_hat=std::ldexp(scaled.d_hat,exponent);
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)scaled.positions[i][k]=scaled.previous[i][k]=std::ldexp(scaled.positions[i][k],exponent);
        f.push_back({scaled,true});
        scaled.d_hat=std::ldexp(1.,exponent);f.push_back({scaled});
    }
    const double tiny=std::numeric_limits<double>::denorm_min();
    f.push_back({parallel(8*tiny,2*tiny),true});
    f.push_back({parallel(tiny,tiny)});
    auto translated=diagonal;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)
        translated.positions[i][k]=translated.previous[i][k]=std::ldexp(1.,500)+std::ldexp(diagonal.positions[i][k],452);
    translated.d_hat=std::ldexp(.25,452);f.push_back({translated,true});
    auto overflow=parallel(1,1);
    for(int i=0;i<4;++i)overflow.positions[i][0]=overflow.previous[i][0]=i<2?-DBL_MAX:DBL_MAX;
    f.push_back({overflow});
    std::mt19937_64 random(0xdccdf00d);
    for(int n=0;n<2500;++n) {
        ContactInput in;in.kind=n%2?ContactKind::PointTriangle:ContactKind::EdgeEdge;
        const int exponent=n%11==0?(int(random()%1801)-900):0;
        const double scale=std::ldexp(1.,exponent);
        in.d_hat=(1+random()%32)*scale/8;
        in.friction=n%3?.2:0;
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)
            in.positions[i][k]=in.previous[i][k]=(int(random()%33)-16)*scale/8;
        if(n%13==0)for(int k=0;k<3;++k)in.positions[1][k]=in.previous[1][k]=in.positions[0][k];
        if(n%17==0)for(int k=0;k<3;++k)in.positions[3][k]=in.previous[3][k]=in.positions[2][k];
        f.push_back({in});
    }
    // Primitive-specific threshold neighbors, not just axis-aligned gaps.
    // Rounded sqrt only PROPOSES d_hat; exact rationals below judge both sides.
    const auto random_cases=f;
    int near_cases=0;
    for(std::size_t i=25;i<random_cases.size()&&near_cases<80;++i) {
        if(i%11==0)continue;
        const auto& in=random_cases[i].in;
        const double d2=exact_squared_distance(in).convert_to<double>();
        if(!(d2>1e-8&&d2<100))continue;
        const double distance=std::sqrt(d2);
        for(int neighbor=-2;neighbor<=2;++neighbor) {
            auto near=in;near.d_hat=distance;
            for(int j=0;j<std::abs(neighbor);++j)
                near.d_hat=std::nextafter(near.d_hat,neighbor<0?0.:double(INFINITY));
            f.push_back({near});
        }
        ++near_cases;
    }
    return f;
}
std::vector<ResidentContactRejectionResult> evaluate(const std::vector<ContactInput>& input) {
    std::vector<ResidentContactRejectionResult> result(input.size());
#ifdef IPC_RESIDENT_CONTACT_REJECTION_CUDA_TEST
    resident_contact_rejection_gpu(input.data(),result.data(),input.size());
#else
    for(std::size_t i=0;i<input.size();++i) {
        result[i].certified=certified_contact_inactive(input[i]);
        result[i].output=evaluate_contact(input[i]);
    }
#endif
    return result;
}
void check_invalid() {
    std::vector<ContactInput> input;
    auto add=[&](double x) {auto in=parallel(.2,.1);in.positions[0][0]=x;input.push_back(in);};
    add(INFINITY);add(-INFINITY);add(NAN);
    for(double d:{0.,-1.,double(INFINITY),double(NAN)}) {auto in=parallel(.2,.1);in.d_hat=d;input.push_back(in);}
    auto kind=parallel(.2,.1);kind.kind=static_cast<ContactKind>(9);input.push_back(kind);
    auto actual=evaluate(input);
    for(const auto& out:actual)require(!out.certified&&out.output.status==ContactStatus::InvalidInput,"invalid geometry accepted");
    input.clear();
    auto bad_material=parallel(.2,.1);bad_material.friction=-1;input.push_back(bad_material);
    bad_material=parallel(.2,.1);bad_material.previous[0][0]=NAN;input.push_back(bad_material);
    actual=evaluate(input);
    for(const auto& out:actual)require(out.output.status==ContactStatus::InvalidInput&&!out.output.active,"inactive certificate bypassed input validation");
}
}

int main() {
    try {
        const auto cases=fixtures();std::vector<ContactInput> inputs;
        for(const auto& f:cases)inputs.push_back(f.in);
        const auto result=evaluate(inputs);
        std::size_t certified=0,active_or_boundary=0,avoided_exact=0;
        for(std::size_t i=0;i<cases.size();++i) {
            const auto& in=cases[i].in;const auto& actual=result[i];
            const std::string tag="fixture "+std::to_string(i);
            const R distance2=exact_squared_distance(in),dhat=rational(in.d_hat),threshold2=dhat*dhat;
            if(distance2<=threshold2) {++active_or_boundary;require(!actual.certified,tag+": false inactive certificate");}
            if(actual.certified) {++certified;require(distance2>threshold2,tag+": exact reference disproves certificate");}
            if(cases[i].must_certify)require(actual.certified,tag+": separated fixture not certified");
            contact_detail::Feature ordinary{};double normal[3]{};
            const auto status=in.kind==ContactKind::PointTriangle?contact_detail::point_triangle(in,ordinary,normal):contact_detail::edge_edge(in,ordinary,normal);
            if(status==ContactStatus::NeedsExactGeometry&&actual.certified) {
                ++avoided_exact;
                require(actual.output.status==ContactStatus::Ok&&!actual.output.active,tag+": inactive result contract");
                require(actual.output.distance==in.d_hat,tag+": inactive distance sentinel");
                for(int role=0;role<4;++role) {
                    require(actual.output.weights[role]==0,tag+": inactive weight");
                    for(int k=0;k<3;++k) {
                        require(actual.output.barrier_gradient[role][k]==0,tag+": inactive gradient");
                        for(int j=0;j<4;++j)for(int l=0;l<3;++l)
                            require(actual.output.barrier_hessian[role][j][3*k+l]==0,tag+": inactive Hessian");
                    }
                }
            }
            if(!actual.certified&&status==ContactStatus::NeedsExactGeometry)
                require(actual.output.status==ContactStatus::NeedsExactGeometry,tag+": unresolved exact fallback lost");
        }
        check_invalid();
        std::cout<<"Certified contact rejection: "<<cases.size()<<" fixtures, "<<certified
                 <<" certificates, "<<active_or_boundary<<" active/boundary, "<<avoided_exact
                 <<" unnecessary exact requests avoided; exact rational reference passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
