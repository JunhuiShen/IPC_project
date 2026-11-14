#include "ccd.cuh"
#include "original_rigid.cuh"
#include "reference_projection.cuh"

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

namespace detail=ipc_gpu::ccd::detail;
namespace cache=ipc_gpu::gs_ccd_cache;
using Query=ipc_gpu::ccd::LinearQuery;

static std::uint64_t bits(double value) {
    std::uint64_t result;std::memcpy(&result,&value,sizeof(result));return result;
}
static void require(bool value,const std::string& description) {
    if(!value)throw std::runtime_error(description);
}
static void equal(double a,double b,const char* field) {
    require(bits(a)==bits(b),std::string("bit mismatch: ")+field);
}

// Independent raw axis enumeration, checked against the original full
// separated() result as well as every original projection() result.
static std::vector<detail::Vec> axes(const Query& q) {
    std::vector<detail::Vec> result;
    result.push_back({{1,0,0}});result.push_back({{0,1,0}});result.push_back({{0,0,1}});
    for(const double (*p)[3]:{q.start,q.end}) {
        if(q.vertex_face) {
            const auto normal=detail::cross(detail::difference(p[2],p[1]),detail::difference(p[3],p[1]));
            result.push_back(normal);
            for(const auto& pair:std::array<std::array<int,2>,3>{{{{2,1}},{{3,2}},{{1,3}}}})
                result.push_back(detail::cross(normal,detail::difference(p[pair[0]],p[pair[1]])));
        } else {
            const auto a=detail::difference(p[1],p[0]);
            const auto b=detail::difference(p[3],p[2]);
            const auto w=detail::difference(p[2],p[0]);
            result.push_back(detail::cross(a,b));
            result.push_back(detail::cross(detail::cross(a,w),a));
            result.push_back(detail::cross(detail::cross(b,w),b));
        }
        const int split=q.vertex_face?1:2;
        for(int i=0;i<split;++i)for(int j=split;j<4;++j)
            result.push_back(detail::difference(p[i],p[j]));
    }
    return result;
}

static Query face() {
    Query q;
    const double p[4][3]={{.2,.2,1},{0,0,0},{1,0,0},{0,1,0}};
    std::memcpy(q.start,p,sizeof(p));std::memcpy(q.end,p,sizeof(p));return q;
}
static Query edges() {
    Query q;q.vertex_face=false;
    const double p[4][3]={{-1,0,1},{1,0,1},{0,-1,0},{0,1,0}};
    std::memcpy(q.start,p,sizeof(p));std::memcpy(q.end,p,sizeof(p));return q;
}
static Query translate(Query q,double delta) {
    for(auto& p:q.end)p[2]+=delta;
    return q;
}

template<int C>
static unsigned check_status(const Query& q) {
    const auto a=ipc_gpu::original_rigid::represented_path<C>(q.start,q.end,q.vertex_face);
    const auto b=cache::represented_path<C>(q.start,q.end,q.vertex_face);
    require(a==b,"represented_path status mismatch at capacity "+std::to_string(C));
    return static_cast<unsigned>(a);
}

int main() {
    try {
        std::vector<Query> cases;
        std::vector<Query> status_cases;
        for(bool vf:{false,true}) {
            Query q=vf?face():edges();
            status_cases.push_back(q);
            status_cases.push_back(translate(q,2));
            for(int i=0;i<4;++i)q.end[i][2]+=i<(vf?1:2)?-1:1;
            status_cases.push_back(q);
            q=vf?face():edges();
            q.start[0][2]=q.end[0][2]=0;
            if(!vf)q.start[1][2]=q.end[1][2]=0;
            status_cases.push_back(translate(q,2));
            q=vf?face():edges();q.end[0][2]=-1;
            status_cases.push_back(q);
            q=Query{};q.vertex_face=vf;
            status_cases.push_back(q);
            q=vf?face():edges();q.start[0][0]=std::numeric_limits<double>::quiet_NaN();
            status_cases.push_back(q);
            q=vf?face():edges();q.end[1][1]=std::numeric_limits<double>::infinity();
            status_cases.push_back(q);
            // Degenerate primitives retain exact endpoint membership behavior.
            q=Query{};q.vertex_face=vf;
            q.start[0][0]=q.end[0][0]=-1;
            if(!vf)q.start[1][0]=q.end[1][0]=-1;
            for(auto& p:q.end)p[0]+=2;
            status_cases.push_back(q);
        }
        cases=status_cases;
        std::mt19937_64 rng(7301631);
        std::uniform_real_distribution<double> unit(-1,1);
        for(int i=0;i<384;++i) {
            Query q;q.vertex_face=i%2==0;
            for(int v=0;v<4;++v)for(int k=0;k<3;++k) {
                double x=2*unit(rng),y=x+unit(rng);
                if(i%2==0){x=double(float(x));y=double(float(y));}
                q.start[v][k]=x;q.end[v][k]=y;
            }
            cases.push_back(q);
        }
        constexpr int exponents[]={-1074,-1070,-1022,-800,-100,-1,0,1,100,500,900,1023};
        for(int exponent:exponents)for(int repeat=0;repeat<4;++repeat) {
            Query q;q.vertex_face=repeat%2==0;
            for(int v=0;v<4;++v)for(int k=0;k<3;++k) {
                q.start[v][k]=std::scalbn(unit(rng),exponent);
                q.end[v][k]=std::scalbn(unit(rng),exponent);
            }
            cases.push_back(q);
        }
        for(int i=0;i<48;++i) {
            Query q;q.vertex_face=i%2==0;
            for(int v=0;v<4;++v)for(int k=0;k<3;++k) {
                q.start[v][k]=std::scalbn(unit(rng),exponents[rng()%12]);
                q.end[v][k]=std::scalbn(unit(rng),exponents[rng()%12]);
            }
            cases.push_back(q);
        }
        for(double gap:{0.0,-0.0,1e-10,1e-8,double(std::numeric_limits<float>::denorm_min()),
                         std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::min()}) {
            Query q=face();q.start[0][2]=gap;q.end[0][2]=std::nextafter(gap,INFINITY);
            cases.push_back(q);
        }
        std::uint64_t interval_checks=0,projection_checks=0,invariant_checks=0,status_checks=0;
        const std::array<int,8> selected{{0,1,63,127,128,191,254,255}};
        for(const Query& q:cases) {
            const auto prepared=cache::prepare(q);
            const auto raw=axes(q);
            require(raw.size()==17&&prepared.count==17,"axis count/order shape");
            for(int v=0;v<4;++v)for(int k=0;k<3;++k) {
                const auto base=detail::subtract(detail::point(q.start[v][k]),detail::point(q.start[3][k]));
                const auto motion=detail::subtract(detail::point(q.end[v][k]),detail::point(q.start[v][k]));
                equal(base.lo,prepared.base[v][k].lo,"base.lo");equal(base.hi,prepared.base[v][k].hi,"base.hi");
                equal(motion.lo,prepared.motion[v][k].lo,"motion.lo");equal(motion.hi,prepared.motion[v][k].hi,"motion.hi");
                invariant_checks+=4;
            }
            for(int i=0;i<256;++i) {
                const double lo=double(i)/256,hi=double(i+1)/256;
                const bool original=detail::separated(q,lo,hi,0.0);
                require(original==cache::separated_zero_gap(prepared,lo,hi),"subinterval separated mismatch");
                ++interval_checks;
            }
            std::vector<std::array<double,2>> intervals{{{0,1}},{{0,0}},{{1,1}}};
            for(int i:selected)intervals.push_back({{double(i)/256,double(i+1)/256}});
            for(const auto& time:intervals)for(int i=0;i<17;++i) {
                cache::Axis invariant;cache::ProjectionTrace expected,actual;
                const bool original=detail::projection(q,raw[i],time[0],time[1],0.0);
                const bool traced=detail::reference_projection(q,raw[i],time[0],time[1],0.0,&invariant,&expected);
                const bool cached=cache::projection_zero_gap(prepared,prepared.axes[i],time[0],time[1],&actual);
                require(original==traced&&original==cached,"per-axis projection mismatch");
                require(invariant.usable==prepared.axes[i].usable,"usable mismatch");
                if(invariant.usable) {
                    for(int k=0;k<3;++k)equal(invariant.direction.v[k],prepared.axes[i].direction.v[k],"axis.direction");
                    equal(invariant.norm,prepared.axes[i].norm,"axis.norm");
                    equal(invariant.threshold,prepared.axes[i].threshold,"axis.threshold");
                    equal(expected.first_lo,actual.first_lo,"first_lo");equal(expected.first_hi,actual.first_hi,"first_hi");
                    equal(expected.second_lo,actual.second_lo,"second_lo");equal(expected.second_hi,actual.second_hi,"second_hi");
                    equal(expected.ab,actual.ab,"ab");equal(expected.ba,actual.ba,"ba");
                    invariant_checks+=11;
                }
                ++projection_checks;
            }
        }
        std::array<unsigned,5> status_counts{};
        for(const Query& q:status_cases) {
            ++status_counts[check_status<2>(q)];
            ++status_counts[check_status<256>(q)];
            ++status_counts[check_status<2048>(q)];
            status_checks+=3;
        }
        for(unsigned count:status_counts)require(count!=0,"missing represented status coverage");
        // Verify the important cold branch is actually reached by the fixture.
        for(Query q:{translate(face(),2),translate(edges(),2)}) {
            require(!detail::separated(q,0,1,0),"cold fixture was rejected by whole-step proof");
            require(ipc_gpu::exact::validate_collision_step<2048>(q.start,q.start,q.vertex_face)==ipc_gpu::exact::Status::Safe,"cold initial not safe");
            require(cache::represented_path<2048>(q.start,q.end,q.vertex_face)==ipc_gpu::original_rigid::RepresentationStatus::Safe,"subdivided safe fixture");
        }
        std::cout<<"{\"cases\":"<<cases.size()<<",\"interval_checks\":"<<interval_checks
            <<",\"projection_checks\":"<<projection_checks<<",\"bitwise_invariant_checks\":"<<invariant_checks
            <<",\"represented_status_checks\":"<<status_checks
            <<",\"status_counts_safe_uncertain_initial_invalid_capacity\":["
            <<status_counts[0]<<","<<status_counts[1]<<","<<status_counts[2]<<","<<status_counts[3]<<","<<status_counts[4]<<"]"
            <<",\"prepared_path_bytes\":"<<sizeof(cache::PreparedPath)
            <<",\"all_passed\":true}"<<std::endl;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<std::endl;return 1;
    }
}
