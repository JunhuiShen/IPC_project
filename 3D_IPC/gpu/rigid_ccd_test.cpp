#include "rigid_ccd.cuh"
#include "ccd.h"
#include "parallel_helper.h"
#include "rigid_body_ipc.h"
#include "safe_step.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu::solver;
void gpu_rigid_ccd_test_cuda(const RigidRotationQuery*,const double*,std::size_t,
    RigidRotationResult*,double*,RigidRotationStatus*);

#ifdef IPC_GPU_RIGID_CCD_HOST_REFERENCE
void gpu_rigid_ccd_test_cuda(const RigidRotationQuery* queries,const double* parameters,
    std::size_t count,RigidRotationResult* results,double* values,RigidRotationStatus* statuses) {
    for (std::size_t i = 0; i < count; ++i) {
        const auto& q = queries[i]; const double* p = parameters+10*i;
        double* v = values+17*i; auto* s = statuses+4*i;
        results[i] = rigid_rotation_ccd(q);
        s[0] = interpolate_rigid_orientation(q.current,q.target,p[0],v);
        s[1] = bound_rigid_orientation(p+6,q.current,q.target,p[2],v+4);
        s[2] = rigid_angular_velocity_from_orientation(q.target,q.current,p[1],v+8);
        BpAabb box{};
        s[3] = rigid_spherical_cap_box(q.com,q.current,p+3,p[2],box);
        for (int k = 0; k < 3; ++k) { v[11+k] = box.lo[k]; v[14+k] = box.hi[k]; }
    }
}
#endif

static void require(bool value,const std::string& message) { if (!value) throw std::runtime_error(message); }
static std::size_t comparisons = 0, hits = 0;
static double maximum_error = 0;
static void compare(double value,double reference,const std::string& label) {
    require(std::isfinite(value) && std::isfinite(reference),label+" nonfinite");
    const double error = std::abs(value-reference)/(1+std::abs(reference));
    maximum_error = std::max(maximum_error,error); ++comparisons;
    if (error > 2e-9) {
        std::ostringstream message;
        message << std::setprecision(17) << label << " mismatch: actual=" << value
                << " CPU=" << reference << " normalized error=" << error;
        throw std::runtime_error(message.str());
    }
}
static Vec3 v3(const double* p) { return Vec3(p[0],p[1],p[2]); }
static Vec4 v4(const double* p) { return Vec4(p[0],p[1],p[2],p[3]); }
static Vec4 axis_angle(Vec3 axis,double angle) {
    axis.normalize(); const double s = std::sin(.5*angle);
    return Vec4(std::cos(.5*angle),s*axis[0],s*axis[1],s*axis[2]);
}

static std::string describe(const RigidRotationQuery& q) {
    std::ostringstream message;
    message << std::setprecision(17) << " kind=" << (q.vertex_face ? "NT" : "EE");
    for (int i=0;i<4;++i)message << " x" << i << "=(" << v3(q.positions[i]).transpose() << ')';
    message << " com=(" << v3(q.com).transpose() << ") current=("
            << v4(q.current).transpose() << ") target=(" << v4(q.target).transpose() << ')';
    return message.str();
}

static std::vector<RigidRotationQuery> fixtures() {
    constexpr double pi = rotation_detail::pi;
    std::vector<RigidRotationQuery> queries;
    auto add = [&](bool nt,const std::array<Vec3,4>& x,Vec3 axis,double angle,Vec3 com=Vec3::Zero()) {
        RigidRotationQuery q{}; q.vertex_face = nt;
        const Vec4 current = queries.size()%2 ? Vec4(.8,.2,-.4,.4).normalized() : Vec4(1,0,0,0);
        const Vec4 target = quaternion_multiply(axis_angle(axis,angle),current);
        for (int i = 0; i < 4; ++i) {
            q.current[i] = current[i]; q.target[i] = target[i];
            for (int k = 0; k < 3; ++k) q.positions[i][k] = x[i][k]+com[k];
        }
        for (int k = 0; k < 3; ++k) q.com[k] = com[k];
        queries.push_back(q);
    };
    for (double angle : {0.0,1e-11,pi/4,pi/2,pi,1.5*pi,1.999*pi,-pi/2}) {
        // NT: coplanar, tilted, skew, tangent/grazing, degenerate, axis point.
        add(true,{Vec3(0,-2,0),Vec3(1,0,0),Vec3(3,0,0),Vec3(2,1,0)},Vec3::UnitZ(),angle);
        add(true,{Vec3(0,-2,0),Vec3(1,0,-1),Vec3(3,0,-1),Vec3(2,0,1)},Vec3::UnitZ(),angle);
        add(true,{Vec3(0,-2,0),Vec3(1.5,-.5,-.5),Vec3(2.7,.6,.4),Vec3(1.6,.7,.8)},Vec3::UnitZ(),angle);
        add(true,{Vec3(0,-2,0),Vec3(2,-1,-1),Vec3(2,1,-1),Vec3(2,0,1)},Vec3::UnitZ(),angle);
        add(true,{Vec3(0,-2,0),Vec3(1,0,0),Vec3(2,0,0),Vec3(3,0,0)},Vec3::UnitZ(),angle);
        add(true,{Vec3(0,0,1),Vec3(-1,-1,1),Vec3(1,-1,1),Vec3(0,1,1)},Vec3::UnitZ(),angle);
        // SS: general quadratic surface, annulus piercing, in-plane,
        // different parallel planes, true inner radius, axis-degenerate.
        add(false,{Vec3(0,-1,0),Vec3(0,-2,2),Vec3(1,-.5,1),Vec3(2,.5,1)},Vec3::UnitZ(),angle);
        add(false,{Vec3(0,-1,0),Vec3(0,-2,0),Vec3(1.5,0,-1),Vec3(1.5,0,1)},Vec3::UnitZ(),angle);
        add(false,{Vec3(0,-1,0),Vec3(0,-2,0),Vec3(1.5,0,0),Vec3(3,0,0)},Vec3::UnitZ(),angle);
        add(false,{Vec3(0,-1,0),Vec3(0,-2,0),Vec3(1.5,0,5),Vec3(3,0,5)},Vec3::UnitZ(),angle);
        add(false,{Vec3(-1,.2,0),Vec3(1,.2,0),Vec3(.5,0,-1),Vec3(.5,0,1)},Vec3::UnitZ(),angle);
        add(false,{Vec3(0,0,0),Vec3(0,0,1),Vec3(-1,0,.5),Vec3(1,0,.5)},Vec3::UnitZ(),angle);
    }
    // Existing arbitrary-axis and skew fixtures with known impacts.
    add(false,{Vec3(1,0,-1),Vec3(0,2,2),Vec3(.7351048854627524,.60605919321519798,.65883592132205056),
        Vec3(-.01080185722448657,1.2666188030347671,.74418305418971882)},Vec3(1,1,1),pi/2,Vec3(.5,-.3,.2));
    add(false,{Vec3(-1,.5,-1),Vec3(1,-.3,2),Vec3(.23511410091698948,-.32360679774997875,.5),
        Vec3(-.35267115137548433,.48541019662496826,.5)},Vec3::UnitZ(),pi/2);
    add(false,{Vec3(2,0,-2),Vec3(-2,0,2),Vec3(0,1,1),Vec3(0,1,-1)},Vec3::UnitZ(),pi);
    // Deterministic arbitrary geometry, including negative-axis arcs.
    for (int i = 0; i < 160; ++i) {
        std::array<Vec3,4> positions;
        for (int j = 0; j < 4; ++j) positions[j] = Vec3(std::sin(.37*(i+3*j)),std::cos(.61*(i+2*j)),std::sin(.23*(2*i+j)));
        add(i%2 == 0,positions,Vec3(.3+std::sin(.7*i),.5+std::cos(.2*i),.9),
            (i%2 ? -1 : 1)*(.02+1.9*pi*(i+1)/161.0),Vec3(.2,-.4,.7));
    }
    return queries;
}

static void verify(const std::vector<RigidRotationQuery>& queries,std::size_t count) {
    std::vector<double> parameters(10*count),values(17*count);
    std::vector<RigidRotationResult> results(count);
    std::vector<RigidRotationStatus> statuses(4*count);
    for (std::size_t i = 0; i < count; ++i) {
        double* p = parameters.data()+10*i;
        p[0] = .37; p[1] = .031; p[2] = i%5 == 0 ? 4.0 : .3+.2*(i%5);
        p[3] = .2; p[4] = -.7; p[5] = 1.3;
        for (int k = 0; k < 4; ++k) p[6+k] = queries[i].current[k];
    }
    gpu_rigid_ccd_test_cuda(queries.data(),parameters.data(),count,results.data(),values.data(),statuses.data());
    for (std::size_t i = 0; i < count; ++i) {
        const auto& q = queries[i]; const double* p = parameters.data()+10*i;
        const double* v = values.data()+17*i;
        const std::string label = "query "+std::to_string(i);
        require(results[i].status == RigidRotationStatus::Success,label+" invalid status");
        double reference_time = -1;
        const bool reference = q.vertex_face
            ? point_triangle_rb_rotation_ccd(v3(q.positions[0]),v3(q.com),v4(q.target),v4(q.current),v3(q.positions[1]),v3(q.positions[2]),v3(q.positions[3]),reference_time)
            : segment_segment_rb_rotation_ccd(v3(q.positions[0]),v3(q.positions[1]),v3(q.com),v4(q.target),v4(q.current),v3(q.positions[2]),v3(q.positions[3]),reference_time);
        if (results[i].collision != reference) {
            std::ostringstream message;
            message << std::setprecision(17) << label << " collision classification mismatch: actual="
                    << results[i].collision << " TOI=" << results[i].time << " CPU=" << reference
                    << " CPU TOI=" << reference_time << describe(q);
            throw std::runtime_error(message.str());
        }
        compare(results[i].time,reference_time,label+" TOI");
        hits += reference;
        for (int k = 0; k < 4; ++k) require(statuses[4*i+k] == RigidRotationStatus::Success,label+" helper status");
        const Vec4 interpolated = interpolate_orientation_full_arc(v4(q.current),v4(q.target),p[0]);
        const Vec4 bounded = bound_quaternion(v4(p+6),v4(q.current),v4(q.target),p[2]);
        const Vec3 omega = angular_velocity_from_orientation_full_arc(v4(q.target),v4(q.current),p[1]);
        const AABB box = spherical_cap_node_aabb(v3(q.com),v4(q.current),v3(p+3),p[2]);
        for (int k = 0; k < 4; ++k) { compare(v[k],interpolated[k],label+" interpolation"); compare(v[4+k],bounded[k],label+" quaternion bound"); }
        for (int k = 0; k < 3; ++k) {
            compare(v[8+k],omega[k],label+" omega recovery");
            compare(v[11+k],box.min[k],label+" cap lower"); compare(v[14+k],box.max[k],label+" cap upper");
        }
    }
}

static void tiny_rotation_cases() {
    // Nonzero angular increments below the retired 1e-10 cutoff must retain
    // initial NT contact and positive-time EE crossings. The latter also
    // detects loss from subtracting two world-scale atan2 results.
    for (double angle : {1e-11,1e-12})for (bool nt : {false,true}) {
        RigidRotationQuery q{};q.vertex_face=nt;q.current[0]=1;
        q.target[0]=std::cos(.5*angle);q.target[3]=std::sin(.5*angle);
        const double y=nt ? 0.0 : .5*angle;
        const std::array<Vec3,4> x = nt
            ? std::array<Vec3,4>{{Vec3(1,0,0),Vec3(.5,y,-1),Vec3(1.5,y,-1),Vec3(1,y,1)}}
            : std::array<Vec3,4>{{Vec3(1,0,-1),Vec3(1,0,1),Vec3(.9,y,0),Vec3(1.1,y,0)}};
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)q.positions[i][k]=x[i][k];
        // Keep the existing strict CPU comparison and all helper checks.
        verify({q},1);
        const double p[10]={.37,.031,.8,.2,-.7,1.3,1,0,0,0};
        RigidRotationResult result;double values[17]{};RigidRotationStatus statuses[4];
        gpu_rigid_ccd_test_cuda(&q,p,1,&result,values,statuses);
        require(result.status==RigidRotationStatus::Success&&result.collision,
            "tiny rotation missed a known contact"+describe(q));
        compare(result.time,std::asin(y)/angle,"tiny rotation analytic TOI"+describe(q));
    }
}

static void swept_family_cases() {
    // Permanent CPU family-coverage fixtures, plus independent known event
    // times. A near-zero radial polynomial is not enough to claim a family.
    struct Case {RigidRotationQuery query;bool hit;double first;};
    std::vector<Case> cases;
    const auto point=[](double radius,double height,double angle) {
        return Vec3(radius*std::cos(angle),radius*std::sin(angle),height);
    };
    const auto reflection=[](double t,double angle) {
        return Vec3(std::cos(angle)+t*std::sin(angle),std::sin(angle)-t*std::cos(angle),t);
    };
    const auto add=[&](const std::array<Vec3,4>& x,bool hit,double first,double angle=.04) {
        Case c{};c.hit=hit;c.first=first;c.query.current[0]=1;
        c.query.target[0]=std::cos(angle/2);c.query.target[3]=std::sin(angle/2);
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)c.query.positions[i][k]=x[i][k];
        cases.push_back(c);
    };
    for(bool reverse_moving:{false,true})for(bool reverse_stationary:{false,true}) {
        std::array<Vec3,4>x{{Vec3(1,0,1),Vec3(2,0,2),point(.5,.5,.02),point(2.5,2.5,.02)}};
        if(reverse_moving)std::swap(x[0],x[1]);
        if(reverse_stationary)std::swap(x[2],x[3]);
        add(x,true,.5);
    }
    const double phase=rotation_detail::pi/2+.02;
    add({{Vec3(1,0,0),Vec3(1,1,1),reflection(-.5,phase),reflection(1.5,phase)}},true,.5);
    const double initial_phase=2*std::atan(.5);
    add({{Vec3(1,0,0),Vec3(1,1,1),reflection(-.5,initial_phase),reflection(1.5,initial_phase)}},true,0);
    for(double angle:{.02,0.})
        add({{Vec3(1e-5,0,0),Vec3(1e-5,0,1),point(1.2e-5,0,angle),point(1.2e-5,1,angle)}},false,0);
    add({{Vec3(1,0,1),Vec3(2,0,2),point(.5,.5,.08),point(2.5,2.5,.08)}},false,0);
    add({{Vec3(1,0,1),Vec3(2,0,2),point(.5,.5,-.02),point(2.5,2.5,-.02)}},true,.5,-.04);
    add({{Vec3(1,0,1),Vec3(2,0,2),point(.5,.5,.02),point(2.5,2.5,.02)}},false,0,-.04);
    add({{Vec3(1,0,1),Vec3(2,0,2),point(3,3,.02),point(4,4,.02)}},false,0);
    std::vector<RigidRotationQuery> queries;for(const auto& c:cases)queries.push_back(c.query);
    verify(queries,queries.size());
    std::vector<RigidRotationResult> results(cases.size());
    std::vector<double> parameters(10*cases.size()),values(17*cases.size());
    std::vector<RigidRotationStatus> statuses(4*cases.size());
    for(std::size_t i=0;i<cases.size();++i) {
        double* p=parameters.data()+10*i;
        p[0]=.37;p[1]=.031;p[2]=.8;p[3]=.2;p[4]=-.7;p[5]=1.3;p[6]=1;
    }
    gpu_rigid_ccd_test_cuda(queries.data(),parameters.data(),queries.size(),results.data(),values.data(),statuses.data());
    for(std::size_t i=0;i<cases.size();++i) {
        const auto& c=cases[i];const auto& result=results[i];
        const auto label="finite swept family "+std::to_string(i)+describe(c.query);
        require(result.status==RigidRotationStatus::Success&&result.collision==c.hit,label+" classification");
        if(c.hit)compare(result.time,c.first,label+" analytic first event");
    }
}

static void recorded_near_start_contact() {
    // Exact binary64 callee arguments from CPU frame245. Do not normalize the
    // supplied quaternions again: the independent contact witness uses them.
    const std::array<Vec3,4> points{{
        Vec3(0x1.419226b948f29p+3,0x1.1f1e78fb3f4d7p+4,-0x1.fa077b31bbc00p-4),
        Vec3(0x1.41f5a7a328e14p+3,0x1.1f788280b7aa6p+4,-0x1.b63b6c7b9dc76p-3),
        Vec3(0x1.3f1eec1e1cfb7p+3,0x1.1f769596b28cep+4,-0x1.6e6609bb69eb7p-4),
        Vec3(0x1.41bcc79805a87p+3,0x1.1f1dac9fc34a5p+4,-0x1.0a72dcb80e2ffp-3)}};
    const Vec3 center(0x1.49f110c7198d8p+3,0x1.26cb954a6113bp+4,-0x1.f47b2a00f3064p-5);
    const Vec4 current(0x1.6399e14040a32p-1,0x1.b4b58c7f90ebfp-3,0x1.58e581de887ffp-1,-0x1.15a9f306c6677p-3);
    const Vec4 target(0x1.62b3f88b5e0b4p-1,0x1.b55edec4a4a14p-3,0x1.59d3965cb2d80p-1,-0x1.147de6b9f27d7p-3);
    RigidRotationQuery query{};
    for(int i=0;i<4;++i) {
        query.current[i]=current[i];query.target[i]=target[i];
        for(int k=0;k<3;++k)query.positions[i][k]=points[i][k];
    }
    for(int k=0;k<3;++k)query.com[k]=center[k];
    verify({query},1);
    const double p[10]={.37,.031,.8,.2,-.7,1.3,current[0],current[1],current[2],current[3]};
    RigidRotationResult result;double values[17]{};RigidRotationStatus statuses[4];
    gpu_rigid_ccd_test_cuda(&query,p,1,&result,values,statuses);
    const double first_contact=2.505511442023970287069885789088298082524785816353792442326697412476885054634755448391E-9;
    require(result.status==RigidRotationStatus::Success&&result.collision
        &&std::isfinite(result.time)&&result.time>0&&result.time<=1,
        "recorded near-start rigid contact was not found"+describe(query));
    std::ostringstream message;message << std::setprecision(17)
        << "recorded near-start accepted weight=" << .9*result.time
        << " must precede independent first contact=" << first_contact << describe(query);
    require(.9*result.time<std::nextafter(first_contact,0.0),message.str());
}

static void error_cases() {
    auto q = fixtures().front();
    q.current[0] = 1; q.current[1] = q.current[2] = q.current[3] = 0;
    q.target[0] = -1; q.target[1] = q.target[2] = q.target[3] = 0;
    const double p[10] = {.5,.03,.8,.2,.3,.7,1,0,0,0};
    RigidRotationResult result; double values[17]{}; RigidRotationStatus statuses[4];
    gpu_rigid_ccd_test_cuda(&q,p,1,&result,values,statuses);
    require(result.status == RigidRotationStatus::AmbiguousFullTurn,"full turn must fail closed");
    for (int k = 0; k < 3; ++k) require(statuses[k] == RigidRotationStatus::AmbiguousFullTurn,"full-arc helper must reject indeterminate axis");
    q.target[0] = 1; q.extra_turns = 2;
    gpu_rigid_ccd_test_cuda(&q,p,1,&result,values,statuses);
    require(result.status == RigidRotationStatus::UnsupportedWinding,"unencoded multiple turns must not be accepted");
    q.extra_turns = 0; q.positions[0][0] = std::numeric_limits<double>::infinity();
    gpu_rigid_ccd_test_cuda(&q,p,1,&result,values,statuses);
    require(result.status == RigidRotationStatus::InvalidInput,"infinite positions must be rejected");
}

int main() {
    try {
        const auto queries = fixtures();
        for (std::size_t count : {std::size_t(0),std::size_t(1),std::size_t(3),std::size_t(33),std::size_t(129),queries.size()}) verify(queries,count);
        tiny_rotation_cases();
        swept_family_cases();
        recorded_near_start_contact();
        error_cases();
        require(hits > 10,"test coverage has insufficient actual impacts");
#ifdef IPC_GPU_RIGID_CCD_HOST_REFERENCE
        std::cout << "HOST ARITHMETIC ONLY (not CUDA validation): ";
#else
        std::cout << "CUDA curved rigid CCD: ";
#endif
        std::cout << comparisons << " components, " << hits << " impacts match CPU; max normalized error "
                  << std::setprecision(8) << maximum_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU rigid CCD test failed: " << error.what() << '\n'; return 1;
    }
}
