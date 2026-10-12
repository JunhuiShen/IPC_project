#include "resident_rigid_ccd.cuh"
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
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu::resident;
void resident_rigid_ccd_test_cuda(const RigidRotationQuery*,const double*,std::size_t,
    RigidRotationResult*,double*,RigidRotationStatus*);

#ifdef IPC_RESIDENT_RIGID_CCD_HOST_REFERENCE
void resident_rigid_ccd_test_cuda(const RigidRotationQuery* queries,const double* parameters,
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
    require(error <= 2e-9,label+" mismatch "+std::to_string(error));
}
static Vec3 v3(const double* p) { return Vec3(p[0],p[1],p[2]); }
static Vec4 v4(const double* p) { return Vec4(p[0],p[1],p[2],p[3]); }
static Vec4 axis_angle(Vec3 axis,double angle) {
    axis.normalize(); const double s = std::sin(.5*angle);
    return Vec4(std::cos(.5*angle),s*axis[0],s*axis[1],s*axis[2]);
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
    resident_rigid_ccd_test_cuda(queries.data(),parameters.data(),count,results.data(),values.data(),statuses.data());
    for (std::size_t i = 0; i < count; ++i) {
        const auto& q = queries[i]; const double* p = parameters.data()+10*i;
        const double* v = values.data()+17*i;
        const std::string label = "query "+std::to_string(i);
        require(results[i].status == RigidRotationStatus::Success,label+" invalid status");
        double reference_time = -1;
        const bool reference = q.vertex_face
            ? point_triangle_rb_rotation_ccd(v3(q.positions[0]),v3(q.com),v4(q.target),v4(q.current),v3(q.positions[1]),v3(q.positions[2]),v3(q.positions[3]),reference_time)
            : segment_segment_rb_rotation_ccd(v3(q.positions[0]),v3(q.positions[1]),v3(q.com),v4(q.target),v4(q.current),v3(q.positions[2]),v3(q.positions[3]),reference_time);
        require(results[i].collision == reference,label+" collision classification mismatch");
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

static void error_cases() {
    auto q = fixtures().front();
    q.current[0] = 1; q.current[1] = q.current[2] = q.current[3] = 0;
    q.target[0] = -1; q.target[1] = q.target[2] = q.target[3] = 0;
    const double p[10] = {.5,.03,.8,.2,.3,.7,1,0,0,0};
    RigidRotationResult result; double values[17]{}; RigidRotationStatus statuses[4];
    resident_rigid_ccd_test_cuda(&q,p,1,&result,values,statuses);
    require(result.status == RigidRotationStatus::AmbiguousFullTurn,"full turn must fail closed");
    for (int k = 0; k < 3; ++k) require(statuses[k] == RigidRotationStatus::AmbiguousFullTurn,"full-arc helper must reject indeterminate axis");
    q.target[0] = 1; q.extra_turns = 2;
    resident_rigid_ccd_test_cuda(&q,p,1,&result,values,statuses);
    require(result.status == RigidRotationStatus::UnsupportedWinding,"unencoded multiple turns must not be accepted");
    q.extra_turns = 0; q.positions[0][0] = std::numeric_limits<double>::infinity();
    resident_rigid_ccd_test_cuda(&q,p,1,&result,values,statuses);
    require(result.status == RigidRotationStatus::InvalidInput,"infinite positions must be rejected");
}

int main() {
    try {
        const auto queries = fixtures();
        for (std::size_t count : {std::size_t(0),std::size_t(1),std::size_t(3),std::size_t(33),std::size_t(129),queries.size()}) verify(queries,count);
        error_cases();
        require(hits > 10,"test coverage has insufficient actual impacts");
#ifdef IPC_RESIDENT_RIGID_CCD_HOST_REFERENCE
        std::cout << "HOST ARITHMETIC ONLY (not CUDA validation): ";
#else
        std::cout << "CUDA curved rigid CCD: ";
#endif
        std::cout << comparisons << " components, " << hits << " impacts match CPU; max normalized error "
                  << std::setprecision(8) << maximum_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Resident rigid CCD test failed: " << error.what() << '\n'; return 1;
    }
}
