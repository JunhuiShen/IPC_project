#pragma once

#include "resident_rigid.cuh"
#include "resident_broadphase.h"
#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#define IPC_RCCD_HD __host__ __device__
#else
#define IPC_RCCD_HD
#endif

namespace ipc_gpu::resident {

// Port of the current analytic rotational CCD in ccd.cpp. These are curved
// trajectories, not a linear chord approximation. CPU code uses trigonometric
// roots and a quadratic swept-edge surface (there is no cubic root stage).
// Quaternion signs retain arcs beyond pi; endpoints alone cannot encode an
// arbitrary number of complete turns. The caller must not discard winding.
enum class RigidRotationStatus : unsigned { Success, InvalidInput, AmbiguousFullTurn, OutsideCap, UnsupportedWinding };
struct RigidRotationQuery {
    double positions[4][3];
    double com[3], current[4], target[4];
    unsigned vertex_face;
    // Nonzero winding cannot be represented by this endpoint-only CPU API.
    int extra_turns = 0;
};
struct RigidRotationResult {
    RigidRotationStatus status = RigidRotationStatus::Success;
    bool collision = false;
    double time = 0.0;
};

namespace rotation_detail {
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double two_pi = 2.0 * pi;
IPC_RCCD_HD inline bool finite(double x) { return x-x == 0.0; }
IPC_RCCD_HD inline double minimum(double a, double b) { return a < b ? a : b; }
IPC_RCCD_HD inline double maximum(double a, double b) { return a > b ? a : b; }
IPC_RCCD_HD inline double clamp(double x, double lo, double hi) { return maximum(lo, minimum(hi,x)); }
struct V2 {
    double x, y;
    IPC_RCCD_HD V2 operator+(V2 b) const { return {x+b.x,y+b.y}; }
    IPC_RCCD_HD V2 operator-(V2 b) const { return {x-b.x,y-b.y}; }
    IPC_RCCD_HD V2 operator*(double a) const { return {x*a,y*a}; }
    IPC_RCCD_HD V2 operator/(double a) const { return {x/a,y/a}; }
    IPC_RCCD_HD double dot(V2 b) const { return x*b.x+y*b.y; }
    IPC_RCCD_HD double norm() const { return sqrt(dot(*this)); }
};
struct V3 {
    double x,y,z;
    IPC_RCCD_HD V3 operator+(V3 b) const { return {x+b.x,y+b.y,z+b.z}; }
    IPC_RCCD_HD V3 operator-(V3 b) const { return {x-b.x,y-b.y,z-b.z}; }
    IPC_RCCD_HD V3 operator*(double a) const { return {x*a,y*a,z*a}; }
    IPC_RCCD_HD V3 operator/(double a) const { return {x/a,y/a,z/a}; }
    IPC_RCCD_HD double dot(V3 b) const { return (x*b.x+y*b.y)+z*b.z; }
    IPC_RCCD_HD double squaredNorm() const { return dot(*this); }
    IPC_RCCD_HD double norm() const { return sqrt(squaredNorm()); }
    IPC_RCCD_HD V3 cross(V3 b) const { return {y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x}; }
};
IPC_RCCD_HD inline V3 vec(const double* p) { return {p[0],p[1],p[2]}; }
IPC_RCCD_HD inline void basis(V3 n, V3& e1, V3& e2) {
    const V3 ref = fabs(n.x) < .9 ? V3{1,0,0} : V3{0,1,0};
    e1 = n.cross(ref); e1 = e1/e1.norm(); e2 = n.cross(e1);
}
IPC_RCCD_HD inline V2 project(V3 p, V3 com, V3 e1, V3 e2) {
    const V3 d = p-com; return {d.dot(e1),d.dot(e2)};
}
IPC_RCCD_HD inline bool normalize(const double* q, double* out) {
    const double norm = sqrt(((q[0]*q[0]+q[1]*q[1])+q[2]*q[2])+q[3]*q[3]);
    if (!(norm > 0.0) || !finite(norm)) return false;
    for (int k = 0; k < 4; ++k) out[k] = q[k]/norm;
    return true;
}
IPC_RCCD_HD inline void relative(const double* target, const double* current, double* q) {
    const double conjugate[4] = {current[0],-current[1],-current[2],-current[3]};
    rigid_quaternion_multiply(target,conjugate,q);
}

IPC_RCCD_HD inline bool point_segment_2d(V2 x, V2 com, double theta_n,
    double theta_new, V2 x0, V2 x1, double& step) {
    step = 0.0;
    const V2 dx = x-com;
    const double cn = cos(theta_n), sn = sin(theta_n);
    const V2 r{cn*dx.x+sn*dx.y,-sn*dx.x+cn*dx.y};
    const V2 d = x1-x0;
    const double length = d.norm();
    if (length < 1e-14) return false;
    const V2 direction = d/length;
    const double A = r.x*direction.y-r.y*direction.x;
    const double B = -r.x*direction.x-r.y*direction.y;
    const double C = (com.x-x0.x)*direction.y-(com.y-x0.y)*direction.x;
    const double amplitude = sqrt(A*A+B*B);
    if (fabs(C) > amplitude+1e-14) return false;
    if (amplitude < 1e-14) {
        if (fabs(C) > 1e-14) return false;
        const double t = (x-x0).dot(d)/(length*length);
        return t >= -1e-12 && t <= 1.0+1e-12;
    }
    const double phi = atan2(B,A), ac = acos(clamp(-C/amplitude,-1,1));
    const double candidates[2] = {phi+ac,phi-ac};
    const double dtheta = theta_new-theta_n;
    if (fabs(dtheta) < 1e-14) return false;
    double best = INFINITY;
    for (int i = 0; i < 2; ++i) {
        const double k = dtheta > 0 ? ceil((theta_n-candidates[i]-1e-12)/two_pi)
                                  : floor((theta_n-candidates[i]+1e-12)/two_pi);
        const double theta = candidates[i]+two_pi*k;
        double s = (theta-theta_n)/dtheta;
        if (s < -1e-12 || s > 1.0+1e-12) continue;
        s = clamp(s,0,1);
        const V2 position = V2{cos(theta)*r.x-sin(theta)*r.y,sin(theta)*r.x+cos(theta)*r.y}+com;
        const double t = (position-x0).dot(d)/(length*length);
        if (t < -1e-12 || t > 1.0+1e-12) continue;
        if (s < best) best = s;
    }
    if (best == INFINITY) return false;
    step = best; return true;
}

IPC_RCCD_HD inline bool inside(V3 p, V3 origin, V3 e1, V3 e2,
    double a11, double a12, double a22, double det) {
    const V3 r = p-origin;
    const double b1 = r.dot(e1), b2 = r.dot(e2);
    const double alpha = (b1*a22-b2*a12)/det;
    const double beta = (-b1*a12+b2*a11)/det;
    return alpha >= -1e-10 && beta >= -1e-10 && 1.0-alpha-beta >= -1e-10;
}

// Raw counterparts intentionally preserve the CPU epsilon/rejection behavior.
// Use the checked public query below so full-turn ambiguity cannot be mistaken
// for a certified collision-free step.
IPC_RCCD_HD inline bool point_triangle(V3 x, V3 com, const double* target,
    const double* current, V3 a, V3 b, V3 c, double& s) {
    constexpr double eps = 1e-10;
    s = 0.0;
    double qr[4]; relative(target,current,qr);
    const V3 v = vec(qr+1); const double norm = v.norm();
    if (norm < eps) return false;
    const V3 n = v/norm, dx = x-com;
    const V3 parallel = n*dx.dot(n), perpendicular = dx-parallel;
    if (perpendicular.norm() < eps) return false;
    const double dtheta = 2.0*atan2(norm,qr[0]);
    if (fabs(dtheta) < eps) return false;
    const V3 crossed = n.cross(perpendicular), e1 = b-a, e2 = c-a;
    const double a11 = e1.dot(e1), a12 = e1.dot(e2), a22 = e2.dot(e2);
    const V3 normal = e1.cross(e2); const double normal_length = normal.norm();
    if (normal_length*normal_length <= eps*a11*a22) return false;
    const V3 unit_normal = normal/normal_length;
    const double A = perpendicular.dot(unit_normal), B = crossed.dot(unit_normal), C = (com+parallel-a).dot(unit_normal);
    const double det = a11*a22-a12*a12;
    if (fabs(det) <= eps*a11*a22) return false;
    const double amplitude = sqrt(A*A+B*B);
    if (amplitude < eps) {
        if (fabs(C) > eps) return false;
        if (inside(x,a,e1,e2,a11,a12,a22,det)) return true;
        V3 be1,be2; basis(n,be1,be2);
        const V2 p = project(x,com,be1,be2);
        const V2 vertices[3] = {project(a,com,be1,be2),project(b,com,be1,be2),project(c,com,be1,be2)};
        const double theta = atan2(p.y,p.x);
        double best = INFINITY;
        for (int i = 0; i < 3; ++i) {
            double candidate;
            if (point_segment_2d(p,{0,0},theta,theta+dtheta,vertices[i],vertices[(i+1)%3],candidate)
                && candidate < best) best = candidate;
        }
        if (best == INFINITY) return false;
        s = best; return true;
    }
    if (fabs(C) > amplitude+eps) return false;
    const double phi = atan2(B,A), ac = acos(clamp(-C/amplitude,-1,1));
    const double candidates[2] = {phi+ac,phi-ac};
    double best = INFINITY;
    for (int i = 0; i < 2; ++i) {
        const double k = dtheta > 0 ? ceil((-candidates[i]-eps)/two_pi) : floor((-candidates[i]+eps)/two_pi);
        double candidate = (candidates[i]+two_pi*k)/dtheta;
        if (candidate < -eps || candidate > 1.0+eps) continue;
        candidate = clamp(candidate,0,1);
        const double theta = candidate*dtheta;
        const V3 position = ((com+parallel)+perpendicular*cos(theta))+crossed*sin(theta);
        if (inside(position,a,e1,e2,a11,a12,a22,det) && candidate < best) best = candidate;
    }
    if (best == INFINITY) return false;
    s = best; return true;
}

IPC_RCCD_HD inline bool segment_segment(V3 x0, V3 x1, V3 com,
    const double* target, const double* current, V3 x2, V3 x3, double& s) {
    constexpr double eps = 1e-10;
    s = 0.0;
    double qr[4]; relative(target,current,qr);
    const V3 v = vec(qr+1); const double norm = v.norm();
    if (norm < eps) return false;
    const V3 n = v/norm;
    const double h0 = (x0-com).dot(n), h1 = (x1-com).dot(n);
    const V3 p0 = (x0-com)-n*h0, p1 = (x1-com)-n*h1;
    const double r0 = p0.norm(), r1 = p1.norm();
    if (maximum(r0,r1) < eps) return false;
    const double dtheta = 2.0*atan2(norm,qr[0]);
    if (fabs(dtheta) < eps) return false;
    const V3 ref = r0 >= r1 ? p0 : p1, dp = p1-p0;
    const V3 q2 = x2-com, q3 = x3-com, e = q3-q2;
    const double h2 = q2.dot(n), en = e.dot(n);
    const V3 q2p = q2-n*h2, ep = e-n*en;
    V3 e1,e2; basis(n,e1,e2);
    const double theta_n = atan2(ref.dot(e2),ref.dot(e1));
    bool found = false; double best = INFINITY;
    if (fabs(h1-h0) > eps) {
        const double dh = h1-h0, beta = en/dh;
        const V3 p0prime = p0+dp*((h2-h0)/dh);
        const double A = ep.squaredNorm()-beta*beta*dp.squaredNorm();
        const double B = 2.0*(q2p.dot(ep)-beta*p0prime.dot(dp));
        const double C = q2p.squaredNorm()-p0prime.squaredNorm();
        const double disc = B*B-4.0*A*C;
        if (disc >= 0.0) for (int sign = -1; sign <= 1; sign += 2) {
            double u;
            if (fabs(A) < eps) {
                if (fabs(B) < eps) {
                    if (fabs(C) >= eps || sign == 1) continue;
                    u = 0.0;
                } else u = -C/B;
            } else u = (-B+sign*sqrt(maximum(0.0,disc)))/(2.0*A);
            if (u < -eps || u > 1.0+eps) continue;
            u = clamp(u,0,1);
            double t = (h2+u*en-h0)/dh;
            if (t < -eps || t > 1.0+eps) continue;
            t = clamp(t,0,1);
            const V3 moving = p0+dp*t;
            if (moving.norm() < eps) continue;
            const V3 star = x2*(1.0-u)+x3*u;
            const V3 star_p = (star-com)-n*(star-com).dot(n);
            double angle = atan2(star_p.dot(e2),star_p.dot(e1))-atan2(moving.dot(e2),moving.dot(e1));
            while (dtheta > 0 && angle < -eps) angle += two_pi;
            while (dtheta < 0 && angle > eps) angle -= two_pi;
            double candidate = angle/dtheta;
            if (candidate < -eps || candidate > 1.0+eps) continue;
            candidate = clamp(candidate,0,1);
            if (candidate < best) { best = candidate; found = true; }
        }
    } else {
        const double outer = maximum(r0,r1), dp2 = dp.squaredNorm();
        double inner;
        if (dp2 < eps) inner = r0;
        else {
            const double t = -p0.dot(dp)/dp2;
            inner = t > 0.0 && t < 1.0 ? (p0+dp*t).norm() : minimum(r0,r1);
        }
        if (fabs(en) > eps) {
            double u = (h0-h2)/en;
            if (u >= -eps && u <= 1.0+eps) {
                u = clamp(u,0,1);
                const V3 star = x2*(1.0-u)+x3*u;
                const V3 star_p = (star-com)-n*(star-com).dot(n);
                const double radius = star_p.norm();
                if (radius >= inner-eps && radius <= outer+eps) {
                    double candidate;
                    if (point_segment_2d(project(star,com,e1,e2),{0,0},0,-dtheta,
                        project(x0,com,e1,e2),project(x1,com,e1,e2),candidate)) { best = candidate; found = true; }
                }
            }
        } else if (fabs(h0-h2) < eps) {
            const V2 a = project(x0,com,e1,e2), b = project(x1,com,e1,e2);
            const V2 c = project(x2,com,e1,e2), d = project(x3,com,e1,e2);
            const V2 points[4] = {a,b,c,d};
            for (int i = 0; i < 4; ++i) {
                double candidate;
                const bool forward = i < 2;
                if (point_segment_2d(points[i],{0,0},forward ? theta_n : 0,
                    forward ? theta_n+dtheta : -dtheta,forward ? c : a,forward ? d : b,candidate)
                    && candidate < best) { best = candidate; found = true; }
            }
        }
    }
    if (!found) return false;
    s = best; return true;
}

} // namespace rotation_detail

IPC_RCCD_HD inline RigidRotationResult rigid_rotation_ccd(const RigidRotationQuery& q) {
    using namespace rotation_detail;
    RigidRotationResult result;
    if (q.extra_turns != 0) { result.status = RigidRotationStatus::UnsupportedWinding; return result; }
    for (int i = 0; i < 4; ++i) {
        if (!finite(q.current[i]) || !finite(q.target[i])) { result.status = RigidRotationStatus::InvalidInput; return result; }
        // This branch-heavy CPU algorithm has no scaled-arithmetic fallback.
        // Reject extreme magnitudes before finite inputs can overflow an
        // intermediate discriminant and be misclassified as a non-collision.
        for (int k = 0; k < 3; ++k) if (!finite(q.positions[i][k]) || !finite(q.com[k]) ||
            fabs(q.positions[i][k]) > 1e30 || fabs(q.com[k]) > 1e30) {
            result.status = RigidRotationStatus::InvalidInput; return result;
        }
    }
    double current_norm = 0, target_norm = 0;
    for (int k = 0; k < 4; ++k) { current_norm += q.current[k]*q.current[k]; target_norm += q.target[k]*q.target[k]; }
    if (!finite(current_norm) || !finite(target_norm) || fabs(current_norm-1)>1e-8 || fabs(target_norm-1)>1e-8) {
        result.status = RigidRotationStatus::InvalidInput; return result;
    }
    double relative_q[4]; relative(q.target,q.current,relative_q);
    // Raw CPU kernels treat this as negligible even on a nearly full turn.
    // Fail closed here instead of allowing that ambiguity to authorize motion.
    if (relative_q[0] < 0 && vec(relative_q+1).norm() < 1e-10) {
        result.status = RigidRotationStatus::AmbiguousFullTurn; return result;
    }
    result.collision = q.vertex_face
        ? point_triangle(vec(q.positions[0]),vec(q.com),q.target,q.current,vec(q.positions[1]),vec(q.positions[2]),vec(q.positions[3]),result.time)
        : segment_segment(vec(q.positions[0]),vec(q.positions[1]),vec(q.com),q.target,q.current,vec(q.positions[2]),vec(q.positions[3]),result.time);
    if (!finite(result.time)) result.status = RigidRotationStatus::InvalidInput;
    return result;
}

IPC_RCCD_HD inline RigidRotationStatus interpolate_rigid_orientation(const double* q_current,
    const double* q_target, double alpha, double* out) {
    using namespace rotation_detail;
    if (!finite(alpha) || alpha < 0 || alpha > 1) return RigidRotationStatus::InvalidInput;
    double current[4], target[4], relative_q[4];
    if (!normalize(q_current,current) || !normalize(q_target,target)) return RigidRotationStatus::InvalidInput;
    if (alpha == 0 || alpha == 1) {
        for (int k = 0; k < 4; ++k) out[k] = alpha == 0 ? current[k] : target[k];
        return RigidRotationStatus::Success;
    }
    relative(target,current,relative_q);
    if (!normalize(relative_q,relative_q)) return RigidRotationStatus::InvalidInput;
    const double sine = vec(relative_q+1).norm();
    if (relative_q[0] < 0 && sine <= 64.0*DBL_EPSILON) return RigidRotationStatus::AmbiguousFullTurn;
    if (relative_q[0] >= 0 && sine < 1e-12) {
        for (int k = 0; k < 4; ++k) out[k] = (1-alpha)*current[k]+alpha*target[k];
    } else {
        const double factor = 2.0*alpha*atan2(sine,relative_q[0])/sine;
        const double vector[3] = {factor*relative_q[1],factor*relative_q[2],factor*relative_q[3]};
        rigid_orientation(current,vector,1.0,out);
    }
    return normalize(out,out) ? RigidRotationStatus::Success : RigidRotationStatus::InvalidInput;
}

IPC_RCCD_HD inline RigidRotationStatus rigid_angular_velocity_from_orientation(const double* q,
    const double* previous_q, double dt, double* omega) {
    using namespace rotation_detail;
    if (!finite(dt) || dt <= 0) return RigidRotationStatus::InvalidInput;
    double target[4], previous[4], relative_q[4];
    if (!normalize(q,target) || !normalize(previous_q,previous)) return RigidRotationStatus::InvalidInput;
    relative(target,previous,relative_q);
    if (!normalize(relative_q,relative_q)) return RigidRotationStatus::InvalidInput;
    const double sine = vec(relative_q+1).norm();
    if (relative_q[0] < 0 && sine <= 64.0*DBL_EPSILON) return RigidRotationStatus::AmbiguousFullTurn;
    const double factor = relative_q[0] >= 0 && sine < 1e-12 ? 2.0/dt : 2.0*atan2(sine,relative_q[0])/(dt*sine);
    for (int k = 0; k < 3; ++k) omega[k] = factor*relative_q[k+1];
    return RigidRotationStatus::Success;
}

IPC_RCCD_HD inline RigidRotationStatus bound_rigid_orientation(const double* q_anchor,
    const double* q_current, const double* q_target, double theta_bound, double* out) {
    using namespace rotation_detail;
    if (!finite(theta_bound)) return RigidRotationStatus::InvalidInput;
    double current[4],target[4],rel[4];
    if (!normalize(q_current,current) || !normalize(q_target,target)) return RigidRotationStatus::InvalidInput;
    relative(target,current,rel);
    if (!normalize(rel,rel)) return RigidRotationStatus::InvalidInput;
    const double sine = vec(rel+1).norm();
    if (rel[0] < 0 && sine <= 64.0*DBL_EPSILON) return RigidRotationStatus::AmbiguousFullTurn;
    if (theta_bound >= pi || (rel[0] >= 0 && sine == 0)) {
        for (int k = 0; k < 4; ++k) out[k] = target[k];
        return RigidRotationStatus::Success;
    }
    double anchor[4], tangent[4];
    if (!normalize(q_anchor,anchor)) return RigidRotationStatus::InvalidInput;
    const double axis_q[4] = {0,rel[1]/sine,rel[2]/sine,rel[3]/sine};
    rigid_quaternion_multiply(axis_q,current,tangent);
    double anchor_current = 0, anchor_tangent = 0;
    for (int k = 0; k < 4; ++k) { anchor_current += anchor[k]*current[k]; anchor_tangent += anchor[k]*tangent[k]; }
    const double a = fabs(anchor_current), b = (anchor_current < 0 ? -1.0 : 1.0)*anchor_tangent;
    const double cap_cosine = cos(.5*maximum(0,theta_bound));
    if (a+1e-12 < cap_cosine) return RigidRotationStatus::OutsideCap;
    const double amplitude = hypot(a,b), phase = atan2(b,a);
    const double boundary_sine = sqrt(maximum(0,amplitude*amplitude-cap_cosine*cap_cosine));
    const double first_exit = maximum(0,phase+atan2(boundary_sine,cap_cosine));
    const double half_arc = atan2(sine,rel[0]);
    const double alpha = first_exit >= half_arc ? 1 : clamp(first_exit/half_arc,0,1);
    return interpolate_rigid_orientation(current,target,alpha,out);
}

// Same spherical-cap envelope as parallel_helper.cpp; translation padding is
// applied by the caller. This encloses the full arc, not only its endpoints.
IPC_RCCD_HD inline RigidRotationStatus rigid_spherical_cap_box(const double* com,
    const double* orientation, const double* X, double theta_bound, BpAabb& box) {
    using namespace rotation_detail;
    if (!finite(theta_bound)) return RigidRotationStatus::InvalidInput;
    for (int k = 0; k < 3; ++k) if (!finite(com[k]) || !finite(X[k])) return RigidRotationStatus::InvalidInput;
    double q[4]; if (!normalize(orientation,q)) return RigidRotationStatus::InvalidInput;
    const double zero[3] = {0,0,0}; double offset[3];
    rigid_world_position(X,zero,q,offset);
    const double radius = vec(offset).norm();
    if (!finite(radius)) return RigidRotationStatus::InvalidInput;
    const double extent = maximum(0,theta_bound), cosine = cos(extent), sine = sin(extent);
    for (int k = 0; k < 3; ++k) {
        if (radius == 0) { box.lo[k] = com[k]; box.hi[k] = com[k]; continue; }
        if (extent >= pi) { box.lo[k] = com[k]-radius; box.hi[k] = com[k]+radius; continue; }
        const double d = clamp(offset[k]/radius,-1,1), tangent = sqrt(maximum(0,1-d*d));
        const double hi = d >= cosine ? 1.0 : d*cosine+tangent*sine;
        const double lo = -d >= cosine ? -1.0 : d*cosine-tangent*sine;
        box.lo[k] = com[k]+radius*lo; box.hi[k] = com[k]+radius*hi;
    }
    return RigidRotationStatus::Success;
}

} // namespace ipc_gpu::resident

#undef IPC_RCCD_HD
