#pragma once

#include "rigid.cuh"
#include "broadphase_types.h"
#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#define IPC_RCCD_HD __host__ __device__
#else
#define IPC_RCCD_HD
#endif

namespace ipc_gpu::solver {

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
    // The CPU rigid_rotation_epsilon is zero: a small nonzero rotation can
    // still reach contact and must not be treated as an unchanged pose.
    if (norm == 0.0) return false;
    const V3 n = v/norm, dx = x-com;
    const V3 parallel = n*dx.dot(n), perpendicular = dx-parallel;
    if (perpendicular.norm() < eps) return false;
    const double dtheta = 2.0*atan2(norm,qr[0]);
    if (dtheta == 0.0) return false;
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

// Closed-form counterpart of ccd_rotation_family_detail. CUDA long double has
// no extra precision, so all bounds explicitly use binary64 roundoff. Only the
// legacy near-zero radial-polynomial branch enters this finite-primitive repair.
struct FamilyRoots { double value[2]{}; int count=0; };
IPC_RCCD_HD inline FamilyRoots family_roots(double a,double b,double c) {
    FamilyRoots result;
    if(a==0) {
        if(b!=0)result.value[result.count++]=-c/b;
        return result;
    }
    const double bb=b*b,ac4=4*a*c;
    double disc=fma(-4*a,c,bb);
    const double error=32*DBL_EPSILON*(fabs(bb)+fabs(ac4));
    if(disc < -error)return result;
    disc=maximum(0,disc);
    const double q=-.5*(b+copysign(sqrt(disc),b));
    if(q==0)result.value[result.count++]=-b/(2*a);
    else {result.value[result.count++]=q/a;result.value[result.count++]=c/q;}
    return result;
}
struct FamilyRange {
    V3 p,v,q,e,n;
    double roundoff,arc,direction,best=INFINITY;
    IPC_RCCD_HD void consider(double w,bool zero_event=false) {
        const double parameter_error=64*DBL_EPSILON;
        if(!finite(w)||w < -parameter_error||w > 1+parameter_error)return;
        w=clamp(w,0,1);
        const V3 a=p+v*w,b=q+e*w;
        const double ar2=a.squaredNorm(),br2=b.squaredNorm();
        const double a_error=roundoff*(p.norm()+fabs(w)*v.norm());
        const double b_error=roundoff*(q.norm()+fabs(w)*e.norm());
        const double radial_error=roundoff*(ar2+br2)
            +2*(sqrt(ar2)*a_error+sqrt(br2)*b_error)
            +a_error*a_error+b_error*b_error;
        if(fabs(br2-ar2)>radial_error)return;
        if(ar2==0&&br2==0){best=0;return;}
        if(ar2==0||br2==0)return;
        const double x=a.dot(b),y=direction*n.dot(a.cross(b));
        double angle;
        if(zero_event&&x>0&&fabs(y)<=roundoff*sqrt(ar2*br2))angle=0;
        else {angle=atan2(y,x);if(angle<0)angle+=two_pi;}
        const double fraction=angle/arc;
        if(fraction<=1+1e-10)best=minimum(best,clamp(fraction,0,1));
    }
    IPC_RCCD_HD void roots(double a,double b,double c,bool zero_event=false) {
        const auto candidates=family_roots(a,b,c);
        for(int i=0;i<candidates.count;++i)consider(candidates.value[i],zero_event);
    }
};
IPC_RCCD_HD inline V3 family_moving_at(V3 p,V3 delta,double h0,double h2,
    double en,double dh,double u,int endpoint) {
    if(endpoint==0)return p;
    if(endpoint==1)return p+delta;
    return p+delta*clamp((h2+u*en-h0)/dh,0,1);
}
IPC_RCCD_HD inline bool family_range(V3 moving0,V3 moving_delta,V3 stationary0,
    V3 stationary_delta,V3 axis,double h0,double h1,double h2,double en,
    double coordinate_scale,double dtheta,double lo,double hi,double& s,int endpoint=-1) {
    const double dh=h1-h0;
    const V3 p=family_moving_at(moving0,moving_delta,h0,h2,en,dh,lo,endpoint);
    const V3 v=family_moving_at(moving0,moving_delta,h0,h2,en,dh,hi,endpoint)-p;
    const V3 q=stationary0+stationary_delta*lo,e=stationary_delta*(hi-lo);
    const double scale=maximum(maximum(p.norm(),(p+v).norm()),maximum(q.norm(),(q+e).norm()));
    if(scale==0){s=0;return true;}
    FamilyRange range{p/scale,v/scale,q/scale,e/scale,axis/axis.norm(),
        DBL_EPSILON*(128+64*coordinate_scale/scale),fabs(dtheta),dtheta<0 ? -1.0 : 1.0};
    const double fa=range.e.squaredNorm()-range.v.squaredNorm();
    const double fb=2*(range.q.dot(range.e)-range.p.dot(range.v));
    const double fc=range.q.squaredNorm()-range.p.squaredNorm();
    const double ea=range.roundoff*(range.e.squaredNorm()+range.v.squaredNorm());
    const double eb=2*range.roundoff*(range.q.norm()*range.e.norm()+range.p.norm()*range.v.norm());
    const double ec=range.roundoff*(range.q.squaredNorm()+range.p.squaredNorm());
    const bool family=fabs(fa)<=ea&&fabs(fb)<=eb&&fabs(fc)<=ec;
    range.consider(0);range.consider(1);
    if(!family) {
        // A small nonzero polynomial has only actual roots. In particular,
        // constant unequal radii do not become a swept-surface family.
        range.roots(fa,fb,fc);
    } else {
        const double kp=range.n.dot(range.p.cross(range.v));
        const double kq=range.n.dot(range.q.cross(range.e));
        range.roots(kq*range.v.squaredNorm()-kp*range.e.squaredNorm(),
            2*(kq*range.p.dot(range.v)-kp*range.q.dot(range.e)),
            kq*range.p.squaredNorm()-kp*range.q.squaredNorm());
        // Interior angle-zero and radial-axis events split wrapped branches.
        range.roots(range.n.dot(range.v.cross(range.e)),
            range.n.dot(range.p.cross(range.e)+range.v.cross(range.q)),
            range.n.dot(range.p.cross(range.q)),true);
        range.roots(range.v.squaredNorm(),2*range.p.dot(range.v),range.p.squaredNorm(),true);
        range.roots(range.e.squaredNorm(),2*range.q.dot(range.e),range.q.squaredNorm(),true);
    }
    if(!finite(range.best))return false;
    s=range.best;return true;
}
IPC_RCCD_HD inline bool swept_family(V3 moving0,V3 moving_delta,V3 stationary0,
    V3 stationary_delta,V3 axis,double h0,double h1,double h2,double en,
    double coordinate_scale,double dtheta,double& s) {
    double lo=0,hi=1,best=INFINITY,candidate=0;
    const double lower=minimum(h0,h1),upper=maximum(h0,h1);
    bool nominal=true,found=false;
    if(en==0)nominal=h2>=lower&&h2<=upper;
    else {
        double enter=(lower-h2)/en,exit=(upper-h2)/en;
        if(enter>exit){const double saved=enter;enter=exit;exit=saved;}
        lo=maximum(lo,enter);hi=minimum(hi,exit);nominal=lo<=hi;
    }
    if(nominal&&family_range(moving0,moving_delta,stationary0,stationary_delta,
            axis,h0,h1,h2,en,coordinate_scale,dtheta,lo,hi,candidate)) {
        found=true;best=candidate;
    }
    // Roundoff seams use the actual finite moving endpoint. They never extend
    // the moving edge or substitute an extrapolated infinite-line contact.
    const double height_error=64*DBL_EPSILON*(coordinate_scale
        +maximum(maximum(fabs(h0),fabs(h1)),maximum(fabs(h2),fabs(h2+en))));
    for(int endpoint=0;endpoint<2;++endpoint) {
        const double height=endpoint==0 ? h0 : h1;
        double begin=0,end=1;
        if(en==0) {if(fabs(h2-height)>height_error)continue;}
        else {
            begin=(height-height_error-h2)/en;end=(height+height_error-h2)/en;
            if(begin>end){const double saved=begin;begin=end;end=saved;}
            begin=maximum(begin,0);end=minimum(end,1);if(begin>end)continue;
        }
        if(family_range(moving0,moving_delta,stationary0,stationary_delta,
                axis,h0,h1,h2,en,coordinate_scale,dtheta,begin,end,candidate,endpoint)) {
            found=true;best=minimum(best,candidate);
        }
    }
    if(found)s=best;
    return found;
}

IPC_RCCD_HD inline bool segment_segment(V3 x0, V3 x1, V3 com,
    const double* target, const double* current, V3 x2, V3 x3, double& s) {
    constexpr double eps = 1e-10;
    s = 0.0;
    double qr[4]; relative(target,current,qr);
    const V3 v = vec(qr+1); const double norm = v.norm();
    if (norm == 0.0) return false;
    const V3 n = v/norm;
    const double h0 = (x0-com).dot(n), h1 = (x1-com).dot(n);
    const V3 p0 = (x0-com)-n*h0, p1 = (x1-com)-n*h1;
    const double r0 = p0.norm(), r1 = p1.norm();
    if (maximum(r0,r1) < eps) return false;
    const double dtheta = 2.0*atan2(norm,qr[0]);
    if (dtheta == 0.0) return false;
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
        if(fabs(A)<eps&&fabs(B)<eps&&fabs(C)<eps) {
            const V3 points[5]={x0,x1,x2,x3,com};double coordinate_scale=0;
            for(int i=0;i<5;++i)coordinate_scale=maximum(coordinate_scale,
                maximum(maximum(fabs(points[i].x),fabs(points[i].y)),fabs(points[i].z)));
            return swept_family(p0,dp,q2p,ep,n,h0,h1,h2,en,coordinate_scale,dtheta,s);
        }
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
            const V3 star_p = q2p+ep*u;
            // Recover the relative angle directly, as on the CPU. Subtracting
            // two world-scale angles loses near-start impacts.
            double angle = atan2(n.dot(moving.cross(star_p)),moving.dot(star_p));
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

} // namespace ipc_gpu::solver

#undef IPC_RCCD_HD
