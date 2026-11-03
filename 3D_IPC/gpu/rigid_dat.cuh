#pragma once

#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#define IPC_RIGID_DAT_HD __host__ __device__
#else
#define IPC_RIGID_DAT_HD
#endif

namespace ipc_gpu::solver::rigid_dat_detail {

// Device version of rigid_dat_trajectory.h. All computations are local to the
// calling thread. Compile without floating-point contraction or fast-math.
// q0 is scalar first and retains its supplied (nearly unit) norm. This is the
// angular-coordinate curve, not interpolation between endpoint quaternions:
// c + t*dc + exp(skew(dt*(omega+t*dw))) * (q0*(0,X)*conj(q0)).
struct Trajectory {
    double center[3]{}, delta_center[3]{}, omega[3]{}, delta_omega[3]{}, reference[3]{};
    double q0[4]{1,0,0,0};
    double dt=1;
};
struct Box { double lo[3]{}, hi[3]{}; };
struct Interval { double lo,hi; };
// Immutable per-node preparation, shared by radius and all pair-plane queries.
struct PreparedTrajectory { Interval v0[3]{}; Box initial_box{}; };

enum class Status : int {
    Ok=0, InvalidInput, Nonfinite, AngleRange, InitialSideUnproved,
    InitialRadiusUnproved, DepthExhausted, BudgetExhausted
};
struct Result { Status status=Status::Ok; double factor=0; };
// A depth/budget limit may reduce progress, but its factor still certifies the
// entire prefix. Fatal errors carry zero and must be reported to the caller.
IPC_RIGID_DAT_HD inline bool is_fatal(Status status) {
    return status!=Status::Ok && status!=Status::DepthExhausted &&
        status!=Status::BudgetExhausted;
}
inline constexpr int kMaxDepth=64;
inline constexpr int kMaxBoxBudget=4000000;
inline constexpr int kTaylorOrder=18;
inline constexpr int kMaxScalingExponent=60;
inline constexpr double kTaylorRemainder=1e-33;
inline constexpr double kQuaternionTolerance=1e-8;

IPC_RIGID_DAT_HD inline bool finite(double x) { return x<=DBL_MAX && x>=-DBL_MAX; }
IPC_RIGID_DAT_HD inline double round_down(double x) {
    if(x==INFINITY)return DBL_MAX;
    return ::nextafter(x,-double(INFINITY));
}
IPC_RIGID_DAT_HD inline double round_up(double x) {
    if(x==-INFINITY)return -DBL_MAX;
    return ::nextafter(x,double(INFINITY));
}
IPC_RIGID_DAT_HD inline Interval make_interval(double x) { return {x,x}; }
IPC_RIGID_DAT_HD inline bool is_zero(Interval x) { return x.lo==0 && x.hi==0; }
IPC_RIGID_DAT_HD inline bool all_zero(const double* x) {
    return x[0]==0 && x[1]==0 && x[2]==0;
}
IPC_RIGID_DAT_HD inline Interval add(Interval a,Interval b) {
    if(is_zero(a))return b;
    if(is_zero(b))return a;
    double lo=a.lo+b.lo,hi=a.hi+b.hi;
    if(lo!=lo)lo=-INFINITY;
    if(hi!=hi)hi=INFINITY;
    return {round_down(lo),round_up(hi)};
}
IPC_RIGID_DAT_HD inline Interval sub(Interval a,Interval b) {
    if(is_zero(b))return a;
    double lo=a.lo-b.hi,hi=a.hi-b.lo;
    if(lo!=lo)lo=-INFINITY;
    if(hi!=hi)hi=INFINITY;
    return {round_down(lo),round_up(hi)};
}
IPC_RIGID_DAT_HD inline double safe_product(double a,double b) {
    return a==0 || b==0 ? 0 : a*b;
}
IPC_RIGID_DAT_HD inline Interval mul(Interval a,Interval b) {
    if(is_zero(a)||is_zero(b))return make_interval(0);
    const double products[4]={safe_product(a.lo,b.lo),safe_product(a.lo,b.hi),
        safe_product(a.hi,b.lo),safe_product(a.hi,b.hi)};
    double lo=INFINITY,hi=-INFINITY;
    for(int i=0;i<4;++i) {
        if(products[i]!=products[i])return {-INFINITY,INFINITY};
        if(products[i]<lo)lo=products[i];
        if(products[i]>hi)hi=products[i];
    }
    return {round_down(lo),round_up(hi)};
}
IPC_RIGID_DAT_HD inline Interval negate(Interval a) { return {-a.hi,-a.lo}; }
IPC_RIGID_DAT_HD inline Interval scale_pow2(Interval a,int exponent) {
    if(is_zero(a)||exponent==0)return a;
    return {round_down(::ldexp(a.lo,-exponent)),round_up(::ldexp(a.hi,-exponent))};
}
IPC_RIGID_DAT_HD inline double max_abs(Interval a) {
    if(a.lo!=a.lo||a.hi!=a.hi)return INFINITY;
    const double lo=::fabs(a.lo),hi=::fabs(a.hi);
    return lo>hi ? lo : hi;
}
IPC_RIGID_DAT_HD inline double rotation_norm_upper(const Interval* theta) {
    double sum=0;
    for(int i=0;i<3;++i) {
        const double component=max_abs(theta[i]);
        if(!finite(component))return INFINITY;
        sum=round_up(sum+component);
    }
    return sum;
}
IPC_RIGID_DAT_HD inline Status scaling_exponent(double rho,int& scale) {
    if(!finite(rho)||rho<0)return Status::Nonfinite;
    if(rho==0){scale=0;return Status::Ok;}
    int exponent=0;
    (void)::frexp(rho,&exponent);
    scale=exponent+3;
    if(scale<0)scale=0;
    return scale>kMaxScalingExponent ? Status::AngleRange : Status::Ok;
}

struct IntervalMatrix { Interval m[3][3]; };
IPC_RIGID_DAT_HD inline IntervalMatrix identity_matrix() {
    IntervalMatrix result{};
    for(int i=0;i<3;++i)result.m[i][i]=make_interval(1);
    return result;
}
IPC_RIGID_DAT_HD inline IntervalMatrix matrix_multiply(const IntervalMatrix& a,
    const IntervalMatrix& b) {
    IntervalMatrix result{};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        for(int k=0;k<3;++k)
            result.m[i][j]=add(result.m[i][j],mul(a.m[i][k],b.m[k][j]));
    return result;
}

// Scaling uses ||skew(theta)||_2=||theta||_2<=sum_i |theta_i| with
// upward-rounded component maxima. The Taylor tail at ||B||<=1/8 is bounded by
// (1/8)^19/19!/(1-1/(8*20)) < 6e-35 < 1e-33. Thus the added remainder is a
// proved enclosure, not a tolerance. Interval squaring preserves the enclosure
// for every theta in the box, including full turns and changing rotation axes.
IPC_RIGID_DAT_HD inline Status rotation_action_interval(const Interval* theta,
    const Interval* v,Interval* result) {
    if(is_zero(v[0])&&is_zero(v[1])&&is_zero(v[2])) {
        for(int i=0;i<3;++i)result[i]=v[i];
        return Status::Ok;
    }
    const double rho=rotation_norm_upper(theta);
    int scale=0;
    Status status=scaling_exponent(rho,scale);
    if(status!=Status::Ok)return status;
    if(rho==0) {
        for(int i=0;i<3;++i)result[i]=v[i];
        return Status::Ok;
    }
    Interval scaled[3];
    for(;;) {
        for(int i=0;i<3;++i)scaled[i]=scale_pow2(theta[i],scale);
        if(rotation_norm_upper(scaled)<=.125)break;
        if(++scale>kMaxScalingExponent)return Status::AngleRange;
    }
    IntervalMatrix b{};
    b.m[0][1]=negate(scaled[2]);b.m[0][2]=scaled[1];
    b.m[1][0]=scaled[2];b.m[1][2]=negate(scaled[0]);
    b.m[2][0]=negate(scaled[1]);b.m[2][1]=scaled[0];
    IntervalMatrix term=identity_matrix(),poly{};
    double factorial=1;
    for(int k=0;k<=kTaylorOrder;++k) {
        // All k! through 18! are exactly representable binary64 integers.
        // A one-ulp outward interval encloses each correctly rounded division.
        Interval coefficient=make_interval(1);
        if(k) {
            factorial*=double(k);
            const double reciprocal=1/factorial;
            coefficient={round_down(reciprocal),round_up(reciprocal)};
        }
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            poly.m[i][j]=add(poly.m[i][j],mul(coefficient,term.m[i][j]));
        if(k<kTaylorOrder)term=matrix_multiply(term,b);
    }
    const Interval tail{-kTaylorRemainder,kTaylorRemainder};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)poly.m[i][j]=add(poly.m[i][j],tail);
    for(int k=0;k<scale;++k)poly=matrix_multiply(poly,poly);
    for(int i=0;i<3;++i) {
        result[i]=make_interval(0);
        for(int j=0;j<3;++j)result[i]=add(result[i],mul(poly.m[i][j],v[j]));
    }
    return Status::Ok;
}

IPC_RIGID_DAT_HD inline void quaternion_apply_interval(const double* q,const double* v,
    Interval* out) {
    const Interval w=make_interval(q[0]),x=make_interval(q[1]),
        y=make_interval(q[2]),z=make_interval(q[3]),two=make_interval(2);
    const Interval ww=mul(w,w),xx=mul(x,x),yy=mul(y,y),zz=mul(z,z),
        xy=mul(x,y),xz=mul(x,z),yz=mul(y,z),wx=mul(w,x),wy=mul(w,y),wz=mul(w,z);
    const Interval r[3][3]={
        {sub(add(ww,xx),add(yy,zz)),mul(two,sub(xy,wz)),mul(two,add(xz,wy))},
        {mul(two,add(xy,wz)),sub(add(ww,yy),add(xx,zz)),mul(two,sub(yz,wx))},
        {mul(two,sub(xz,wy)),mul(two,add(yz,wx)),sub(add(ww,zz),add(xx,yy))}};
    for(int i=0;i<3;++i) {
        out[i]=make_interval(0);
        for(int j=0;j<3;++j)out[i]=add(out[i],mul(r[i][j],make_interval(v[j])));
    }
}
IPC_RIGID_DAT_HD inline Status validate(const Trajectory& path) {
    for(int i=0;i<3;++i)
        if(!finite(path.center[i])||!finite(path.delta_center[i])||
           !finite(path.omega[i])||!finite(path.delta_omega[i])||
           !finite(path.reference[i]))return Status::InvalidInput;
    for(int i=0;i<4;++i)if(!finite(path.q0[i]))return Status::InvalidInput;
    if(!finite(path.dt))return Status::InvalidInput;
    const double norm=::hypot(::hypot(path.q0[0],path.q0[1]),
        ::hypot(path.q0[2],path.q0[3]));
    return finite(norm)&&::fabs(norm-1)<=kQuaternionTolerance ? Status::Ok : Status::InvalidInput;
}
IPC_RIGID_DAT_HD inline Status bounds_core(const Trajectory& path,double a,double b,
    const Interval* v0,Box& box) {
    const Interval time{a,b};
    Interval theta[3],rotated[3];
    for(int i=0;i<3;++i) {
        if(!finite(v0[i].lo)||!finite(v0[i].hi))return Status::Nonfinite;
        theta[i]=mul(make_interval(path.dt),
            add(make_interval(path.omega[i]),mul(time,make_interval(path.delta_omega[i]))));
    }
    Status status=rotation_action_interval(theta,v0,rotated);
    if(status!=Status::Ok)return status;
    for(int i=0;i<3;++i) {
        const Interval center=add(make_interval(path.center[i]),
            mul(time,make_interval(path.delta_center[i])));
        const Interval value=add(center,rotated[i]);
        if(!finite(value.lo)||!finite(value.hi))return Status::Nonfinite;
        box.lo[i]=value.lo;box.hi[i]=value.hi;
    }
    return Status::Ok;
}
IPC_RIGID_DAT_HD inline Status trajectory_bounds(const Trajectory& path,double a,double b,
    Box& box) {
    const Status status=validate(path);
    if(status!=Status::Ok)return status;
    if(!finite(a)||!finite(b)||a>b)return Status::InvalidInput;
    Interval v0[3];quaternion_apply_interval(path.q0,path.reference,v0);
    return bounds_core(path,a,b,v0,box);
}
IPC_RIGID_DAT_HD inline Status prepare_trajectory(const Trajectory& path,
    PreparedTrajectory& prepared) {
    const auto status=validate(path);
    if(status!=Status::Ok)return status;
    quaternion_apply_interval(path.q0,path.reference,prepared.v0);
    return bounds_core(path,0,0,prepared.v0,prepared.initial_box);
}
IPC_RIGID_DAT_HD inline Interval dot_box(const double* normal,const Box& box) {
    Interval sum=make_interval(0);
    for(int i=0;i<3;++i)sum=add(sum,mul(make_interval(normal[i]),{box.lo[i],box.hi[i]}));
    return sum;
}
IPC_RIGID_DAT_HD inline Interval dot_scalar_interval(const double* a,const double* b) {
    Interval sum=make_interval(0);
    for(int i=0;i<3;++i)sum=add(sum,mul(make_interval(a[i]),make_interval(b[i])));
    return sum;
}
IPC_RIGID_DAT_HD inline Interval plane_projection(const Box& box,const double* normal,
    const double* point) {
    Interval sum=make_interval(0);
    for(int i=0;i<3;++i)sum=add(sum,mul(make_interval(normal[i]),
        sub({box.lo[i],box.hi[i]},make_interval(point[i]))));
    return sum;
}
// Outward upper bound on the Euclidean distance of every point in box from
// the represented query point. The initial pose/query discrepancy is included.
IPC_RIGID_DAT_HD inline double norm_upper(const Box& box,const double* query) {
    Interval squared=make_interval(0);
    for(int i=0;i<3;++i) {
        if(!finite(query[i]))return INFINITY;
        const double component=max_abs(sub({box.lo[i],box.hi[i]},make_interval(query[i])));
        squared=add(squared,mul(make_interval(component),make_interval(component)));
    }
    if(squared.hi==0)return 0; // sqrt(0) is exact, including a zero-radius sphere.
    return finite(squared.hi)&&squared.hi>=0 ? round_up(::sqrt(squared.hi)) : INFINITY;
}

// Proposal evaluator only. The certificate above does not rely on libm sin,
// cos or on a floating endpoint matching the mathematical curve exactly.
IPC_RIGID_DAT_HD inline Status evaluate(const Trajectory& path,double t,double* out) {
    Status status=validate(path);
    if(status!=Status::Ok)return status;
    if(!finite(t))return Status::InvalidInput;
    const double* q=path.q0;
    const double w=q[0],x=q[1],y=q[2],z=q[3];
    const double ww=w*w,xx=x*x,yy=y*y,zz=z*z,xy=x*y,xz=x*z,yz=y*z,wx=w*x,wy=w*y,wz=w*z;
    const double* v=path.reference;
    const double u[3]={
        (ww+xx-yy-zz)*v[0]+2*(xy-wz)*v[1]+2*(xz+wy)*v[2],
        2*(xy+wz)*v[0]+(ww-xx+yy-zz)*v[1]+2*(yz-wx)*v[2],
        2*(xz-wy)*v[0]+2*(yz+wx)*v[1]+(ww-xx-yy+zz)*v[2]};
    double theta[3]{},upper=0;
    for(int i=0;i<3;++i) {
        if(!finite(u[i]))return Status::Nonfinite;
        if(path.dt!=0)theta[i]=path.dt*(path.omega[i]+t*path.delta_omega[i]);
        if(!finite(theta[i]))return Status::Nonfinite;
        upper=round_up(upper+::fabs(theta[i]));
    }
    int scale=0;status=scaling_exponent(upper,scale);
    if(status!=Status::Ok)return status;
    const double angle=::hypot(theta[0],::hypot(theta[1],theta[2]));
    double first[3],second[3];
    for(int i=0;i<3;++i)first[i]=theta[(i+1)%3]*u[(i+2)%3]-theta[(i+2)%3]*u[(i+1)%3];
    for(int i=0;i<3;++i) {
        second[i]=theta[(i+1)%3]*first[(i+2)%3]-theta[(i+2)%3]*first[(i+1)%3];
        if(!finite(first[i])||!finite(second[i]))return Status::Nonfinite;
    }
    double c1,c2;
    if(angle<1e-4) {
        const double a2=theta[0]*theta[0]+theta[1]*theta[1]+theta[2]*theta[2];
        c1=1+a2*(-1.0/6+a2*(1.0/120+a2*(-1.0/5040)));
        c2=.5+a2*(-1.0/24+a2*(1.0/720+a2*(-1.0/40320)));
    } else {
        c1=::sin(angle)/angle;c2=(1-::cos(angle))/(angle*angle);
    }
    for(int i=0;i<3;++i) {
        const double rotated=u[i]+c1*first[i]+c2*second[i];
        out[i]=path.center[i]+t*path.delta_center[i]+rotated;
        if(!finite(out[i]))return Status::Nonfinite;
    }
    return Status::Ok;
}

struct PrefixScan {
    const Trajectory* path;
    Interval v0[3];
    double normal[3],query[3];
    double radius;
    int max_depth,budget;
    double last_safe=0;
    bool radial;

    IPC_RIGID_DAT_HD Status interval_safe(double a,double b,bool& safe) const {
        Box box{};
        const Status status=bounds_core(*path,a,b,v0,box);
        if(status!=Status::Ok)return status;
        safe=radial ? norm_upper(box,query)<=radius :
            plane_projection(box,normal,query).lo>0;
        return Status::Ok;
    }
    // Iterative left-before-right DFS. Only certified leaves advance
    // last_safe; the first unresolved leaf stops progress. The fixed stack is
    // bounded by max_depth+1, and no device recursion or heap is required.
    IPC_RIGID_DAT_HD Status scan(double a,double b) {
        struct Node { double a,b;int depth; };
        Node pending[kMaxDepth+1];
        int count=1;pending[0]={a,b,0};
        while(count) {
            if(budget<=0)return Status::BudgetExhausted;
            --budget;
            const Node current=pending[--count];
            bool safe=false;
            const Status status=interval_safe(current.a,current.b,safe);
            if(status!=Status::Ok)return status;
            if(safe){last_safe=current.b;continue;}
            if(current.depth>=max_depth)return Status::DepthExhausted;
            const double mid=current.a+.5*(current.b-current.a);
            if(!(mid>current.a&&mid<current.b))return Status::DepthExhausted;
            pending[count++]={mid,current.b,current.depth+1};
            pending[count++]={current.a,mid,current.depth+1};
        }
        return Status::Ok;
    }
};
IPC_RIGID_DAT_HD inline bool no_motion(const Trajectory& path) {
    return all_zero(path.delta_center) &&
        (path.dt==0||all_zero(path.delta_omega)||all_zero(path.reference));
}
IPC_RIGID_DAT_HD inline bool valid_budget(int max_depth,int budget) {
    return max_depth>=1&&max_depth<=kMaxDepth&&budget>=1&&budget<=kMaxBoxBudget;
}

// normal points into the *intended* initial half-space; it is never silently
// reversed based on a differently rounded pose. Gamma is applied exactly once
// to a limited certified prefix. A completely certified path returns 1.
IPC_RIGID_DAT_HD inline Result plane_step(const Trajectory& path,const double* normal,
    const double* plane_point,double gamma,int max_depth=40,int box_budget=4096,
    const PreparedTrajectory* prepared=nullptr) {
    Status status=validate(path);
    if(status!=Status::Ok)return {status,0};
    if(!finite(gamma)||gamma<0||gamma>1||!valid_budget(max_depth,box_budget))
        return {Status::InvalidInput,0};
    for(int i=0;i<3;++i)if(!finite(normal[i])||!finite(plane_point[i]))return {Status::InvalidInput,0};
    const double length=::hypot(normal[0],::hypot(normal[1],normal[2]));
    if(!(length>0)||!finite(length))return {Status::InvalidInput,0};
    if(gamma==0)return {Status::Ok,0};
    PrefixScan scan{};
    scan.path=&path;scan.max_depth=max_depth;scan.budget=box_budget;
    if(prepared)for(int i=0;i<3;++i)scan.v0[i]=prepared->v0[i];
    else quaternion_apply_interval(path.q0,path.reference,scan.v0);
    for(int i=0;i<3;++i){scan.normal[i]=normal[i];scan.query[i]=plane_point[i];}
    Box initial{};
    if(prepared)initial=prepared->initial_box;
    else status=bounds_core(path,0,0,scan.v0,initial);
    if(status!=Status::Ok)return {status,0};
    if(!(plane_projection(initial,normal,plane_point).lo>0))
        return {Status::InitialSideUnproved,0};
    if(no_motion(path))return {Status::Ok,1};
    // Paper sampling proposes a limit; it never certifies the sampled chord.
    double limit=1;
    for(int i=1;i<=8;++i) {
        double proposal[3];status=evaluate(path,double(i)/8,proposal);
        if(status!=Status::Ok)return {status,0};
        const double distance=(normal[0]*(proposal[0]-plane_point[0])+
            normal[1]*(proposal[1]-plane_point[1]))+normal[2]*(proposal[2]-plane_point[2]);
        if(distance<=0){limit=double(i)/8;break;}
    }
    status=scan.scan(0,limit);
    if(status==Status::Ok&&limit<1)status=scan.scan(limit,1);
    if(is_fatal(status))return {status,0};
    if(status==Status::Ok)return {status,1};
    if(!(scan.last_safe>0))return {status,0};
    double relaxed=gamma*scan.last_safe;
    if(!(relaxed<scan.last_safe))relaxed=::nextafter(scan.last_safe,0.0);
    return {status,relaxed>0 ? relaxed : 0};
}

// The sphere is centered on the represented position used by the neighbor
// query, rather than a newly evaluated phi(0). This accounts for the complete
// pose-rounding offset directly. Radius already contains DAT's gamma; this
// routine never applies another relaxation.
IPC_RIGID_DAT_HD inline Result radius_step(const Trajectory& path,const double* query_position,
    double radius,int max_depth=40,int box_budget=4096,
    const PreparedTrajectory* prepared=nullptr) {
    Status status=validate(path);
    if(status!=Status::Ok)return {status,0};
    if(!finite(radius)||radius<0||!valid_budget(max_depth,box_budget))return {Status::InvalidInput,0};
    PrefixScan scan{};
    scan.path=&path;scan.max_depth=max_depth;scan.budget=box_budget;scan.radius=radius;scan.radial=true;
    for(int i=0;i<3;++i) {
        if(!finite(query_position[i]))return {Status::InvalidInput,0};
        scan.query[i]=query_position[i];
    }
    if(prepared)for(int i=0;i<3;++i)scan.v0[i]=prepared->v0[i];
    else quaternion_apply_interval(path.q0,path.reference,scan.v0);
    Box initial{};
    if(prepared)initial=prepared->initial_box;
    else status=bounds_core(path,0,0,scan.v0,initial);
    if(status!=Status::Ok)return {status,0};
    if(!(norm_upper(initial,query_position)<=radius))return {Status::InitialRadiusUnproved,0};
    if(no_motion(path))return {Status::Ok,1};
    status=scan.scan(0,1);
    if(is_fatal(status))return {status,0};
    return {status,status==Status::Ok ? 1 : scan.last_safe};
}

} // namespace ipc_gpu::solver::rigid_dat_detail

#undef IPC_RIGID_DAT_HD
