#include "rigid_pipeline.h"
#include "rigid_ccd.cuh"
#include "rigid_proxy_index.h"
#include "rigid_proxy_values.cuh"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ipc_gpu::solver {
namespace {
constexpr unsigned width = 128;
struct RotationBatchContext {
    std::size_t body=0;
    int mode=0,done=0;unsigned hit=0;
    unsigned long long first_error=~0ull;
    double current[4]{},reverse[4]{},angle=0.,fraction=1.;
};
static_assert(sizeof(RotationBatchContext)==112,"bounded rotation context layout");
unsigned blocks(std::size_t n) { return static_cast<unsigned>((n+width-1)/width); }
void launched() {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(std::string("GPU rigid CUDA launch: ")+cudaGetErrorString(status));
}
void valid_dt(double dt) {
    if (!(dt > 0) || !std::isfinite(dt)) throw std::invalid_argument("GPU rigid dt must be positive and finite");
}
unsigned rotation_pair_threads() {
    const char* value=std::getenv("IPC_GPU_RIGID_ROTATION_PAIR_THREADS");
    if(!value||!*value||std::strcmp(value,"1")==0)return 1;
    for(const unsigned count:{32u,64u,128u})
        if(std::strcmp(value,std::to_string(count).c_str())==0)return count;
    throw std::invalid_argument("IPC_GPU_RIGID_ROTATION_PAIR_THREADS must be 1,32,64,128");
}
bool parallel_proxy_option() {
    const char* value=std::getenv("IPC_GPU_PARALLEL_RIGID_PROXIES");
    if(!value||!*value)return true;
    if(std::strcmp(value,"0")==0)return false;
    if(std::strcmp(value,"1")==0)return true;
    throw std::invalid_argument("IPC_GPU_PARALLEL_RIGID_PROXIES must be 0 or 1");
}
__device__ bool active(const RigidScene& s,const int* colors,int color,std::size_t body) {
    return colors[s.body_blocks[body]] == color;
}
__device__ void fail(int* error,int code) { atomicCAS(error,0,code); }
__device__ bool failed(int* error) { return atomicAdd(error,0)!=0; }
__device__ double outward_neighbor(double value,bool upper) {
    // Bitwise adjacent binary64 value: avoids host-only mixed-type nextafter
    // overloads selected by INFINITY on NVCC, without relaxed constexpr.
    if (!isfinite(value)) return value;
    if (value==0.0) return __longlong_as_double(static_cast<long long>(upper ? 1ull : 0x8000000000000001ull));
    auto bits=static_cast<unsigned long long>(__double_as_longlong(value));
    if ((value>0.0)==upper) ++bits; else --bits;
    return __longlong_as_double(static_cast<long long>(bits));
}
__device__ bool finite3(const double* x) { return isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2]); }
__device__ bool finite_derivatives(const Derivatives& d) {
    for (int i=0;i<3;++i) if (!isfinite(d.gradient[i])) return false;
    for (int i=0;i<9;++i) if (!isfinite(d.hessian[i])) return false;
    return true;
}

// Same lower-triangle pivoted LDLT algorithm and diagonal pseudoinverse
// cutoff as Eigen 3.4 LDLT::solve used by the production rigid solver. This
// is a 3x3 implementation, not the deformable explicit inverse/QR solver.
// Algorithm adapted from Eigen/src/Cholesky/LDLT.h (MPL-2.0); Eigen copyright
// 2008-2010 Gael Guennebaud and Benoit Jacob. Roundoff need not be bitwise equal.
__device__ bool ldlt3(const double* H,const double* g,double* x) {
    double a[3][3],rhs[3]; int swaps[3]={0,1,2}; bool zero_seen=false;
    for (int i=0;i<3;++i) {
        if (!isfinite(g[i])) return false;
        rhs[i]=g[i];
        for (int j=0;j<3;++j) {
            if (!isfinite(H[3*i+j])) return false;
            a[i][j]=H[3*i+j];
        }
    }
    for (int k=0;k<3;++k) {
        int pivot=k;
        for (int i=k+1;i<3;++i) if (fabs(a[i][i])>fabs(a[pivot][pivot])) pivot=i;
        swaps[k]=pivot;
        if (pivot!=k) {
            for (int j=0;j<k;++j) { const double t=a[k][j]; a[k][j]=a[pivot][j]; a[pivot][j]=t; }
            for (int i=pivot+1;i<3;++i) { const double t=a[i][k]; a[i][k]=a[i][pivot]; a[i][pivot]=t; }
            const double t=a[k][k]; a[k][k]=a[pivot][pivot]; a[pivot][pivot]=t;
            for (int i=k+1;i<pivot;++i) { const double u=a[i][k]; a[i][k]=a[pivot][i]; a[pivot][i]=u; }
        }
        if (k>0) {
            double temp[3]{},diagonal=0;
            for (int j=0;j<k;++j) { temp[j]=a[j][j]*a[k][j]; diagonal+=a[k][j]*temp[j]; }
            a[k][k]-=diagonal;
            for (int i=k+1;i<3;++i) {
                double product=0;
                for (int j=0;j<k;++j) product+=a[i][j]*temp[j];
                a[i][k]-=product;
            }
        }
        const double diagonal=a[k][k];
        if (!isfinite(diagonal)) return false;
        const bool nonzero=fabs(diagonal)>0;
        if (k==0 && !nonzero) {
            for (int i=0;i<3;++i) for (int j=0;j<i;++j) if (a[i][j]!=0) return false;
            for (int i=0;i<3;++i) x[i]=0;
            return true;
        }
        if (zero_seen && nonzero) return false;
        if (!nonzero) zero_seen=true;
        for (int i=k+1;i<3;++i) {
            if (nonzero) a[i][k]/=diagonal;
            else if (a[i][k]!=0) return false;
            if (!isfinite(a[i][k])) return false;
        }
    }
    for (int k=0;k<3;++k) { const double t=rhs[k]; rhs[k]=rhs[swaps[k]]; rhs[swaps[k]]=t; }
    for (int i=0;i<3;++i) for (int j=0;j<i;++j) rhs[i]-=a[i][j]*rhs[j];
    for (int i=0;i<3;++i) rhs[i]=fabs(a[i][i])>DBL_MIN ? rhs[i]/a[i][i] : 0.0;
    for (int i=2;i>=0;--i) for (int j=i+1;j<3;++j) rhs[i]-=a[j][i]*rhs[j];
    for (int k=2;k>=0;--k) { const double t=rhs[k]; rhs[k]=rhs[swaps[k]]; rhs[swaps[k]]=t; }
    for (int i=0;i<3;++i) { x[i]=rhs[i]; if (!isfinite(x[i])) return false; }
    return true;
}

__global__ void begin(RigidScene s,RigidState x,int* error) {
    const std::size_t b=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (b>=s.bodies || failed(error)) return;
    if (!finite3(x.com[b]) || !finite3(x.velocity[b]) || !finite3(x.omega[b]) ||
        !(s.mass[b]>0) || !isfinite(s.mass[b])) { fail(error,RigidInvalidState); return; }
    double qnorm=0;
    for (int k=0;k<4;++k) qnorm+=x.orientation[b][k]*x.orientation[b][k];
    if (!isfinite(qnorm) || fabs(qnorm-1.0)>1e-8) { fail(error,RigidInvalidState); return; }
    for (int k=0;k<3;++k) {
        x.previous_com[b][k]=x.com[b][k];
        x.previous_velocity[b][k]=s.translate[b] ? x.velocity[b][k] : 0.0;
        x.previous_omega[b][k]=s.rotate[b] ? x.omega[b][k] : 0.0;
        x.omega[b][k]=0.0;
    }
    for (int k=0;k<4;++k) x.previous_orientation[b][k]=x.orientation[b][k];
}
__global__ void body_bounds(RigidScene s,RigidState x,double radius,BpAabb* boxes,double (*anchors)[4],int* error) {
    const std::size_t b=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (b>=s.bodies || failed(error)) return;
    if (anchors) for (int k=0;k<4;++k) anchors[b][k]=x.orientation[b][k];
    for (int k=0;k<3;++k) {
        const double r=s.translate[b] ? radius : 0.0;
        boxes[b].lo[k]=outward_neighbor(x.com[b][k]-r,false);
        boxes[b].hi[k]=outward_neighbor(x.com[b][k]+r,true);
        if (!isfinite(boxes[b].lo[k]) || !isfinite(boxes[b].hi[k])) fail(error,RigidInvalidBounds);
    }
}
__global__ void proxy_bounds(RigidScene s,RigidState x,std::size_t nodes,
    const BpAabb* body_boxes,BpAabb* node_boxes,double angle,
    const double (*anchors)[4],const double (*represented)[3],int* error) {
    const std::size_t node=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (node>=nodes || failed(error)) return;
    const int b=s.node_body[node]; if (b<0) return;
    if (static_cast<std::size_t>(b)>=s.bodies || !finite3(s.reference[node])) { fail(error,RigidInvalidState); return; }
    double offset[3]; const double zero[3]={0,0,0};
    const double* anchor=anchors ? anchors[b] : x.orientation[b];
    BpAabb cap;
    if (rigid_spherical_cap_box(zero,anchor,s.reference[node],s.rotate[b] ? angle : 0.0,cap)!=RigidRotationStatus::Success) {
        fail(error,RigidInvalidBounds); return;
    }
    double unit_anchor[4];
    if (!rotation_detail::normalize(anchor,unit_anchor)) { fail(error,RigidInvalidBounds); return; }
    rigid_world_position(s.reference[node],zero,unit_anchor,offset);
    const double radius=rotation_detail::vec(offset).norm();
    double discrepancy=0;
    if (represented) {
        if (!finite3(represented[node])) { fail(error,RigidInvalidBounds); return; }
        double difference[3];
        for (int k=0;k<3;++k) difference[k]=represented[node][k]-(x.com[b][k]+offset[k]);
        discrepancy=hypot(difference[0],hypot(difference[1],difference[2]));
    }
    if (!isfinite(radius) || !isfinite(discrepancy) || !finite3(offset)) { fail(error,RigidInvalidBounds); return; }
    for (int k=0;k<3;++k) {
        // The discrepancy norm encloses rotation of the represented offset,
        // not merely its current endpoint. Round outward after Minkowski sum.
        const double margin=discrepancy+64.0*DBL_EPSILON*(fabs(x.com[b][k])+radius+1.0);
        node_boxes[node].lo[k]=outward_neighbor(body_boxes[b].lo[k]+cap.lo[k]-margin,false);
        node_boxes[node].hi[k]=outward_neighbor(body_boxes[b].hi[k]+cap.hi[k]+margin,true);
        if (represented) {
            node_boxes[node].lo[k]=fmin(node_boxes[node].lo[k],represented[node][k]);
            node_boxes[node].hi[k]=fmax(node_boxes[node].hi[k],represented[node][k]);
        }
        if (!isfinite(node_boxes[node].lo[k]) || !isfinite(node_boxes[node].hi[k])) fail(error,RigidInvalidBounds);
    }
}
__global__ void inertia(RigidScene s,RigidState x,RigidWorkspace w,const int* colors,
    int color,bool orientation,double dt,double gravity,int* error) {
    const std::size_t b=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (b>=s.bodies || !active(s,colors,color,b) || failed(error)) return;
    w.systems[b]={};
    if (!(orientation ? s.rotate[b] : s.translate[b])) return;
    RigidInertia p{};
    if (orientation) p=build_rigid_inertia(x.previous_orientation[b],x.previous_omega[b],x.omega[b],dt,s.second_moment[b]);
    else {
        p.dt=dt; p.mass=s.mass[b]; p.gravity=gravity;
        for (int k=0;k<3;++k) { p.position[k]=x.com[b][k]; p.previous_position[k]=x.previous_com[b][k]; p.previous_velocity[k]=x.previous_velocity[b][k]; }
    }
    const auto result=evaluate_rigid_inertia(p);
    if (!finite_derivatives(result)) { fail(error,RigidInvalidDerivative); return; }
    w.systems[b]=result;
}
__global__ void propose(RigidScene s,RigidState x,RigidWorkspace w,const int* colors,
    int color,bool orientation,const BpAabb* boxes,double dt,double damping,double max_rotation,
    const double (*anchors)[4],int* error) {
    const std::size_t b=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (b>=s.bodies || !active(s,colors,color,b) || failed(error)) return;
    if (!(orientation ? s.rotate[b] : s.translate[b])) {
        for (int k=0;k<3;++k) { w.proposed_com[b][k]=x.com[b][k]; w.proposed_omega[b][k]=x.omega[b][k]; }
        for (int k=0;k<4;++k) w.proposed_orientation[b][k]=x.orientation[b][k];
        return;
    }
    double delta[3];
    if (!ldlt3(w.systems[b].hessian,w.systems[b].gradient,delta)) { fail(error,RigidSolveFailed); return; }
    if (!orientation) {
        for (int k=0;k<3;++k) {
            const double proposed=x.com[b][k]-damping*delta[k];
            if (!isfinite(proposed) || !isfinite(boxes[b].lo[k]) || !isfinite(boxes[b].hi[k]) || boxes[b].lo[k]>boxes[b].hi[k]) { fail(error,RigidInvalidBounds); return; }
            w.proposed_com[b][k]=fmin(boxes[b].hi[k],fmax(boxes[b].lo[k],proposed));
        }
    } else {
        // Limit the Newton increment before exponentiating it. A scaled norm
        // avoids overflow for finite but enormous solves, so a large proposed
        // step is safely shortened rather than encoded as an invisible turn.
        const double largest=fmax(fabs(delta[0]),fmax(fabs(delta[1]),fabs(delta[2])));
        double scale=damping;
        if (largest>0) {
            double squared=0;
            for (int k=0;k<3;++k) { const double component=delta[k]/largest; squared+=component*component; }
            const double allowed_scale=(max_rotation/dt)/largest/sqrt(squared);
            if (fabs(scale)>allowed_scale) scale=copysign(allowed_scale,scale);
        }
        for (int k=0;k<3;++k) {
            w.proposed_omega[b][k]=x.omega[b][k]-scale*delta[k];
        }
        if (!finite3(w.proposed_omega[b])) { fail(error,RigidRotationFailure); return; }
        double current[4],bounded[4];
        rigid_orientation(x.previous_orientation[b],x.omega[b],dt,current);
        rigid_orientation(x.previous_orientation[b],w.proposed_omega[b],dt,w.proposed_orientation[b]);
        if (!rotation_detail::normalize(current,current) ||
            !rotation_detail::normalize(w.proposed_orientation[b],w.proposed_orientation[b]) ||
            bound_rigid_orientation(current,current,w.proposed_orientation[b],max_rotation,bounded)!=RigidRotationStatus::Success) {
            fail(error,RigidRotationFailure); return;
        }
        // The production cap helper treats theta >= pi as unrestricted. Still
        // enforce this API's per-update full-arc bound at its pi endpoint.
        double relative[4]; rotation_detail::relative(bounded,current,relative);
        if (!rotation_detail::normalize(relative,relative)) { fail(error,RigidRotationFailure); return; }
        const double angle=2.0*atan2(rotation_detail::vec(relative+1).norm(),relative[0]);
        if (angle>max_rotation &&
            interpolate_rigid_orientation(current,bounded,max_rotation/angle,bounded)!=RigidRotationStatus::Success) {
            fail(error,RigidRotationFailure); return;
        }
        if (anchors && bound_rigid_orientation(anchors[b],current,bounded,max_rotation,bounded)!=RigidRotationStatus::Success) {
            fail(error,RigidRotationFailure); return;
        }
        for (int k=0;k<4;++k) w.proposed_orientation[b][k]=bounded[k];
        if (rigid_angular_velocity_from_orientation(bounded,x.previous_orientation[b],dt,w.proposed_omega[b])!=RigidRotationStatus::Success)
            fail(error,RigidRotationFailure);
    }
}

__global__ void propose_displacements(RigidScene s,RigidWorkspace w,const int* colors,
    int color,bool orientation,double damping,double (*increments)[3],int* error) {
    const std::size_t b=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (b>=s.bodies || !active(s,colors,color,b) || failed(error)) return;
    for (int k=0;k<3;++k) increments[b][k]=0.0;
    if (!(orientation ? s.rotate[b] : s.translate[b]) || damping==0.0) return;
    double delta[3];
    if (!ldlt3(w.systems[b].hessian,w.systems[b].gradient,delta)) {
        fail(error,RigidSolveFailed); return;
    }
    double proposal[3];
    for (int k=0;k<3;++k) proposal[k]=-damping*delta[k];
    if (!finite3(proposal)) { fail(error,RigidInvalidDerivative); return; }
    for (int k=0;k<3;++k) increments[b][k]=proposal[k];
}

__global__ void validate_translation(RigidScene s,RigidWorkspace w,const int* colors,int color,
    const double (*accepted)[3],int* error) {
    const std::size_t b=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (b>=s.bodies || !active(s,colors,color,b) || !s.translate[b] || failed(error)) return;
    if (!finite3(w.accepted_com[b])) { fail(error,RigidInvalidState); return; }
    for (std::size_t i=s.node_offsets[b];i<s.node_offsets[b+1];++i)
        if (!finite3(accepted[s.nodes[i]])) { fail(error,RigidInvalidState); return; }
}
__global__ void commit_translation(RigidScene s,RigidState x,RigidWorkspace w,DeformableState nodes,
    const int* colors,int color,const double (*accepted)[3],int* error) {
    const std::size_t b=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (b>=s.bodies || !active(s,colors,color,b) || !s.translate[b] || failed(error)) return;
    for (int k=0;k<3;++k) x.com[b][k]=w.accepted_com[b][k];
    for (std::size_t i=s.node_offsets[b];i<s.node_offsets[b+1];++i) {
        const int node=s.nodes[i]; for (int k=0;k<3;++k) nodes.positions[node][k]=accepted[node][k];
    }
}

__device__ bool intersects(const BpAabb& a,const BpAabb& b) {
    for (int k=0;k<3;++k) if (a.lo[k]>b.hi[k] || a.hi[k]<b.lo[k]) return false;
    return true;
}
__device__ BpAabb joined(BpAabb a,const BpAabb& b) {
    for (int k=0;k<3;++k) { a.lo[k]=fmin(a.lo[k],b.lo[k]); a.hi[k]=fmax(a.hi[k],b.hi[k]); }
    return a;
}
__device__ bool swept_candidate(const BpContact& contact,int body,RigidScene s,RigidState x,
    DeformableState state,double angle,int* error) {
    BpAabb boxes[4]; const double identity[4]={1,0,0,0};
    for (int i=0;i<4;++i) {
        const int node=contact.vertices[i];
        if (s.node_body[node]==body) {
            double offset[3]; for (int k=0;k<3;++k) offset[k]=state.positions[node][k]-x.com[body][k];
            if (rigid_spherical_cap_box(x.com[body],identity,offset,angle,boxes[i])!=RigidRotationStatus::Success) { fail(error,RigidRotationFailure); return false; }
            // Outward slack also contains the represented proxy position after
            // a translation whose rounded endpoint is not exactly COM+R*X.
            const double radius=rotation_detail::vec(offset).norm();
            for (int k=0;k<3;++k) {
                const double margin=64.0*DBL_EPSILON*(fabs(x.com[body][k])+radius+1.0);
                boxes[i].lo[k]=outward_neighbor(boxes[i].lo[k]-margin,false);
                boxes[i].hi[k]=outward_neighbor(boxes[i].hi[k]+margin,true);
            }
        } else for (int k=0;k<3;++k) boxes[i].lo[k]=boxes[i].hi[k]=state.positions[node][k];
    }
    return contact.kind==0 ? intersects(boxes[0],joined(joined(boxes[1],boxes[2]),boxes[3]))
                           : intersects(joined(boxes[0],boxes[1]),joined(boxes[2],boxes[3]));
}

__global__ void rotation_clip(RigidScene s,RigidState x,RigidWorkspace w,DeformableState state,
    BroadPhaseView broadphase,const int* colors,int color,double dt,double safety,bool use_ccd,
    double (*accepted)[3],int* error,const RotationBatchContext* gate) {
    if(gate&&gate->mode!=0)return;
    const std::size_t body=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (body>=s.bodies || !active(s,colors,color,body) || !s.rotate[body] || failed(error)) return;
    double current[4],relative[4],reverse[4];
    rigid_orientation(x.previous_orientation[body],x.omega[body],dt,current);
    if (!rotation_detail::normalize(current,current)) { fail(error,RigidRotationFailure); return; }
    rotation_detail::relative(w.proposed_orientation[body],current,relative);
    rotation_detail::relative(current,w.proposed_orientation[body],reverse);
    if (!rotation_detail::normalize(relative,relative) || !rotation_detail::normalize(reverse,reverse)) { fail(error,RigidRotationFailure); return; }
    const double angle=2.0*atan2(rotation_detail::vec(relative+1).norm(),relative[0]);
    double fraction=1.0; bool collision=false;
    if (use_ccd) for (std::size_t index=0;index<broadphase.contact_count;++index) {
        const BpContact c=broadphase.contacts[index];
        int owner[4]; bool involved=false;
        for (int i=0;i<4;++i) { owner[i]=s.node_body[c.vertices[i]]; involved=involved || owner[i]==static_cast<int>(body); }
        if (!involved) continue;
        const bool first=owner[0]==static_cast<int>(body);
        bool second;
        if (c.kind==0) {
            second=owner[1]==static_cast<int>(body);
            if ((owner[2]==static_cast<int>(body))!=second || (owner[3]==static_cast<int>(body))!=second) { fail(error,RigidContactOwnership); return; }
        } else {
            second=owner[2]==static_cast<int>(body);
            if ((owner[1]==static_cast<int>(body))!=first || (owner[3]==static_cast<int>(body))!=second) { fail(error,RigidContactOwnership); return; }
        }
        if (first && second) continue;
        if (!swept_candidate(c,static_cast<int>(body),s,x,state,angle,error)) { if (failed(error)) return; continue; }
        RigidRotationQuery query{}; query.vertex_face=c.kind==0;
        for (int k=0;k<3;++k) query.com[k]=x.com[body][k];
        for (int i=0;i<4;++i) {
            query.current[i]=first ? current[i] : (i==0 ? 1.0 : 0.0);
            query.target[i]=first ? w.proposed_orientation[body][i] : reverse[i];
            for (int k=0;k<3;++k) query.positions[i][k]=state.positions[c.vertices[i]][k];
        }
        const auto result=rigid_rotation_ccd(query);
        if (result.status!=RigidRotationStatus::Success) { fail(error,RigidRotationFailure); return; }
        if (result.collision) { collision=true; fraction=fmin(fraction,result.time); }
    }
    if (collision) fraction*=safety;
    if (interpolate_rigid_orientation(current,w.proposed_orientation[body],fraction,w.accepted_orientation[body])!=RigidRotationStatus::Success ||
        rigid_angular_velocity_from_orientation(w.accepted_orientation[body],x.previous_orientation[body],dt,w.accepted_omega[body])!=RigidRotationStatus::Success) {
        fail(error,RigidRotationFailure); return;
    }
    for (std::size_t i=s.node_offsets[body];i<s.node_offsets[body+1];++i) {
        const int node=s.nodes[i];
        rigid_world_position(s.reference[node],x.com[body],w.accepted_orientation[body],accepted[node]);
        if (!finite3(accepted[node])) { fail(error,RigidInvalidState); return; }
    }
}
// Prototype: the original scalar kernel above remains the differential oracle.
// A block owns one body. Every lane evaluates one candidate from the current
// chunk, and lane zero consumes all results in ORIGINAL candidate order.
// There is no tree reduction, changed per-query arithmetic, or global scratch.
__device__ int rotation_pair_result(const BpContact& c,std::size_t body,
    RigidScene s,RigidState x,RigidWorkspace w,DeformableState state,
    const double* current,const double* reverse,double angle,
    bool& collision,double& time) {
    collision=false;time=1.0;
    int owner[4];bool involved=false;
    for(int i=0;i<4;++i) {
        owner[i]=s.node_body[c.vertices[i]];
        involved=involved||owner[i]==static_cast<int>(body);
    }
    if(!involved)return 0;
    const bool first=owner[0]==static_cast<int>(body);
    bool second;
    if(c.kind==0) {
        second=owner[1]==static_cast<int>(body);
        if((owner[2]==static_cast<int>(body))!=second
            ||(owner[3]==static_cast<int>(body))!=second)return RigidContactOwnership;
    } else {
        second=owner[2]==static_cast<int>(body);
        if((owner[1]==static_cast<int>(body))!=first
            ||(owner[3]==static_cast<int>(body))!=second)return RigidContactOwnership;
    }
    if(first&&second)return 0;
    // Same swept_candidate arithmetic, with a lane-local status instead of
    // racing the global error. The ordered consumer publishes the first error.
    BpAabb boxes[4];const double identity[4]={1,0,0,0};
    for(int i=0;i<4;++i) {
        const int node=c.vertices[i];
        if(s.node_body[node]==static_cast<int>(body)) {
            double offset[3];
            for(int k=0;k<3;++k)offset[k]=state.positions[node][k]-x.com[body][k];
            if(rigid_spherical_cap_box(x.com[body],identity,offset,angle,boxes[i])
                !=RigidRotationStatus::Success)return RigidRotationFailure;
            const double radius=rotation_detail::vec(offset).norm();
            for(int k=0;k<3;++k) {
                const double margin=64.0*DBL_EPSILON*(fabs(x.com[body][k])+radius+1.0);
                boxes[i].lo[k]=outward_neighbor(boxes[i].lo[k]-margin,false);
                boxes[i].hi[k]=outward_neighbor(boxes[i].hi[k]+margin,true);
            }
        } else for(int k=0;k<3;++k)boxes[i].lo[k]=boxes[i].hi[k]=state.positions[node][k];
    }
    const bool candidate=c.kind==0
        ?intersects(boxes[0],joined(joined(boxes[1],boxes[2]),boxes[3]))
        :intersects(joined(boxes[0],boxes[1]),joined(boxes[2],boxes[3]));
    if(!candidate)return 0;
    RigidRotationQuery query{};query.vertex_face=c.kind==0;
    for(int k=0;k<3;++k)query.com[k]=x.com[body][k];
    for(int i=0;i<4;++i) {
        query.current[i]=first?current[i]:(i==0?1.0:0.0);
        query.target[i]=first?w.proposed_orientation[body][i]:reverse[i];
        for(int k=0;k<3;++k)query.positions[i][k]=state.positions[c.vertices[i]][k];
    }
    const auto result=rigid_rotation_ccd(query);
    if(result.status!=RigidRotationStatus::Success)return RigidRotationFailure;
    collision=result.collision;time=result.time;
    return 0;
}

__global__ void rotation_clip_pair_parallel(RigidScene s,RigidState x,RigidWorkspace w,
    DeformableState state,BroadPhaseView broadphase,const int* colors,int color,
    double dt,double safety,bool use_ccd,double(*accepted)[3],int* error,const RotationBatchContext* gate) {
    if(gate&&gate->mode!=0)return;
    const std::size_t body=blockIdx.x;
    const unsigned lane=threadIdx.x;
    __shared__ double current[4],reverse[4],angle,fraction;
    __shared__ int enabled,collision;
    __shared__ double pair_times[128];
    __shared__ int pair_status[128],pair_hits[128];
    if(lane==0) {
        enabled=body<s.bodies&&active(s,colors,color,body)&&s.rotate[body]&&!failed(error);
        collision=0;fraction=1.0;
        if(enabled) {
            double relative[4];
            rigid_orientation(x.previous_orientation[body],x.omega[body],dt,current);
            if(!rotation_detail::normalize(current,current)) {
                fail(error,RigidRotationFailure);enabled=0;
            } else {
                rotation_detail::relative(w.proposed_orientation[body],current,relative);
                rotation_detail::relative(current,w.proposed_orientation[body],reverse);
                if(!rotation_detail::normalize(relative,relative)
                    ||!rotation_detail::normalize(reverse,reverse)) {
                    fail(error,RigidRotationFailure);enabled=0;
                } else angle=2.0*atan2(rotation_detail::vec(relative+1).norm(),relative[0]);
            }
        }
    }
    __syncthreads();
    if(!enabled)return;
    if(use_ccd)for(std::size_t first=0;first<broadphase.contact_count;first+=blockDim.x) {
        const std::size_t index=first+lane;
        bool hit=false;double time=1.0;int status=0;
        if(index<broadphase.contact_count)
            status=rotation_pair_result(broadphase.contacts[index],body,s,x,w,state,
                current,reverse,angle,hit,time);
        pair_times[lane]=time;pair_hits[lane]=hit;pair_status[lane]=status;
        __syncthreads();
        if(lane==0) {
            for(unsigned item=0;item<blockDim.x&&first+item<broadphase.contact_count;++item) {
                if(pair_status[item]) {
                    fail(error,pair_status[item]);enabled=0;break;
                }
                if(pair_hits[item]) {collision=1;fraction=fmin(fraction,pair_times[item]);}
            }
        }
        // All lanes observe the same error decision before leaving a chunk.
        __syncthreads();
        if(!enabled)return;
    }
    if(lane!=0)return;
    if(collision)fraction*=safety;
    if(interpolate_rigid_orientation(current,w.proposed_orientation[body],fraction,w.accepted_orientation[body])
        !=RigidRotationStatus::Success
        ||rigid_angular_velocity_from_orientation(w.accepted_orientation[body],x.previous_orientation[body],dt,w.accepted_omega[body])
        !=RigidRotationStatus::Success) {fail(error,RigidRotationFailure);return;}
    for(std::size_t i=s.node_offsets[body];i<s.node_offsets[body+1];++i) {
        const int node=s.nodes[i];
        rigid_world_position(s.reference[node],x.com[body],w.accepted_orientation[body],accepted[node]);
        if(!finite3(accepted[node])) {fail(error,RigidInvalidState);return;}
    }
}

__global__ void rotation_batch_prepare(RigidScene s,RigidState x,RigidWorkspace w,
    BroadPhaseView bp,const int* colors,int color,double dt,int* error,RotationBatchContext* ctx) {
    if(blockIdx.x||threadIdx.x)return;*ctx=RotationBatchContext{};
    if(failed(error)){ctx->mode=1;ctx->done=1;return;}
    if(bp.contact_count>0xffffffffull){return;}
    std::size_t selected=0;bool found=false;
    for(std::size_t body=0;body<s.bodies;++body)if(active(s,colors,color,body)&&s.rotate[body]) {
        if(found){return;}
        found=true;selected=body;
    }
    if(!found){return;}
    ctx->mode=1;ctx->body=selected;
    // Exactly the original scalar preparation, retaining all normalization
    // and relative-quaternion operations in their original order.
    double relative[4];
    rigid_orientation(x.previous_orientation[selected],x.omega[selected],dt,ctx->current);
    if(!rotation_detail::normalize(ctx->current,ctx->current)) {
        fail(error,RigidRotationFailure);ctx->done=1;return;
    }
    rotation_detail::relative(w.proposed_orientation[selected],ctx->current,relative);
    rotation_detail::relative(ctx->current,w.proposed_orientation[selected],ctx->reverse);
    if(!rotation_detail::normalize(relative,relative)||!rotation_detail::normalize(ctx->reverse,ctx->reverse)) {
        fail(error,RigidRotationFailure);ctx->done=1;return;
    }
    ctx->angle=2.0*atan2(rotation_detail::vec(relative+1).norm(),relative[0]);
}
__device__ void rotation_batch_min(double* destination,double value) {
    auto* bits=reinterpret_cast<unsigned long long*>(destination);
    auto observed=atomicCAS(bits,0ull,0ull);
    while(true) {
        const auto desired=static_cast<unsigned long long>(__double_as_longlong(
            fmin(__longlong_as_double(static_cast<long long>(observed)),value)));
        if(desired==observed)return;
        const auto previous=atomicCAS(bits,observed,desired);
        if(previous==observed)return;observed=previous;
    }
}
__global__ void rotation_batch_pairs(RigidScene s,RigidState x,RigidWorkspace w,DeformableState state,
    BroadPhaseView bp,bool use_ccd,RotationBatchContext* ctx) {
    if(ctx->mode!=1||ctx->done||!use_ccd)return;
    for(std::size_t index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;index<bp.contact_count;
        index+=std::size_t(gridDim.x)*blockDim.x) {
        bool hit=false;double time=1.0;
        const int status=rotation_pair_result(bp.contacts[index],ctx->body,s,x,w,state,
            ctx->current,ctx->reverse,ctx->angle,hit,time);
        if(status)atomicMin(&ctx->first_error,
            (static_cast<unsigned long long>(index)<<32)|static_cast<unsigned>(status));
        else if(hit){atomicOr(&ctx->hit,1u);rotation_batch_min(&ctx->fraction,time);}
    }
}
__global__ void rotation_batch_finish(RigidScene s,RigidState x,RigidWorkspace w,
    double dt,double safety,double(*accepted)[3],int* error,RotationBatchContext* ctx,
    bool publish_proxies) {
    if(blockIdx.x||threadIdx.x||ctx->mode!=1||ctx->done)return;
    if(ctx->first_error!=~0ull) {
        atomicCAS(error,0,static_cast<int>(static_cast<unsigned>(ctx->first_error)));ctx->done=1;return;
    }
    const auto body=ctx->body;double fraction=ctx->fraction;
    if(ctx->hit)fraction*=safety;
    if(interpolate_rigid_orientation(ctx->current,w.proposed_orientation[body],fraction,w.accepted_orientation[body])
        !=RigidRotationStatus::Success
        ||rigid_angular_velocity_from_orientation(w.accepted_orientation[body],x.previous_orientation[body],dt,w.accepted_omega[body])
        !=RigidRotationStatus::Success) {fail(error,RigidRotationFailure);ctx->done=1;return;}
    if(!publish_proxies) {ctx->done=1;return;}
    // Keep proxy publication scalar and ordered, including partial private
    // scratch writes on a late invalid reference. The commit remains separate.
    for(std::size_t i=s.node_offsets[body];i<s.node_offsets[body+1];++i) {
        const int node=s.nodes[i];
        rigid_world_position(s.reference[node],x.com[body],w.accepted_orientation[body],accepted[node]);
        if(!finite3(accepted[node])) {fail(error,RigidInvalidState);ctx->done=1;return;}
    }
    ctx->done=1;
}

__global__ void rotation_commit(RigidScene s,RigidState x,RigidWorkspace w,DeformableState nodes,
    const int* colors,int color,const double (*accepted)[3],int* error) {
    const std::size_t body=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (body>=s.bodies || !active(s,colors,color,body) || !s.rotate[body] || failed(error)) return;
    for (int k=0;k<4;++k) x.orientation[body][k]=w.accepted_orientation[body][k];
    for (int k=0;k<3;++k) x.omega[body][k]=w.accepted_omega[body][k];
    for (std::size_t i=s.node_offsets[body];i<s.node_offsets[body+1];++i) {
        const int node=s.nodes[i]; for (int k=0;k<3;++k) nodes.positions[node][k]=accepted[node][k];
    }
}
__global__ void finish(RigidScene s,RigidState x,DeformableState nodes,double dt,int* error,
    const int* fallback_only) {
    if(fallback_only&&!*fallback_only)return;
    const std::size_t b=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (b>=s.bodies || failed(error)) return;
    for (int k=0;k<3;++k) {
        x.velocity[b][k]=s.translate[b] ? (x.com[b][k]-x.previous_com[b][k])/dt : 0.0;
        if (!s.rotate[b]) x.omega[b][k]=0.0;
    }
    if (!finite3(x.velocity[b]) || !finite3(x.omega[b])) { fail(error,RigidInvalidState); return; }
    const double zero[3]={0,0,0};
    for (std::size_t i=s.node_offsets[b];i<s.node_offsets[b+1];++i) {
        const int node=s.nodes[i]; double offset[3];
        rigid_world_position(s.reference[node],zero,x.orientation[b],offset);
        const auto tangent=rotation_detail::vec(x.omega[b]).cross(rotation_detail::vec(offset));
        const double rotational[3]={tangent.x,tangent.y,tangent.z};
        for (int k=0;k<3;++k) nodes.velocities[node][k]=x.velocity[b][k]+rotational[k];
        if (!finite3(nodes.velocities[node])) { fail(error,RigidInvalidState); return; }
    }
}

// These kernels distribute authoritative CSR slots. They never infer a body
// from a node ID or assume contiguous node IDs, and only run for caller-checked
// unique topology. Empty rows still have exactly one body-value writer.
__global__ void validate_translation_body(RigidScene s,RigidWorkspace w,
    const int* colors,int color,int* error) {
    const std::size_t b=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(b>=s.bodies||!active(s,colors,color,b)||!s.translate[b]||failed(error))return;
    if(!finite3(w.accepted_com[b]))fail(error,RigidInvalidState);
}
__global__ void validate_translation_proxies(RigidScene s,const int* colors,int color,
    const double(*accepted)[3],int* error,proxy_index::Plan plan) {
    const unsigned tile=proxy_index::tile(blockIdx.x,plan);
    for(auto b=proxy_index::first_body(blockIdx.x,plan);b<s.bodies;
        b=proxy_index::next_body(b,s.bodies,plan)) {
        if(!active(s,colors,color,b)||!s.translate[b]||failed(error))continue;
        for(auto row=proxy_index::row_cursor(s.node_offsets[b],s.node_offsets[b+1],tile,threadIdx.x,plan);
            row.valid();row.advance())
            if(!finite3(accepted[s.nodes[row.slot()]]))fail(error,RigidInvalidState);
    }
}
__global__ void publish_translation_proxies(RigidScene s,RigidState x,RigidWorkspace w,
    DeformableState nodes,const int* colors,int color,const double(*accepted)[3],
    int* error,proxy_index::Plan plan) {
    const unsigned tile=proxy_index::tile(blockIdx.x,plan);
    for(auto b=proxy_index::first_body(blockIdx.x,plan);b<s.bodies;
        b=proxy_index::next_body(b,s.bodies,plan)) {
        if(!active(s,colors,color,b)||!s.translate[b]||failed(error))continue;
        if(tile==0&&threadIdx.x==0)
            for(int k=0;k<3;++k)x.com[b][k]=w.accepted_com[b][k];
        for(auto row=proxy_index::row_cursor(s.node_offsets[b],s.node_offsets[b+1],tile,threadIdx.x,plan);
            row.valid();row.advance()) {
            const int node=s.nodes[row.slot()];
            for(int k=0;k<3;++k)nodes.positions[node][k]=accepted[node][k];
        }
    }
}
__global__ void publish_rotation_proxies(RigidScene s,RigidState x,RigidWorkspace w,
    DeformableState nodes,const int* colors,int color,const double(*accepted)[3],
    int* error,proxy_index::Plan plan) {
    const unsigned tile=proxy_index::tile(blockIdx.x,plan);
    for(auto b=proxy_index::first_body(blockIdx.x,plan);b<s.bodies;
        b=proxy_index::next_body(b,s.bodies,plan)) {
        if(!active(s,colors,color,b)||!s.rotate[b]||failed(error))continue;
        if(tile==0&&threadIdx.x==0) {
            for(int k=0;k<4;++k)x.orientation[b][k]=w.accepted_orientation[b][k];
            for(int k=0;k<3;++k)x.omega[b][k]=w.accepted_omega[b][k];
        }
        for(auto row=proxy_index::row_cursor(s.node_offsets[b],s.node_offsets[b+1],tile,threadIdx.x,plan);
            row.valid();row.advance()) {
            const int node=s.nodes[row.slot()];
            for(int k=0;k<3;++k)nodes.positions[node][k]=accepted[node][k];
        }
    }
}

// The prospective values below preserve finish's scalar component arithmetic.
// Preflight writes only a Boolean fault. A fault routes the COMPLETE call to
// the untouched ordered kernel, preserving even private/partial error writes.
__device__ void finish_values(RigidScene s,RigidState x,std::size_t b,double dt,
    double* velocity,double* omega) {
    proxy_values::finish_body(x.com[b],x.previous_com[b],x.omega[b],s.translate[b],
        s.rotate[b],dt,velocity,omega);
}
__device__ void finish_proxy_value(RigidScene s,RigidState x,std::size_t b,int node,
    const double* velocity,const double* omega,double* result) {
    proxy_values::finish_proxy(s.reference[node],x.orientation[b],velocity,omega,result);
}
__global__ void preflight_finish_bodies(RigidScene s,RigidState x,double dt,
    int* error,int* fault) {
    const auto b=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(b>=s.bodies||failed(error))return;
    double velocity[3],omega[3];finish_values(s,x,b,dt,velocity,omega);
    if(!finite3(velocity)||!finite3(omega))atomicOr(fault,1);
}
__global__ void preflight_finish_proxies(RigidScene s,RigidState x,double dt,
    int* error,int* fault,proxy_index::Plan plan) {
    if(atomicAdd(fault,0)||failed(error))return;
    const unsigned tile=proxy_index::tile(blockIdx.x,plan);
    for(auto b=proxy_index::first_body(blockIdx.x,plan);b<s.bodies;
        b=proxy_index::next_body(b,s.bodies,plan)) {
        double velocity[3],omega[3];finish_values(s,x,b,dt,velocity,omega);
        for(auto row=proxy_index::row_cursor(s.node_offsets[b],s.node_offsets[b+1],tile,threadIdx.x,plan);
            row.valid();row.advance()) {
            double value[3];finish_proxy_value(s,x,b,s.nodes[row.slot()],velocity,omega,value);
            if(!finite3(value))atomicOr(fault,1);
        }
    }
}
__global__ void publish_finish_proxies(RigidScene s,RigidState x,DeformableState nodes,
    double dt,int* error,const int* fault,proxy_index::Plan plan) {
    if(*fault||failed(error))return;
    const unsigned tile=proxy_index::tile(blockIdx.x,plan);
    for(auto b=proxy_index::first_body(blockIdx.x,plan);b<s.bodies;
        b=proxy_index::next_body(b,s.bodies,plan)) {
        double velocity[3],omega[3];finish_values(s,x,b,dt,velocity,omega);
        if(tile==0&&threadIdx.x==0)for(int k=0;k<3;++k) {
            x.velocity[b][k]=velocity[k];
            if(!s.rotate[b])x.omega[b][k]=0.0;
        }
        for(auto row=proxy_index::row_cursor(s.node_offsets[b],s.node_offsets[b+1],tile,threadIdx.x,plan);
            row.valid();row.advance()) {
            const int node=s.nodes[row.slot()];
            finish_proxy_value(s,x,b,node,velocity,omega,nodes.velocities[node]);
        }
    }
}

// Batch rotation's accepted quaternion/omega are prepared by the SAME scalar
// finish prefix. Finite preflight writes no accepted positions. Invalid data
// replays only the old ordered proxy suffix, including its first-bad prefix.
__global__ void preflight_batch_rotation_proxies(RigidScene s,RigidState x,
    RigidWorkspace w,int* error,const RotationBatchContext* ctx,int* fault) {
    if(ctx->mode!=1||!ctx->done||failed(error))return;
    const auto b=ctx->body;
    const auto count=s.node_offsets[b+1]>s.node_offsets[b]
        ?s.node_offsets[b+1]-s.node_offsets[b]:0;
    const auto stride=std::size_t(gridDim.x)*blockDim.x;
    for(auto relative=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;relative<count;) {
        const int node=s.nodes[s.node_offsets[b]+relative];double value[3];
        rigid_world_position(s.reference[node],x.com[b],w.accepted_orientation[b],value);
        if(!finite3(value))atomicOr(fault,1);
        if(count-relative<=stride)break;relative+=stride;
    }
}
__global__ void publish_batch_rotation_proxies(RigidScene s,RigidState x,
    RigidWorkspace w,double(*accepted)[3],int* error,const RotationBatchContext* ctx,
    const int* fault) {
    if(ctx->mode!=1||!ctx->done||*fault||failed(error))return;
    const auto b=ctx->body;
    const auto count=s.node_offsets[b+1]>s.node_offsets[b]
        ?s.node_offsets[b+1]-s.node_offsets[b]:0;
    const auto stride=std::size_t(gridDim.x)*blockDim.x;
    for(auto relative=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;relative<count;) {
        const int node=s.nodes[s.node_offsets[b]+relative];
        rigid_world_position(s.reference[node],x.com[b],w.accepted_orientation[b],accepted[node]);
        if(count-relative<=stride)break;relative+=stride;
    }
}
__global__ void fallback_batch_rotation_proxies(RigidScene s,RigidState x,
    RigidWorkspace w,double(*accepted)[3],int* error,RotationBatchContext* ctx,
    const int* fault) {
    if(blockIdx.x||threadIdx.x||ctx->mode!=1||!ctx->done||!*fault||failed(error))return;
    const auto body=ctx->body;
    for(std::size_t i=s.node_offsets[body];i<s.node_offsets[body+1];++i) {
        const int node=s.nodes[i];
        rigid_world_position(s.reference[node],x.com[body],w.accepted_orientation[body],accepted[node]);
        if(!finite3(accepted[node])) {fail(error,RigidInvalidState);ctx->done=1;return;}
    }
}
} // namespace

void begin_rigid_substep(const RigidScene& s,const RigidState& x,int* error,cudaStream_t stream) {
    if (s.bodies) begin<<<blocks(s.bodies),width,0,stream>>>(s,x,error); launched();
}
void build_rigid_blue_boxes(const RigidScene& s,const RigidState& x,std::size_t nodes,
    double radius,BpAabb* body_boxes,BpAabb* node_boxes,int* error,double angle,
    double (*anchors)[4],const double (*represented)[3],cudaStream_t stream) {
    if (!(radius>=0) || !std::isfinite(radius)) throw std::invalid_argument("Rigid COM radius must be finite and nonnegative");
    if (!(angle>=0 && angle<=rotation_detail::pi)) throw std::invalid_argument("Rigid blue-box angle must be in [0, pi]");
    if (s.bodies && angle<rotation_detail::pi && !anchors) throw std::invalid_argument("Restricted rigid blue boxes require persistent quaternion anchors");
    if (s.bodies) body_bounds<<<blocks(s.bodies),width,0,stream>>>(s,x,radius,body_boxes,anchors,error);
    if (s.bodies && nodes) proxy_bounds<<<blocks(nodes),width,0,stream>>>(s,x,nodes,body_boxes,node_boxes,angle,anchors,represented,error); launched();
}
void assemble_rigid_inertia_color(const RigidScene& s,const RigidState& x,const RigidWorkspace& w,
    const int* colors,int color,bool orientation,double dt,double gravity,int* error,cudaStream_t stream) {
    valid_dt(dt); if (!std::isfinite(gravity)) throw std::invalid_argument("Nonfinite rigid gravity");
    if (s.bodies) inertia<<<blocks(s.bodies),width,0,stream>>>(s,x,w,colors,color,orientation,dt,gravity,error); launched();
}
void propose_rigid_color(const RigidScene& s,const RigidState& x,const RigidWorkspace& w,
    const int* colors,int color,bool orientation,const BpAabb* boxes,double dt,double damping,int* error,double max_rotation,
    const double (*anchors)[4],cudaStream_t stream) {
    valid_dt(dt); if (!std::isfinite(damping)) throw std::invalid_argument("Nonfinite rigid damping");
    if (!(max_rotation>=0 && max_rotation<=rotation_detail::pi)) throw std::invalid_argument("Rigid rotation cap must be in [0, pi]");
    if (s.bodies) propose<<<blocks(s.bodies),width,0,stream>>>(s,x,w,colors,color,orientation,boxes,dt,damping,max_rotation,anchors,error); launched();
}
void propose_rigid_displacements(const RigidScene& s,const RigidWorkspace& w,
    const int* colors,int color,bool orientation,double damping,double (*increments)[3],
    int* error,cudaStream_t stream) {
    if (!std::isfinite(damping)) throw std::invalid_argument("Nonfinite rigid damping");
    if (s.bodies && (!colors || !increments || !error))
        throw std::invalid_argument("Rigid displacement proposal requires device storage");
    if (s.bodies) propose_displacements<<<blocks(s.bodies),width,0,stream>>>(s,w,colors,color,orientation,damping,increments,error);
    launched();
}
void commit_rigid_translation_color(const RigidScene& s,const RigidState& x,const RigidWorkspace& w,
    const DeformableState& nodes,const int* colors,int color,const double (*accepted)[3],int* error,cudaStream_t stream) {
    if (s.bodies) {
        validate_translation<<<blocks(s.bodies),width,0,stream>>>(s,w,colors,color,accepted,error);
        commit_translation<<<blocks(s.bodies),width,0,stream>>>(s,x,w,nodes,colors,color,accepted,error);
    } launched();
}
void clip_rigid_rotation_color(const RigidScene& s,const RigidState& x,const RigidWorkspace& w,
    const DeformableState& nodes,const BroadPhaseView& bp,const int* colors,int color,double dt,
    double safety,bool use_ccd,double (*accepted)[3],int* error,cudaStream_t stream) {
    valid_dt(dt); if (!(safety>=0 && safety<=1)) throw std::invalid_argument("Rigid CCD safety must be in [0,1]");
    const unsigned pair_threads=rotation_pair_threads();
    if(s.bodies) {
        if(pair_threads==1)
            rotation_clip<<<blocks(s.bodies),width,0,stream>>>(s,x,w,nodes,bp,colors,color,dt,safety,use_ccd,accepted,error,nullptr);
        else rotation_clip_pair_parallel<<<static_cast<unsigned>(s.bodies),pair_threads,0,stream>>>(s,x,w,nodes,bp,colors,color,dt,safety,use_ccd,accepted,error,nullptr);
    }
    launched();
}
struct GpuRigidRotationClipper::Impl {
    RotationBatchContext* context=nullptr;
    int* proxy_fault=nullptr;
    const bool parallel_proxies;
    explicit Impl(bool validated):parallel_proxies(validated&&parallel_proxy_option()){}
    ~Impl(){if(context)cudaFree(context);if(proxy_fault)cudaFree(proxy_fault);}
    void reserve() {
        if(!context) {
            RotationBatchContext* replacement=nullptr;
            const auto status=cudaMalloc(&replacement,sizeof(*replacement));
            if(status!=cudaSuccess)throw std::runtime_error(std::string("GPU rotation context allocation: ")+cudaGetErrorString(status));
            context=replacement;
        }
        if(parallel_proxies&&!proxy_fault) {
            const auto status=cudaMalloc(reinterpret_cast<void**>(&proxy_fault),sizeof(*proxy_fault));
            if(status!=cudaSuccess)throw std::runtime_error(std::string("GPU rotation proxy flag allocation: ")+cudaGetErrorString(status));
        }
    }
};
GpuRigidRotationClipper::GpuRigidRotationClipper():impl_(std::make_unique<Impl>(false)){}
GpuRigidRotationClipper::GpuRigidRotationClipper(bool validated):impl_(std::make_unique<Impl>(validated)){}
GpuRigidRotationClipper::~GpuRigidRotationClipper()=default;
void GpuRigidRotationClipper::clip_color(const RigidScene& s,const RigidState& x,const RigidWorkspace& w,
    const DeformableState& nodes,const BroadPhaseView& bp,const int* colors,int color,double dt,
    double safety,bool use_ccd,double(*accepted)[3],int* error,cudaStream_t stream) {
    valid_dt(dt);if(!(safety>=0&&safety<=1))throw std::invalid_argument("Rigid CCD safety must be in [0,1]");
    const unsigned pair_threads=rotation_pair_threads();if(!s.bodies){launched();return;}
    impl_->reserve();auto* context=impl_->context;
    const auto wanted=bp.contact_count/128+(bp.contact_count%128!=0);
    const unsigned grid=static_cast<unsigned>(wanted==0?1:wanted>65535?65535:wanted);
    rotation_batch_prepare<<<1,1,0,stream>>>(s,x,w,bp,colors,color,dt,error,context);launched();
    rotation_batch_pairs<<<grid,128,0,stream>>>(s,x,w,nodes,bp,use_ccd,context);launched();
    if(impl_->parallel_proxies) {
        const auto status=cudaMemsetAsync(impl_->proxy_fault,0,sizeof(int),stream);
        if(status!=cudaSuccess)throw std::runtime_error(std::string("GPU rotation proxy flag clear: ")+cudaGetErrorString(status));
        rotation_batch_finish<<<1,1,0,stream>>>(s,x,w,dt,safety,accepted,error,context,false);launched();
        preflight_batch_rotation_proxies<<<128,width,0,stream>>>(s,x,w,error,context,impl_->proxy_fault);launched();
        publish_batch_rotation_proxies<<<128,width,0,stream>>>(s,x,w,accepted,error,context,impl_->proxy_fault);launched();
        fallback_batch_rotation_proxies<<<1,1,0,stream>>>(s,x,w,accepted,error,context,impl_->proxy_fault);launched();
    } else {
        rotation_batch_finish<<<1,1,0,stream>>>(s,x,w,dt,safety,accepted,error,context,true);launched();
    }
    if(pair_threads==1)
        rotation_clip<<<blocks(s.bodies),width,0,stream>>>(s,x,w,nodes,bp,colors,color,dt,safety,use_ccd,accepted,error,context);
    else rotation_clip_pair_parallel<<<static_cast<unsigned>(s.bodies),pair_threads,0,stream>>>(s,x,w,nodes,bp,colors,color,dt,safety,use_ccd,accepted,error,context);
    launched();
}
void commit_rigid_rotation_color(const RigidScene& s,const RigidState& x,const RigidWorkspace& w,
    const DeformableState& nodes,const int* colors,int color,const double (*accepted)[3],int* error,cudaStream_t stream) {
    if (s.bodies) rotation_commit<<<blocks(s.bodies),width,0,stream>>>(s,x,w,nodes,colors,color,accepted,error); launched();
}
void finish_rigid_substep(const RigidScene& s,const RigidState& x,const DeformableState& nodes,
    double dt,int* error,cudaStream_t stream) {
    valid_dt(dt); if (s.bodies) finish<<<blocks(s.bodies),width,0,stream>>>(s,x,nodes,dt,error,nullptr); launched();
}

struct GpuRigidProxyPipeline::Impl {
    const bool enabled;
    int* fault=nullptr;
    explicit Impl(bool validated):enabled(validated&&parallel_proxy_option()){}
    ~Impl(){if(fault)cudaFree(fault);}
    void reserve() {
        if(fault)return;
        const auto status=cudaMalloc(reinterpret_cast<void**>(&fault),sizeof(*fault));
        if(status!=cudaSuccess)throw std::runtime_error(std::string("GPU rigid proxy flag allocation: ")+cudaGetErrorString(status));
    }
};
GpuRigidProxyPipeline::GpuRigidProxyPipeline(bool validated):impl_(std::make_unique<Impl>(validated)){}
GpuRigidProxyPipeline::~GpuRigidProxyPipeline()=default;
void GpuRigidProxyPipeline::commit_translation_color(const RigidScene& s,const RigidState& x,
    const RigidWorkspace& w,const DeformableState& nodes,const int* colors,int color,
    const double(*accepted)[3],int* error,cudaStream_t stream) {
    if(!impl_->enabled) {
        commit_rigid_translation_color(s,x,w,nodes,colors,color,accepted,error,stream);return;
    }
    if(s.bodies) {
        const auto plan=proxy_index::make_plan(s.bodies);
        validate_translation_body<<<blocks(s.bodies),width,0,stream>>>(s,w,colors,color,error);launched();
        validate_translation_proxies<<<plan.blocks,width,0,stream>>>(s,colors,color,accepted,error,plan);launched();
        publish_translation_proxies<<<plan.blocks,width,0,stream>>>(s,x,w,nodes,colors,color,accepted,error,plan);
    }
    launched();
}
void GpuRigidProxyPipeline::commit_rotation_color(const RigidScene& s,const RigidState& x,
    const RigidWorkspace& w,const DeformableState& nodes,const int* colors,int color,
    const double(*accepted)[3],int* error,cudaStream_t stream) {
    if(!impl_->enabled) {
        commit_rigid_rotation_color(s,x,w,nodes,colors,color,accepted,error,stream);return;
    }
    if(s.bodies) {
        const auto plan=proxy_index::make_plan(s.bodies);
        publish_rotation_proxies<<<plan.blocks,width,0,stream>>>(s,x,w,nodes,colors,color,accepted,error,plan);
    }
    launched();
}
void GpuRigidProxyPipeline::finish_substep(const RigidScene& s,const RigidState& x,
    const DeformableState& nodes,double dt,int* error,cudaStream_t stream) {
    if(!impl_->enabled) {finish_rigid_substep(s,x,nodes,dt,error,stream);return;}
    valid_dt(dt);
    if(s.bodies) {
        impl_->reserve();
        const auto status=cudaMemsetAsync(impl_->fault,0,sizeof(int),stream);
        if(status!=cudaSuccess)throw std::runtime_error(std::string("GPU rigid proxy flag clear: ")+cudaGetErrorString(status));
        const auto plan=proxy_index::make_plan(s.bodies);
        preflight_finish_bodies<<<blocks(s.bodies),width,0,stream>>>(s,x,dt,error,impl_->fault);launched();
        preflight_finish_proxies<<<plan.blocks,width,0,stream>>>(s,x,dt,error,impl_->fault,plan);launched();
        publish_finish_proxies<<<plan.blocks,width,0,stream>>>(s,x,nodes,dt,error,impl_->fault,plan);launched();
        finish<<<blocks(s.bodies),width,0,stream>>>(s,x,nodes,dt,error,impl_->fault);
    }
    launched();
}
} // namespace ipc_gpu::solver
