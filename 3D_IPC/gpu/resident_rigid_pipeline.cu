#include "resident_rigid_pipeline.h"
#include "resident_rigid_ccd.cuh"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ipc_gpu::resident {
namespace {
constexpr unsigned width = 128;
unsigned blocks(std::size_t n) { return static_cast<unsigned>((n+width-1)/width); }
void launched() {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(std::string("Resident rigid CUDA launch: ")+cudaGetErrorString(status));
}
void valid_dt(double dt) {
    if (!(dt > 0) || !std::isfinite(dt)) throw std::invalid_argument("Resident rigid dt must be positive and finite");
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
    double (*accepted)[3],int* error) {
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
__global__ void finish(RigidScene s,RigidState x,DeformableState nodes,double dt,int* error) {
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
    if (s.bodies) rotation_clip<<<blocks(s.bodies),width,0,stream>>>(s,x,w,nodes,bp,colors,color,dt,safety,use_ccd,accepted,error); launched();
}
void commit_rigid_rotation_color(const RigidScene& s,const RigidState& x,const RigidWorkspace& w,
    const DeformableState& nodes,const int* colors,int color,const double (*accepted)[3],int* error,cudaStream_t stream) {
    if (s.bodies) rotation_commit<<<blocks(s.bodies),width,0,stream>>>(s,x,w,nodes,colors,color,accepted,error); launched();
}
void finish_rigid_substep(const RigidScene& s,const RigidState& x,const DeformableState& nodes,
    double dt,int* error,cudaStream_t stream) {
    valid_dt(dt); if (s.bodies) finish<<<blocks(s.bodies),width,0,stream>>>(s,x,nodes,dt,error); launched();
}
} // namespace ipc_gpu::resident
