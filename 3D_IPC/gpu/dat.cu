#include "dat.h"
#include "dat.cuh"
#include "rigid.cuh"
#include "rigid_dat.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ipc_gpu::solver {
namespace {
constexpr unsigned threads=128;
constexpr int exact_slots=512;
constexpr int small_limbs=256,large_limbs=1600;
constexpr unsigned max_small_workers=64,max_large_workers=2;
constexpr unsigned small_worker_threads=8;
void checked(cudaError_t status,const char* what) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string("GPU DAT ")+what+": "+cudaGetErrorString(status));
}
unsigned blocks(std::size_t count) { return static_cast<unsigned>((count+threads-1)/threads); }
template<class T> struct Buffer {
    T* data=nullptr;std::size_t capacity=0;
    ~Buffer(){if(data)cudaFree(data);}
    void reserve(std::size_t n) {
        if(n<=capacity)return;
        if(n>std::numeric_limits<std::size_t>::max()/sizeof(T))throw std::overflow_error("GPU DAT allocation extent");
        T* replacement=nullptr;checked(cudaMalloc(reinterpret_cast<void**>(&replacement),n*sizeof(T)),"allocate");
        if(data)checked(cudaFree(data),"release old allocation");
        data=replacement;capacity=n;
    }
};
using V=dat_detail::Vec;
namespace rd=rigid_dat_detail;
static_assert(sizeof(V)==3*sizeof(double));
__device__ void fail(int* status,int value){atomicCAS(status,0,value);}
__device__ int failed(int* status){return atomicAdd(status,0);}
__device__ double load_factor(const double* address) {
    auto* bits=reinterpret_cast<unsigned long long*>(const_cast<double*>(address));
    return __longlong_as_double(static_cast<long long>(atomicCAS(bits,0ULL,0ULL)));
}
__device__ void atomic_min_factor(double* address,double value) {
    auto* bits=reinterpret_cast<unsigned long long*>(address);
    auto previous=atomicCAS(bits,0ULL,0ULL);
    while(value<__longlong_as_double(static_cast<long long>(previous))) {
        const auto expected=previous;
        previous=atomicCAS(bits,expected,static_cast<unsigned long long>(__double_as_longlong(value)));
        if(previous==expected)break;
    }
}
struct RigidWork {
    RigidScene scene{};
    RigidState state{};
    const double (*delta_com)[3]=nullptr;
    const double (*delta_omega)[3]=nullptr;
    double dt=0;
    double* factors=nullptr;
    rd::PreparedTrajectory* prepared=nullptr;
};
struct EndpointConstraint { V normal{}; int split=0; };
__device__ rd::Trajectory trajectory(const RigidWork& rigid,int node) {
    const int body=rigid.scene.node_body[node];rd::Trajectory path;
    for(int k=0;k<3;++k) {
        path.center[k]=rigid.state.com[body][k];
        path.delta_center[k]=rigid.delta_com[body][k];
        path.omega[k]=rigid.state.omega[body][k];
        path.delta_omega[k]=rigid.delta_omega[body][k];
        path.reference[k]=rigid.scene.reference[node][k];
    }
    for(int k=0;k<4;++k)path.q0[k]=rigid.state.previous_orientation[body][k];
    path.dt=rigid.dt;return path;
}
__device__ rd::Interval project(const rd::Box& box,V normal,V point) {
    return rd::plane_projection(box,normal.v,point.v);
}
__device__ rd::Box point_box(V point) {
    rd::Box box;for(int k=0;k<3;++k)box.lo[k]=box.hi[k]=point.v[k];return box;
}
__device__ bool safe_result(rd::Result result,int* status) {
    if(rd::is_fatal(result.status)||!dat_detail::finite(result.factor)
       ||result.factor<0||result.factor>1) {
        fail(status,DatRigidTrajectory);return false;
    }
    return true;
}
__global__ void prepare_bodies(std::size_t nodes,RigidWork rigid,
    unsigned* membership,int* status) {
    const auto body=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(body>=rigid.scene.bodies||failed(status))return;
    const auto begin=rigid.scene.node_offsets[body],end=rigid.scene.node_offsets[body+1];
    if((body==0&&begin!=0)||begin>=end||end>nodes) {fail(status,DatRigidOwnership);return;}
    double qnorm=0;
    for(int k=0;k<4;++k) {
        const double q=rigid.state.previous_orientation[body][k];
        if(!dat_detail::finite(q)){fail(status,DatInvalidInput);return;}
        qnorm+=q*q;
    }
    if(!dat_detail::finite(qnorm)||::fabs(qnorm-1)>1e-10) {fail(status,DatInvalidInput);return;}
    for(int k=0;k<3;++k) {
        const double c=rigid.state.com[body][k],w=rigid.state.omega[body][k];
        const double dc=rigid.delta_com[body][k],dw=rigid.delta_omega[body][k];
        if(!dat_detail::finite(c)||!dat_detail::finite(w)||!dat_detail::finite(dc)
           ||!dat_detail::finite(dw)||::fabs(w)>::sqrt(DBL_MAX)/8
           ||::fabs(dw)>::sqrt(DBL_MAX)/8
           ||(!rigid.scene.translate[body]&&dc!=0)||(!rigid.scene.rotate[body]&&dw!=0)) {
            fail(status,DatInvalidInput);return;
        }
    }
    rigid.factors[body]=1;
    for(auto i=begin;i<end;++i) {
        const int node=rigid.scene.nodes[i];
        if(node<0||std::size_t(node)>=nodes||rigid.scene.node_body[node]!=int(body)) {
            fail(status,DatRigidOwnership);return;
        }
        atomicAdd(membership+node,1u);
    }
}
__global__ void prepare_mixed_nodes(std::size_t count,const double (*positions)[3],
    const double (*raw)[3],double (*capped)[3],double* factors,BpAabb* boxes,
    double limit,double radius,RigidWork rigid,const unsigned* membership,int* status) {
    const auto node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=count||failed(status))return;
    const int body=rigid.scene.node_body[node];
    if(body< -1||body>=int(rigid.scene.bodies)||membership[node]!=(body<0?0u:1u)) {
        fail(status,DatRigidOwnership);return;
    }
    const V x=dat_detail::load(positions[node]),update=dat_detail::load(raw[node]);
    if(!dat_detail::finite(x)||!dat_detail::finite(update)){fail(status,DatInvalidInput);return;}
    for(int k=0;k<3;++k) {
        const double lo=x.v[k]-radius,hi=x.v[k]+radius;
        if(!dat_detail::finite(lo)||!dat_detail::finite(hi)){fail(status,DatInvalidInput);return;}
        boxes[node].lo[k]=lo;boxes[node].hi[k]=hi;
    }
    factors[node]=1;
    if(body<0) {
        const V value=dat_detail::capped_update(update,limit);
        for(int k=0;k<3;++k)capped[node][k]=value.v[k];
        return;
    }
    const auto path=trajectory(rigid,int(node));
    double q[4],expected[3];
    rigid_orientation(path.q0,path.omega,path.dt,q);
    rigid_world_position(path.reference,path.center,q,expected);
    double scale=1;
    for(int k=0;k<3;++k)scale=dat_detail::maximum(scale,
        dat_detail::maximum(::fabs(expected[k]),::fabs(x.v[k])));
    for(int k=0;k<3;++k)if(!dat_detail::finite(path.reference[k])
       ||!dat_detail::finite(expected[k])||::fabs(expected[k]-x.v[k])>128*DBL_EPSILON*scale) {
        fail(status,DatInvalidInput);return;
    }
    if(rd::prepare_trajectory(path,rigid.prepared[node])!=rd::Status::Ok) {
        fail(status,DatRigidTrajectory);return;
    }
    const double cap=load_factor(rigid.factors+body);
    const auto radius_factor=rd::radius_step_capped(path,positions[node],limit,cap,40,4096,rigid.prepared+node);
    if(safe_result(radius_factor,status))atomic_min_factor(rigid.factors+body,radius_factor.factor);
}
// This second pass starts only after all vertices have reduced their body's
// curved-path radius factor. Plane allocation uses local motion to avoid loss
// from subtracting large rounded world coordinates.
__global__ void mixed_proposal_motion(std::size_t count,const double (*positions)[3],
    double (*capped)[3],RigidWork rigid,int* status) {
    const auto node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=count||failed(status))return;
    const int body=rigid.scene.node_body[node];if(body<0)return;
    const double alpha=rigid.factors[body];auto path=trajectory(rigid,int(node));
    bool rotating=false;for(int k=0;k<3;++k)rotating|=path.delta_omega[k]!=0;
    if(!rotating) {
        for(int k=0;k<3;++k)capped[node][k]=alpha*path.delta_center[k];
    } else {
        for(int k=0;k<3;++k)path.center[k]=0;
        double from[3],to[3];
        if(rd::evaluate(path,0,from)!=rd::Status::Ok||rd::evaluate(path,alpha,to)!=rd::Status::Ok) {
            fail(status,DatRigidTrajectory);return;
        }
        for(int k=0;k<3;++k)capped[node][k]=to[k]-from[k];
    }
    for(int k=0;k<3;++k)if(!dat_detail::finite(capped[node][k]))fail(status,DatRigidTrajectory);
}
__global__ void prepare_kernel(std::size_t count,const double (*positions)[3],
    const double (*raw)[3],double (*capped)[3],double* factors,BpAabb* boxes,
    double limit,double radius,int* status) {
    const auto node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=count||failed(status))return;
    const V x=dat_detail::load(positions[node]),update=dat_detail::load(raw[node]);
    if(!dat_detail::finite(x)||!dat_detail::finite(update)){fail(status,DatInvalidInput);return;}
    const V value=dat_detail::capped_update(update,limit);
    for(int k=0;k<3;++k) {
        const double lo=x.v[k]-radius,hi=x.v[k]+radius;
        if(!dat_detail::finite(lo)||!dat_detail::finite(hi)){fail(status,DatInvalidInput);return;}
        capped[node][k]=value.v[k];boxes[node].lo[k]=lo;boxes[node].hi[k]=hi;
    }
    factors[node]=1;
}
struct Work {
    std::size_t nodes,pairs;
    const double (*positions)[3];
    const double (*capped)[3];
    const BpContact* contacts;
    double* factors;
    unsigned *exact_count,*retry_count;
    int *exact_queue,*retry_queue;
    std::uint32_t* exact_words;
    int* exact_free_slots;
    DatParameters parameters;
    bool mixed=false;
    RigidWork rigid{};
    EndpointConstraint* constraints=nullptr;
};
// The public entry point fixes the mode, so pure deformable kernels never
// access optional rigid ownership or trajectory storage.
template<bool Mixed> __device__ bool gather(const Work& w,std::size_t index,V positions[4],V updates[4],int* status) {
    const auto& pair=w.contacts[index];
    if(pair.kind!=0&&pair.kind!=1){fail(status,DatInvalidInput);return false;}
    for(int i=0;i<4;++i) {
        const int node=pair.vertices[i];
        if(node<0||std::size_t(node)>=w.nodes){fail(status,DatInvalidInput);return false;}
        for(int j=0;j<i;++j)if(node==pair.vertices[j]){fail(status,DatInvalidInput);return false;}
        positions[i]=dat_detail::load(w.positions[node]);updates[i]=dat_detail::load(w.capped[node]);
    }
    if constexpr(Mixed) {
        const int* owner=w.rigid.scene.node_body;
        const int split=pair.kind==0?1:2;
        for(int i=1;i<split;++i)if(owner[pair.vertices[i]]!=owner[pair.vertices[0]]) {
            fail(status,DatRigidOwnership);return false;
        }
        for(int i=split+1;i<4;++i)if(owner[pair.vertices[i]]!=owner[pair.vertices[split]]) {
            fail(status,DatRigidOwnership);return false;
        }
        if(owner[pair.vertices[0]]>=0&&owner[pair.vertices[0]]==owner[pair.vertices[split]])return false;
    }
    return true;
}
__device__ int constrain_mixed(const Work& w,std::size_t index,const V x[4],
    const V update[4],V separation,int* status) {
    const int split=w.contacts[index].kind==0?1:2;
    dat_detail::PairPlane selected;
    const int selection=dat_detail::select_pair_plane(x,split,separation,w.parameters.query_radius,selected);
    if(selection)return selection;
    if(selected.outside_query)return 0;
    const V normal=selected.normal,origin=x[0];
    bool rigid_pair=false;
    for(int i=0;i<4;++i)rigid_pair|=w.rigid.scene.node_body[w.contacts[index].vertices[i]]>=0;
    if(!rigid_pair) {
        double factors[4];
        const int code=dat_detail::plane_factors_from_plane(update,split,selected,w.parameters,factors);
        if(code)return code;
        w.constraints[index]={normal,split};
        for(int i=0;i<4;++i)atomic_min_factor(w.factors+w.contacts[index].vertices[i],factors[i]);
        return 0;
    }
    const double left=selected.left,right=selected.right;
    double velocity[4],a=0,b=0;
    for(int i=0;i<4;++i) {
        velocity[i]=dat_detail::dot(update[i],normal);
        if(!dat_detail::finite(velocity[i]))return DatUnresolvedGeometry;
        if(i<split)a=dat_detail::maximum(a,velocity[i]);
        else b=dat_detail::maximum(b,-velocity[i]);
    }
    const double gap=right-left;
    if(!(gap>0)||!dat_detail::finite(gap))return DatUnresolvedGeometry;
    const double scale=dat_detail::maximum(a,b);
    a=scale>0?a/scale:1;b=scale>0?b/scale:1;
    // Every side receives a strict interior plane even when a rotating endpoint
    // appears stationary or retreating; endpoint direction is only a proposal.
    a=dat_detail::maximum(a,1e-3);b=dat_detail::maximum(b,1e-3);
    const double budget=gap*(a/(a+b));
    const V plane=dat_detail::add(origin,dat_detail::multiply(normal,left+budget));
    if(!dat_detail::finite(plane))return DatUnresolvedGeometry;
    w.constraints[index]={normal,split};
    for(int i=0;i<4;++i) {
        const int node=w.contacts[index].vertices[i],body=w.rigid.scene.node_body[node];
        const bool side_a=i<split;
        const auto stored=project(point_box(x[i]),normal,plane);
        const double margin=side_a?-stored.hi:stored.lo;
        if(!dat_detail::finite(stored.lo)||!dat_detail::finite(stored.hi)||!(margin>0)) {
            return DatUnresolvedGeometry;
        }
        if(body>=0) {
            const auto initial=project(w.rigid.prepared[node].initial_box,normal,plane);
            if(side_a?!(initial.hi<0):!(initial.lo>0))return DatRigidTrajectory;
            const V safe_normal=side_a?dat_detail::multiply(normal,-1):normal;
            const auto result=rd::plane_step(trajectory(w.rigid,node),safe_normal.v,plane.v,w.parameters.gamma,40,4096,w.rigid.prepared+node);
            if(!safe_result(result,status))return DatRigidTrajectory;
            atomic_min_factor(w.rigid.factors+body,result.factor);
        } else {
            const V safe_normal=side_a?normal:dat_detail::multiply(normal,-1);
            const auto speed=rd::dot_scalar_interval(safe_normal.v,update[i].v);
            const double approach=speed.hi;
            if(approach<=0)continue;
            const double allowed=rd::round_down(w.parameters.gamma*margin);
            if(approach>allowed)atomic_min_factor(w.factors+node,
                dat_detail::clamp(rd::round_down(allowed/approach)));
        }
    }
    return 0;
}
template<bool Mixed> __device__ int constrain(const Work& w,std::size_t index,const V x[4],const V update[4],V separation,int* status) {
    if constexpr(Mixed)return constrain_mixed(w,index,x,update,separation,status);
    double factors[4];
    const int code=dat_detail::plane_factors(x,update,w.contacts[index].kind==0?1:2,separation,w.parameters,factors);
    if(code)return code;
    for(int i=0;i<4;++i)atomic_min_factor(w.factors+w.contacts[index].vertices[i],factors[i]);
    return 0;
}
template<bool Mixed> __global__ void ordinary_pairs(Work work,int* status) {
    const auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=work.pairs||failed(status))return;
    V positions[4],updates[4],separation;
    if(!gather<Mixed>(work,index,positions,updates,status))return;
    if(dat_detail::aabb_separated(positions,work.contacts[index].kind==0?1:2,work.parameters.query_radius))return;
    int code=work.contacts[index].kind==0
        ?dat_detail::vertex_triangle(positions,separation):dat_detail::edge_edge(positions,separation);
    if(!code)code=constrain<Mixed>(work,index,positions,updates,separation,status);
    if(code==dat_detail::NeedsExact) {
        const unsigned next=atomicAdd(work.exact_count,1u);work.exact_queue[next]=static_cast<int>(index);
    } else if(code)fail(status,code);
}
template<int Limbs,bool Retry,bool Mixed> __global__ void exact_pairs(Work work,unsigned workers,int* status) {
    const unsigned worker=blockIdx.x*blockDim.x+threadIdx.x;
    if(worker>=workers||failed(status))return;
    const unsigned count=Retry?*work.retry_count:*work.exact_count;
    const int* queue=Retry?work.retry_queue:work.exact_queue;
    for(unsigned ordinal=worker;ordinal<count;ordinal+=workers) {
        if(failed(status))return;
        const int index=queue[ordinal];V positions[4],updates[4],separation;
        if(!gather<Mixed>(work,std::size_t(index),positions,updates,status))return;
        exact::DevicePool pool;
        exact::initialize_pool(pool,work.exact_words+std::size_t(worker)*Limbs*exact_slots,
            work.exact_free_slots+std::size_t(worker)*exact_slots,Limbs,exact_slots);
        int code=dat_detail::exact_pair_geometry<Limbs>(positions,
            work.contacts[index].kind==0?1:2,separation);
        const bool arena_failed=pool.failed||pool.available!=pool.slot_count;
        exact::release_pool();
        if(arena_failed){fail(status,DatExactCapacity);return;}
        if(!code) {
            code=constrain<Mixed>(work,std::size_t(index),positions,updates,separation,status);
            if(code==dat_detail::NeedsExact)code=DatUnresolvedGeometry;
        }
        if(code==DatExactCapacity&&!Retry) {
            const unsigned next=atomicAdd(work.retry_count,1u);work.retry_queue[next]=index;
        } else if(code){fail(status,code);return;}
    }
}
__global__ void finish_kernel(std::size_t count,const double (*capped)[3],const double* factors,
    double (*accepted)[3],int* status) {
    const auto node=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=count||failed(status))return;
    for(int k=0;k<3;++k) {
        const double value=capped[node][k]*factors[node];
        if(!dat_detail::finite(value)){fail(status,DatNonfiniteOutput);return;}
        accepted[node][k]=value;
    }
}
struct EndpointWork {
    Work work;
    double (*positions)[3],(*com)[3],(*omega)[3],(*orientation)[4];
    unsigned *bad_nodes,*bad_bodies,*unresolved,*converged;
};
__device__ bool inactive(const EndpointWork& e,int* status) {
    return failed(status)||atomicAdd(e.converged,0u)!=0;
}
__device__ void mark(const EndpointWork& e,int node) {
    const int body=e.work.rigid.scene.node_body[node];
    if(body<0)atomicExch(e.bad_nodes+node,1u);
    else atomicExch(e.bad_bodies+body,1u);
    atomicExch(e.unresolved,1u);
}
__device__ void endpoint_poses_at(EndpointWork e,std::size_t body,int* status) {
    if(body>=e.work.rigid.scene.bodies||inactive(e,status))return;
    const auto& r=e.work.rigid;const double alpha=r.factors[body];
    for(int k=0;k<3;++k) {
        e.com[body][k]=r.state.com[body][k]+alpha*r.delta_com[body][k];
        e.omega[body][k]=r.state.omega[body][k]+alpha*r.delta_omega[body][k];
    }
    rigid_orientation(r.state.previous_orientation[body],e.omega[body],r.dt,e.orientation[body]);
}
__global__ void endpoint_poses(EndpointWork e,int* status) {
    endpoint_poses_at(e,std::size_t(blockIdx.x)*blockDim.x+threadIdx.x,status);
}
__device__ void endpoint_nodes_at(EndpointWork e,std::size_t node,int* status) {
    if(node>=e.work.nodes||inactive(e,status))return;
    const auto& w=e.work;const int body=w.rigid.scene.node_body[node];
    if(body<0)for(int k=0;k<3;++k)
        e.positions[node][k]=w.positions[node][k]+w.capped[node][k]*w.factors[node];
    else rigid_world_position(w.rigid.scene.reference[node],e.com[body],e.orientation[body],e.positions[node]);
    const V endpoint=dat_detail::load(e.positions[node]);
    if(!dat_detail::finite(endpoint)) {mark(e,int(node));return;}
    if(body>=0) {
        for(int k=0;k<3;++k)if(!dat_detail::finite(e.com[body][k])||!dat_detail::finite(e.omega[body][k])) {
            mark(e,int(node));return;
        }
        for(int k=0;k<4;++k)if(!dat_detail::finite(e.orientation[body][k])) {mark(e,int(node));return;}
    }
    const double limit=(.5*w.parameters.gamma)*w.parameters.query_radius;
    if(!(rd::norm_upper(point_box(endpoint),w.positions[node])<=limit))mark(e,int(node));
}
__global__ void endpoint_nodes(EndpointWork e,int* status) {
    endpoint_nodes_at(e,std::size_t(blockIdx.x)*blockDim.x+threadIdx.x,status);
}
__device__ void endpoint_pairs_at(EndpointWork e,std::size_t index,int* status) {
    if(index>=e.work.pairs||inactive(e,status))return;
    const auto constraint=e.work.constraints[index];if(!constraint.split)return;
    const auto pair=e.work.contacts[index];
    const V origin=dat_detail::load(e.positions[pair.vertices[0]]);
    double left=-INFINITY,right=INFINITY;
    for(int i=0;i<4;++i) {
        const auto interval=project(point_box(dat_detail::load(e.positions[pair.vertices[i]])),constraint.normal,origin);
        if(i<constraint.split)left=dat_detail::maximum(left,interval.hi);
        else right=dat_detail::minimum(right,interval.lo);
    }
    if(!(right>left))for(int i=0;i<4;++i)mark(e,pair.vertices[i]);
}
__global__ void endpoint_pairs(EndpointWork e,int* status) {
    endpoint_pairs_at(e,std::size_t(blockIdx.x)*blockDim.x+threadIdx.x,status);
}
__global__ void endpoint_join(EndpointWork e,int* status) {
    if(inactive(e,status))return;
    if(!*e.unresolved)*e.converged=1;
}
__device__ void endpoint_backtrack_at(EndpointWork e,std::size_t index,int attempt,int* status) {
    if(inactive(e,status))return;
    // The final retry uses zero, exactly as the CPU filter. At zero even the
    // represented pose must still pass; failure is never accepted as no motion.
    constexpr int last_halving=std::numeric_limits<double>::digits+1;
    if(index<e.work.nodes&&e.bad_nodes[index])
        e.work.factors[index]=attempt==last_halving?0:.5*e.work.factors[index];
    if(index<e.work.rigid.scene.bodies&&e.bad_bodies[index])
        e.work.rigid.factors[index]=attempt==last_halving?0:.5*e.work.rigid.factors[index];
}
// Keep the common single-halving retry parallel as well. This joined kernel
// reads each index's flags before clearing them, then the next launch may
// recompute every endpoint without a grid-wide in-kernel barrier.
__global__ void endpoint_next_attempt(EndpointWork e,int* status) {
    const auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(inactive(e,status))return;
    endpoint_backtrack_at(e,index,0,status);
    if(index<e.work.nodes)e.bad_nodes[index]=0;
    if(index<e.work.rigid.scene.bodies)e.bad_bodies[index]=0;
    if(index==0)*e.unresolved=0;
}
// The normal case uses two parallel passes above. Only a still-failed
// represented endpoint enters this bounded retry; every phase joins the whole
// block before another phase reads its writes. All decisions stay on device.
__global__ void endpoint_retry(EndpointWork e,int* status) {
    __shared__ bool stop;
    constexpr int attempts=std::numeric_limits<double>::digits+3;
    for(int attempt=2;attempt<attempts;++attempt) {
        if(threadIdx.x==0)stop=inactive(e,status);
        __syncthreads();
        if(stop)return;
        for(std::size_t i=threadIdx.x;i<e.work.nodes;i+=blockDim.x)
            endpoint_backtrack_at(e,i,attempt-1,status);
        __syncthreads();
        if(threadIdx.x==0)*e.unresolved=0;
        for(std::size_t i=threadIdx.x;i<e.work.nodes;i+=blockDim.x)e.bad_nodes[i]=0;
        for(std::size_t i=threadIdx.x;i<e.work.rigid.scene.bodies;i+=blockDim.x)e.bad_bodies[i]=0;
        __syncthreads();
        for(std::size_t i=threadIdx.x;i<e.work.rigid.scene.bodies;i+=blockDim.x)endpoint_poses_at(e,i,status);
        __syncthreads();
        for(std::size_t i=threadIdx.x;i<e.work.nodes;i+=blockDim.x)endpoint_nodes_at(e,i,status);
        __syncthreads();
        for(std::size_t i=threadIdx.x;i<e.work.pairs;i+=blockDim.x)endpoint_pairs_at(e,i,status);
        __syncthreads();
        if(threadIdx.x==0 && !failed(status) && !*e.unresolved)*e.converged=1;
        __syncthreads();
    }
}
__global__ void endpoint_finish(EndpointWork e,int* status) {
    if(!failed(status)&&!*e.converged)fail(status,DatEndpointUnproved);
}
} // namespace

struct GpuDatFilter::Impl {
    Buffer<V> capped;
    Buffer<double> factors;
    Buffer<BpAabb> boxes;
    Buffer<int> exact_queue,retry_queue,exact_free_slots;
    Buffer<unsigned> exact_count,retry_count;
    Buffer<std::uint32_t> exact_words;
    Buffer<double> body_factors;
    Buffer<rd::PreparedTrajectory> prepared;
    Buffer<unsigned> membership,bad_nodes,bad_bodies,unresolved,converged;
    Buffer<EndpointConstraint> constraints;
    std::size_t count=0;
    const double (*positions)[3]=nullptr;
    const double (*raw)[3]=nullptr;
    int* error=nullptr;
    DatParameters parameters;
    cudaStream_t stream=nullptr;
    bool ready=false;
    bool mixed=false;
    RigidWork rigid{};
};

GpuDatFilter::GpuDatFilter():impl_(new Impl){}
GpuDatFilter::~GpuDatFilter()=default;

const BpAabb* GpuDatFilter::prepare(std::size_t count,const double (*positions)[3],
    const double (*raw)[3],const DatParameters& parameters,int* error,cudaStream_t stream) {
    auto& p=*impl_;p.ready=false;p.mixed=false;
    const double limit=(0.5*parameters.gamma)*parameters.query_radius;
    const double radius=parameters.query_radius*parameters.node_box_padding;
    if(!std::isfinite(parameters.gamma)||parameters.gamma<=0||parameters.gamma>=1
       ||!std::isfinite(parameters.query_radius)||parameters.query_radius<=0
       ||!std::isfinite(parameters.node_box_padding)||parameters.node_box_padding<1
       ||!(limit>0)||!std::isfinite(radius))throw std::invalid_argument("GPU DAT requires 0<gamma<1, positive finite radius/limit and finite padding>=1");
    if(count>INT_MAX||!error||(count&&(!positions||!raw)))throw std::invalid_argument("GPU DAT invalid node extent/pointer");
    p.count=count;p.positions=positions;p.raw=raw;p.error=error;p.parameters=parameters;p.stream=stream;
    p.capped.reserve(count);p.factors.reserve(count);p.boxes.reserve(count);
    if(count)prepare_kernel<<<blocks(count),threads,0,stream>>>(count,positions,raw,
        reinterpret_cast<double(*)[3]>(p.capped.data),p.factors.data,p.boxes.data,limit,radius,error);
    checked(cudaGetLastError(),"prepare launch");p.ready=true;return p.boxes.data;
}

void GpuDatFilter::filter(const BroadPhaseView& broad,double (*accepted)[3],int* error,cudaStream_t stream) {
    auto& p=*impl_;
    const auto overlaps=[&](const void* source) {
        const auto a=reinterpret_cast<std::uintptr_t>(accepted),b=reinterpret_cast<std::uintptr_t>(source);
        const auto bytes=p.count*sizeof(V);
        return p.count&&(a<=b?b-a<bytes:a-b<bytes);
    };
    if(!p.ready||p.mixed||error!=p.error||stream!=p.stream||(p.count&&!accepted)
       ||overlaps(p.positions)||overlaps(p.raw)
       ||broad.contact_count>INT_MAX||(broad.contact_count&&!broad.contacts))
        throw std::invalid_argument("GPU DAT requires one matching prepared view/stream and valid candidates");
    p.ready=false;
    const auto pairs=broad.contact_count;
    if(pairs) {
        p.exact_queue.reserve(pairs);p.retry_queue.reserve(pairs);p.exact_count.reserve(1);p.retry_count.reserve(1);
        const auto small=std::min<std::size_t>(max_small_workers,pairs),large=std::min<std::size_t>(max_large_workers,pairs);
        p.exact_words.reserve(std::max(small*small_limbs,large*large_limbs)*exact_slots);
        p.exact_free_slots.reserve(small*exact_slots);
        checked(cudaMemsetAsync(p.exact_count.data,0,sizeof(unsigned),stream),"clear exact queue");
        checked(cudaMemsetAsync(p.retry_count.data,0,sizeof(unsigned),stream),"clear retry queue");
        const Work work{p.count,pairs,p.positions,reinterpret_cast<const double(*)[3]>(p.capped.data),
            broad.contacts,p.factors.data,p.exact_count.data,p.retry_count.data,p.exact_queue.data,p.retry_queue.data,
            p.exact_words.data,p.exact_free_slots.data,p.parameters};
        ordinary_pairs<false><<<blocks(pairs),threads,0,stream>>>(work,error);
        exact_pairs<small_limbs,false,false><<<static_cast<unsigned>((small+small_worker_threads-1)/small_worker_threads),small_worker_threads,0,stream>>>(work,static_cast<unsigned>(small),error);
        exact_pairs<large_limbs,true,false><<<static_cast<unsigned>(large),1,0,stream>>>(work,static_cast<unsigned>(large),error);
    }
    if(p.count)finish_kernel<<<blocks(p.count),threads,0,stream>>>(p.count,
        reinterpret_cast<const double(*)[3]>(p.capped.data),p.factors.data,accepted,error);
    checked(cudaGetLastError(),"filter launch");
}

const BpAabb* GpuDatFilter::prepare(std::size_t count,const double (*positions)[3],
    const double (*raw)[3],const RigidScene& scene,const RigidState& state,
    const double (*delta_com)[3],const double (*delta_omega)[3],double dt,
    const DatParameters& parameters,int* error,cudaStream_t stream) {
    auto& p=*impl_;p.ready=false;p.mixed=true;
    const double limit=(.5*parameters.gamma)*parameters.query_radius;
    const double radius=parameters.query_radius*parameters.node_box_padding;
    if(!std::isfinite(parameters.gamma)||parameters.gamma<=0||parameters.gamma>=1
       ||!std::isfinite(parameters.query_radius)||parameters.query_radius<=0
       ||!std::isfinite(parameters.node_box_padding)||parameters.node_box_padding<1
       ||!(limit>0)||!std::isfinite(radius)||!(dt>0)||!std::isfinite(dt)
       ||!(dt*dt>0)||!std::isfinite(dt*dt))
        throw std::invalid_argument("GPU mixed DAT requires valid finite filter parameters and dt");
    if(count>INT_MAX||scene.bodies>count||!error
       ||(count&&(!positions||!raw||!scene.node_body))
       ||(scene.bodies&&(!scene.reference||!scene.node_offsets||!scene.nodes
           ||!scene.translate||!scene.rotate||!state.com||!state.omega
           ||!state.previous_orientation||!delta_com||!delta_omega)))
        throw std::invalid_argument("GPU mixed DAT invalid extent/pointer");
    p.count=count;p.positions=positions;p.raw=raw;p.error=error;p.parameters=parameters;p.stream=stream;
    p.capped.reserve(count);p.factors.reserve(count);p.boxes.reserve(count);
    p.body_factors.reserve(scene.bodies);p.prepared.reserve(count);p.membership.reserve(count);
    p.rigid={scene,state,delta_com,delta_omega,dt,p.body_factors.data,p.prepared.data};
    if(count)checked(cudaMemsetAsync(p.membership.data,0,count*sizeof(unsigned),stream),"clear rigid membership");
    if(scene.bodies)prepare_bodies<<<blocks(scene.bodies),threads,0,stream>>>(count,p.rigid,p.membership.data,error);
    if(count) {
        prepare_mixed_nodes<<<blocks(count),threads,0,stream>>>(count,positions,raw,
            reinterpret_cast<double(*)[3]>(p.capped.data),p.factors.data,p.boxes.data,
            limit,radius,p.rigid,p.membership.data,error);
        mixed_proposal_motion<<<blocks(count),threads,0,stream>>>(count,positions,
            reinterpret_cast<double(*)[3]>(p.capped.data),p.rigid,error);
    }
    checked(cudaGetLastError(),"mixed prepare launch");p.ready=true;return p.boxes.data;
}

void GpuDatFilter::filter(const BroadPhaseView& broad,double (*positions)[3],
    double (*com)[3],double (*omega)[3],double (*orientation)[4],int* error,cudaStream_t stream) {
    auto& p=*impl_;
    struct Range { const void* data; std::size_t bytes; };
    const auto overlaps=[](Range a,Range b) {
        if(!a.bytes||!b.bytes)return false;
        const auto x=reinterpret_cast<std::uintptr_t>(a.data),y=reinterpret_cast<std::uintptr_t>(b.data);
        return x<=y?y-x<a.bytes:x-y<b.bytes;
    };
    const std::size_t body_bytes=p.rigid.scene.bodies*sizeof(V),node_bytes=p.count*sizeof(V);
    const Range outputs[]={{positions,node_bytes},{com,body_bytes},{omega,body_bytes},
        {orientation,p.rigid.scene.bodies*4*sizeof(double)}};
    const auto optional=[](const void* data,std::size_t bytes) {return Range{data,data?bytes:0};};
    const Range inputs[]={{p.positions,node_bytes},{p.raw,node_bytes},
        {p.rigid.scene.reference,p.rigid.scene.bodies?node_bytes:0},
        {p.rigid.scene.node_body,p.count*sizeof(int)},
        optional(p.rigid.scene.nodes,p.rigid.scene.bodies?sizeof(int):0),
        optional(p.rigid.scene.node_offsets,(p.rigid.scene.bodies+1)*sizeof(std::size_t)),
        optional(p.rigid.scene.translate,p.rigid.scene.bodies*sizeof(unsigned char)),
        optional(p.rigid.scene.rotate,p.rigid.scene.bodies*sizeof(unsigned char)),
        {p.rigid.state.com,body_bytes},{p.rigid.state.omega,body_bytes},
        {p.rigid.state.previous_orientation,p.rigid.scene.bodies*4*sizeof(double)},
        optional(p.rigid.state.orientation,p.rigid.scene.bodies*4*sizeof(double)),
        optional(p.rigid.state.velocity,body_bytes),optional(p.rigid.state.previous_com,body_bytes),
        optional(p.rigid.state.previous_velocity,body_bytes),optional(p.rigid.state.previous_omega,body_bytes),
        {p.rigid.delta_com,body_bytes},{p.rigid.delta_omega,body_bytes},
        {broad.contacts,broad.contact_count<=INT_MAX?broad.contact_count*sizeof(BpContact):0},
        {error,sizeof(int)}};
    bool alias=false;
    for(std::size_t i=0;i<4;++i) {
        for(const auto& input:inputs)alias|=overlaps(outputs[i],input);
        for(std::size_t j=0;j<i;++j)alias|=overlaps(outputs[i],outputs[j]);
    }
    if(!p.ready||!p.mixed||error!=p.error||stream!=p.stream||alias||(p.count&&!positions)
       ||(p.rigid.scene.bodies&&(!com||!omega||!orientation))
       ||broad.contact_count>INT_MAX||(broad.contact_count&&!broad.contacts))
        throw std::invalid_argument("GPU mixed DAT requires matching preparation, distinct output storage and valid candidates");
    p.ready=false;
    const auto pairs=broad.contact_count;
    p.constraints.reserve(pairs);
    if(pairs)checked(cudaMemsetAsync(p.constraints.data,0,pairs*sizeof(EndpointConstraint),stream),"clear endpoint constraints");
    Work work{p.count,pairs,p.positions,reinterpret_cast<const double(*)[3]>(p.capped.data),
        broad.contacts,p.factors.data,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,p.parameters};
    work.mixed=true;work.rigid=p.rigid;work.constraints=p.constraints.data;
    if(pairs) {
        p.exact_queue.reserve(pairs);p.retry_queue.reserve(pairs);p.exact_count.reserve(1);p.retry_count.reserve(1);
        const auto small=std::min<std::size_t>(max_small_workers,pairs),large=std::min<std::size_t>(max_large_workers,pairs);
        p.exact_words.reserve(std::max(small*small_limbs,large*large_limbs)*exact_slots);
        p.exact_free_slots.reserve(small*exact_slots);
        checked(cudaMemsetAsync(p.exact_count.data,0,sizeof(unsigned),stream),"clear mixed exact queue");
        checked(cudaMemsetAsync(p.retry_count.data,0,sizeof(unsigned),stream),"clear mixed retry queue");
        work.exact_count=p.exact_count.data;work.retry_count=p.retry_count.data;
        work.exact_queue=p.exact_queue.data;work.retry_queue=p.retry_queue.data;
        work.exact_words=p.exact_words.data;work.exact_free_slots=p.exact_free_slots.data;
        ordinary_pairs<true><<<blocks(pairs),threads,0,stream>>>(work,error);
        exact_pairs<small_limbs,false,true><<<static_cast<unsigned>((small+small_worker_threads-1)/small_worker_threads),small_worker_threads,0,stream>>>(work,static_cast<unsigned>(small),error);
        exact_pairs<large_limbs,true,true><<<static_cast<unsigned>(large),1,0,stream>>>(work,static_cast<unsigned>(large),error);
    }
    p.bad_nodes.reserve(p.count);p.bad_bodies.reserve(p.rigid.scene.bodies);
    p.unresolved.reserve(1);p.converged.reserve(1);
    checked(cudaMemsetAsync(p.converged.data,0,sizeof(unsigned),stream),"clear endpoint convergence");
    const EndpointWork endpoints{work,positions,com,omega,orientation,p.bad_nodes.data,p.bad_bodies.data,
        p.unresolved.data,p.converged.data};
    // Scalable first and single-halving passes, then device-controlled retries.
    checked(cudaMemsetAsync(p.unresolved.data,0,sizeof(unsigned),stream),"clear endpoint retry");
    if(p.count)checked(cudaMemsetAsync(p.bad_nodes.data,0,p.count*sizeof(unsigned),stream),"clear bad nodes");
    if(p.rigid.scene.bodies)checked(cudaMemsetAsync(p.bad_bodies.data,0,p.rigid.scene.bodies*sizeof(unsigned),stream),"clear bad bodies");
    if(p.rigid.scene.bodies)endpoint_poses<<<blocks(p.rigid.scene.bodies),threads,0,stream>>>(endpoints,error);
    if(p.count)endpoint_nodes<<<blocks(p.count),threads,0,stream>>>(endpoints,error);
    if(pairs)endpoint_pairs<<<blocks(pairs),threads,0,stream>>>(endpoints,error);
    endpoint_join<<<1,1,0,stream>>>(endpoints,error);
    if(p.count)endpoint_next_attempt<<<blocks(p.count),threads,0,stream>>>(endpoints,error);
    if(p.rigid.scene.bodies)endpoint_poses<<<blocks(p.rigid.scene.bodies),threads,0,stream>>>(endpoints,error);
    if(p.count)endpoint_nodes<<<blocks(p.count),threads,0,stream>>>(endpoints,error);
    if(pairs)endpoint_pairs<<<blocks(pairs),threads,0,stream>>>(endpoints,error);
    endpoint_join<<<1,1,0,stream>>>(endpoints,error);
    endpoint_retry<<<1,threads,0,stream>>>(endpoints,error);
    endpoint_finish<<<1,1,0,stream>>>(endpoints,error);
    checked(cudaGetLastError(),"mixed filter launch");
}

} // namespace ipc_gpu::solver
