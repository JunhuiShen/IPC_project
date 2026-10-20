#include "resident_level_schedule.h"
#include "resident_level_schedule.cuh"
#include "resident_box_certificate.cuh"
#include "resident_memory.h"
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <climits>
#include <limits>

namespace ipc_gpu::resident::level_predictor {
namespace {
using Key=std::uint64_t;
struct RetainContact { __host__ __device__ bool operator()(unsigned char separated)const{return separated==0;} };
unsigned blocks(std::size_t n){return static_cast<unsigned>((n+127)/128);}
__global__ void validate_nodes(int n,const double (*x)[3],const double (*t)[3],
    const BpAabb* boxes,const unsigned char* moving,int* error) {
    auto v=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(v>=static_cast<std::size_t>(n)||atomicAdd(error,0))return;
    if(moving[v]>1){atomicCAS(error,0,71);return;}
    for(int a=0;a<3;++a) {
        if(!::isfinite(x[v][a])||!::isfinite(t[v][a])||!::isfinite(boxes[v].lo[a])
            ||!::isfinite(boxes[v].hi[a])||boxes[v].lo[a]>boxes[v].hi[a]
            ||x[v][a]<boxes[v].lo[a]||x[v][a]>boxes[v].hi[a]
            ||t[v][a]<boxes[v].lo[a]||t[v][a]>boxes[v].hi[a]
            ||(!moving[v]&&x[v][a]!=t[v][a])){atomicCAS(error,0,71);return;}
    }
}
__global__ void emit_graph(int n,const BpContact* contacts,std::size_t count,
    const double (*x)[3],const BpAabb* boxes,double gap,Key* keys,
    unsigned char* separated,std::uint64_t* costs,int* error) {
    auto p=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(p>=count)return;
    for(int i=0;i<12;++i)keys[12*p+i]=UINT64_MAX;
    separated[p]=0;
    if(atomicAdd(error,0))return;
    const auto c=contacts[p];
    if(c.kind!=0&&c.kind!=1){atomicCAS(error,0,71);return;}
    BpAabb region[4];double positions[4][3];
    for(int a=0;a<4;++a) {
        int v=c.vertices[a];
        if(v<0||v>=n){atomicCAS(error,0,71);return;}
        for(int b=0;b<a;++b)if(c.vertices[b]==v){atomicCAS(error,0,71);return;}
        region[a]=boxes[v];
        for(int k=0;k<3;++k)positions[a][k]=x[v][k];
    }
    // Existing GPU outward-rounded complete-box certificate, not a current
    // distance heuristic. Unknown retains all four-node dependencies.
    separated[p]=certify_movement_box_contact(region,positions,c.kind==0,gap);
    if(separated[p])return;
    int e=0;
    for(int a=0;a<4;++a) {
        atomicAdd(reinterpret_cast<unsigned long long*>(costs+c.vertices[a]),1ull);
        for(int b=0;b<4;++b)if(a!=b)
            keys[12*p+e++]=(Key(static_cast<unsigned>(c.vertices[a]))<<32)
                |static_cast<unsigned>(c.vertices[b]);
    }
}
__global__ void csr_rows(const Key* keys,std::size_t count,std::size_t* offsets,int* adj) {
    auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) {
        atomicAdd(reinterpret_cast<unsigned long long*>(offsets+(keys[i]>>32)+1),1ull);
        adj[i]=static_cast<int>(keys[i]&UINT32_MAX);
    }
}
__global__ void color_graph(int n,const std::size_t* row,const int* adj,Scratch scratch,
    Shape* shape,int* error) {
    if(atomicAdd(error,0))return;
    auto result=color(n,row,adj,scratch,*shape);
    if(result!=Status::Ok)atomicCAS(error,0,static_cast<int>(result));
}
__global__ void trial_groups(int n,const int* colors,const int* order,int* offsets,
    Shape shape,int* error) {
    if(atomicAdd(error,0))return;
    int at=0;offsets[0]=0;
    for(int c=0;c<shape.colors;++c) {
        while(at<n&&colors[order[at]]==c)++at;
        offsets[c+1]=at;
    }
}
__global__ void measure(int n,const double (*start)[3],const double (*trial)[3],
    double* movement,int* error) {
    auto v=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(v>=static_cast<std::size_t>(n)||atomicAdd(error,0))return;
    double d[3];for(int a=0;a<3;++a)d[a]=sub_rn(trial[v][a],start[v][a]);
    movement[v]=stable_norm3(d);
    if(!::isfinite(movement[v]))atomicCAS(error,0,73);
}
__global__ void build_levels(int n,const std::size_t* row,const int* adj,
    const unsigned char* moving,const double* movement,Scratch scratch,Shape* shape,int* error) {
    if(atomicAdd(error,0))return;
    auto result=schedule(n,row,adj,moving,movement,scratch,*shape);
    if(result!=Status::Ok)atomicCAS(error,0,static_cast<int>(result));
}
__global__ void project_nodes(int active,const int* order,const int* offsets,int groups,
    const unsigned char* moving,const int* blocks,int block_count,int* owners,int* colors,int* error) {
    auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=static_cast<std::size_t>(active)||atomicAdd(error,0))return;
    const int v=order[i],block=blocks[v];
    if(block<0||block>=block_count){atomicCAS(error,0,71);return;}
    if(!moving[v])return;
    if(atomicCAS(owners+block,-1,v)!=-1){atomicCAS(error,0,71);return;}
    colors[block]=group_of(static_cast<int>(i),groups,offsets);
}
__global__ void validate_block_projection(int n,const unsigned char* moving,
    const int* blocks,int block_count,const int* owners,int* error) {
    auto v=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(v>=static_cast<std::size_t>(n)||atomicAdd(error,0))return;
    const int block=blocks[v];
    if(block<0||block>=block_count||(!moving[v]&&owners[block]!=-1)
        ||(moving[v]&&owners[block]!=static_cast<int>(v)))atomicCAS(error,0,71);
}
template<class T> auto ptr(DeviceArray<T>& a){return thrust::device_pointer_cast(a.data());}
void check(int* error) {
    cuda_checked(cudaGetLastError());int value=0;
    cuda_checked(cudaMemcpy(&value,error,sizeof value,cudaMemcpyDeviceToHost));
    if(value)throw std::runtime_error("private GPU level scheduler failed closed: "+std::to_string(value));
}
}
struct ResidentLevelSchedule::Impl {
    int n=0;bool ready=false,finished=false;Shape shape{};
    DeviceArray<double> start,target,movement;
    DeviceArray<BpAabb> boxes;
    DeviceArray<BpContact> retained;std::size_t retained_count=0;
    DeviceArray<unsigned char> moving,separated;
    DeviceArray<Key> keys;
    DeviceArray<std::uint64_t> costs;
    DeviceArray<std::size_t> rows;
    DeviceArray<int> adjacency,colors,seen,levels,component,queue,trial_order,order,offsets,level_offsets,block_colors,block_owners;
    DeviceArray<Shape> device_shape{1};
    Scratch scratch(){return {colors.data(),seen.data(),levels.data(),component.data(),queue.data(),
        trial_order.data(),order.data(),offsets.data(),level_offsets.data(),static_cast<std::size_t>(n)};}
};
ResidentLevelSchedule::ResidentLevelSchedule():impl_(new Impl){}
ResidentLevelSchedule::~ResidentLevelSchedule()=default;
void ResidentLevelSchedule::prepare(int n,const BpContact* contacts,std::size_t count,
    const double (*start)[3],const double (*target)[3],const BpAabb* boxes,
    const unsigned char* moving,double gap,bool heavy,int* error) {
    auto& p=*impl_;p.ready=false;p.finished=false;
    if(n<0||!error||(!contacts&&count)||((!start||!target||!boxes||!moving)&&n)
        ||!(gap>0)||!std::isfinite(gap)||count>std::size_t(INT_MAX)
        ||count>std::numeric_limits<std::size_t>::max()/12/sizeof(Key))
        throw std::invalid_argument("private level graph pointers/extent/gap");
    static_assert(sizeof(std::size_t)==sizeof(unsigned long long));
    p.n=n;p.start.resize(std::size_t(n)*3);p.target.resize(std::size_t(n)*3);
    p.boxes.resize(n);p.moving.resize(n);p.movement.resize(n);p.separated.resize(count);
    p.keys.resize(count*12);p.costs.resize(n);p.costs.clear();p.rows.resize(std::size_t(n)+1);p.rows.clear();
    for(auto* a:{&p.colors,&p.seen,&p.levels,&p.component,&p.queue,&p.trial_order,&p.order})a->resize(n);
    p.offsets.resize(std::size_t(n)+1);p.level_offsets.resize(std::size_t(n)+1);
    if(n) {
        cuda_checked(cudaMemcpy(p.start.data(),start,std::size_t(n)*3*sizeof(double),cudaMemcpyDeviceToDevice));
        cuda_checked(cudaMemcpy(p.target.data(),target,std::size_t(n)*3*sizeof(double),cudaMemcpyDeviceToDevice));
        cuda_checked(cudaMemcpy(p.boxes.data(),boxes,std::size_t(n)*sizeof(BpAabb),cudaMemcpyDeviceToDevice));
        cuda_checked(cudaMemcpy(p.moving.data(),moving,std::size_t(n),cudaMemcpyDeviceToDevice));
        validate_nodes<<<blocks(n),128>>>(n,starts(),targets(),p.boxes.data(),p.moving.data(),error);
    }
    // Do not erase a caller's preexisting error. No graph/certificate may
    // authorize a trial if validation failed, even for an empty contact list.
    check(error);
    if(count)emit_graph<<<blocks(count),128>>>(n,contacts,count,starts(),p.boxes.data(),gap,p.keys.data(),p.separated.data(),p.costs.data(),error);
    check(error);
    p.retained.resize(count);p.retained_count=0;
    if(count) {
        auto begin=thrust::device_pointer_cast(contacts);
        auto end=thrust::copy_if(begin,begin+count,ptr(p.separated),ptr(p.retained),RetainContact{});
        p.retained_count=static_cast<std::size_t>(end-ptr(p.retained));
    }
    std::size_t edges=0;
    if(count) {
        auto begin=ptr(p.keys),end=begin+p.keys.size();
        end=thrust::remove(begin,end,UINT64_MAX);thrust::sort(begin,end);end=thrust::unique(begin,end);
        edges=static_cast<std::size_t>(end-begin);
    }
    p.adjacency.resize(edges);
    if(edges)csr_rows<<<blocks(edges),128>>>(p.keys.data(),edges,p.rows.data(),p.adjacency.data());
    thrust::inclusive_scan(ptr(p.rows),ptr(p.rows)+p.rows.size(),ptr(p.rows));
    color_graph<<<1,1>>>(n,p.rows.data(),p.adjacency.data(),p.scratch(),p.device_shape.data(),error);check(error);
    p.shape=p.device_shape.download()[0];
    if(n)thrust::sort(ptr(p.trial_order),ptr(p.trial_order)+n,TrialLess{p.colors.data(),p.costs.data(),p.moving.data(),heavy});
    trial_groups<<<1,1>>>(n,p.colors.data(),p.trial_order.data(),p.offsets.data(),p.shape,error);check(error);
    p.ready=true;
}
DeviceSchedule ResidentLevelSchedule::trial()const {
    const auto& p=*impl_;if(!p.ready||p.finished)throw std::logic_error("private level trial epoch unavailable");
    return {p.colors.data(),nullptr,p.trial_order.data(),p.offsets.data(),p.shape.colors,p.n};
}
DeviceSchedule ResidentLevelSchedule::finish_trial(const double (*trial)[3],int* error) {
    auto& p=*impl_;
    if(!p.ready||p.finished||!error||(!trial&&p.n))throw std::invalid_argument("private level trial completion epoch");
    if(p.n)measure<<<blocks(p.n),128>>>(p.n,starts(),trial,p.movement.data(),error);
    build_levels<<<1,1>>>(p.n,p.rows.data(),p.adjacency.data(),p.moving.data(),p.movement.data(),p.scratch(),p.device_shape.data(),error);
    check(error);p.shape=p.device_shape.download()[0];p.finished=true;
    return {p.colors.data(),p.levels.data(),p.order.data(),p.offsets.data(),p.shape.groups,p.shape.active};
}
const int* ResidentLevelSchedule::project_blocks(const int* node_blocks,int block_count,int* error) {
    auto& p=*impl_;
    if(!p.ready||!error||block_count<0||(!node_blocks&&p.n))throw std::invalid_argument("private level block projection epoch/extent");
    p.block_colors.resize(block_count);p.block_owners.resize(block_count);
    if(block_count) {
        cuda_checked(cudaMemset(p.block_colors.data(),0xff,std::size_t(block_count)*sizeof(int)));
        cuda_checked(cudaMemset(p.block_owners.data(),0xff,std::size_t(block_count)*sizeof(int)));
    }
    auto view=p.finished?DeviceSchedule{p.colors.data(),p.levels.data(),p.order.data(),p.offsets.data(),p.shape.groups,p.shape.active}:trial();
    if(view.vertices)project_nodes<<<blocks(view.vertices),128>>>(view.vertices,view.ordered_vertices,view.group_offsets,view.groups,
        p.moving.data(),node_blocks,block_count,p.block_owners.data(),p.block_colors.data(),error);
    if(p.n)validate_block_projection<<<blocks(p.n),128>>>(p.n,p.moving.data(),node_blocks,block_count,p.block_owners.data(),error);
    check(error);return p.block_colors.data();
}
const BpContact* ResidentLevelSchedule::retained_contacts()const {
    if(!impl_->ready)throw std::logic_error("private level retained contacts unavailable");
    return impl_->retained.data();
}
std::size_t ResidentLevelSchedule::retained_contact_count()const {
    if(!impl_->ready)throw std::logic_error("private level retained contact count unavailable");
    return impl_->retained_count;
}
const unsigned char* ResidentLevelSchedule::separated()const{return impl_->separated.data();}
const double* ResidentLevelSchedule::movement()const{return impl_->movement.data();}
const double (*ResidentLevelSchedule::starts()const)[3]{return reinterpret_cast<const double(*)[3]>(impl_->start.data());}
const double (*ResidentLevelSchedule::targets()const)[3]{return reinterpret_cast<const double(*)[3]>(impl_->target.data());}
const BpAabb* ResidentLevelSchedule::boxes()const{return impl_->boxes.data();}
const std::size_t* ResidentLevelSchedule::graph_offsets()const{return impl_->rows.data();}
const int* ResidentLevelSchedule::graph_neighbors()const{return impl_->adjacency.data();}
} // namespace ipc_gpu::resident::level_predictor
