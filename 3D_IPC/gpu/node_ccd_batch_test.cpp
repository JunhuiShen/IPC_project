#include "linear_pipeline.h"
#include "original_ccd.cuh"
#include "ccd.cuh"
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu=ipc_gpu::solver;
using V=std::array<double,3>;
void require(bool value,const std::string& label){if(!value)throw std::runtime_error(label);}
void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
template<class T>struct Buffer {
    T* p=nullptr;std::size_t count;
    explicit Buffer(const std::vector<T>& values):count(values.size()) {
        if(count){check(cudaMalloc(&p,count*sizeof(T)));check(cudaMemcpy(p,values.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
    }
    ~Buffer(){if(p)cudaFree(p);}
    std::vector<T> read()const {
        std::vector<T> values(count);if(count)check(cudaMemcpy(values.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return values;
    }
};
double(*xyz(V* p))[3]{return reinterpret_cast<double(*)[3]>(p);}
template<class T>bool bytes(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
struct Environment {
    std::vector<std::pair<std::string,std::string>> saved;std::vector<bool> present;
    Environment() {
        for(const char* name:{"IPC_GPU_ORIGINAL_NODE_CCD_BATCH","IPC_GPU_ORIGINAL_NODE_CCD_BATCH_QUEUE_CAPACITY","IPC_GPU_ORIGINAL_CCD_WORKERS"}) {
            const char* old=std::getenv(name);saved.push_back({name,old?old:""});present.push_back(old!=nullptr);
        }
        unsetenv("IPC_GPU_ORIGINAL_NODE_CCD_BATCH_QUEUE_CAPACITY");
    }
    ~Environment() {
        for(std::size_t i=0;i<saved.size();++i)
            if(present[i])setenv(saved[i].first.c_str(),saved[i].second.c_str(),1);
            else unsetenv(saved[i].first.c_str());
    }
};
struct Fixture {
    std::string name;std::vector<V> positions,targets;
    std::vector<unsigned char> deformable,selected;
    std::vector<int> colors;std::vector<gpu::BpContact> contacts;
    std::vector<std::vector<int>> rows;std::vector<gpu::BpAabb> boxes;
    gpu::LinearClipMode mode=gpu::LinearClipMode::OriginalSolver;
    gpu::OriginalUpdateConvention convention=gpu::OriginalUpdateConvention::StoredNoop;
    bool ccd=true;int prior_error=0,expected_error=0;
};
struct Outcome {int error;std::vector<V> accepted;};
Outcome run(const Fixture& f,bool batch,gpu::GpuLinearClipper& clipper) {
    setenv("IPC_GPU_ORIGINAL_NODE_CCD_BATCH",batch?"1":"0",1);
    const auto count=f.positions.size();require(f.rows.size()==count,"fixture rows do not cover blocks");
    std::vector<int> blocks(count),indices;std::vector<std::size_t> rows{0};
    for(std::size_t node=0;node<count;++node) {
        blocks[node]=static_cast<int>(node);indices.insert(indices.end(),f.rows[node].begin(),f.rows[node].end());rows.push_back(indices.size());
    }
    Buffer<V> x(f.positions),proposed(f.targets),accepted(std::vector<V>(count,V{-91,-92,-93}));
    Buffer<int> device_blocks(blocks),colors(f.colors),pairs(indices),error(std::vector<int>{f.prior_error});
    Buffer<unsigned char> moving(f.deformable),selected(f.selected);Buffer<std::size_t> offsets(rows);
    Buffer<gpu::BpContact> contacts(f.contacts);Buffer<gpu::BpAabb> boxes(f.boxes);
    gpu::DeformableScene scene{};scene.nodes=count;scene.deformable=moving.p;scene.node_blocks=device_blocks.p;
    gpu::DeformableState state{};state.positions=xyz(x.p);
    gpu::ElasticWorkspace work{};work.proposed=xyz(proposed.p);work.error=error.p;
    gpu::BroadPhaseView bp{};bp.contacts=contacts.p;bp.contact_count=f.contacts.size();
    clipper.clip_deformable_color(scene,state,work,bp,offsets.p,pairs.p,boxes.p,colors.p,0,xyz(accepted.p),error.p,
        f.ccd,nullptr,{},false,f.mode,f.convention,selected.p);
    check(cudaDeviceSynchronize());
    require(bytes(x.read(),f.positions)&&bytes(proposed.read(),f.targets),f.name+" modified immutable positions/proposals");
    return {error.read()[0],accepted.read()};
}
Fixture point(const std::string& name,double start=1.,double target=-1.) {
    Fixture f;f.name=name;
    f.positions={V{0,.25,start},V{-10,-10,0},V{10,-10,0},V{0,10,0}};
    f.targets=f.positions;f.targets[0][2]=target;f.deformable.assign(4,1);f.colors={0,1,1,1};
    f.contacts={gpu::BpContact{{0,1,2,3},0}};f.rows.assign(4,std::vector<int>{0});
    f.boxes.assign(4,gpu::BpAabb{{-100,-100,-100},{100,100,100}});return f;
}
void compare(const Fixture& f,gpu::GpuLinearClipper& reused) {
    const auto reference=run(f,false,reused);
    require(reference.error==f.expected_error,f.name+" unexpected legacy status "+std::to_string(reference.error));
    for(int round=0;round<3;++round) {
        const auto actual=run(f,true,reused);
        require(actual.error==reference.error,f.name+" batch status differs");
        require(bytes(actual.accepted,reference.accepted),f.name+" accepted endpoint bytes differ");
    }
}
int main() {
    try {
        Environment restore;setenv("IPC_GPU_ORIGINAL_CCD_WORKERS","16",1);gpu::GpuLinearClipper reused;
        std::vector<Fixture> fixtures;
        for(auto mode:{gpu::LinearClipMode::OriginalSolver,gpu::LinearClipMode::OriginalColoredGuess})
            for(auto convention:{gpu::OriginalUpdateConvention::StoredNoop,gpu::OriginalUpdateConvention::CooperativeSquaredNorm})
                for(bool ccd:{false,true})for(bool selected:{false,true}) {
                    auto f=point("crossing mode="+std::to_string(int(mode))+" convention="+std::to_string(int(convention))
                        +" CCD="+std::to_string(ccd)+" selected="+std::to_string(selected));f.mode=mode;f.convention=convention;f.ccd=ccd;
                    if(convention==gpu::OriginalUpdateConvention::CooperativeSquaredNorm)f.selected={static_cast<unsigned char>(selected),0,0,0};
                    fixtures.push_back(f);
                }
        for(bool ccd:{false,true})for(auto convention:{gpu::OriginalUpdateConvention::StoredNoop,gpu::OriginalUpdateConvention::CooperativeSquaredNorm}) {
            for(double delta:{0.,0x1.6849b86a12b9ap-47,0x1.6849b86a12b9bp-47,0x1.6849b86a12b9cp-47}) {
                auto f=point("tiny selected cutoff",ccd?1.:0.,ccd?1.:0.);f.ccd=ccd;f.convention=convention;
                if(!ccd)f.positions={V{0,0,0},V{0,.25,1},V{-10,-10,0},V{10,-10,0}};
                f.targets=f.positions;f.targets[0][2]+=delta;
                if(convention==gpu::OriginalUpdateConvention::CooperativeSquaredNorm)f.selected={1,0,0,0};
                fixtures.push_back(f);f.name="tiny unselected cutoff";
                if(!f.selected.empty())f.selected[0]=0;fixtures.push_back(f);
                f.name="tiny predictor ignores selected cutoff";f.mode=gpu::LinearClipMode::OriginalColoredGuess;
                if(!f.selected.empty())f.selected[0]=1;fixtures.push_back(f);
            }
        }
        auto f=point("healthy no-op",1.,1.);fixtures.push_back(f);
        Fixture empty;empty.name="empty scene";fixtures.push_back(empty);
        f=point("positive tiny-gap no-op",1e-12,1e-12);fixtures.push_back(f);
        f=point("true initial contact no-op",0.,0.);f.expected_error=int(gpu::LinearClipError::InitialContact);fixtures.push_back(f);
        f=point("true initial contact moving",0.,1.);f.expected_error=int(gpu::LinearClipError::InitialContact);fixtures.push_back(f);
        f=point("true initial contact selected cutoff",0.,1e-15);f.selected={1,0,0,0};
        f.convention=gpu::OriginalUpdateConvention::CooperativeSquaredNorm;f.expected_error=int(gpu::LinearClipError::InitialContact);fixtures.push_back(f);
        for(double start:{5e-9,5e-8}) {
            f=point("ordinary near-gap move",start,2.5e-9);fixtures.push_back(f);
            f.name="predictor 48-step tested refinement";f.mode=gpu::LinearClipMode::OriginalColoredGuess;fixtures.push_back(f);
            const auto ordinary=run(point("floor premise",start,2.5e-9),false,reused),predictor=run(f,false,reused);
            require(ordinary.error==0&&predictor.error==0&&predictor.accepted[0][2]>ordinary.accepted[0][2],"predictor floor fixture is vacuous");
            f.ccd=false;f.name="predictor CCD-off bypasses refinement";fixtures.push_back(f);
        }
        f=point("EE crossing");f.positions={V{-1,0,1},V{1,0,1},V{0,-1,0},V{0,1,0}};
        f.targets=f.positions;f.targets[0][2]=-1.;f.contacts={gpu::BpContact{{0,1,2,3},1}};fixtures.push_back(f);
        f.name="EE predictor refinement";f.positions[0][2]=f.positions[1][2]=5e-9;
        f.targets=f.positions;f.targets[0][2]=2.5e-9;f.mode=gpu::LinearClipMode::OriginalColoredGuess;fixtures.push_back(f);
        f=point("arbitrary duplicated and shuffled CSR");
        f.positions.insert(f.positions.end(),{V{-10,-10,.5},V{10,-10,.5},V{0,10,.5}});
        f.targets=f.positions;f.targets[0][2]=-1.;f.contacts.push_back(gpu::BpContact{{0,4,5,6},0});
        f.deformable.assign(7,1);f.colors.assign(7,1);f.colors[0]=0;f.rows.assign(7,{});f.rows[0]={1,0,1,0,0,1};
        f.boxes.assign(7,gpu::BpAabb{{-100,-100,-100},{100,100,100}});fixtures.push_back(f);
        std::reverse(f.rows[0].begin(),f.rows[0].end());f.name="CSR reverse preserves minimum TOI";fixtures.push_back(f);
        f=point("rounded 90-percent endpoint requires safety backtrack");
        f.positions={V{1e16+2,.25,.25},V{1e16,0,0},V{1e16,1,0},V{1e16,0,1}};
        f.targets=f.positions;f.targets[0][0]=1e16;
        f.boxes.assign(4,gpu::BpAabb{{1e16-100,-4,-4},{1e16+100,4,4}});
        ipc_gpu::original_ccd::Query rounding;
        double rounded_end[4][3];
        for(int node=0;node<4;++node)for(int axis=0;axis<3;++axis)
            rounding.positions[node][axis]=rounded_end[node][axis]=f.positions[node][axis];
        rounding.motion[0][0]=-2.;volatile double raw_product=.9*(-2.);
        rounded_end[0][0]=f.positions[0][0]+raw_product;
        const auto raw_toi=ipc_gpu::original_ccd::evaluate_fast(rounding);
        require(raw_toi.status==ipc_gpu::original_ccd::Status::Ok&&raw_toi.collision&&raw_toi.toi==1.
            &&ipc_gpu::exact::validate_collision_step<2048>(rounding.positions,rounded_end,true)==ipc_gpu::exact::Status::Unsafe,
            "rounded node backtrack fixture lost its independent Unsafe premise");
        fixtures.push_back(f);
        f=point("first InitialContact before InvalidInput",0.,0.);
        f.positions.push_back(V{INFINITY,1,0});f.targets=f.positions;f.deformable.push_back(1);f.colors.push_back(1);
        f.boxes.push_back(gpu::BpAabb{{-100,-100,-100},{100,100,100}});f.rows.resize(5);
        f.contacts.push_back(gpu::BpContact{{0,1,2,4},0});f.rows[0]={0,1};
        f.expected_error=int(gpu::LinearClipError::InitialContact);fixtures.push_back(f);
        f.name="shuffled first InvalidInput before InitialContact";f.rows[0]={1,0};
        f.expected_error=int(gpu::LinearClipError::InvalidInput);fixtures.push_back(f);
        f=point("empty contacts");f.contacts.clear();f.rows.assign(4,{});fixtures.push_back(f);
        f=point("no active nodes");f.colors.assign(4,1);fixtures.push_back(f);
        f=point("fixed inactive node");f.deformable[0]=0;fixtures.push_back(f);
        f=point("prior error");f.prior_error=77;f.expected_error=77;fixtures.push_back(f);
        f=point("invalid cooperative mask");f.convention=gpu::OriginalUpdateConvention::CooperativeSquaredNorm;
        f.selected={1,2,0,0};f.expected_error=int(gpu::LinearClipError::InvalidInput);fixtures.push_back(f);
        f.selected.clear();f.name="missing cooperative mask";fixtures.push_back(f);
        f=point("out-of-box start",5.,1.);f.boxes[0]=gpu::BpAabb{{-4,-4,-4},{4,4,4}};
        f.expected_error=int(gpu::LinearClipError::OutsideBlueBox);fixtures.push_back(f);
        f=point("nonfinite proposal");f.targets[0][0]=INFINITY;f.expected_error=int(gpu::LinearClipError::InvalidInput);fixtures.push_back(f);
        const double maximum=std::numeric_limits<double>::max(),minimum=std::numeric_limits<double>::denorm_min();
        auto wide=point("256 to 2048 exact retry",1.,2.);
        wide.positions={V{.25,.25,1},V{-maximum,0,0},V{maximum,0,0},V{0,minimum,0}};
        wide.targets=wide.positions;wide.targets[0][2]=2.;
        wide.boxes.assign(4,gpu::BpAabb{{-maximum,-maximum,-maximum},{maximum,maximum,maximum}});
        wide.boxes[0]=gpu::BpAabb{{-4,-4,-4},{4,4,4}};
        ipc_gpu::original_ccd::Query retry;
        for(int node=0;node<4;++node)for(int axis=0;axis<3;++axis)retry.positions[node][axis]=wide.positions[node][axis];
        retry.motion[0][2]=1.;
        require(ipc_gpu::original_ccd::evaluate_fast(retry).status==ipc_gpu::original_ccd::Status::NeedsExact
            &&ipc_gpu::original_ccd::evaluate_exact<256>(retry).status==ipc_gpu::original_ccd::Status::CapacityExceeded,
            "node retry fixture does not exercise true small-capacity failure");
        fixtures.push_back(wide);
        auto queued=point("257 exact jobs with duplicated row",5e-11,5e-11-2.);
        queued.positions[0][0]=queued.targets[0][0]=.25;queued.rows[0].assign(257,0);fixtures.push_back(queued);
        ipc_gpu::original_ccd::Query queue_query;
        for(int node=0;node<4;++node)for(int axis=0;axis<3;++axis)queue_query.positions[node][axis]=queued.positions[node][axis];
        for(int axis=0;axis<3;++axis)queue_query.motion[0][axis]=queued.targets[0][axis]-queued.positions[0][axis];
        require(ipc_gpu::original_ccd::evaluate_fast(queue_query).status==ipc_gpu::original_ccd::Status::NeedsExact,
            "duplicated node queue fixture bypassed exact work");
        auto trial=point("257 represented-only exact health jobs",1.,1.);
        trial.positions={V{.25,.25,std::nextafter(.5,1.)},V{1,0,0},V{0,1,0},V{0,0,1}};
        trial.targets=trial.positions;trial.rows[0].assign(257,0);fixtures.push_back(trial);
        ipc_gpu::ccd::LinearQuery stationary;
        for(int node=0;node<4;++node)for(int axis=0;axis<3;++axis)
            stationary.start[node][axis]=stationary.end[node][axis]=trial.positions[node][axis];
        require(!ipc_gpu::ccd::detail::separated(stationary,0.,1.,0.)
            &&ipc_gpu::exact::validate_collision_step<256>(stationary.start,stationary.end,true)==ipc_gpu::exact::Status::Safe,
            "duplicated node health queue lacks an uncertain but safe exact premise");
        // Several independent same-color vertices spread pair work over CTAs.
        Fixture multi;multi.name="33 independent active rows";const int active=33,nodes=active+3;
        multi.positions.resize(nodes);multi.targets.resize(nodes);multi.deformable.assign(nodes,1);multi.colors.assign(nodes,1);
        multi.rows.resize(nodes);multi.boxes.assign(nodes,gpu::BpAabb{{-4,-4,-4},{4,4,4}});
        for(int node=0;node<active;++node) {
            multi.positions[node]=V{.125+.0625*(node%8),.125,1.};multi.targets[node]=multi.positions[node];
            multi.targets[node][2]+=double(node+1)/128.;multi.colors[node]=0;
            multi.contacts.push_back(gpu::BpContact{{node,active,active+1,active+2},0});multi.rows[node]={node};
        }
        multi.positions[active]=V{0,0,0};multi.positions[active+1]=V{1,0,0};multi.positions[active+2]=V{0,1,0};
        for(int node=active;node<nodes;++node)multi.targets[node]=multi.positions[node];fixtures.push_back(multi);
        for(const auto& item:fixtures)compare(item,reused);
        setenv("IPC_GPU_ORIGINAL_NODE_CCD_BATCH_QUEUE_CAPACITY","1",1);
        for(const auto* item:{&queued,&trial})compare(*item,reused);
        unsetenv("IPC_GPU_ORIGINAL_NODE_CCD_BATCH_QUEUE_CAPACITY");
        for(const char* workers:{"1","256"}) {
            setenv("IPC_GPU_ORIGINAL_CCD_WORKERS",workers,1);gpu::GpuLinearClipper limited;
            compare(wide,limited);compare(queued,limited);compare(multi,limited);
        }
        std::cout<<"Original node batch: "<<fixtures.size()<<" fixtures, scalar/batch endpoint bits, modes/masks/cutoff,"
            <<" no-op health, predictor refinement, arbitrary CSR, exact retry, queue fallback/reuse PASS\n";
        return 0;
    } catch(const std::exception& error){std::cerr<<"Original node batch test failed: "<<error.what()<<'\n';return 1;}
}
