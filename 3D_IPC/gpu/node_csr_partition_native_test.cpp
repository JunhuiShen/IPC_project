// Native-ready extension of the qualified node-batch fixture. The original
// node_ccd_batch_test.cpp remains unchanged. NATIVE_UNQUALIFIED locally.
#include "linear_pipeline.h"
#include "original_ccd.cuh"
#include "ccd.cuh"
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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
        for(const char* name:{"IPC_GPU_ORIGINAL_NODE_CCD_BATCH","IPC_GPU_ORIGINAL_NODE_CCD_BATCH_QUEUE_CAPACITY","IPC_GPU_ORIGINAL_CCD_WORKERS",
            "IPC_GPU_NODE_CSR_PARTITION"}) {
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
struct Variant {
    std::string flags;
    std::unique_ptr<gpu::GpuLinearClipper> owner;
};
void feature_flags(const char* partition) {
    require(setenv("IPC_GPU_NODE_CSR_PARTITION",partition,1)==0,"cannot set partition test environment");
}
Variant captured_variant(const char* partition) {
    feature_flags(partition);
    return {partition,std::make_unique<gpu::GpuLinearClipper>()};
}
void compare(const Fixture& f,gpu::GpuLinearClipper& reused,
    const std::vector<Variant>* variants=nullptr,const std::string& reused_flags="0") {
    const auto reference=run(f,false,reused);
    require(reference.error==f.expected_error,f.name+" unexpected legacy status "+std::to_string(reference.error));
    for(int round=0;round<3;++round) {
        const auto actual=run(f,true,reused);
        require(actual.error==reference.error,f.name+" batch flags="+reused_flags+" status differs");
        require(bytes(actual.accepted,reference.accepted),f.name+" flags="+reused_flags+" accepted endpoint bytes differ");
        if(variants)for(const auto& variant:*variants) {
            const auto candidate=run(f,true,*variant.owner);
            const auto label=f.name+" captured flags="+variant.flags+" round="+std::to_string(round);
            require(candidate.error==reference.error,label+" status differs");
            require(bytes(candidate.accepted,reference.accepted),label+" endpoint bytes differ");
        }
    }
}
// Finite inactive padding drives pair_grid to1024 while leaving one row active.
// CSR indices are allowed to repeat and need not be sorted by global pair ID.
Fixture long_collision_row(std::size_t count) {
    auto f=point("one active long CSR late minimum count="+std::to_string(count),1.25,-1.);
    f.positions.insert(f.positions.end(),{V{-10,-10,.5},V{10,-10,.5},V{0,10,.5}});
    f.contacts.push_back(gpu::BpContact{{0,4,5,6},0});
    f.positions.resize(7+1024,V{20,20,20});f.targets=f.positions;f.targets[0][2]=-1.;
    f.deformable.assign(f.positions.size(),0);f.deformable[0]=1;
    f.colors.assign(f.positions.size(),1);f.colors[0]=0;f.rows.assign(f.positions.size(),{});
    f.rows[0].assign(count,0);if(count)f.rows[0].back()=1;
    f.boxes.assign(f.positions.size(),gpu::BpAabb{{-100,-100,-100},{100,100,100}});
    return f;
}
Fixture independent_rows(std::size_t active,bool pad) {
    Fixture f;f.name="uneven independent rows active="+std::to_string(active);
    const std::size_t nodes=active+3+(pad?1024:0);
    f.positions.assign(nodes,V{20,20,20});f.targets=f.positions;
    f.deformable.assign(nodes,0);f.colors.assign(nodes,1);f.rows.resize(nodes);
    f.boxes.assign(nodes,gpu::BpAabb{{-100,-100,-100},{100,100,100}});
    for(std::size_t node=0;node<active;++node) {
        f.positions[node]=V{.125+.03125*double(node%8),.125,1.};f.targets[node]=f.positions[node];
        f.targets[node][2]=node%2?-1.:1.125;f.deformable[node]=1;f.colors[node]=0;
        const int pair=static_cast<int>(f.contacts.size());
        f.contacts.push_back(gpu::BpContact{{static_cast<int>(node),static_cast<int>(active),
            static_cast<int>(active+1),static_cast<int>(active+2)},0});
        const std::size_t count=node%7==0?0:node%7==1?129:node%7==2?17:1;
        f.rows[node].assign(count,pair);
    }
    if(active&&active<1024)f.rows[active/2].assign(4097,static_cast<int>(active/2));
    f.positions[active]=V{0,0,0};f.positions[active+1]=V{1,0,0};f.positions[active+2]=V{0,1,0};
    for(std::size_t node=active;node<nodes;++node)f.targets[node]=f.positions[node];
    return f;
}
Fixture partition_boundary_rows(std::size_t active) {
    auto f=independent_rows(active,active<1024);
    f.name+=" every mover has a unique smaller TOI at rank128";
    const int base=static_cast<int>(f.positions.size());
    f.positions.insert(f.positions.end(),{V{0,0,.5},V{1,0,.5},V{0,1,.5}});
    f.targets.insert(f.targets.end(),{V{0,0,.5},V{1,0,.5},V{0,1,.5}});
    f.deformable.resize(f.positions.size(),0);f.colors.resize(f.positions.size(),1);
    f.rows.resize(f.positions.size());f.boxes.resize(f.positions.size(),gpu::BpAabb{{-100,-100,-100},{100,100,100}});
    // node_batch_prepare appends active nodes atomically, so the physical
    // node owning active-list row0 is not fixed. Make every possible owner
    // require its extra CTA's rank128, independent of append order.
    for(std::size_t node=0;node<active;++node) {
        f.positions[node][2]=1.25;f.targets[node][2]=-1.;
        const int smaller=static_cast<int>(f.contacts.size());
        f.contacts.push_back(gpu::BpContact{{static_cast<int>(node),base,base+1,base+2},0});
        f.rows[node].assign(257,static_cast<int>(node));f.rows[node][128]=smaller;
        require(std::count(f.rows[node].begin(),f.rows[node].end(),smaller)==1,
            "active boundary fixture lost a mover's unique smaller-TOI rank");
    }
    return f;
}
Fixture padded_health(const Fixture& healthy,std::size_t count) {
    auto f=healthy;f.name="one active cold health CSR count="+std::to_string(count);
    const auto original=f.positions.size();f.positions.resize(original+1024,V{20,20,20});
    f.targets=f.positions;f.deformable.assign(f.positions.size(),0);f.deformable[0]=1;
    f.colors.assign(f.positions.size(),1);f.colors[0]=0;f.rows.assign(f.positions.size(),{});
    f.rows[0].assign(count,0);f.boxes.assign(f.positions.size(),gpu::BpAabb{{-100,-100,-100},{100,100,100}});
    return f;
}
Fixture health_first_bad(const Fixture& healthy,bool invalid_first,std::size_t later_rank=65) {
    auto f=healthy;f.name=invalid_first?"long health first InvalidInput before InitialContact"
        :"long health first InitialContact before InvalidInput";
    f.name+=" ranks3/"+std::to_string(later_rank);
    require(later_rank>3&&f.rows[0].size()>later_rank,"long health fixture lacks error ranks");
    const int base=static_cast<int>(f.positions.size());const double z=f.positions[0][2];
    f.positions.insert(f.positions.end(),{V{-10,-10,z},V{10,-10,z},V{0,10,z},V{INFINITY,0,z}});
    f.targets=f.positions;f.deformable.resize(f.positions.size(),0);f.colors.resize(f.positions.size(),1);
    f.rows.resize(f.positions.size());f.boxes.resize(f.positions.size(),gpu::BpAabb{{-100,-100,-100},{100,100,100}});
    f.contacts.push_back(gpu::BpContact{{0,base,base+1,base+2},0});
    f.contacts.push_back(gpu::BpContact{{0,base,base+1,base+3},0});
    f.rows[0][3]=invalid_first?2:1;f.rows[0][later_rank]=invalid_first?1:2;
    f.expected_error=int(invalid_first?gpu::LinearClipError::InvalidInput:gpu::LinearClipError::InitialContact);
    return f;
}
int main() {
    try {
        Environment restore;setenv("IPC_GPU_ORIGINAL_CCD_WORKERS","16",1);
        feature_flags("0");gpu::GpuLinearClipper reused;
        std::vector<Variant> variants;
        variants.push_back(captured_variant("1"));
        for(const char* bad:{"invalid","2","-1"}) {
            feature_flags(bad);bool rejected=false;
            try {gpu::GpuLinearClipper invalid;}catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"fresh constructor accepted malformed CSR feature flag");
        }
        // Previously created partition0/1 owners must retain their captured flags;
        // these values would reject any accidental per-call environment read.
        feature_flags("invalid-after-capture");
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
        // The same proved rounded-Unsafe trial now spans33 CTAs. Every query
        // must keep the represented-path pass; earliest trial-unsafety remains
        // a backtrackable packed zero-code verdict rather than a hard error.
        auto rounded_speculative=f;rounded_speculative.name="4097 cross-CTA rounded-Unsafe trials";
        rounded_speculative.positions.resize(1028,V{20,20,20});
        rounded_speculative.targets.resize(1028,V{20,20,20});
        rounded_speculative.deformable.resize(1028,0);rounded_speculative.colors.resize(1028,1);
        rounded_speculative.rows.resize(1028);rounded_speculative.rows[0].assign(4097,0);
        rounded_speculative.boxes.resize(1028,gpu::BpAabb{{-100,-100,-100},{100,100,100}});
        fixtures.push_back(rounded_speculative);
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
        auto long4097=long_collision_row(4097),long65537=long_collision_row(65537);
        const auto query_for=[](const Fixture& item,int pair) {
            ipc_gpu::original_ccd::Query query;query.vertex_face=true;
            for(int role=0;role<4;++role)for(int axis=0;axis<3;++axis)
                query.positions[role][axis]=item.positions[item.contacts[pair].vertices[role]][axis];
            for(int axis=0;axis<3;++axis)
                query.motion[0][axis]=item.targets[0][axis]-item.positions[0][axis];
            return query;
        };
        const auto far_hit=ipc_gpu::original_ccd::evaluate_fast(query_for(long65537,0));
        const auto late_hit=ipc_gpu::original_ccd::evaluate_fast(query_for(long65537,1));
        require(far_hit.status==ipc_gpu::original_ccd::Status::Ok&&far_hit.collision
            &&late_hit.status==ipc_gpu::original_ccd::Status::Ok&&late_hit.collision
            &&late_hit.toi<far_hit.toi&&std::abs(late_hit.toi/far_hit.toi-.6)<1e-12,
            "late-rank plane.5 fixture lost its independent smaller-TOI premise");
        // Start1.25, planes0/.5 gives TOI ratio.6. A mistaken early raw hit
        // followed by halving cannot accidentally reproduce the correct min.
        const auto late_reference=run(long65537,false,reused);
        require(late_reference.error==0&&std::abs(late_reference.accepted[0][2]-.575)<2e-14,
            "late raw-minimum fixture lost its independently expected endpoint");
        fixtures.push_back(long4097);fixtures.push_back(long65537);
        // pair_grid=1024, blockDim=128 gives a maximum partition rank stride
        // of131072 for one active row. Only the final CSR incident uses the
        // smaller TOI, so tails on either side of that stride must be visited.
        for(std::size_t count:{131071u,131072u,131073u}) {
            auto boundary=long_collision_row(count);
            require(boundary.positions.size()>1024&&boundary.rows[0].size()==count
                &&boundary.rows[0].back()==1
                &&std::count(boundary.rows[0].begin(),boundary.rows[0].end(),1)==1,
                "rank-stride boundary lost its unique final smaller-TOI incident");
            const auto boundary_reference=run(boundary,false,reused);
            require(boundary_reference.error==0
                &&std::abs(boundary_reference.accepted[0][2]-.575)<2e-14,
                "rank-stride boundary lost its independent .575 endpoint oracle");
            fixtures.push_back(std::move(boundary));
        }
        auto shuffled=long4097;shuffled.name="shuffled duplicated4097 CSR mixed TOIs";
        for(std::size_t rank=0;rank<shuffled.rows[0].size();++rank)
            shuffled.rows[0][rank]=rank%5==0?1:0;
        std::uint64_t shuffle_state=0x5c031af24b8719deull;
        for(std::size_t rank=shuffled.rows[0].size();rank>1;--rank) {
            shuffle_state=shuffle_state*6364136223846793005ull+1442695040888963407ull;
            std::swap(shuffled.rows[0][rank-1],shuffled.rows[0][shuffle_state%rank]);
        }
        fixtures.push_back(shuffled);
        auto long_health=padded_health(trial,4097);fixtures.push_back(long_health);
        fixtures.push_back(health_first_bad(long_health,false));
        fixtures.push_back(health_first_bad(long_health,true));
        // With one active row, rank3 lies in CTA0;129 lies in CTA1 and4096
        // in CTA32. Completion order must not replace earliest CSR error order.
        for(std::size_t later_rank:{129u,4096u})for(bool invalid_first:{false,true})
            fixtures.push_back(health_first_bad(long_health,invalid_first,later_rank));
        fixtures.push_back(independent_rows(7,true));
        for(std::size_t active:{15u,16u,17u,1023u,1024u,1025u}) {
            // Atactive1023, active-list row0's rank128 belongs to CTA1023; rank256
            // belongs to CTA0's next iteration. At/above1024 the old row
            // mapping applies. Every mover has the unique smaller TOI, so
            // this premise does not assume atomic active-list append order.
            auto boundary=partition_boundary_rows(active);
            const auto reference=run(boundary,false,reused);
            require(reference.error==0,"active-row boundary legacy reference failed");
            for(std::size_t node=0;node<active;++node)
                require(std::abs(reference.accepted[node][2]-.575)<2e-14,
                    "active-row boundary lost a mover's independent .575 oracle");
            fixtures.push_back(std::move(boundary));
        }
        // Empty/short rows still enter active_count. This exceeds pair_grid1024
        // and therefore exercises the unchanged row-stride path with bothflagsON.
        fixtures.push_back(independent_rows(1031,false));
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
        for(const auto& item:fixtures)compare(item,reused,&variants);
        setenv("IPC_GPU_ORIGINAL_NODE_CCD_BATCH_QUEUE_CAPACITY","1",1);
        for(const auto* item:{&queued,&trial,&long_health,&rounded_speculative})compare(*item,reused,&variants);
        unsetenv("IPC_GPU_ORIGINAL_NODE_CCD_BATCH_QUEUE_CAPACITY");
        // Reset valid capture settings BEFORE constructing worker-limit owners.
        // Only one256-worker owner exists at a time; never allocate one1GiB
        // arena per feature variant simultaneously.
        for(const char* workers:{"1","256"}) {
            setenv("IPC_GPU_ORIGINAL_CCD_WORKERS",workers,1);
            for(const char* flags:{"0","1"}) {
                feature_flags(flags);
                gpu::GpuLinearClipper limited;
                compare(wide,limited,nullptr,flags);compare(queued,limited,nullptr,flags);
                compare(multi,limited,nullptr,flags);
            }
        }
        std::cout<<"Node CSR partition native acceptance: "<<fixtures.size()<<" fixtures; legacybatchOFF / qualified partition0 / partition1"
            <<" endpoint bits/status, captured flags, long/later-TOI CSR, cold health/firstbad priority,"
            <<" rank131071/131072/131073, cross-CTA error ranks3/129/4096, uneven and>grid active rows,"
            <<" original modes/masks/noop/predictor/exactretry, queue fallback/reuse PASS\n";
        return 0;
    } catch(const std::exception& error){std::cerr<<"Original node batch test failed: "<<error.what()<<'\n';return 1;}
}
