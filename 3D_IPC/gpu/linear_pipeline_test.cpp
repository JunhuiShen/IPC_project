#include "linear_pipeline.h"
#include "ccd.cuh"
#include "exact.cuh"
#include "original_ccd.cuh"
#include "original_rigid.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <limits>

namespace gpu=ipc_gpu::solver;
using V=std::array<double,3>;
void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool value,const char*message) { if(!value)throw std::runtime_error(message); }
template<class T> struct Buffer {
    T* pointer=nullptr;std::size_t count=0;
    explicit Buffer(const std::vector<T>&values):count(values.size()) {
        if(count) { check(cudaMalloc(&pointer,count*sizeof(T)));check(cudaMemcpy(pointer,values.data(),count*sizeof(T),cudaMemcpyHostToDevice)); }
    }
    ~Buffer() { if(pointer)cudaFree(pointer); }
    std::vector<T> read()const {
        std::vector<T> result(count);if(count)check(cudaMemcpy(result.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost));return result;
    }
};
double(*xyz(V* p))[3] { return reinterpret_cast<double(*)[3]>(p); }

std::vector<gpu::BpAabb> boxes() { return std::vector<gpu::BpAabb>(4,gpu::BpAabb{{-2,-2,-2},{2,2,2}}); }

// Independent analytic oracle for the CPU's coordinatewise inset clamp and
// unfused before + (clipped-before) endpoint. Volatile prevents host
// contraction so it mirrors safe_step.cpp's specified rounding.
V analytic_box_endpoint(const V&start,const V&raw,const V&lo,const V&hi) {
    V result;
    for(int k=0;k<3;++k) {
        volatile double lower=lo[k]+1e-10;
        volatile double upper=hi[k]-1e-10;
        volatile double clipped=std::min(std::max(raw[k],double(lower)),double(upper));
        volatile double delta=clipped-start[k];
        volatile double weighted=1.0*delta;
        result[k]=start[k]+weighted;
    }
    return result;
}

struct BoxClampFixture { const char*name; V start,raw,lo,hi; bool deformable0=true; };

void run_box_clamp_fixture(const BoxClampFixture&fixture,bool ccd,bool expect_active) {
    const V sentinel{{-91,-92,-93}};
    const std::vector<V> positions{fixture.start,V{{-.25,.5,.75}}};
    const std::vector<V> proposals{fixture.raw,V{{9,9,9}}};
    Buffer<V> x(positions),proposed(proposals),accepted(std::vector<V>(2,sentinel));
    Buffer<unsigned char> deformable(std::vector<unsigned char>{
        static_cast<unsigned char>(fixture.deformable0?1:0),static_cast<unsigned char>(1)});
    Buffer<int> blocks(std::vector<int>{0,1}),colors(std::vector<int>{0,1}),error(std::vector<int>{0});
    Buffer<std::size_t> offsets(std::vector<std::size_t>{0,0,0});
    Buffer<gpu::BpAabb> bounds(std::vector<gpu::BpAabb>{
        gpu::BpAabb{{fixture.lo[0],fixture.lo[1],fixture.lo[2]},
                    {fixture.hi[0],fixture.hi[1],fixture.hi[2]}},
        gpu::BpAabb{{-1,-1,-1},{1,1,1}}});
    gpu::DeformableScene scene{};scene.nodes=2;scene.deformable=deformable.pointer;scene.node_blocks=blocks.pointer;
    gpu::DeformableState state{};state.positions=xyz(x.pointer);
    gpu::ElasticWorkspace work{};work.proposed=xyz(proposed.pointer);work.error=error.pointer;
    gpu::BroadPhaseView bp{}; // no contacts: only the proposal clamp is exercised
    gpu::GpuLinearClipper clipper;
    clipper.clip_deformable_color(scene,state,work,bp,offsets.pointer,nullptr,bounds.pointer,
        colors.pointer,0,xyz(accepted.pointer),error.pointer,ccd);
    check(cudaDeviceSynchronize());
    const int status=error.read()[0];
    std::ostringstream context;context<<std::setprecision(17)<<"box clamp fixture="<<fixture.name
        <<" ccd="<<ccd;
    if(status!=0)throw std::runtime_error(context.str()+" error expected=0 actual="+std::to_string(status));
    const auto result=accepted.read();
    require(result[1]==sentinel,(context.str()+" inactive node overwritten").c_str());
    if(expect_active) {
        const V expected=analytic_box_endpoint(fixture.start,fixture.raw,fixture.lo,fixture.hi);
        for(int k=0;k<3;++k)if(result[0][k]!=expected[k]) {
            std::ostringstream mismatch;mismatch<<context.str()<<" axis="<<k
                <<" expected="<<expected[k]<<" actual="<<result[0][k];
            throw std::runtime_error(mismatch.str());
        }
    } else require(result[0]==sentinel,(context.str()+" inactive proposal was written").c_str());
    require(x.read()==positions,(context.str()+" clipper committed positions").c_str());
    require(proposed.read()==proposals,(context.str()+" clipper modified proposal").c_str());
}

void box_clamp_test() {
    // One exiting axis must not shrink motion on axes that stay inside; this
    // distinguishes the CPU clamp from the former uniform sweep fraction.
    const std::vector<BoxClampFixture> fixtures{
        {"x_high_yz_inside",{{0,0,0}},{{2,.5,.25}},{{-1,-2,-3}},{{1,2,3}}},
        {"x_low_yz_inside",{{.125,.25,0}},{{-3,.75,.5}},{{-1,-2,-3}},{{1,2,3}}},
        {"y_high_xz_inside",{{.25,.125,-.25}},{{.75,4,.5}},{{-1,-2,-3}},{{1,2,3}}},
        {"z_low_xy_inside",{{-.25,.25,.125}},{{.5,.75,-5}},{{-1,-2,-3}},{{1,2,3}}},
        {"two_axis",{{.125,.25,-.25}},{{3,-4,.5}},{{-1,-2,-3}},{{1,2,3}}},
        {"exact_face_inset",{{0,0,0}},{{1,.5,.25}},{{-1,-2,-3}},{{1,2,3}}},
        {"inside_untouched",{{.125,.25,-.25}},{{.5,.75,-.5}},{{-1,-2,-3}},{{1,2,3}}},
        {"anisotropic_offset",{{8.25,-3.5,.125}},{{10,-3,.5}},{{8,-4,-1}},{{9,2,3}}},
        {"inactive_node",{{0,0,0}},{{2,.5,.25}},{{-1,-2,-3}},{{1,2,3}},false},
    };
    for(const auto& fixture:fixtures)for(bool ccd:{false,true})
        run_box_clamp_fixture(fixture,ccd,fixture.deformable0);
}

void box_clamp_failclosed_test() {
    const V sentinel{{-91,-92,-93}};
    struct BadCase { const char*name; V start,raw,lo,hi; int error; };
    const std::vector<BadCase> cases{
        // Current position outside the blue box.
        {"start_outside",{{0,0,3.5}},{{0,0,-1}},{{-1,-2,-3}},{{1,2,3}},
            int(gpu::LinearClipError::OutsideBlueBox)},
        // Nonfinite raw proposal.
        {"nan_proposal",{{0,0,0}},{{std::numeric_limits<double>::quiet_NaN(),0,0}},{{-1,-2,-3}},{{1,2,3}},
            int(gpu::LinearClipError::InvalidInput)},
        // Box narrower than the two 1e-10 insets: the CPU inner interval is
        // empty. Nothing may be written and the solver must not move outside
        // the actual box.
        {"empty_inset_interval",{{5e-12,0,0}},{{5e-12,0,0}},{{0,0,0}},{{1e-11,1,1}},
            int(gpu::LinearClipError::OutsideBlueBox)},
    };
    for(const auto& bad:cases)for(bool ccd:{false,true}) {
        Buffer<V> x(std::vector<V>{bad.start}),proposed(std::vector<V>{bad.raw}),
            accepted(std::vector<V>{sentinel});
        Buffer<unsigned char> deformable(std::vector<unsigned char>{1});
        Buffer<int> blocks(std::vector<int>{0}),colors(std::vector<int>{0}),
            error(std::vector<int>{0});
        Buffer<std::size_t> offsets(std::vector<std::size_t>{0,0});
        Buffer<gpu::BpAabb> bounds(std::vector<gpu::BpAabb>{
            gpu::BpAabb{{bad.lo[0],bad.lo[1],bad.lo[2]},{bad.hi[0],bad.hi[1],bad.hi[2]}}});
        gpu::DeformableScene scene{};scene.nodes=1;scene.deformable=deformable.pointer;scene.node_blocks=blocks.pointer;
        gpu::DeformableState state{};state.positions=xyz(x.pointer);
        gpu::ElasticWorkspace work{};work.proposed=xyz(proposed.pointer);work.error=error.pointer;
        gpu::BroadPhaseView bp{};
        gpu::GpuLinearClipper clipper;
        clipper.clip_deformable_color(scene,state,work,bp,offsets.pointer,nullptr,bounds.pointer,
            colors.pointer,0,xyz(accepted.pointer),error.pointer,ccd);
        check(cudaDeviceSynchronize());
        const int status=error.read()[0];
        std::ostringstream context;context<<std::setprecision(17)<<"box clamp failclosed fixture="
            <<bad.name<<" ccd="<<ccd;
        if(status!=bad.error)throw std::runtime_error(context.str()+" error expected="
            +std::to_string(bad.error)+" actual="+std::to_string(status));
        require(accepted.read()[0]==sentinel,(context.str()+" published a partial endpoint").c_str());
        require(x.read()[0]==bad.start,(context.str()+" clipper committed positions").c_str());
        const auto read_back=proposed.read()[0];
        require(std::memcmp(read_back.data(),bad.raw.data(),sizeof(V))==0,
            (context.str()+" clipper modified proposal").c_str());
    }
}

void deformable_test(double start_gap,double end_gap,bool ccd,int expected_error,
    double expected_z,bool tangential=false,bool two_contacts=false,bool huge_box=false,
    int prior_error=0) {
    std::vector<V> positions{{.25,.25,start_gap},{0,0,0},{1,0,0},{0,1,0}};
    if(two_contacts) { positions.push_back(V{{0,0,.75}});positions.push_back(V{{1,0,.75}});positions.push_back(V{{0,1,.75}}); }
    auto targets=positions;const std::size_t count=positions.size();
    targets[0][2]=end_gap;if(tangential)targets[0][0]=.375;
    Buffer<V> x(positions),proposed(targets),accepted(std::vector<V>(count,V{{-99,-99,-99}}));
    std::vector<unsigned char> mask(count,0);mask[0]=1;Buffer<unsigned char> deformable(mask);
    std::vector<int> block_ids(count);for(std::size_t i=0;i<count;++i)block_ids[i]=int(i);
    Buffer<int> blocks(block_ids),colors(block_ids),indices(two_contacts?std::vector<int>{0,1}:std::vector<int>{0}),error(std::vector<int>{prior_error});
    std::vector<std::size_t> offset_values(count+1,two_contacts?2:1);offset_values[0]=0;
    Buffer<std::size_t> offsets(offset_values);
    std::vector<gpu::BpContact> contacts{{{0,1,2,3},0}};if(two_contacts)contacts.push_back({{0,4,5,6},0});
    Buffer<gpu::BpContact> contact(contacts);
    const double extent=huge_box?1e201:2.;
    Buffer<gpu::BpAabb> bounds(std::vector<gpu::BpAabb>(count,gpu::BpAabb{{-extent,-extent,-extent},{extent,extent,extent}}));
    gpu::DeformableScene scene{};scene.nodes=count;scene.deformable=deformable.pointer;scene.node_blocks=blocks.pointer;
    gpu::DeformableState state{};state.positions=xyz(x.pointer);
    gpu::ElasticWorkspace work{};work.proposed=xyz(proposed.pointer);work.error=error.pointer;
    gpu::BroadPhaseView bp{};bp.contacts=contact.pointer;bp.contact_count=contacts.size();
    gpu::GpuLinearClipper clipper;
    clipper.clip_deformable_color(scene,state,work,bp,offsets.pointer,indices.pointer,bounds.pointer,
        colors.pointer,0,xyz(accepted.pointer),error.pointer,ccd);
    const int status=error.read()[0];
    std::ostringstream context;context<<std::setprecision(17)<<"deformable fixture z0="<<start_gap
        <<" z1="<<end_gap<<" ccd="<<ccd<<" tangential="<<tangential
        <<" two_contacts="<<two_contacts<<" huge_box="<<huge_box<<" prior_error="<<prior_error;
    if(status!=expected_error)throw std::runtime_error(context.str()+" error expected="
        +std::to_string(expected_error)+" actual="+std::to_string(status));
    if(!status) {
        const auto out=accepted.read();if(out[0][2]!=expected_z) {
            context<<" endpoint expected="<<expected_z<<" actual="<<out[0][2];
            throw std::runtime_error(context.str());
        }
        require(out[1][0]==-99,"inactive accepted slots must remain untouched");
        if(tangential)require(out[0][0]==targets[0][0],"near-contact tangential motion must not lock");
    }
    if(prior_error)require(accepted.read()[0][0]==-99,"prior failure must not inspect/update proposal");
    require(x.read()==positions,"clipper must not commit positions itself");
}

void rigid_test(double gap,double target_gap,double expected_gap,bool ccd,bool moving_triangle=false) {
    const std::vector<V> positions=moving_triangle
        ?std::vector<V>{{.25,.25,0},{0,0,gap},{1,0,gap},{0,1,gap}}
        :std::vector<V>{{-1,0,gap},{1,0,gap},{0,-1,0},{0,1,0}};
    Buffer<V> x(positions),accepted(std::vector<V>(4,V{{-99,-99,-99}}));
    Buffer<V> com(std::vector<V>{{0,0,gap}}),proposed(std::vector<V>{{0,0,target_gap}}),accepted_com(std::vector<V>{{-99,-99,-99}});
    Buffer<int> body_blocks(std::vector<int>{0}),node_body(moving_triangle?std::vector<int>{-1,0,0,0}:std::vector<int>{0,0,-1,-1}),
        nodes(moving_triangle?std::vector<int>{1,2,3}:std::vector<int>{0,1});
    Buffer<int> colors(std::vector<int>{0}),indices(std::vector<int>{0}),error(std::vector<int>{0});
    Buffer<unsigned char> translate(std::vector<unsigned char>{1});
    Buffer<std::size_t> node_offsets(std::vector<std::size_t>{0,moving_triangle?3UL:2UL}),contact_offsets(std::vector<std::size_t>{0,1});
    Buffer<gpu::BpContact> contact(std::vector<gpu::BpContact>{{{0,1,2,3},moving_triangle?0:1}});
    Buffer<gpu::BpAabb> bounds(boxes());
    gpu::RigidScene scene{};scene.bodies=1;scene.body_blocks=body_blocks.pointer;scene.node_body=node_body.pointer;
    scene.node_offsets=node_offsets.pointer;scene.nodes=nodes.pointer;scene.translate=translate.pointer;
    gpu::RigidState state{};state.com=xyz(com.pointer);
    gpu::RigidWorkspace work{};work.proposed_com=xyz(proposed.pointer);work.accepted_com=xyz(accepted_com.pointer);
    gpu::DeformableState vertices{};vertices.positions=xyz(x.pointer);
    gpu::BroadPhaseView bp{};bp.contacts=contact.pointer;bp.contact_count=1;
    gpu::GpuLinearClipper clipper;
    clipper.clip_rigid_translation_color(scene,state,work,vertices,bp,contact_offsets.pointer,indices.pointer,
        bounds.pointer,colors.pointer,0,xyz(accepted.pointer),error.pointer,ccd);
    const int status=error.read()[0];
    std::ostringstream context;context<<std::setprecision(17)<<"rigid fixture z0="<<gap
        <<" z1="<<target_gap<<" ccd="<<ccd<<" moving_triangle="<<moving_triangle;
    if(status)throw std::runtime_error(context.str()+" error expected=0 actual="+std::to_string(status));
    const double actual_gap=accepted_com.read()[0][2];
    if(actual_gap!=expected_gap) {
        context<<" COM expected="<<expected_gap<<" actual="<<actual_gap;
        throw std::runtime_error(context.str());
    }
    const auto endpoints=accepted.read();
    const int first=moving_triangle?1:0,second=moving_triangle?2:1,stationary=moving_triangle?0:2;
    require(endpoints[first][2]==expected_gap&&endpoints[second][2]==expected_gap,"rigid exact represented proxies");
    require(endpoints[stationary][0]==-99,"static proxy must remain untouched");
    require(x.read()==positions,"rigid clipper must not commit positions itself");
}

void cooperative_endpoint_test(bool cache_backtrack) {
    // A large incident range makes many workers publish uncertainty for the
    // SAME node, while other contacts publish Safe. The special cache case
    // has a safe full trajectory whose midpoint violates the endpoint floor:
    // full-proposal certificates cannot be reused after another pair clips it.
    const std::vector<V> positions=cache_backtrack
        ?std::vector<V>{{-1,0,1e-9},{-.1,-1,0},{.1,-1,0},{0,1,0},
                       {.5,-1,-1},{.5,1,-1},{.5,0,1}}
        :std::vector<V>{{.25,.25,1},{0,0,0},{1,0,0},{0,1,0},
                       {0,0,.75},{1,0,.75},{0,1,.75}};
    auto targets=positions;
    if(cache_backtrack)targets[0][0]=1.;else targets[0][2]=-1.;
    const std::size_t count=positions.size(),pair_count=cache_backtrack?2:1025;
    std::vector<gpu::BpContact> pairs;
    std::vector<int> pair_indices;
    for(std::size_t i=0;i<pair_count;++i) {
        pairs.push_back(i%2?gpu::BpContact{{0,4,5,6},0}:gpu::BpContact{{0,1,2,3},0});
        pair_indices.push_back(int(i));
    }
    std::vector<int> block_ids(count);for(std::size_t i=0;i<count;++i)block_ids[i]=int(i);
    std::vector<unsigned char> mask(count,0);mask[0]=1;
    std::vector<std::size_t> offset_values(count+1,pair_count);offset_values[0]=0;
    Buffer<V> x(positions),proposed(targets),accepted(std::vector<V>(count,V{{-99,-99,-99}}));
    Buffer<unsigned char> deformable(mask);
    Buffer<int> blocks(block_ids),colors(block_ids),indices(pair_indices),error(std::vector<int>{0});
    Buffer<std::size_t> offsets(offset_values);
    Buffer<gpu::BpContact> contacts(pairs);
    Buffer<gpu::BpAabb> bounds(std::vector<gpu::BpAabb>(count,gpu::BpAabb{{-2,-2,-2},{2,2,2}}));
    gpu::DeformableScene scene{};scene.nodes=count;scene.deformable=deformable.pointer;scene.node_blocks=blocks.pointer;
    gpu::DeformableState state{};state.positions=xyz(x.pointer);
    gpu::ElasticWorkspace work{};work.proposed=xyz(proposed.pointer);work.error=error.pointer;
    gpu::BroadPhaseView bp{};bp.contacts=contacts.pointer;bp.contact_count=pairs.size();
    gpu::GpuLinearClipper clipper;
    for(int sweep=0;sweep<3;++sweep) {
        const bool receding=sweep==1;
        targets[0]=positions[0];
        if(cache_backtrack)targets[0][0]=receding?-1.5:1.;
        else targets[0][2]=receding?1.5:-1.;
        check(cudaMemcpy(proposed.pointer,targets.data(),count*sizeof(V),cudaMemcpyHostToDevice));
        clipper.clip_deformable_color(scene,state,work,bp,offsets.pointer,indices.pointer,bounds.pointer,
            colors.pointer,0,xyz(accepted.pointer),error.pointer,true);
        const int status=error.read()[0];
        const double actual=accepted.read()[0][cache_backtrack?0:2];
        const double expected=cache_backtrack?(receding?-1.5:-.5):(receding?1.5:.875);
        if(status||actual!=expected) {
            std::ostringstream message;message<<std::setprecision(17)<<"cooperative/cache fixture cache_backtrack="
                <<cache_backtrack<<" sweep="<<sweep<<" status="<<status<<" endpoint="<<actual<<" expected="<<expected;
            throw std::runtime_error(message.str());
        }
        require(x.read()==positions,"cooperative check must not commit positions");
    }
    // This graph is invalid: two deformable vertices of one contact cannot
    // share a color. The parallel predicate must fail closed, not certify
    // each independently while ignoring the simultaneous other motion.
    mask[1]=1;block_ids[1]=0;
    check(cudaMemcpy(deformable.pointer,mask.data(),count*sizeof(unsigned char),cudaMemcpyHostToDevice));
    check(cudaMemcpy(colors.pointer,block_ids.data(),count*sizeof(int),cudaMemcpyHostToDevice));
    clipper.clip_deformable_color(scene,state,work,bp,offsets.pointer,indices.pointer,bounds.pointer,
        colors.pointer,0,xyz(accepted.pointer),error.pointer,true);
    require(error.read()[0]==int(gpu::LinearClipError::UnsupportedMotion),"invalid shared contact color must fail closed");
}

// The premise is checked independently on the host using the existing exact
// arithmetic. The production calls below must therefore traverse real small
// -> full-width retries; passing only allocation-free certificates is excluded.
void wide_capacity_reuse_test() {
    namespace ex=ipc_gpu::exact;
    namespace oc=ipc_gpu::original_ccd;
    namespace cc=ipc_gpu::ccd;
    const double maximum=std::numeric_limits<double>::max();
    const double minimum=std::numeric_limits<double>::denorm_min();
    const std::vector<V> wide{{.25,.25,1},{-maximum,0,0},{maximum,0,0},{0,minimum,0}};
    cc::LinearQuery q;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)q.start[i][k]=q.end[i][k]=wide[i][k];
    q.end[0][2]=2;
    q.gap_floor=128.0*std::numeric_limits<double>::epsilon()*maximum;
    require(cc::validate_linear(q)==cc::Status::Uncertain,"wide fixture bypassed the cold queue");
    require(ex::validate_step<128>(q.start,q.end,true,q.gap_floor)==ex::Status::CapacityExceeded,
        "wide fixture did not exhaust robust small limbs");
    require(ex::validate_step<2048>(q.start,q.end,true,q.gap_floor)==ex::Status::Safe,
        "wide robust retry premise did not resolve");
    oc::Query original;
    std::memcpy(original.positions,q.start,sizeof(q.start));original.motion[0][2]=1;
    require(oc::evaluate_fast(original).status==oc::Status::NeedsExact,
        "wide Original fixture bypassed exact arithmetic");
    require(oc::evaluate_exact<256>(original).status==oc::Status::CapacityExceeded,
        "wide fixture did not exhaust Original small limbs");
    const auto result=oc::evaluate_exact<2048>(original);
    require(result.status==oc::Status::Ok&&!result.collision,
        "wide Original retry premise did not resolve to a miss");
    require(!cc::detail::separated(q,0.,1.,0.),"wide represented path bypassed exact collision arithmetic");
    require(ex::validate_collision_step<256>(q.start,q.end,true)==ex::Status::CapacityExceeded
        &&ex::validate_collision_step<2048>(q.start,q.end,true)==ex::Status::Safe,
        "wide represented-path retry premise missing");
    require(ex::validate_motion<128>(q.start,original.motion,true,q.gap_floor)==ex::Status::CapacityExceeded
        &&ex::validate_motion<2048>(q.start,original.motion,true,q.gap_floor)==ex::Status::Safe,
        "wide rigid common-motion retry premise missing");
    require(ex::validate_endpoint_gap<128>(q.start,q.end,true,q.gap_floor)==ex::Status::CapacityExceeded
        &&ex::validate_endpoint_gap<2048>(q.start,q.end,true,q.gap_floor)==ex::Status::Safe,
        "wide rigid represented-endpoint retry premise missing");

    // Original rigid motion first rejects disjoint swept boxes. Cross the
    // broad triangle edge so this production route must perform its TOI retry.
    auto rigid_original=original;rigid_original.positions[0][1]=0.;rigid_original.motion[0][2]=-2.;
    const double rigid_delta[3]={0,0,-2};
    require(ipc_gpu::original_rigid::translation_boxes_overlap(rigid_original.positions,true,true,rigid_delta),
        "wide Original rigid query bypassed CCD via swept boxes");
    require(oc::evaluate_fast(rigid_original).status==oc::Status::NeedsExact
        &&oc::evaluate_exact<256>(rigid_original).status==oc::Status::CapacityExceeded,
        "wide Original rigid TOI retry premise missing");
    const auto rigid_toi=oc::evaluate_exact<2048>(rigid_original);
    require(rigid_toi.status==oc::Status::Ok&&rigid_toi.collision&&rigid_toi.toi==.5,
        "wide Original rigid full-width TOI was not the analytic crossing");

    const V sentinel{{-91,-92,-93}};
    struct Route {bool rigid,direct;gpu::LinearClipMode mode;};
    const Route routes[]={{false,false,gpu::LinearClipMode::Robust},
        {true,false,gpu::LinearClipMode::OriginalSolver},
        {false,false,gpu::LinearClipMode::OriginalSolver},
        {true,false,gpu::LinearClipMode::Robust},
        {false,true,gpu::LinearClipMode::Robust}};
    gpu::GpuLinearClipper reused;
    for(int round=0;round<2;++round)for(const auto route:routes)for(bool full_width:{true,false}) {
        std::vector<V> positions=full_width?wide:
            std::vector<V>{{.25,.25,1},{0,0,0},{1,0,0},{0,1,0}};
        const bool wide_rigid_original=full_width&&route.rigid&&route.mode==gpu::LinearClipMode::OriginalSolver;
        if(wide_rigid_original)positions[0][1]=0.;
        auto targets=positions;targets[0][2]=(full_width||route.direct)&&!wide_rigid_original?2.:-1.;
        Buffer<V> x(positions),proposed(targets),accepted(std::vector<V>(4,sentinel));
        Buffer<int> blocks(std::vector<int>{0,1,2,3}),colors(std::vector<int>{0,1,1,1}),
            indices(std::vector<int>{0}),error(std::vector<int>{0});
        Buffer<std::size_t> offsets(std::vector<std::size_t>{0,1,1,1,1});
        Buffer<unsigned char> deformable(std::vector<unsigned char>{1,0,0,0});
        Buffer<gpu::BpContact> contacts(std::vector<gpu::BpContact>{{{0,1,2,3},0}});
        std::vector<gpu::BpAabb> bounds(4,gpu::BpAabb{{-maximum,-maximum,-maximum},{maximum,maximum,maximum}});
        bounds[0]=gpu::BpAabb{{-4,-4,-4},{4,4,4}};
        Buffer<gpu::BpAabb> boxes(bounds);
        gpu::BroadPhaseView bp{};bp.contacts=contacts.pointer;bp.contact_count=1;
        gpu::DeformableState state{};state.positions=xyz(x.pointer);
        double expected=2.;
        if((!full_width&&!route.direct)||wide_rigid_original) {
            if(route.mode==gpu::LinearClipMode::Robust)expected=.5;
            else {volatile double weight=.9*.5;volatile double delta=weight*(-2.);expected=1.+delta;}
        }
        if(route.rigid) {
            Buffer<V> center(std::vector<V>{{0,0,0}}),target_center(std::vector<V>{{0,0,targets[0][2]-1.}}),
                accepted_center(std::vector<V>{sentinel});
            Buffer<int> body_block(std::vector<int>{0}),body_map(std::vector<int>{0,-1,-1,-1}),body_nodes(std::vector<int>{0});
            Buffer<std::size_t> body_offsets(std::vector<std::size_t>{0,1});
            Buffer<unsigned char> translate(std::vector<unsigned char>{1});
            gpu::RigidScene scene{};scene.bodies=1;scene.body_blocks=body_block.pointer;
            scene.node_body=body_map.pointer;scene.node_offsets=body_offsets.pointer;
            scene.nodes=body_nodes.pointer;scene.translate=translate.pointer;
            gpu::RigidState rigid_state{};rigid_state.com=xyz(center.pointer);
            gpu::RigidWorkspace work{};work.proposed_com=xyz(target_center.pointer);work.accepted_com=xyz(accepted_center.pointer);
            reused.clip_rigid_translation_color(scene,rigid_state,work,state,bp,offsets.pointer,indices.pointer,
                boxes.pointer,colors.pointer,0,xyz(accepted.pointer),error.pointer,true,nullptr,route.mode);
            check(cudaDeviceSynchronize());
            const auto actual_center=accepted_center.read()[0];
            // COM and proxy use the same represented raw displacement. For
            // Original crossing this subtraction is exact by Sterbenz's lemma.
            require(actual_center==V{{0,0,expected-1.}},"reused rigid clipper changed the accepted COM");
            require(center.read()[0]==V{{0,0,0}},"reused rigid clipper committed the COM");
        } else {
            gpu::DeformableScene scene{};scene.nodes=4;scene.deformable=deformable.pointer;scene.node_blocks=blocks.pointer;
            gpu::ElasticWorkspace work{};work.proposed=xyz(proposed.pointer);work.error=error.pointer;
            reused.clip_deformable_color(scene,state,work,bp,offsets.pointer,indices.pointer,boxes.pointer,
                colors.pointer,0,xyz(accepted.pointer),error.pointer,true,nullptr,{},route.direct,route.mode);
            check(cudaDeviceSynchronize());
        }
        if(error.read()[0]!=0) {
            std::ostringstream message;message<<"wide/reuse route failed: round="<<round<<" rigid="<<route.rigid
                <<" direct="<<route.direct<<" Original="<<(route.mode==gpu::LinearClipMode::OriginalSolver)
                <<" full_width="<<full_width<<" status="<<error.read()[0];throw std::runtime_error(message.str());
        }
        const auto out=accepted.read();
        require(out[0]==V{{positions[0][0],positions[0][1],expected}},"wide/reuse changed the certified represented endpoint");
        for(int node=1;node<4;++node)require(out[node]==sentinel,"wide/reuse published an inactive endpoint");
        require(x.read()==positions,"wide/reuse committed physical positions");
        require(proposed.read()==targets,"wide/reuse modified the input proposal");
    }
    std::cout<<"GPU production exact retries/reuse: 20 calls across five kernel routes PASS\n";
}

void original_worker_stride_test() {
    constexpr int active=33,nodes=active+3;
    std::vector<V> positions(nodes),targets(nodes);
    std::vector<int> block_values(nodes),color_values(nodes,1),pair_indices(active);
    std::vector<std::size_t> offset_values(nodes+1,active);
    std::vector<unsigned char> moving(nodes,0);
    std::vector<gpu::BpContact> pair_values(active);
    for(int node=0;node<nodes;++node)block_values[node]=node;
    for(int node=0;node<active;++node) {
        positions[node]=V{{.125+.0625*(node%8),.125,1.}};
        targets[node]=positions[node];targets[node][2]+=double(node+1)/128.;
        color_values[node]=0;moving[node]=1;pair_indices[node]=node;offset_values[node]=node;
        pair_values[node]=gpu::BpContact{{node,active,active+1,active+2},0};
    }
    positions[active]=V{{0,0,0}};positions[active+1]=V{{1,0,0}};positions[active+2]=V{{0,1,0}};
    for(int node=active;node<nodes;++node)targets[node]=positions[node];
    const V sentinel{{-91,-92,-93}};
    Buffer<V> x(positions),proposed(targets),accepted(std::vector<V>(nodes,sentinel));
    Buffer<int> blocks(block_values),colors(color_values),indices(pair_indices),error(std::vector<int>{0});
    Buffer<unsigned char> deformable(moving);Buffer<std::size_t> offsets(offset_values);
    Buffer<gpu::BpContact> contacts(pair_values);
    Buffer<gpu::BpAabb> bounds(std::vector<gpu::BpAabb>(nodes,gpu::BpAabb{{-2,-2,-2},{2,2,2}}));
    gpu::DeformableScene scene{};scene.nodes=nodes;scene.deformable=deformable.pointer;scene.node_blocks=blocks.pointer;
    gpu::DeformableState state{};state.positions=xyz(x.pointer);
    gpu::ElasticWorkspace work{};work.proposed=xyz(proposed.pointer);work.error=error.pointer;
    gpu::BroadPhaseView bp{};bp.contacts=contacts.pointer;bp.contact_count=active;
    gpu::GpuLinearClipper clipper;
    for(int repeat=0;repeat<2;++repeat) {
        clipper.clip_deformable_color(scene,state,work,bp,offsets.pointer,indices.pointer,bounds.pointer,
            colors.pointer,0,xyz(accepted.pointer),error.pointer,true,nullptr,{},false,gpu::LinearClipMode::OriginalSolver);
        check(cudaDeviceSynchronize());require(error.read()[0]==0,"Original strided worker resolve failed");
        const auto out=accepted.read();
        for(int node=0;node<active;++node)require(out[node]==targets[node],"Original worker stride lost or mixed an endpoint");
        for(int node=active;node<nodes;++node)require(out[node]==sentinel,"Original worker stride published an inactive node");
        require(x.read()==positions&&proposed.read()==targets,"Original worker stride modified immutable inputs");
    }
    std::cout<<"GPU Original worker stride: 33 active nodes across 16 workers, two sweeps PASS\n";
}

// Captured coplanar crossings that exposed live exact-arena slot reuse.
// The oracle certifies the actual rounded proposal and returned path.
void coplanar_production_test() {
    namespace ex=ipc_gpu::exact;
    namespace cc=ipc_gpu::ccd;
    struct Fixture {const char*name;bool face;int moving;double positions[4][3],target[3];};
    const Fixture fixtures[]={
        {"PT51",true,0,
            {{0x1.95c17a5de0f58p-1,0x1.64f4bcd2cea86p+0,0},
             {-0x1.5746ddb00057fp+0,-0x1.ae1af7f2ac87p-2,0},
             {-0x1.83ed5ab1fa76cp-1,0x1.9de3b8aa1ae18p-2,0},
             {0x1.2312dd1df02d6p+0,0x1.b1ab5a3976bf8p+0,0}},
            {-0x1.796d45c7d23d2p+0,0x1.4fbf5ac35ae9p-3,0}},
        {"EE222",false,3,
            {{-0x1.f15fc2ea017b8p+0,-0x1.ac14b7e89888p-5,0},
             {0x1.3e5ac3234aa24p-1,0x1.5f022cb352cf8p-2,0},
             {0x1.f60c182c52d58p-1,0x1.80678b515915p-2,0},
             {0x1.6bc276a8a14fcp+0,0x1.be095547e82p-1,0}},
            {-0x1.14c5542c2deccp+0,-0x1.94d5f9a104564p-2,0}}
    };
    struct Route {bool direct;gpu::LinearClipMode mode;};
    const Route routes[]={{false,gpu::LinearClipMode::Robust},
        {true,gpu::LinearClipMode::Robust},{false,gpu::LinearClipMode::OriginalSolver}};
    const V sentinel{{-91,-92,-93}},lo{{-4,-4,-4}},hi{{4,4,4}};
    gpu::GpuLinearClipper clipper;
    for(const auto& fixture:fixtures)for(const auto route:routes) {
        std::vector<V> positions(4);
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)positions[i][k]=fixture.positions[i][k];
        auto targets=positions;
        for(int k=0;k<3;++k)targets[fixture.moving][k]=fixture.target[k];
        const auto represented=analytic_box_endpoint(positions[fixture.moving],targets[fixture.moving],lo,hi);
        cc::LinearQuery q;q.vertex_face=fixture.face;q.gap_floor=1e-8;
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)q.start[i][k]=q.end[i][k]=positions[i][k];
        for(int k=0;k<3;++k)q.end[fixture.moving][k]=represented[k];
        require(ex::validate_step<2048>(q.start,q.start,q.vertex_face,q.gap_floor)==ex::Status::Safe,
            "coplanar fixture has an unhealthy initial state");
        require(ex::validate_step<2048>(q.start,q.end,q.vertex_face,q.gap_floor)==ex::Status::Unsafe
            &&ex::validate_collision_step<2048>(q.start,q.end,q.vertex_face)==ex::Status::Unsafe,
            "coplanar represented proposal must cross the other primitive");
        require(cc::validate_linear(q)!=cc::Status::Safe,"coplanar proposal bypassed the exact resolver");
        Buffer<V> x(positions),proposed(targets),accepted(std::vector<V>(4,sentinel));
        Buffer<int> blocks(std::vector<int>{0,1,2,3}),indices(std::vector<int>{0}),error(std::vector<int>{0});
        std::vector<int> color_values(4,1);color_values[fixture.moving]=0;Buffer<int> colors(color_values);
        std::vector<unsigned char> mask(4,0);mask[fixture.moving]=1;Buffer<unsigned char> deformable(mask);
        std::vector<std::size_t> rows(5,1);for(int i=0;i<=fixture.moving;++i)rows[i]=0;Buffer<std::size_t> offsets(rows);
        Buffer<gpu::BpContact> contacts(std::vector<gpu::BpContact>{{{0,1,2,3},fixture.face?0:1}});
        Buffer<gpu::BpAabb> bounds(std::vector<gpu::BpAabb>(4,gpu::BpAabb{{-4,-4,-4},{4,4,4}}));
        gpu::DeformableScene scene{};scene.nodes=4;scene.deformable=deformable.pointer;scene.node_blocks=blocks.pointer;
        gpu::DeformableState state{};state.positions=xyz(x.pointer);
        gpu::ElasticWorkspace work{};work.proposed=xyz(proposed.pointer);work.error=error.pointer;
        gpu::BroadPhaseView broad{};broad.contacts=contacts.pointer;broad.contact_count=1;
        clipper.clip_deformable_color(scene,state,work,broad,offsets.pointer,indices.pointer,bounds.pointer,
            colors.pointer,0,xyz(accepted.pointer),error.pointer,true,nullptr,{},route.direct,route.mode);
        check(cudaDeviceSynchronize());
        std::ostringstream context;context<<"coplanar "<<fixture.name<<" direct="<<route.direct
            <<" mode="<<int(route.mode);
        require(error.read()[0]==0,(context.str()+" production resolver failed").c_str());
        const auto out=accepted.read();const auto& endpoint=out[fixture.moving];
        require(endpoint!=represented,(context.str()+" accepted the known crossing proposal").c_str());
        require(endpoint!=positions[fixture.moving],(context.str()+" lost healthy nonzero progress").c_str());
        for(int k=0;k<3;++k)q.end[fixture.moving][k]=endpoint[k];
        const auto actual=route.mode==gpu::LinearClipMode::OriginalSolver
            ?ex::validate_collision_step<2048>(q.start,q.end,q.vertex_face)
            :ex::validate_step<2048>(q.start,q.end,q.vertex_face,q.gap_floor);
        require(actual==ex::Status::Safe,(context.str()+" returned an uncertified rounded path").c_str());
        for(int node=0;node<4;++node)if(node!=fixture.moving)
            require(out[node]==sentinel,"coplanar resolver published an inactive endpoint");
        require(x.read()==positions&&proposed.read()==targets,"coplanar resolver modified immutable input");
    }
    std::cout<<"GPU coplanar PT/EE: baseline, direct and Original rounded paths independently certified PASS\n";
}

struct TranslationPairEnvironment {
    bool existed=false;std::string value;
    bool batch_existed=false,cap_existed=false,queue_existed=false;std::string batch_value,cap_value,queue_value;
    TranslationPairEnvironment() {
        if(const char* old=std::getenv("IPC_GPU_RIGID_TRANSLATION_PAIR_THREADS")){existed=true;value=old;}
        if(const char* old=std::getenv("IPC_GPU_RIGID_CCD_BATCH")){batch_existed=true;batch_value=old;}
        if(const char* old=std::getenv("IPC_GPU_RIGID_CCD_BATCH_MAX_PAIRS")){cap_existed=true;cap_value=old;}
        if(const char* old=std::getenv("IPC_GPU_RIGID_CCD_BATCH_QUEUE_CAPACITY")){queue_existed=true;queue_value=old;}
        unsetenv("IPC_GPU_RIGID_CCD_BATCH_MAX_PAIRS");
        unsetenv("IPC_GPU_RIGID_CCD_BATCH_QUEUE_CAPACITY");
    }
    ~TranslationPairEnvironment() {
        if(existed)setenv("IPC_GPU_RIGID_TRANSLATION_PAIR_THREADS",value.c_str(),1);
        else unsetenv("IPC_GPU_RIGID_TRANSLATION_PAIR_THREADS");
        if(batch_existed)setenv("IPC_GPU_RIGID_CCD_BATCH",batch_value.c_str(),1);
        else unsetenv("IPC_GPU_RIGID_CCD_BATCH");
        if(cap_existed)setenv("IPC_GPU_RIGID_CCD_BATCH_MAX_PAIRS",cap_value.c_str(),1);
        else unsetenv("IPC_GPU_RIGID_CCD_BATCH_MAX_PAIRS");
        if(queue_existed)setenv("IPC_GPU_RIGID_CCD_BATCH_QUEUE_CAPACITY",queue_value.c_str(),1);
        else unsetenv("IPC_GPU_RIGID_CCD_BATCH_QUEUE_CAPACITY");
    }
};
struct TranslationPairFixture {
    std::string name;
    std::vector<V> positions,com,target;
    std::vector<int> owners,colors;
    std::vector<unsigned char> translate;
    std::vector<gpu::BpContact> contacts;
    std::vector<int> contact_order; // Optional explicit CSR order, independent of global pair index.
    std::vector<gpu::BpAabb> bounds;
    bool ccd=true;int color=0,prior_error=0,expected_error=0;
};
struct TranslationPairOutcome {int error;std::vector<V> com,proxy;};

TranslationPairOutcome run_translation_pair_fixture(const TranslationPairFixture& f,
    unsigned width,gpu::GpuLinearClipper& reused,bool batch=false) {
    setenv("IPC_GPU_RIGID_CCD_BATCH",batch?"1":"0",1);
    if(width==0)unsetenv("IPC_GPU_RIGID_TRANSLATION_PAIR_THREADS");
    else setenv("IPC_GPU_RIGID_TRANSLATION_PAIR_THREADS",std::to_string(width).c_str(),1);
    const V sentinel{{-91,-92,-93}};
    std::vector<int> body_blocks(f.com.size()),node_list,pair_indices;
    std::vector<std::size_t> node_offsets{0},contact_offsets{0};
    for(std::size_t body=0;body<f.com.size();++body) {
        body_blocks[body]=static_cast<int>(body);
        for(std::size_t node=0;node<f.positions.size();++node)
            if(f.owners[node]==static_cast<int>(body))node_list.push_back(static_cast<int>(node));
        node_offsets.push_back(node_list.size());
        require(f.contact_order.empty()||f.contact_order.size()==f.contacts.size(),"invalid fixture CSR permutation size");
        for(std::size_t index=0;index<f.contacts.size();++index) {
            const auto pair=f.contact_order.empty()?index:static_cast<std::size_t>(f.contact_order[index]);
            require(pair<f.contacts.size(),"invalid fixture CSR permutation index");
            bool involved=false;
            for(int node:f.contacts[pair].vertices)involved=involved||f.owners[node]==static_cast<int>(body);
            if(involved)pair_indices.push_back(static_cast<int>(pair));
        }
        contact_offsets.push_back(pair_indices.size());
    }
    Buffer<V> x(f.positions),center(f.com),target(f.target),
        accepted_com(std::vector<V>(f.com.size(),sentinel)),accepted(std::vector<V>(f.positions.size(),sentinel));
    Buffer<int> owners(f.owners),nodes(node_list),blocks(body_blocks),colors(f.colors),
        indices(pair_indices),error(std::vector<int>{f.prior_error});
    Buffer<unsigned char> translate(f.translate),rotate(std::vector<unsigned char>(f.com.size(),0));
    Buffer<std::size_t> body_rows(node_offsets),pair_rows(contact_offsets);
    Buffer<gpu::BpContact> contacts(f.contacts);Buffer<gpu::BpAabb> bounds(f.bounds);
    gpu::RigidScene scene{};scene.bodies=f.com.size();scene.body_blocks=blocks.pointer;
    scene.node_body=owners.pointer;scene.node_offsets=body_rows.pointer;scene.nodes=nodes.pointer;
    scene.translate=translate.pointer;scene.rotate=rotate.pointer;
    gpu::RigidState state{};state.com=xyz(center.pointer);
    gpu::RigidWorkspace work{};work.proposed_com=xyz(target.pointer);work.accepted_com=xyz(accepted_com.pointer);
    gpu::DeformableState vertices{};vertices.positions=xyz(x.pointer);
    gpu::BroadPhaseView bp{};bp.contacts=contacts.pointer;bp.contact_count=f.contacts.size();
    reused.clip_rigid_translation_color(scene,state,work,vertices,bp,pair_rows.pointer,indices.pointer,
        bounds.pointer,colors.pointer,f.color,xyz(accepted.pointer),error.pointer,f.ccd,nullptr,
        gpu::LinearClipMode::OriginalSolver);
    check(cudaDeviceSynchronize());
    const auto actual_positions=x.read(),actual_centers=center.read(),actual_targets=target.read();
    require(actual_positions.size()==f.positions.size()
        &&std::memcmp(actual_positions.data(),f.positions.data(),f.positions.size()*sizeof(V))==0,
        "translation pair prototype committed immutable positions");
    require(actual_centers==f.com&&actual_targets==f.target,
        "translation pair prototype modified immutable COM/proposal");
    return {error.read()[0],accepted_com.read(),accepted.read()};
}

void original_rigid_pair_parallel_test() {
    TranslationPairEnvironment restore_environment;
    gpu::GpuLinearClipper reused;
    auto fixture=[](const char* name,bool face,bool first,V raw=V{{0,0,-2}}) {
        TranslationPairFixture f;f.name=name;
        f.positions=face?std::vector<V>{{.2,.2,1},{0,0,0},{1,0,0},{0,1,0}}
            :std::vector<V>{{-1,0,1},{1,0,1},{0,-1,0},{0,1,0}};
        if(!first)for(auto& position:f.positions)position[2]=1.-position[2];
        f.com={V{{0,0,0}}};f.target={raw};f.owners.assign(4,-1);f.colors={0};f.translate={1};
        for(int node=0;node<4;++node)if((node<(face?1:2))==first)f.owners[node]=0;
        f.contacts={gpu::BpContact{{0,1,2,3},face?0:1}};
        f.bounds.assign(4,gpu::BpAabb{{-4,-4,-4},{4,4,4}});
        return f;
    };
    std::vector<TranslationPairFixture> fixtures;
    for(bool face:{true,false})for(bool first:{true,false}) {
        auto f=fixture(face?(first?"point":"triangle"):(first?"first edge":"second edge"),face,first);
        // Cross both chunk boundaries and a partial last chunk.
        const auto contact=f.contacts[0];f.contacts.clear();
        for(int index=0;index<131;++index)f.contacts.push_back(contact);
        fixtures.push_back(f);
        f.ccd=false;f.name+=" CCD off";fixtures.push_back(f);
    }
    auto base=fixture("empty contacts",true,true);base.contacts.clear();fixtures.push_back(base);
    base=fixture("no rigid bodies",true,true);base.com.clear();base.target.clear();base.colors.clear();
    base.translate.clear();base.owners.assign(4,-1);fixtures.push_back(base);
    base=fixture("healthy no-op",true,true,V{{0,0,0}});fixtures.push_back(base);
    base.name="nonfinite fixed geometry no-op";base.positions[1][0]=INFINITY;
    base.expected_error=int(gpu::LinearClipError::InvalidInput);fixtures.push_back(base);
    base=fixture("healthy no-op",true,true,V{{0,0,0}});
    base.name="initial-contact no-op";base.positions[0][2]=0;
    base.expected_error=int(gpu::LinearClipError::InitialContact);fixtures.push_back(base);
    base=fixture("positive-band no-op",true,true);base.positions[0][2]=1e-12;fixtures.push_back(base);
    base=fixture("tiny motion",true,true,V{{0,0,1e-15}});fixtures.push_back(base);
    base=fixture("rounded COM keeps raw proxy motion",true,true);
    base.com[0][2]=1e16;base.target[0][2]=1e16-2.;fixtures.push_back(base);
    base=fixture("box backtracking",true,true,V{{0,0,4}});base.contacts.clear();
    base.bounds.assign(4,gpu::BpAabb{{-2,-2,-2},{2,2,2}});fixtures.push_back(base);
    base=fixture("start outside every trial",true,true);base.ccd=false;
    base.bounds[0]=gpu::BpAabb{{-4,-4,-.5},{4,4,.5}};
    base.expected_error=int(gpu::LinearClipError::Unresolved);fixtures.push_back(base);
    base.name="start outside all 65 staged trials";base.ccd=true;fixtures.push_back(base);
    base=fixture("nonfinite proposal",true,true);base.target[0][0]=INFINITY;
    base.expected_error=int(gpu::LinearClipError::InvalidInput);fixtures.push_back(base);
    base=fixture("prior error",true,true);base.prior_error=77;base.expected_error=77;fixtures.push_back(base);
    base=fixture("inactive color",true,true);base.color=1;fixtures.push_back(base);
    base=fixture("translation disabled",true,true);base.translate[0]=0;fixtures.push_back(base);
    // The raw TOI pass must inspect every CSR pair even after a collision.
    // Ownership errors after a hit must never be mistaken for an early miss.
    base=fixture("ownership after raw collision",true,true);
    const gpu::BpContact malformed{{0,0,2,3},0};
    base.contacts.push_back(malformed);base.expected_error=int(gpu::LinearClipError::UnsupportedMotion);
    fixtures.push_back(base);
    std::reverse(base.contacts.begin(),base.contacts.end());base.name="ownership before raw collision";fixtures.push_back(base);
    base=fixture("initial-contact error before speculative ownership",true,true,V{{0,0,0}});
    base.positions[0][2]=0;base.contacts.push_back(malformed);
    base.expected_error=int(gpu::LinearClipError::InitialContact);fixtures.push_back(base);
    std::reverse(base.contacts.begin(),base.contacts.end());base.name="ownership before initial-contact error";
    base.expected_error=int(gpu::LinearClipError::UnsupportedMotion);fixtures.push_back(base);

    // Explicit exact-retry premise: this crossing cannot fit the small arena.
    const double maximum=std::numeric_limits<double>::max(),minimum=std::numeric_limits<double>::denorm_min();
    base=fixture("256 to 2048 exact retry",true,true);
    base.positions={V{{.25,0,1}},V{{-maximum,0,0}},V{{maximum,0,0}},V{{0,minimum,0}}};
    base.bounds.assign(4,gpu::BpAabb{{-maximum,-maximum,-maximum},{maximum,maximum,maximum}});
    base.bounds[0]=gpu::BpAabb{{-4,-4,-4},{4,4,4}};
    ipc_gpu::original_ccd::Query retry;
    for(int node=0;node<4;++node)for(int k=0;k<3;++k)retry.positions[node][k]=base.positions[node][k];
    retry.motion[0][2]=-2.;
    require(ipc_gpu::original_ccd::evaluate_fast(retry).status==ipc_gpu::original_ccd::Status::NeedsExact
        &&ipc_gpu::original_ccd::evaluate_exact<256>(retry).status==ipc_gpu::original_ccd::Status::CapacityExceeded,
        "translation pair test lacks real small-capacity retry");
    const auto full=ipc_gpu::original_ccd::evaluate_exact<2048>(retry);
    require(full.status==ipc_gpu::original_ccd::Status::Ok&&full.collision&&full.toi==.5,
        "translation pair exact retry has no analytic .5 crossing");
    fixtures.push_back(base);
    const auto retry_fixture=base;

    base=fixture("different collision TOIs",true,true);
    base.positions.insert(base.positions.end(),{V{{0,0,.5}},V{{1,0,.5}},V{{0,1,.5}}});
    base.owners.insert(base.owners.end(),{-1,-1,-1});
    base.bounds.assign(7,gpu::BpAabb{{-4,-4,-4},{4,4,4}});
    base.contacts.push_back(gpu::BpContact{{0,4,5,6},0});fixtures.push_back(base);
    base.name="different collision TOIs shuffled CSR";base.contact_order={1,0};fixtures.push_back(base);
    base=fixture("shuffled CSR error priority",true,true,V{{0,0,0}});
    base.positions[0][2]=0;base.contacts.push_back(malformed);base.contact_order={1,0};
    base.expected_error=int(gpu::LinearClipError::UnsupportedMotion);fixtures.push_back(base);

    base=fixture("multiple bodies same color",true,true);
    base.positions.insert(base.positions.end(),{V{{10.2,.2,1}},V{{10,0,0}},V{{11,0,0}},V{{10,1,0}}});
    base.com.push_back(V{{10,0,0}});base.target.push_back(V{{10,0,-2}});
    base.owners.insert(base.owners.end(),{1,-1,-1,-1});base.colors.push_back(0);base.translate.push_back(1);
    base.contacts.push_back(gpu::BpContact{{4,5,6,7},0});
    base.bounds.assign(8,gpu::BpAabb{{-20,-4,-4},{20,4,4}});fixtures.push_back(base);
    base.name="multiple colors";base.colors[1]=1;fixtures.push_back(base);
    base.name="other color selected";base.color=1;fixtures.push_back(base);
    base.name="later disabled body initializes proxies";base.colors[1]=0;base.color=0;base.translate[1]=0;fixtures.push_back(base);
    base=fixture("fixed bodies before/after selected success",true,true);
    for(int body=1;body<3;++body) {
        const int first=static_cast<int>(base.positions.size());const double shift=10.*body;
        base.positions.insert(base.positions.end(),{V{{shift+.2,.2,1}},V{{shift,0,0}},V{{shift+1,0,0}},V{{shift,1,0}}});
        base.com.push_back(V{{shift,0,0}});base.target.push_back(V{{shift,0,-2}});
        base.owners.insert(base.owners.end(),{body,-1,-1,-1});base.colors.push_back(0);base.translate.push_back(0);
        base.contacts.push_back(gpu::BpContact{{first,first+1,first+2,first+3},0});
    }
    base.translate={0,1,0};base.bounds.assign(12,gpu::BpAabb{{-30,-4,-4},{30,4,4}});fixtures.push_back(base);
    base.name="fixed frontier raw ownership error";base.contacts[1]=gpu::BpContact{{4,4,6,7},0};
    base.expected_error=int(gpu::LinearClipError::UnsupportedMotion);fixtures.push_back(base);
    base.name="fixed frontier no-op initial contact";base.contacts[1]=gpu::BpContact{{4,5,6,7},0};
    base.positions[4][2]=0.;base.target[1]=base.com[1];
    base.expected_error=int(gpu::LinearClipError::InitialContact);fixtures.push_back(base);
    base=fixture("alternating inactive and disabled body controls",true,true);
    for(int body=1;body<12;++body) {
        const int first=static_cast<int>(base.positions.size());const double shift=10.*body;
        base.positions.insert(base.positions.end(),{V{{shift+.2,.2,1}},V{{shift,0,0}},V{{shift+1,0,0}},V{{shift,1,0}}});
        base.com.push_back(V{{shift,0,0}});base.target.push_back(V{{shift,0,-2}});
        base.owners.insert(base.owners.end(),{body,-1,-1,-1});
        base.colors.push_back(body%3==1?1:0);base.translate.push_back(body%3==2?0:1);
        base.contacts.push_back(gpu::BpContact{{first,first+1,first+2,first+3},0});
    }
    base.bounds.assign(base.positions.size(),gpu::BpAabb{{-200,-4,-4},{200,4,4}});fixtures.push_back(base);

    for(const auto& f:fixtures) {
        const auto reference=run_translation_pair_fixture(f,1,reused);
        require(reference.error==f.expected_error,(f.name+" unexpected serial error "+std::to_string(reference.error)).c_str());
        for(bool batch:{false,true})for(unsigned width:{0u,8u,16u,32u,64u,128u})for(int round=0;round<2;++round) {
            const auto actual=run_translation_pair_fixture(f,width,reused,batch);
            const std::string context=f.name+" batch="+std::to_string(batch)+" width="+std::to_string(width)+" round="+std::to_string(round);
            require(actual.error==reference.error,(context+" status differs").c_str());
            require(actual.com.size()==reference.com.size()
                &&(actual.com.empty()||std::memcmp(actual.com.data(),reference.com.data(),actual.com.size()*sizeof(V))==0),
                (context+" accepted COM bits differ").c_str());
            require(actual.proxy.size()==reference.proxy.size()
                &&std::memcmp(actual.proxy.data(),reference.proxy.data(),actual.proxy.size()*sizeof(V))==0,
                (context+" accepted proxy bits differ").c_str());
        }
    }
    {
        struct RestoreWorkerLimit {
            bool present=false;std::string value;
            RestoreWorkerLimit() {
                if(const char* old=std::getenv("IPC_GPU_ORIGINAL_CCD_WORKERS")){present=true;value=old;}
            }
            ~RestoreWorkerLimit() {
                if(present)setenv("IPC_GPU_ORIGINAL_CCD_WORKERS",value.c_str(),1);
                else unsetenv("IPC_GPU_ORIGINAL_CCD_WORKERS");
            }
        } restore_worker_limit;
        // A small near-band query requires exact evaluation without wide-limb
        // expansion. 257 duplicate jobs exceed even the largest 256-worker
        // pool, forcing queue continuation and a fresh-arena second job.
        auto queue_fixture=fixture("257 NeedsExact jobs",true,true);
        queue_fixture.positions[0]=V{{.25,.25,5e-11}};
        ipc_gpu::original_ccd::Query queued;
        for(int node=0;node<4;++node)for(int axis=0;axis<3;++axis)
            queued.positions[node][axis]=queue_fixture.positions[node][axis];
        queued.motion[0][2]=-2.;
        require(ipc_gpu::original_ccd::evaluate_fast(queued).status==ipc_gpu::original_ccd::Status::NeedsExact,
            "queue-extreme fixture bypassed NeedsExact");
        const auto resolved=ipc_gpu::original_ccd::evaluate_exact<256>(queued);
        require(resolved.status==ipc_gpu::original_ccd::Status::Ok&&resolved.collision&&resolved.toi==0.,
            "queue-extreme fixture is not a cheap small-arena initial-band hit");
        queue_fixture.contacts.assign(257,queue_fixture.contacts.front());
        const auto retry_reference=run_translation_pair_fixture(retry_fixture,1,reused,false);
        const auto queue_reference=run_translation_pair_fixture(queue_fixture,1,reused,false);
        require(retry_reference.error==0&&queue_reference.error==0,"worker-limit scalar premise failed");
        auto identical=[](const TranslationPairOutcome& a,const TranslationPairOutcome& b,const std::string& label) {
            require(a.error==b.error&&a.com.size()==b.com.size()&&a.proxy.size()==b.proxy.size()
                &&std::memcmp(a.com.data(),b.com.data(),a.com.size()*sizeof(V))==0
                &&std::memcmp(a.proxy.data(),b.proxy.data(),a.proxy.size()*sizeof(V))==0,
                (label+" scalar/batch output bits differ").c_str());
        };
        for(const char* limit:{"1","256"}) {
            setenv("IPC_GPU_ORIGINAL_CCD_WORKERS",limit,1);
            // The limit is captured by construction; reuse this same instance
            // across wide retry and queued jobs, then repeat both calls.
            gpu::GpuLinearClipper limited;
            for(int round=0;round<2;++round) {
                identical(run_translation_pair_fixture(retry_fixture,1,limited,true),retry_reference,
                    std::string("wide retry workers=")+limit);
                identical(run_translation_pair_fixture(queue_fixture,1,limited,true),queue_reference,
                    std::string("257 queued jobs workers=")+limit);
            }
        }
        // Overflow is a dispatch fallback, never a truncated query set. Force
        // both raw-TOI and represented-trial cold queues past one slot.
        auto trial_queue=fixture("257 trial certificate jobs",true,true,V{{0,0,0}});
        trial_queue.positions={V{{.25,.25,std::nextafter(.5,1.)}},V{{1,0,0}},V{{0,1,0}},V{{0,0,1}}};
        ipc_gpu::ccd::LinearQuery stationary;
        for(int node=0;node<4;++node)for(int axis=0;axis<3;++axis)
            stationary.start[node][axis]=stationary.end[node][axis]=trial_queue.positions[node][axis];
        require(!ipc_gpu::ccd::detail::separated(stationary,0.,1.,0.)
            &&ipc_gpu::original_rigid::represented_path<256>(stationary.start,stationary.end,true)
                ==ipc_gpu::original_rigid::RepresentationStatus::Safe,
            "trial queue overflow fixture lacks a healthy uncertain prefix");
        trial_queue.contacts.assign(257,trial_queue.contacts.front());
        auto frontier=queue_fixture;frontier.name="queue overflow fixed frontier";
        frontier.com.insert(frontier.com.begin(),V{{10,0,2}});frontier.com.push_back(V{{-10,0,2}});
        frontier.target.insert(frontier.target.begin(),V{{10,0,2}});frontier.target.push_back(V{{-10,0,2}});
        frontier.positions.insert(frontier.positions.end(),{V{{10,0,2}},V{{-10,0,2}}});
        frontier.owners[0]=1;frontier.owners.insert(frontier.owners.end(),{0,2});
        frontier.colors={0,0,0};frontier.translate={0,1,0};
        frontier.bounds.assign(6,gpu::BpAabb{{-20,-4,-4},{20,4,4}});
        auto intersecting=fixture("overflow trial initial contact",true,true,V{{0,0,0}});
        intersecting.positions[0][2]=0.;intersecting.contacts.assign(257,intersecting.contacts.front());
        intersecting.expected_error=int(gpu::LinearClipError::InitialContact);
        setenv("IPC_GPU_RIGID_CCD_BATCH_QUEUE_CAPACITY","1",1);
        for(const auto* f:{&queue_fixture,&trial_queue,&frontier,&intersecting}) {
            const auto reference=run_translation_pair_fixture(*f,1,reused,false);
            require(reference.error==f->expected_error,"overflow scalar fixture premise failed");
            for(unsigned width:{1u,32u,128u})for(int round=0;round<2;++round)
                identical(run_translation_pair_fixture(*f,width,reused,true),reference,
                    f->name+" queue-cap=1 width="+std::to_string(width));
        }
        for(const char* bad:{"0","67108865","-1","32x"}) {
            setenv("IPC_GPU_RIGID_CCD_BATCH_QUEUE_CAPACITY",bad,1);bool rejected=false;
            try {(void)run_translation_pair_fixture(queue_fixture,1,reused,true);}
            catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"invalid rigid batch queue capacity was accepted");
        }
        unsetenv("IPC_GPU_RIGID_CCD_BATCH_QUEUE_CAPACITY");
    }
    // Capacity is a dispatch limit, never contact pruning. Force a 131-pair
    // row through a one-pair allocation and verify scalar/CTA fallback bytes.
    setenv("IPC_GPU_RIGID_CCD_BATCH_MAX_PAIRS","1",1);
    for(unsigned width:{1u,32u,128u}) {
        const auto reference=run_translation_pair_fixture(fixtures.front(),1,reused,false);
        const auto actual=run_translation_pair_fixture(fixtures.front(),width,reused,true);
        require(actual.error==reference.error&&actual.com.size()==reference.com.size()
            &&actual.proxy.size()==reference.proxy.size()
            &&std::memcmp(actual.com.data(),reference.com.data(),actual.com.size()*sizeof(V))==0
            &&std::memcmp(actual.proxy.data(),reference.proxy.data(),actual.proxy.size()*sizeof(V))==0,
            "bounded batch capacity fallback changed a complete contact row");
    }
    for(const char* bad:{"0","67108865","-1","32x"}) {
        setenv("IPC_GPU_RIGID_CCD_BATCH_MAX_PAIRS",bad,1);bool rejected=false;
        try {(void)run_translation_pair_fixture(fixtures.front(),1,reused,true);}
        catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"invalid rigid batch capacity was accepted");
    }
    unsetenv("IPC_GPU_RIGID_CCD_BATCH_MAX_PAIRS");
    for(const char* bad:{"0","2","4","256","32x","-1"}) {
        setenv("IPC_GPU_RIGID_TRANSLATION_PAIR_THREADS",bad,1);
        bool rejected=false;
        // Reuse the public caller with valid storage so parser rejection is
        // distinguished from unrelated scene/data validation.
        auto f=fixture("invalid parser",true,true);
        Buffer<V> x(f.positions),center(f.com),target(f.target),accepted(f.positions),accepted_center(f.com);
        Buffer<int> body_blocks(std::vector<int>{0}),owners(f.owners),nodes(std::vector<int>{0}),
            colors(f.colors),indices(std::vector<int>{0}),error(std::vector<int>{0});
        Buffer<std::size_t> body_rows(std::vector<std::size_t>{0,1}),pair_rows(std::vector<std::size_t>{0,1});
        Buffer<unsigned char> translate(f.translate),rotate(std::vector<unsigned char>{0});
        Buffer<gpu::BpContact> pairs(f.contacts);Buffer<gpu::BpAabb> bounds(f.bounds);
        gpu::RigidScene scene{};scene.bodies=1;scene.body_blocks=body_blocks.pointer;scene.node_body=owners.pointer;
        scene.node_offsets=body_rows.pointer;scene.nodes=nodes.pointer;scene.translate=translate.pointer;scene.rotate=rotate.pointer;
        gpu::RigidState state{};state.com=xyz(center.pointer);
        gpu::RigidWorkspace work{};work.proposed_com=xyz(target.pointer);work.accepted_com=xyz(accepted_center.pointer);
        gpu::DeformableState vertices{};vertices.positions=xyz(x.pointer);
        gpu::BroadPhaseView bp{};bp.contacts=pairs.pointer;bp.contact_count=1;
        try {reused.clip_rigid_translation_color(scene,state,work,vertices,bp,pair_rows.pointer,indices.pointer,
            bounds.pointer,colors.pointer,0,xyz(accepted.pointer),error.pointer,true,nullptr,gpu::LinearClipMode::OriginalSolver);}
        catch(const std::invalid_argument&){rejected=true;}
        require(rejected,(std::string("invalid translation pair width accepted: ")+bad).c_str());
    }
    std::cout<<"GPU original rigid translation: scalar/CTA/stream bitwise ordered differential, atomic TOI, frontier, row/queue fallback/reuse PASS\n";
}

int main() {
    try {
        deformable_test(1.,-1.,true,0,.5);
        deformable_test(1.,-1.,true,0,.875,false,true);
        deformable_test(1.,-1.,false,0,-1.);
        deformable_test(1e-12,2e-12,true,0,2e-12);
        deformable_test(1e-12,1e-12,true,0,1e-12);
        deformable_test(1e-12,1e-12,true,0,1e-12,true);
        deformable_test(1e-12,-1e200,true,0,1e-12,false,false,true);
        deformable_test(0.,1.,true,int(gpu::LinearClipError::InitialContact),0.);
        deformable_test(3.,1.,true,int(gpu::LinearClipError::OutsideBlueBox),0.);
        deformable_test(1.,std::numeric_limits<double>::quiet_NaN(),true,77,0.,false,false,false,77);
        rigid_test(1.,-1.,.5,true);
        rigid_test(1.,-1.,-1.,false);
        rigid_test(1e-12,2e-12,2e-12,true);
        rigid_test(1e-12,1e-12,1e-12,true);
        rigid_test(1.,-1.,.5,true,true);
        rigid_test(1e-12,2e-12,2e-12,true,true);
        cooperative_endpoint_test(false);
        cooperative_endpoint_test(true);
        box_clamp_test();
        box_clamp_failclosed_test();
        wide_capacity_reuse_test();
        original_worker_stride_test();
        coplanar_production_test();
        original_rigid_pair_parallel_test();
        std::cout<<"Passed GPU linear pipeline: crossings, CCD-off, tiny-gap receding/tangential, initial contact, blue bounds, componentwise inset clamp, rigid COM/proxy endpoints\n";
        return 0;
    } catch(const std::exception&e) { std::cerr<<e.what()<<'\n';return 1; }
}
