#include "linear_pipeline.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstring>
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
        std::cout<<"Passed GPU linear pipeline: crossings, CCD-off, tiny-gap receding/tangential, initial contact, blue bounds, componentwise inset clamp, rigid COM/proxy endpoints\n";
        return 0;
    } catch(const std::exception&e) { std::cerr<<e.what()<<'\n';return 1; }
}
