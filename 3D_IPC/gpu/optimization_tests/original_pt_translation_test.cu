#include "pipeline_fixture_extracted.inc"
#include "publisher_extracted.h"
#include "qualified_evaluator_oracle.h"

namespace {
using namespace ipc_gpu::solver;
template<class T>__device__ bool pt_equal_bytes(const T& a,const T& b) {
    const auto* x=reinterpret_cast<const unsigned char*>(&a);
    const auto* y=reinterpret_cast<const unsigned char*>(&b);
    for(std::size_t k=0;k<sizeof(T);++k)if(x[k]!=y[k])return false;
    return true;
}
__global__ void pt_publication_differential(const ContactInput* inputs,std::size_t n,
    ContactOutput* expected,ContactOutput* actual,unsigned* failures) {
    const auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=n)return;
    const auto& input=inputs[index];
    const auto full=evaluate_contact_original_qualified_oracle(input);
    const auto thin=evaluate_contact_original_self_zero(input,0);
    unsigned failure=0;
    for(int gated=0;gated<2;++gated) {
        unsigned char expected_activity=0x7f,actual_activity=0x7f;
        const auto slot=2*index+gated;
        store_contact_output(input,full,expected+slot,nullptr,gated?&expected_activity:nullptr);
        store_original_pt_translation_zero(thin,actual+slot,gated?&actual_activity:nullptr);
        if(expected_activity!=actual_activity||!pt_equal_bytes(expected[slot],actual[slot]))failure|=1u<<gated;
    }
    failures[index]=failure;
}
GpuPipelineTestInput pt_fixture() {
    GpuPipelineTestInput input;
    const double x[8][3]={{.2,.3,.17},{0,0,0},{1,0,0},{0,1,0},
        {.2,.3,.17},{3,0,0},{2.5,-.5,.17},{2.5,.5,.17}};
    for(int n=0;n<8;++n)for(int k=0;k<3;++k) {
        input.positions[n][k]=x[n][k];input.previous[n][k]=x[n][k]-.001*(n+1)*(k+1);
        input.references[n][k]=x[n][k];
    }
    return input;
}
void pt_public_pipeline() {
    using namespace ipc_gpu;
    ColorLocalSetting no_local(false);
    ContactOptionSetting no_box("IPC_GPU_BOX_CONTACT_PRUNING","0");
    ContactOptionSetting no_ordinary_thin("IPC_GPU_ORIGINAL_SELF_ZERO_GATHER","0");
    std::size_t comparisons=0,active_witnesses=0;
    const auto input=pt_fixture();
    for(const char* value:{"2","-1","invalid"}) {
        ContactOptionSetting malformed("IPC_GPU_ORIGINAL_PT_TRANSLATION_ZERO_GATHER",value);
        bool rejected=false;
        try{GpuContactPipeline invalid(65);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"fresh PT constructor accepted malformed execution option");
    }
    auto paired=[&](const GpuPipelineTestInput& fixture,MixedOptions option,
        GpuContactPipeline* off=nullptr,GpuContactPipeline* on=nullptr) {
        GpuPipelineTestOutput expected{},actual{};ContactStorageBytes a,b;
        {
            ContactOptionSetting flag("IPC_GPU_ORIGINAL_PT_TRANSLATION_ZERO_GATHER","0");
            run_mixed(fixture,expected,option,&a,off);
        }
        {
            ContactOptionSetting flag("IPC_GPU_ORIGINAL_PT_TRANSLATION_ZERO_GATHER","1");
            run_mixed(fixture,actual,option,&b,on);
        }
        require(expected.error==actual.error,"thin PT public status changed");
        compare_mixed_bytes(actual,expected);
        require(a.packet_bytes==b.packet_bytes&&a.exact_bytes==b.exact_bytes,
            "thin PT changed packet/exact allocation bounds");
        ++comparisons;return expected;
    };
    for(bool activity:{false,true})for(bool cache:{false,true})for(bool compact:{false,true})
        for(double friction:{0.0,-0.0})
        for(std::size_t capacity:{std::size_t(65),std::size_t(129),std::size_t(257)})
        for(int owner=0;owner<3;++owner) {
            ContactOptionSetting activity_setting("IPC_GPU_CONTACT_OUTPUT_ACTIVITY",activity?"1":"0");
            RigidCacheSetting c(cache);RigidCompactSetting k(compact);
            MixedOptions option;option.capacity=capacity;option.original=true;option.seeded=true;
            option.point_only=true;option.repeat_contacts=2*capacity+1;option.d_hat=.2;option.no_sdf=true;
            option.pt_zero_test_friction=friction;option.pt_zero_test_point_owner=owner==1;
            option.missing_owner=owner==2;
            const auto expected=paired(input,option);
            require(expected.error==0,"healthy PT owner fixture failed");
            auto control=option;control.pt_zero_test_kbarrier=0.;
            const auto no_force=paired(input,control);
            require(std::memcmp(expected.rigid_com,no_force.rigid_com,sizeof(expected.rigid_com))!=0,
                "PT owner fixture has no active barrier contribution");
            ++active_witnesses;
        }
    // Same captured owners cross tiled/untiled/empty/disabled/reordered states.
    RigidCacheSetting cache(true);RigidCompactSetting compact(true);
    for(bool activity:{false,true}) {
        ContactOptionSetting activity_setting("IPC_GPU_CONTACT_OUTPUT_ACTIVITY",activity?"1":"0");
        std::unique_ptr<GpuContactPipeline> off,on;
        {ContactOptionSetting f("IPC_GPU_ORIGINAL_PT_TRANSLATION_ZERO_GATHER","0");off=std::make_unique<GpuContactPipeline>(65);}
        {ContactOptionSetting f("IPC_GPU_ORIGINAL_PT_TRANSLATION_ZERO_GATHER","1");on=std::make_unique<GpuContactPipeline>(65);}
        for(int epoch=0;epoch<8;++epoch) {
            MixedOptions option;option.capacity=65;option.original=true;option.seeded=true;
            option.point_only=true;option.repeat_contacts=epoch==0||epoch==7?131:epoch==1?3:67;
            option.d_hat=epoch==4?.001:.2;option.no_sdf=true;option.pt_zero_test_friction=0.;
            option.empty_contacts=epoch==2;option.disable_translation=epoch==3;
            option.reverse_contacts=epoch==5;option.duplicate_incidence_row=epoch==6;
            if(epoch==5){option.point_only=false;option.interleaved=true;}
            paired(input,option,off.get(),on.get());
        }
    }
    // Unsupported/frozen/Robust/friction paths retain the original gather.
    for(int kind=0;kind<4;++kind) {
        MixedOptions option;option.capacity=65;option.original=kind!=0;option.frozen=kind==1;
        option.point_only=true;option.repeat_contacts=67;option.d_hat=.2;option.no_sdf=true;
        option.pt_zero_test_friction=kind==2?.1:kind==3?std::numeric_limits<double>::quiet_NaN():0.;
        const auto result=paired(input,option);
        require(result.error==(kind==3?20:0),"unsupported PT control status premise changed");
    }
    // A late PT error follows healthy translated PT tiles. Pending systems must
    // remain unpublished, and ordinary failed packets must not enter an exact queue.
    for(bool invalid_history:{false,true})for(bool activity:{false,true}) {
        ContactOptionSetting activity_setting("IPC_GPU_CONTACT_OUTPUT_ACTIVITY",activity?"1":"0");
        auto fixture=input;fixture.positions[6][0]=fixture.previous[6][0]=.25;
        fixture.positions[6][1]=fixture.previous[6][1]=.25;
        fixture.positions[6][2]=fixture.previous[6][2]=invalid_history?.17:0.;
        if(invalid_history)fixture.previous[6][0]=std::numeric_limits<double>::quiet_NaN();
        MixedOptions option;option.capacity=65;option.original=true;option.seeded=true;option.point_only=true;
        option.repeat_contacts=131;option.d_hat=.2;option.no_sdf=true;option.pt_zero_test_friction=0.;
        option.pt_zero_test_translation_only=true;option.pt_zero_test_late_point=true;
        const auto result=paired(fixture,option);
        require(result.error==(invalid_history?20:21),"late PT status priority changed");
        for(int body=0;body<2;++body) {
            const auto seed=initial_system(body+8);
            require(std::memcmp(&result.rigid_com[body],&seed,sizeof(seed))==0,
                "late PT error partially published a rigid system");
        }
    }
    for(std::size_t bodies:{std::size_t(64),std::size_t(65)}) {
        std::unique_ptr<GpuContactPipeline> off,on;
        {ContactOptionSetting f("IPC_GPU_ORIGINAL_PT_TRANSLATION_ZERO_GATHER","0");off=std::make_unique<GpuContactPipeline>(65);}
        {ContactOptionSetting f("IPC_GPU_ORIGINAL_PT_TRANSLATION_ZERO_GATHER","1");on=std::make_unique<GpuContactPipeline>(65);}
        // Constructors capture the flag. A later invalid value must not be
        // reparsed by either existing owner while assembling the same scene.
        ContactOptionSetting malformed("IPC_GPU_ORIGINAL_PT_TRANSLATION_ZERO_GATHER","invalid");
        const auto expected=compact_many_run(*off,bodies,65,false,false,true);
        const auto actual=compact_many_run(*on,bodies,65,false,false,true);
        compare_many_compact(actual,expected);
        require(actual.storage.packet_bytes==expected.storage.packet_bytes,
            "64/65 thin gather changed compact-fallback allocation bounds");
        ++comparisons;
    }
    require(active_witnesses>0,"PT translation force fixture is vacuous");
    std::cout<<"Original PT translation functional pipeline PASS: comparisons="<<comparisons
        <<" active_owner_witnesses="<<active_witnesses<<'\n';
}
}
void gpu_original_pt_translation_test(const ipc_gpu::solver::ContactInput* inputs,std::size_t count) {
    std::vector<ContactInput> selected;
    for(std::size_t i=0;i<count;++i)if(inputs[i].kind==ContactKind::PointTriangle
        &&inputs[i].hessian_blocks==1u&&inputs[i].friction==0.0)selected.push_back(inputs[i]);
    require(!selected.empty(),"PT native publication input matrix is empty");
    Memory<ContactInput> device_inputs(selected.size(),selected.data());
    Memory<ContactOutput> expected(2*selected.size()),actual(2*selected.size());
    check(cudaMemset(expected.data,0xa5,2*selected.size()*sizeof(ContactOutput)));
    check(cudaMemset(actual.data,0xa5,2*selected.size()*sizeof(ContactOutput)));
    Memory<unsigned> failures(selected.size());
    pt_publication_differential<<<(selected.size()+127)/128,128>>>(device_inputs.data,selected.size(),
        expected.data,actual.data,failures.data);check(cudaGetLastError());
    std::vector<unsigned> host_failures(selected.size());failures.download(host_failures.data());
    for(auto failure:host_failures)require(failure==0,"PT native full packet publication changed bytes");
    pt_public_pipeline();
    std::cout<<"Original PT translation device publication PASS: inputs="<<selected.size()<<'\n';
}
