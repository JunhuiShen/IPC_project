#define main queue_elision_reference_main
#include "two_rigid_color_queue_native_test.cpp"
#undef main
#include "two_rigid_dense_csr_policy.h"
namespace {
SceneDescription graded_fixture() {
    auto h=fixture();const double heights[4]={.1,.12,.14,.16};
    h.com[0]=Vector3{};h.body_second_moment[0]=Matrix3{};
    for(int node=0;node<4;++node) {
        h.positions[node][2]=heights[node];h.pin_targets[node]=h.positions[node];
        for(int axis=0;axis<3;++axis)h.com[0][axis]+=.25*h.positions[node][axis];
    }
    for(int node=0;node<4;++node) {
        auto& local=h.body_reference[node];
        for(int axis=0;axis<3;++axis)local[axis]=h.positions[node][axis]-h.com[0][axis];
        for(int row=0;row<3;++row)for(int col=0;col<3;++col)
            h.body_second_moment[0][3*row+col]+=local[row]*local[col];
        const auto& v=h.com_velocity[0];const auto& w=h.omega[0];
        h.velocities[node]=Vector3{v[0]+w[1]*local[2]-w[2]*local[1],
            v[1]+w[2]*local[0]-w[0]*local[2],v[2]+w[0]*local[1]-w[1]*local[0]};
    }
    return h;
}
Run dense_run(const char* dense_flag,const char* queue_flag,const char* forced_pairs,
    const std::vector<double>& dhats,const std::vector<bool>& modes,int capacity,
    bool pruning=false,bool graded=false) {
    Environment dense_env("IPC_GPU_TWO_RIGID_DENSE_CSR"),queue_env("IPC_GPU_TWO_RIGID_COLOR_QUEUE_ELISION"),
        pairs_env("IPC_GPU_COLOR_PAIR_QUEUES"),cap_env("IPC_GPU_CONTACT_PACKET_CAPACITY"),prune_env("IPC_GPU_BOX_CONTACT_PRUNING");
    dense_env.set(dense_flag);queue_env.set(queue_flag);pairs_env.set(forced_pairs);
    const auto capacity_text=std::to_string(capacity);cap_env.set(capacity_text.c_str());prune_env.set(pruning?"1":"0");
    require(dhats.size()==modes.size(),"invalid dense lifecycle protocol");
    const auto h=graded?graded_fixture():fixture();const auto before=resident_bytes();Run result;
    {
        GpuSolver solver(h);
        for(std::size_t epoch=0;epoch<dhats.size();++epoch) {
            auto p=parameters(modes[epoch]);p.contact.d_hat=dhats[epoch];
            if(graded){p.box_radius_min=p.box_radius=1e-5;p.rotation_radius=1e-5;p.damping=.001;}
            solver.advance_substep(p,h.pin_targets,{});auto snapshot=solver.snapshot();finite(snapshot);
            result.states.push_back(std::move(snapshot));result.resident.push_back(resident_bytes()-before);
        }
    }
    require(resident_bytes()==before&&!tracking_failed,"dense owner retained allocation");return result;
}
void compare_runs(const Run& a,const Run& b,const std::string& label) {
    require(a.states.size()==b.states.size(),label+" state count");
    for(std::size_t epoch=0;epoch<a.states.size();++epoch)
        equal(a.states[epoch],b.states[epoch],label+" epoch="+std::to_string(epoch));
}
void dense_lifetimes() {
    Environment kind_accounting("IPC_GPU_TWO_RIGID_KIND_PACKETS"),slot_accounting("IPC_GPU_RIGID_ONE_SLOT_TERMS"),
        sparse_accounting("IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER");
    kind_accounting.set("0");slot_accounting.set("0");sparse_accounting.set("0");
    // N4 at first/third epoch; generic middleN1. The preflight word remains4B.
    // Dense32B CSR vs generic64B gives an exact28B whole-owner difference.
    for(bool exact:{false,true})for(const char* queue_flag:{"0","1"})
        for(const char* pair_flag:{static_cast<const char*>(nullptr),"1","0"}) {
            const std::vector<double> dhats{.2,.11,.2};const std::vector<bool> modes(3,exact);
            const auto off=dense_run("0",queue_flag,pair_flag,dhats,modes,3);
            const auto unset=dense_run(nullptr,queue_flag,pair_flag,dhats,modes,3);
            const auto on=dense_run("1",queue_flag,pair_flag,dhats,modes,3);
            compare_runs(on,unset,"dense defaultON");compare_runs(off,on,"dense independent color policy");
            for(std::size_t epoch=0;epoch<dhats.size();++epoch) {
                require(on.resident[epoch]==unset.resident[epoch],"unset dense did not select defaultON storage");
                require(off.resident[epoch]==on.resident[epoch]+28,
                    "dense CSR28B retained/reentry witness failed epoch="+std::to_string(epoch));
            }
        }
    // Actual proxy heights and tiny bounds separate these counts by at least
    // .009, well above two damped updates: N4→3→2 remain dense atcap1, then
    // genericN1→emptyN0→denseN4. Constant28B proves dense high-water reuse,
    // and final28B also requires releasing the retained small Incident array.
    for(bool exact:{false,true}) {
        const std::vector<double> dhats{.2,.15,.13,.11,.05,.2};
        const std::vector<bool> modes(dhats.size(),exact);
        const auto off=dense_run("0","1",nullptr,dhats,modes,1,false,true);
        const auto on=dense_run("1","1",nullptr,dhats,modes,1,false,true);
        compare_runs(off,on,"dense highwater/tails/empty");
        for(std::size_t epoch=0;epoch<dhats.size();++epoch)
            require(off.resident[epoch]==on.resident[epoch]+28,
                "dense decreasing-N/highwater or stale Incident storage epoch="+std::to_string(epoch));
    }
    // Under BOX_CONTACT_PRUNING1: Original has no pruning cache for this
    // pure-rigid scene, Robust builds one (allflags0). Dense must still choose
    // the complete generic4N route, then shrink once on reentry to Original.
    const std::vector<double> dhats{.2,.2,.2};const std::vector<bool> modes{false,true,false};
    const auto off=dense_run("0","1",nullptr,dhats,modes,3,true);
    const auto on=dense_run("1","1",nullptr,dhats,modes,3,true);
    compare_runs(off,on,"no-pruning restriction/reentry");
    require(off.resident[0]==on.resident[0]+28&&off.resident[2]==on.resident[2]+28,
        "generic4N→dense2N transition failed");
    require(on.resident[1]==off.resident[1]+sizeof(unsigned),
        "pruned mode failed complete generic fallback or lost retained preflight");
}
void dense_raw_error_gates() {
    Environment dense_env("IPC_GPU_TWO_RIGID_DENSE_CSR");
    dense_env.set("1");raw_pipeline_gates();
    {
        // Public raw callers never receive the hint. In particular a valid
        // EE/PT/EE/PT input must keep the generic(kind,pair) sorted rows.
        const auto h=fixture();DeviceArray<int> blocks(h.node_blocks);
        DeviceArray<BpContact> contacts(std::vector<BpContact>{
            BpContact{{0,1,4,5},1},BpContact{{0,4,5,6},0},
            BpContact{{2,3,5,6},1},BpContact{{1,4,5,6},0}});
        DeviceArray<int> colors(std::vector<int>{0,1}),ordered(std::vector<int>{0,1}),
            color_offsets(std::vector<int>{0,1,2});
        BroadPhaseView view;view.contacts=contacts.data();view.contact_count=4;
        view.block_colors=colors.data();view.color_count=2;
        view.color_blocks=ordered.data();view.color_offsets=color_offsets.data();
        GpuContactPipeline raw(3);raw.rebuild(view,2,blocks.data());
        const auto csr=raw.incidence();std::size_t offsets[3]{};int columns[8]{};
        cuda_checked(cudaMemcpy(offsets,csr.block_offsets,sizeof(offsets),cudaMemcpyDeviceToHost));
        cuda_checked(cudaMemcpy(columns,csr.pair_indices,sizeof(columns),cudaMemcpyDeviceToHost));
        const std::size_t expected_offsets[3]={0,4,8};const int expected_columns[8]={1,3,0,2,1,3,0,2};
        require(std::memcmp(offsets,expected_offsets,sizeof(offsets))==0
            &&std::memcmp(columns,expected_columns,sizeof(columns))==0,
            "raw nonmonotone contacts bypassed complete generic sorting");
    }
    for(bool exact:{false,true}) {
        dense_env.set("0");const auto error_off=force_error("1",exact);
        dense_env.set("1");const auto error_on=force_error("1",exact);
        require(error_off==error_on,"dense single zero-gap code/invalidation changed");
    }
}
}
int main() {
    try {
        Environment dense_env("IPC_GPU_TWO_RIGID_DENSE_CSR");dense_env.set("0");
        require(queue_elision_reference_main()==0,"all sealed queue-elision assertions must still pass");
        tracking=true;dense_lifetimes();dense_raw_error_gates();
        require(resident_bytes()==0&&!tracking_failed,"dense allocation ledger not empty");tracking=false;
        std::cout<<"Two-rigid dense CSR: independent queue flags, six-array bytes, dense highwater/generic/pruned/empty reuse, raw/errors PASS\n";return 0;
    }catch(const std::exception& e){std::cerr<<"two-rigid dense test: "<<e.what()<<'\n';return 1;}
}
