// Native-ready acceptance only. No CUDA execution is claimed by this source.
// Include the exact current117-fixture suite without modifying its assertions.
#include "linear_pipeline_launch.h"
#define main qualified_node_csr_native_main
#include "node_csr_partition_native_test.cpp"
#undef main

struct CooperativeEnvironment {
    bool existed=false;std::string value;
    CooperativeEnvironment() {
        if(const char* old=std::getenv("IPC_GPU_NODE_CCD_COOPERATIVE")) {
            existed=true;value=old;
        }
    }
    ~CooperativeEnvironment() {
        if(existed)setenv("IPC_GPU_NODE_CCD_COOPERATIVE",value.c_str(),1);
        else unsetenv("IPC_GPU_NODE_CCD_COOPERATIVE");
    }
};

void controller_capture_and_coverage(ipc_gpu::solver::node_controller::Plan plan) {
    Environment inherited;
    setenv("IPC_GPU_ORIGINAL_CCD_WORKERS","16",1);
    feature_flags("1");
    require(setenv("IPC_GPU_NODE_CCD_COOPERATIVE","0",1)==0,"cannot capture fixed controller");
    gpu::GpuLinearClipper fixed;
    require(setenv("IPC_GPU_NODE_CCD_COOPERATIVE","1",1)==0,"cannot capture cooperative controller");
    gpu::GpuLinearClipper cooperative;
    auto collision=long_collision_row(4097);
    const auto reference=run(collision,false,fixed);
    require(reference.error==0,"controller capture reference failed");
    for(const char* runtime:{"0","1","invalid","2","-1"}) {
        require(setenv("IPC_GPU_NODE_CCD_COOPERATIVE",runtime,1)==0,"cannot alter runtime controller environment");
        const auto scalar=run(collision,true,fixed),current=run(collision,true,cooperative);
        require(scalar.error==reference.error&&bytes(scalar.accepted,reference.accepted),
            "captured fixed controller changed after environment mutation");
        require(current.error==reference.error&&bytes(current.accepted,reference.accepted),
            "captured cooperative controller changed after environment mutation");
    }
    for(const char* malformed:{"invalid","2","-1"}) {
        require(setenv("IPC_GPU_NODE_CCD_COOPERATIVE",malformed,1)==0,"cannot set malformed controller option");
        bool rejected=false;
        try {gpu::GpuLinearClipper owner;}
        catch(const std::invalid_argument&) {rejected=true;}
        require(rejected,"malformed controller option did not reject fresh owner construction");
    }
    require(setenv("IPC_GPU_NODE_CCD_COOPERATIVE","1",1)==0,"cannot restore valid controller option");
    require(plan.eligible(),"coverage fixture requires an eligible combined kernel");
    const auto count=std::size_t(plan.blocks)*ipc_gpu::solver::node_controller::width+1;
    // This is a resource-derived native fallback case, never an unbounded
    // allocation. The tested current device is far below this defensive cap.
    require(count<=1000000,"native node-coverage fixture exceeds its declared memory bound");
    const auto outside=ipc_gpu::solver::linear_detail::query_node_cooperative_plan(count,16);
    require(outside.reason==ipc_gpu::solver::node_controller::Ineligible::NodesNotCovered,
        "actual combined-kernel node coverage did not reject one excess node");
    require(plan.blocks<static_cast<unsigned>(std::numeric_limits<int>::max()),
        "native worker-coverage fixture cannot represent resident+1");
    const auto extra_worker=static_cast<int>(plan.blocks)+1;
    const auto workers_outside=ipc_gpu::solver::linear_detail::query_node_cooperative_plan(4,extra_worker);
    require(workers_outside.reason==ipc_gpu::solver::node_controller::Ineligible::WorkerCount,
        "actual combined-kernel worker coverage did not reject one excess worker");
    if(extra_worker<=256) {
        const auto workers=std::to_string(extra_worker);
        require(setenv("IPC_GPU_ORIGINAL_CCD_WORKERS",workers.c_str(),1)==0,"cannot set uncovered worker count");
        require(setenv("IPC_GPU_NODE_CCD_COOPERATIVE","1",1)==0,"cannot capture worker fallback controller");
        gpu::GpuLinearClipper workers_fallback;
        compare(collision,workers_fallback,nullptr,"actual resident worker+1 fallback");
    } else {
        // The public worker limit is capped at256. Do not change it or allocate
        // an artificial oversized arena to manufacture this ineligible path.
        std::cout<<"Worker+1 metadata guard PASS; public cap256 prevents executing that fallback on this device\n";
    }
    auto larger=point("actual resident coverage+1 falls back with a real collision",1.25,-1.);
    larger.positions.resize(count,V{20,20,20});larger.targets.resize(count,V{20,20,20});
    larger.deformable.resize(count,0);larger.colors.resize(count,1);larger.rows.resize(count);
    larger.boxes.resize(count,gpu::BpAabb{{-100,-100,-100},{100,100,100}});
    compare(larger,cooperative,nullptr,"cooperative node-coverage fallback");
    compare(collision,cooperative,nullptr,"eligible reuse after coverage fallback");
}

int main() {
    try {
        CooperativeEnvironment restore;
        require(setenv("IPC_GPU_NODE_CCD_COOPERATIVE","0",1)==0,"cannot disable cooperative execution");
        const int old=qualified_node_csr_native_main();if(old)return old;
        // Current retained CSR fixtures reach2053 nodes. Eligibility at2053
        // and16workers proves the regular ON matrix dispatches the new kernel.
        // Worker256 cases can independently fall back if residency is smaller.
        const auto plan=ipc_gpu::solver::linear_detail::query_node_cooperative_plan(2053,16);
        std::cout<<"Native cooperative residency: blocks="<<plan.blocks
            <<" reason="<<static_cast<int>(plan.reason)<<'\n';
        require(setenv("IPC_GPU_NODE_CCD_COOPERATIVE","1",1)==0,"cannot enable cooperative execution");
        const int current=qualified_node_csr_native_main();if(current)return current;
        if(!plan.eligible()) {
            std::cout<<"Fallback equivalence PASS; cooperative kernel NOT QUALIFIED: preflight ineligible\n";
            return 77; // A runner must report SKIP, never native qualification.
        }
        controller_capture_and_coverage(plan);
        std::cout<<"Cooperative native acceptance PASS: exact retained117-fixture suite OFF/ON,"
            <<" CSR0/1, workers1/16/256, capture/malformed flags, actual coverage+1 fallback/reuse\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"Cooperative native acceptance failed: "<<error.what()<<'\n';return 1;
    }
}
