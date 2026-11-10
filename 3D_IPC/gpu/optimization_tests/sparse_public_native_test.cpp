// Public Solver default/OFF/ON numerical and bounded-allocation regression.
// The included allocator wrappers serve assertions, not performance tracing.
#define main queue_fixture_main_not_executed
#include "two_rigid_color_queue_native_test.cpp"
#undef main
namespace {
Run sparse_public_run(const char* flag,bool contact=true) {
    Environment sparse("IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER"),cap("IPC_GPU_CONTACT_PACKET_CAPACITY");
    sparse.set(flag);cap.set("3");const auto h=fixture();const auto before=resident_bytes();Run result;
    {
        GpuSolver solver(h);
        for(double activation:{.2,.11,.2}) {
            auto p=parameters(false);p.contact.d_hat=activation;p.contact.friction=0.;
            if(!contact)p.contact.k_barrier=0.;
            solver.advance_substep(p,h.pin_targets,{});auto state=solver.snapshot();finite(state);
            result.states.push_back(std::move(state));result.resident.push_back(resident_bytes()-before);
        }
    }
    require(resident_bytes()==before&&!tracking_failed,"sparse public owner retained allocation");return result;
}
std::string sparse_public_error(const char* flag) {
    Environment sparse("IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER"),cap("IPC_GPU_CONTACT_PACKET_CAPACITY");
    sparse.set(flag);cap.set("3");const auto h=fixture(0.);auto p=parameters(false);p.contact.friction=0.;
    const auto before=resident_bytes();std::string message;
    {
        GpuSolver solver(h);finite(solver.snapshot());
        try{solver.advance_substep(p,h.pin_targets,{});}catch(const std::runtime_error& e){message=e.what();}
        require(message.find("device status 21;")!=std::string::npos,"sparse public zero gap lost code21");
        bool snapshot_rejected=false,retry_rejected=false;
        try{(void)solver.snapshot();}catch(const std::runtime_error&){snapshot_rejected=true;}
        try{solver.advance_substep(p,h.pin_targets,{});}catch(const std::runtime_error&){retry_rejected=true;}
        require(snapshot_rejected&&retry_rejected,"sparse force failure left owner usable");
    }
    require(resident_bytes()==before&&!tracking_failed,"sparse public error owner retained allocation");return message;
}
}
int main(){try {
    tracking=true;
    // Other selected options retain TRUE final defaults. This test requires
    // selected F6/dense prerequisites; no global CTest ENV masks them.
    const auto off=sparse_public_run("0"),on=sparse_public_run("1"),unset=sparse_public_run(nullptr);
    for(std::size_t epoch=0;epoch<off.states.size();++epoch) {
        equal(off.states[epoch],on.states[epoch],"sparse public OFF/ON six-array bytes");
        equal(on.states[epoch],unset.states[epoch],"sparse public UNSET/ON six-array bytes");
        require(on.resident[epoch]==unset.resident[epoch],"unset sparse did not select defaultON storage");
        require(on.resident[epoch]==off.resident[epoch]+3*sizeof(unsigned)+sizeof(unsigned),
            "real public sparse route/highwater allocation witness failed");
    }
    const auto no_force=sparse_public_run(nullptr,false);
    require(!equal_bytes(unset.states[0].com,no_force.states[0].com)
        ||!equal_bytes(unset.states[0].omega,no_force.states[0].omega),"sparse public force fixture is vacuous");
    require(sparse_public_error("0")==sparse_public_error("1"),"sparse public error/invalidation changed");
    {Environment feature("IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER");feature.set("invalid");bool rejected=false;
        try{GpuSolver invalid(fixture());}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"invalid sparse option accepted by public constructor");}
    require(resident_bytes()==0&&!tracking_failed,"sparse public allocation ledger not empty");tracking=false;
    std::cout<<"Sparse public Solver: UNSET=ON/OFF exact six-array states, active-force, allocation/reuse, error/invalidation PASS\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
