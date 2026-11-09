#define main queue_qualified_original_main
#include "two_rigid_color_queue_native_test.cpp"
#undef main

namespace {
Run one_slot_run(const char* slot_flag,bool exact,bool dat=false,bool dense=true,
                 bool contact=true) {
    Environment one("IPC_GPU_RIGID_ONE_SLOT_TERMS"),csr("IPC_GPU_TWO_RIGID_DENSE_CSR");
    one.set(slot_flag);csr.set(dense?"1":"0");
    return run("1",nullptr,exact,dat,contact);
}
void slot_public_checks() {
    // Known source arrays/capacity from the independent qualified public fixture.
    // Reserve transitions are intentionally tested on the same owner at N4→1→4.
    Run off,on,unset;
    {
        Environment kind("IPC_GPU_TWO_RIGID_KIND_PACKETS"),sparse("IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER");
        kind.set("0");sparse.set("0");
        off=one_slot_run("0",false);on=one_slot_run("1",false);unset=one_slot_run(nullptr,false);
    }
    const std::size_t expected[3]={1720,1528,1720}; // 3*192*cap minus8B color proof.
    for(std::size_t i=0;i<3;++i) {
        equal(off.states[i],on.states[i],"one-slot Original six-array bytes");
        equal(on.states[i],unset.states[i],"one-slot defaultON six-array bytes");
        require(off.resident[i]==on.resident[i]+expected[i],"one-slot exact allocation/real-route witness failed");
        require(on.resident[i]==unset.resident[i],"unset one-slot did not select defaultON resources");
    }
    const auto true_on=one_slot_run("1",false),true_unset=one_slot_run(nullptr,false);
    for(std::size_t i=0;i<true_on.states.size();++i) {
        equal(true_on.states[i],true_unset.states[i],"one-slot true-default UNSET/ON six-array bytes");
        require(true_on.resident[i]==true_unset.resident[i],"one-slot true-default UNSET/ON resources");
    }
    for(bool exact:{false,true})for(bool dat:{false,true})for(bool dense:{false,true}) {
        const auto a=one_slot_run("0",exact,dat,dense),b=one_slot_run("1",exact,dat,dense);
        for(std::size_t epoch=0;epoch<a.states.size();++epoch)
            equal(a.states[epoch],b.states[epoch],"one-slot policy/frozen/generic fallback bytes");
    }
    const auto no_force=one_slot_run("0",false,false,true,false);
    require(!equal_bytes(off.states[0].com,no_force.states[0].com)
        ||!equal_bytes(off.states[0].omega,no_force.states[0].omega),"one-slot active-force fixture is vacuous");
    for(bool exact:{false,true}) {
        std::string a,b;
        {Environment one("IPC_GPU_RIGID_ONE_SLOT_TERMS"),csr("IPC_GPU_TWO_RIGID_DENSE_CSR");one.set("0");csr.set("1");a=force_error("1",exact);}
        {Environment one("IPC_GPU_RIGID_ONE_SLOT_TERMS"),csr("IPC_GPU_TWO_RIGID_DENSE_CSR");one.set("1");csr.set("1");b=force_error("1",exact);}
        require(a==b,"one-slot changed exact failure/invalidation contract");
    }
    {Environment one("IPC_GPU_RIGID_ONE_SLOT_TERMS");one.set("1");raw_pipeline_gates();}
    {Environment one("IPC_GPU_RIGID_ONE_SLOT_TERMS");one.set("invalid");bool rejected=false;
        try{GpuSolver bad(fixture());}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"fresh invalid one-slot option accepted");}
}
}
int main(){try{
    tracking=true;slot_public_checks();
    require(resident_bytes()==0&&!tracking_failed,"one-slot ledger leaked/missed allocation");tracking=false;
    std::cout<<"One-slot public Solver functional/memory witness PASS; native qualification only, no speed claim\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"one-slot public test: "<<e.what()<<'\n';return 1;}}
