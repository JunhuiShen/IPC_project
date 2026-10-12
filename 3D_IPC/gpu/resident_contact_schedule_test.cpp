#include "resident_contact_schedule.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using ipc_gpu::resident::ExactContactSchedule;

static constexpr std::array<const char*,4> names{
    "IPC_GPU_CONTACT_EXACT_SMALL_WORKERS","IPC_GPU_CONTACT_EXACT_SMALL_THREADS",
    "IPC_GPU_CONTACT_EXACT_LARGE_WORKERS","IPC_GPU_CONTACT_EXACT_LARGE_THREADS"};

struct EnvironmentGuard {
    std::array<std::string,4> values;
    std::array<bool,4> present{};
    EnvironmentGuard() {
        for(std::size_t i=0;i<names.size();++i) {
            const char* value=std::getenv(names[i]);
            present[i]=value!=nullptr;if(value)values[i]=value;
            unsetenv(names[i]);
        }
    }
    ~EnvironmentGuard() {
        for(std::size_t i=0;i<names.size();++i)
            if(present[i])setenv(names[i],values[i].c_str(),1);else unsetenv(names[i]);
    }
};

static void require(bool okay,const char* message) {
    if(!okay)throw std::runtime_error(message);
}
static void set(const char* name,const char* value) {
    if(setenv(name,value,1))throw std::runtime_error("setenv failed");
}
static void rejected(const char* name,const char* value) {
    set(name,value);bool threw=false;
    try {ExactContactSchedule::from_environment();}
    catch(const std::invalid_argument&){threw=true;}
    unsetenv(name);require(threw,"invalid exact contact schedule accepted");
}
static void check_arena(const ExactContactSchedule& schedule,std::size_t candidates) {
    const auto arena=schedule.arena_sizes(candidates,256,1600,512);
    for(const auto pass:{std::array<int,3>{schedule.small_workers,schedule.small_threads,256},
                         std::array<int,3>{schedule.large_workers,schedule.large_threads,1600}}) {
        require(pass[0]%pass[1]==0,"launch is not exactly divisible");
        const auto active=std::min(candidates,std::size_t(pass[0]));
        require(arena.words>=active*pass[2]*512,"exact words underallocated");
        require(arena.free_slots>=active*512,"exact free slots underallocated");
    }
}

int main() {
    try {
        EnvironmentGuard environment;
        const auto defaults=ExactContactSchedule::from_environment();
        require(defaults.small_workers==1024&&defaults.small_threads==1&&
                defaults.large_workers==256&&defaults.large_threads==4,"default schedule changed");
        for(const auto value:{"","0","-1","+1"," 8","8 ","8x","1.0","4097","999999999999999999999"})
            rejected(names[0],value);
        rejected(names[2],"1025");
        for(const auto value:{"0","3","5","31","33","64"}) {
            rejected(names[1],value);rejected(names[3],value);
        }
        set(names[1],"8");
        rejected(names[0],"9"); // not divisible by explicitly selected eight threads
        unsetenv(names[1]);
        rejected(names[2],"5"); // not divisible by default four threads
        for(const auto threads:{1,2,4,8,16,32}) {
            const auto text=std::to_string(threads);
            set(names[1],text.c_str());set(names[3],text.c_str());
            const auto schedule=ExactContactSchedule::from_environment();
            require(schedule.small_threads==threads&&schedule.large_threads==threads,"supported threads rejected");
        }
        for(const auto& schedule:{defaults,ExactContactSchedule{4096,32,1,1},
                                 ExactContactSchedule{1,1,1024,32},ExactContactSchedule{32,1,32,1}}) {
            schedule.validate();
            for(const std::size_t count:{0,1,2,7,31,1031,100000})check_arena(schedule,count);
        }
        const auto reversed=ExactContactSchedule{1,1,1024,32}.arena_sizes(2000,256,1600,512);
        require(reversed.free_slots==std::size_t(1024)*512,"large retry pool does not own enough free slots");
        set(names[0],"4096");set(names[2],"1024");
        const auto maximum=ExactContactSchedule::from_environment();
        require(maximum.small_workers==4096&&maximum.large_workers==1024,"maximum worker limits rejected");
        // An already-created pipeline's configuration is immutable even if a
        // caller later changes process environment variables.
        set(names[0],"32");
        require(maximum.small_workers==4096&&ExactContactSchedule::from_environment().small_workers==32,
                "schedule is not a once-created value");
        std::cout<<"exact contact schedule parsing, limits, launch divisibility and arena sizing passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
