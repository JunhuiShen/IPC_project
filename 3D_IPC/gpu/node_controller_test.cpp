#include "node_cooperative_controller.h"
#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace controller=ipc_gpu::solver::node_controller;
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
enum class Initial {Empty,Direct,Health,Raw};
enum class Fault {None,Prior,PrepareFallback,RawOverflow,RawInternal,RawError,
    TrialOverflow,TrialInternal,TrialError,HealthError};
struct Configuration {Initial initial=Initial::Raw;Fault fault=Fault::None;int unsafe_trials=0,fault_attempt=0;};
struct Control {int mode=1,done=0;};
struct Model {
    Configuration config;Control control;
    int error=0,pending=0,phase=0,attempts=0,last_attempt=-1,step_halvings=0;
    int accepted=-91,raw_jobs=0,trial_jobs=0,exact_jobs=0;
    bool boxed=true,overflow=false,internal=false,bad=false,forced_zero=false;
    std::vector<std::string> trace;
    explicit Model(Configuration c):config(c){}
    bool running() const{return control.mode==1&&!control.done;}
    void begin(){trace.push_back("begin");if(config.fault==Fault::Prior){error=77;control.done=1;}}
    void prepare(){
        if(control.done)return;
        trace.push_back("prepare");
        pending=config.initial==Initial::Empty?0:1;
        phase=config.initial==Initial::Raw?0:config.initial==Initial::Health?1:2;
        if(config.fault==Fault::PrepareFallback)control.mode=0;
        if(phase==0&&pending)raw_jobs=7;
    }
    void fast(){
        if(!running())return;
        trace.push_back("fast");
        if(phase==0&&config.fault==Fault::RawOverflow)overflow=true;
        if(phase==0&&config.fault==Fault::RawError)bad=true;
        if(phase==1&&config.fault==Fault::HealthError)bad=true;
        if(phase==3&&last_attempt==config.fault_attempt&&config.fault==Fault::TrialOverflow)overflow=true;
        if(phase==3&&last_attempt==config.fault_attempt&&config.fault==Fault::TrialError)bad=true;
    }
    void exact(){
        if(!running()||overflow)return;
        trace.push_back("exact");++exact_jobs;
        if(phase==0&&config.fault==Fault::RawInternal)internal=true;
        if(phase==3&&last_attempt==config.fault_attempt&&config.fault==Fault::TrialInternal)internal=true;
    }
    void gate(){
        if(!running())return;
        trace.push_back("gate");
        if(internal){error=33;control.done=1;return;}
        if(overflow){control.mode=0;return;}
        if(!pending)control.done=1;
    }
    void consume(int attempt){
        if(!running())return;
        trace.push_back("consume"+std::to_string(attempt));
        if(bad){error=phase==1?32:31;control.done=1;return;}
        if(phase==0){phase=3;return;}
        if(phase==1||phase==2){accepted=17;pending=0;control.done=1;return;}
        if(phase==3){
            if(boxed){accepted=forced_zero?0:100+attempt;pending=0;control.done=1;return;}
            if(attempt==64){error=34;control.done=1;return;}
            ++step_halvings;
        }
    }
    void prepare_trial(int attempt){
        if(!running())return;
        // Deliberately matches the checked flag footprint: this stage may NOT
        // write mode/done before the next grid join/other loop entrants.
        trace.push_back("trial"+std::to_string(attempt));++attempts;last_attempt=attempt;
        overflow=internal=bad=false;forced_zero=attempt==64;
        boxed=attempt>=config.unsafe_trials;if(boxed)trial_jobs+=7;
    }
    void tail(){
        if(control.mode==0){trace.push_back("restore");accepted=-91;
            trace.push_back("full_legacy_fallback");accepted=19;}
    }
    auto image() const {
        return std::make_tuple(control.mode,control.done,error,pending,phase,attempts,last_attempt,
            step_halvings,accepted,raw_jobs,trial_jobs,exact_jobs,boxed,overflow,internal,bad,forced_zero,trace);
    }
};

class Barrier {
    std::mutex mutex_;std::condition_variable changed_;unsigned arrived_=0,generation_=0;
    const unsigned participants_;
public:
    explicit Barrier(unsigned n):participants_(n){}
    void wait(){
        std::unique_lock<std::mutex> lock(mutex_);const auto generation=generation_;
        if(++arrived_==participants_){arrived_=0;++generation_;changed_.notify_all();return;}
        changed_.wait(lock,[&]{return generation_!=generation;});
    }
};
struct Adapter {
    Model& model;unsigned rank=0,joins=0;Barrier* barrier=nullptr;
    void begin(){if(!rank)model.begin();}
    void prepare(){if(!rank)model.prepare();}
    void fast(){if(!rank)model.fast();}
    void exact(){if(!rank)model.exact();}
    void gate(){if(!rank)model.gate();}
    void consume(int attempt){if(!rank)model.consume(attempt);}
    void prepare_trial(int attempt){if(!rank)model.prepare_trial(attempt);}
    void join(){++joins;if(barrier)barrier->wait();}
};
// Independent fixed host-launch oracle: all65 scheduled trials remain, and
// their stage guards decide whether they do work. This is not execute().
template<class Stage> void fixed_schedule(Stage& stage){
    stage.begin();stage.join();stage.prepare();stage.join();
    stage.fast();stage.join();stage.exact();stage.join();stage.gate();stage.join();
    stage.consume(-1);stage.join();
    for(int attempt=0;attempt<=64;++attempt){
        stage.prepare_trial(attempt);stage.join();stage.fast();stage.join();
        stage.exact();stage.join();stage.gate();stage.join();stage.consume(attempt);stage.join();
    }
}
void compare(Configuration config,bool threaded){
    Model original(config),candidate(config);Adapter fixed{original},single{candidate};
    fixed_schedule(fixed);original.tail();
    require(fixed.joins==331,"fixed scheduled-stage oracle changed");
    if(!threaded){controller::execute(single,&candidate.control);candidate.tail();
        require(single.joins==6+5*unsigned(candidate.attempts),"conditional barrier count differs from completed trial calls");}
    else {
        // Nonleader methods return before their work, yet EVERY participant
        // must still reach every join. Real grid/device memory is NOT modeled.
        constexpr unsigned threads=8;Barrier barrier(threads);
        std::vector<unsigned> joins(threads,0);std::vector<std::thread> group;
        for(unsigned rank=0;rank<threads;++rank)group.emplace_back([&,rank]{
            Adapter stage{candidate,rank,0,&barrier};controller::execute(stage,&candidate.control);joins[rank]=stage.joins;});
        for(auto& thread:group)thread.join();
        candidate.tail();
        require(std::all_of(joins.begin(),joins.end(),[&](unsigned n){return n==6+5*unsigned(candidate.attempts);}),
            "nonleader early return lost a barrier participant");
    }
    require(original.image()==candidate.image(),"conditional controller changed stage/counter/error/budget/restore semantics");
    require(candidate.attempts<=65,"conditional controller exceeded65trial budget");
    if(config.initial==Initial::Raw&&config.fault==Fault::None&&config.unsafe_trials>=65)
        require(candidate.attempts==65&&candidate.last_attempt==64&&candidate.forced_zero&&candidate.error==34,
            "all64+finalzero trial/unresolved contract differs");
}
void plans(){
    using I=controller::Ineligible;
    require(controller::make_plan(11776,16,true,46,2).eligible(),"covered launch rejected");
    require(controller::make_plan(1,1,true,1,1).blocks==1,"minimal resident grid differs");
    require(controller::make_plan(128,1,true,1,1).eligible(),"exact node coverage rejected");
    require(controller::make_plan(129,1,true,1,1).reason==I::NodesNotCovered,"one-node excess accepted");
    require(controller::make_plan(128,2,true,1,1).reason==I::WorkerCount,"one-worker excess accepted");
    require(controller::make_plan(1,-1,true,1,1).reason==I::WorkerCount,"negative worker count accepted");
    require(controller::make_plan(std::numeric_limits<std::size_t>::max(),1,true,1,1).reason==I::NodesNotCovered,
        "maximum scene count wrapped node coverage");
    for(int sms:{1,2,46,142,256})for(int occupancy:{1,2,4,8}) {
        const auto blocks=static_cast<unsigned>(sms*occupancy);
        const auto nodes=std::size_t(blocks)*controller::width;
        const auto exact=controller::make_plan(nodes,static_cast<int>(blocks),true,sms,occupancy);
        require(exact.eligible()&&exact.blocks==blocks,"resident node/worker boundary rejected");
        require(controller::make_plan(nodes+1,static_cast<int>(blocks),true,sms,occupancy).reason==I::NodesNotCovered,
            "resident node boundary overflow accepted");
        require(controller::make_plan(nodes,static_cast<int>(blocks)+1,true,sms,occupancy).reason==I::WorkerCount,
            "resident worker boundary overflow accepted");
    }
    require(controller::make_plan(11777,16,true,46,2).reason==I::NodesNotCovered,"coverage guard missing");
    require(controller::make_plan(1,93,true,46,2).reason==I::WorkerCount,"workerpool block mapping oversubscribed");
    require(controller::make_plan(1,0,true,46,2).reason==I::WorkerCount,"zero workers accepted");
    require(controller::make_plan(1,1,false,46,2).reason==I::Unsupported,"unsupported cooperative launch accepted");
    require(controller::make_plan(0,1,true,46,2).reason==I::Empty,"empty scene launched");
    require(controller::make_plan(1,1,true,46,0).reason==I::NoResidency,"zero actual occupancy accepted");
    require(controller::make_plan(1,1,true,std::numeric_limits<int>::max(),std::numeric_limits<int>::max()).reason==I::Overflow,
        "unsafe resident block product accepted");
}
int main(){try{
    plans();
    for(auto initial:{Initial::Empty,Initial::Direct,Initial::Health,Initial::Raw})
        for(auto fault:{Fault::None,Fault::Prior,Fault::PrepareFallback,Fault::RawOverflow,Fault::RawInternal,
            Fault::RawError,Fault::TrialOverflow,Fault::TrialInternal,Fault::TrialError,Fault::HealthError})
            for(int unsafe:{0,1,3,63,64,65,66})
                compare({initial,fault,unsafe,std::min(unsafe,64)},false);
    std::mt19937 random(0x4862dc);
    for(int trial=0;trial<2000;++trial)compare({Initial(random()%4),Fault(random()%10),int(random()%67),int(random()%65)},false);
    for(const Configuration config:std::vector<Configuration>{{Initial::Empty},{Initial::Direct},{Initial::Health},
        {Initial::Raw},{Initial::Raw,Fault::Prior},{Initial::Raw,Fault::PrepareFallback},
        {Initial::Raw,Fault::RawOverflow},{Initial::Raw,Fault::RawInternal},
        {Initial::Raw,Fault::TrialOverflow,3,2},{Initial::Raw,Fault::TrialError,3,2},
        {Initial::Raw,Fault::None,64},{Initial::Raw,Fault::None,65}})compare(config,true);
    std::cout<<"Host shared-controller oracle PASS: fixed331stages vsconditional, allbudget/error/overflow/direct/empty cases, 2000random, 8participant early-return barriers, residency bounds; NOT CUDA qualification\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
