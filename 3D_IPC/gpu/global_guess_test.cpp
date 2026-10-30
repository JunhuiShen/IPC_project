#include "global_guess.cuh"
#include <tight_inclusion/ccd.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef TIGHT_INCLUSION_WITH_DOUBLE_PRECISION
#error "The GPU standard guess matches the CPU double-precision Tight-Inclusion build"
#endif
#ifdef TIGHT_INCLUSION_USE_MAX_ABS_TOL
#error "Port the CPU's changed absolute tolerance caps before enabling this comparison"
#endif
#ifdef IPC_GLOBAL_GUESS_CUDA_TEST
#include "global_guess.h"
#include "device_memory.h"
#include <future>
#endif

using ipc_gpu::solver::global_guess_detail::Input;
using ipc_gpu::solver::global_guess_detail::Cell;
using Point=std::array<double,3>;
static void require(bool result,const char* message){if(!result)throw std::runtime_error(message);}
static std::vector<Input> fixtures(bool include_wide_frontier) {
    const Input pt={{{.25,.25,1},{0,0,0},{1,0,0},{0,1,0}},
                    {{.25,.25,-1},{0,0,0},{1,0,0},{0,1,0}},true};
    const Input ee={{{-1,0,1},{1,0,1},{0,-1,0},{0,1,0}},
                    {{-1,0,-1},{1,0,-1},{0,-1,0},{0,1,0}},false};
    std::vector<Input> result;
    for(const Input& base:{pt,ee}) {
        result.push_back(base);
        auto q=base;
        for(int v=0;v<4;++v)for(int a=0;a<3;++a)q.end[v][a]+=double(a+1)*.37;
        result.push_back(q); // Every vertex moves; relative crossing unchanged.
        q=base;std::memcpy(q.end,q.start,sizeof(q.end));result.push_back(q); // Static separated.
        for(int v=0;v<4;++v)for(int a=0;a<3;++a)q.end[v][a]+=.25*(a+1);
        result.push_back(q); // Rigid translation, no collision.
        q=base;for(int v=0;v<(q.vertex_face?1:2);++v)q.end[v][2]=0;
        result.push_back(q); // Contact at t=1.
        q=base;for(int v=0;v<(q.vertex_face?1:2);++v)q.start[v][2]=0;
        result.push_back(q); // Initial overlap -> zero (no_zero_toi false).
        q=base;for(int v=0;v<4;++v)q.start[v][2]=q.end[v][2]=0;
        result.push_back(q); // Identically coplanar.
        q=base;for(int v=0;v<4;++v)for(int a=0;a<3;++a)q.start[v][a]=q.end[v][a]=0;
        result.push_back(q); // Degenerate coincident points.
        for(double scale:{1e-6,.01,100.,1e6}) {
            q=base;for(int v=0;v<4;++v)for(int a=0;a<3;++a){q.start[v][a]*=scale;q.end[v][a]*=scale;}
            result.push_back(q);
        }
        q=base;
        if(q.vertex_face) {
            for(int t=0;t<2;++t) {
                double (*x)[3]=t?q.end:q.start;
                x[0][0]=2;x[0][1]=2; // Cross plane outside finite triangle.
            }
        } else {
            for(int t=0;t<2;++t) {
                double (*x)[3]=t?q.end:q.start;
                x[2][0]=2;x[3][0]=2; // Cross lines outside finite edges.
            }
        }
        result.push_back(q);
        q=base;
        for(int t=0;t<2;++t) {
            double (*x)[3]=t?q.end:q.start;
            if(q.vertex_face)for(int a=0;a<3;++a)x[3][a]=x[2][a];
            else for(int a=0;a<3;++a)x[1][a]=x[0][a];
        }
        result.push_back(q); // Degenerate triangle/edge remains finite feature.
    }
    // Parallel overlapping edges become coincident together. The diagonal
    // u=v solution family creates a genuinely wide BFS frontier (>256), so
    // the real CUDA grow-and-retry path is exercised, not just host size-1.
    if(include_wide_frontier)
        result.push_back({{{-1,0,1},{1,0,1},{-1,0,0},{1,0,0}},
                          {{-1,0,-1},{1,0,-1},{-1,0,0},{1,0,0}},false});
    std::mt19937_64 generator(916872);
    std::uniform_real_distribution<double> random(-.5,.5);
    for(int i=0;i<512;++i) {
        Input q=i%2?pt:ee;
        for(int v=0;v<4;++v)for(int a=0;a<3;++a) {
            q.start[v][a]+=random(generator);
            q.end[v][a]+=random(generator);
        }
        result.push_back(q);
    }
    return result;
}
static double cpu_reference(const Input& q) {
    ticcd::Vector3 start[4],end[4];
    for(int v=0;v<4;++v)for(int a=0;a<3;++a){start[v][a]=q.start[v][a];end[v][a]=q.end[v][a];}
    double toi=INFINITY,output_tolerance=1e-6;
    const auto function=q.vertex_face?ticcd::vertexFaceCCD:ticcd::edgeEdgeCCD;
    bool hit=function(start[0],start[1],start[2],start[3],end[0],end[1],end[2],end[3],
        ticcd::Array3::Constant(-1),1e-10,toi,1e-6,1.0,1000000,output_tolerance,false,
        ticcd::CCDRootFindingMethod::BREADTH_FIRST_SEARCH);
    return !hit||!std::isfinite(toi)?1:std::min(1.0,std::max(0.0,toi));
}
static double portable(const Input& q,bool& retried,int& largest_capacity) {
    for(int size=1;size<=1<<20;size*=2) {
        std::vector<Cell> heap(size);
        const auto result=ipc_gpu::solver::global_guess_detail::solve(q,heap.data(),size);
        require(result.status==0,"portable BFS status");
        if(!result.capacity){largest_capacity=std::max(largest_capacity,size);return result.toi;}
        retried=true;
    }
    throw std::runtime_error("portable BFS queue bound");
}

#ifdef IPC_GLOBAL_GUESS_CUDA_TEST
using namespace ipc_gpu::solver;
struct GuessEnvironment {
    const std::array<const char*,3> names{{"IPC_GPU_GLOBAL_GUESS_WORKERS",
        "IPC_GPU_GLOBAL_GUESS_THREADS","IPC_GPU_GLOBAL_GUESS_CANDIDATES_ONLY"}};
    std::array<std::string,3> values;
    std::array<bool,3> present{};
    GuessEnvironment() {
        for(std::size_t i=0;i<names.size();++i)if(const char* value=std::getenv(names[i])) {
            present[i]=true;values[i]=value;
        }
    }
    ~GuessEnvironment() {
        for(std::size_t i=0;i<names.size();++i)
            if(present[i])setenv(names[i],values[i].c_str(),1);else unsetenv(names[i]);
    }
};
static void configure_guess(int workers,int threads,bool candidates_only=false) {
    require(setenv("IPC_GPU_GLOBAL_GUESS_WORKERS",std::to_string(workers).c_str(),1)==0,"set guess workers");
    require(setenv("IPC_GPU_GLOBAL_GUESS_THREADS",std::to_string(threads).c_str(),1)==0,"set guess threads");
    require(setenv("IPC_GPU_GLOBAL_GUESS_CANDIDATES_ONLY",candidates_only?"1":"0",1)==0,"set guess build mode");
}
static void scheduling_checks(const std::vector<Input>& fixtures,const std::vector<double>& expected) {
    GuessEnvironment environment;
    // More jobs than the largest worker count exercises cursor reuse and a
    // partial last block. Mix crossing, all-moving and separated predicates.
    constexpr std::size_t count=8193;
    std::vector<Point> starts,ends;std::vector<BpContact> contacts;
    std::vector<double> reference;
    constexpr std::array<std::size_t,8> selected{{0,1,2,3,14,15,16,17}};
    for(std::size_t i=0;i<count;++i) {
        const auto fixture=selected[i%selected.size()];
        const auto& q=fixtures[fixture];const int offset=static_cast<int>(starts.size());
        for(int v=0;v<4;++v) {
            starts.push_back({q.start[v][0],q.start[v][1],q.start[v][2]});
            ends.push_back({q.end[v][0],q.end[v][1],q.end[v][2]});
        }
        contacts.push_back({{offset,offset+1,offset+2,offset+3},q.vertex_face?0:1});
        reference.push_back(expected[fixture]);
    }
    DeviceArray<Point> ds(starts),de(ends);DeviceArray<BpContact> dc(contacts);
    int device=0;cuda_checked(cudaGetDevice(&device));
    const auto query=[&]() {
        cuda_checked(cudaSetDevice(device));
        DeviceArray<double> result(count);DeviceArray<int> error(1);error.clear();
        GpuGlobalGuess guess;
        guess.query(reinterpret_cast<double(*)[3]>(ds.data()),reinterpret_cast<double(*)[3]>(de.data()),
            static_cast<int>(starts.size()),dc.data(),count,result.data(),error.data());
        require(error.download()[0]==0,"scheduled global guess status");
        const auto actual=result.download();
        require(std::memcmp(actual.data(),reference.data(),count*sizeof(double))==0,
            "global guess worker/thread scheduling changed TOI bytes");
        // Reuse the same owner with fewer jobs, then an empty query. This must
        // not consume stale retries, cursor values or output from a prior call.
        result.upload(std::vector<double>(count,-17.0));error.clear();
        guess.query(reinterpret_cast<double(*)[3]>(ds.data()),reinterpret_cast<double(*)[3]>(de.data()),
            static_cast<int>(starts.size()),dc.data(),3,result.data(),error.data());
        require(error.download()[0]==0,"reused global guess status");
        const auto short_result=result.download();
        require(std::memcmp(short_result.data(),reference.data(),3*sizeof(double))==0
            &&short_result[3]==-17.0,"reused global guess touched stale jobs");
        guess.query(nullptr,nullptr,0,nullptr,0,nullptr,error.data());
        require(result.download()==short_result,"empty query changed output");
    };
    for(const int workers:{512,2048,8192})for(const int threads:{1,8,32}) {
        configure_guess(workers,threads);
        query();
    }
    // Independent owners may be invoked by separate host threads; all query
    // arenas and retry queues remain instance-private. Mutate no environment
    // while either thread is running.
    configure_guess(2048,8);
    auto first=std::async(std::launch::async,query);
    auto second=std::async(std::launch::async,query);
    first.get();second.get();
    for(const auto& setting:std::vector<std::pair<const char*,std::vector<const char*>>>{
        {"IPC_GPU_GLOBAL_GUESS_WORKERS",{"","0","-1","+1"," 1","8193","999999999999999999999","1x"}},
        {"IPC_GPU_GLOBAL_GUESS_THREADS",{"","0","3","33","-1","8x"}},
        {"IPC_GPU_GLOBAL_GUESS_CANDIDATES_ONLY",{"","2","true","false","-1","1x","01","00"}}}) {
        for(const char* value:setting.second) {
            configure_guess(512,32);
            require(setenv(setting.first,value,1)==0,"set invalid guess configuration");
            bool rejected=false;
            try {GpuGlobalGuess invalid;}
            catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"invalid global guess configuration accepted");
        }
    }
    std::cout<<"Global guess schedules: 9 layouts, concurrent owners, reuse and strict options PASS"<<std::endl;
}
static void apply_checks(bool quick) {
    GuessEnvironment environment;
    configure_guess(512,32,false);
    // Two nonadjacent triangles: the upper one moves through the lower one.
    // Compare the whole initial-guess output to CPU all-pair global minimum.
    const std::vector<Point> start=quick
        ?std::vector<Point>{{0,0,1},{1,0,1},{0,1,1},{-.25,-.25,0},{1.20,-.25,0},{-.25,1.30,0}}
        :std::vector<Point>{{0,0,1},{1,0,1},{0,1,1},{-.25,-.25,0},{1.25,-.25,0},{-.25,1.25,0}};
    if(quick)std::cout<<"Quick whole-scene apply uses nonparallel hypotenuse geometry"<<std::endl;
    auto end=start;for(int i=0;i<3;++i)end[i][2]=-1;
    std::vector<BpTriangle> triangles={{{0,1,2}},{{3,4,5}}};
    std::vector<BpEdge> edges={{{0,1}},{{0,2}},{{1,2}},{{3,4}},{{3,5}},{{4,5}}};
    std::vector<BpContact> candidates;
    for(int n=0;n<6;++n)for(const auto& t:triangles) {
        if(n==t.vertices[0]||n==t.vertices[1]||n==t.vertices[2])continue;
        candidates.push_back({{n,t.vertices[0],t.vertices[1],t.vertices[2]},0});
    }
    for(std::size_t i=0;i<edges.size();++i)for(std::size_t j=i+1;j<edges.size();++j) {
        bool shared=false;for(int a:edges[i].vertices)for(int b:edges[j].vertices)shared|=a==b;
        if(!shared)candidates.push_back({{edges[i].vertices[0],edges[i].vertices[1],edges[j].vertices[0],edges[j].vertices[1]},1});
    }
    DeviceArray<Point> dx(start),previous(start),predicted(end),velocity(6);
    DeviceArray<int> node_blocks(std::vector<int>{0,1,2,3,4,5}),error(1);
    DeviceArray<unsigned char> deformable(std::vector<unsigned char>(6,1));
    DeviceArray<BpTriangle> dt(triangles);DeviceArray<BpEdge> de(edges);
    auto ptr=[](DeviceArray<Point>& a){return reinterpret_cast<double(*)[3]>(a.data());};
    DeformableScene scene{};scene.nodes=6;scene.node_blocks=node_blocks.data();scene.deformable=deformable.data();
    DeformableState state{ptr(dx),ptr(velocity),ptr(previous),ptr(predicted)};
    BroadPhaseTopology topology;topology.vertex_count=6;topology.block_count=6;
    topology.triangle_count=2;topology.edge_count=6;topology.triangles=dt.data();topology.edges=de.data();topology.vertex_block=node_blocks.data();
    GpuGlobalGuess guess;
    configure_guess(8192,8,true);
    GpuGlobalGuess candidate_guess;
    configure_guess(512,32,false); // Options are latched, not reread on apply.
    for(int mode=0;mode<3;++mode) {
        end=start;
        if(mode==0)for(int n=0;n<3;++n)end[n][2]=-1;
        if(mode==1)for(int n=0;n<6;++n)end[n][0]+=.17;
        if(mode==2)for(int n=0;n<3;++n)end[n][2]=0; // t=1 contact.
        predicted.upload(end);dx.upload(start);error.clear();
        double minimum=1;
        for(const auto& c:candidates) {
            Input q;q.vertex_face=c.kind==0;
            for(int v=0;v<4;++v)for(int a=0;a<3;++a) {
                int n=c.vertices[v];q.start[v][a]=start[n][a];q.end[v][a]=start[n][a]+(end[n][a]-start[n][a]);
            }
            minimum=std::min(minimum,cpu_reference(q));
        }
        std::cout<<"CUDA whole-scene apply mode "<<mode<<" starting"<<std::endl;
        guess.apply(scene,state,topology,error.data());
        require(error.download()[0]==0,"global guess apply status");
        const auto actual=dx.download();const double omega=minimum>=1?1:.9*minimum;
        for(int n=0;n<6;++n)for(int a=0;a<3;++a) {
            const double expected=start[n][a]+omega*(end[n][a]-start[n][a]);
            require(std::abs(actual[n][a]-expected)<=1e-6,"global guess whole-scene output");
        }
        dx.upload(start);error.clear();candidate_guess.apply(scene,state,topology,error.data());
        require(error.download()[0]==0,"candidate-only global guess apply status");
        const auto candidate_actual=dx.download();
        require(std::memcmp(candidate_actual.data(),actual.data(),actual.size()*sizeof(Point))==0,
            "candidate-only global guess changed represented endpoints");
    }
    // No topology means no contacts: accept original target using CPU order.
    topology.triangle_count=topology.edge_count=0;
    std::cout<<"CUDA whole-scene no-contact apply starting"<<std::endl;
    error.clear();guess.apply(scene,state,topology,error.data());
    const auto actual=dx.download();require(error.download()[0]==0,"empty guess status");
    for(int n=0;n<6;++n)for(int a=0;a<3;++a)
        require(actual[n][a]==start[n][a]+(end[n][a]-start[n][a]),"empty guess exact target");
    dx.upload(start);error.clear();candidate_guess.apply(scene,state,topology,error.data());
    require(error.download()[0]==0,"empty candidate-only guess status");
    const auto candidate_actual=dx.download();
    require(std::memcmp(candidate_actual.data(),actual.data(),actual.size()*sizeof(Point))==0,
        "empty candidate-only guess changed represented endpoints");
}
static void cuda_checks(const std::vector<Input>& inputs,const std::vector<double>& reference,
                        bool include_wide_frontier) {
    std::vector<Point> starts,ends;std::vector<BpContact> contacts;
    for(const auto& q:inputs) {
        const int offset=starts.size();
        for(int v=0;v<4;++v){starts.push_back({q.start[v][0],q.start[v][1],q.start[v][2]});ends.push_back({q.end[v][0],q.end[v][1],q.end[v][2]});}
        contacts.push_back({{offset,offset+1,offset+2,offset+3},q.vertex_face?0:1});
    }
    DeviceArray<Point> ds(starts),de(ends);DeviceArray<BpContact> dc(contacts);
    DeviceArray<double> results(inputs.size());DeviceArray<int> error(1);error.clear();
    GpuGlobalGuess guess;
    std::cout<<"CUDA query fixtures starting: "<<inputs.size()
             <<(include_wide_frontier?" (includes wide frontier)":" (quick: no wide frontier)")<<std::endl;
    guess.query(reinterpret_cast<double(*)[3]>(ds.data()),reinterpret_cast<double(*)[3]>(de.data()),
        starts.size(),dc.data(),contacts.size(),results.data(),error.data());
    require(error.download()[0]==0,"CUDA global CCD status");
    const auto actual=results.download();double max_error=0;
    std::cout<<"CUDA query fixtures completed"<<std::endl;
    require(actual[0]<=.5&&actual[0]>=.5-1e-6,"CUDA known PT crossing must be conservative");
    require(actual[14]<=.5&&actual[14]>=.5-1e-6,"CUDA known EE crossing must be conservative");
    if(include_wide_frontier)
        require(actual[28]<=.5,"CUDA wide-frontier EE crossing must be conservative");
    for(std::size_t i=0;i<actual.size();++i) {
        max_error=std::max(max_error,std::abs(actual[i]-reference[i]));
        if(std::abs(actual[i]-reference[i])>1e-6) {
            std::cerr<<"CUDA mismatch "<<i<<" reference "<<reference[i]<<" actual "<<actual[i]<<"\n";
            throw std::runtime_error("CUDA vs CPU Tight-Inclusion TOI");
        }
    }
    scheduling_checks(inputs,actual);
    // Invalid and preexisting-error queries fail closed, not silently safe.
    std::cout<<"CUDA invalid/preexisting-error checks starting"<<std::endl;
    starts[0][0]=INFINITY;ds.upload(starts);error.clear();
    guess.query(reinterpret_cast<double(*)[3]>(ds.data()),reinterpret_cast<double(*)[3]>(de.data()),
        starts.size(),dc.data(),1,results.data(),error.data());
    require(error.download()[0]==50,"nonfinite query must fail");
    error.upload(std::vector<int>{71});
    guess.query(reinterpret_cast<double(*)[3]>(ds.data()),reinterpret_cast<double(*)[3]>(de.data()),
        starts.size(),dc.data(),1,results.data(),error.data());
    require(error.download()[0]==71,"existing error preserved");
    std::cout<<"CUDA invalid/preexisting-error checks completed"<<std::endl;
    apply_checks(!include_wide_frontier);
    std::cout<<"CUDA global guess: "<<inputs.size()<<" PT/EE fixtures; max TOI error "<<max_error<<"; whole-guess checks passed"<<std::endl;
}
#endif

int main(int argc,char** argv) {
    try {
        const bool quick=argc==2&&std::strcmp(argv[1],"--quick")==0;
        require(argc==1||quick,"usage: gpu_global_guess_test [--quick]");
        const auto inputs=fixtures(!quick);std::vector<double> reference;bool retried=false;
        std::cout<<(quick?"Quick mode: 540 fixtures; wide-frontier stress omitted":"Full mode: 541 fixtures; wide-frontier stress included")<<std::endl;
        std::cout<<"CPU reference and portable BFS checks starting"<<std::endl;
        double maximum_error=0;int exact=0,largest_capacity=0;
        for(std::size_t i=0;i<inputs.size();++i) {
            const double expected=cpu_reference(inputs[i]),actual=portable(inputs[i],retried,largest_capacity);
            reference.push_back(expected);maximum_error=std::max(maximum_error,std::abs(expected-actual));
            exact+=expected==actual;
            if(std::abs(expected-actual)>1e-6) {
                std::cerr<<"portable mismatch "<<i<<" reference "<<expected<<" actual "<<actual<<"\n";
                throw std::runtime_error("portable vs CPU Tight-Inclusion TOI");
            }
        }
        require(retried,"capacity retries not exercised");
        if(!quick)require(largest_capacity>256,"real CUDA queue capacity retry fixture missing");
        require(reference[0]<=.5&&reference[0]>=.5-1e-6,"CPU known PT crossing must be conservative");
        {
            auto invalid=inputs[0];invalid.start[0][0]=INFINITY;
            std::vector<Cell> heap(256);
            require(ipc_gpu::solver::global_guess_detail::solve(invalid,heap.data(),256).status==50,
                    "portable nonfinite query must fail");
        }
        std::cout<<"Global guess portable BFS: "<<inputs.size()<<" fixtures, "<<exact<<" exact TOIs, max error "<<maximum_error
                 <<", maximum queue capacity "<<largest_capacity<<std::endl;
#ifdef IPC_GLOBAL_GUESS_CUDA_TEST
        cuda_checks(inputs,reference,!quick);
#endif
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
