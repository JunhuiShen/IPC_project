#define main old_one_slot_facade_main
#include "sparse_fixture.inc"
#undef main
#include <cfloat>
#include <limits>
namespace {
std::size_t cases=0,force_witnesses=0,storage_witnesses=0;
void compare_case(bool trusted,int graph,bool frozen,bool activity,bool cache,bool orientation,
    int kinds,bool far,std::size_t capacity,const ContactParameters& params,int fault=0,int expected=-1) {
    const std::size_t contacts=2*capacity+3;
    Result serial,sparse,repeated,unset;
    Setting activity_option("IPC_GPU_CONTACT_OUTPUT_ACTIVITY",activity?"1":"0");
    {Setting feature("IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER","0");
        serial=run(true,trusted,graph,frozen,fault,orientation,contacts,capacity,false,false,nullptr,&params,kinds,far,cache);}
    {Setting feature("IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER","1");
        sparse=run(true,trusted,graph,frozen,fault,orientation,contacts,capacity,false,false,nullptr,&params,kinds,far,cache);}
    {Setting feature("IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER","1");
        repeated=run(true,trusted,graph,frozen,fault,orientation,contacts,capacity,false,false,nullptr,&params,kinds,far,cache);}
    {Setting feature("IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER",nullptr);
        unset=run(true,trusted,graph,frozen,fault,orientation,contacts,capacity,false,false,nullptr,&params,kinds,far,cache);}
    need(sparse.codes==unset.codes&&std::memcmp(sparse.values.data(),unset.values.data(),
        sparse.values.size()*sizeof(Derivatives))==0,"sparse UNSET/ON default bytes/status differ");
    for(std::size_t epoch=0;epoch<sparse.storage.size();++epoch)
        need(sparse.storage[epoch].packet_bytes==unset.storage[epoch].packet_bytes
            &&sparse.storage[epoch].exact_bytes==unset.storage[epoch].exact_bytes,"sparse UNSET/ON resources differ");
    need(serial.codes==sparse.codes&&sparse.codes==repeated.codes,"sparse gather changed error status");
    need(serial.values.size()==sparse.values.size()
         &&std::memcmp(serial.values.data(),sparse.values.data(),serial.values.size()*sizeof(Derivatives))==0
         &&std::memcmp(sparse.values.data(),repeated.values.data(),sparse.values.size()*sizeof(Derivatives))==0,
         "sparse gather changed force/H/publication bytes or repeatability");
    if(expected>=0)for(std::size_t epoch=0;epoch<4;++epoch)
        need(serial.codes[epoch]==(epoch==2?0:expected),"native error/inactive premise changed");
    if(!fault&&expected==0&&!far&&params.k_barrier>0) {
        need(std::memcmp(serial.values.data(),serial.values.data()+4,2*sizeof(Derivatives))!=0,
             "active native force fixture is vacuous");++force_witnesses;
    }
    const bool eligible=trusted&&!frozen&&activity&&cache&&(graph==0||graph==5)&&params.friction==0.0;
    for(std::size_t epoch:{std::size_t(0),std::size_t(3)}) {
        const auto delta=eligible?(capacity*sizeof(unsigned)+sizeof(unsigned)):0;
        need(sparse.storage[epoch].packet_bytes==serial.storage[epoch].packet_bytes+delta,
             "sparse bounded storage/fallback witness changed");
        need(sparse.storage[epoch].exact_bytes==serial.storage[epoch].exact_bytes,"sparse changed exact arenas");
        ++storage_witnesses;
    }
    ++cases;
}
}
int main(){try{
    for(const char* bad:{"invalid","2","-1"}) {
        Setting flag("IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER",bad);bool rejected=false;
        try{GpuContactPipeline invalid(3);}catch(const std::invalid_argument&){rejected=true;}
        need(rejected,"malformed sparse selector accepted");
    }
    ContactParameters params{.5,37,.031,0,.1,1e-12};
    for(const char* kind_packets:{"0","1"}) {
        Setting spans("IPC_GPU_TWO_RIGID_KIND_PACKETS",kind_packets);
        for(int kinds:{0,1,2})for(bool far:{false,true})for(bool orientation:{false,true})
            for(std::size_t cap:{std::size_t(1),std::size_t(3),std::size_t(65),std::size_t(129),std::size_t(257)})
                compare_case(true,0,false,true,true,orientation,kinds,far,cap,params,0,0);
    }
    // Each unsupported capability is exercised independently, retaining the
    // original complete gather and full publication when activity is disabled.
    for(bool orientation:{false,true})for(int kinds:{0,1,2}) {
        compare_case(false,0,false,true,true,orientation,kinds,false,3,params,0,0);
        compare_case(true,0,true,true,true,orientation,kinds,false,3,params,0,0);
        compare_case(true,0,false,false,true,orientation,kinds,false,3,params,0,0);
        compare_case(true,0,false,true,false,orientation,kinds,false,3,params,0,0);
        for(int graph:{1,2,3,4,5})compare_case(true,graph,false,true,true,orientation,kinds,false,3,params,0,0);
        for(double friction:{-0.0,.1}) {
            auto p=params;p.friction=friction;
            compare_case(true,0,false,true,true,orientation,kinds,false,3,p,0,0);
        }
        auto zero=params;zero.k_barrier=0;
        compare_case(true,0,false,true,true,orientation,kinds,false,65,zero,0,0);
        auto inactive=params;inactive.k_barrier=DBL_MAX;inactive.dt=2;
        compare_case(true,0,false,true,true,orientation,kinds,true,65,inactive,0,0);
        for(unsigned bad=0;bad<6;++bad) {
            auto p=params;
            if(bad==0)p.k_barrier=-1;
            if(bad==1)p.k_barrier=std::numeric_limits<double>::infinity();
            if(bad==2)p.k_barrier=std::numeric_limits<double>::quiet_NaN();
            if(bad==3)p.d_hat=std::numeric_limits<double>::quiet_NaN();
            if(bad==4){p.dt=1e-200;p.eps_v=1e-200;}
            if(bad==5){p.dt=1e200;p.eps_v=1e-200;}
            compare_case(true,0,false,true,true,orientation,kinds,true,65,p,0,20);
        }
        compare_case(true,0,false,true,true,orientation,kinds,false,65,params,2,-1);
    }
    for(bool orientation:{false,true})compare_case(true,0,false,true,true,orientation,0,false,65,params,1,-1);
    std::cout<<"Original sparse gather functional PASS: cases="<<cases<<" force_witnesses="<<force_witnesses
        <<" storage_witnesses="<<storage_witnesses
        <<"; PT/EE/mixed tails, exact systems/status/publication/reuse, malformed/raw/frozen/activity/cache fallbacks; activation and trajectory separate\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
