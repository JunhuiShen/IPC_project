#include "node_cooperative_grid.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace grid=ipc_gpu::solver::node_cooperative_grid;
void require(bool ok,const char* text) {if(!ok)throw std::runtime_error(text);}
int main() {
    std::size_t checks=0;
    for(unsigned incoming:{0u,1u,2u,16u,79u,128u,142u,143u,256u,284u,1024u})
        for(std::size_t nodes:{std::size_t(0),std::size_t(1),std::size_t(127),std::size_t(128),
                std::size_t(129),std::size_t(10000),std::size_t(18176),std::size_t(18177),
                std::size_t(36352),std::size_t(36353),std::numeric_limits<std::size_t>::max()})
            for(int workers:{-1,0,1,2,16,128,142,143,255,256,284,285})
                for(unsigned preferred:{0u,1u,79u,142u,143u,256u,284u,1024u,std::numeric_limits<unsigned>::max()}) {
                    unsigned first=0;
                    if(nodes&&workers>0)
                        for(unsigned candidate=1;candidate<=incoming;++candidate)
                            if((nodes-1)/128<candidate&&unsigned(workers)<=candidate) {first=candidate;break;}
                    const auto actual=grid::choose(incoming,nodes,workers,preferred);
                    require(actual.valid==(first!=0),"eligibility differs from explicit coverage search");
                    if(actual.valid) {
                        unsigned expected=incoming;
                        if(preferred) {
                            expected=first;
                            while(expected<incoming&&expected<preferred)++expected;
                        }
                        require(actual.grid==expected,"selected grid differs from coverage/preference search");
                        require(actual.grid<=incoming&&(nodes-1)/128<actual.grid
                            &&unsigned(workers)<=actual.grid,"node/arena/residency coverage lost");
                    }
                    ++checks;
                }
    require(grid::choose(284,10000,16,142).grid==142,"real scene policy");
    require(grid::choose(284,18177,16,142).grid==143,"last node not covered");
    require(grid::choose(284,10000,256,142).grid==256,"last arena not covered");
    std::cout<<"Cooperative grid independent coverage PASS: "<<checks<<" cases\n";
}
