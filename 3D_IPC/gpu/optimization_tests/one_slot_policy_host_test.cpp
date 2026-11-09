#include "../rigid_one_slot_policy.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
namespace p=ipc_gpu::solver::rigid_one_slot;
void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
int main(){try{
    std::size_t checks=0;
    for(int a=-1;a<=2;++a)for(int b=-1;b<=2;++b)
    for(int o0=-1;o0<=2;++o0)for(int o1=-1;o1<=2;++o1)for(int o2=1;o2<=3;++o2)
    for(int x=-1;x<=2;++x)for(int y=-1;y<=2;++y) {
        const int c[2]={a,b},o[3]={o0,o1,o2},blocks[2]={x,y};
        const bool valid_colors=(a==0&&b==1)||(a==1&&b==0);
        const bool valid_offsets=o0==0&&o1==1&&o2==2;
        const bool valid_blocks=(x==0&&y==1)||(x==1&&y==0);
        const bool expected=valid_colors&&valid_offsets&&valid_blocks
            &&((x==0?a:b)==0)&&((y==0?a:b)==1);
        require(p::proper_colors(c,o,blocks,2,2)==expected,"proper-color capability changed");++checks;
    }
    const int c[2]={0,1},o[3]={0,1,2},b[2]={0,1};
    require(!p::proper_colors(nullptr,o,b,2,2)&&!p::proper_colors(c,nullptr,b,2,2)
        &&!p::proper_colors(c,o,nullptr,2,2),"null map was accepted");
    for(int blocks:{0,1,3})require(!p::proper_colors(c,o,b,blocks,2),"wrong block count accepted");
    for(int colors:{0,1,3})require(!p::proper_colors(c,o,b,2,colors),"wrong color count accepted");
    for(unsigned mask=0;mask<(1u<<11);++mask) {
        auto bit=[&](unsigned n){return(mask&(1u<<n))!=0;};
        const bool expected=bit(0)&&bit(1)&&bit(2)&&!bit(3)&&bit(4)&&bit(5)&&!bit(6)
            &&bit(7)&&bit(8)&&bit(9)&&bit(10);
        require(p::eligible(bit(0),bit(1),bit(2),bit(3),bit(4),bit(5),bit(6),
            bit(7),bit(8),bit(9),2,7,7,bit(10))==expected,"eligibility guard changed");++checks;
    }
    require(!p::eligible(true,true,true,false,true,true,false,true,true,true,1,7,7,true)
        &&!p::eligible(true,true,true,false,true,true,false,true,true,true,2,6,7,true),
        "body/node provenance mismatch accepted");
    for(std::size_t slot:{std::size_t(0),std::size_t(1),std::size_t(1048575),std::size_t(1ull<<40)})
        for(unsigned role=0;role<4;++role) {
            require(p::term_index(false,slot,role)==slot*4+role,"generic first-role slot changed");
            require(p::term_index(true,slot,role)==slot,"single owner slot depends on role");checks+=2;
        }
    require(p::release_on_transition(false,true,12,3),"generic→one did not shrink");
    require(!p::release_on_transition(true,true,12,3),"same-mode highwater was shrunk");
    require(!p::release_on_transition(true,false,3,4),"grow was incorrectly released first");
    require(p::release_on_transition(true,false,30,4),"small generic return retained excess capacity");
    std::cout<<"One-slot integer capability policy PASS: checks="<<checks
        <<"; no CUDA, FP arithmetic or solver simulation\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
