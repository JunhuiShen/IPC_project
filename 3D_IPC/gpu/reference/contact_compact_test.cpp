#include "contact_compact.cuh"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace ipc_gpu::solver;
void require(bool ok){if(!ok)throw std::runtime_error("compact contact bit preservation failed");}
void same(double a,double b){require(std::memcmp(&a,&b,sizeof(double))==0);}
int main() {
    ContactOutput original;
    original.active=1;
    for(int role=0;role<4;++role) {
        original.weights[role]=role?-double(role)/7.:-0.;
        for(int k=0;k<3;++k)original.barrier_gradient[role][k]=std::ldexp(double(role*3+k)-4.,-71);
        for(int k=0;k<9;++k)original.barrier_hessian[role][role][k]=std::ldexp(double(role*9+k)-7.,43);
    }
    for(int k=0;k<3;++k)original.friction_relative_gradient[k]=k?1./k:-0.;
    for(int k=0;k<9;++k)original.friction_relative_hessian[k]=k?-1./k:0.;
    for(int role=0;role<4;++role) {
        ContactInput in;in.hessian_blocks=1u<<(5*role);
        ContactOutput unused;CompactDeformableContact compact;
        store_contact_output(in,original,&unused,&compact);
        require(compact.role==role&&compact.status==original.status&&compact.active==original.active);
        same(compact.weight,original.weights[role]);
        for(int k=0;k<3;++k){same(compact.gradient[k],original.barrier_gradient[role][k]);same(compact.friction_gradient[k],original.friction_relative_gradient[k]);}
        for(int k=0;k<9;++k){same(compact.hessian[k],original.barrier_hessian[role][role][k]);same(compact.friction_hessian[k],original.friction_relative_hessian[k]);}
    }
    for(unsigned mask:{0u,0xffffu,33u}) {
        ContactInput in;in.hessian_blocks=mask;
        ContactOutput full;CompactDeformableContact compact;
        store_contact_output(in,original,&full,&compact);
        require(compact.role==-1&&std::memcmp(&full,&original,sizeof(full))==0);
    }
    std::cout<<"Compact contact stores "<<sizeof(CompactDeformableContact)<<" vs "<<sizeof(ContactOutput)<<" bytes; bit preservation PASS\n";
}
