// Host-only arithmetic-order check; the CUDA test also covers scheduling/SDF.
#include "contact.cuh"
#include "contact_compact.cuh"
#include "kernels.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace ipc_gpu;
using namespace ipc_gpu::solver;
#include "contact_warp_reduce.cuh"

int main() {
    std::mt19937_64 random(20261011);
    std::uniform_real_distribution<double> value(-100.,100.);
    for(int test=0;test<1000;++test) {
        Derivatives original{},lanes{},compact_lanes{};
        for(int k=0;k<3;++k)original.gradient[k]=lanes.gradient[k]=compact_lanes.gradient[k]=value(random);
        for(int k=0;k<9;++k)original.hessian[k]=lanes.hessian[k]=compact_lanes.hessian[k]=value(random);
        const double scale=value(random);
        for(int pair=0;pair<37;++pair) {
            ContactOutput out{};
            for(int role=0;role<4;++role) {
                out.weights[role]=value(random);
                for(int k=0;k<3;++k)out.barrier_gradient[role][k]=value(random);
                for(int k=0;k<9;++k)out.barrier_hessian[role][role][k]=value(random);
            }
            for(int k=0;k<3;++k)out.friction_relative_gradient[k]=test%2?value(random):0.;
            for(int k=0;k<9;++k)out.friction_relative_hessian[k]=test%2?value(random):0.;
            for(int role=0;role<4;++role)if((pair+role)%3) {
                const double weight=out.weights[role];
                for(int k=0;k<3;++k) {
                    original.gradient[k]+=scale*out.barrier_gradient[role][k];
                    original.gradient[k]+=weight*out.friction_relative_gradient[k];
                    for(int l=0;l<3;++l) {
                        original.hessian[3*k+l]+=scale*out.barrier_hessian[role][role][3*k+l];
                        original.hessian[3*k+l]+=(weight*out.friction_relative_hessian[3*k+l])*weight;
                    }
                }
                for(int lane=0;lane<12;++lane) {
                    double& entry=lane<3?lanes.gradient[lane]:lanes.hessian[lane-3];
                    entry=contact_warp_add_component(entry,out,role,lane,scale);
                }
                ContactInput input;input.hessian_blocks=1u<<(5*role);
                out.active=1;
                ContactOutput unused;CompactDeformableContact compact;
                store_contact_output(input,out,&unused,&compact);
                for(int lane=0;lane<12;++lane) {
                    double& entry=lane<3?compact_lanes.gradient[lane]:compact_lanes.hessian[lane-3];
                    entry=contact_warp_add_compact_component(entry,compact,lane,scale);
                }
            }
        }
        if(std::memcmp(&original,&lanes,sizeof(original)))throw std::runtime_error("Arithmetic-order byte mismatch");
        if(std::memcmp(&original,&compact_lanes,sizeof(original)))throw std::runtime_error("Compact arithmetic-order byte mismatch");
    }
    std::cout<<"Full/compact warp contact component arithmetic: 1000 host byte-equivalence fixtures PASS\n";
}
