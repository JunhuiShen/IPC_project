#include <qualified_evaluator_oracle.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
namespace gpu=ipc_gpu::solver;
namespace trial=gpu::original_contact_storage;
#ifdef IPC_ORIGINAL_SELF_STORAGE_CUDA_TEST
void gpu_original_self_storage_check(const gpu::ContactInput*,std::size_t);
#endif
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
std::size_t checks=0,active_count=0,error_count=0;
void compare(gpu::ContactInput in){
    const auto old=gpu::evaluate_contact_original_qualified_oracle(in);
    const auto shared=gpu::evaluate_contact_original(in);
    require(std::memcmp(&old,&shared,sizeof(old))==0,"full shared-body bytes changed");
    ++checks;active_count+=old.active!=0;error_count+=old.status!=gpu::ContactStatus::Ok;
    if(in.friction==0.)for(int role=0;role<4;++role)if(in.hessian_blocks==(1u<<(5*role))){
        const auto thin=gpu::evaluate_contact_original_self_zero(in,role);
        require(thin.status==old.status&&thin.active==old.active,"thin status/active changed");
        require(std::memcmp(&thin.distance,&old.distance,sizeof(double))==0
            &&std::memcmp(thin.weights,old.weights,sizeof(old.weights))==0
            &&std::memcmp(thin.normal,old.normal,sizeof(old.normal))==0
            &&std::memcmp(thin.barrier_gradient,old.barrier_gradient,sizeof(old.barrier_gradient))==0,
            "thin geometry/all12 gradients changed");
        require(std::memcmp(thin.selected_hessian,old.barrier_hessian[role][role],sizeof(thin.selected_hessian))==0,
            "thin self Hessian changed");
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)if(i!=role||j!=role)
            for(double value:old.barrier_hessian[i][j])require(value==0.&&!std::signbit(value),"omitted full block not logical+zero");
        for(double value:old.friction_relative_gradient)require(value==0.&&!std::signbit(value),"zero friction gradient not+zero");
        for(double value:old.friction_relative_hessian)require(value==0.&&!std::signbit(value),"zero friction Hessian not+zero");
        ++checks;
    }
}
gpu::ContactInput make(bool edge,double z){
    gpu::ContactInput in;
    const double pt[4][3]={{.25,.25,z},{0,0,0},{1,0,0},{0,1,0}};
    const double ee[4][3]={{-.5,0,z},{.5,0,z},{0,-.5,0},{0,.5,0}};
    std::memcpy(in.positions,edge?ee:pt,sizeof(pt));std::memcpy(in.previous,in.positions,sizeof(pt));
    in.kind=edge?gpu::ContactKind::EdgeEdge:gpu::ContactKind::PointTriangle;
    return in;
}
int main(){try{
    std::vector<gpu::ContactInput> fixtures,all_inputs;
    for(bool edge:{false,true})for(double z:{0.,0x1p-100,1e-12,.001,.004,.005,.006,1.}){
        auto in=make(edge,z);fixtures.push_back(in);
        in.positions[2][0]=in.positions[1][0];in.positions[2][1]=in.positions[1][1];fixtures.push_back(in);
        in=make(edge,z);for(int k=0;k<3;++k)in.positions[3][k]=in.positions[2][k];fixtures.push_back(in);
    }
    std::mt19937_64 rng(0x715b6c1);std::uniform_real_distribution<double> u(-1,1);
    for(int n=0;n<2000;++n){auto in=make(n%2,.001);const double scale=std::ldexp(1.,int(rng()%121)-60);
        for(int i=0;i<4;++i)for(int k=0;k<3;++k){in.positions[i][k]=scale*u(rng);in.previous[i][k]=in.positions[i][k]+scale*.0001*u(rng);}
        in.d_hat=scale*.2;fixtures.push_back(in);
    }
    for(auto in:fixtures)for(unsigned mask:{0u,1u,32u,1024u,32768u,65535u,2u,65536u})
        for(double friction:{0.,-0.,.1}){in.hessian_blocks=mask;in.friction=friction;all_inputs.push_back(in);}
    for(bool edge:{false,true})for(int role=0;role<4;++role){
        auto base=make(edge,.001);base.hessian_blocks=1u<<(5*role);
        for(int vertex=0;vertex<4;++vertex)for(int axis=0;axis<3;++axis)
            for(double invalid:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}){
                auto in=base;in.positions[vertex][axis]=invalid;all_inputs.push_back(in);
                in=base;in.previous[vertex][axis]=invalid;all_inputs.push_back(in);
            }
        for(int field=0;field<6;++field)for(double value:{0.,-1.,std::numeric_limits<double>::max(),std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::infinity()}){
            auto in=base;double* scalars[]={&in.d_hat,&in.k_barrier,&in.dt,&in.friction,&in.eps_v,&in.eps};*scalars[field]=value;all_inputs.push_back(in);
        }
        auto in=base;in.dt=std::numeric_limits<double>::denorm_min();in.eps_v=.001;all_inputs.push_back(in);
        require(gpu::evaluate_contact_original(in).status==gpu::ContactStatus::InvalidInput,"dt*eps_v underflow zero-friction premise lost");
        in=base;in.dt=std::numeric_limits<double>::max();all_inputs.push_back(in);
        require(gpu::evaluate_contact_original(in).status==gpu::ContactStatus::InvalidInput,"dt2 overflow premise lost");
    }
    for(const auto& in:all_inputs)compare(in);
    for(int role=-1;role<=4;++role)for(unsigned mask:{0u,1u,32u,65535u,65536u})
        for(double friction:{0.,-0.,.1,std::numeric_limits<double>::quiet_NaN()}) {
            auto in=make(false,.001);in.hessian_blocks=mask;in.friction=friction;
            const bool eligible=role>=0&&role<4&&mask==(1u<<(5*role))&&friction==0.;
            const auto out=gpu::evaluate_contact_original_self_zero(in,role);
            if(!eligible)require(out.status==gpu::ContactStatus::InvalidInput&&!out.active,
                "unsupported thin call did not fail closed");
            else compare(in);
        }
    trial::SelfZeroOutput write_probe{};
    require(!trial::Access<trial::SelfZeroOutput>::put_hessian(write_probe,0,1,0,1.,0),
        "thin storage silently accepted off-self write");
    require(!trial::Access<trial::SelfZeroOutput>::put_hessian(write_probe,0,0,9,1.,0),
        "thin storage silently accepted out-of-range write");
#ifdef IPC_ORIGINAL_SELF_STORAGE_CUDA_TEST
    gpu_original_self_storage_check(all_inputs.data(),all_inputs.size());
#endif
    require(active_count>0&&error_count>0,"host matrix has no active/error witnesses");
    std::cout<<"Original shared-value-storage differential PASS: checks="<<checks<<" active="<<active_count
        <<" errors="<<error_count<<" full1416/thin240/all12grad/soleH9; no compiler-resource/speedup claim\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
