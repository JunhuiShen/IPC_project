#include "make_shape.h"
#include "physics.h"

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

using VecX = Eigen::VectorXd;
using MatX = Eigen::MatrixXd;


VecX flatten_positions(const std::vector<Vec3>& x){
    VecX q(3 * x.size());
    for(int i=0;i<x.size();++i) q.segment<3>(3*i)=x[i];
    return q;
}

std::vector<Vec3> unflatten_positions(const VecX& q){
    std::vector<Vec3> x(q.size()/3);
    for(int i=0;i<x.size();++i) x[i]=q.segment<3>(3*i);
    return x;
}

double total_energy(const RefMesh& ref_mesh,const LumpedMass& M,const std::vector<Pin>& pins,const SimParams& params,const VecX& q,const std::vector<Vec3>& xhat){
    return compute_incremental_potential(ref_mesh,M,pins,params,unflatten_positions(q),xhat);
}


Vec3 local_gradient_fd(int vi,const RefMesh& ref_mesh,const LumpedMass& M,const VertexAdjacency& adj,const std::vector<Pin>& pins,const SimParams& params,const std::vector<Vec3>& x,const std::vector<Vec3>& xhat,double eps){
    Vec3 gfd=Vec3::Zero();
    for(int d=0;d<3;++d){
        auto xp=x,xm=x;
        xp[vi](d)+=eps;
        xm[vi](d)-=eps;
        double Ep=compute_incremental_potential(ref_mesh,M,pins,params,xp,xhat);
        double Em=compute_incremental_potential(ref_mesh,M,pins,params,xm,xhat);
        gfd(d)=(Ep-Em)/(2.0*eps);
    }
    return gfd;
}

Mat33 local_hessian_fd(int vi,const RefMesh& ref_mesh,const LumpedMass& M,const VertexAdjacency& adj,const std::vector<Pin>& pins,const SimParams& params,const std::vector<Vec3>& x,const std::vector<Vec3>& xhat,double eps){
    Mat33 H=Mat33::Zero();
    for(int d=0;d<3;++d){
        auto xp=x,xm=x;
        xp[vi](d)+=eps;
        xm[vi](d)-=eps;
        Vec3 gp=compute_local_gradient(vi,ref_mesh,M,adj,pins,params,xp,xhat);
        Vec3 gm=compute_local_gradient(vi,ref_mesh,M,adj,pins,params,xm,xhat);
        H.col(d)=(gp-gm)/(2.0*eps);
    }
    return H;
}


void slope2_check(const RefMesh& ref_mesh,const LumpedMass& M,const std::vector<Pin>& pins,const SimParams& params,const std::vector<Vec3>& x,const std::vector<Vec3>& xhat,int vi){
    std::cout<<"\n=== slope-2 check vertex "<<vi<<" ===\n";

    VertexAdjacency adj=build_vertex_adjacency(ref_mesh);

    Vec3 g=compute_local_gradient(vi,ref_mesh,M,adj,pins,params,x,xhat);
    Mat33 H=compute_local_hessian(vi,ref_mesh,M,adj,pins,params,x);

    Vec3 dir(0.3,-0.5,0.8); dir.normalize();

    std::vector<double> hs={1e-2,5e-3,2.5e-3,1.25e-3};
    std::vector<double> errs;

    for(double h:hs){
        auto xh=x;
        xh[vi]+=h*dir;

        Vec3 gh=compute_local_gradient(vi,ref_mesh,M,adj,pins,params,xh,xhat);
        Vec3 lin=g+h*H*dir;

        double err=(gh-lin).norm();
        errs.push_back(err);

        std::cout<<"h="<<h<<" err="<<err<<"\n";
    }

    for(int i=1;i<errs.size();++i)
        std::cout<<"ratio="<<errs[i-1]/errs[i]<<" (~4 expected)\n";
}

int main(){
    SimParams params;
    params.dt=1.0/30.0;
    params.mu=100.0;
    params.lambda=100.0;
    params.density=1.0;
    params.thickness=0.1;
    params.kpin=1e3;
    params.gravity=Vec3(0,-9.81,0);
    params.max_global_iters=50;
    params.tol_abs=1e-8;
    params.step_weight=1.0;

    RefMesh ref_mesh;
    DeformedState state;
    std::vector<Pin> pins;

    clear_model(ref_mesh,state,pins);

    int base=build_square_mesh(ref_mesh,state,1,1,1.0,1.0,Vec3(0,0,0));

    state.velocities.assign(state.deformed_positions.size(),Vec3::Zero());

    // perturb
    state.deformed_positions[0]+=Vec3(0.02,-0.01,0.03);
    state.deformed_positions[1]+=Vec3(-0.01,0.03,-0.02);
    state.deformed_positions[2]+=Vec3(0.01,0.02,0.01);
    state.deformed_positions[3]+=Vec3(-0.02,-0.01,0.02);

    append_pin(pins,base+2,state.deformed_positions);
    append_pin(pins,base+3,state.deformed_positions);

    LumpedMass M=build_lumped_mass(ref_mesh,params.density,params.thickness);
    VertexAdjacency adj=build_vertex_adjacency(ref_mesh);

    std::vector<Vec3> xhat=state.deformed_positions;
    std::vector<Vec3> x=state.deformed_positions;

    std::cout<<std::setprecision(12);

   // directional derivative
    {
        std::cout<<"=== total energy check ===\n";

        VecX q=flatten_positions(x);
        VecX dir=VecX::Random(q.size()); dir.normalize();

        double eps=1e-6;

        double fd=(total_energy(ref_mesh,M,pins,params,q+eps*dir,xhat)
                   - total_energy(ref_mesh,M,pins,params,q-eps*dir,xhat))/(2.0*eps);

        VecX g(3*x.size());
        for(int vi=0;vi<x.size();++vi)
            g.segment<3>(3*vi)=compute_local_gradient(vi,ref_mesh,M,adj,pins,params,x,xhat);

        double an=g.dot(dir);

        std::cout<<"FD="<<fd<<" analytic="<<an<<" error="<<std::abs(fd-an)<<"\n";
    }

    // gradient check
    {
        std::cout<<"\n=== gradient check ===\n";
        double eps=1e-6;

        for(int vi=0;vi<x.size();++vi){
            Vec3 g=compute_local_gradient(vi,ref_mesh,M,adj,pins,params,x,xhat);
            Vec3 gfd=local_gradient_fd(vi,ref_mesh,M,adj,pins,params,x,xhat,eps);

            std::cout<<"v "<<vi<<" err="<<(g-gfd).norm()<<"\n";
        }
    }

    // Hessian check

    {
        std::cout<<"\n=== Hessian check ===\n";
        double eps=1e-6;

        for(int vi=0;vi<x.size();++vi){
            Mat33 H=compute_local_hessian(vi,ref_mesh,M,adj,pins,params,x);
            Mat33 Hfd=local_hessian_fd(vi,ref_mesh,M,adj,pins,params,x,xhat,eps);

            std::cout<<"v "<<vi<<" err="<<(H-Hfd).lpNorm<Eigen::Infinity>()<<"\n";
        }
    }

    // slope-2

    slope2_check(ref_mesh,M,pins,params,x,xhat,0);

    return 0;
}
