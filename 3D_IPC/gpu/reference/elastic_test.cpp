#include "elastic.h"
#include "test_support.h"
#include "cloth_math.cuh"
#include "solid_math.cuh"
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace ipc_gpu;
using namespace ipc_gpu::solver;
using V=std::array<double,3>;
static_assert(sizeof(V)==3*sizeof(double));
static void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
static double (*triples(V* p))[3] {return reinterpret_cast<double(*)[3]>(p);}
static const double (*triples(const V* p))[3] {return reinterpret_cast<const double(*)[3]>(p);}
static void add(Derivatives& a,const Derivatives& b,double scale) {
    for(int i=0;i<3;++i)a.gradient[i]+=scale*b.gradient[i];
    for(int i=0;i<9;++i)a.hessian[i]+=scale*b.hessian[i];
}
static V proposed(const V& x,const Derivatives& system,double damping) {
    const double* h=system.hessian;
    double c[9]={h[4]*h[8]-h[5]*h[7],h[2]*h[7]-h[1]*h[8],h[1]*h[5]-h[2]*h[4],
        h[5]*h[6]-h[3]*h[8],h[0]*h[8]-h[2]*h[6],h[2]*h[3]-h[0]*h[5],
        h[3]*h[7]-h[4]*h[6],h[1]*h[6]-h[0]*h[7],h[0]*h[4]-h[1]*h[3]};
    double det=(h[0]*c[0]+h[1]*c[3])+h[2]*c[6]; V out{};
    const double inv_det=1.0/det;
    for(int a=0;a<3;++a)out[a]=x[a]-damping*(((c[3*a]*inv_det)*system.gradient[0]
        +(c[3*a+1]*inv_det)*system.gradient[1])+(c[3*a+2]*inv_det)*system.gradient[2]);
    return out;
}
int main()try {
    // Separate cloth and solid components plus a rigid proxy sentinel. The
    // arithmetic kernels already have independent scalar/SIMD reference tests;
    // this test checks device-GPU phase ordering and update/integration.
    std::vector<V> x={{0,0,0},{1.01,0,0},{0,.99,.02},{1,1,.01},
        {2,0,0},{3.01,0,0},{2,1.02,0},{2,0,.99},{7,8,9}};
    std::vector<V> velocities(x.size(),V{.01,-.03,.02}),targets=x;
    std::vector<int> node_blocks={0,1,2,3,4,5,6,7,8},colors=node_blocks;
    std::vector<unsigned char> deformable={1,1,1,1,1,1,1,1,0},solid={0,0,0,0,1,1,1,1,0},pinned={1,0,0,0,0,1,0,0,0};
    std::vector<double> masses(x.size(),1),fractions(x.size(),.9);
    std::vector<TriangleIncident> triangles;std::vector<SolidIncident> solids;std::vector<HingeIncident> hinges;
    std::vector<std::size_t> to(x.size()+1),so(x.size()+1),ho(x.size()+1);
    const int faces[2][3]={{0,1,2},{1,3,2}};
    for(std::size_t n=0;n<x.size();++n) {
        to[n]=triangles.size();so[n]=solids.size();ho[n]=hinges.size();
        for(int f=0;f<2;++f)for(int role=0;role<3;++role)if(faces[f][role]==int(n)) {
            TriangleIncident t{};t.owner=n;
            for(int a=0;a<3;++a)t.nodes[a]=faces[f][a];
            // A valid arbitrary reference metric is enough for phase tests.
            t.material.dm_inverse[0]=1;t.material.dm_inverse[3]=1;t.material.area=.5;
            t.material.shape[0]=role==0?-1:(role==1?1:0);
            t.material.shape[1]=role==0?-1:(role==2?1:0);triangles.push_back(t);
        }
        if(n<4) {
            HingeIncident h{};h.owner=n;
            const int corners[4]={1,2,0,3};
            for(int a=0;a<4;++a){h.nodes[a]=corners[a];if(corners[a]==int(n))h.material.active_node=a;}
            h.material.coefficient=1;h.material.rest_angle=0;hinges.push_back(h);
        }
        if(n>=4&&n<8) {
            SolidIncident s{};s.owner=n;
            for(int a=0;a<4;++a)s.nodes[a]=4+a;
            for(int a=0;a<3;++a){s.material.dm_inverse[3*a+a]=1;s.material.shape[a]=n==4?-1:(n==std::size_t(5+a)?1:0);}
            s.material.measure=1.0/6.0;solids.push_back(s);
        }
    }
    to.back()=triangles.size();so.back()=solids.size();ho.back()=hinges.size();
    Parameters p{};p.gravity[1]=-9.81;p.kpin=100;p.dt2=.01*.01;p.mu=2;p.lambda=3;p.bending_stiffness=.001;
    SolidParameters sp{2,3};
    GpuTestBuffer<V> dx(x),dv(velocities),dprev(x.size()),dpred(x.size()),dtargets(targets),dproposed(x.size());
    GpuTestBuffer<int> db(node_blocks),dc(colors),error(std::vector<int>{0});
    GpuTestBuffer<unsigned char> dd(deformable),ds(solid),dp(pinned);
    GpuTestBuffer<double> dm(masses),df(fractions);
    GpuTestBuffer<TriangleIncident> dt(triangles);GpuTestBuffer<SolidIncident> dst(solids);GpuTestBuffer<HingeIncident> dh(hinges);
    GpuTestBuffer<std::size_t> dto(to),dso(so),dho(ho);
    GpuTestBuffer<Point> gp(x.size());GpuTestBuffer<Triangle> gt(triangles.size());GpuTestBuffer<Solid> gs(solids.size());GpuTestBuffer<Hinge> gh(hinges.size());
    GpuTestBuffer<Derivatives> op(x.size()),ot(triangles.size()),os(solids.size()),oh(hinges.size()),totals(x.size());
    DeformableScene scene{x.size(),triangles.size(),solids.size(),hinges.size(),db.data(),dd.data(),ds.data(),dp.data(),dm.data(),triples(dtargets.data()),dt.data(),dst.data(),dh.data(),dto.data(),dso.data(),dho.data()};
    DeformableState state{triples(dx.data()),triples(dv.data()),triples(dprev.data()),triples(dpred.data())};
    ElasticWorkspace work{gp.data(),gt.data(),gs.data(),gh.data(),op.data(),ot.data(),os.data(),oh.data(),totals.data(),triples(dproposed.data()),error.data()};
    // No device->host state transfers occur in this complete 5-substep loop.
    for(int sub=0;sub<5;++sub) {
        begin_deformable_substep(scene,state,.01);
        for(int sweep=0;sweep<3;++sweep)for(int color=0;color<9;++color) {
            assemble_elastic_color(scene,state,work,dc.data(),color,p,sp);
            propose_deformable_color(scene,state,work,dc.data(),color,.7);
            commit_deformable_color(scene,state,work,dc.data(),color,df.data());
        }
        finish_deformable_substep(scene,state,.01);
    }
    gpu_cuda_check(cudaDeviceSynchronize());require(error.download()[0]==0,"Device elastic error");
    for(int sub=0;sub<5;++sub) {
        auto previous=x,predicted=x;
        for(std::size_t n=0;n<x.size();++n)if(deformable[n])for(int a=0;a<3;++a)predicted[n][a]+=.01*velocities[n][a];
        for(int sweep=0;sweep<3;++sweep)for(std::size_t n=0;n<x.size();++n)if(deformable[n]) {
            Point point{};point.mass=masses[n];point.pinned=pinned[n];point.solid=solid[n];
            for(int a=0;a<3;++a){point.position[a]=x[n][a];point.predicted[a]=predicted[n][a];point.target[a]=targets[n][a];}
            auto sum=evaluate(point,p);
            for(auto e=to[n];e<to[n+1];++e){auto t=triangles[e].material;for(int c=0;c<3;++c)for(int a=0;a<3;++a)t.positions[3*c+a]=x[triangles[e].nodes[c]][a];add(sum,evaluate(t,p),p.dt2);}
            for(auto e=so[n];e<so[n+1];++e){auto t=solids[e].material;for(int c=0;c<4;++c)for(int a=0;a<3;++a)t.positions[3*c+a]=x[solids[e].nodes[c]][a];add(sum,evaluate(t,sp),p.dt2);}
            for(auto e=ho[n];e<ho[n+1];++e){auto t=hinges[e].material;for(int c=0;c<4;++c)for(int a=0;a<3;++a)t.positions[3*c+a]=x[hinges[e].nodes[c]][a];add(sum,evaluate(t,p),p.dt2);}
            const auto target=proposed(x[n],sum,.7);
            for(int a=0;a<3;++a)x[n][a]+=.9*(target[a]-x[n][a]);
        }
        for(std::size_t n=0;n<x.size();++n)if(deformable[n])for(int a=0;a<3;++a)velocities[n][a]=(x[n][a]-previous[n][a])/.01;
    }
    const auto result=dx.download(),result_v=dv.download();double worst=0;
    for(std::size_t n=0;n<x.size();++n)for(int a=0;a<3;++a){
        require(std::isfinite(result[n][a])&&std::isfinite(result_v[n][a]),"Nonfinite GPU trajectory");
        worst=std::max(worst,std::abs(result[n][a]-x[n][a]));
        require(std::abs(result[n][a]-x[n][a])<1e-9,"GPU trajectory mismatch");
        require(std::abs(result_v[n][a]-velocities[n][a])<1e-7,"GPU velocity mismatch");
    }
    require(result.back()==x.back(),"Rigid proxy changed in deformable pipeline");
    std::cout<<"PASS GPU elastic gather/compute/reduce/solve/commit/integrate; 5 substeps, max position error="<<worst<<"\n";
    std::cout<<"Component test only: no contact/CCD or rigid solver claimed.\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
