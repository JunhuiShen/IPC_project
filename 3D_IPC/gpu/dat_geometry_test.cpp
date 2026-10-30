#include "dat.cuh"
#include "../DAT.h"
#include "../broad_phase.h"
#include "../physics.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace ipc_gpu::solver;
using dat_detail::Vec;
struct Scene {RefMesh mesh;std::vector<Vec3> positions,updates;};
Vec pack(const Vec3& x){return {{x[0],x[1],x[2]}};}
void require(bool x,const char* what){if(!x)throw std::runtime_error(what);}
std::vector<Vec> evaluate(const Scene& s,const DatParameters& p) {
    const double limit=(.5*p.gamma)*p.query_radius,radius=p.query_radius*p.node_box_padding;
    std::vector<Vec> updates;std::vector<AABB> boxes;std::vector<double> factors(s.positions.size(),1);
    for(std::size_t i=0;i<s.positions.size();++i) {
        updates.push_back(dat_detail::capped_update(pack(s.updates[i]),limit));
        boxes.emplace_back(s.positions[i]-Vec3::Constant(radius),s.positions[i]+Vec3::Constant(radius));
    }
    if(updates.empty())return updates;
    BroadPhase phase;
    if(s.mesh.tets.empty()&&s.mesh.tet_nodes.empty())phase.initialize(boxes,s.mesh,0,BroadPhase::InitializationMode::DeformableSolver);
    else phase.initialize_surface_nodes(boxes,s.mesh,0,BroadPhase::InitializationMode::DeformableSolver);
    const auto pair=[&](const int nodes[4],int split) {
        Vec x[4],u[4],separation;for(int i=0;i<4;++i){x[i]=pack(s.positions[nodes[i]]);u[i]=updates[nodes[i]];}
        int status=split==1?dat_detail::vertex_triangle(x,separation):dat_detail::edge_edge(x,separation);
        if(status==dat_detail::NeedsExact) {
            status=dat_detail::exact_edge_edge<256>(x,separation);
            if(status==DatExactCapacity)status=dat_detail::exact_edge_edge<1600>(x,separation);
        }
        require(status==0,"host DAT geometry failed");
        double limits[4];status=dat_detail::plane_factors(x,u,split,separation,p,limits);
        if(status==DatZeroGap)throw std::domain_error("host DAT zero gap");
        require(status==0,"host DAT plane failed");
        for(int i=0;i<4;++i)factors[nodes[i]]=std::min(factors[nodes[i]],limits[i]);
    };
    for(const auto& candidate:phase.cache().nt_pairs) {
        const int nodes[4]={candidate.node,candidate.tri_v[0],candidate.tri_v[1],candidate.tri_v[2]};pair(nodes,1);
    }
    for(const auto& candidate:phase.cache().ss_pairs)pair(candidate.v,2);
    for(std::size_t i=0;i<updates.size();++i)updates[i]=dat_detail::multiply(updates[i],factors[i]);
    return updates;
}
std::size_t cases=0;
void compare(const Scene& s,DatParameters p,double units=1) {
    const auto expected=planar_DAT(p.gamma,p.query_radius,s.mesh,s.updates,s.positions,p.node_box_padding);
    const auto actual=evaluate(s,p);require(actual.size()==expected.size(),"host DAT size mismatch");
    for(std::size_t i=0;i<actual.size();++i)for(int k=0;k<3;++k) {
        const double x=actual[i].v[k]/units,y=expected[i][k]/units;
        require(std::isfinite(x)&&std::abs(x-y)<=2e-11*std::max(1.,std::abs(y)),"host DAT differs from CPU oracle");
    }
    ++cases;
}
Scene point_triangle() {
    Scene s;s.positions={Vec3(0,0,.4),Vec3(-3,-3,0),Vec3(3,-3,0),Vec3(0,3,0)};
    s.updates={Vec3(0,0,-.4),Vec3(0,0,.1),Vec3(0,0,.1),Vec3(0,0,.1)};
    s.mesh.num_positions=4;s.mesh.tris={1,2,3};return s;
}
Scene crossed() {
    Scene s;s.positions={Vec3(-10,0,0),Vec3(10,0,0),Vec3(0,-10,-10),Vec3(0,-10,.4),Vec3(0,10,.4),Vec3(10,0,10.4)};
    s.updates={Vec3(0,0,.4),Vec3(0,0,.4),Vec3(0,0,.4),Vec3(0,0,-.1),Vec3(0,0,-.1),Vec3(0,0,-.1)};
    s.mesh.num_positions=6;s.mesh.tris={0,1,2,3,4,5};return s;
}
void run() {
    const DatParameters p{.9,1,1.2};compare({},p);
    Scene cap;cap.positions={Vec3::Zero()};cap.updates={Vec3(1e308,-1e308,1e308)};cap.mesh.num_positions=1;compare(cap,p);
    for(double scale:{1.,1e-100,1e80})for(auto s:{point_triangle(),crossed()}) {
        for(auto& x:s.positions)x*=scale;for(auto& u:s.updates)u*=scale;
        auto q=p;q.query_radius*=scale;compare(s,q,scale);
    }
    auto parallel=crossed();parallel.positions[3]=Vec3(-10,0,.4);parallel.positions[4]=Vec3(10,0,.4);parallel.positions[5]=Vec3(0,10,10.4);compare(parallel,p);
    auto wide=crossed();
    for(auto& x:wide.positions){x.x()*=1e100;x.y()*=1e100;if(std::abs(x.z())>1)x.z()*=1e100;else x.z()*=1e-100;}
    for(auto& u:wide.updates)u*=1e-100;
    Vec exact_points[4];const int nodes[4]={0,1,3,4};for(int i=0;i<4;++i)exact_points[i]=pack(wide.positions[nodes[i]]);
    Vec separation;require(dat_detail::exact_edge_edge<256>(exact_points,separation)==DatExactCapacity,"wide fixture must exhaust small exact limbs");
    compare(wide,DatParameters{.9,1e-99,1.2},1e-100);
    auto translated=parallel;for(auto& x:translated.positions)x+=Vec3(1e12,-1e12,1e12);compare(translated,p);
    auto subulp=point_triangle();const double world=1e12,ulp=std::nextafter(world,INFINITY)-world;
    subulp.positions={Vec3(world+100*ulp,0,0),Vec3(world,-3,-3),Vec3(world,3,-3),Vec3(world,0,3)};
    subulp.updates[0]=Vec3(-1e-5,.2,0);for(int i=1;i<4;++i)subulp.updates[i]=Vec3((world+.4-world)-2e-5,0,0);compare(subulp,p);
    auto rounded=subulp;for(std::size_t i=0;i<rounded.positions.size();++i)rounded.updates[i]=(rounded.positions[i]+rounded.updates[i])-rounded.positions[i];
    const auto direct=planar_DAT(p.gamma,p.query_radius,subulp.mesh,subulp.updates,subulp.positions,p.node_box_padding);
    const auto wrong=planar_DAT(p.gamma,p.query_radius,rounded.mesh,rounded.updates,rounded.positions,p.node_box_padding);
    require(std::abs(direct[0].y()-wrong[0].y())>1e-8,"sub-ULP fixture cannot distinguish endpoint subtraction");
    for(int i=0;i<64;++i) {
        auto s=point_triangle();s.positions[0].x()=.0125*i;s.positions[0].z()=.01+.012*i;
        s.updates[0]=Vec3(.002*i,.001*i,-.025-.008*i);
        for(int j=1;j<4;++j)s.updates[j]=Vec3(.001*j,-.001*j,.005*j);
        compare(s,p);
    }
    Scene obtuse;obtuse.positions={Vec3(.5,-.01,.1),Vec3::Zero(),Vec3(1,0,0),Vec3(-10,.1,0)};
    obtuse.updates.assign(4,Vec3::Zero());obtuse.updates[0]=Vec3(0,.05,-.2);obtuse.mesh.num_positions=4;obtuse.mesh.tris={1,2,3};compare(obtuse,DatParameters{.9,.3,1.2});
    for(double padding:{1.,1.2,3.}) {auto q=p;q.node_box_padding=padding;compare(point_triangle(),q);}
    for(int split:{1,2}) {
        auto s=split==1?point_triangle():crossed();
        if(split==1)s.positions[0].z()=0;
        else for(int j=3;j<6;++j)s.positions[j].z()-=.4;
        bool cpu=false,host=false;try{planar_DAT(p.gamma,p.query_radius,s.mesh,s.updates,s.positions,p.node_box_padding);}catch(const std::domain_error&){cpu=true;}
        try{evaluate(s,p);}catch(const std::domain_error&){host=true;}
        require(cpu&&host,"CPU/host DAT must both reject zero gap");
    }
}
} // namespace
int main() {
    try{run();std::cout<<"Host DAT geometry: "<<cases<<" independent CPU-oracle cases, exact EE and zero-gap checks PASS\n";return 0;}
    catch(const std::exception& error){std::cerr<<"Host DAT geometry failed: "<<error.what()<<'\n';return 1;}
}
