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
        if(dat_detail::aabb_separated(x,split,p.query_radius))return;
        int status=split==1?dat_detail::vertex_triangle(x,separation):dat_detail::edge_edge(x,separation);
        double limits[4];
        if(status==0)status=dat_detail::plane_factors(x,u,split,separation,p,limits);
        if(status==dat_detail::NeedsExact) {
            status=dat_detail::exact_pair_geometry<256>(x,split,separation);
            if(status==DatExactCapacity)status=dat_detail::exact_pair_geometry<1600>(x,split,separation);
            if(status==0)status=dat_detail::plane_factors(x,u,split,separation,p,limits);
        }
        if(status==DatZeroGap)throw std::domain_error("host DAT zero gap");
        require(status==0,"host DAT geometry/plane certificate failed");
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
void supporting_plane_cases() {
    const DatParameters p{.9,1,1.2};
    auto s=point_triangle();
    Vec x[4],u[4];
    for(int i=0;i<4;++i) {x[i]=pack(s.positions[i]);u[i]=pack(s.updates[i]);}
    dat_detail::PairPlane plane;
    // Residual magnitude is not a lower bound on finite-primitive distance.
    require(dat_detail::select_pair_plane(x,1,{{0,0,-100}},1,plane)==0,
        "valid oversized supporting direction rejected");
    require(!plane.outside_query,"raw residual norm incorrectly discarded an in-range pair");
    double factors[4];
    require(dat_detail::plane_factors(x,u,1,{{0,0,-100}},p,factors)==0,
        "oversized residual plane allocation failed");
    for(double factor:factors)require(std::abs(factor-.72)<1e-12,"proportional allocation changed");
    double shared_factors[4];
    require(dat_detail::plane_factors_from_plane(u,1,plane,p,shared_factors)==0,
        "shared supporting-plane allocation failed");
    for(int i=0;i<4;++i)require(shared_factors[i]==factors[i],"shared plane changed allocation arithmetic");
    require(dat_detail::select_pair_plane(x,1,{{100,0,0}},1,plane)==dat_detail::NeedsExact,
        "nonseparating normal bypassed exact retry via its large norm");
    Vec exact;
    require(dat_detail::exact_vertex_triangle<256>(x,exact)==0
        &&dat_detail::select_pair_plane(x,1,exact,1,plane)==0&&!plane.outside_query,
        "exact vertex/triangle retry did not recover the supporting plane");
    x[0].v[2]=2;
    require(dat_detail::select_pair_plane(x,1,{{0,0,-100}},1,plane)==0&&plane.outside_query,
        "certified projected gap did not discard an out-of-range pair");

    // Captured README Example 1 frame 126: the fast EE residual points into a
    // primitive although the exact finite edges are separated by ~6.16e-10.
    Scene captured;
    captured.positions={
        Vec3(-1.2499964352680171,.54358597764887895,-.43915608458931166),
        Vec3(-1.2722323852803259,.53028835202992686,-.36137318480164726),
        Vec3(-1.2499964600268660,.54358615217010631,-.43915583489559568),
        Vec3(-1.2499995411271900,.50375014954939679,-.47291205323261482)};
    const Vec3 normal(.95038202192690158,-.20115535389100869,.23729841128654272);
    for(int i=0;i<4;++i)x[i]=pack(captured.positions[i]);
    Vec ordinary;
    require(dat_detail::edge_edge(x,ordinary)==0,"captured EE no longer exercises its ordinary proposal");
    require(dat_detail::select_pair_plane(x,2,ordinary,.0075,plane)==dat_detail::NeedsExact,
        "captured non-supporting EE normal was accepted");
    require(dat_detail::exact_edge_edge<256>(x,exact)==0
        &&dat_detail::select_pair_plane(x,2,exact,.0075,plane)==0,
        "captured EE exact supporting plane was not certified");
    require(plane.right>plane.left&&!plane.outside_query,"captured positive EE gap was discarded");
    captured.positions.push_back(captured.positions[0]-.03*normal);
    captured.positions.push_back(captured.positions[2]+.03*normal);
    captured.mesh.num_positions=6;captured.mesh.tris={0,1,4,2,3,5};
    captured.updates.assign(6,Vec3::Zero());
    compare(captured,DatParameters{.9,.0075,1.2});
    for(int i=0;i<4;++i)captured.updates[i]=(i<2?1e-4:-1e-4)*normal;
    compare(captured,DatParameters{.9,.0075,1.2});
    const auto accepted=evaluate(captured,DatParameters{.9,.0075,1.2});
    std::vector<Vec3> endpoints=captured.positions;
    for(int i=0;i<6;++i)endpoints[i]+=Vec3(accepted[i].v[0],accepted[i].v[1],accepted[i].v[2]);
    const auto signed_gap=[&](const std::vector<Vec3>& points) {
        const Vec3 origin=points[0];
        return std::min((points[2]-origin).dot(normal),(points[3]-origin).dot(normal))
            -std::max((points[0]-origin).dot(normal),(points[1]-origin).dot(normal));
    };
    require(signed_gap(endpoints)>0&&signed_gap(endpoints)<signed_gap(captured.positions),
        "captured filtered EE approach did not preserve its original sides");

    // Frame 215: rounded closest points coincide, but the original binary64
    // finite segments have exact positive distance 1.67453456537080605e-17.
    const Vec positive_gap[4]={
        {{-1.2508422453585308,.30142330872441742,-.3441007037557508}},
        {{-1.249271199389888,.43384953344448823,-.36383141911477945}},
        {{-1.252873745189554,.35850525592266047,-.35936719358941788}},
        {{-1.2435860863993213,.36377791656553377,-.33712447408743129}}};
    require(dat_detail::edge_edge(positive_gap,ordinary)==dat_detail::NeedsExact
        &&dat_detail::stable_norm(ordinary)==0,"positive-gap fixture did not exercise rounded zero distance");
    require(dat_detail::exact_pair_geometry<256>(positive_gap,2,exact)==0,
        "exact positive gap was misclassified as contact");
    require(std::abs(dat_detail::stable_norm(exact)-1.67453456537080605e-17)<5e-31,
        "exact retry changed the independently computed positive gap");
    require(dat_detail::select_pair_plane(positive_gap,2,exact,.005,plane)==0&&!plane.outside_query,
        "positive rounded-zero pair did not certify its exact plane");

    for(int split:{1,2}) {
        auto touching=split==1?point_triangle():crossed();
        if(split==1)touching.positions[0].z()=0;
        else for(int i=3;i<6;++i)touching.positions[i].z()-=.4;
        const int nodes[4]={0,1,split==1?2:3,split==1?3:4};
        for(int i=0;i<4;++i)x[i]=pack(touching.positions[nodes[i]]);
        require(dat_detail::select_pair_plane(x,split,{},1,plane)==dat_detail::NeedsExact,
            "ordinary zero residual declared contact without exact retry");
        require(dat_detail::exact_pair_geometry<256>(x,split,exact)==DatZeroGap,
            "true exact contact was artificially separated");
    }
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
    Scene sliver;sliver.positions={Vec3(.5,1e-3,1e-4),Vec3::Zero(),Vec3(1,0,0),Vec3(1,1e-9,0)};
    sliver.updates.assign(4,Vec3::Zero());sliver.updates[0]=Vec3(0,0,-1e-4);
    sliver.mesh.num_positions=4;sliver.mesh.tris={1,2,3};
    compare(sliver,DatParameters{.9,5e-4,1.2});
    supporting_plane_cases();
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
