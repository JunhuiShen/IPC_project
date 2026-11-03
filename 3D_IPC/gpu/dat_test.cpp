#include "dat.h"
#include "dat.cuh"
#include "device_memory.h"
#include "rigid.cuh"
#include "../DAT.h"
#include "../physics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ipc_gpu::solver;
using Value=std::array<double,3>;
struct Scene {RefMesh mesh;std::vector<Vec3> positions,updates;};
void require(bool value,const char* what){if(!value)throw std::runtime_error(what);}
template<class F> void invalid(F&& f) {
    bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"expected invalid-input rejection");
}
std::vector<Value> packed(const std::vector<Vec3>& values) {
    std::vector<Value> out;for(const auto& p:values)out.push_back({p[0],p[1],p[2]});return out;
}
void throw_status(int status) {
    if(status==DatInvalidInput)throw std::invalid_argument("DAT device invalid input");
    if(status==DatZeroGap||status==DatUnresolvedGeometry)throw std::domain_error("DAT device unresolved/zero-gap geometry");
    if(status)throw std::runtime_error("DAT unexpected device status "+std::to_string(status));
}
std::vector<Value> gpu_filter(const Scene& s,const DatParameters& p,GpuDatFilter& filter,
    GpuBroadPhase& phase,cudaStream_t stream=nullptr,bool malformed_candidate=false) {
    DeviceArray<Value> positions(packed(s.positions)),updates(packed(s.updates)),accepted(s.positions.size());
    DeviceArray<int> error(1);error.clear();
    std::vector<int> owners(s.positions.size());std::iota(owners.begin(),owners.end(),0);
    std::vector<BpTriangle> triangles;std::set<std::array<int,2>> unique_edges;
    for(std::size_t i=0;i<s.mesh.tris.size();i+=3) {
        BpTriangle triangle{{s.mesh.tris[i],s.mesh.tris[i+1],s.mesh.tris[i+2]}};triangles.push_back(triangle);
        for(int j=0;j<3;++j) {
            int a=triangle.vertices[j],b=triangle.vertices[(j+1)%3];
            if(a>b)std::swap(a,b);unique_edges.insert({a,b});
        }
    }
    std::vector<BpEdge> edges;for(const auto& pair:unique_edges)edges.push_back({{pair[0],pair[1]}});
    std::vector<unsigned char> surface(s.positions.size(),1);
    for(int node:s.mesh.tet_nodes)surface[node]=0;
    for(int node:s.mesh.surface_nodes)surface[node]=1;
    DeviceArray<int> device_owners(owners);DeviceArray<unsigned char> device_surface(surface);
    DeviceArray<BpTriangle> device_triangles(triangles);DeviceArray<BpEdge> device_edges(edges);
    const auto* boxes=filter.prepare(s.positions.size(),reinterpret_cast<const double(*)[3]>(positions.data()),
        reinterpret_cast<const double(*)[3]>(updates.data()),p,error.data(),stream);
    cuda_checked(cudaStreamSynchronize(stream));throw_status(error.download()[0]);
    BroadPhaseTopology topology;topology.vertex_count=topology.block_count=static_cast<int>(s.positions.size());
    topology.triangle_count=static_cast<int>(triangles.size());topology.edge_count=static_cast<int>(edges.size());
    topology.vertex_block=device_owners.data();topology.surface_query=device_surface.data();
    topology.triangles=device_triangles.data();topology.edges=device_edges.data();
    topology.include_triangle_dependencies=false;topology.protect_linear_gap=false;
    auto candidates=phase.rebuild(topology,boxes,0.0,BroadPhaseMode::CandidatesOnly);
    DeviceArray<BpContact> invalid_pair;
    if(malformed_candidate) {
        invalid_pair.upload(std::vector<BpContact>{{{0,1,2,INT_MAX},0}});
        candidates.contacts=invalid_pair.data();candidates.contact_count=1;
    }
    filter.filter(candidates,reinterpret_cast<double(*)[3]>(accepted.data()),error.data(),stream);
    cuda_checked(cudaStreamSynchronize(stream));const int status=error.download()[0];
    require(positions.download()==packed(s.positions),"DAT modified frozen positions");
    require(updates.download()==packed(s.updates),"DAT modified raw increments");
    throw_status(status);return accepted.download();
}
std::size_t checked_cases=0;
void compare(const Scene& s,GpuDatFilter& filter,GpuBroadPhase& phase,
    DatParameters p={},double units=1,cudaStream_t stream=nullptr) {
    const auto expected=planar_DAT(p.gamma,p.query_radius,s.mesh,s.updates,s.positions,p.node_box_padding);
    const auto actual=gpu_filter(s,p,filter,phase,stream);
    require(actual.size()==expected.size(),"DAT result size differs from CPU");
    for(std::size_t i=0;i<actual.size();++i)for(int k=0;k<3;++k) {
        const double a=actual[i][k]/units,e=expected[i][k]/units;
        require(std::isfinite(a)&&std::abs(a-e)<=2e-11*std::max(1.0,std::abs(e)),"DAT increment differs from CPU oracle");
    }
    ++checked_cases;
}
Scene point_triangle(double gap=.4) {
    Scene s;s.positions={Vec3(0,0,gap),Vec3(-3,-3,0),Vec3(3,-3,0),Vec3(0,3,0)};
    s.updates.assign(4,Vec3::Zero());s.mesh.num_positions=4;s.mesh.tris={1,2,3};return s;
}
Scene crossed_edges(double gap=.4) {
    Scene s;s.positions={Vec3(-10,0,0),Vec3(10,0,0),Vec3(0,-10,-10),
        Vec3(0,-10,gap),Vec3(0,10,gap),Vec3(10,0,10+gap)};
    s.updates={Vec3(0,0,.4),Vec3(0,0,.4),Vec3(0,0,.4),Vec3(0,0,-.1),Vec3(0,0,-.1),Vec3(0,0,-.1)};
    s.mesh.num_positions=6;s.mesh.tris={0,1,2,3,4,5};return s;
}
Scene shared_point() {
    auto s=point_triangle();for(int i=1;i<4;++i)s.positions[i].z()=.2;
    s.positions.insert(s.positions.end(),{Vec3(-3,-3,.1),Vec3(3,-3,.1),Vec3(0,3,.1)});
    s.updates.assign(7,Vec3::Zero());s.updates[0]=Vec3(0,0,-.4);
    s.mesh.num_positions=7;s.mesh.tris={1,2,3,4,5,6};return s;
}
void component_cases() {
    GpuDatFilter filter;GpuBroadPhase phase;DatParameters p{.9,1,1.2};
    Scene cap;cap.positions={Vec3::Zero(),Vec3(1,0,0),Vec3(2,0,0)};
    cap.updates={Vec3(3,4,0),Vec3(.1,-.2,.1),Vec3::Zero()};cap.mesh.num_positions=3;compare(cap,filter,phase,p);
    cap.updates[0]=Vec3(1e308,-1e308,1e308);compare(cap,filter,phase,p);
    auto point=point_triangle();point.updates[0]=Vec3(0,0,-.4);
    for(int i=1;i<4;++i)point.updates[i]=Vec3(0,0,.1);
    compare(point,filter,phase,p);
    for(bool face_moves:{false,true}) {
        auto s=point_triangle();
        if(face_moves)for(int i=1;i<4;++i)s.updates[i]=Vec3(0,0,.4);
        else s.updates[0]=Vec3(0,0,-.4);
        compare(s,filter,phase,p);
    }
    for(bool separating:{false,true}) {
        auto s=point_triangle();s.updates[0]=Vec3(.1,.2,separating?.1:0);
        for(int i=1;i<4;++i)s.updates[i]=Vec3(-.2,.1,separating?-.1:0);
        compare(s,filter,phase,p);
    }
    auto maximum=point_triangle();maximum.updates[0]=Vec3(0,0,-.2);maximum.updates[1]=Vec3(0,0,.4);compare(maximum,filter,phase,p);
    auto capped=point;capped.updates[0]=Vec3(0,0,-4);compare(capped,filter,phase,p);
    for(double gap:{1.0,1.5}) {auto s=point_triangle(gap);s.updates[0]=Vec3(0,0,-.4);compare(s,filter,phase,p);}
    for(double scale:{1e-100,1e80}) {
        auto s=point;for(auto& x:s.positions)x*=scale;for(auto& u:s.updates)u*=scale;
        auto scaled=p;scaled.query_radius*=scale;compare(s,filter,phase,scaled,scale);
    }
    for(double scale:{1.0,1e-100,1e80}) {
        auto s=crossed_edges();for(auto& x:s.positions)x*=scale;for(auto& u:s.updates)u*=scale;
        auto scaled=p;scaled.query_radius*=scale;compare(s,filter,phase,scaled,scale);
    }
    auto wide=crossed_edges(4e-101);
    for(auto& x:wide.positions){x.x()*=1e100;x.y()*=1e100;if(std::abs(x.z())>1)x.z()*=1e100;}
    for(auto& u:wide.updates)u*=1e-100;
    dat_detail::Vec exact_points[4];const int wide_nodes[4]={0,1,3,4};
    for(int i=0;i<4;++i)exact_points[i]=dat_detail::load(wide.positions[wide_nodes[i]].data());
    dat_detail::Vec residual;
    require(dat_detail::exact_edge_edge<256>(exact_points,residual)==DatExactCapacity,
        "wide-exponent fixture must require the large exact retry");
    compare(wide,filter,phase,DatParameters{.9,1e-99,1.2},1e-100);
    auto parallel=crossed_edges();parallel.positions[3]=Vec3(-10,0,.4);parallel.positions[4]=Vec3(10,0,.4);parallel.positions[5]=Vec3(0,10,10.4);compare(parallel,filter,phase,p);
    auto translated=parallel;for(auto& x:translated.positions)x+=Vec3(1e12,-1e12,1e12);compare(translated,filter,phase,p);
    auto subulp=point_triangle();const double world=1e12,ulp=std::nextafter(world,INFINITY)-world;
    subulp.positions={Vec3(world+100*ulp,0,0),Vec3(world,-3,-3),Vec3(world,3,-3),Vec3(world,0,3)};
    subulp.updates[0]=Vec3(-1e-5,.2,0);
    for(int i=1;i<4;++i)subulp.updates[i]=Vec3((world+.4-world)-2e-5,0,0);
    compare(subulp,filter,phase,p);
    auto rounded=subulp;for(std::size_t i=0;i<rounded.positions.size();++i)rounded.updates[i]=(rounded.positions[i]+rounded.updates[i])-rounded.positions[i];
    const auto direct_expected=planar_DAT(p.gamma,p.query_radius,subulp.mesh,subulp.updates,subulp.positions,p.node_box_padding);
    const auto rounded_wrong=planar_DAT(p.gamma,p.query_radius,rounded.mesh,rounded.updates,rounded.positions,p.node_box_padding);
    require(std::abs(direct_expected[0].y()-rounded_wrong[0].y())>1e-8,"sub-ULP fixture must detect endpoint-minus-position loss");
    auto shared=shared_point();const auto original=gpu_filter(shared,p,filter,phase);compare(shared,filter,phase,p);
    shared.mesh.tris={4,5,6,1,2,3};const auto permuted=gpu_filter(shared,p,filter,phase);require(original==permuted,"DAT minimum depends on triangle order");
    for(double padding:{1.0,1.2,3.0}) {auto q=p;q.node_box_padding=padding;compare(point,filter,phase,q);}
    Scene sliver;sliver.positions={Vec3(.9,2e-8,1e-4),Vec3::Zero(),Vec3(1,0,0),Vec3(1,1e-8,0)};
    sliver.updates.assign(4,Vec3::Zero());sliver.updates[0]=Vec3(0,0,-1e-4);sliver.mesh.num_positions=4;sliver.mesh.tris={1,2,3};
    compare(sliver,filter,phase,DatParameters{.9,5e-4,1.2});
    Scene obtuse;obtuse.positions={Vec3(.5,-.01,.1),Vec3::Zero(),Vec3(1,0,0),Vec3(-10,.1,0)};
    obtuse.updates.assign(4,Vec3::Zero());obtuse.updates[0]=Vec3(0,.05,-.2);obtuse.mesh.num_positions=4;obtuse.mesh.tris={1,2,3};
    compare(obtuse,filter,phase,DatParameters{.9,.3,1.2});
    Scene solid;solid.positions={Vec3(-3,-3,0),Vec3(3,-3,0),Vec3(0,3,0),Vec3(0,0,3),Vec3(0,0,.1),Vec3(0,0,-.2)};
    solid.updates.assign(6,Vec3::Zero());solid.updates[4]=Vec3(0,0,-.095);solid.updates[5]=Vec3(0,0,.4);
    solid.mesh.num_positions=6;solid.mesh.tris={0,2,1,0,1,3,1,2,3,2,0,3};solid.mesh.tets={4,0,2,1,4,0,1,3,4,1,2,3,4,2,0,3};
    solid.mesh.tet_nodes={0,1,2,3,4};solid.mesh.surface_nodes={0,1,2,3};compare(solid,filter,phase,p);
    Scene large;
    for(int copy=0;copy<48;++copy) {
        const auto part=copy%2?crossed_edges():shared_point();const int base=static_cast<int>(large.positions.size());
        for(const auto& x:part.positions)large.positions.push_back(x+Vec3(30*copy,0,0));
        large.updates.insert(large.updates.end(),part.updates.begin(),part.updates.end());
        for(int node:part.mesh.tris)large.mesh.tris.push_back(base+node);
    }
    large.mesh.num_positions=large.positions.size();compare(large,filter,phase,p);
    const auto first=gpu_filter(large,p,filter,phase);for(int repeat=0;repeat<3;++repeat)require(gpu_filter(large,p,filter,phase)==first,"DAT GPU order/rebuild is nondeterministic");
    cudaStream_t stream=nullptr;cuda_checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    try {compare(point,filter,phase,p,1,stream);}catch(...){cudaStreamDestroy(stream);throw;}
    cuda_checked(cudaStreamDestroy(stream));
    compare(Scene{},filter,phase,p);compare(point,filter,phase,p);compare(cap,filter,phase,p);compare(Scene{},filter,phase,p);
    for(auto s:{point_triangle(0),crossed_edges(0)}) {
        bool cpu=false,gpu=false;try{planar_DAT(p.gamma,p.query_radius,s.mesh,s.updates,s.positions,p.node_box_padding);}catch(const std::domain_error&){cpu=true;}
        try{gpu_filter(s,p,filter,phase);}catch(const std::domain_error&){gpu=true;}
        require(cpu&&gpu,"zero-gap primitives must fail closed on CPU and GPU");
    }
    auto nonfinite=cap;nonfinite.updates[0].x()=std::numeric_limits<double>::infinity();
    invalid([&]{gpu_filter(nonfinite,p,filter,phase);});
    auto overflow=point;for(auto& x:overflow.positions)x=Vec3::Constant(DBL_MAX);
    auto huge=p;huge.query_radius=1e308;huge.node_box_padding=1;invalid([&]{gpu_filter(overflow,huge,filter,phase);});
    invalid([&]{gpu_filter(point,p,filter,phase,nullptr,true);});
    DeviceArray<int> status(1);status.clear();
    for(double gamma:{0.,1.,-.1,std::numeric_limits<double>::quiet_NaN()}) {auto q=p;q.gamma=gamma;invalid([&]{filter.prepare(0,nullptr,nullptr,q,status.data());});}
    for(double radius:{0.,-1.,std::numeric_limits<double>::infinity()}) {auto q=p;q.query_radius=radius;invalid([&]{filter.prepare(0,nullptr,nullptr,q,status.data());});}
    for(double padding:{0.,.9,std::numeric_limits<double>::infinity()}) {auto q=p;q.node_box_padding=padding;invalid([&]{filter.prepare(0,nullptr,nullptr,q,status.data());});}
    invalid([&]{filter.filter({},nullptr,status.data());});
    DeviceArray<Value> positions(packed(point.positions)),raw(packed(point.updates)),accepted(point.positions.size());
    filter.prepare(point.positions.size(),reinterpret_cast<const double(*)[3]>(positions.data()),reinterpret_cast<const double(*)[3]>(raw.data()),p,status.data());
    cuda_checked(cudaDeviceSynchronize());
    invalid([&]{filter.filter({},reinterpret_cast<double(*)[3]>(positions.data()),status.data());});
    invalid([&]{filter.filter({},reinterpret_cast<double(*)[3]>(raw.data()),status.data());});
    filter.filter({},reinterpret_cast<double(*)[3]>(accepted.data()),status.data());cuda_checked(cudaDeviceSynchronize());throw_status(status.download()[0]);
    status.upload({42});filter.prepare(point.positions.size(),reinterpret_cast<const double(*)[3]>(positions.data()),reinterpret_cast<const double(*)[3]>(raw.data()),p,status.data());
    filter.filter({},reinterpret_cast<double(*)[3]>(accepted.data()),status.data());cuda_checked(cudaDeviceSynchronize());
    require(status.download()[0]==42,"DAT cleared an existing nonzero status");
}

using Quat=std::array<double,4>;
struct MixedScene {
    std::vector<Value> positions,updates,reference,com,omega,dc,dw;
    std::vector<Quat> q0;
    std::vector<int> owner;
    std::vector<unsigned char> translate,rotate;
    std::vector<BpTriangle> triangles;
};
struct MixedOutput {
    std::vector<Value> positions,com,omega;
    std::vector<Quat> orientation;
    int status=0;
};
MixedOutput mixed_filter(MixedScene scene,DatParameters parameters,bool duplicate=false,
    int initial_status=0) {
    const auto count=scene.positions.size(),bodies=scene.com.size();
    require(scene.owner.size()==count&&scene.reference.size()==count,"invalid mixed test fixture");
    if(scene.updates.empty())scene.updates.resize(count);
    std::vector<int> nodes,blocks(count),body_blocks(bodies);
    std::vector<std::size_t> offsets{0};int next=0;
    for(std::size_t node=0;node<count;++node)if(scene.owner[node]<0)blocks[node]=next++;
    for(std::size_t body=0;body<bodies;++body) {
        body_blocks[body]=next++;
        for(std::size_t node=0;node<count;++node)if(scene.owner[node]==int(body)) {
            nodes.push_back(int(node));blocks[node]=body_blocks[body];
        }
        offsets.push_back(nodes.size());
    }
    if(duplicate&&nodes.size()>1)nodes[1]=nodes[0];
    std::set<std::array<int,2>> unique_edges;
    for(const auto& triangle:scene.triangles)for(int i=0;i<3;++i) {
        int a=triangle.vertices[i],b=triangle.vertices[(i+1)%3];
        if(a>b)std::swap(a,b);unique_edges.insert({a,b});
    }
    std::vector<BpEdge> edges;for(const auto& edge:unique_edges)edges.push_back({{edge[0],edge[1]}});
    DeviceArray<Value> positions(scene.positions),raw(scene.updates),reference(scene.reference);
    DeviceArray<Value> com(scene.com),omega(scene.omega),dc(scene.dc),dw(scene.dw);
    DeviceArray<Quat> q0(scene.q0);
    DeviceArray<int> owner(scene.owner),device_nodes(nodes),device_blocks(blocks),device_body_blocks(body_blocks);
    DeviceArray<std::size_t> device_offsets(offsets);
    DeviceArray<unsigned char> translate(scene.translate),rotate(scene.rotate);
    DeviceArray<BpTriangle> triangles(scene.triangles);DeviceArray<BpEdge> device_edges(edges);
    DeviceArray<Value> accepted(count),accepted_com(bodies),accepted_omega(bodies);
    DeviceArray<Quat> accepted_q(bodies);DeviceArray<int> status(std::vector<int>{initial_status});
    const RigidScene rigid{bodies,device_body_blocks.data(),owner.data(),
        reinterpret_cast<const double(*)[3]>(reference.data()),device_offsets.data(),device_nodes.data(),
        nullptr,nullptr,translate.data(),rotate.data()};
    RigidState state{};state.com=reinterpret_cast<double(*)[3]>(com.data());
    state.omega=reinterpret_cast<double(*)[3]>(omega.data());
    state.orientation=state.previous_orientation=reinterpret_cast<double(*)[4]>(q0.data());
    GpuDatFilter filter;GpuBroadPhase phase;
    const auto* boxes=filter.prepare(count,reinterpret_cast<const double(*)[3]>(positions.data()),
        reinterpret_cast<const double(*)[3]>(raw.data()),rigid,state,
        reinterpret_cast<const double(*)[3]>(dc.data()),reinterpret_cast<const double(*)[3]>(dw.data()),
        1,parameters,status.data());
    cuda_checked(cudaDeviceSynchronize());
    MixedOutput out;out.status=status.download()[0];
    if(out.status)return out;
    BroadPhaseTopology topology;topology.vertex_count=int(count);topology.block_count=next;
    topology.vertex_block=device_blocks.data();topology.triangle_count=int(scene.triangles.size());
    topology.triangles=triangles.data();topology.edge_count=int(edges.size());topology.edges=device_edges.data();
    topology.include_triangle_dependencies=false;topology.protect_linear_gap=false;
    const auto candidates=phase.rebuild(topology,boxes,0,BroadPhaseMode::CandidatesOnly);
    invalid([&]{filter.filter(candidates,reinterpret_cast<double(*)[3]>(positions.data()),
        reinterpret_cast<double(*)[3]>(accepted_com.data()),reinterpret_cast<double(*)[3]>(accepted_omega.data()),
        reinterpret_cast<double(*)[4]>(accepted_q.data()),status.data());});
    filter.filter(candidates,reinterpret_cast<double(*)[3]>(accepted.data()),
        reinterpret_cast<double(*)[3]>(accepted_com.data()),reinterpret_cast<double(*)[3]>(accepted_omega.data()),
        reinterpret_cast<double(*)[4]>(accepted_q.data()),status.data());
    cuda_checked(cudaDeviceSynchronize());out.status=status.download()[0];
    require(positions.download()==scene.positions&&raw.download()==scene.updates,
        "mixed DAT modified frozen node input");
    require(com.download()==scene.com&&omega.download()==scene.omega&&q0.download()==scene.q0
        &&dc.download()==scene.dc&&dw.download()==scene.dw,"mixed DAT modified frozen rigid input");
    if(!out.status) {
        out.positions=accepted.download();out.com=accepted_com.download();
        out.omega=accepted_omega.download();out.orientation=accepted_q.download();
    }
    return out;
}
void place_rigid_nodes(MixedScene& scene) {
    for(std::size_t node=0;node<scene.positions.size();++node) {
        const int body=scene.owner[node];if(body<0||std::size_t(body)>=scene.com.size())continue;
        double q[4];rigid_orientation(scene.q0[body].data(),scene.omega[body].data(),1,q);
        rigid_world_position(scene.reference[node].data(),scene.com[body].data(),q,scene.positions[node].data());
    }
}
void mixed_component_cases() {
    const DatParameters parameters{.9,10,1.2};
    MixedScene curved;
    curved.positions={{},{},{-3,-3,0},{3,-3,0},{0,3,0}};
    curved.reference={{1,0,.3},{0,1,.3},{},{},{}};
    curved.owner={0,0,-1,-1,-1};curved.com={{0,0,0}};curved.omega={{0,0,0}};
    curved.dc={{.15,0,0}};curved.dw={{0,2*std::acos(-1.),0}};curved.q0={{1,0,0,0}};
    curved.translate={1};curved.rotate={1};curved.triangles={{{2,3,4}}};place_rigid_nodes(curved);
    // One full revolution has a safe endpoint but crosses the stationary face
    // in between. A chord-only or endpoint-only rigid filter accepts it.
    double full_q[4],full_point[3];rigid_orientation(curved.q0[0].data(),curved.dw[0].data(),1,full_q);
    rigid_world_position(curved.reference[0].data(),curved.dc[0].data(),full_q,full_point);
    require(full_point[2]>.29,"hidden rigid crossing fixture endpoint is not safe");
    const auto result=mixed_filter(curved,parameters);require(result.status==0,"curved mixed DAT failed");
    const double alpha=result.omega[0][1]/curved.dw[0][1];
    require(alpha>0&&alpha<.1,"rigid DAT missed hidden full-turn plane crossing");
    require(std::abs(result.com[0][0]/curved.dc[0][0]-alpha)<1e-12,
        "rigid DAT used different COM and angular factors");
    for(int sample=0;sample<=1000;++sample) {
        const double t=alpha*sample/1000;double c[3],w[3],q[4],point[3];
        for(int k=0;k<3;++k){c[k]=curved.com[0][k]+t*curved.dc[0][k];w[k]=t*curved.dw[0][k];}
        rigid_orientation(curved.q0[0].data(),w,1,q);
        for(int node:{0,1}) {
            rigid_world_position(curved.reference[node].data(),c,q,point);
            require(point[2]>0,"accepted rigid curve crosses stationary face");
            double squared=0;for(int k=0;k<3;++k)squared+=std::pow(point[k]-curved.positions[node][k],2);
            require(std::sqrt(squared)<=.5*parameters.gamma*parameters.query_radius,
                "accepted rigid path exceeds DAT query sphere");
        }
    }
    for(int node:{0,1}) {
        double point[3];rigid_world_position(curved.reference[node].data(),result.com[0].data(),
            result.orientation[0].data(),point);
        for(int k=0;k<3;++k)require(std::abs(point[k]-result.positions[node][k])<2e-14,
            "accepted rigid proxy disagrees with accepted pose");
    }
    // A far-pair radius limit must see the complete loop even without contacts.
    auto loop=curved;loop.positions.resize(2);loop.reference.resize(2);loop.owner.resize(2);loop.triangles.clear();
    loop.dc[0]={0,0,0};const DatParameters sphere{.9,1,1.2};
    const auto radial=mixed_filter(loop,sphere);require(radial.status==0,"curved rigid radius DAT failed");
    const double radial_alpha=radial.omega[0][1]/loop.dw[0][1];
    require(radial_alpha>0&&radial_alpha<.2,"query sphere missed a hidden full loop");
    // Two independently moving bodies must share the same division plane.
    MixedScene bodies; bodies.positions={{},{},{},{},{},{}};
    bodies.reference={{-.3,0,0},{-.3,1,0},{-.3,0,1},{.3,-2,-2},{.3,2,-2},{.3,0,2}};
    bodies.owner={0,0,0,1,1,1};bodies.com={{0,0,0},{0,0,0}};
    bodies.omega={{0,0,0},{0,0,0}};bodies.dc={{.5,0,0},{-.5,0,0}};bodies.dw={{0,0,0},{0,0,0}};
    bodies.q0={{1,0,0,0},{1,0,0,0}};bodies.translate={1,1};bodies.rotate={0,0};
    bodies.triangles={{{0,1,2}},{{3,4,5}}};place_rigid_nodes(bodies);
    const auto body_result=mixed_filter(bodies,DatParameters{.9,2,1.2});
    require(body_result.status==0,"body/body DAT failed");
    require(body_result.com[0][0]>0&&body_result.com[1][0]<0
        &&body_result.positions[0][0]<body_result.positions[3][0],"body/body DAT failed separation");
    for(int node:{1,2})require(std::abs((body_result.positions[node][0]-bodies.positions[node][0])
        -(body_result.positions[0][0]-bodies.positions[0][0]))<1e-14,"body vertices used independent factors");
    // Full same-body primitives are culled, including zero-gap self contact.
    MixedScene self;self.positions.resize(4);self.reference={{0,0,0},{-1,-1,0},{1,-1,0},{0,1,0}};
    self.owner={0,0,0,0};self.com={{0,0,0}};self.omega={{0,0,0}};
    self.dc={{.1,0,0}};self.dw={{0,0,.2}};self.q0={{1,0,0,0}};
    self.translate={1};self.rotate={1};self.triangles={{{1,2,3}}};place_rigid_nodes(self);
    const auto self_result=mixed_filter(self,parameters);require(self_result.status==0,"same-body candidates were not culled");
    require(std::abs(self_result.com[0][0]-.1)<1e-14&&std::abs(self_result.omega[0][2]-.2)<1e-14,
        "unconstrained rigid DAT unnecessarily damped a safe update");
    auto pinned=self;pinned.translate[0]=0;pinned.rotate[0]=0;pinned.dc[0]={};pinned.dw[0]={};
    const auto fixed=mixed_filter(pinned,parameters);require(fixed.status==0&&fixed.positions==pinned.positions,
        "fixed rigid modes moved in DAT");
    // The represented endpoint can move farther than its mathematical capped
    // increment after adding it to a large position. The endpoint join must
    // shorten the affected free vertex before publishing any part of the sweep.
    auto rounded=pinned;rounded.positions.push_back({1e12,0,0});rounded.reference.push_back({});
    rounded.owner.push_back(-1);rounded.updates.resize(5);rounded.updates.back()={.3,0,0};
    const DatParameters narrow{.9,2./3,1.2};
    require((rounded.positions.back()[0]+.3)-rounded.positions.back()[0]>.3,
        "rounded-endpoint fixture must exceed its raw radius cap");
    const auto rounded_result=mixed_filter(rounded,narrow);
    require(rounded_result.status==0,"rounded represented endpoint did not backtrack");
    const double represented=rounded_result.positions.back()[0]-rounded.positions.back()[0];
    require(represented>0&&represented<=.5*narrow.gamma*narrow.query_radius,
        "DAT published a represented endpoint outside its query sphere");
    // Exercise retry striding past one block while every represented .3
    // increment rounds outside its .3 sphere on the first parallel pass.
    auto many_rounded=rounded;
    for(int i=0;i<300;++i) {
        many_rounded.positions.push_back({1e12+8*(i+1),0,0});
        many_rounded.reference.push_back({});many_rounded.owner.push_back(-1);
        many_rounded.updates.push_back({.3,0,0});
    }
    const auto retried=mixed_filter(many_rounded,narrow);
    require(retried.status==0,"multi-block represented endpoint retry failed");
    for(std::size_t i=4;i<many_rounded.positions.size();++i) {
        const double delta=retried.positions[i][0]-many_rounded.positions[i][0];
        require(delta>0&&delta<=.5*narrow.gamma*narrow.query_radius,
            "strided retry published an endpoint outside its sphere");
    }
    require(mixed_filter(self,parameters,true).status==DatRigidOwnership,"duplicate rigid ownership was accepted");
    auto bad=self;bad.owner[0]=17;
    require(mixed_filter(bad,parameters).status==DatRigidOwnership,"out-of-range rigid owner was accepted");
    bad=self;bad.translate[0]=0;
    require(mixed_filter(bad,parameters).status==DatInvalidInput,"fixed translation accepted a nonzero increment");
    bad=curved;bad.positions[0][0]+=.01;
    require(mixed_filter(bad,parameters).status==DatInvalidInput,"inconsistent proxy pose was accepted");
    bad=curved;bad.owner[2]=0;
    require(mixed_filter(bad,parameters).status!=0,"mixed-owner rigid triangle was accepted");
    require(mixed_filter(self,parameters,false,42).status==42,"mixed DAT cleared an existing error");
    std::cout<<"GPU mixed DAT: hidden rotation crossings, full-path radius, shared body/body planes, represented endpoint backtracking, modes, ownership and immutable snapshots PASS\n";
}
} // namespace

int main() {
    try {
        int count=0;cuda_checked(cudaGetDeviceCount(&count));require(count>0,"GPU DAT test requires a CUDA device");
        component_cases();
        mixed_component_cases();
        std::cout<<"GPU DAT: "<<checked_cases<<" CPU-oracle cases; capped updates, NT/EE exact retries, obtuse/sliver/scale/translation, boundary/interior masks, permutation/rebuild/stream invariants, zero-gap and invalid-input rejection PASS\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<"GPU DAT test failed: "<<error.what()<<'\n';return 1;}
}
