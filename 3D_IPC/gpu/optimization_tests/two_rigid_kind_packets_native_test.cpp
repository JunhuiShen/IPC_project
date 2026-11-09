// Reuse the public solver/allocation fixture. The unchanged dense reference
// suite also runs as its own maintained test in the complete validation target.
#define main queue_reference_main
#include "two_rigid_color_queue_native_test.cpp"
#undef main
#include <numeric>

namespace {
void kind_compare_runs(const Run& a,const Run& b,const std::string& label) {
    require(a.states.size()==b.states.size(),label+" state count");
    for(std::size_t epoch=0;epoch<a.states.size();++epoch)
        equal(a.states[epoch],b.states[epoch],label+" epoch="+std::to_string(epoch));
}
SceneDescription many_points_fixture(std::size_t points) {
    auto h=fixture();
    if(points==4)return h;
    require(points>=4,"invalid point fixture extent");
    const auto fixed=std::vector<Vector3>(h.positions.end()-3,h.positions.end());
    h.positions.resize(points+3);
    for(std::size_t i=0;i<points;++i)
        h.positions[i]=Vector3{.1+.01*double(i%3),.02*double(i%5),.1+.0001*double(i%7)};
    for(std::size_t i=0;i<3;++i)h.positions[points+i]=fixed[i];
    const auto nodes=h.positions.size();
    h.velocities.assign(nodes,Vector3{});h.pin_targets=h.positions;
    h.body_reference.assign(nodes,Vector3{});
    h.deformable.assign(nodes,0);h.solid.assign(nodes,0);h.pinned.assign(nodes,0);
    h.surface.assign(nodes,1);h.mass.assign(nodes,0.0);
    h.node_body.assign(nodes,0);
    for(auto i=points;i<nodes;++i)h.node_body[i]=1;
    h.node_blocks=h.node_body;h.body_nodes.resize(nodes);
    std::iota(h.body_nodes.begin(),h.body_nodes.end(),0);
    h.body_node_offsets={0,points,nodes};
    h.body_mass={double(points),3.};
    h.com.assign(2,Vector3{});h.body_second_moment.assign(2,Matrix3{});
    h.triangle_offsets.assign(nodes+1,0);h.solid_offsets.assign(nodes+1,0);
    h.hinge_offsets.assign(nodes+1,0);
    const int a=static_cast<int>(points),b=a+1,c=a+2;
    h.surface_triangles={BpTriangle{{a,b,c}}};
    h.edges={BpEdge{{a,b}},BpEdge{{b,c}},BpEdge{{c,a}}};
    for(std::size_t body=0;body<2;++body) {
        const auto first=h.body_node_offsets[body],last=h.body_node_offsets[body+1];
        const double nodal_mass=h.body_mass[body]/double(last-first);
        for(auto i=first;i<last;++i)for(int k=0;k<3;++k)
            h.com[body][k]+=nodal_mass*h.positions[h.body_nodes[i]][k];
        for(auto& value:h.com[body])value/=h.body_mass[body];
        for(auto i=first;i<last;++i) {
            const int node=h.body_nodes[i];auto& local=h.body_reference[node];
            for(int k=0;k<3;++k)local[k]=h.positions[node][k]-h.com[body][k];
            for(int r=0;r<3;++r)for(int c_=0;c_<3;++c_)
                h.body_second_moment[body][3*r+c_]+=nodal_mass*local[r]*local[c_];
            const auto& v=h.com_velocity[body];const auto& w=h.omega[body];
            h.velocities[node]=Vector3{v[0]+w[1]*local[2]-w[2]*local[1],
                v[1]+w[2]*local[0]-w[0]*local[2],v[2]+w[0]*local[1]-w[1]*local[0]};
        }
    }
    return h;
}
SceneDescription kind_fixture(int mode,std::size_t points=4) {
    auto h=many_points_fixture(points);
    if(mode) {
        h.edges.push_back(BpEdge{{0,1}});
        h.edges.push_back(BpEdge{{2,3}});
    }
    if(mode==1)std::fill(h.surface.begin(),h.surface.end(),0);
    return h;
}
Run kind_run(int mode,const char* feature,int capacity,bool exact=false,
    bool dat=false,bool force=true,std::size_t points=4) {
    Environment kind("IPC_GPU_TWO_RIGID_KIND_PACKETS"),
        dense("IPC_GPU_TWO_RIGID_DENSE_CSR"),cap("IPC_GPU_CONTACT_PACKET_CAPACITY"),
        prune("IPC_GPU_BOX_CONTACT_PRUNING");
    kind.set(feature);dense.set("1");prune.set("0");
    const auto cap_text=std::to_string(capacity);cap.set(cap_text.c_str());
    const auto h=kind_fixture(mode,points);
    const std::vector<double> dhats=mode==1?std::vector<double>{3.,1.95,1.,3.}
        :mode==2?std::vector<double>{3.,.2,.05,3.}
        :std::vector<double>{.2,.11,.05,.2};
    const auto before=resident_bytes();Run result;
    {
        GpuSolver solver(h);
        for(double dhat:dhats) {
            auto p=parameters(exact,dat);p.contact.d_hat=dhat;
            p.box_radius_min=p.box_radius=1e-5;p.rotation_radius=1e-5;p.damping=.001;
            if(!force){p.contact.k_barrier=0.;p.contact.friction=0.;}
            solver.advance_substep(p,h.pin_targets,{});
            auto state=solver.snapshot();finite(state);
            result.states.push_back(std::move(state));
            result.resident.push_back(resident_bytes()-before);
        }
    }
    require(resident_bytes()==before&&!tracking_failed,"kind packet owner leaked storage");
    return result;
}
void kind_lifecycle() {
    for(int mode:{0,1,2})for(int capacity:{1,2,3,4,7,12}) {
        const auto off=kind_run(mode,"0",capacity);
        const auto on=kind_run(mode,"1",capacity);
        const auto unset=kind_run(mode,nullptr,capacity);
        const auto label="kind mode="+std::to_string(mode)+" capacity="+std::to_string(capacity);
        kind_compare_runs(off,on,label+" ON/OFF");
        kind_compare_runs(on,unset,label+" selected defaultON");
        const auto true_on=kind_run(mode,"1",capacity),true_unset=kind_run(mode,nullptr,capacity);
        kind_compare_runs(true_on,true_unset,label+" true-default UNSET/ON");
        require(true_on.resident==true_unset.resident,label+" true-default UNSET/ON resources");
        Run allocation_off,allocation_on,allocation_unset;
        {
            Environment slot("IPC_GPU_RIGID_ONE_SLOT_TERMS"),sparse("IPC_GPU_ORIGINAL_RIGID_SPARSE_GATHER");
            slot.set("0");sparse.set("0");
            allocation_off=kind_run(mode,"0",capacity);
            allocation_on=kind_run(mode,"1",capacity);
            allocation_unset=kind_run(mode,nullptr,capacity);
        }
        bool witnessed_metadata=false;
        for(std::size_t epoch=0;epoch<allocation_off.resident.size();++epoch) {
            require(allocation_on.resident[epoch]==allocation_off.resident[epoch]
                ||allocation_on.resident[epoch]==allocation_off.resident[epoch]+sizeof(unsigned),
                label+" unexpected metadata storage");
            witnessed_metadata=witnessed_metadata
                ||allocation_on.resident[epoch]==allocation_off.resident[epoch]+sizeof(unsigned);
            require(allocation_unset.resident[epoch]==allocation_on.resident[epoch],label+" unset did not select defaultON storage");
        }
        if(capacity==1)require(witnessed_metadata,label+" dense metadata witness is vacuous");
        const auto no_force=kind_run(mode,"0",capacity,false,false,false);
        require(!equal_bytes(off.states.front().com,no_force.states.front().com)
            ||!equal_bytes(off.states.front().omega,no_force.states.front().omega),
            label+" active contact-force witness is vacuous");
    }
    // Actual device publication and stale packet/chunk tails at larger grids.
    // Mixed→pure PT→empty→mixed occurs on one owner in every run.
    for(std::size_t points:{127u,128u,129u,257u})for(int capacity:{128,129})
        for(int mode:{0,2}) {
            kind_compare_runs(kind_run(mode,"0",capacity,false,false,true,points),
                kind_run(mode,"1",capacity,false,false,true,points),
                "kind device tails points="+std::to_string(points)+" cap="+std::to_string(capacity));
        }
    // The optional Robust empty-exact-queue/health protocol is deliberately
    // unchanged. DAT/raw owners receive no dense kind-range capability.
    for(bool dat:{false,true})for(bool exact:{false,true}) {
        kind_compare_runs(kind_run(2,"0",3,exact,dat),kind_run(2,"1",3,exact,dat),
            "kind Robust/DAT complete route");
    }
    Environment feature("IPC_GPU_TWO_RIGID_KIND_PACKETS");
    feature.set("1");raw_pipeline_gates();
    feature.set("0");const auto before=force_error("1",false);
    feature.set("1");const auto after=force_error("1",false);
    require(before==after,"kind packet zero-gap error/owner invalidation changed");
}
}
int main() {
    try {
        Environment feature("IPC_GPU_TWO_RIGID_KIND_PACKETS");feature.set("0");
        require(queue_reference_main()==0,"sealed queue reference suite failed");
        tracking=true;kind_lifecycle();
        require(resident_bytes()==0&&!tracking_failed,"kind packet allocation ledger not empty");
        tracking=false;
        std::cout<<"Kind packets: public six-array bytes, PT/EE/mixed, aligned/misaligned and above/below-cap epochs, "
            <<"empty/reentry, active forces, exact4B metadata, Robust/DAT/raw and rollback PASS\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"kind packet test: "<<e.what()<<'\n';return 1;}
}
