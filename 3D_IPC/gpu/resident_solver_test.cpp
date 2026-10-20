#include "resident_bridge.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "solid_ipc.h"
#include "broad_phase.h"
#include "barrier_energy.h"
#include "ipc_args.h"
#include <cmath>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value,const char* what) {if(!value)throw std::runtime_error(what);}
ipc_gpu::resident::SceneDescription point_scene(std::size_t count) {
    ipc_gpu::resident::SceneDescription h;
    h.positions.resize(count);h.velocities.resize(count);h.pin_targets.resize(count);h.body_reference.resize(count);
    h.deformable.assign(count,1);h.solid.assign(count,0);h.pinned.assign(count,0);h.surface.assign(count,1);
    h.mass.assign(count,1);h.node_body.assign(count,-1);h.node_blocks.resize(count);
    for(std::size_t n=0;n<count;++n)h.node_blocks[n]=static_cast<int>(n);
    h.block_count=static_cast<int>(count);
    h.triangle_offsets.assign(count+1,0);h.solid_offsets.assign(count+1,0);h.hinge_offsets.assign(count+1,0);
    h.body_node_offsets={0};
    return h;
}
void readme_defaults_test() {
    using namespace ipc_gpu::resident;
    IPCArgs3D args;
    std::vector<std::string> values={"3D_sim", "--example", "1", "--num_frames", "240",
        "--E", "115000", "--nu", "0.25", "--kB", "0.009", "--kpin", "1e9", "--twist_rate", "0.5",
        "--d_hat", "0.005", "--k_barrier", "100", "--node_box_min", "0.001", "--node_box_max", "0.01",
        "--fixed_iters", "--substeps", "5", "--max_substep_iters", "4", "--node_box_update_count", "10",
        "--use_basic_experimental", "true", "--use_simd", "true", "--use_parallel", "true",
        "--outdir", "example1_output", "--format", "geo"};
    std::vector<char*> argv;for(auto& value:values)argv.push_back(value.data());
    require(args.parse(static_cast<int>(argv.size()),argv.data()),"README argument parsing failed");
    auto params=args.to_sim_params();
    require(!params.ccd_exact_step_bound,"GPU default must keep exact-step-bound opt-in disabled");
    auto step=describe_step(params);
    require(step.global_ccd_guess&&step.colored_guess_sweeps==0,"README defaults must select standard global CCD guess");
    require(step.iterations==4&&params.substeps==5,"README iteration count changed");
    require(step.box_radius_min==.001&&step.box_radius==.01,"README adaptive node bounds changed");
    // End-to-end integration without elasticity isolates predictor dispatch.
    auto scene=point_scene(1);scene.positions[0]={.1,.2,.3};scene.velocities[0]={.3,-.2,.1};
    ResidentSolver solver(scene);step.iterations=0;
    solver.advance_substep(step,scene.pin_targets,{});
    auto output=solver.snapshot();
    for(int a=0;a<3;++a)
        require(std::abs(output.positions[0][a]-(scene.positions[0][a]+step.dt*scene.velocities[0][a]))<1e-15,
            "standard global predictor was not applied at solver entry");
    scene.solid[0]=1;
    ResidentSolver general(scene);general.advance_substep(step,scene.pin_targets,{});
    require(general.snapshot().positions==scene.positions,
        "general path must keep the CPU's legacy particle-guess suppression");
    // A zero-sweep colored guess still overrides the ordinary guess, exactly
    // as the CPU's flag dispatch does; zero is not interpreted as 'unset'.
    params.use_colored_ccd_guess=true;params.colored_ccd_guess_iters=0;
    const auto colored=describe_step(params);
    require(!colored.global_ccd_guess&&colored.colored_guess_sweeps==0,"colored-guess priority changed");
    params.use_colored_ccd_guess=false;params.use_ccd_guess=false;
    require(!describe_step(params).global_ccd_guess,"explicit no-guess selection ignored");
    // Forward the exact-bound CLI opt-in into the resident device backend.
    IPCArgs3D exact_args;
    values.insert(values.end(),{"--use_ticcd","false","--ccd_exact_step_bound","true"});
    argv.clear();for(auto& value:values)argv.push_back(value.data());
    require(exact_args.parse(static_cast<int>(argv.size()),argv.data()),"exact-step-bound argument parsing failed");
    const auto exact_params=exact_args.to_sim_params();
    require(exact_params.ccd_exact_step_bound&&describe_step(exact_params).ccd_exact_step_bound,
        "exact-step-bound CLI flag was not forwarded to the resident solver");
}
void stage_skip_test() {
    using namespace ipc_gpu::resident;
    StepParameters p;p.dt=.01;p.elastic.dt2=p.dt*p.dt;p.contact.dt=p.dt;p.sdf.dt=p.dt;
    p.contact.eps=1e-12;p.contact.eps_v=.001;
    p.iterations=2;p.rebuild_interval=1;p.colored_guess_sweeps=2;p.elastic.kpin=100;
    // No empty CUDA launches or attempts to consume nonexistent color arrays.
    ResidentSolver empty(point_scene(0));empty.advance_substep(p,{},{});
    require(empty.snapshot().positions.empty(),"empty resident scene changed size");
    // A soft pin is not a fixed body. Its point energy must still be assembled.
    auto h=point_scene(1);h.pinned[0]=1;h.pin_targets[0]={.001,0,0};
    ResidentSolver pinned(h);pinned.advance_substep(p,h.pin_targets,{});
    require(pinned.snapshot().positions[0][0]>0,"soft-pinned node was incorrectly scheduled as fixed");
    // All-fixed rigid scenes skip solver/guess work, but retain the body and
    // all proxy coordinates byte-for-byte through begin/end bookkeeping.
    h=point_scene(3);h.positions={Vector3{0,0,0},Vector3{1,0,0},Vector3{0,1,0}};
    h.body_reference=h.positions;h.deformable.assign(3,0);h.node_body.assign(3,0);h.node_blocks.assign(3,0);
    h.block_count=1;h.body_blocks={0};h.body_nodes={0,1,2};h.body_node_offsets={0,3};h.body_mass={1};
    h.body_second_moment={Matrix3{1,0,0,0,1,0,0,0,1}};h.body_translate={0};h.body_rotate={0};
    h.com={Vector3{0,0,0}};h.com_velocity={Vector3{0,0,0}};h.omega={Vector3{0,0,0}};
    h.orientation={Quaternion{1,0,0,0}};h.surface_triangles={BpTriangle{{0,1,2}}};
    ResidentSolver fixed(h);fixed.advance_substep(p,h.pin_targets,{});
    const auto fixed_state=fixed.snapshot();
    require(fixed_state.positions==h.positions&&fixed_state.com==h.com&&fixed_state.orientation==h.orientation,
        "all-fixed resident scene changed pose");
}
// Preserve the caller's environment while exercising both opt-in and rollback.
class ScopedTestEnvironment {
    std::string name_,saved_;bool present_;
public:
    explicit ScopedTestEnvironment(const char* name):name_(name),present_(std::getenv(name)!=nullptr) {
        if(present_)saved_=std::getenv(name);
    }
    ~ScopedTestEnvironment() {
        if(present_)::setenv(name_.c_str(),saved_.c_str(),1);else ::unsetenv(name_.c_str());
    }
    void assign(const char* value) {require(::setenv(name_.c_str(),value,1)==0,"test environment update failed");}
};
template<class T> bool same_state_bytes(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
void pure_cloth_selected_ordinary_test() {
    using namespace ipc_gpu::resident;
    ScopedTestEnvironment selected("IPC_GPU_SELECTED_ORDINARY_GATHER"),defer("IPC_GPU_DEFER_CONTACT_INPUTS"),
        queues("IPC_GPU_COLOR_PAIR_QUEUES"),
        compact("IPC_GPU_COMPACT_DEFORMABLE_CONTACTS");
    queues.assign("1");compact.assign("1");
    // The owning lifecycle fixture's positive cloth PT scene: one distinct
    // block per node, holes in the block namespace, and an active surface gap.
    auto h=point_scene(4);h.positions={Vector3{.25,.25,.001},Vector3{0,0,0},Vector3{1,0,0},Vector3{0,1,0}};
    h.pin_targets=h.positions;h.body_reference=h.positions;h.node_blocks={7,2,5,0};h.block_count=8;
    h.surface_triangles={BpTriangle{{1,2,3}}};h.edges={BpEdge{{1,2}},BpEdge{{2,3}},BpEdge{{3,1}}};
    StepParameters p;p.dt=.01;p.elastic.dt2=p.dt*p.dt;p.contact.dt=p.dt;p.sdf.dt=p.dt;
    p.contact.d_hat=.01;p.contact.k_barrier=1;p.contact.friction=0;
    p.contact.eps_v=.001;p.contact.eps=1e-12;p.damping=.001;
    p.iterations=1;p.rebuild_interval=1;p.box_radius_min=.001;p.box_radius=.01;
    p.collisions=true;p.global_ccd_guess=false;p.colored_guess_sweeps=0;
    for(const char* deferred:{"0","1"}) {
        defer.assign(deferred);selected.assign("0");
        ResidentSolver control(h);control.advance_substep(p,h.pin_targets,{});const auto expected=control.snapshot();
        selected.assign("1");
        ResidentSolver candidate(h);candidate.advance_substep(p,h.pin_targets,{});const auto actual=candidate.snapshot();
        require(same_state_bytes(expected.positions,actual.positions)&&same_state_bytes(expected.velocities,actual.velocities)
            &&same_state_bytes(expected.com,actual.com)&&same_state_bytes(expected.com_velocity,actual.com_velocity)
            &&same_state_bytes(expected.omega,actual.omega)&&same_state_bytes(expected.orientation,actual.orientation),
            "pure cloth selected ordinary OFF/ON saved-state bytes differ");
        bool moved=false;
        for(std::size_t n=0;n<actual.positions.size();++n)for(int a=0;a<3;++a) {
            require(std::isfinite(actual.positions[n][a])&&std::isfinite(actual.velocities[n][a]),"nonfinite pure cloth selected ordinary state");
            moved|=actual.positions[n][a]!=h.positions[n][a];
        }
        require(moved,"pure cloth selected ordinary fixture lost active contact response");
    }
    std::cout<<"Resident pure cloth selected ordinary: OFF/ON saved-state bytes, active contact response, deferred/nondeferred PASS\n";
}
void predictor_ports_test() {
    using namespace ipc_gpu::resident;
    StepParameters p;p.dt=.01;p.elastic.dt2=p.dt*p.dt;p.contact.dt=p.dt;p.sdf.dt=p.dt;
    p.contact.d_hat=.005;p.contact.eps=1e-12;p.contact.eps_v=.001;
    p.iterations=0;p.colored_guess_sweeps=1;
    auto h=point_scene(4);
    h.positions={Vector3{.25,.25,1},Vector3{0,0,0},Vector3{1,0,0},Vector3{0,1,0}};
    h.velocities[0]={0,0,-200};h.pin_targets=h.positions;h.body_reference=h.positions;
    h.deformable={1,0,0,0};h.node_body={-1,0,0,0};h.node_blocks={0,1,1,1};h.block_count=2;
    h.body_blocks={1};h.body_nodes={1,2,3};h.body_node_offsets={0,3};h.body_mass={1};
    h.body_second_moment={Matrix3{1,0,0,0,1,0,0,0,1}};h.body_translate={0};h.body_rotate={0};
    h.com={Vector3{0,0,0}};h.com_velocity={Vector3{0,0,0}};h.omega={Vector3{0,0,0}};
    h.orientation={Quaternion{1,0,0,0}};h.surface_triangles={BpTriangle{{1,2,3}}};
    h.edges={BpEdge{{1,2}},BpEdge{{2,3}},BpEdge{{3,1}}};
    double legacy_gap=0;
    for(bool direct:{false,true}) {
        p.ccd_exact_step_bound=direct;p.colored_guess_level_based=false;
        ResidentSolver legacy(h);legacy.advance_substep(p,h.pin_targets,{});const auto expected=legacy.snapshot();
        p.colored_guess_level_based=true;
        ResidentSolver levels(h);levels.advance_substep(p,h.pin_targets,{});const auto actual=levels.snapshot();
        // Three rigid proxy vertices make four vertex colors but only two
        // update blocks. Exactly one discarded trial must be restored before
        // the ONE requested actual sweep; failure gives a different gap.
        require(same_state_bytes(expected.positions,actual.positions)&&same_state_bytes(expected.velocities,actual.velocities),
            "level predictor changed one-active-node schedule or failed to restore discarded trial");
        for(std::size_t n=1;n<h.positions.size();++n)
            require(actual.positions[n]==h.positions[n],"level predictor moved fixed rigid obstacle");
        require(actual.com==h.com&&actual.orientation==h.orientation,"level predictor changed rigid pose");
        const double gap=actual.positions[0][2];
        require(std::isfinite(gap)&&gap>=1e-8&&gap<1,"ported predictor lost strict endpoint separation or all motion");
        if(!direct)legacy_gap=gap;
        else require(gap<legacy_gap,"exact direct-bound fixture did not exercise accepted partial proposal");
    }
    // Zero requested sweeps performs no trial, even with both opt-ins set.
    p.colored_guess_sweeps=0;
    ResidentSolver zero(h);zero.advance_substep(p,h.pin_targets,{});
    require(zero.snapshot().positions==h.positions,"zero-sweep level predictor ran a discarded or actual update");
    // Empty graphs and isolated active vertices still have valid epochs.
    for(std::size_t n:{std::size_t(0),std::size_t(1)}) {
        auto isolated=point_scene(n);
        if(n) {isolated.positions[0]={.1,.2,.3};isolated.velocities[0]={.3,-.2,.1};}
        p.colored_guess_sweeps=2;
        ResidentSolver solver(isolated);solver.advance_substep(p,isolated.pin_targets,{});const auto state=solver.snapshot();
        for(std::size_t v=0;v<n;++v)for(int a=0;a<3;++a)
            require(state.positions[v][a]==isolated.positions[v][a]+p.dt*isolated.velocities[v][a],
                "isolated level predictor changed fixed target or counted trial as actual motion");
    }
    // Public StepParameters must enforce the configured level mode even
    // when zero sweeps would otherwise bypass its numerical work.
    for(int invalid=0;invalid<6;++invalid) {
        auto bad=p;bad.colored_guess_sweeps=0;
        if(invalid==0)bad.global_ccd_guess=true;
        if(invalid==1)bad.contact.d_hat=-1;
        if(invalid==2)bad.contact.d_hat=std::numeric_limits<double>::quiet_NaN();
        if(invalid==3)bad.contact.d_hat=std::numeric_limits<double>::infinity();
        if(invalid==4) {bad.global_ccd_guess=true;bad.colored_guess_sweeps=2;}
        if(invalid==5)bad.colored_guess_sweeps=-1;
        ResidentSolver guarded(h);bool rejected=false;
        try {guarded.advance_substep(bad,h.pin_targets,{});}
        catch(const std::invalid_argument&) {rejected=true;}
        require(rejected&&guarded.snapshot().positions==h.positions,
            "invalid manual level configuration changed state or bypassed zero-sweep validation");
    }
    std::cout<<"Resident CPU-feature ports: direct proposal, restored trial, rigid obstacles, four colors/two blocks, empty/isolated/zero sweeps PASS\n";
}
void check_contacts(const RefMesh& mesh,const DeformedState& state,const SimParams& p) {
    BroadPhase bp;std::vector<AABB> boxes;
    for(const auto& x:state.deformed_positions)boxes.emplace_back(x-Vec3::Constant(1e-8),x+Vec3::Constant(1e-8));
    bp.initialize_surface_nodes(boxes,mesh,p.d_hat,BroadPhase::InitializationMode::GeneralSolver);
    std::size_t active=0;
    for(const auto& pair:bp.cache().nt_pairs) {
        const std::array<Vec3,4> x{state.deformed_positions[pair.node],state.deformed_positions[pair.tri_v[0]],
            state.deformed_positions[pair.tri_v[1]],state.deformed_positions[pair.tri_v[2]]};
        const auto c=make_node_triangle_contact_evaluation(x,p.d_hat,p.k_barrier,1e-12,nullptr,true);
        require(std::isfinite(c.dr.distance)&&c.dr.distance>0,"resident solver produced zero/nonfinite NT gap");
        active+=c.active;
    }
    for(const auto& pair:bp.cache().ss_pairs) {
        const std::array<Vec3,4> x{state.deformed_positions[pair.v[0]],state.deformed_positions[pair.v[1]],
            state.deformed_positions[pair.v[2]],state.deformed_positions[pair.v[3]]};
        const auto c=make_segment_segment_contact_evaluation(x,p.d_hat,p.k_barrier,1e-12,nullptr,true);
        require(std::isfinite(c.dr.distance)&&c.dr.distance>0,"resident solver produced zero/nonfinite EE gap");
        active+=c.active;
    }
    require(active>0,"resident mixed fixture lost all active contact coverage");
}
void run(double friction,bool guess,RigidBodyUpdateMode mode=RigidBodyUpdateMode::TranslationAndOrientation,
    bool levels=false,bool direct=false) {
    SimParams p=SimParams::zeros();p.fps=60;p.substeps=1;p.mu=2;p.lambda=3;p.kB=1e-4;p.kpin=1000;
    p.solid_mu=2;p.solid_lambda=3;p.gravity=Vec3(0,-.1,0);p.node_box_min=p.node_box_max=.01;
    p.node_box_update_count=2;p.max_global_iters=3;p.fixed_iters=true;p.damping=.1;
    p.theta_box_min=p.theta_box_max=.1;
    p.use_ccd=true;p.use_ticcd=false;p.use_ccd_guess=false;p.use_colored_ccd_guess=guess&&!levels;p.colored_ccd_guess_iters=2;
    p.use_colored_ccd_guess_level_based=levels;p.ccd_exact_step_bound=direct;
    p.d_hat=.03;p.k_barrier=1;p.friction_coefficient=friction;p.friction_velocity_epsilon=.01;
    p.k_sdf=.1;p.eps_sdf=.03;p.sdf_planes.push_back({Vec3(0,-.02,0),Vec3::UnitY()});
    RefMesh mesh;DeformedState state;std::vector<Vec2> material;std::vector<Pin> pins;
    build_square_mesh(mesh,state,material,3,3,1,1,Vec3(-.5,0,-.5));
    mesh.build_lumped_mass(900,.001);
    state.velocities.assign(state.deformed_positions.size(),Vec3(-.005,-.003,.002));
    append_pin(pins,0,state.deformed_positions);append_pin(pins,3,state.deformed_positions);
    const auto cloth_count=state.deformed_positions.size();
    std::vector<Vec3> solid{Vec3(-.08,.006,-.08),Vec3(.08,.006,-.08),Vec3(-.08,.006,.08),Vec3(-.08,.012,-.08)};
    solid.push_back(.25*(solid[0]+solid[1]+solid[2]+solid[3]));
    create_solid(solid,{4,2,1,3,0,4,1,3,0,2,4,3,0,2,1,4},900,mesh,state);
    for(std::size_t n=cloth_count;n<state.velocities.size();++n)state.velocities[n]=Vec3(.003,-.002,-.004);
    const int body=create_rigid_body({Vec3(-.4,.018,-.4),Vec3(.4,.018,-.4),Vec3(0,.018,.4)},
        Vec3(.01,-.01,.02),Vec4(1,0,0,0),Vec3(.02,-.01,.01),1,mesh,state,mode);
    mesh.tris.insert(mesh.tris.end(),mesh.rb_nodes[body].begin(),mesh.rb_nodes[body].end());
    mesh.node_to_rb.resize(state.deformed_positions.size(),-1);mesh.build_deformable_nodes();
    const auto original=state.deformed_positions;
    const auto original_com=state.x_coms[body];
    const auto original_orientation=state.orientations[body];
    check_contacts(mesh,state,p);
    ipc_gpu::resident::ResidentSolver solver(ipc_gpu::resident::describe_scene(mesh,state,pins,p));
    for(int step=0;step<4;++step) {
        solver.advance_substep(ipc_gpu::resident::describe_step(p),
            ipc_gpu::resident::describe_pin_targets(state.deformed_positions.size(),pins),
            ipc_gpu::resident::describe_colliders(p));
        ipc_gpu::resident::copy_snapshot(solver.snapshot(),state);
        for(const auto& x:state.deformed_positions)require(x.allFinite(),"nonfinite resident mixed state");
        require(std::abs(state.orientations[body].norm()-1)<1e-10,"resident rigid orientation norm");
        if(!updates_rigid_translation(mode))
            require(state.x_coms[body]==original_com,"disabled rigid translation stage changed COM");
        if(!updates_rigid_orientation(mode))
            require(state.orientations[body]==original_orientation,"disabled rigid orientation stage changed quaternion");
        check_contacts(mesh,state,p);
    }
    double motion=0;
    for(std::size_t n=0;n<original.size();++n)motion+=(original[n]-state.deformed_positions[n]).norm();
    require(motion>1e-8,"resident integration did not move scene");
    std::cout<<"Resident mixed solver: friction="<<friction<<", colored guess="<<guess
        <<", level="<<levels<<", exact bound="<<direct
        <<", 4 substeps, cloth+solid+rigid+SDF; displacement sum="<<motion<<"\n";
}
}
int main() {
    try {
        SimParams unsupported=SimParams::zeros();
        unsupported.fixed_iters=true;unsupported.use_ccd_guess=false;
        unsupported.use_ticcd=true;
        bool rejected=false;
        try {(void)ipc_gpu::resident::describe_step(unsupported);}
        catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"resident solver must not silently replace Tight-Inclusion");
        unsupported.use_ticcd=false;
        for(bool collisions:{false,true})for(bool direct:{false,true}) {
            unsupported.use_ccd=collisions;unsupported.ccd_exact_step_bound=direct;
            const auto mapped=ipc_gpu::resident::describe_step(unsupported);
            require(mapped.collisions==collisions&&mapped.ccd_exact_step_bound==direct,
                "exact-step-bound or collision flag mapping changed");
        }
        for(bool legacy:{false,true})for(bool global:{false,true})for(int sweeps:{0,2}) {
            unsupported.use_colored_ccd_guess=legacy;unsupported.use_ccd_guess=global;
            unsupported.use_colored_ccd_guess_level_based=true;unsupported.colored_ccd_guess_iters=sweeps;
            if(legacy) {
                rejected=false;
                try {(void)ipc_gpu::resident::describe_step(unsupported);}
                catch(const std::invalid_argument& error) {
                    rejected=std::string(error.what()).find("Choose only one")!=std::string::npos;
                }
                require(rejected,"CPU mutually exclusive colored-guess flags were accepted by GPU bridge");
            } else {
                const auto mapped=ipc_gpu::resident::describe_step(unsupported);
                require(mapped.colored_guess_level_based&&mapped.colored_guess_sweeps==sweeps&&!mapped.global_ccd_guess,
                    "level colored predictor did not suppress global guess or preserve zero sweeps");
            }
        }
        for(int invalid=0;invalid<8;++invalid) {
            SimParams bad=SimParams::zeros();bad.fixed_iters=true;bad.use_ticcd=false;
            bad.use_ccd_guess=false;bad.use_colored_ccd_guess_level_based=true;bad.colored_ccd_guess_iters=0;
            if(invalid==0)bad.colored_ccd_guess_iters=-1;
            if(invalid==1)bad.d_hat=-1;
            if(invalid==2)bad.d_hat=std::numeric_limits<double>::quiet_NaN();
            if(invalid==3)bad.d_hat=std::numeric_limits<double>::infinity();
            if(invalid>=4)bad.ccd_exact_step_bound=true;
            if(invalid==4)bad.exact_computation_fallback=false;
            if(invalid==5)bad.use_ticcd=true;
            if(invalid==6)bad.use_ogc=true;
            if(invalid==7)bad.use_ogc_solver=true;
            rejected=false;
            try {(void)ipc_gpu::resident::describe_step(bad);}
            catch(const std::invalid_argument&) {rejected=true;}
            require(rejected,"resident bridge bypassed CPU colored-guess or exact-bound parameter validation");
        }
        readme_defaults_test();
        stage_skip_test();
        pure_cloth_selected_ordinary_test();
        predictor_ports_test();
        run(0,false);run(.1,false);run(.1,true);
        run(0,true,RigidBodyUpdateMode::TranslationAndOrientation,true,false);
        run(.1,true,RigidBodyUpdateMode::TranslationAndOrientation,true,true);
        run(.1,false,RigidBodyUpdateMode::TranslationAndOrientation,false,true);
        run(0,true,RigidBodyUpdateMode::TranslationOnly);
        run(0,true,RigidBodyUpdateMode::OrientationOnly);
        run(0,true,RigidBodyUpdateMode::None);
        {
            // Keep the generic gather rollback's mixed cloth/solid/rigid
            // collision and finite checks alongside the selected path.
            ScopedTestEnvironment selected("IPC_GPU_SELECTED_ORDINARY_GATHER");selected.assign("0");
            run(0,false);
        }
        return 0;
    }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
