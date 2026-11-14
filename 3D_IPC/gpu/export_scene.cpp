// Original scene preparation and prescribed inputs for the Newton-based IPC GPU solver.
// No evolving energy, contact, CCD or time-integration calculation runs here.
#include "example.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr const char* schema="ipc-newton-scene-v1";
void text(std::ostream& out,const std::string& value) {
    constexpr char hex[]="0123456789abcdef";
    out.put('"');
    for(unsigned char c:value) {
        switch(c) {
        case '"':out<<"\\\"";break;
        case '\\':out<<"\\\\";break;
        case '\n':out<<"\\n";break;
        case '\r':out<<"\\r";break;
        case '\t':out<<"\\t";break;
        default:if(c<32)out<<"\\u00"<<hex[c>>4]<<hex[c&15];else out.put(char(c));
        }
    }
    out.put('"');
}
void value(std::ostream& out,double v) {if(!std::isfinite(v))throw std::runtime_error("Nonfinite native scene value");out<<v;}
void value(std::ostream& out,int v){out<<v;}
void value(std::ostream& out,std::size_t v){out<<v;}
void value(std::ostream& out,bool v){out<<(v?"true":"false");}
template<class R,class F>void array(std::ostream& out,const R& r,F emit){out<<'[';bool first=true;for(const auto& v:r){if(!first)out<<',';first=false;emit(v);}out<<']';}
template<class R>void scalars(std::ostream& out,const R& r){array(out,r,[&](const auto& v){value(out,v);});}
template<class V>void vector(std::ostream& out,const V& v){out<<'[';for(int i=0;i<v.size();++i){if(i)out<<',';value(out,v[i]);}out<<']';}
template<class R>void vectors(std::ostream& out,const R& r){array(out,r,[&](const auto& v){vector(out,v);});}
template<class M>void matrix(std::ostream& out,const M& m){out<<'[';bool first=true;for(int i=0;i<m.rows();++i)for(int j=0;j<m.cols();++j){if(!first)out<<',';first=false;value(out,m(i,j));}out<<']';}
void cells(std::ostream& out,const std::vector<int>& v,int size){if(v.size()%size)throw std::runtime_error("Malformed topology");out<<'[';for(std::size_t i=0;i<v.size();i+=size){if(i)out<<',';out<<'[';for(int j=0;j<size;++j){if(j)out<<',';value(out,v[i+j]);}out<<']';}out<<']';}
struct Object {
    std::ostream& out;bool first=true;
    explicit Object(std::ostream& stream):out(stream){out<<'{';}
    void key(const char* name){if(!first)out<<',';first=false;text(out,name);out<<':';}
    template<class T>void scalar(const char* name,T v){key(name);value(out,v);}
    void string(const char* name,const std::string& v){key(name);text(out,v);}
    void end(){out<<'}';}
};
void pose(std::ostream& out,const SDFMaterialPose& p){Object o(out);o.key("rotation");matrix(out,p.rotation);o.key("translation");vector(out,p.translation);o.end();}
void colliders(std::ostream& out,const SimParams& p){
    out<<'[';bool first=true;
    const auto emit=[&](const char* kind,const Vec3& point,const Vec3& axis,double radius,const SDFMaterialMotion& motion){
        if(!first)out<<',';first=false;Object o(out);o.string("kind",kind);o.key("point");vector(out,point);o.key("axis");vector(out,axis);o.scalar("radius",radius);o.key("previous_pose");pose(out,motion.previous);o.key("current_pose");pose(out,motion.current);o.end();
    };
    for(const auto& s:p.sdf_planes)emit("plane",s.point,s.normal,0,s.material_motion);
    for(const auto& s:p.sdf_cylinders)emit("cylinder",s.point,s.axis,s.radius,s.material_motion);
    for(const auto& s:p.sdf_spheres)emit("sphere",s.center,Vec3::UnitY(),s.radius,s.material_motion);
    out<<']';
}
void params(std::ostream& out,const SimParams& p){Object o(out);
#define F(name) o.scalar(#name,p.name)
    F(fps);F(substeps);F(mu);F(lambda);F(density);F(thickness);F(solid_mu);F(solid_lambda);F(solid_density);F(rigid_density);
    F(kpin);F(kB);F(tol_abs);F(tol_rel);F(d_hat);F(k_barrier);F(friction_coefficient);F(friction_velocity_epsilon);F(k_sdf);F(eps_sdf);
    F(max_global_iters);F(damping);F(fixed_iters);F(use_parallel);F(use_simd);F(use_ccd);F(use_ccd_guess);F(use_colored_ccd_guess);F(use_colored_ccd_guess_level_based);F(colored_ccd_guess_iters);F(colored_ccd_guess_fraction);
    F(use_dat);F(use_ogc);F(use_jacobi_level_based_ccd);F(use_cloth_grid);F(node_box_min);F(node_box_max);F(theta_box_min);F(theta_box_max);F(node_box_update_count);F(exact_computation_fallback);F(exact_computation_fallback_2);F(ccd_exact_step_bound);F(rigid_pin_time);F(write_substeps);F(verbose);
    F(dat_gamma);F(dat_inertial_predictor);F(dat_collision_detection_interval);F(dat_query_radius);F(dat_node_box_padding);
#undef F
    o.scalar("dt",p.dt());o.scalar("dt2",p.dt2());o.key("gravity");vector(out,p.gravity);o.end();
}
void soft_pins(std::ostream& out,const std::vector<Pin>& p){array(out,p,[&](const Pin& pin){Object o(out);o.scalar("node",pin.vertex_index);o.key("target");vector(out,pin.target_position);o.scalar("prescribed",pin.prescribed);o.end();});}
void rigid_pins(std::ostream& out,const std::vector<RigidPin>& pins){array(out,pins,[&](const RigidPin& p){Object o(out);o.scalar("body",p.body);o.key("local_position");vector(out,p.local_position);o.key("initial_target");vector(out,p.initial_target);o.key("rotation_center");vector(out,p.rotation_center);o.key("rotation_axis");vector(out,p.rotation_axis);o.scalar("angular_velocity_z",p.angular_velocity_z);o.key("target_translation");vector(out,p.target_translation);o.scalar("translation_start_time",p.translation_start_time);o.scalar("translation_duration",p.translation_duration);o.scalar("angular_velocity_ramp_duration",p.angular_velocity_ramp_duration);o.end();});}

struct Scene {
    IPCArgs3D args;SimParams p=SimParams::zeros();RefMesh mesh;DeformedState state;std::vector<Vec2> X;std::vector<Pin> pins;
    std::vector<Vec3> visual;std::vector<int> visual_tris;TwistSpec twist;CylinderTwistSpec cylinders;TwistUntwistSpec untwist;OscillatingClothLayersSpec oscillating;
    bool has_rigid=false,general=false;
    void build(){
        p=args.to_sim_params();
        switch(args.example){
        case 1:build_twisting_cloth_example(args,mesh,state,X,pins,twist);break;
        case 2:build_two_cylinder_twist_example(args,mesh,state,X,pins,p,visual,visual_tris,cylinders);break;
        case 3:build_twist_untwist_example(args,mesh,state,X,pins,p,visual,visual_tris,untwist);break;
        case 4:build_avatar_clothing_example(args,mesh,state,pins,p,visual,visual_tris);break;
        case 5:build_rotating_tennis_racket_example(args,mesh,state,X,pins,p);break;
        case 6:build_rotating_space_tool_example(args,mesh,state,X,pins,p);break;
        case 7:build_twenty_rigid_polygon_static_stack_example(args,mesh,state,X,pins,p,visual,visual_tris);break;
        case 8:build_two_bunny_spot_cube_gear_cycles_on_pinned_cloth_example(args,mesh,state,X,pins,p);break;
        case 9:build_dynamic_bolt_into_fixed_nut_example(args,mesh,state,X,pins,p);break;
        case 10:build_four_bunny_spot_cube_gear_rows_on_pinned_cloth_example(args,mesh,state,X,pins,p);break;
        case 11:build_cloth_unrolling_down_fixed_ramp_example(args,mesh,state,X,pins,p,visual,visual_tris);break;
        case 12:build_oscillating_cloth_layers_example(args,mesh,state,X,pins,p,oscillating);break;
        case 13:build_wrecking_ball_example(args,mesh,state,X,pins,p,visual,visual_tris);break;
        case 14:case 15:build_cloth_cylinder_drop_example(args,mesh,state,X,pins,p,visual,visual_tris);break;
        case 16:build_chain_net_ball_example(args,mesh,state,X,pins,p);break;
        case 17:build_rigid_objects_in_box_example(args,mesh,state,X,pins,p);break;
        case 18:build_bike_chain_example(args,mesh,state,X,pins,p);break;
        case 19:build_anchor_example(args,mesh,state,X,pins,p);break;
        case 20:build_arch_example(args,mesh,state,X,pins,p,visual,visual_tris);break;
        default:throw std::invalid_argument("Unknown example ID");}
        auto n=state.deformed_positions.size();has_rigid=!mesh.total_mass.empty();
        if(mesh.node_to_rb.empty()&&!has_rigid)mesh.node_to_rb.assign(n,-1);
        if(mesh.node_to_rb.size()!=n||state.velocities.size()!=n)throw std::runtime_error("Inconsistent native scene arrays");
        bool deformable=false;for(int b:mesh.node_to_rb){if(b < -1||b>=int(mesh.total_mass.size()))throw std::runtime_error("Invalid body owner");deformable=deformable||b<0;}
        general=!mesh.tet_nodes.empty()||(has_rigid&&deformable);
        if(general)mesh.build_deformable_lumped_mass(p.density,p.thickness);else if(!has_rigid)mesh.build_lumped_mass(p.density,p.thickness);
        mesh.build_deformable_nodes();
        if(mesh.mass.size()!=n||mesh.rb_update_modes.size()!=mesh.total_mass.size())throw std::runtime_error("Incomplete native mass/body labels");
    }
    void update(double t){
        if(args.example==1)update_twist_pins(pins,twist,t);
        else if(args.example==2){update_cylinder_twist_pins(pins,cylinders,t);update_cylinder_sdfs(p,cylinders,t);update_cylinder_visuals(visual,cylinders,t);}
        else if(args.example==3){update_twist_untwist_pins(pins,untwist,t);update_twist_untwist_sdf(p,untwist,t);update_twist_untwist_visual(visual,untwist,t);}
        else if(args.example==12)update_oscillating_cloth_layer_pins(pins,oscillating,t);
        if(has_rigid&&!general)p.rigid_pin_time=t;
    }
    std::vector<Vec3> targets()const{std::vector<Vec3> v(state.deformed_positions.size(),Vec3::Zero());for(const auto& pin:pins)v.at(pin.vertex_index)=pin.target_position;return v;}
    void motions(std::ostream& out)const{
        Object o(out);o.key("pins");out<<'[';bool first=true;
        const auto group=[&](const std::vector<int>& indices,const std::vector<Vec3>& initial,const Vec3& center,const Vec3& axis,double speed,double settle,double ramp,double maximum,bool reverse,double hold,double amplitude=0.0,double frequency=0.0){
            for(std::size_t k=0;k<indices.size();++k){if(!first)out<<',';first=false;Object item(out);item.scalar("node",pins.at(indices[k]).vertex_index);item.key("initial");vector(out,initial[k]);item.key("center");vector(out,center);item.key("axis");vector(out,axis);item.scalar("speed",speed);item.scalar("settle",settle);item.scalar("ramp",ramp);item.scalar("maximum",maximum);item.scalar("reverse",reverse);item.scalar("hold",hold);item.scalar("amplitude",amplitude);item.scalar("frequency",frequency);item.end();}
        };
        if(args.example==1){group(twist.left_pin_indices,twist.left_initial_targets,twist.axis_point,Vec3::UnitX(),twist.omega_left,0,0,0,false,0);group(twist.right_pin_indices,twist.right_initial_targets,twist.axis_point,Vec3::UnitX(),twist.omega_right,0,0,0,false,0);}
        else if(args.example==2){group(cylinders.top_pin_indices,cylinders.top_initial_targets,cylinders.top_axis_point,Vec3::UnitY(),cylinders.omega_top,cylinders.t_settle,cylinders.t_ramp,cylinders.max_abs_theta,cylinders.untwist,cylinders.t_hold);group(cylinders.bot_pin_indices,cylinders.bot_initial_targets,cylinders.bot_axis_point,Vec3::UnitY(),cylinders.omega_bot,cylinders.t_settle,cylinders.t_ramp,cylinders.max_abs_theta,cylinders.untwist,cylinders.t_hold);}
        else if(args.example==3)group(untwist.wrap_pin_indices,untwist.wrap_initial_targets,untwist.cyl_axis_point,Vec3::UnitY(),untwist.omega,untwist.t_settle,untwist.t_ramp,untwist.max_abs_theta,untwist.untwist,untwist.t_hold);
        else if(args.example==12)group(oscillating.driven_pin_indices,oscillating.driven_initial_targets,Vec3::Zero(),oscillating.motion_direction,0,0,0,0,false,0,oscillating.amplitude,oscillating.frequency_hz);
        out<<']';o.key("colliders");out<<'[';first=true;
        const auto collider=[&](int index,const Vec3& center,double speed,double settle,double ramp,double maximum,bool reverse,double hold){if(!first)out<<',';first=false;Object item(out);item.scalar("index",int(p.sdf_planes.size())+index);item.key("center");vector(out,center);item.scalar("speed",speed);item.scalar("settle",settle);item.scalar("ramp",ramp);item.scalar("maximum",maximum);item.scalar("reverse",reverse);item.scalar("hold",hold);item.end();};
        if(args.example==2){collider(0,cylinders.top_axis_point,cylinders.omega_top,cylinders.t_settle,cylinders.t_ramp,cylinders.max_abs_theta,cylinders.untwist,cylinders.t_hold);collider(1,cylinders.bot_axis_point,cylinders.omega_bot,cylinders.t_settle,cylinders.t_ramp,cylinders.max_abs_theta,cylinders.untwist,cylinders.t_hold);}
        else if(args.example==3)collider(untwist.cyl_sdf_index,untwist.cyl_axis_point,untwist.omega,untwist.t_settle,untwist.t_ramp,untwist.max_abs_theta,untwist.untwist,untwist.t_hold);
        out<<']';o.end();
    }
    void boundary_fields(std::ostream& out,Object& o)const{
        o.key("pins");soft_pins(out,pins);o.key("pin_targets");vectors(out,targets());o.key("colliders");colliders(out,p);o.scalar("rigid_pin_time",p.rigid_pin_time);
        o.key("rigid_pin_targets");array(out,p.rigid_pins,[&](const RigidPin& pin){vector(out,rigid_pin_target_position(pin,p.rigid_pin_time));});
    }
    void boundary(std::ostream& out,double time)const{Object o(out);o.string("schema",schema);o.string("kind","boundary");o.scalar("time",time);boundary_fields(out,o);if(args.example==2||args.example==3){o.key("static_positions");vectors(out,visual);}o.end();out<<'\n'<<std::flush;}
    void write(std::ostream& out)const{
        Object o(out);o.string("schema",schema);o.string("kind","scene");o.scalar("example",args.example);o.key("params");params(out,p);
        o.key("positions");vectors(out,state.deformed_positions);o.key("velocities");vectors(out,state.velocities);o.key("mass");scalars(out,mesh.mass);o.key("tris");cells(out,mesh.tris,3);o.key("tets");cells(out,mesh.tets,4);o.key("rest_coordinates");vectors(out,X);
        o.key("triangle_dm_inverse");array(out,mesh.Dm_inverse,[&](const Mat22& m){matrix(out,m);});o.key("triangle_areas");scalars(out,mesh.area);
        o.key("hinges");array(out,mesh.hinges,[&](const Hinge& h){Object item(out);item.key("nodes");scalars(out,h.v);item.scalar("rest_angle",h.bar_theta);item.scalar("coefficient",h.c_e);item.end();});
        o.key("tet_rest_data");array(out,mesh.tet_rest_data,[&](const TetRestData& r){Object item(out);item.key("dm_inverse");matrix(out,r.Dm_inverse);item.scalar("measure",r.measure);item.key("grad_N");vectors(out,r.grad_N);item.end();});
        o.key("node_to_rb");scalars(out,mesh.node_to_rb);o.key("tet_nodes");scalars(out,mesh.tet_nodes);o.key("surface_nodes");scalars(out,mesh.surface_nodes);o.key("deformable_nodes");scalars(out,mesh.deformable_nodes);
        o.key("bodies");std::vector<int> ids(mesh.total_mass.size());for(std::size_t b=0;b<ids.size();++b)ids[b]=int(b);
        array(out,ids,[&](int b){Object item(out);item.key("nodes");scalars(out,mesh.rb_nodes[b]);item.key("reference_positions");vectors(out,mesh.ref_positions[b]);item.scalar("mass",mesh.total_mass[b]);item.key("I_hat");matrix(out,mesh.I_hat[b]);item.scalar("mode",static_cast<int>(mesh.rb_update_modes[b]));item.scalar("translate",updates_rigid_translation(mesh.rb_update_modes[b]));item.scalar("rotate",updates_rigid_orientation(mesh.rb_update_modes[b]));item.key("com");vector(out,state.x_coms[b]);item.key("velocity");vector(out,state.v_coms[b]);item.key("orientation");vector(out,state.orientations[b]);item.key("omega");vector(out,state.omega[b]);item.end();});
        o.key("rigid_pins");rigid_pins(out,p.rigid_pins);boundary_fields(out,o);o.key("motion");motions(out);
        o.key("static_geometry");Object visual_obj(out);visual_obj.key("positions");vectors(out,visual);visual_obj.key("tris");cells(out,visual_tris,3);visual_obj.scalar("dynamic",args.example==2||args.example==3);visual_obj.end();
        o.key("output");Object output(out);output.string("outdir",args.outdir);output.string("format",args.format);output.scalar("num_frames",args.num_frames);output.scalar("restart_frame",args.restart_frame);output.scalar("write_substeps",p.write_substeps);output.end();o.scalar("uses_general_solver",general);o.end();out<<'\n'<<std::flush;
    }
};
}
int main(int argc,char** argv){
    std::locale::global(std::locale::classic());
    std::ostream protocol(std::cout.rdbuf());protocol<<std::setprecision(17);
    std::cout.rdbuf(std::cerr.rdbuf());
    try{Scene scene;if(!scene.args.parse(argc,argv))return 1;scene.build();scene.write(protocol);std::string line;
        while(std::getline(std::cin,line)){if(line.find_first_not_of(" \t\r")==std::string::npos)continue;std::istringstream in(line);double t;std::string extra;if(!(in>>t)||in>>extra||!std::isfinite(t)||t<0)throw std::invalid_argument("Boundary request must be one finite nonnegative time");scene.update(t);scene.boundary(protocol,t);}return 0;
    }catch(const std::exception& e){std::cerr<<"Newton IPC scene export: "<<e.what()<<'\n';return 1;}
}
