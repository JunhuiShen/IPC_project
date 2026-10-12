#include "resident_rigid_pipeline.h"
#include "resident_rigid_ccd.cuh"
#include "ccd.h"
#include "rigid_body_ipc.h"

#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu::resident;
using D3=std::array<double,3>;
using D4=std::array<double,4>;
using D9=std::array<double,9>;
static void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void require(bool b,const std::string& s) { if(!b) throw std::runtime_error(s); }
static void close(double a,double b,const char* label) {
    require(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=2e-9*(1+std::abs(b)),label);
}
template<class T> struct Device {
    T* p=nullptr; std::size_t count;
    explicit Device(std::size_t n):count(n) { if(n)check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))); }
    explicit Device(const std::vector<T>& v):Device(v.size()) { set(v); }
    ~Device(){if(p)cudaFree(p);}
    void set(const std::vector<T>& v) { require(v.size()==count,"wrong upload size");if(count)check(cudaMemcpy(p,v.data(),count*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> get()const {std::vector<T> v(count);if(count)check(cudaMemcpy(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
template<std::size_t N> static auto ptr(Device<std::array<double,N>>& d)->double(*)[N] {
    return reinterpret_cast<double(*)[N]>(d.p);
}
static Vec3 vec(const D3& a){return Vec3(a[0],a[1],a[2]);}
static D3 pod(const Vec3& a){return {a[0],a[1],a[2]};}

int main(){
    try{
        constexpr std::size_t nb=4,nn=15; constexpr double dt=.1;
        std::vector<int> body_blocks{0,1,2,3},node_body(nn,-1),nodes(12),colors{0,1,1,1};
        std::vector<std::size_t> offsets{0,3,6,9,12};
        std::vector<D3> refs(nn),positions(nn),velocities(nn),com(nb),velocity(nb),omega(nb);
        std::vector<D4> q(nb,D4{1,0,0,0});
        for(std::size_t b=0;b<nb;++b){
            com[b]={10.0*b,0,0}; velocity[b]={.2,-.1,.3}; omega[b]={0,0,.5};
            for(int j=0;j<3;++j){const int n=3*b+j;nodes[n]=n;node_body[n]=static_cast<int>(b);refs[n]={.1*j,-2.0+.5*j,.2*j};positions[n]=pod(vec(com[b])+vec(refs[n]));}
        }
        positions[12]={1,0,-1}; positions[13]={3,0,-1}; positions[14]={2,0,1};
        const auto initial_positions=positions;
        std::vector<unsigned char> translate{1,1,0,0},rotate{1,0,1,0};
        std::vector<D9> moment(nb,D9{2,.1,0,.1,3,.2,0,.2,4});
        Device<int> d_body_blocks(body_blocks),d_node_body(node_body),d_nodes(nodes),d_colors(colors),error(std::vector<int>{0});
        Device<std::size_t> d_offsets(offsets);
        Device<unsigned char>d_translate(translate),d_rotate(rotate);
        Device<double> mass(std::vector<double>(nb,2.0));
        Device<D3> d_refs(refs),d_positions(positions),d_velocities(velocities),previous(nn),predicted(nn),d_com(com),d_velocity(velocity),d_omega(omega);
        Device<D3> previous_com(nb),previous_velocity(nb),previous_omega(nb),proposed_com(nb),proposed_omega(nb),accepted_com(nb),accepted_omega(nb),accepted_proxy(nn);
        Device<D4> orientation(q),previous_orientation(nb),proposed_orientation(nb),accepted_orientation(nb);
        Device<D9> d_moment(moment);
        Device<ipc_gpu::Derivatives>systems(nb);
        Device<BpAabb>body_boxes(nb),node_boxes(nn);
        RigidScene scene{nb,d_body_blocks.p,d_node_body.p,ptr(d_refs),d_offsets.p,d_nodes.p,mass.p,ptr(d_moment),d_translate.p,d_rotate.p};
        RigidState state{ptr(d_com),ptr(d_velocity),ptr(orientation),ptr(d_omega),ptr(previous_com),ptr(previous_velocity),ptr(previous_orientation),ptr(previous_omega)};
        RigidWorkspace work{systems.p,ptr(proposed_com),ptr(proposed_omega),ptr(proposed_orientation),ptr(accepted_com),ptr(accepted_orientation),ptr(accepted_omega)};
        DeformableState particles{ptr(d_positions),ptr(d_velocities),ptr(previous),ptr(predicted)};
        auto success=[&]{require(error.get()[0]==0,"resident rigid pipeline device error "+std::to_string(error.get()[0]));};
        begin_rigid_substep(scene,state,error.p);success();
        const auto old_velocity=previous_velocity.get(),old_omega=previous_omega.get(),zero_omega=d_omega.get();
        for(std::size_t b=0;b<nb;++b)for(int k=0;k<3;++k){
            close(old_velocity[b][k],translate[b]?velocity[b][k]:0,"constrained previous velocity");
            close(old_omega[b][k],rotate[b]?omega[b][k]:0,"constrained previous omega");
            close(zero_omega[b][k],0,"iteration omega not reset");
        }
        build_rigid_blue_boxes(scene,state,nn,2.0,body_boxes.p,node_boxes.p,error.p);success();
        const auto nbounds=node_boxes.get();
        for(int n=0;n<12;++n)for(int k=0;k<3;++k)
            require(nbounds[n].lo[k]<=positions[n][k]&&nbounds[n].hi[k]>=positions[n][k],"proxy missing from blue box");
        for(int color:{0,1}){
            assemble_rigid_inertia_color(scene,state,work,d_colors.p,color,false,dt,-9.81,error.p);
            propose_rigid_color(scene,state,work,d_colors.p,color,false,body_boxes.p,dt,1.0,error.p);success();
        }
        const auto predicted_com=proposed_com.get();
        for(std::size_t b=0;b<nb;++b)for(int k=0;k<3;++k){
            const double expected=translate[b]?com[b][k]+dt*velocity[b][k]+(k==1?-9.81*dt*dt:0):com[b][k];
            close(predicted_com[b][k],expected,"COM inertia proposal");
        }
        // Exercise pivoted LDLT independently of diagonal mass matrices.
        std::vector<Mat33> matrices;
        Mat33 A;A<<4,1,.2,1,2,-.3,.2,-.3,3;matrices.push_back(A);
        A=Mat33::Zero();A.diagonal()<<1,-2,3;matrices.push_back(A);
        A=Mat33::Zero();A.diagonal()<<1,0,2;matrices.push_back(A);
        matrices.push_back(Mat33::Zero());
        const Vec3 gradient(.2,-.7,1.3);
        for(const auto& H:matrices){
            std::vector<ipc_gpu::Derivatives> d(nb);
            for(int r=0;r<3;++r){d[0].gradient[r]=gradient[r];for(int c=0;c<3;++c)d[0].hessian[3*r+c]=H(r,c);}
            systems.set(d);propose_rigid_color(scene,state,work,d_colors.p,0,false,body_boxes.p,dt,.1,error.p);success();
            const Vec3 expected=vec(com[0])-.1*H.ldlt().solve(gradient);
            const auto actual=proposed_com.get()[0];for(int k=0;k<3;++k)close(actual[k],expected[k],"pivoted LDLT proposal");
        }
        // Queue-owned, represented translation endpoint. Commit must copy
        // this buffer byte-for-byte, rather than recompute from COM.
        const Vec3 displacement(.01,-.02,.03);
        auto committed_com=com;committed_com[0]=pod(vec(com[0])+displacement);
        auto accepted_positions=positions;
        for(int n=0;n<3;++n)accepted_positions[n]=pod(vec(positions[n])+displacement);
        accepted_com.set(committed_com);accepted_proxy.set(accepted_positions);
        commit_rigid_translation_color(scene,state,work,particles,d_colors.p,0,ptr(accepted_proxy),error.p);success();
        const auto translated=d_positions.get();
        require(std::memcmp(translated.data(),accepted_positions.data(),nn*sizeof(D3))==0,"translation changed validated endpoints");
        const auto translated_com=d_com.get();
        // Orientation inertia preparation at updated COM is independently
        // compared to original CPU derivatives, before contact assembly.
        assemble_rigid_inertia_color(scene,state,work,d_colors.p,0,true,dt,-9.81,error.p);success();
        const Mat33 inertia_moment=(Mat33()<<2,.1,0,.1,3,.2,0,.2,4).finished();
        const auto expected_inertia=inertia_rotation_gradient_hessian(Vec3::Zero(),Vec4(1,0,0,0),vec(omega[0]),dt,inertia_moment);
        const auto actual_inertia=systems.get()[0];
        for(int r=0;r<3;++r){close(actual_inertia.gradient[r],expected_inertia.first[r],"rotation inertia gradient");for(int c=0;c<3;++c)close(actual_inertia.hessian[3*r+c],expected_inertia.second(r,c),"rotation inertia Hessian");}
        // Huge finite raw Newton omega proposals must be bounded before any
        // quaternion exponentiation; a direct squared norm would overflow.
        std::vector<ipc_gpu::Derivatives> enormous_systems(nb);
        const Vec3 raw_direction(1,-2,3),axis=-raw_direction.normalized();
        for(int k=0;k<3;++k){enormous_systems[0].gradient[k]=raw_direction[k]*1e300;enormous_systems[0].hessian[3*k+k]=1;}
        systems.set(enormous_systems);
        for(double cap:{0.0,.07,3.14159265358979323846}){
            propose_rigid_color(scene,state,work,d_colors.p,0,true,body_boxes.p,dt,1.0,error.p,cap);success();
            const auto bounded_q=proposed_orientation.get()[0];
            const auto bounded_omega=proposed_omega.get()[0];
            const Vec3 expected_capped_omega=axis*(cap/dt);
            const Vec4 expected_capped_q=quaternion_from_angular_velocity(Vec4(1,0,0,0),expected_capped_omega,dt);
            for(int k=0;k<4;++k)close(bounded_q[k],expected_capped_q[k],"bounded huge Newton orientation");
            for(int k=0;k<3;++k)close(bounded_omega[k],expected_capped_omega[k],"bounded huge Newton omega");
            const double actual_angle=2*std::atan2(std::hypot(bounded_q[1],std::hypot(bounded_q[2],bounded_q[3])),bounded_q[0]);
            require(actual_angle<=cap+1e-12,"Newton orientation exceeds configured cap");
        }
        // Rebuild-anchored caps remain valid across multiple proposed updates,
        // including translation roundoff in the represented proxy positions.
        Device<D4> anchors(nb);
        constexpr double cap_radius=.07,com_radius=.02;
        build_rigid_blue_boxes(scene,state,nn,com_radius,body_boxes.p,node_boxes.p,error.p,cap_radius,ptr(anchors),ptr(d_positions));success();
        const auto anchored_boxes=node_boxes.get();const auto anchor_values=anchors.get();
        require(anchor_values==q,"blue-box quaternion anchor snapshot differs");
        for(int n=0;n<12;++n)for(int k=0;k<3;++k)
            require(anchored_boxes[n].lo[k]<=translated[n][k]&&anchored_boxes[n].hi[k]>=translated[n][k],"anchored cap misses represented position");
        require(anchored_boxes[0].hi[0]-anchored_boxes[0].lo[0]<1.0,"restricted cap unexpectedly uses full sphere");
        const Vec3 axes[]={Vec3::UnitX(),Vec3::UnitY(),Vec3::UnitZ(),Vec3(1,2,3).normalized()};
        for(const auto& axis_sample:axes)for(double angle:{-cap_radius,0.0,cap_radius})for(int n=0;n<3;++n){
            const Vec4 q_sample=quaternion_from_angular_velocity(Vec4(1,0,0,0),axis_sample*angle,1.0);
            const Vec3 offset=quaternion_rotate(q_sample,vec(refs[n]));
            for(int k=0;k<3;++k){
                const double lo=translated_com[0][k]-com_radius+offset[k],hi=translated_com[0][k]+com_radius+offset[k];
                require(anchored_boxes[n].lo[k]<=lo&&anchored_boxes[n].hi[k]>=hi,"angular cap fails to enclose reachable pose");
            }
        }
        auto staged_q=q;auto staged_omega=d_omega.get();
        const Vec3 current_omega=axis*(.06/dt);
        const Vec4 current_q=quaternion_from_angular_velocity(Vec4(1,0,0,0),current_omega,dt);
        for(int k=0;k<4;++k)staged_q[0][k]=current_q[k];
        staged_omega[0]=pod(current_omega);orientation.set(staged_q);d_omega.set(staged_omega);
        propose_rigid_color(scene,state,work,d_colors.p,0,true,body_boxes.p,dt,1.0,error.p,cap_radius,ptr(anchors));success();
        const auto anchor_limited=proposed_orientation.get()[0];
        const double anchor_angle=2*std::atan2(std::hypot(anchor_limited[1],std::hypot(anchor_limited[2],anchor_limited[3])),anchor_limited[0]);
        require(anchor_angle<=cap_radius+1e-12,"successive proposal escaped rebuild cap");
        require(anchor_angle>.069,"anchor cap unexpectedly suppressed valid progress");
        orientation.set(q);d_omega.set(std::vector<D3>(nb,D3{0,0,0}));
        build_rigid_blue_boxes(scene,state,nn,2.0,body_boxes.p,node_boxes.p,error.p);success();
        // A known point/triangle rotational hit, after the COM translation.
        auto target=q;target[0]={0,0,0,1};proposed_orientation.set(target);
        Device<BpContact>contacts(std::vector<BpContact>{{{0,12,13,14},0}});
        BroadPhaseView bp;bp.contacts=contacts.p;bp.contact_count=1;
        double toi=0;
        const bool hit=point_triangle_rb_rotation_ccd(vec(translated[0]),vec(translated_com[0]),Vec4(0,0,0,1),Vec4(1,0,0,0),vec(positions[12]),vec(positions[13]),vec(positions[14]),toi);
        require(hit,"rotation fixture has no collision");
        clip_rigid_rotation_color(scene,state,work,particles,bp,d_colors.p,0,dt,.9,true,ptr(accepted_proxy),error.p);success();
        const Vec4 expected_q=interpolate_orientation_full_arc(Vec4(1,0,0,0),Vec4(0,0,0,1),.9*toi);
        const auto accepted_q=accepted_orientation.get()[0];
        for(int k=0;k<4;++k)close(accepted_q[k],expected_q[k],"curved CCD accepted orientation");
        const auto before_commit=accepted_proxy.get();
        commit_rigid_rotation_color(scene,state,work,particles,d_colors.p,0,ptr(accepted_proxy),error.p);success();
        const auto rotated=d_positions.get();
        for(int n=0;n<3;++n)require(std::memcmp(rotated[n].data(),before_commit[n].data(),sizeof(D3))==0,"rotation changed accepted endpoint");
        for(int n=3;n<static_cast<int>(nn);++n)require(rotated[n]==positions[n],"rotation touched another color/object");
        finish_rigid_substep(scene,state,particles,dt,error.p);success();
        const auto final_velocity=d_velocity.get(),final_omega=d_omega.get(),proxy_velocity=d_velocities.get();
        const Vec3 expected_omega=angular_velocity_from_orientation_full_arc(expected_q,Vec4(1,0,0,0),dt);
        for(int k=0;k<3;++k){close(final_velocity[0][k],displacement[k]/dt,"body velocity");close(final_omega[0][k],expected_omega[k],"body angular velocity");}
        for(int n=0;n<3;++n){const Vec3 expected=displacement/dt+expected_omega.cross(quaternion_rotate(expected_q,vec(refs[n])));for(int k=0;k<3;++k)close(proxy_velocity[n][k],expected[k],"proxy velocity");}
        for(int b:{2,3})for(int k=0;k<3;++k)close(final_velocity[b][k],0,"constrained COM velocity");
        for(int b:{1,3})for(int k=0;k<3;++k)close(final_omega[b][k],0,"constrained angular velocity");
        // An ambiguous full turn must not commit anything.
        target=orientation.get();for(int k=0;k<4;++k)target[0][k]=-target[0][k];proposed_orientation.set(target);
        const auto original_state=d_positions.get();
        clip_rigid_rotation_color(scene,state,work,particles,bp,d_colors.p,0,dt,.9,true,ptr(accepted_proxy),error.p);
        commit_rigid_rotation_color(scene,state,work,particles,d_colors.p,0,ptr(accepted_proxy),error.p);
        require(error.get()[0]==RigidRotationFailure,"ambiguous rotation was accepted");
        const auto unchanged=d_positions.get();require(std::memcmp(unchanged.data(),original_state.data(),nn*sizeof(D3))==0,"error did not block commit");
        std::cout<<"CUDA rigid pipeline: state/modes, boxes, inertia, LDLT, exact endpoint commits, curved CCD and velocities PASS\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"Resident rigid pipeline test failed: "<<e.what()<<'\n';return 1;}
}
