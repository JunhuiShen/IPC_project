"""Our rigid GS equations on the upstream float32 Model/State convention.

Public quaternions use Newton/Warp's xyzw convention. Native derivatives use
the CPU solver's normalized scalar-first quaternion and are converted at the
boundary. Outputs are float32; the small signed-curvature/LDLT calculations use
the maintained binary64 device routines to retain their robustness.
"""

import warp as wp

from .warp_build import configure_module as _configure
from .contact import FullPacket


HEADERS=("rigid.cuh","rigid_pins.cuh","rigid_solve.cuh")

_HELPERS=r"""
#ifndef IPC_WARP_RIGID_F32_BOUNDARY
#define IPC_WARP_RIGID_F32_BOUNDARY
namespace ipc_warp_rigid {
CUDA_CALLABLE inline bool unit(float x,float y,float z,float w,double* q) {
    q[0]=w;q[1]=x;q[2]=y;q[3]=z;
    const double norm=sqrt(((q[0]*q[0]+q[1]*q[1])+q[2]*q[2])+q[3]*q[3]);
    if(!(norm>0)||!isfinite(norm))return false;
    for(int k=0;k<4;++k)q[k]/=norm;
    return true;
}
CUDA_CALLABLE inline void vec(const wp::vec_t<3,float>& input,double* output) {
    for(int k=0;k<3;++k)output[k]=input[k];
}
}
#endif
"""


def configure_module(module=None):
    return _configure(__name__ if module is None else module,HEADERS,extra_preamble=_HELPERS)


_UNIT=r"""
    double q[4];
    const bool unit=ipc_warp_rigid::unit(previous_q[0],previous_q[1],previous_q[2],previous_q[3],q);
"""
_OUTPUT=r"""
    for(int a=0;a<3;++a){gradient[a]=static_cast<float>(value.gradient[a]);
        for(int b=0;b<3;++b)hessian.data[a][b]=static_cast<float>(value.hessian[3*a+b]);}
"""


@wp.func_native(_UNIT+r"""
    wp::vec_t<4,float> out;
    if(!unit){for(int k=0;k<4;++k)out[k]=static_cast<float>(NAN);return out;}
    double angular[3],target[4];ipc_warp_rigid::vec(omega,angular);
    ipc_gpu::solver::rigid_orientation(q,angular,dt,target);
    out[0]=static_cast<float>(target[1]);out[1]=static_cast<float>(target[2]);out[2]=static_cast<float>(target[3]);out[3]=static_cast<float>(target[0]);
    return out;
""")
def _orientation(previous_q: wp.quat,omega: wp.vec3,dt: float)->wp.vec4: ...


@wp.func
def orientation(previous_q: wp.quat,omega: wp.vec3,dt: float):
    value=_orientation(previous_q,omega,dt)
    return wp.normalize(wp.quat(value[0],value[1],value[2],value[3]))


@wp.func_native(r"""
    double q[4],x[3],center[3],result[3];
    wp::vec_t<3,float> output;
    if(!ipc_warp_rigid::unit(rotation[0],rotation[1],rotation[2],rotation[3],q)) {
        for(int k=0;k<3;++k)output[k]=static_cast<float>(NAN);return output;
    }
    ipc_warp_rigid::vec(reference,x);ipc_warp_rigid::vec(com,center);
    ipc_gpu::solver::rigid_world_position(x,center,q,result);
    for(int k=0;k<3;++k)output[k]=static_cast<float>(result[k]);return output;
""")
def world_point(reference: wp.vec3,com: wp.vec3,rotation: wp.quat)->wp.vec3: ...


@wp.func
def world_position(reference: wp.vec3,pose: wp.transform):
    return world_point(reference,wp.transform_get_translation(pose),wp.transform_get_rotation(pose))


@wp.func_native(_UNIT+r"""
    double angular[3],x[3],j[9],c[3][9];
    ipc_warp_rigid::vec(omega,angular);ipc_warp_rigid::vec(reference,x);
    const auto kin=ipc_gpu::solver::rigid_kinematics(q,angular,dt);
    ipc_gpu::solver::rigid_reference_derivatives(x,kin,j,c);
    for(int r=0;r<3;++r)for(int s=0;s<3;++s){
        jacobian.data[r][s]=unit?static_cast<float>(j[3*r+s]):static_cast<float>(NAN);
        cx.data[r][s]=unit?static_cast<float>(c[0][3*r+s]):static_cast<float>(NAN);
        cy.data[r][s]=unit?static_cast<float>(c[1][3*r+s]):static_cast<float>(NAN);
        cz.data[r][s]=unit?static_cast<float>(c[2][3*r+s]):static_cast<float>(NAN);
    }
""")
def _reference(reference: wp.vec3,previous_q: wp.quat,omega: wp.vec3,dt: float,
               jacobian: wp.ref[wp.mat33],cx: wp.ref[wp.mat33],cy: wp.ref[wp.mat33],cz: wp.ref[wp.mat33]): ...


@wp.func
def reference_derivatives(reference: wp.vec3,previous_q: wp.quat,omega: wp.vec3,dt: float):
    j=wp.mat33();cx=wp.mat33();cy=wp.mat33();cz=wp.mat33()
    _reference(reference,previous_q,omega,dt,j,cx,cy,cz)
    return j,cx,cy,cz


@wp.func
def translation_inertia(com: wp.vec3,previous_com: wp.vec3,previous_velocity: wp.vec3,
                        mass: float,gravity_y: float,dt: float):
    gradient=mass*(com-previous_com-dt*previous_velocity)
    gradient[1]-=dt*dt*mass*gravity_y
    return gradient,mass*wp.identity(n=3,dtype=float)


@wp.func_native(_UNIT+r"""
    double old_angular[3],angular[3],moment[9];
    ipc_warp_rigid::vec(previous_omega,old_angular);ipc_warp_rigid::vec(omega,angular);
    for(int r=0;r<3;++r)for(int s=0;s<3;++s)moment[3*r+s]=second_moment.data[r][s];
    const auto prepared=ipc_gpu::solver::build_rigid_inertia(q,old_angular,angular,dt,moment);
    auto value=ipc_gpu::evaluate_rigid_inertia(prepared);
    if(!unit){for(int k=0;k<3;++k)value.gradient[k]=NAN;}
"""+_OUTPUT)
def _rotation(previous_q: wp.quat,previous_omega: wp.vec3,omega: wp.vec3,dt: float,
              second_moment: wp.mat33,gradient: wp.ref[wp.vec3],hessian: wp.ref[wp.mat33]): ...


@wp.func
def rotation_inertia(previous_q: wp.quat,previous_omega: wp.vec3,omega: wp.vec3,dt: float,second_moment: wp.mat33):
    g=wp.vec3();h=wp.mat33()
    _rotation(previous_q,previous_omega,omega,dt,second_moment,g,h)
    return g,h


@wp.func_native(_UNIT+r"""
    using namespace ipc_gpu;
    if(packet[176]!=0){for(int k=0;k<3;++k)gradient[k]=static_cast<float>(NAN);return;}
    if(packet[177]==0){for(int k=0;k<3;++k)gradient[k]=0;for(int r=0;r<3;++r)for(int s=0;s<3;++s)hessian.data[r][s]=0;return;}
    bool selected[4]{};RigidPreparedContact p{};
    p.segment_segment=kind!=0;
    p.flags=rotational?RigidOrientationGradient|RigidOrientationHessian|RigidUpperCrossBlocks
        :RigidTranslationGradient|RigidTranslationHessian;
    p.flags|=RigidFriction;p.update_mask=rotational?2u:1u;
    double angular[3];ipc_warp_rigid::vec(omega,angular);
    const auto kin=ipc_gpu::solver::rigid_kinematics(q,angular,dt);
    bool valid=unit,any=false;
    for(int i=0;i<4;++i){
        const int vertex=nodes[i];
        if(vertex<0||vertex>=owners.shape[0]||vertex>=references.shape[0]){valid=false;continue;}
        selected[i]=owners.data[vertex]==body;
        if(!selected[i])continue;
        any=true;
        p.friction_weights[i]=packet[1+i];
        for(int k=0;k<3;++k)p.gradients[i][k]=packet[8+3*i+k];
        if(rotational){double x[3];ipc_warp_rigid::vec(references.data[vertex],x);
            ipc_gpu::solver::rigid_reference_derivatives(x,kin,p.jacobians[i],p.curvature[i]);}
    }
    for(int k=0;k<3;++k)p.friction_gradient[k]=packet[164+k];
    for(int k=0;k<9;++k)p.friction_hessian[k]=packet[167+k];
    if(!rotational&&kind==0&&any){for(int k=0;k<9;++k)p.translation_hessian[k]=packet[20+k];}
    else for(int i=0;i<4;++i)if(selected[i])for(int j=i;j<4;++j)if(selected[j])
        for(int r=0;r<3;++r)for(int s=0;s<3;++s){
            p.hessians[i][j][3*r+s]=packet[20+36*i+9*j+3*r+s];
            if(!rotational&&i!=j)p.hessians[j][i][3*s+r]=p.hessians[i][j][3*r+s];
        }
    const auto result=evaluate_rigid(p);
    const double* g=rotational?result.barrier.orientation_gradient:result.barrier.translation_gradient;
    const double* h=rotational?result.barrier.orientation_orientation_hessian:result.barrier.translation_translation_hessian;
    const double* fg=rotational?result.friction.orientation_gradient:result.friction.translation_gradient;
    const double* fh=rotational?result.friction.orientation_orientation_hessian:result.friction.translation_translation_hessian;
    for(int r=0;r<3;++r){gradient[r]=valid?static_cast<float>(normal_scale*g[r]+fg[r]):static_cast<float>(NAN);
        for(int s=0;s<3;++s)hessian.data[r][s]=static_cast<float>(normal_scale*h[3*r+s]+fh[3*r+s]);}
""")
def _contact(packet: FullPacket,nodes: wp.vec4i,owners: wp.array[int],references: wp.array[wp.vec3],
              body: int,previous_q: wp.quat,omega: wp.vec3,dt: float,kind: int,rotational: int,normal_scale: float,
              gradient: wp.ref[wp.vec3],hessian: wp.ref[wp.mat33]): ...


@wp.func
def contact_terms(packet: FullPacket,nodes: wp.vec4i,owners: wp.array[int],references: wp.array[wp.vec3],
                  body: int,previous_q: wp.quat,omega: wp.vec3,dt: float,kind: int,rotational: int,normal_scale: float):
    g=wp.vec3();h=wp.mat33()
    _contact(packet,nodes,owners,references,body,previous_q,omega,dt,kind,rotational,normal_scale,g,h)
    return g,h


@wp.func_native(_UNIT+r"""
    ipc_gpu::solver::RigidPinConstraint pin{};double center[3],angular[3];
    ipc_warp_rigid::vec(com,center);ipc_warp_rigid::vec(omega,angular);
    for(int k=0;k<3;++k){pin.local_position[k]=reference[k];pin.initial_target[k]=target[k];}
    const auto kin=ipc_gpu::solver::rigid_kinematics(q,angular,dt);
    ipc_gpu::RigidDerivatives result{};
    valid=unit&&ipc_gpu::solver::evaluate_rigid_pin(pin,center,kin,dt,kpin,0.0,rotational!=0,result);
    const double* g=rotational?result.orientation_gradient:result.translation_gradient;
    const double* h=rotational?result.orientation_orientation_hessian:result.translation_translation_hessian;
    for(int r=0;r<3;++r){gradient[r]=static_cast<float>(g[r]);for(int s=0;s<3;++s)hessian.data[r][s]=static_cast<float>(h[3*r+s]);}
""")
def _pin(reference: wp.vec3,target: wp.vec3,com: wp.vec3,previous_q: wp.quat,omega: wp.vec3,
          dt: float,kpin: float,rotational: int,gradient: wp.ref[wp.vec3],hessian: wp.ref[wp.mat33],valid: wp.ref[int]): ...


@wp.func
def point_pin_derivatives(reference: wp.vec3,target: wp.vec3,com: wp.vec3,previous_q: wp.quat,
                          omega: wp.vec3,dt: float,kpin: float,rotational: int):
    g=wp.vec3();h=wp.mat33();valid=int(0)
    _pin(reference,target,com,previous_q,omega,dt,kpin,rotational,g,h,valid)
    return g,h,valid


@wp.func_native(r"""
    wp::vec_t<3,float> target;
    bool okay=isfinite(rate)&&isfinite(time)&&isfinite(ramp)&&ramp>=0&&isfinite(start)&&isfinite(duration);
    double target64[3],offset[3],axis64[3];
    for(int k=0;k<3;++k){target64[k]=initial[k];offset[k]=static_cast<double>(initial[k])-center[k];axis64[k]=axis[k];
        okay=okay&&isfinite(initial[k])&&isfinite(center[k])&&isfinite(translation[k]);}
    if(rate!=0&&!(ramp>0&&time<=0)){
        double largest=fmax(fabs(axis64[0]),fmax(fabs(axis64[1]),fabs(axis64[2])));
        okay=okay&&largest>0&&isfinite(largest);
        if(okay){
            for(int k=0;k<3;++k)axis64[k]/=largest;
            const double norm=sqrt((axis64[0]*axis64[0]+axis64[1]*axis64[1])+axis64[2]*axis64[2]);
            for(int k=0;k<3;++k)axis64[k]/=norm;
            double rotation_time=time;
            if(ramp>0){if(time>=ramp)rotation_time=static_cast<double>(time)-0.5*static_cast<double>(ramp);
                else {const double u=static_cast<double>(time)/ramp;rotation_time=static_cast<double>(ramp)*u*u*u*u*(2.5+u*(-3.0+u));}}
            const double theta=static_cast<double>(rate)*rotation_time,c=cos(theta),s=sin(theta);
            okay=okay&&isfinite(theta);
            if(axis[0]==0&&axis[1]==0&&axis[2]>0){
                target64[0]=center[0]+(c*offset[0]-s*offset[1]);
                target64[1]=center[1]+(s*offset[0]+c*offset[1]);target64[2]=center[2]+offset[2];
            }else{
                const double dot=(axis64[0]*offset[0]+axis64[1]*offset[1])+axis64[2]*offset[2];
                const double cross[3]{axis64[1]*offset[2]-axis64[2]*offset[1],axis64[2]*offset[0]-axis64[0]*offset[2],axis64[0]*offset[1]-axis64[1]*offset[0]};
                for(int k=0;k<3;++k)target64[k]=center[k]+c*offset[k]+s*cross[k]+((1-c)*dot)*axis64[k];
            }
        }
    }
    const bool moves=translation[0]!=0||translation[1]!=0||translation[2]!=0;
    okay=okay&&(!moves||duration>0);
    if(moves&&time>start&&okay){
        const double elapsed=static_cast<double>(time)-start,u=elapsed>=duration?1.0:elapsed/duration;
        const double weight=u*u*u*(10.0+u*(-15.0+6.0*u));
        for(int k=0;k<3;++k)target64[k]+=weight*translation[k];
    }
    valid=okay?1:0;
    for(int k=0;k<3;++k){target[k]=okay?static_cast<float>(target64[k]):static_cast<float>(NAN);if(!isfinite(target[k]))valid=0;}
    return target;
""")
def _target(initial: wp.vec3,center: wp.vec3,axis: wp.vec3,rate: float,time: float,ramp: float,
            translation: wp.vec3,start: float,duration: float,valid: wp.ref[int])->wp.vec3: ...


@wp.func
def pin_target(initial: wp.vec3,center: wp.vec3,axis: wp.vec3,rate: float,time: float,
               ramp: float,translation: wp.vec3,start: float,duration: float):
    valid=int(0)
    target=_target(initial,center,axis,rate,time,ramp,translation,start,duration,valid)
    return target,valid


@wp.func_native(r"""
    double H[9],g[3],x[3];
    for(int r=0;r<3;++r){g[r]=force[r];for(int s=0;s<3;++s)H[3*r+s]=hessian.data[r][s];}
    valid=ipc_gpu::solver::rigid_ldlt3(H,g,x)?1:0;
    for(int k=0;k<3;++k)delta[k]=static_cast<float>(x[k]);
""")
def _solve(hessian: wp.mat33,force: wp.vec3,delta: wp.ref[wp.vec3],valid: wp.ref[int]): ...


@wp.func
def solve3(hessian: wp.mat33,gradient: wp.vec3):
    delta=wp.vec3();valid=int(0)
    _solve(hessian,gradient,delta,valid)
    return delta,valid


configure_module()
