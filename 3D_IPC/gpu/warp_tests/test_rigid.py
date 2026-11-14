"""Independent rigid/SDF equations on actual float32 Newton state types."""

import math
import unittest

import numpy as np
import warp as wp

from gpu import rigid as r,sdf as s,contact


@wp.func_native(r"""
    ipc_gpu::solver::ContactInput input;
    input.kind=ipc_gpu::solver::ContactKind::PointTriangle;
    input.d_hat=0.1;input.k_barrier=300;input.dt=0.02;input.friction=0;input.eps_v=.001;
    input.hessian_blocks=0xffffu;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k){input.positions[i][k]=positions.data[i][k];input.previous[i][k]=positions.data[i][k];}
    const auto out=ipc_gpu::solver::evaluate_contact(input);
    wp::vec_t<178,double> packet;
    packet[0]=out.distance;
    for(int i=0;i<4;++i)packet[1+i]=out.weights[i];
    for(int k=0;k<3;++k){packet[5+k]=out.normal[k];packet[164+k]=out.friction_relative_gradient[k];}
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)packet[8+3*i+k]=out.barrier_gradient[i][k];
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int k=0;k<9;++k)packet[20+36*i+9*j+k]=out.barrier_hessian[i][j][k];
    for(int k=0;k<9;++k)packet[167+k]=out.friction_relative_hessian[k];
    packet[176]=static_cast<int>(out.status);packet[177]=out.active;return packet;
""")
def _packet(positions: wp.array[wp.vec3])->contact.FullPacket: ...


@wp.kernel
def _contact_packet(positions: wp.array[wp.vec3],out: wp.array[contact.FullPacket]):
    out[0]=_packet(positions)


@wp.kernel
def _contact_pullback(packet: wp.array[contact.FullPacket],owners: wp.array[int],references: wp.array[wp.vec3],
                       rotational: int,g: wp.array[wp.vec3],h: wp.array[wp.mat33]):
    gradient,hessian=r.contact_terms(packet[0],wp.vec4i(0,1,2,3),owners,references,0,
        wp.quat_identity(),wp.vec3(0),.02,0,rotational,.02*.02*300.)
    g[0]=gradient;h[0]=hessian


@wp.kernel
def _kinematics(reference: wp.vec3,com: wp.vec3,previous: wp.quat,omega: wp.vec3,dt: float,
                qout: wp.array[wp.quat],xout: wp.array[wp.vec3],matrices: wp.array[wp.mat33]):
    q=r.orientation(previous,omega,dt)
    qout[0]=q
    xout[0]=r.world_position(reference,wp.transform(com,q))
    j,cx,cy,cz=r.reference_derivatives(reference,previous,omega,dt)
    matrices[0]=j;matrices[1]=cx;matrices[2]=cy;matrices[3]=cz


@wp.kernel
def _rotation(previous: wp.quat,oldomega: wp.vec3,omega: wp.vec3,dt: float,moment: wp.mat33,
              g: wp.array[wp.vec3],h: wp.array[wp.mat33]):
    gradient,hessian=r.rotation_inertia(previous,oldomega,omega,dt,moment)
    g[0]=gradient;h[0]=hessian


@wp.kernel
def _pin(reference: wp.vec3,target: wp.vec3,com: wp.vec3,previous: wp.quat,omega: wp.vec3,
          dt: float,kpin: float,rotational: int,g: wp.array[wp.vec3],h: wp.array[wp.mat33],valid: wp.array[int]):
    gradient,hessian,okay=r.point_pin_derivatives(reference,target,com,previous,omega,dt,kpin,rotational)
    g[0]=gradient;h[0]=hessian;valid[0]=okay


@wp.kernel
def _target(initial: wp.vec3,center: wp.vec3,axis: wp.vec3,rate: float,time: float,ramp: float,
             translation: wp.vec3,start: float,duration: float,out: wp.array[wp.vec3],valid: wp.array[int]):
    target,okay=r.pin_target(initial,center,axis,rate,time,ramp,translation,start,duration)
    out[0]=target;valid[0]=okay


@wp.kernel
def _solve(h: wp.array[wp.mat33],g: wp.array[wp.vec3],out: wp.array[wp.vec3],valid: wp.array[int]):
    i=wp.tid()
    delta,okay=r.solve3(h[i],g[i])
    out[i]=delta;valid[i]=okay


@wp.kernel
def _sdf(positions: wp.array[wp.vec3],previous: wp.array[wp.vec3],colliders: wp.array[s.Collider],params: s.Parameters,
          packets: wp.array[s.Packet],forces: wp.array[wp.vec3],hessians: wp.array[wp.mat33]):
    i=wp.tid()
    packet=s.evaluate(positions[i],previous[i],colliders,params)
    f,h=s.vertex_terms(packet,wp.float32(params.dt))
    packets[i]=packet;forces[i]=f;hessians[i]=h


r.configure_module(__name__)
s.configure_module(__name__)
contact.configure_module(__name__, include_filter=False)


def quat_multiply(a,b):
    aw,ax,ay,az=a;bw,bx,by,bz=b
    return np.array([aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,
                     aw*by+ay*bw+az*bx-ax*bz,aw*bz+az*bw+ax*by-ay*bx])


def orientation(q,omega,dt):
    magnitude=np.linalg.norm(omega)
    exp=np.r_[np.cos(.5*dt*magnitude),(.5*dt if magnitude==0 else np.sin(.5*dt*magnitude)/magnitude)*omega]
    return quat_multiply(exp,q/np.linalg.norm(q))


def matrix(q):
    w=q[0];v=q[1:];x,y,z=v
    cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return (w*w-np.dot(v,v))*np.eye(3)+2*np.outer(v,v)+2*w*cross


def rotation_predictor(q,omega,dt):
    q=q/np.linalg.norm(q)
    old=orientation(q,-omega,dt)
    qdot=.5*quat_multiply(np.r_[0,omega],old)
    h=1e-6
    derivative=(matrix(q+h*qdot)-matrix(q-h*qdot))/(2*h)
    derivative-=2*np.dot(q,qdot)*matrix(q)
    return matrix(q)+dt*derivative


def fd_gradient(fun,x,h=1e-5):
    result=np.zeros_like(x,dtype=np.float64)
    for a in range(len(x)):
        plus=x.copy();minus=x.copy();plus[a]+=h;minus[a]-=h
        result[a]=(fun(plus)-fun(minus))/(2*h)
    return result


def fd_hessian(fun,x,h=1e-4):
    result=np.zeros((len(x),len(x)))
    for a in range(len(x)):
        for b in range(len(x)):
            pp=x.copy();pm=x.copy();mp=x.copy();mm=x.copy()
            pp[a]+=h;pp[b]+=h;pm[a]+=h;pm[b]-=h
            mp[a]-=h;mp[b]+=h;mm[a]-=h;mm[b]-=h
            result[a,b]=(fun(pp)-fun(pm)-fun(mp)+fun(mm))/(4*h*h)
    return result


class RigidAndSdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init();cls.devices=[wp.get_device("cpu"),*wp.get_cuda_devices()]

    def test_actual_xyzw_state_kinematics_and_curvature(self):
        reference=np.array([.2,-.3,.5],np.float32);com=np.array([1,.5,-.2],np.float32)
        q=np.array([.9,.1,-.2,.3],np.float64);q/=np.linalg.norm(q)
        qxyzw=q[[1,2,3,0]].astype(np.float32);q=qxyzw[[3,0,1,2]].astype(np.float64);q/=np.linalg.norm(q)
        for angular in ([0,0,0],[.3,-.2,.15]):
            omega=np.array(angular,np.float32).astype(np.float64);dt=float(np.float32(.02))
            fun=lambda value:matrix(orientation(q,value,dt))@reference+com
            expected_j=np.column_stack([fd_gradient(lambda z:fun(z)[a],omega) for a in range(3)]).T
            expected_c=np.array([fd_hessian(lambda z:fun(z)[a],omega) for a in range(3)])
            for device in self.devices:
                qo=wp.empty(1,wp.quat,device=device);xo=wp.empty(1,wp.vec3,device=device);ms=wp.empty(4,wp.mat33,device=device)
                wp.launch(_kinematics,1,[wp.vec3(reference),wp.vec3(com),wp.quat(*qxyzw),wp.vec3(omega),dt,qo,xo,ms],device=device)
                np.testing.assert_allclose(xo.numpy()[0],fun(omega),rtol=3e-6,atol=3e-6)
                np.testing.assert_allclose(ms.numpy()[0],expected_j,rtol=1e-4,atol=1e-6)
                np.testing.assert_allclose(ms.numpy()[1:],expected_c,rtol=4e-3,atol=1e-6)

    def test_rigid_rotation_inertia_matches_energy_finite_differences(self):
        qxyzw=np.array([.1,-.2,.15,.95],np.float32)
        q=qxyzw[[3,0,1,2]].astype(np.float64);q/=np.linalg.norm(q)
        old=np.array([.2,-.1,.3],np.float32).astype(np.float64)
        omega=np.array([.3,-.4,.1],np.float32).astype(np.float64)
        moment=np.diag([.4,.7,1.1]).astype(np.float32);dt=float(np.float32(.03))
        predictor=rotation_predictor(q,old,dt)
        def energy(value):
            d=matrix(orientation(q,value,dt))-predictor
            return .5*np.trace(d.T@d@moment)
        expected_g=fd_gradient(energy,omega);expected_h=fd_hessian(energy,omega)
        for device in self.devices:
            g=wp.empty(1,wp.vec3,device=device);h=wp.empty(1,wp.mat33,device=device)
            wp.launch(_rotation,1,[wp.quat(*qxyzw),wp.vec3(old),wp.vec3(omega),dt,wp.mat33(moment),g,h],device=device)
            np.testing.assert_allclose(g.numpy()[0],expected_g,rtol=2e-4,atol=2e-6)
            np.testing.assert_allclose(h.numpy()[0],expected_h,rtol=2e-3,atol=2e-6)

    def test_finite_rigid_pin_keeps_signed_curvature(self):
        reference=np.array([.3,-.2,.4],np.float32);target=np.array([.1,.4,.2],np.float32);com=np.array([.1,.2,.3],np.float32)
        q=np.array([1.,0,0,0]);omega=np.array([.3,-.2,.1],np.float32).astype(np.float64)
        dt=float(np.float32(.04));k=float(np.float32(42.))
        def energy(value):
            difference=com+matrix(orientation(q,value,dt))@reference-target
            return .5*dt*dt*k*np.dot(difference,difference)
        for device in self.devices:
            g=wp.empty(1,wp.vec3,device=device);h=wp.empty(1,wp.mat33,device=device);valid=wp.empty(1,int,device=device)
            wp.launch(_pin,1,[wp.vec3(reference),wp.vec3(target),wp.vec3(com),wp.quat_identity(),wp.vec3(omega),dt,k,1,g,h,valid],device=device)
            self.assertEqual(int(valid.numpy()[0]),1)
            np.testing.assert_allclose(g.numpy()[0],fd_gradient(energy,omega),rtol=3e-4,atol=2e-6)
            np.testing.assert_allclose(h.numpy()[0],fd_hessian(energy,omega),rtol=3e-3,atol=2e-6)

    def test_motor_speed_ramp_and_mount_translation_are_preserved(self):
        for time in (0.,.25,.75,1.5):
            ramp=1.;rate=.8;u=min(1.,max(0.,time/ramp))
            rt=time-.5*ramp if time>=ramp else ramp*u**4*(2.5+u*(-3+u))
            theta=rate*rt
            expected=np.array([np.cos(theta),np.sin(theta),0.])
            v=min(1.,max(0.,(time-.25)/1.))
            expected+=v**3*(10+v*(-15+6*v))*np.array([.2,0,.1])
            for device in self.devices:
                out=wp.empty(1,wp.vec3,device=device);valid=wp.empty(1,int,device=device)
                wp.launch(_target,1,[wp.vec3(1,0,0),wp.vec3(0),wp.vec3(0,0,1),rate,time,ramp,
                    wp.vec3(.2,0,.1),.25,1.,out,valid],device=device)
                self.assertEqual(int(valid.numpy()[0]),1)
                np.testing.assert_allclose(out.numpy()[0],expected,rtol=3e-6,atol=3e-6)

    def test_own_rigid_ldlt_regular_rank_deficient_and_zero(self):
        matrices=np.array([[[4,1,.2],[1,3,.1],[.2,.1,2]],np.diag([1,0,2]),np.zeros((3,3))],dtype=np.float32)
        rhs=np.array([[1,2,-1],[1,1,4],[3,-2,1]],np.float32)
        expected=np.array([np.linalg.solve(matrices[0],rhs[0]),[1,0,2],[0,0,0]],np.float32)
        for device in self.devices:
            out=wp.empty(3,wp.vec3,device=device);valid=wp.empty(3,int,device=device)
            wp.launch(_solve,3,[wp.array(matrices,wp.mat33,device=device),wp.array(rhs,wp.vec3,device=device),out,valid],device=device)
            np.testing.assert_array_equal(valid.numpy(),[1,1,1])
            np.testing.assert_allclose(out.numpy(),expected,rtol=3e-6,atol=3e-6)

    def test_multi_corner_contact_pullback_and_frozen_friction(self):
        x=np.array([[.2,.1,.02],[-1,-1,0],[1,-1,0],[0,1,0]],np.float32)
        dt=float(np.float32(.02));scale=float(np.float32(.02*.02*300.))
        def energy(com,omega):
            tri=(matrix(orientation(np.array([1.,0,0,0]),omega,dt))@x[1:].T).T+com
            normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
            distance=abs(np.dot(x[0]-tri[0],normal))
            return scale*(-(distance-.1)**2*np.log(distance/.1))
        zeros=np.zeros(3)
        for device in self.devices:
            positions=wp.array(x,wp.vec3,device=device);packets=wp.empty(1,contact.FullPacket,device=device)
            owners=wp.array([-1,0,0,0],int,device=device);refs=wp.array(x,wp.vec3,device=device)
            wp.launch(_contact_packet,1,[positions,packets],device=device)
            raw=packets.numpy().copy()
            self.assertEqual(raw[0,176],0);self.assertEqual(raw[0,177],1)
            for rotational in (0,1):
                g=wp.empty(1,wp.vec3,device=device);h=wp.empty(1,wp.mat33,device=device)
                wp.launch(_contact_pullback,1,[packets,owners,refs,rotational,g,h],device=device)
                baseg,baseh=g.numpy()[0].copy(),h.numpy()[0].copy()
                fun=(lambda value:energy(zeros,value)) if rotational else (lambda value:energy(value,zeros))
                np.testing.assert_allclose(baseg,fd_gradient(fun,zeros),rtol=3e-4,atol=2e-6)
                np.testing.assert_allclose(baseh,fd_hessian(fun,zeros),rtol=3e-3,atol=2e-6)
                friction_g=np.array([.03,-.02,.01]);friction_h=np.diag([.2,.3,.4])
                altered=raw.copy();altered[0,164:167]=friction_g;altered[0,167:176]=friction_h.reshape(-1)
                packets.assign(altered)
                wp.launch(_contact_pullback,1,[packets,owners,refs,rotational,g,h],device=device)
                weights=raw[0,2:5]
                if rotational:
                    weighted=np.zeros((3,3))
                    for weight,point in zip(weights,x[1:]):
                        px,py,pz=point
                        weighted+=weight*(-dt*np.array([[0,-pz,py],[pz,0,-px],[-py,px,0]]))
                    fg=weighted.T@friction_g;fh=weighted.T@friction_h@weighted
                else:
                    fg=weights.sum()*friction_g;fh=weights.sum()**2*friction_h
                np.testing.assert_allclose(g.numpy()[0]-baseg,fg,rtol=3e-4,atol=2e-6)
                np.testing.assert_allclose(h.numpy()[0]-baseh,fh,rtol=3e-4,atol=2e-6)
                packets.assign(raw)

    def test_sdf_union_moving_material_force_units_and_float_pose_roundoff(self):
        records=[dict(kind="plane",point=[0,0,0],axis=[0,1,0])]
        records[0]['previous_pose']={'rotation':np.eye(3).tolist(),'translation':[0,0,0]}
        angle=.41;rotation=[[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]]
        records[0]['current_pose']={'rotation':rotation,'translation':[.05,0,0]}
        for device in self.devices:
            raw=wp.array(s.pack_colliders(records),dtype=float,device=device)
            bank=wp.empty(1,s.Collider,device=device);error=wp.zeros(1,int,device=device)
            wp.launch(s.prepare_colliders,1,[raw,bank,1,error],device=device)
            self.assertEqual(int(error.numpy()[0]),0)
            x=wp.array([[0,-.01,0]],wp.vec3,device=device)
            packet=wp.empty(1,s.Packet,device=device);f=wp.empty(1,wp.vec3,device=device);h=wp.empty(1,wp.mat33,device=device)
            p=s.parameters(stiffness=100,epsilon=0,dt=.01,friction=.2,eps_v=.001)
            wp.launch(_sdf,1,[x,x,bank,p,packet,f,h],device=device)
            self.assertEqual(int(packet.numpy()[0,41]),0)
            self.assertAlmostEqual(float(f.numpy()[0,1]),1.,places=5)
            self.assertGreater(float(f.numpy()[0,0]),0)
            self.assertAlmostEqual(float(h.numpy()[0,1,1]),100.,places=4)
            invalid=s.pack_colliders(records);invalid[0,20]=2
            error.zero_();raw.assign(invalid)
            wp.launch(s.prepare_colliders,1,[raw,bank,1,error],device=device)
            self.assertEqual(int(error.numpy()[0]),48)


if __name__=="__main__":unittest.main()
