"""Independent material checks for the actual float32 tiled VBD extension."""

import unittest

import numpy as np
import warp as wp
import newton

from gpu import energy as m
from gpu import kernels
from gpu.scene import SCHEMA, create_model


@wp.kernel
def _triangle(x: wp.array[wp.vec3], inverse: wp.mat22, area: float, mu: float, lam: float,
              fast: wp.array[m.Pack], reference: wp.array[m.Pack]):
    role = wp.tid()
    fast[role] = m.triangle(x[0],x[1],x[2],inverse,area,mu,lam,role)
    reference[role] = m.triangle_reference(x[0],x[1],x[2],inverse,area,mu,lam,role)


@wp.kernel
def _hinge(x: wp.array[wp.vec3], coefficient: float, rest: float, stiffness: float,
           fast: wp.array[m.Pack], reference: wp.array[m.Pack]):
    role = wp.tid()
    fast[role] = m.hinge(x[0],x[1],x[2],x[3],coefficient,rest,stiffness,role)
    reference[role] = m.hinge_reference(x[0],x[1],x[2],x[3],coefficient,rest,stiffness,role)


@wp.kernel
def _tet(x: wp.array[wp.vec3], inverse: wp.mat33, shape: wp.array[wp.vec3], measure: float, mu: float, lam: float,
         fast: wp.array[m.Pack], reference: wp.array[m.Pack]):
    role = wp.tid()
    fast[role] = m.solid(x[0],x[1],x[2],x[3],inverse,shape[role],measure,mu,lam)
    reference[role] = m.solid_reference(x[0],x[1],x[2],x[3],inverse,shape[role],measure,mu,lam)


@wp.kernel
def _solve(hessian: wp.array[wp.mat33], force: wp.array[wp.vec3], delta: wp.array[wp.vec3], valid: wp.array[int]):
    i = wp.tid()
    value, status = m.solve(hessian[i],force[i])
    delta[i] = value
    valid[i] = status


m.configure_module(__name__)


def _gradient(energy, x, h=2e-6):
    g = np.zeros_like(x,dtype=np.float64)
    for i in range(len(x)):
        for a in range(3):
            plus,minus=x.astype(np.float64).copy(),x.astype(np.float64).copy()
            plus[i,a]+=h;minus[i,a]-=h
            g[i,a]=(energy(plus)-energy(minus))/(2*h)
    return g


def _membrane_energy(x,inverse,area,mu,lam):
    f=np.column_stack((x[1]-x[0],x[2]-x[0]))@inverse
    u,s,vt=np.linalg.svd(f,full_matrices=False)
    r=u@vt
    return area*(mu*np.sum((f-r)**2)+.5*lam*(np.prod(s)-1)**2)


def _hinge_angle(x):
    e=x[1]-x[0];a=x[2]-x[0];b=x[3]-x[0]
    na=np.cross(e,a);nb=np.cross(b,e)
    return np.arctan2(np.dot(np.cross(na,nb),e/np.linalg.norm(e)),np.dot(na,nb))


def _hinge_energy(x,coefficient,rest,stiffness):
    return stiffness*coefficient*(_hinge_angle(x)-rest)**2


def _solid_energy(x,inverse,measure,mu,lam):
    f=np.column_stack((x[1]-x[0],x[2]-x[0],x[3]-x[0]))@inverse
    u,s,vt=np.linalg.svd(f)
    flip=np.eye(3);flip[2,2]=np.linalg.det(u@vt)
    r=u@flip@vt
    return measure*(mu*np.sum((f-r)**2)+.5*lam*(np.linalg.det(f)-1)**2)


class Float32MaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices=[wp.get_device("cpu"),*wp.get_cuda_devices()]

    def packet_arrays(self,n,device):
        return wp.empty(n,m.Pack,device=device),wp.empty(n,m.Pack,device=device)

    def test_corotated_membrane_matches_own_reference_and_energy(self):
        x=np.array([[.1,-.2,.3],[1.2,.1,.25],[.2,1.1,.55]],dtype=np.float32)
        inverse=np.array([[1.1,-.2],[.05,.9]],dtype=np.float32)
        area,mu,lam=.48,17.3,23.1
        expected=-_gradient(lambda q:_membrane_energy(q,inverse,area,mu,lam),x)
        for device in self.devices:
            out,ref=self.packet_arrays(3,device)
            wp.launch(_triangle,3,[wp.array(x,wp.vec3,device=device),wp.mat22(inverse),area,mu,lam,out,ref],device=device)
            actual,reference=out.numpy(),ref.numpy()
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual,reference,rtol=3e-4,atol=3e-4)
            np.testing.assert_allclose(actual[:,:3],expected,rtol=3e-4,atol=3e-4)

    def test_bending_preserves_native_model_and_psd_self_hessians(self):
        x=np.array([[0,0,0],[1.1,.1,.1],[.2,.8,.25],[.3,-.9,-.2]],dtype=np.float32)
        c,rest,k=.62,-.17,2.3
        expected=-_gradient(lambda q:_hinge_energy(q,c,rest,k),x)
        for device in self.devices:
            out,ref=self.packet_arrays(4,device)
            wp.launch(_hinge,4,[wp.array(x,wp.vec3,device=device),c,rest,k,out,ref],device=device)
            actual,reference=out.numpy(),ref.numpy()
            np.testing.assert_allclose(actual,reference,rtol=5e-4,atol=5e-4)
            np.testing.assert_allclose(actual[:,:3],expected,rtol=5e-4,atol=5e-4)
            for p in actual:
                self.assertGreaterEqual(np.linalg.eigvalsh(p[3:].reshape(3,3)).min(),-1e-5)

    def test_signed_corotated_tets_including_inverted_state(self):
        inverse=np.array([[1,.1,-.1],[.05,.9,.02],[0,.1,1.1]],dtype=np.float32)
        shapes=np.vstack((-inverse.sum(axis=0),inverse)).astype(np.float32)
        for sign in (1,-1):
            x=np.array([[.1,.2,.1],[1.1,.1,.15],[.2,1.2,.3],[.1,.15,sign*.9]],dtype=np.float32)
            measure,mu,lam=.13,13.7,21.9
            expected=-_gradient(lambda q:_solid_energy(q,inverse,measure,mu,lam),x)
            for device in self.devices:
                out,ref=self.packet_arrays(4,device)
                wp.launch(_tet,4,[wp.array(x,wp.vec3,device=device),wp.mat33(inverse),wp.array(shapes,wp.vec3,device=device),measure,mu,lam,out,ref],device=device)
                actual,reference=out.numpy(),ref.numpy()
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual,reference,rtol=1e-3,atol=1e-3)
                np.testing.assert_allclose(actual[:,:3],expected,rtol=1e-3,atol=1e-3)

    def test_float32_local_solve_regular_rank_deficient_and_zero(self):
        matrices=np.array([[[4,1,.2],[1,3,.1],[.2,.1,2]],np.diag([1,0,2]),np.zeros((3,3))],dtype=np.float32)
        forces=np.array([[1,2,-1],[1,1,4],[3,-2,1]],dtype=np.float32)
        expected=np.array([np.linalg.solve(matrices[0],forces[0]),[1,0,2],[0,0,0]],dtype=np.float32)
        for device in self.devices:
            result=wp.empty(3,wp.vec3,device=device);valid=wp.empty(3,int,device=device)
            wp.launch(_solve,3,[wp.array(matrices,wp.mat33,device=device),wp.array(forces,wp.vec3,device=device),result,valid],device=device)
            np.testing.assert_array_equal(valid.numpy(),[1,1,1])
            np.testing.assert_allclose(result.numpy(),expected,rtol=3e-5,atol=3e-5)

    def _scalar_scene(self,device,*,elastic=True,inactive_middle=False):
        rest=np.array([[0,0,0],[1,0,0],[0,1,0],[3,0,0],[4,0,0],[3,1,0],
                       [6,0,0],[7,0,0],[6,1,0]],dtype=np.float32)
        builder=newton.ModelBuilder(gravity=(0,0,0))
        builder.add_cloth_mesh(pos=wp.vec3(0),rot=wp.quat_identity(),scale=1.0,vel=wp.vec3(0),
            vertices=[wp.vec3(v) for v in rest],indices=[0,1,2,3,4,5,6,7,8],density=.2,
            tri_ke=17.3 if elastic else 0.0,tri_ka=23.1 if elastic else 0.0,tri_kd=0,edge_ke=0,edge_kd=0)
        builder.color()
        model=builder.finalize(device=device)
        x=rest.copy();x[[1,4,7],1]+=.08;x[[2,5,8],2]+=.06
        model.particle_q.assign(x)
        if inactive_middle:
            flags=model.particle_flags.numpy();flags[3]=0;model.particle_flags.assign(flags)
        ids=wp.array([0,3,6],int,device=device)
        inertia=x+np.array([.01,-.02,.005],np.float32)
        targets=x+np.array([.03,.01,-.02],np.float32)
        pins=np.zeros(9,np.int32);pins[0]=1
        forces=np.zeros((9,3),np.float32);forces[[0,3,6]]=[.03,-.01,.02]
        extra_h=np.repeat((np.eye(3,dtype=np.float32)*.3)[None],9,axis=0)
        displacement=wp.zeros(9,wp.vec3,device=device)
        proposal=wp.full(9,wp.vec3(-123),wp.vec3,device=device)
        error=wp.zeros(1,int,device=device)
        args=[.01,ids,wp.array(rest,wp.vec3,device=device),model.particle_q,model.particle_mass,
            wp.array(inertia,wp.vec3,device=device),model.particle_flags,model.tri_indices,model.tri_poses,
            model.tri_materials,model.tri_areas,model.edge_indices,model.edge_rest_angle,model.edge_rest_length,
            model.edge_bending_properties,model.tet_indices,model.tet_poses,model.tet_materials,
            model.soft_mesh_adjacency_device,wp.array(forces,wp.vec3,device=device),
            wp.array(extra_h,wp.mat33,device=device),wp.array(targets,wp.vec3,device=device),wp.array(pins,int,device=device),
            7.0,wp.zeros(model.edge_count,float,device=device),wp.empty((0,4),wp.vec3,device=device),
            wp.empty(0,float,device=device),.6,displacement,proposal,error]
        return model,args,(x,inertia,targets,pins,forces,extra_h)

    def test_actual_newton_arrays_scalar_energy_and_finite_pin_update(self):
        model,args,values=self._scalar_scene("cpu",elastic=True)
        wp.launch(kernels.solve_elasticity_scalar,3,args,device="cpu")
        self.assertEqual(int(args[-1].numpy()[0]),0)
        result=args[-2].numpy()
        x,inertia,targets,pins,forces,extra_h=values
        mass=model.particle_mass.numpy();inverse=model.tri_poses.numpy();areas=model.tri_areas.numpy()
        for p in (0,3,6):
            ti=p//3
            packets,refs=self.packet_arrays(3,"cpu")
            wp.launch(_triangle,3,[wp.array(x[p:p+3],wp.vec3,device="cpu"),wp.mat22(inverse[ti]),float(areas[ti]),17.3,23.1,packets,refs],device="cpu")
            value=refs.numpy()[0]
            f=value[:3]+mass[p]/.01**2*(inertia[p]-x[p])+forces[p]
            h=value[3:].reshape(3,3)+np.eye(3)*mass[p]/.01**2+extra_h[p]
            if pins[p]:f-=7*(x[p]-targets[p]);h+=np.eye(3)*7
            expected=x[p]+.6*np.linalg.solve(h,f)
            np.testing.assert_allclose(result[p],expected,rtol=3e-6,atol=3e-6)
        np.testing.assert_array_equal(result[[1,2,4,5,7,8]],np.full((6,3),-123,np.float32))

    def test_native_scene_hinge_coefficient_enters_full_local_update_once(self):
        # Exercise the authoring adapter, Newton's edge/role adjacency, and the
        # complete local system. A direct energy.hinge test cannot catch a
        # coefficient accidentally folded into both Model.edge_ke and c_e.
        x=np.array([[0,0,0],[1.25,.125,0],[.125,.875,.25],[.375,-.875,.5]],np.float32)
        masses=np.array([.25,.375,.5,.625],np.float32)
        coefficient,stiffness,rest,dt,damping=2.75,1.75,-.125,.125,.75
        payload={"schema":SCHEMA,"kind":"scene","example":1,
            "params":{"mu":0.0,"lambda":0.0,"solid_mu":0.0,"solid_lambda":0.0,
                      "kB":stiffness,"kpin":0.0,"gravity":[0,0,0],"rigid_pin_time":0.0},
            "positions":x.tolist(),"velocities":[[0,0,0]]*4,"mass":masses.tolist(),
            "node_to_rb":[-1]*4,"tris":[[0,1,2],[1,0,3]],"tets":[],
            "triangle_dm_inverse":[[1,0,0,1]]*2,"triangle_areas":[.5,.5],
            "hinges":[{"nodes":[0,1,2,3],"coefficient":coefficient,"rest_angle":rest}],
            "pins":[],"pin_targets":[[0,0,0]]*4,"bodies":[],"tet_nodes":[],
            "surface_nodes":[0,1,2,3],"rigid_pins":[],"rigid_pin_targets":[],"colliders":[]}
        # Independent energy finite differences and the CPU model's PSD
        # Gauss-Newton angle block; no native/Warp derivative helper is used.
        gradient=_gradient(lambda q:_hinge_energy(q,coefficient,rest,stiffness),x)
        angle_gradient=_gradient(_hinge_angle,x)
        hessians=np.array([2*stiffness*coefficient*np.outer(g,g) for g in angle_gradient])
        expected=np.array([x[i]-damping*np.linalg.solve(hessians[i]+masses[i]/dt**2*np.eye(3),gradient[i])
                           for i in range(4)])
        doubled=np.array([x[i]-damping*np.linalg.solve(coefficient*hessians[i]+masses[i]/dt**2*np.eye(3),
                                                       coefficient*gradient[i]) for i in range(4)])
        self.assertGreater(np.max(np.abs(expected-doubled)),1e-3,
                           "The fixture must distinguish c_e from c_e squared")
        for device in self.devices:
            model=create_model(payload,device)
            implementations=[kernels.solve_elasticity_scalar]
            if device.is_cuda:
                implementations.append(kernels.make_solve_elasticity_tile(True,False,True))
            for kernel in implementations:
                with self.subTest(device=str(device),kernel=kernel.key):
                    displacement=wp.zeros(4,wp.vec3,device=device)
                    proposal=wp.full(4,wp.vec3(-123),wp.vec3,device=device)
                    error=wp.zeros(1,int,device=device)
                    for ids in model.particle_color_groups:
                        args=[dt,ids,wp.array(x,wp.vec3,device=device),model.particle_q,model.particle_mass,
                            wp.array(x,wp.vec3,device=device),model.particle_flags,model.tri_indices,model.tri_poses,
                            model.tri_materials,model.tri_areas,model.edge_indices,model.edge_rest_angle,
                            model.edge_rest_length,model.edge_bending_properties,model.tet_indices,model.tet_poses,
                            model.tet_materials,model.soft_mesh_adjacency_device,wp.zeros(4,wp.vec3,device=device),
                            wp.zeros(4,wp.mat33,device=device),model.ipc.pin_targets,model.ipc.pinned,0.0,
                            model.ipc.edge_coefficient,model.ipc.tet_grad_N,model.ipc.tet_measure,damping,
                            displacement,proposal,error]
                        if kernel is kernels.solve_elasticity_scalar:
                            wp.launch(kernel,ids.size,args,device=device)
                        else:
                            wp.launch(kernel,((ids.size+1)//2)*32,args,device=device,block_dim=32)
                    self.assertEqual(int(error.numpy()[0]),0)
                    np.testing.assert_allclose(proposal.numpy(),expected,rtol=3e-6,atol=3e-6)

    def test_cooperative_cuda_kernel_matches_scalar_for_odd_color_and_inactive_slot(self):
        devices=wp.get_cuda_devices()
        if not devices:self.skipTest("No CUDA device is available")
        for device in devices:
            _,reference_args,_=self._scalar_scene(device,inactive_middle=True)
            wp.launch(kernels.solve_elasticity_scalar,3,reference_args,device=device)
            expected=reference_args[-2].numpy()
            _,args,_=self._scalar_scene(device,inactive_middle=True)
            tiled=kernels.make_solve_elasticity_tile(True,False,True)
            wp.launch(tiled,64,args,device=device,block_dim=32)
            self.assertEqual(int(args[-1].numpy()[0]),0)
            np.testing.assert_allclose(args[-2].numpy(),expected,rtol=3e-6,atol=3e-6)

    def _tet_scene(self, device, include_triangles, include_tets=True):
        tet=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
        rest=np.vstack((tet,tet+[3,0,0],[[6,0,0],[7,0,0],[6,1,0]])).astype(np.float32)
        builder=newton.ModelBuilder(gravity=(0,0,0))
        for point in rest:
            builder.add_particle(wp.vec3(point),wp.vec3(0),mass=1.0)
        measures=[]
        if include_tets:
            for first in (0,4):
                measures.append(builder.add_tetrahedron(first,first+1,first+2,first+3,
                    k_mu=13.7,k_lambda=21.9,k_damp=0.0))
        if include_triangles:
            builder.add_triangle(8,9,10,tri_ke=17.3,tri_ka=23.1,tri_kd=0.0)
        builder.color()
        model=builder.finalize(device=device)
        x=rest.copy();x[[1,5,9],1]+=.08;x[[2,6,10],2]+=.06
        model.particle_q.assign(x)
        flags=model.particle_flags.numpy();flags[4]=0;model.particle_flags.assign(flags)
        gradients=np.empty((0,4,3),np.float32)
        if include_tets:
            inverse=model.tet_poses.numpy()
            gradients=np.stack([np.vstack((-value.sum(axis=0),value)) for value in inverse]).astype(np.float32)
        args=[.01,wp.array([0,4,8],int,device=device),wp.array(rest,wp.vec3,device=device),
            model.particle_q,model.particle_mass,wp.array(x+[.01,-.02,.005],wp.vec3,device=device),
            model.particle_flags,model.tri_indices,model.tri_poses,model.tri_materials,model.tri_areas,
            model.edge_indices,model.edge_rest_angle,model.edge_rest_length,model.edge_bending_properties,
            model.tet_indices,model.tet_poses,model.tet_materials,model.soft_mesh_adjacency_device,
            wp.zeros(11,wp.vec3,device=device),wp.zeros(11,wp.mat33,device=device),
            wp.array(x,wp.vec3,device=device),wp.zeros(11,int,device=device),7.0,
            wp.zeros(model.edge_count,float,device=device),wp.array(gradients,wp.vec3,device=device),
            wp.array(measures,float,device=device),.6,wp.zeros(11,wp.vec3,device=device),
            wp.full(11,wp.vec3(-123),wp.vec3,device=device),wp.zeros(1,int,device=device)]
        return model,args

    def test_cooperative_cuda_tets_mixed_and_point_layouts_match_scalar(self):
        devices=wp.get_cuda_devices()
        if not devices:self.skipTest("No CUDA device is available")
        for device in devices:
            for include_triangles,include_tets in ((False,True),(True,True),(False,False)):
                with self.subTest(device=str(device),triangles=include_triangles,tets=include_tets):
                    _,reference_args=self._tet_scene(device,include_triangles,include_tets)
                    wp.launch(kernels.solve_elasticity_scalar,3,reference_args,device=device)
                    expected=reference_args[-2].numpy()
                    self.assertEqual(int(reference_args[-1].numpy()[0]),0)
                    _,args=self._tet_scene(device,include_triangles,include_tets)
                    tiled=kernels.make_solve_elasticity_tile(include_triangles,include_tets,not include_tets)
                    dimension,block_dim=(48,16) if include_tets else (64,32)
                    wp.launch(tiled,dimension,args,device=device,block_dim=block_dim)
                    self.assertEqual(int(args[-1].numpy()[0]),0)
                    np.testing.assert_allclose(args[-2].numpy(),expected,rtol=3e-6,atol=3e-6)


if __name__=="__main__":
    unittest.main()
