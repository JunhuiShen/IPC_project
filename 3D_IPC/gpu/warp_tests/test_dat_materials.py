"""Strict stock-layout energy adapter acceptance; no controller substitutions."""
from types import SimpleNamespace
import inspect
import unittest
from unittest import mock
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverVBD
from newton._src.solvers.vbd import particle_vbd_kernels as stock
from gpu import energy
from gpu import dat_materials as material
from gpu import dat_material_kernels as kernels
from gpu.warp_tests.test_energies import _gradient,_membrane_energy,_hinge_energy,_hinge_angle,_solid_energy


@wp.kernel
def _tri_probe(x:wp.array[wp.vec3],previous:wp.array[wp.vec3],triangles:wp.array2d[int],inverse:wp.mat22,
    area:float,mu:float,lam:float,own_materials:wp.array2d[float],forces:wp.array[wp.vec3],hessians:wp.array[wp.mat33]):
    role=wp.tid()
    f,h=material.evaluate_own_membrane_force_hessian(0,role,x,previous,triangles,inverse,area,mu,lam,123.0,.1,own_materials)
    forces[role]=f;hessians[role]=h


@wp.kernel
def _small_strain_probe(x:wp.array[wp.vec3],triangles:wp.array2d[int],own_materials:wp.array2d[float],
    native_force:wp.array[wp.vec3],own_force:wp.array[wp.vec3]):
    i=wp.tid()
    f,h=stock.evaluate_neo_hookean_membrane_force_hessian(i,1,x,x,triangles,wp.identity(n=2,dtype=float),
        1.0,1000.0,1000.0,0.0,.1)
    own_f,own_h=material.evaluate_own_membrane_force_hessian(i,1,x,x,triangles,wp.identity(n=2,dtype=float),
        1.0,1000.0,1000.0,0.0,.1,own_materials)
    native_force[i]=f;own_force[i]=own_f


@wp.kernel
def _bend_probe(x:wp.array[wp.vec3],previous:wp.array[wp.vec3],edges:wp.array2d[int],rest:wp.array[float],
    length:wp.array[float],coefficient:wp.array[float],stiffness:wp.array[float],forces:wp.array[wp.vec3],hessians:wp.array[wp.mat33]):
    role=wp.tid()
    f,h=material.evaluate_own_bending_force_hessian(0,role,x,previous,edges,rest,length,99.0,12.0,.1,coefficient,stiffness)
    forces[role]=f;hessians[role]=h


@wp.kernel
def _matched_bend_probe(x:wp.array[wp.vec3],edges:wp.array2d[int],rest:wp.array[float],length:wp.array[float],
    coefficient:wp.array[float],stiffness:wp.array[float],native_f:wp.array[wp.vec3],native_h:wp.array[wp.mat33],
    own_f:wp.array[wp.vec3],own_h:wp.array[wp.mat33]):
    role=wp.tid()
    f,h=stock.evaluate_dihedral_angle_based_bending_force_hessian(0,role,x,x,edges,rest,length,.001,0.0,.1)
    pf,ph=material.evaluate_own_bending_force_hessian(0,role,x,x,edges,rest,length,.001,0.0,.1,coefficient,stiffness)
    native_f[role]=f;native_h[role]=h;own_f[role]=pf;own_h[role]=ph


@wp.kernel
def _tet_probe(x:wp.array[wp.vec3],previous:wp.array[wp.vec3],tets:wp.array2d[int],inverse:wp.mat33,
    grad:wp.array2d[wp.vec3],measure:wp.array[float],mu:float,lam:float,
    forces:wp.array[wp.vec3],hessians:wp.array[wp.mat33]):
    role=wp.tid()
    f,h=material.evaluate_own_solid_force_hessian(0,role,previous,x,tets,inverse,mu,lam,19.0,.1,grad,measure)
    forces[role]=f;hessians[role]=h


def fingerprint(model):
    names=['particle_q','particle_qd','particle_mass','particle_inv_mass','particle_flags','particle_colors',
        'tri_indices','tri_poses','tri_materials','tri_areas','edge_indices','edge_rest_angle',
        'edge_rest_length','edge_bending_properties','tet_indices','tet_poses','tet_materials']
    result={name:getattr(model,name).numpy().tobytes() if getattr(model,name) is not None else None for name in names}
    result['groups']=tuple(group.numpy().tobytes() for group in model.particle_color_groups)
    return result


def model_fixture():
    builder=newton.ModelBuilder(gravity=(0,0,0))
    builder.add_particles([[0,0,0],[1,0,0],[0,1,0],[0,-1,0]],[[0,0,0]]*4,[1]*4,
        radius=[.1]*4,flags=[0,1,1,1])
    builder.add_triangles([0,1],[1,0],[2,3],tri_ke=[3,3],tri_ka=[5,5],tri_kd=[.1,.1])
    builder.add_edges([2],[3],[0],[1],rest=[.2],edge_ke=[.001],edge_kd=[.01])
    builder.color()
    return builder.finalize(device='cpu')


class StockLayoutMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):wp.init();cls.device=wp.get_device('cpu')

    def ar(self,x,dtype):return wp.array(x,dtype=dtype,device=self.device)

    def test_static_guard_preserves_original_tile_scalar_inverse_and_updates(self):
        receipt=kernels.assert_stock_layout()
        self.assertTrue(all(x['unchanged_non_energy_ast'] for x in receipt.values()))
        original=inspect.getsource
        def damaged(function):
            source=original(function)
            if function is kernels.make_solve_elasticity_tile:
                source=source.replace('h_inv = wp.inverse(h_total)','h_inv = h_total')
            return source
        with mock.patch.object(kernels.inspect,'getsource',side_effect=damaged):
            with self.assertRaisesRegex(AssertionError,'non-energy body/layout'):kernels.assert_stock_layout()

    def test_preparation_keeps_stock_model_arrays_and_hard_flags_unchanged(self):
        model=model_fixture();before=fingerprint(model)
        data=material.prepare(model)
        self.assertEqual(fingerprint(model),before)
        np.testing.assert_array_equal(data.pin_mask.numpy(),np.zeros(4))
        self.assertEqual(data.pin_stiffness,0)
        self.assertAlmostEqual(float(data.edge_coefficients.numpy()[0]),1.0)
        np.testing.assert_array_equal(data.bending_stiffness.numpy(),model.edge_bending_properties.numpy()[:,0])
        # Effective stock bending stiffness is mapped separately by the bridge.
        mapped=self.ar([.0005],wp.float32);coefficient=self.ar([1],wp.float32)
        mapped_tri=self.ar(model.tri_materials.numpy()[:,:2].copy(),wp.float32)
        model.ipc=SimpleNamespace(dat_bending_stiffness=mapped,edge_coefficient=coefficient,dat_triangle_materials=mapped_tri)
        again=material.prepare(model)
        self.assertIs(again.bending_stiffness,mapped);self.assertIs(again.edge_coefficients,coefficient)
        self.assertIs(again.triangle_materials,mapped_tri)
        self.assertEqual(fingerprint(model),before)

    def test_triangle_force_and_rest_hessian_are_our_energy(self):
        inverse=np.array([[1.1,-.2],[.05,.9]],np.float32);x=np.array([[.1,-.2,.3],[1.2,.1,.25],[.2,1.1,.55]],np.float32)
        f=wp.empty(3,wp.vec3,device=self.device);h=wp.empty(3,wp.mat33,device=self.device)
        wp.launch(_tri_probe,3,[self.ar(x,wp.vec3),self.ar(x*1.2+.7,wp.vec3),self.ar([[0,1,2]],int),wp.mat22(inverse),.48,99.,111.,self.ar([[17.3,23.1]],wp.float32),f,h],device=self.device)
        expected=-_gradient(lambda q:_membrane_energy(q,inverse,.48,17.3,23.1),x)
        np.testing.assert_allclose(f.numpy(),expected,rtol=4e-4,atol=4e-4)
        rest=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
        wp.launch(_tri_probe,3,[self.ar(rest,wp.vec3),self.ar(rest+.3,wp.vec3),self.ar([[0,1,2]],int),wp.mat22(np.eye(2)),.5,99.,111.,self.ar([[3.,5.]],wp.float32),f,h],device=self.device)
        gradients=np.array([[-1,-1],[1,0],[0,1]])
        for role,(a,b) in enumerate(gradients):
            expected_h=.5*np.array([[11*a*a+3*b*b,8*a*b,0],[8*a*b,3*a*a+11*b*b,0],[0,0,0]])
            np.testing.assert_allclose(h.numpy()[role],expected_h,rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(f.numpy(),0,atol=2e-6)

    def test_actual_stock_lame_conversion_requires_direct_slot_mapping(self):
        # The CLEAN reference converts lambda_NH=lambda_slot+mu internally.
        # Evaluate its actual callback, not the unconverted formula comment.
        x=np.array([[0,0,0],[1.01,0,0],[0,1,0],[0,0,0],[.99,0,0],[0,1,0]],np.float32)
        triangles=self.ar([[0,1,2],[3,4,5]],int)
        native_f=wp.empty(2,wp.vec3,device=self.device);own_f=wp.empty_like(native_f)
        wp.launch(_small_strain_probe,2,[self.ar(x,wp.vec3),triangles,self.ar([[1000,1000]]*2,wp.float32),native_f,own_f],device=self.device)
        width=float(x[1,0]-x[4,0]);native_tangent=-float(native_f.numpy()[0,0]-native_f.numpy()[1,0])/width
        own_tangent=-float(own_f.numpy()[0,0]-own_f.numpy()[1,0])/width
        self.assertAlmostEqual(native_tangent,3000,delta=.1);self.assertAlmostEqual(own_tangent,3000,delta=.1)
        np.testing.assert_allclose(native_f.numpy(),own_f.numpy(),rtol=2e-5,atol=2e-3)
        wp.launch(_small_strain_probe,2,[self.ar(x,wp.vec3),triangles,self.ar([[1000,0]]*2,wp.float32),native_f,own_f],device=self.device)
        wrong_tangent=-float(own_f.numpy()[0,0]-own_f.numpy()[1,0])/width
        self.assertAlmostEqual(wrong_tangent,2000,delta=.1)

    def test_bending_uses_separate_kB_and_coefficient_once_with_stock_role_mapping(self):
        x=np.array([[0,0,0],[1.1,.1,.1],[.2,.8,.25],[.3,-.9,-.2]],np.float32)
        c,k,rest=2.75,1.75,-.17
        f=wp.empty(4,wp.vec3,device=self.device);h=wp.empty(4,wp.mat33,device=self.device)
        # Stored Newton roles are[oppositeA,oppositeB,e0,e1]. The old model
        # material99 is deliberately different from the new own kB1.75.
        wp.launch(_bend_probe,4,[self.ar(x,wp.vec3),self.ar(x+.2,wp.vec3),self.ar([[2,3,0,1]],int),
            self.ar([rest],wp.float32),self.ar([1.1],wp.float32),self.ar([c],wp.float32),self.ar([k],wp.float32),f,h],device=self.device)
        gradient=_gradient(lambda q:_hinge_energy(q,c,rest,k),x)
        angle_gradient=_gradient(_hinge_angle,x)
        for role,node in enumerate([2,3,0,1]):
            np.testing.assert_allclose(f.numpy()[role],-gradient[node],rtol=5e-4,atol=5e-4)
            np.testing.assert_allclose(h.numpy()[role],2*k*c*np.outer(angle_gradient[node],angle_gradient[node]),rtol=7e-4,atol=7e-4)

    def test_matched_effective_bending_agrees_with_actual_stock_callback(self):
        x=np.array([[0,0,0],[1.1,.1,.1],[.2,.8,.25],[.3,-.9,-.2]],np.float32)
        c,length=2.75,1.1;k=.001*length/(2*c)
        nf=wp.empty(4,wp.vec3,device=self.device);nh=wp.empty(4,wp.mat33,device=self.device)
        pf=wp.empty_like(nf);ph=wp.empty_like(nh)
        wp.launch(_matched_bend_probe,4,[self.ar(x,wp.vec3),self.ar([[2,3,0,1]],int),self.ar([-.17],wp.float32),
            self.ar([length],wp.float32),self.ar([c],wp.float32),self.ar([k],wp.float32),nf,nh,pf,ph],device=self.device)
        np.testing.assert_allclose(pf.numpy(),nf.numpy(),rtol=5e-4,atol=1e-7)
        np.testing.assert_allclose(ph.numpy(),nh.numpy(),rtol=5e-4,atol=1e-7)

    def test_solid_force_and_native_psd_self_hessian(self):
        inverse=np.eye(3,dtype=np.float32);grad=np.array([[[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]]],np.float32)
        x=np.array([[0,0,0],[1.1,.05,.02],[.1,.9,.03],[.02,.04,1.2]],np.float32)
        f=wp.empty(4,wp.vec3,device=self.device);h=wp.empty(4,wp.mat33,device=self.device)
        args=[self.ar(x,wp.vec3),self.ar(x+.4,wp.vec3),self.ar([[0,1,2,3]],int),wp.mat33(inverse),
            wp.array(grad,dtype=wp.vec3,ndim=2,device=self.device),self.ar([1/6],wp.float32),3.,5.,f,h]
        wp.launch(_tet_probe,4,args,device=self.device)
        expected=-_gradient(lambda q:_solid_energy(q,inverse,1/6,3.,5.),x)
        np.testing.assert_allclose(f.numpy(),expected,rtol=5e-4,atol=5e-4)
        rest=np.vstack((np.zeros(3),np.eye(3))).astype(np.float32);args[0]=self.ar(rest,wp.vec3)
        wp.launch(_tet_probe,4,args,device=self.device)
        for role,g in enumerate(grad[0]):
            # Our declared solid self-Hessian is the frozen-rotation PSD
            # approximation 2μ||gradN||²I + λ(cof(F)gradN)(...)^T.
            expected_h=(6*np.dot(g,g)*np.eye(3)+5*np.outer(g,g))/6
            np.testing.assert_allclose(h.numpy()[role],expected_h,rtol=3e-5,atol=3e-5)

    def test_stock_scalar_inverse_accumulates_displacement_and_keeps_hard_pin(self):
        builder=newton.ModelBuilder(gravity=(0,0,0))
        builder.add_particles([[.1,-.2,.3],[1,2,3]],[[0,0,0]]*2,[2,1],flags=[1,0],radius=[0,0]);builder.color()
        model=builder.finalize(device='cpu');solver=SolverVBD(model,enable_fast_math=False)
        before=fingerprint(model);data=material.prepare(model,pin_mask=np.array([1,1],np.int32),
            pin_targets=np.array([[.3,.1,-.2],[10,20,30]],np.float32),pin_stiffness=7)
        q=model.particle_q;inertia=self.ar(q.numpy()+np.array([.4,-.2,.1],np.float32),wp.vec3)
        contact_f=self.ar([[.2,-.1,.4],[0,0,0]],wp.vec3);contact_h=wp.zeros(2,wp.mat33,device=self.device)
        old=np.array([[.03125,.0625,-.125],[7,8,9]],np.float32);displacements=self.ar(old,wp.vec3)
        arguments=[.25,self.ar([0,1],int),q,q,model.particle_mass,inertia,model.particle_flags,
            model.tri_indices,model.tri_poses,model.tri_materials,model.tri_areas,model.edge_indices,
            model.edge_rest_angle,model.edge_rest_length,model.edge_bending_properties,
            model.tet_indices,model.tet_poses,model.tet_materials,solver.particle_adjacency,contact_f,contact_h]
        wp.launch(kernels.solve_elasticity,2,[*arguments,*data.arguments(),displacements],device=self.device)
        point=q.numpy()[0];force=2*16*(inertia.numpy()[0]-point)-7*(point-data.pin_targets.numpy()[0])+contact_f.numpy()[0]
        expected=old[0]+np.linalg.inv(np.eye(3,dtype=np.float32)*39)@force
        np.testing.assert_allclose(displacements.numpy()[0],expected,rtol=3e-6,atol=3e-6)
        np.testing.assert_array_equal(displacements.numpy()[1],[0,0,0])
        self.assertEqual(fingerprint(model),before)


energy.configure_module(__name__)

if __name__=='__main__':unittest.main()
