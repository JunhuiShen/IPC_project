"""The no-planar-DAT option disables all collision processing, not materials."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverVBD
from newton._src.solvers.solver import SolverBase

from gpu import solver_dat as candidate
from gpu.solver_dat import SolverDAT
from gpu import sdf


def attach(model, *, pin=0., with_sdf=False, coefficients=None):
    q = model.particle_q.numpy()
    targets = q.copy()
    targets[0] += [2.75, -2.25, 1.]
    params = {'kpin': pin, 'd_hat': .002, 'k_barrier': 100., 'friction_coefficient': .2,
              'friction_velocity_epsilon': .01, 'use_ccd_guess': False,
              'use_colored_ccd_guess': False, 'use_colored_ccd_guess_level_based': False,
              'dat_inertial_predictor': True, 'dt': .1, 'k_sdf': 1.e6 if with_sdf else 0., 'eps_sdf': .5}
    records = [{'kind': 'plane', 'point': [0., 0., 0.], 'axis': [0., 1., 0.]}] if with_sdf else []
    model.ipc = SimpleNamespace(params=params,
        pinned=wp.array([int(pin > 0.)] + [0] * (len(q)-1), int, device=model.device),
        pin_targets=wp.array(targets, wp.vec3, device=model.device),
        dat_triangle_materials=wp.array(model.tri_materials.numpy()[:, :2], float, device=model.device),
        collider_data=wp.array(sdf.pack_colliders(records), float, device=model.device),
        deformable=wp.array([1] * len(q), int, device=model.device), solid=wp.zeros(len(q), int, device=model.device),
        surface=wp.array([1] * len(q), int, device=model.device))
    if coefficients is not None:
        model.ipc.edge_coefficient=wp.array(coefficients,float,device=model.device)
    return model


def fixture(device, *, velocity=(3., -2., 1.), pin=0., height=.2, face_z=-10., with_sdf=False):
    builder = newton.ModelBuilder(gravity=(0., 0., 0.))
    points = [[.25, .25, height], [0., 0., face_z], [1., 0., face_z], [0., 1., face_z]]
    builder.add_particles(points, [list(velocity), [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [2., 1., 1., 1.], radius=[0.] * 4, flags=[1, 0, 0, 0])
    builder.add_triangles([1], [2], [3], tri_ke=[0.], tri_ka=[0.], tri_kd=[0.])
    builder.color()
    return attach(builder.finalize(device=device),pin=pin,with_sdf=with_sdf)


def solver(model, *, enabled=True, iterations=1, **options):
    return SolverDAT(model, iterations=iterations, allow_cpu=True, enable_planar_dat=enabled,
        particle_enable_self_contact=True, particle_self_contact_margin=.002,
        particle_self_contact_gap=.0015, vertex_contact_capacity=8, edge_contact_capacity=8,
        enable_fast_math=False, **options)


def gradient(function,x,epsilon=2.e-6):
    result=np.zeros_like(x,dtype=np.float64)
    for vertex in range(len(x)):
        for axis in range(3):
            plus=x.astype(np.float64).copy();minus=x.astype(np.float64).copy()
            plus[vertex,axis]+=epsilon;minus[vertex,axis]-=epsilon
            result[vertex,axis]=(function(plus)-function(minus))/(2.*epsilon)
    return result


def angle(q):
    edge=q[1]-q[0]
    first=np.cross(edge,q[2]-q[0]);second=np.cross(q[3]-q[0],edge)
    return np.arctan2(np.dot(np.cross(first,second),edge/np.linalg.norm(edge)),np.dot(first,second))


def membrane_energy(q,inverse,area,mu,lam):
    deformation=np.column_stack((q[1]-q[0],q[2]-q[0]))@inverse
    u,singular,vt=np.linalg.svd(deformation,full_matrices=False)
    return area*(mu*np.sum((deformation-u@vt)**2)+.5*lam*(np.prod(singular)-1.)**2)


def membrane_psd(q,inverse,area,mu,lam,role):
    deformation=np.column_stack((q[1]-q[0],q[2]-q[0]))@inverse
    f0,f1=deformation[:,0],deformation[:,1]
    cross=np.cross(f0,f1);ratio=max(np.linalg.norm(cross),1.e-10);normal=cross/ratio
    a,b=np.vstack((-inverse.sum(axis=0),inverse))[role]
    derivative=a*np.cross(f1,normal)+b*np.cross(normal,f0)
    w=a*f1-b*f0;native_lam=lam+mu
    pressure=native_lam*(ratio-1.-mu/native_lam);r=max(0.,pressure)/ratio
    return area*((mu*(a*a+b*b)+r*np.dot(w,w))*np.eye(3)
                 +(native_lam-r)*np.outer(derivative,derivative)-r*np.outer(w,w))


class ForbiddenCollisionObject:
    def __getattr__(self,name):
        raise AssertionError('Collision object was accessed: '+name)


class PlanarDatToggleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device('cpu'), *wp.get_cuda_devices()]

    def test_default_preserves_native_class_lifecycle_and_enabled_workspaces(self):
        for device in self.devices:
            model = fixture(device, pin=80., with_sdf=True)
            default = solver(model)
            disabled = solver(model, enabled=False)
            for name in ('step','_initialize_particles','_collision_detection_penetration_free',
                         '_penetration_free_truncation','_refresh_collision_sets',
                         '_reset_dat_references','_finalize_particles'):
                self.assertIs(getattr(SolverDAT,name),getattr(SolverVBD,name))
            self.assertIs(default.step.__func__,SolverVBD.step)
            self.assertIs(default._penetration_free_truncation.__func__,SolverVBD._penetration_free_truncation)
            self.assertIsNotNone(default._energy_contacts)
            self.assertIsNotNone(default.sdf_backend)
            self.assertTrue(default.particle_enable_self_contact)
            self.assertFalse(disabled.particle_enable_self_contact)
            self.assertFalse(disabled.rigid_soft_enable_dat)
            self.assertIsNone(disabled._energy_contacts)
            self.assertIsNone(disabled.sdf_backend)
            self.assertIsNone(getattr(disabled,'trimesh_collision_detector',None))
            self.assertIsNone(getattr(disabled,'trimesh_collision_info',None))
            self.assertIsNone(disabled.collision_pipeline)
            self.assertFalse(disabled.configuration['enable_planar_dat'])
            self.assertFalse(disabled.configuration['collision_processing'])
            self.assertFalse(disabled.configuration['energy_contact']['enabled'])
            self.assertFalse(disabled.configuration['energy_sdf']['enabled'])
            self.assertIs(disabled._energy_materials.pin_targets,model.ipc.pin_targets)
            for slot in SolverBase.CollisionSlot:
                self.assertEqual(disabled.collision_frequency_type[slot],SolverBase.CollisionFrequencyType.NONE)

    def test_toggle_requires_explicit_boolean(self):
        model = fixture('cpu')
        for value in (None,0,1,'false',np.bool_(False)):
            with self.subTest(value=value),self.assertRaisesRegex(TypeError,'enable_planar_dat'):
                SolverDAT(model,allow_cpu=True,enable_planar_dat=value)

    def test_constructor_and_step_ignore_owned_external_and_reassigned_collision_inputs(self):
        native = __import__(SolverVBD.__module__,fromlist=['SolverVBD'])
        for device in self.devices:
            model=fixture(device,with_sdf=True)
            with patch.object(native,'TriMeshCollisionDetector',side_effect=AssertionError('Detector created')), \
                 patch.object(candidate.energies,'EnergyWorkspace',side_effect=AssertionError('Contact workspace created')), \
                 patch.object(sdf,'SdfBackend',side_effect=AssertionError('SDF workspace created')):
                own=solver(model,enabled=False,rigid_soft_enable_dat=True,
                    collision_pipeline=ForbiddenCollisionObject(),particle_collision_detection_interval=1,
                    collision_frequency_type={SolverBase.CollisionSlot.SOFT_SELF_CONTACT:
                                              SolverBase.CollisionFrequencyType.ITERATIONS})
            state,out=model.state(),model.state()
            # A supplied/reattached pipeline or contacts buffer must never
            # resurrect collision processing in this explicit ablation.
            own.collision_pipeline=ForbiddenCollisionObject()
            own._pipeline_contacts=ForbiddenCollisionObject()
            own.sdf_backend=ForbiddenCollisionObject()
            own.particle_enable_self_contact=True
            own.rigid_soft_enable_dat=True
            with patch.object(own,'_collision_detection_penetration_free',side_effect=AssertionError('Detection ran')), \
                 patch.object(wp,'launch',wraps=wp.launch) as launches:
                own.step(state,out,model.control(),ForbiddenCollisionObject(),.1)
            own.check_status()
            self.assertIsNone(own.collision_pipeline)
            self.assertIsNone(own._pipeline_contacts)
            self.assertIsNone(own.sdf_backend)
            launched=[call.kwargs.get('kernel',call.args[0] if call.args else None)
                      for call in launches.call_args_list]
            self.assertIn(candidate._publish_collision_free_motion,launched)
            self.assertNotIn(native.apply_truncation_ts,launched)
            self.assertNotIn(native.apply_planar_truncation_parallel_by_collision,launched)
            self.assertNotIn(native.accumulate_self_contact_force_and_hessian,launched)
            self.assertTrue(all('contact' not in getattr(kernel,'key','').lower() for kernel in launched))
            np.testing.assert_array_equal(own.truncation_ts.numpy(),0.)

    def test_free_and_finite_pin_closed_forms_exceed_the_previous_motion_cap(self):
        dt=.1
        for device in self.devices:
            for pin in (0.,80.):
                with self.subTest(device=str(device),pin=pin):
                    model=fixture(device,pin=pin)
                    own=solver(model,enabled=False)
                    state,out=model.state(),model.state()
                    initial=state.particle_q.numpy().copy()
                    inertia=initial[0]+dt*model.particle_qd.numpy()[0]
                    expected=(2.*inertia+dt**2*pin*model.ipc.pin_targets.numpy()[0])/(2.+dt**2*pin)
                    own.step(state,out,model.control(),None,dt)
                    own.check_status()
                    np.testing.assert_allclose(out.particle_q.numpy()[0],expected,rtol=3e-6,atol=8e-8)
                    np.testing.assert_allclose(out.particle_qd.numpy()[0],(expected-initial[0])/dt,
                                               rtol=3e-6,atol=8e-7)
                    old_cap=.5*own.dat_conservative_bound_relaxation*own._self_contact_query_radius
                    self.assertGreater(np.linalg.norm(expected-initial[0]),100.*old_cap)
                    np.testing.assert_array_equal(out.particle_q.numpy()[1:],initial[1:])

    def test_zero_iterations_publish_unrestricted_predictor_and_velocity(self):
        for device in self.devices:
            model=fixture(device)
            own=solver(model,enabled=False,iterations=0)
            state,out=model.state(),model.state()
            initial=state.particle_q.numpy().copy()
            expected=initial+.1*model.particle_qd.numpy()
            own.step(state,out,model.control(),ForbiddenCollisionObject(),.1)
            own.check_status()
            np.testing.assert_allclose(out.particle_q.numpy(),expected,rtol=3e-6,atol=8e-8)
            np.testing.assert_allclose(out.particle_qd.numpy(),model.particle_qd.numpy(),rtol=3e-6,atol=8e-7)
            self.assertIs(SolverDAT._initialize_particles,SolverVBD._initialize_particles)

    def test_mesh_and_strong_sdf_contact_energies_are_both_absent(self):
        for device in self.devices:
            model=fixture(device,velocity=(0.,0.,0.),height=.001,face_z=0.,with_sdf=True)
            own=solver(model,enabled=False)
            state,out=model.state(),model.state()
            initial=state.particle_q.numpy().copy()
            own.step(state,out,model.control(),None,.1)
            own.check_status()
            # The point is inside both mesh d_hat and the SDF epsilon band.
            # With all collision forces absent its zero velocity is unchanged.
            np.testing.assert_array_equal(out.particle_q.numpy(),initial)
            np.testing.assert_array_equal(out.particle_qd.numpy(),0.)

    def test_corotated_membrane_and_mapped_bending_remain_active_without_collisions(self):
        dt=.2;coefficient=2.75
        for device in self.devices:
            for membrane,bending in ((True,False),(False,True)):
                with self.subTest(device=str(device),membrane=membrane,bending=bending):
                    mu,lam=(9.,13.) if membrane else (0.,0.)
                    stiffness=.7 if bending else 0.
                    builder=newton.ModelBuilder(gravity=(0.,0.,0.))
                    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]],np.float32)
                    builder.add_particles(points.tolist(),[[0.,0.,0.]]*4,[1.]*4,radius=[0.]*4,flags=[0,0,0,1])
                    builder.add_triangles([0,1],[1,0],[2,3],tri_ke=[mu]*2,tri_ka=[lam]*2,tri_kd=[0.]*2)
                    builder.add_edges([2],[3],[0],[1],rest=[0.],edge_ke=[stiffness],edge_kd=[0.])
                    builder.color()
                    model=attach(builder.finalize(device=device),coefficients=[coefficient])
                    own=solver(model,enabled=False)
                    state,out=model.state(),model.state()
                    q=state.particle_q.numpy().copy();q[3,2]=.25;state.particle_q.assign(q)
                    triangles=model.tri_indices.numpy();inverses=model.tri_poses.numpy();areas=model.tri_areas.numpy()
                    def total(value):
                        result=sum(membrane_energy(value[t],inverse,area,mu,lam)
                                   for t,inverse,area in zip(triangles,inverses,areas)) if membrane else 0.
                        return result+stiffness*coefficient*angle(value)**2
                    force=-gradient(total,q)[3]
                    hessian=np.eye(3)/dt**2
                    if membrane:
                        for t,inverse,area in zip(triangles,inverses,areas):
                            if 3 in t:
                                hessian+=membrane_psd(q[t],inverse,area,mu,lam,int(np.flatnonzero(t==3)[0]))
                    if bending:
                        derivative=gradient(angle,q)[3]
                        hessian+=2.*stiffness*coefficient*np.outer(derivative,derivative)
                    expected=q[3]+np.linalg.solve(hessian,force)
                    own.step(state,out,model.control(),None,dt)
                    own.check_status()
                    np.testing.assert_allclose(out.particle_q.numpy()[3],expected,rtol=4e-5,atol=2e-6)
                    self.assertGreater(np.linalg.norm(out.particle_q.numpy()[3]-q[3]),1.e-4)
                    np.testing.assert_array_equal(out.particle_q.numpy()[:3],q[:3])

    def test_raw_sub_ulp_motion_is_preserved(self):
        for device in self.devices:
            model=fixture(device,velocity=(.001,0.,0.))
            q=model.particle_q.numpy();q[:,0]+=4096.;model.particle_q.assign(q)
            own=solver(model,enabled=False)
            state,out=model.state(),model.state()
            own._initialize_particles(state,out,None,1./600.)
            own.check_status()
            self.assertNotEqual(float(own.particle_displacements.numpy()[0,0]),0.)
            self.assertEqual(float(state.particle_q.numpy()[0,0]-q[0,0]),0.)

    def test_eager_cpu_cuda_and_cuda_graph_replay_agree(self):
        for device in self.devices:
            expected=None
            for captured in ((False,True) if device.is_cuda else (False,)):
                model=fixture(device,pin=80.)
                own=solver(model,enabled=False,iterations=2)
                a,b=model.state(),model.state();control=model.control();dt=.01
                def run_pair():
                    own.step(a,b,control,None,dt)
                    own.step(b,a,control,None,dt)
                if captured:
                    with wp.ScopedCapture(device=device) as graph:
                        run_pair()
                    for _ in range(3):wp.capture_launch(graph.graph)
                else:
                    for _ in range(3):run_pair()
                own.check_status()
                result=(a.particle_q.numpy(),a.particle_qd.numpy())
                if expected is None:expected=result
                else:
                    for actual,oracle in zip(result,expected):
                        np.testing.assert_allclose(actual,oracle,rtol=3e-6,atol=8e-7)


if __name__=='__main__':unittest.main()
