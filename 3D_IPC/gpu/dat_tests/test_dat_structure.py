"""Independent clean-runtime proof of the strict DAT energy substitutions.

Run in a fresh process. These fixtures never import project/GS test modules.
An isolated source overlay is supported without editing the project package.
"""
import ast
import builtins
import copy
import hashlib
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
import gpu

gpu.select_newton_runtime('reference')

import newton
from newton.solvers import SolverVBD as Native
from newton._src.solvers.vbd import particle_vbd_kernels as stock

LIFECYCLE = ('step','_initialize_particles','_pre_initialization_detection','_mid_step_detection',
    '_self_contact_is_due','_resolve_self_contact_schedule','_collision_detection_penetration_free',
    '_penetration_free_truncation','_refresh_collision_sets','_reset_dat_references',
    '_finalize_particles','rebuild_bvh')
REFERENCE_METHODS = {name:getattr(Native,name) for name in LIFECYCLE}
KERNEL_TYPES = (type(stock.solve_elasticity),type(stock.evaluate_neo_hookean_membrane_force_hessian))
REFERENCE_GLOBALS = {name:value for name,value in vars(stock).items() if isinstance(value,KERNEL_TYPES)}
REFERENCE_ITERATION = Native._solve_particle_iteration

from gpu import solver_dat as candidate
from gpu.solver_dat import SolverDAT
from gpu import dat_material_kernels as material_kernels
from gpu import dat_materials,energy


def dump(node):
    return ast.dump(node,include_attributes=False)


def expression(source):
    return ast.parse(source,mode='eval').body


def native_iteration_ast():
    tree=ast.parse(candidate._ORIGINAL.read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='SolverVBD')
    return copy.deepcopy(next(node for node in cls.body if isinstance(node,ast.FunctionDef)
                              and node.name=='_solve_particle_iteration'))


def compiled_candidate_ast():
    """Inspect the AST actually compiled, not inspect.getsource's old file.

    The generated method intentionally retains the reference source filename,
    so ordinary inspect.getsource would misleadingly return the original body.
    Capture its real compile input and verify it matches the installed code.
    """
    captured=[];original_compile=builtins.compile
    def capture(source,*args,**kwargs):
        if isinstance(source,ast.Module):
            methods=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_solve_particle_iteration']
            if methods:captured.append(copy.deepcopy(methods[0]))
        return original_compile(source,*args,**kwargs)
    with patch.object(builtins,'compile',side_effect=capture):
        function,counts=candidate._energy_iteration()
    if len(captured)!=1:raise AssertionError('Expected exactly one compiled particle iteration')
    for name in ('co_code','co_consts','co_names','co_varnames'):
        if getattr(function.__code__,name)!=getattr(SolverDAT._solve_particle_iteration.__code__,name):
            raise AssertionError('Inspected generated function differs from installed SolverDAT method')
    return captured[0],counts


def reverse_energy_edits(tree):
    counts={'contact':0,'tile':0,'scalar':0,'begin':0,'replay':0,'sdf':0}
    expected_replay=expression('self._energy_contacts.resolve(state_in.particle_q,self.particle_q_prev,self.model.particle_colors,color,self.particle_forces,self.particle_hessians)')
    expected_sdf=expression('self._accumulate_own_sdf(state_in,dt,color)')
    expected_begin=expression('self._energy_contacts.begin(dt,self._energy_dhat,self._energy_friction,self._energy_eps_v)')
    expected_material_args=expression('self._energy_materials.arguments()')
    class Reverse(ast.NodeTransformer):
        def visit_Expr(self,node):
            call=node.value
            if isinstance(call,ast.Call):
                name=ast.unparse(call.func)
                if name=='self._energy_contacts.resolve':
                    if dump(call)!=dump(expected_replay):raise AssertionError('Changed exact-energy replay')
                    counts['replay']+=1;return None
                if name=='self._accumulate_own_sdf':
                    if dump(call)!=dump(expected_sdf):raise AssertionError('Changed SDF energy call')
                    counts['sdf']+=1;return None
                if isinstance(call.func,ast.Attribute) and call.func.attr=='launch':
                    keywords={k.arg:k for k in call.keywords}
                    kernel=keywords.get('kernel')
                    if kernel is not None:
                        target=ast.unparse(kernel.value)
                        inputs=keywords['inputs'].value
                        if target=='self._energy_contacts.kernel':
                            if len(inputs.elts)!=16 or dump(inputs.elts[-1])!=dump(expected_begin):
                                raise AssertionError('Changed contact energy context/reset')
                            for index,value in ((8,'self._energy_dhat'),(11,'self._energy_friction'),(12,'self._energy_eps_v')):
                                if dump(inputs.elts[index])!=dump(expression(value)):
                                    raise AssertionError('Unexpected contact energy scalar substitution')
                            inputs.elts.pop();counts['begin']+=1
                            for index,value in ((8,'self.particle_self_contact_margin'),(11,'self.model.soft_contact_mu'),(12,'self.friction_epsilon')):
                                inputs.elts[index]=expression(value)
                            kernel.value=expression('accumulate_self_contact_force_and_hessian');counts['contact']+=1
                        elif target in ('self._tiled_elasticity_kernel','_own_solve_elasticity'):
                            if not isinstance(inputs.elts[-1],ast.Starred) or dump(inputs.elts[-1].value)!=dump(expected_material_args):
                                raise AssertionError('Changed appended material inputs')
                            inputs.elts.pop()
                            if target=='_own_solve_elasticity':
                                kernel.value=expression('solve_elasticity');counts['scalar']+=1
                            else:counts['tile']+=1
            return self.generic_visit(node)
    restored=Reverse().visit(copy.deepcopy(tree))
    if counts!={'contact':1,'tile':1,'scalar':1,'begin':1,'replay':1,'sdf':2}:
        raise AssertionError('Changed energy-only site counts: '+repr(counts))
    if dump(restored)!=dump(native_iteration_ast()):
        raise AssertionError('Non-energy particle iteration AST changed')
    return counts


def fingerprint(model):
    """All existing public model arrays, plus every grouped array identity."""
    result={}
    for name,value in vars(model).items():
        if isinstance(value,wp.array):
            host=value.numpy()
            result[name]=(value.ptr,value.shape,str(value.dtype),host.tobytes())
        elif isinstance(value,(list,tuple)) and value and all(isinstance(x,wp.array) for x in value):
            result[name]=tuple((x.ptr,x.shape,str(x.dtype),x.numpy().tobytes()) for x in value)
    return result


def model_fixture(*,mu=0.,lam=0.,origin=0.,velocity=0.,pin=0.):
    builder=newton.ModelBuilder(gravity=(0.,0.,0.))
    points=[[origin,0,0],[origin+1,0,0],[origin,1,0]]
    builder.add_particles(points,[[velocity,0,0],[0,0,0],[0,0,0]],[1.,1.,1.],
                          radius=[.01]*3,flags=[1,0,0])
    builder.add_triangles([0],[1],[2],tri_ke=[mu],tri_ka=[lam],tri_kd=[.1])
    builder.color();model=builder.finalize(device='cpu')
    targets=np.asarray(points,np.float32);targets[0]+=[.2,-.1,.05]
    model.ipc=SimpleNamespace(params={'kpin':pin,'d_hat':.002,'k_barrier':100.,
        'friction_coefficient':.2,'friction_velocity_epsilon':.01,'exact_computation_fallback':False,
        'use_ccd_guess':False,'use_colored_ccd_guess':False,'use_colored_ccd_guess_level_based':False,
        'dat_inertial_predictor':True},
        pinned=wp.array([int(pin>0),0,0],dtype=int,device='cpu'),
        pin_targets=wp.array(targets,dtype=wp.vec3,device='cpu'),
        dat_triangle_materials=wp.array([[mu,lam]],dtype=float,device='cpu'))
    return model


@wp.kernel
def own_terms(q:wp.array[wp.vec3],triangles:wp.array2d[int],inverse:wp.mat22,area:float,
              mu:float,lam:float,targets:wp.array[wp.vec3],kpin:float,
              triangle:wp.array[energy.Pack],pin_f:wp.array[wp.vec3],pin_h:wp.array[wp.mat33]):
    triangle[0]=energy.triangle(q[triangles[0,0]],q[triangles[0,1]],q[triangles[0,2]],inverse,area,mu,lam,0)
    f,h=dat_materials.evaluate_own_pin_force_hessian(0,q,targets,kpin)
    pin_f[0]=f;pin_h[0]=h


energy.configure_module(__name__)


class StrictDatStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device=wp.get_device('cpu')
        if 'newton_reference' not in Path(newton.__file__).resolve().parts:
            raise AssertionError('This acceptance requires the pristine reference runtime')

    def test_reverse_exact_compiled_ast_preserves_all_nonenergy_code(self):
        tree,counts=compiled_candidate_ast()
        self.assertEqual(counts,{'contact':1,'tile':1,'scalar':1})
        self.assertEqual(reverse_energy_edits(tree),{'contact':1,'tile':1,'scalar':1,'begin':1,'replay':1,'sdf':2})
        self.assertIsNot(SolverDAT._solve_particle_iteration,Native._solve_particle_iteration)
        self.assertIs(Native._solve_particle_iteration,REFERENCE_ITERATION)
        # Negative control: a changed launch policy must fail even though every
        # allowed energy substitution is still present.
        damaged=copy.deepcopy(tree)
        key=next(n for n in ast.walk(damaged) if isinstance(n,ast.keyword) and n.arg=='max_blocks')
        key.value=ast.Constant(value=1)
        with self.assertRaisesRegex(AssertionError,'Non-energy'):reverse_energy_edits(damaged)
        self.assertTrue(all(x['unchanged_non_energy_ast'] for x in material_kernels.assert_stock_layout().values()))

    def test_lifecycle_and_reference_kernels_unchanged_after_construction(self):
        model=model_fixture(mu=3,lam=5,pin=80)
        before=fingerprint(model)
        reference=Native(model,iterations=1,particle_enable_self_contact=True,
            particle_self_contact_margin=.002,particle_self_contact_gap=.0015,
            particle_vertex_contact_buffer_size=2,particle_edge_contact_buffer_size=2,enable_fast_math=False)
        for name,value in REFERENCE_GLOBALS.items():self.assertIs(getattr(stock,name),value)
        own=SolverDAT(model,iterations=1,allow_cpu=True,particle_enable_self_contact=True,
            particle_self_contact_margin=.002,particle_self_contact_gap=.0015,
            particle_vertex_contact_buffer_size=2,particle_edge_contact_buffer_size=2,enable_fast_math=False)
        self.assertEqual(before,fingerprint(model))
        self.assertIs(own.model,model);self.assertIs(reference.model,model)
        for name,method in REFERENCE_METHODS.items():
            self.assertIs(getattr(Native,name),method,name)
            self.assertIs(getattr(SolverDAT,name),method,name)
        for name,value in REFERENCE_GLOBALS.items():self.assertIs(getattr(stock,name),value,name)
        self.assertIsNone(own.ccd_backend);self.assertFalse(own._guess_requested)
        self.assertEqual(own.collision_frequency_type,reference.collision_frequency_type)
        self.assertEqual(own.collision_frequency,reference.collision_frequency)
        self.assertEqual(own._self_contact_query_radius,reference._self_contact_query_radius)
        self.assertEqual(own.dat_conservative_bound_relaxation,reference.dat_conservative_bound_relaxation)

    def test_native_predictor_keeps_unrounded_accumulated_displacement(self):
        # At x=4096, F32 ULP exceeds this velocity step. Native DAT retains a
        # nonzero raw displacement even though q+displacement rounds to q.
        states=[];solvers=[];original=None
        for cls in (Native,SolverDAT):
            model=model_fixture(origin=4096.,velocity=.001)
            kwargs={'allow_cpu':True} if cls is SolverDAT else {}
            solver=cls(model,iterations=1,particle_enable_self_contact=False,enable_fast_math=False,**kwargs)
            state,out=model.state(),model.state();q0=state.particle_q.numpy().copy()
            solver._initialize_particles(state,out,None,1/600)
            arrays={name:getattr(solver,name).numpy() for name in
                    ('particle_q_prev','inertia','particle_displacements','pos_prev_collision_detection')}
            arrays['q']=state.particle_q.numpy()
            self.assertNotEqual(float(arrays['particle_displacements'][0,0]),0.)
            self.assertEqual(float(arrays['q'][0,0]-arrays['pos_prev_collision_detection'][0,0]),0.)
            np.testing.assert_array_equal(arrays['q'],q0)
            if original is None:original=arrays
            else:
                for name in arrays:self.assertEqual(arrays[name].tobytes(),original[name].tobytes(),name)
            states.append((state,out));solvers.append(solver)
        for solver,(state,out) in zip(solvers,states):solver._solve_particle_iteration(state,out,None,1/600)
        for name in ('particle_displacements','pos_prev_collision_detection'):
            self.assertEqual(getattr(solvers[0],name).numpy().tobytes(),getattr(solvers[1],name).numpy().tobytes())
        self.assertEqual(states[0][0].particle_q.numpy().tobytes(),states[1][0].particle_q.numpy().tobytes())

    def terms(self,model,state):
        tri=wp.empty(1,energy.Pack,device='cpu');pf=wp.empty(1,wp.vec3,device='cpu');ph=wp.empty(1,wp.mat33,device='cpu')
        materials=model.ipc.dat_triangle_materials.numpy()[0]
        wp.launch(own_terms,1,[state.particle_q,model.tri_indices,wp.mat22(model.tri_poses.numpy()[0]),
            float(model.tri_areas.numpy()[0]),float(materials[0]),float(materials[1]),
            model.ipc.pin_targets,float(model.ipc.params['kpin']),tri,pf,ph],device='cpu')
        return tri.numpy()[0],pf.numpy()[0],ph.numpy()[0]

    def test_finite_pin_closed_form_uses_original_inverse_and_DAT_update(self):
        model=model_fixture(pin=80);before=fingerprint(model)
        solver=SolverDAT(model,iterations=1,allow_cpu=True,particle_enable_self_contact=False,enable_fast_math=False)
        state,out=model.state(),model.state();initial=state.particle_q.numpy().copy()
        triangle,pin_f,pin_h=self.terms(model,state)
        expected_force=80*(model.ipc.pin_targets.numpy()[0]-initial[0])
        np.testing.assert_array_equal(pin_f,expected_force)
        np.testing.assert_array_equal(pin_h,80*np.eye(3,dtype=np.float32))
        np.testing.assert_array_equal(triangle,np.zeros(12,np.float32))
        dt=.05;solver.step(state,out,model.control(),None,dt);solver.check_status()
        expected=initial.copy();expected[0]+=(80/(1/dt**2+80))*(model.ipc.pin_targets.numpy()[0]-initial[0])
        np.testing.assert_allclose(out.particle_q.numpy(),expected,rtol=2e-6,atol=2e-8)
        self.assertEqual(before,fingerprint(model))

    def test_triangle_and_pin_full_step_matches_own_force_hessian_local_system(self):
        model=model_fixture(mu=9.,lam=13.,pin=40.)
        solver=SolverDAT(model,iterations=1,allow_cpu=True,particle_enable_self_contact=False,enable_fast_math=False)
        state,out=model.state(),model.state()
        q=state.particle_q.numpy();q[0]=[.13,-.07,.05];state.particle_q.assign(q)
        tri,pf,ph=self.terms(model,state);dt=.02
        force=tri[:3]+pf;hessian=tri[3:].reshape(3,3)+ph+np.eye(3,dtype=np.float32)/(np.float32(dt)*np.float32(dt))
        self.assertGreater(float(np.linalg.norm(tri[:3])),0.)
        expected=q.copy();expected[0]+=np.linalg.solve(hessian.astype(np.float64),force.astype(np.float64)).astype(np.float32)
        solver.step(state,out,model.control(),None,dt);solver.check_status()
        np.testing.assert_allclose(out.particle_q.numpy(),expected,rtol=3e-6,atol=3e-8)
        np.testing.assert_array_equal(out.particle_q.numpy()[1:],q[1:])


if __name__=='__main__':unittest.main()
