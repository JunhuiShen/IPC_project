"""Native DAT execution with only project energy evaluations replaced.

The upstream step, prediction, detection, clipping, local inverse and velocity
publication are inherited. The original particle-iteration AST is changed only
at energy kernel bindings and their extra energy inputs.
No global upstream function or kernel is modified. For the explicit
``enable_planar_dat=False`` ablation, all collision processing is disabled:
mesh/SDF contact energies, detection, external/owned contacts and DAT limits.
Native inertial prediction, material/pin energies and publication remain.
"""
from __future__ import annotations
import ast,copy,hashlib,importlib,inspect
from pathlib import Path
import numpy as np
import warp as wp
from . import select_newton_runtime

select_newton_runtime('reference')

from newton.solvers import SolverVBD
from newton._src.solvers.solver import SolverBase
from . import dat_energies as energies

_ORIGINAL = Path(__file__).resolve().parent / 'third_party/newton_reference/newton/_src/solvers/vbd/solver_vbd.py'
# An overlay test can point at the same pinned original; the production path
# above is always relative to this project, with no network fetch.
if not _ORIGINAL.is_file():
    import gpu
    _ORIGINAL=Path(gpu.__file__).resolve().parent/'third_party/newton_reference/newton/_src/solvers/vbd/solver_vbd.py'
_ORIGINAL_SHA='20ed9e41c1fdbd630dc27df6be22d471ca83692d0805d48ddbe4ba5ca78b39f9'


def _energy_iteration():
    source=_ORIGINAL.read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=_ORIGINAL_SHA:
        raise RuntimeError('Reference DAT source changed; review the energy-only substitutions')
    tree=ast.parse(source);base=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SolverVBD')
    original=next(x for x in base.body if isinstance(x,ast.FunctionDef) and x.name=='_solve_particle_iteration')
    function=copy.deepcopy(original);counts={'contact':0,'tile':0,'scalar':0}
    def expression(s):return ast.parse(s,mode='eval').body
    class EnergyCalls(ast.NodeTransformer):
        def visit_Expr(self,node):
            node=self.generic_visit(node)
            call=node.value
            if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or call.func.attr!='launch':return node
            keywords={k.arg:k for k in call.keywords}
            target=keywords.get('kernel')
            if target is None:return node
            name=ast.unparse(target.value)
            inputs=keywords['inputs'].value
            if name=='accumulate_self_contact_force_and_hessian':
                counts['contact']+=1
                target.value=expression('self._energy_contacts.kernel')
                inputs.elts[8]=expression('self._energy_dhat')
                inputs.elts[11]=expression('self._energy_friction')
                inputs.elts[12]=expression('self._energy_eps_v')
                inputs.elts.append(expression('self._energy_contacts.begin(dt,self._energy_dhat,self._energy_friction,self._energy_eps_v)'))
                return node
            if name in ('self._tiled_elasticity_kernel','solve_elasticity'):
                counts['tile' if name.startswith('self.') else 'scalar']+=1
                if name=='solve_elasticity':target.value=expression('_own_solve_elasticity')
                inputs.elts.append(ast.Starred(value=expression('self._energy_materials.arguments()'),ctx=ast.Load()))
                # Any project SDF is an additional energy term; stock cloth
                # has no SDFs and receives no additional work here.
                sdf=ast.parse('self._accumulate_own_sdf(state_in,dt,color)').body[0]
                return [sdf,node]
            return node
    function=EnergyCalls().visit(function)
    if counts!={'contact':1,'tile':1,'scalar':1}:raise RuntimeError('Unexpected upstream iteration layout')
    # Independent reversal: every non-energy node must remain exactly stock.
    class RestoreEnergyCalls(ast.NodeTransformer):
        def visit_Expr(self,node):
            call=node.value
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute):
                label=ast.unparse(call.func)
                if label=='self._accumulate_own_sdf':
                    return None
                if call.func.attr=='launch':
                    keywords={k.arg:k for k in call.keywords};target=keywords.get('kernel')
                    if target is not None:
                        name=ast.unparse(target.value);values=keywords['inputs'].value.elts
                        if name=='self._energy_contacts.kernel':
                            target.value=expression('accumulate_self_contact_force_and_hessian')
                            values[8]=expression('self.particle_self_contact_margin')
                            values[11]=expression('self.model.soft_contact_mu')
                            values[12]=expression('self.friction_epsilon')
                            values.pop()
                        elif name in ('self._tiled_elasticity_kernel','_own_solve_elasticity'):
                            if name=='_own_solve_elasticity':target.value=expression('solve_elasticity')
                            values.pop()
            return self.generic_visit(node)
    restored=RestoreEnergyCalls().visit(copy.deepcopy(function))
    if ast.dump(restored,include_attributes=False)!=ast.dump(original,include_attributes=False):
        raise RuntimeError('Non-energy DAT iteration change detected')
    function.decorator_list=[]
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    native=importlib.import_module(SolverVBD.__module__)
    globals_=dict(vars(native));globals_['_own_solve_elasticity']=energies.solve_elasticity
    globals_['__name__']=__name__
    exec(compile(module,str(_ORIGINAL),'exec'),globals_)
    return globals_['_solve_particle_iteration'],counts


@wp.kernel(module="unique", enable_backward=False)
def _publish_collision_free_motion(reference: wp.array[wp.vec3], displacement: wp.array[wp.vec3],
                                    positions: wp.array[wp.vec3]):
    vertex=wp.tid()
    positions[vertex]=reference[vertex]+displacement[vertex]


def _publish_without_collisions(self,state,contacts):
    if self.model.particle_count:
        wp.launch(_publish_collision_free_motion,self.model.particle_count,
            [self.pos_prev_collision_detection,self.particle_displacements,state.particle_q],device=self.device)


def _step_without_collisions(self,state_in,state_out,control,contacts,dt,*,observables=None):
    # Ignore externally supplied contacts and any pipeline attached after
    # construction. These host assignments are outside the captured GPU work.
    self.collision_pipeline=None
    self._pipeline_contacts=None
    self.particle_enable_self_contact=False
    self.rigid_soft_enable_dat=False
    self.sdf_backend=None
    return SolverVBD.step(self,state_in,state_out,control,None,dt,observables=observables)


def _initialize_without_collisions_zero_iterations(self,state_in,state_out,contacts,dt):
    # Normally the first color sweep publishes state_out. Without any sweeps,
    # explicitly publish the unrestricted predictor before velocity finalization.
    SolverVBD._initialize_particles(self,state_in,state_out,None,dt)
    if self.model.particle_count:
        wp.copy(state_out.particle_q,state_in.particle_q)


class SolverDAT(SolverVBD):
    """Original DAT controller/local solve with our material and IPC energies."""
    solver_mode='dat'
    def __init__(self,model,*,iterations=None,allow_cpu=False,energy_overrides=None,
                 vertex_contact_capacity=None,edge_contact_capacity=None,
                 contact_precision='float32',enable_planar_dat=True,**stock_options):
        if not isinstance(enable_planar_dat,bool):
            raise TypeError('enable_planar_dat must be a bool')
        if contact_precision not in ('float32','float64'):
            raise ValueError("contact_precision must be 'float32' or 'float64'")
        if hashlib.sha256(Path(inspect.getfile(SolverVBD)).read_bytes()).hexdigest()!=_ORIGINAL_SHA:
            raise ImportError('Energy-only DAT requires the pristine reference runtime in a fresh process')
        if not hasattr(model,'ipc'):raise TypeError('Attach our energy data to the unchanged Newton Model')
        if model.body_count:raise ValueError('Our DAT energies currently support deformables; use our GS for rigid bodies')
        if not model.device.is_cuda and not allow_cpu:raise ValueError('Production DAT requires CUDA')
        p=model.ipc.params
        if any(p.get(k,False) for k in ('use_ccd_guess','use_colored_ccd_guess','use_colored_ccd_guess_level_based')):
            raise ValueError('Energy-only DAT uses the original native predictor; disable GS initial-guess flags')
        if not p.get('dat_inertial_predictor',True):raise ValueError('Energy-only DAT preserves the native inertial predictor')
        if vertex_contact_capacity is not None:stock_options.setdefault('particle_vertex_contact_buffer_size',vertex_contact_capacity)
        if edge_contact_capacity is not None:stock_options.setdefault('particle_edge_contact_buffer_size',edge_contact_capacity)
        if not enable_planar_dat:
            # The ablation disables collision work rather than only clipping.
            # Override caller-provided native flags before any detector/contact
            # buffers or owned pipeline state can be initialized.
            stock_options['particle_enable_self_contact']=False
            stock_options['rigid_soft_enable_dat']=False
            stock_options['collision_pipeline']=None
            stock_options['collision_frequency_type']={slot:SolverBase.CollisionFrequencyType.NONE
                for slot in SolverBase.CollisionSlot}
            stock_options.pop('particle_collision_detection_interval',None)
        self.params=p;self.error=wp.zeros(1,int,device=model.device);self._failed=False
        self.ccd_backend=None;self._guess_requested=False
        super().__init__(model,iterations=iterations,**stock_options)
        self.enable_planar_dat=enable_planar_dat
        if not enable_planar_dat:
            # Bind only this instance; default class lifecycle identities and
            # energy-only substitutions remain unchanged.
            self.step=_step_without_collisions.__get__(self,type(self))
            self._penetration_free_truncation=_publish_without_collisions.__get__(self,type(self))
            if self.iterations==0:
                self._initialize_particles=_initialize_without_collisions_zero_iterations.__get__(self,type(self))
        self._energy_materials=energies.prepare(model)
        overrides={} if energy_overrides is None else dict(energy_overrides)
        self._energy_dhat=float(overrides.get('d_hat',p.get('d_hat',getattr(self,'particle_self_contact_margin',.002))))
        self._energy_friction=float(overrides.get('friction',p.get('friction_coefficient',model.soft_contact_mu)))
        self._energy_eps_v=float(overrides.get('eps_v',p.get('friction_velocity_epsilon',self.friction_epsilon)))
        if not all(np.isfinite(x) for x in (self._energy_dhat,self._energy_friction,self._energy_eps_v)) or self._energy_dhat<=0 or self._energy_friction<0 or self._energy_eps_v<=0:
            raise ValueError('Invalid own contact energy parameters')
        self._energy_contacts=None
        self._contact_k_barrier=float(overrides.get('k_barrier',p.get('k_barrier',100)))
        if self.particle_enable_self_contact:
            self._energy_contacts=energies.EnergyWorkspace(self.trimesh_collision_info,device=model.device,
                k_barrier=self._contact_k_barrier,
                precision=contact_precision,error=self.error)
        if self.use_particle_tile_solve:
            include_tets=model.tet_count>0
            native=importlib.import_module(SolverVBD.__module__)
            include_triangles=not native._is_tet_only_elasticity_model(model)
            two=not include_tets
            self._tiled_elasticity_kernel=energies.make_solve_elasticity_tile(include_triangles,include_tets,two)
            options=wp.get_module_options(native.make_solve_elasticity_tile(include_triangles,include_tets,two).module)
            self._set_module_options({k:options[k] for k in ('fast_math','fuse_fp') if k in options},module=self._tiled_elasticity_kernel.module)
        self.sdf_backend=None
        if enable_planar_dat and getattr(model.ipc,'collider_data',None) is not None and model.ipc.collider_data.shape[0]:
            from .sdf import SdfBackend
            self.sdf_backend=SdfBackend(model,self.error)
        self.configuration={'algorithm':'unmodified native DAT execution; energy-only substitutions',
            'predictor':'native forward_step','local_solve':'native determinant/inverse',
            'source_commit':'f89e82fed043fe78d1844b26d3f289a25db03810',
            'energy_membrane':{'force':'corotated','hessian':'native PSD approximation',
                               'singular_geometry':'native area-ratio floor'},
            'energy_bending':{'evaluator':'native with stiffness mapping','degeneracy':'native guards'},
            'energy_contact':{'d_hat':self._energy_dhat,'friction':self._energy_friction,'eps_v':self._energy_eps_v,
                              'k_barrier':self._contact_k_barrier,'exact_geometry':False,
                              'precision':contact_precision,
                              'distance':'reference float32 closest-point routines',
                              'normal_hessian':'native rank-one approximation',
                              'friction_evaluator':'native reference routine',
                              'precision_replay':'none'}}
        self.configuration['enable_planar_dat']=enable_planar_dat
        self.configuration['collision_processing']=enable_planar_dat
        self.configuration['mesh_collision_detection']=bool(self.particle_enable_self_contact)
        self.configuration['energy_contact']['enabled']=self._energy_contacts is not None
        self.configuration['motion_acceptance']=('native planar DAT and epoch radius cap' if enable_planar_dat
            else 'unrestricted publication; all collision processing and DAT limits disabled')
        if not enable_planar_dat:
            self.configuration['algorithm']='native VBD with collision processing disabled; own material/pin energies'
            self.configuration['energy_sdf']={'enabled':False,'reason':'all collision processing disabled'}
        if self.sdf_backend is not None:
            self.configuration['energy_sdf']={
                'precision':self.sdf_backend.precision,
                'include_curvature':bool(self.sdf_backend.params.include_curvature),
                'normal_hessian':'full spatial curvature' if self.sdf_backend.params.include_curvature else 'PSD normal outer product',
                'geometry':'float32 with binary64 replay' if self.sdf_backend.precision=='float32' else 'binary64',
                'material_transport':'binary64',
                'friction_hessian':'frozen tangent PSD; zero at zero slip'}
    def _accumulate_own_sdf(self,state,dt,color):
        if self.sdf_backend is not None:
            self.sdf_backend.prepare(dt)
            self.sdf_backend.accumulate(state.particle_q,self.particle_q_prev,color,self.particle_forces,self.particle_hessians,dt)
    def reset_status(self):
        self.error.zero_();self._failed=False
    def check_status(self):
        if self._failed:raise RuntimeError('DAT energy error must be reset before continuing')
        code=int(self.error.numpy()[0])
        if code:
            self._failed=True;raise RuntimeError(f'DAT energy evaluation failed with status {code}')
    def prepare_capture(self):
        pass


SolverDAT._solve_particle_iteration,_ENERGY_SUBSTITUTIONS=_energy_iteration()
assert SolverDAT.step is SolverVBD.step
assert SolverDAT._initialize_particles is SolverVBD._initialize_particles
assert SolverDAT._collision_detection_penetration_free is SolverVBD._collision_detection_penetration_free
assert SolverDAT._penetration_free_truncation is SolverVBD._penetration_free_truncation
