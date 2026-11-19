"""Native DAT execution with only project energy evaluations replaced.

The upstream step, prediction, detection, clipping, local inverse and velocity
publication are inherited. The original particle-iteration AST is changed only
at energy kernel bindings and their extra energy inputs.
No global upstream function or kernel is modified. For the explicit
``enable_planar_dat=False`` ablation, all collision processing is disabled:
mesh/SDF contact energies, detection, external/owned contacts and DAT limits.
Native inertial prediction, material/pin energies and publication remain.
An independent ``enable_broad_phase_boxes`` opt-in adds only project box/BVH
maintenance to that collision-off path, without querying those hierarchies.
The explicit level-based CCD predictor replaces initialization only, reusing
our GPU implementation. It may run even when collision response in VBD is off.
The separate ``enable_color_ccd`` option clips actual material color updates
using independent per-vertex CCD; it leaves contact energy assembly disabled.
``color_ccd_method`` selects direct_rounding (default) or plane independently
of the predictor CCD method.
``enable_color_ccd_contact_forces`` adds live barrier/friction assembly to that
CCD path without enabling Planar DAT or a second collision detector.
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


def _energy_iteration(scalar_kernel=None):
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
    globals_=dict(vars(native));globals_['_own_solve_elasticity']=energies.solve_elasticity if scalar_kernel is None else scalar_kernel
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


def _initialize_with_boxes(self,state_in,state_out,contacts,dt):
    boxes=self.broad_phase_boxes
    boxes.begin(state_in.particle_qd,dt)
    if self._guess_requested:
        from .dat_initial_guess import initialize
        initialize(self,state_in,state_out,contacts,dt)
    elif self.color_ccd is not None:
        # Keep the gravity-inclusive native objective, but do not publish an
        # unprotected inertial predictor before any CCD candidates exist.
        from newton._src.solvers.vbd.particle_vbd_kernels import forward_step
        model=self.model
        if model.particle_count:
            wp.launch(forward_step,model.particle_count,
                inputs=[dt,model.gravity,model.particle_world,self.particle_q_prev,
                        state_in.particle_q,state_in.particle_qd,model.particle_inv_mass,
                        state_in.particle_f,model.particle_flags],
                outputs=[self.inertia,self.particle_displacements],device=self.device)
            self.pos_prev_collision_detection.assign(state_in.particle_q)
            self.particle_displacements.zero_()
    else:
        SolverVBD._initialize_particles(self,state_in,state_out,None,dt)
    # The predictor has published into state_in. Keep our history completely
    # separate from native DAT's reference/displacement buffers.
    boxes.anchor(state_in.particle_q)
    boxes.update(state_in.particle_q)
    if self.barrier_coloring is not None:
        self.barrier_coloring.update()
        if self.color_ccd is not None:
            self.color_ccd.refresh(state_in.particle_q)
            if self.color_contact_forces is not None:
                self.color_contact_forces.refresh()
    self._box_iteration=0
    if self.iterations==0 and self.model.particle_count:
        wp.copy(state_out.particle_q,state_in.particle_q)


def _solve_iteration_with_boxes(self,state_in,state_out,contacts,dt):
    if self._box_iteration and self._box_iteration%self.broad_phase_boxes.update_interval==0:
        self.broad_phase_boxes.update(state_in.particle_q)
        if self.barrier_coloring is not None:
            self.barrier_coloring.update()
            if self.color_ccd is not None:
                self.color_ccd.refresh(state_in.particle_q)
                if self.color_contact_forces is not None:
                    self.color_contact_forces.refresh()
    # Invoke the exact energy-substituted native iteration, unchanged.
    if self.barrier_coloring is None:
        self._native_energy_iteration(state_in,state_out,None,dt)
    else:
        from .dat_color_schedule import solve_iteration
        solve_iteration(self,state_in,state_out,dt)
    self._box_iteration+=1


def _finalize_with_boxes(self,state_out,dt):
    self.broad_phase_boxes.finish(state_out.particle_q)
    SolverVBD._finalize_particles(self,state_out,dt)


def _rebuild_boxes(self,state):
    self.broad_phase_boxes.update(state.particle_q,rebuild=True)
    if self.color_ccd is not None:
        self.barrier_coloring.update()
        self.color_ccd.refresh(state.particle_q)
        if self.color_contact_forces is not None:
            self.color_contact_forces.refresh()


def _reset_with_boxes(self,state,world_mask=None,flags=None):
    SolverVBD.reset(self,state,world_mask,flags)
    self.broad_phase_boxes.reset(state.particle_q,world_mask)
    if self.color_ccd is not None and state.particle_q is not None:
        self.barrier_coloring.update()
        self.color_ccd.refresh(state.particle_q)
        if self.color_contact_forces is not None:
            self.color_contact_forces.refresh()


class SolverDAT(SolverVBD):
    """Original DAT controller/local solve with our material and IPC energies."""
    solver_mode='dat'
    def __init__(self,model,*,iterations=None,allow_cpu=False,energy_overrides=None,
                 vertex_contact_capacity=None,edge_contact_capacity=None,
                 contact_precision='float32',enable_planar_dat=True,enable_broad_phase_boxes=False,
                 enable_barrier_coloring=False,enable_color_ccd=False,
                 enable_color_ccd_contact_forces=False,color_ccd_method='direct_rounding',local_solve='native',local_solve_damping=None,**stock_options):
        if local_solve not in ('native','cpu_fp32'):
            raise ValueError("local_solve must be 'native' or 'cpu_fp32'")
        self.local_solve=local_solve
        if color_ccd_method not in ('direct_rounding','plane'):
            raise ValueError('color_ccd_method must be direct_rounding or plane')
        if color_ccd_method != 'direct_rounding' and not enable_color_ccd:
            raise ValueError('color_ccd_method=plane requires enable_color_ccd=True')
        if not isinstance(enable_color_ccd_contact_forces,bool):
            raise TypeError('enable_color_ccd_contact_forces must be a bool')
        if enable_color_ccd_contact_forces and not enable_color_ccd:
            raise ValueError('enable_color_ccd_contact_forces requires enable_color_ccd=True')
        if not isinstance(enable_color_ccd,bool):
            raise TypeError('enable_color_ccd must be a bool')
        if enable_color_ccd and (enable_planar_dat or not enable_broad_phase_boxes or not enable_barrier_coloring):
            raise ValueError('enable_color_ccd requires enable_planar_dat=False, enable_broad_phase_boxes=True and enable_barrier_coloring=True')
        if not isinstance(enable_planar_dat,bool):
            raise TypeError('enable_planar_dat must be a bool')
        if not isinstance(enable_broad_phase_boxes,bool):
            raise TypeError('enable_broad_phase_boxes must be a bool')
        if enable_broad_phase_boxes and enable_planar_dat:
            raise ValueError('enable_broad_phase_boxes requires enable_planar_dat=False')
        if not isinstance(enable_barrier_coloring,bool):
            raise TypeError('enable_barrier_coloring must be a bool')
        if enable_barrier_coloring and (enable_planar_dat or not enable_broad_phase_boxes):
            raise ValueError('enable_barrier_coloring requires enable_planar_dat=False and enable_broad_phase_boxes=True')
        if contact_precision not in ('float32','float64'):
            raise ValueError("contact_precision must be 'float32' or 'float64'")
        if hashlib.sha256(Path(inspect.getfile(SolverVBD)).read_bytes()).hexdigest()!=_ORIGINAL_SHA:
            raise ImportError('Energy-only DAT requires the pristine reference runtime in a fresh process')
        if not hasattr(model,'ipc'):raise TypeError('Attach our energy data to the unchanged Newton Model')
        if model.body_count:raise ValueError('Our DAT energies currently support deformables; use our GS for rigid bodies')
        if not model.device.is_cuda and not allow_cpu:raise ValueError('Production DAT requires CUDA')
        p=model.ipc.params
        self.local_solve_damping=float(p.get('damping',1.0) if local_solve_damping is None else local_solve_damping)
        if not np.isfinite(self.local_solve_damping) or self.local_solve_damping<=0:
            raise ValueError('local_solve_damping must be finite and positive')
        if local_solve=='native' and local_solve_damping is not None and self.local_solve_damping!=1.0:
            raise ValueError('The native local solve does not apply damping')
        if any(p.get(k,False) for k in ('use_ccd_guess','use_colored_ccd_guess')):
            raise ValueError('DAT supports the native or level-based CCD predictor; disable other GS initial-guess flags')
        level_guess=p.get('use_colored_ccd_guess_level_based',False)
        if not isinstance(level_guess,bool):
            raise TypeError('use_colored_ccd_guess_level_based must be a bool')
        if level_guess and model.particle_count and not model.tri_count:
            raise ValueError('The VBD level-based CCD predictor requires surface triangles')
        self._ccd_guess_params=None
        if level_guess:
            from .dat_initial_guess import parameters
            self._ccd_guess_params=parameters(p)
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
        self.ccd_backend=None;self._guess_requested=level_guess and model.particle_count>0
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
        self._own_scalar_elasticity_kernel=energies.solve_elasticity
        self._native_energy_iteration=SolverDAT._solve_particle_iteration.__get__(self,type(self))
        if local_solve=='cpu_fp32':
            from . import dat_local_solve_adapter as local_adapter
            self._own_scalar_elasticity_kernel=local_adapter.scalar(self.local_solve_damping)
            self._native_energy_iteration=_energy_iteration(self._own_scalar_elasticity_kernel)[0].__get__(self,type(self))
            self._solve_particle_iteration=self._native_energy_iteration
        overrides={} if energy_overrides is None else dict(energy_overrides)
        self._energy_dhat=float(overrides.get('d_hat',p.get('d_hat',getattr(self,'particle_self_contact_margin',.002))))
        self._energy_friction=float(overrides.get('friction',p.get('friction_coefficient',model.soft_contact_mu)))
        self._energy_eps_v=float(overrides.get('eps_v',p.get('friction_velocity_epsilon',self.friction_epsilon)))
        if not all(np.isfinite(x) for x in (self._energy_dhat,self._energy_friction,self._energy_eps_v)) or self._energy_dhat<=0 or self._energy_friction<0 or self._energy_eps_v<=0:
            raise ValueError('Invalid own contact energy parameters')
        self.broad_phase_boxes=None
        self.barrier_coloring=None
        self.color_ccd=None
        self.color_contact_forces=None
        if enable_broad_phase_boxes:
            from .broad_phase_boxes import BroadPhaseBoxes
            self.broad_phase_boxes=BroadPhaseBoxes(model,self.error,
                minimum=p.get('node_box_min',.001),maximum=p.get('node_box_max',.01),
                d_hat=self._energy_dhat,update_interval=p.get('node_box_update_count',10))
            self._box_iteration=0
            self._initialize_particles=_initialize_with_boxes.__get__(self,type(self))
            self._solve_particle_iteration=_solve_iteration_with_boxes.__get__(self,type(self))
            self._finalize_particles=_finalize_with_boxes.__get__(self,type(self))
            self.rebuild_bvh=_rebuild_boxes.__get__(self,type(self))
            self.reset=_reset_with_boxes.__get__(self,type(self))
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
            if local_solve=='cpu_fp32':
                self._tiled_elasticity_kernel=local_adapter.tiled(include_triangles,include_tets,two,self.local_solve_damping)
            options=wp.get_module_options(native.make_solve_elasticity_tile(include_triangles,include_tets,two).module)
            self._set_module_options({k:options[k] for k in ('fast_math','fuse_fp') if k in options},module=self._tiled_elasticity_kernel.module)
        if enable_barrier_coloring:
            from .barrier_coloring import BarrierColoring
            from .dat_color_schedule import prepare
            self.barrier_coloring=BarrierColoring(model,self.broad_phase_boxes,self.error,
                vertex_capacity=stock_options.get('particle_vertex_contact_buffer_size',128),
                edge_capacity=stock_options.get('particle_edge_contact_buffer_size',256))
            prepare(self)
            if enable_color_ccd:
                from .dat_color_ccd import ColorCcd
                self.color_ccd=ColorCcd(model,self.broad_phase_boxes,self.barrier_coloring,self.error,
                                        method=color_ccd_method)
                if enable_color_ccd_contact_forces:
                    from .dat_color_contacts import ColorContactForces
                    self.color_contact_forces=ColorContactForces(
                        model,self.barrier_coloring,self.color_ccd,self.error,
                        k_barrier=self._contact_k_barrier,precision=contact_precision,
                        edge_parallel_epsilon=stock_options.get('particle_edge_parallel_epsilon',1e-5),
                        damping_bidirectional=self.soft_contact_damping_bidirectional)
        self.sdf_backend=None
        if enable_planar_dat and getattr(model.ipc,'collider_data',None) is not None and model.ipc.collider_data.shape[0]:
            from .sdf import SdfBackend
            self.sdf_backend=SdfBackend(model,self.error)
        self.configuration={'algorithm':'unmodified native DAT execution; energy-only substitutions',
            'predictor':'native forward_step','local_solve':('native determinant/inverse' if local_solve=='native'
                else 'cpu-style float32 scaled inverse / minimum-norm QR fallback'),
            'source_commit':'f89e82fed043fe78d1844b26d3f289a25db03810',
            'energy_membrane':{'force':'project corotated','hessian':'project analytic self-Hessian; no PSD projection',
                               'precision':'float32; binary64 GPU evaluation for ill-conditioned geometry',
                               'precision_guard':'det(F^T F) <= 1e-6 * trace(F^T F)^2',
                               'singular_geometry':'project polar eigenvalue floor; unregularized metric inverse'},
            'energy_bending':{'evaluator':'project dihedral gradient and Gauss-Newton self-Hessian',
                              'precision':'float32; binary64 GPU evaluation for ill-conditioned geometry',
                              'precision_guard':'squared edge/extent or squared sine <= 1e-6',
                              'degeneracy':'project zero-area/edge guards'},
            'energy_contact':{'d_hat':self._energy_dhat,'friction':self._energy_friction,'eps_v':self._energy_eps_v,
                              'k_barrier':self._contact_k_barrier,'exact_geometry':False,
                              'precision':contact_precision,
                              'distance':'reference float32 closest-point routines',
                              'normal_hessian':'native rank-one approximation',
                              'friction_evaluator':'native reference routine',
                              'precision_replay':'none'}}
        self.configuration['local_solve_policy']={'mode':local_solve,'precision':'float32',
            'damping':self.local_solve_damping if local_solve=='cpu_fp32' else 1.0,
            'determinant_threshold':'64*FLT_EPSILON*maxabs(H)^3' if local_solve=='cpu_fp32' else 'absolute 1e-8',
            'singular_fallback':'column-pivoted QR plus QR(R^T), minimum norm' if local_solve=='cpu_fp32' else 'zero update',
            'psd_projection':False,'line_search':False}
        self.configuration['enable_planar_dat']=enable_planar_dat
        self.configuration['broad_phase_boxes']={'enabled':enable_broad_phase_boxes}
        if self.broad_phase_boxes is not None:
            boxes=self.broad_phase_boxes
            self.configuration['broad_phase_boxes'].update({
                'node_count':model.particle_count,'edge_count':model.edge_count,'triangle_count':model.tri_count,
                'node_radius_min':boxes.minimum,'node_radius_max':boxes.maximum,'padding':boxes.padding,
                'contact_padding':boxes.d_hat,'update_interval':boxes.update_interval,
                'radius_policy':'clamp(1.2*max(previous correction, dt*speed), minimum, maximum)',
                'hierarchies':['blue nodes','green triangles (node union + d_hat)','red edges (node union)'],
                'green_edges':'red edge bounds expanded by d_hat; stored without separate BVH',
                'refresh':'after predictor and before iterations k,2k,...; explicit rebuild_bvh rebuilds topology',
                'queries':False,'bounds_enforced':False})
        self.configuration['collision_processing']=enable_planar_dat
        self.configuration['barrier_coloring']={'enabled':enable_barrier_coloring,
            'candidate_queries':enable_barrier_coloring,'drives_vbd_updates':enable_barrier_coloring,
            'contact_response':False if enable_barrier_coloring else bool(enable_planar_dat)}
        if enable_barrier_coloring:
            self.configuration['broad_phase_boxes']['queries']=True
            self.configuration['barrier_coloring'].update({
                'connectivity':['triangle membrane','four-vertex bending hinge','tetrahedron',
                                'four-vertex node-triangle and edge-edge candidates'],
                'candidate_selection':'box overlaps; excludes incidence, incompatible worlds and fully fixed pairs',
                'distance_filter':False,'refresh':'after predictor and at solver box-refresh checkpoints',
                'material_kernel_mapping':self._dynamic_color_proof})
        self.configuration['mesh_collision_detection']=bool(self.particle_enable_self_contact)
        self.configuration['energy_contact']['enabled']=self._energy_contacts is not None
        self.configuration['motion_acceptance']=('native planar DAT and epoch radius cap' if enable_planar_dat
            else 'unrestricted publication; all collision processing and DAT limits disabled')
        if not enable_planar_dat:
            self.configuration['algorithm']='native VBD with collision processing disabled; own material/pin energies'
            self.configuration['energy_sdf']={'enabled':False,'reason':'all collision processing disabled'}
            if enable_barrier_coloring:
                self.configuration['algorithm']='native VBD material/local solve with dynamic barrier+elasticity coloring; collision response disabled'
                self.configuration['motion_acceptance']='unrestricted publication; collision response, CCD and DAT limits disabled'
        if self.sdf_backend is not None:
            self.configuration['energy_sdf']={
                'precision':self.sdf_backend.precision,
                'include_curvature':bool(self.sdf_backend.params.include_curvature),
                'normal_hessian':'full spatial curvature' if self.sdf_backend.params.include_curvature else 'PSD normal outer product',
                'geometry':'float32 with binary64 replay' if self.sdf_backend.precision=='float32' else 'binary64',
                'material_transport':'binary64',
                'friction_hessian':'frozen tangent PSD; zero at zero slip'}
        if self._guess_requested:
            from .dat_initial_guess import prepare,initialize
            prepare(self,stock_options.get('particle_vertex_contact_buffer_size',128),
                    stock_options.get('particle_edge_contact_buffer_size',256))
            if self.broad_phase_boxes is None:
                self._initialize_particles=initialize.__get__(self,type(self))
            self.configuration['predictor']='project level-based colored CCD'
            self.configuration['predictor_settings']={
                'ccd_precision':'float32',
                'ccd_method':self.ccd_backend.method,
                'ccd_uncertain_policy':{
                    'plane':'bounded plane proposal; rounded-endpoint proof or stationary update',
                    'direct_rounding':'closest-feature direct bound with FP32 rounding allowance; endpoint proof or stationary update',
                    'interval':'conservative backtracking; no higher-precision fallback'}[self.ccd_backend.method],
                'updating_sweep_cap':self._ccd_guess_params['colored_ccd_guess_iters'],
                'discarded_probe_sweeps':1 if self._ccd_guess_params['colored_ccd_guess_iters'] else 0,
                'target_fraction':self._ccd_guess_params['colored_ccd_guess_fraction'],
                'target':'step-start position + fraction * dt * velocity (no gravity)',
                'inertial_target':'unchanged native gravity-inclusive forward_step',
                'early_exit':'all targets reached or no represented movement',
                'collision_scope':'deformable mesh vertex-triangle and edge-edge',
                'clipping':'independent per vertex in both probe and updating sweeps',
                'exact_computation_fallback':self._ccd_guess_params.get('exact_computation_fallback',True),
                'ccd_exact_step_bound':self._ccd_guess_params.get('ccd_exact_step_bound',False)}
            active=self._ccd_guess_params['colored_ccd_guess_iters']>0
            self.configuration['predictor_ccd_enabled']=active
            self.configuration['solver_collision_processing']=enable_planar_dat
            self.configuration['collision_processing']=bool(enable_planar_dat or active)
            self.configuration['mesh_collision_detection']=bool(self.particle_enable_self_contact or active)
            self.configuration['algorithm']+='; project level-based CCD initialization'
            if not enable_planar_dat:
                self.configuration['motion_acceptance']='CCD-filtered initialization; unrestricted VBD corrections without contact response or DAT'
        self.configuration['color_ccd']={'enabled':enable_color_ccd}
        if enable_color_ccd:
            self.configuration['color_ccd'].update({
                'method':color_ccd_method,'precision':'float32','gap_floor':1e-8,
                'reduction':'per-vertex atomic minimum over all incident current-color pairs',
                'uncertain_policy':'rounded-endpoint certificate or stationary update',
                'publication':'only current color; accepted reference rebased and rejected displacement discarded'})
            self.configuration['collision_processing']=True
            self.configuration['solver_collision_processing']=True
            self.configuration['mesh_collision_detection']=True
            self.configuration['barrier_coloring']['contact_response']=True
            self.configuration['broad_phase_boxes']['bounds_enforced']=True
            self.configuration['algorithm']='native VBD material/local solve with combined coloring and independent per-color CCD'
            self.configuration['motion_acceptance']='blue-box clamp, per-vertex pairwise CCD minimum, endpoint certificate, current-color publication'
            if not self._guess_requested:
                self.configuration['predictor']='stationary step-start positions; native gravity-inclusive inertia retained'
        self.configuration['color_ccd_contact_forces']={'enabled':enable_color_ccd_contact_forces}
        if enable_color_ccd_contact_forces:
            self.configuration['energy_contact'].update({
                'enabled':True,'energy_model':'project',
                'precision':'float64','assembled_storage_precision':'float32',
                'distance':'project closest-feature geometry',
                'normal_hessian':'full project geometric self-Hessian',
                'friction_evaluator':'project frozen friction gradient and Hessian',
                'precision_replay':'none','exact_geometry':False})
            self.configuration['energy_sdf']={'enabled':False,'reason':'per-color CCD supports mesh contacts only'}
            self.configuration['color_ccd_contact_forces'].update({
                'assembly':'live current-color candidate pairs; per-vertex force/Hessian atomics',
                'barrier':'project IPC gradient and full geometric self-Hessian',
                'friction':'project frozen friction gradient and Hessian; substep-start anchor',
                'friction_coefficient':self._energy_friction,
                'friction_enabled':self._energy_friction>0.0 and self._contact_k_barrier>0.0,
                'barrier_enabled':self._contact_k_barrier>0.0,
                'd_hat':self._energy_dhat,'k_barrier':self._contact_k_barrier,
                'precision':'float64','assembled_storage_precision':'float32','normal_damping':False,
                'evaluator':dict(self.color_contact_forces.configuration),
                'candidate_source':'same box/BVH pairs and color incidence as per-color CCD',
                'planar_dat':False})
            self.configuration['algorithm']+='; live IPC barrier and friction contributions'
    def _accumulate_own_sdf(self,state,dt,color):
        if self.sdf_backend is not None:
            self.sdf_backend.prepare(dt)
            self.sdf_backend.accumulate(state.particle_q,self.particle_q_prev,color,self.particle_forces,self.particle_hessians,dt)
    def reset_status(self):
        self.error.zero_();self._failed=False
    def check_status(self):
        if self._failed:raise RuntimeError('DAT energy error must be reset before continuing')
        if self.barrier_coloring is not None:
            self.barrier_coloring.validate_independence()
        code=int(self.error.numpy()[0])
        if code:
            self._failed=True;raise RuntimeError(f'DAT energy/box/CCD evaluation failed with status {code}')
    def prepare_capture(self):
        pass


SolverDAT._solve_particle_iteration,_ENERGY_SUBSTITUTIONS=_energy_iteration()
assert SolverDAT.step is SolverVBD.step
assert SolverDAT._initialize_particles is SolverVBD._initialize_particles
assert SolverDAT._collision_detection_penetration_free is SolverVBD._collision_detection_penetration_free
assert SolverDAT._penetration_free_truncation is SolverVBD._penetration_free_truncation
