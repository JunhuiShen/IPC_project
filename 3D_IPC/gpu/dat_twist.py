"""Shared Newton cloth-twist scene and our energy metadata bridge.

Reference and own-energy DAT use the same 480-frame scene and frame loop.
Runtime-dependent solver imports stay lazy so callers select the appropriate
Newton snapshot before constructing models or solvers.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np


DEFAULT_FRAMES = 480
ANGULAR_VELOCITY = math.pi / 2
FPS = 60
SUBSTEPS = 10
ITERATIONS = 4
VERTICES = 2500
TRIANGLES = 4802


def _sha(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def model_array_snapshot(model):
    """All existing direct device arrays plus native color/adjacency arrays."""
    import warp as wp
    result={}
    def add(name,value):
        if isinstance(value,wp.array):
            host=value.numpy()
            result[name]={'shape':list(host.shape),'dtype':str(host.dtype),'warp_dtype':str(value.dtype),
                          'sha256':_sha(host),'pointer':int(value.ptr) if value.ptr else 0,'object_id':id(value)}
    for name,value in sorted(vars(model).items()):
        if name!='ipc':add(name,value)
    for i,value in enumerate(model.particle_color_groups):add(f'particle_color_groups[{i}]',value)
    adjacency=getattr(model,'soft_mesh_adjacency_device',None)
    if adjacency is not None:
        fields=getattr(adjacency,'_cls',None)
        names=getattr(fields,'vars',{}) if fields is not None else {}
        names=set(names)|set(getattr(adjacency,'__dict__',{}))
        for name in sorted(names):
            try:add('soft_mesh_adjacency_device.'+name,getattr(adjacency,name))
            except AttributeError:pass
    return result


def portable_array_hashes(snapshot):
    return {key:{name:value for name,value in item.items() if name not in ('pointer','object_id')}
            for key,item in snapshot.items()}


def rest_edge_coefficients(triangles,areas,edges,rest_lengths):
    """Our CPU hinge c_e=|e|²/(A_left+A_right), once in existing row order."""
    triangles=np.asarray(triangles,np.int32).reshape(-1,3)
    edges=np.asarray(edges,np.int32).reshape(-1,4)
    areas=np.asarray(areas,np.float64).reshape(-1)
    lengths=np.asarray(rest_lengths,np.float64).reshape(-1)
    if len(areas)!=len(triangles) or len(lengths)!=len(edges):raise ValueError('Incomplete stock rest metadata')
    if not np.isfinite(areas).all() or np.any(areas<=0) or not np.isfinite(lengths).all() or np.any(lengths<0):raise ValueError('Invalid stock rest measure')
    faces={}
    for i,triangle in enumerate(triangles):
        for j in range(3):
            a,b=int(triangle[j]),int(triangle[(j+1)%3])
            faces.setdefault(tuple(sorted((a,b))),[]).append((i,int(triangle[(j+2)%3])))
    coefficients=np.zeros(len(edges),np.float32);native_order=np.empty((len(edges),4),np.int32)
    for i,(o0,o1,e0,e1) in enumerate(edges):
        native_order[i]=(e0,e1,o0,o1)
        incident=faces.get(tuple(sorted((int(e0),int(e1)))),[])
        if len(incident)>2:raise ValueError('Nonmanifold stock edge')
        if o0<0 or o1<0:
            continue
        if len(incident)!=2 or {v for _,v in incident}!={int(o0),int(o1)}:raise ValueError('Stock edge/triangle rest adjacency differs')
        area_sum=areas[incident[0][0]]+areas[incident[1][0]]
        coefficients[i]=np.float32(lengths[i]*lengths[i]/area_sum)
        if not np.isfinite(coefficients[i]) or coefficients[i]<=0:raise ValueError('Invalid represented hinge coefficient')
    return coefficients,native_order


def attach_stock_ipc(model,*,fps=60,substeps=10,iterations=4,d_hat=.002,k_barrier=100.,
                     friction=.2,eps_v=.01,exact_geometry=True,gs_predictor='colored',bending_mapping='matched-effective',
                     colored_ccd_guess_iters=1,colored_ccd_guess_fraction=1.):
    """Add only model.ipc; stock material/flags/colors/rest/state arrays stay intact."""
    import warp as wp
    if hasattr(model,'ipc'):raise ValueError('Bridge requires an unmodified stock model')
    if model.body_count or model.tet_count:raise ValueError('This bridge is for the official cloth twist model')
    if gs_predictor not in ('none','global','colored','level'):raise ValueError('Unknown explicit GS predictor')
    if isinstance(colored_ccd_guess_iters,bool) or not isinstance(colored_ccd_guess_iters,int) or colored_ccd_guess_iters<0:
        raise ValueError('Colored CCD updating cap must be a nonnegative integer')
    if isinstance(colored_ccd_guess_fraction,bool) or not math.isfinite(colored_ccd_guess_fraction) or not 0<=colored_ccd_guess_fraction<=1:
        raise ValueError('Colored CCD target fraction must be finite and in [0,1]')
    before=model_array_snapshot(model)
    q=model.particle_q.numpy();mass=model.particle_mass.numpy();flags=model.particle_flags.numpy()
    triangles=model.tri_indices.numpy();areas=model.tri_areas.numpy();edges=model.edge_indices.numpy()
    coeff,native_order=rest_edge_coefficients(triangles,areas,edges,model.edge_rest_length.numpy())
    tri_materials=model.tri_materials.numpy();edge_materials=model.edge_bending_properties.numpy()
    lengths=model.edge_rest_length.numpy()
    if bending_mapping not in ('matched-effective','same-numeric'):raise ValueError('Unknown explicit bending mapping')
    mapped_bending=np.zeros(len(edges),np.float32)
    active=coeff>0
    if bending_mapping=='matched-effective':
        mapped_bending[active]=(edge_materials[active,0].astype(np.float64)*lengths[active].astype(np.float64)/(2.*coeff[active].astype(np.float64))).astype(np.float32)
    else:mapped_bending[active]=edge_materials[active,0]
    energy_properties=np.array(edge_materials,copy=True);energy_properties[:,0]=mapped_bending
    stock_scale=np.asarray(edge_materials[:,0]*lengths,np.float32)
    own_scale=np.asarray(np.float32(2.)*mapped_bending*coeff,np.float32)
    if not np.isfinite(mapped_bending).all():raise ValueError('Invalid represented stock bending conversion')
    if not np.isfinite(q).all() or not np.isfinite(mass).all() or np.any(mass<0):raise ValueError('Invalid stock particle input')
    dt=1./(fps*substeps);gamma=.85;stock_radius=.002+.0015;budget=.5*gamma*stock_radius;n=model.particle_count
    gravity=model.gravity.numpy().reshape(-1,3)
    if not np.all(gravity==0):raise ValueError('Official stock twist gravity must remain zero')
    p={'fps':float(fps),'substeps':int(substeps),'max_global_iters':int(iterations),'dt':dt,'dt2':dt*dt,
       'fixed_iters':True,'damping':1.,'gravity':[0.,0.,0.],'mu':float(tri_materials[0,0]),'lambda':float(tri_materials[0,1]),
       'kB':float(mapped_bending.max(initial=0)),'kpin':0.,'density':.2,'thickness':1.,
       'solid_mu':0.,'solid_lambda':0.,'solid_density':.2,'rigid_density':.2,
       'd_hat':float(d_hat),'k_barrier':float(k_barrier),'friction_coefficient':float(friction),
       'friction_velocity_epsilon':float(eps_v),'exact_computation_fallback':bool(exact_geometry),
       'exact_computation_fallback_2':bool(exact_geometry),'ccd_exact_step_bound':False,
       'node_box_max':budget,'node_box_min':min(.001,budget),'node_box_update_count':int(iterations),
       'theta_box_min':.01,'theta_box_max':.1,'use_ccd':True,'use_ccd_guess':gs_predictor=='global',
       'use_colored_ccd_guess':gs_predictor=='colored','use_colored_ccd_guess_level_based':gs_predictor=='level',
       'colored_ccd_guess_iters':colored_ccd_guess_iters,'colored_ccd_guess_fraction':float(colored_ccd_guess_fraction),'use_jacobi_level_based_ccd':False,
       'use_dat':False,'use_ogc':False,'use_cloth_grid':False,'use_parallel':True,'use_simd':False,
       'dat_gamma':gamma,'dat_query_radius':stock_radius,'dat_node_box_padding':1.,'dat_collision_detection_interval':0,
       'dat_inertial_predictor':True,'k_sdf':0.,'eps_sdf':.002,'rigid_pin_time':0.,'verbose':False,'write_substeps':False}
    def upload(value,dtype,shape=None):
        array=np.asarray(value)
        if shape is not None:array=array.reshape(shape)
        return wp.array(array,dtype=dtype,device=model.device)
    empty3=np.empty((0,3),np.float32)
    payload={'example':'official_cloth_twist','uses_general_solver':False,'bodies':[],'rigid_pins':[],
             'pins':[],'colliders':[],'motion':{'pins':[],'colliders':[]},'params':p}
    ipc=SimpleNamespace(payload=payload,params=p,precision='float32',node_count=n,
        node_body=upload(np.full(n,-1,np.int32),wp.int32),deformable=upload(np.ones(n,np.int32),wp.int32),
        solid=upload(np.zeros(n,np.int32),wp.int32),surface=upload(np.ones(n,np.int32),wp.int32),
        pinned=upload(np.zeros(n,np.int32),wp.int32),pin_targets=upload(q,wp.vec3),original_mass=model.particle_mass,
        tri_inverse=model.tri_poses,tri_area=model.tri_areas,
        tri_active=upload(((tri_materials[:,0]>0)|(tri_materials[:,1]>0)).astype(np.int32),wp.int32),
        dat_triangle_materials=wp.array(np.ascontiguousarray(tri_materials[:,:2],dtype=np.float32),dtype=wp.float32,device=model.device),
        tet_inverse=model.tet_poses,tet_measure=upload([],wp.float32),
        tet_grad_N=wp.array(np.empty((0,4,3),np.float32),dtype=wp.vec3,ndim=2,device=model.device),
        edge_coefficient=upload(coeff,wp.float32),dat_bending_stiffness=upload(mapped_bending,wp.float32),
        stock_energy_edge_properties=wp.array(energy_properties,dtype=wp.float32,device=model.device),
        edge_native_order=upload(native_order,wp.vec4i),
        edge_active=upload(((coeff>0)&(edge_materials[:,0]>0)).astype(np.int32),wp.int32),
        body_references=upload(np.zeros_like(q),wp.vec3),body_nodes=upload([],wp.int32),body_node_offsets=upload([0],wp.int32),
        body_translate=upload([],wp.int32),body_rotate=upload([],wp.int32),
        body_second_moment=upload(np.empty((0,3,3),np.float32),wp.mat33),
        rigid_pin_bodies=upload([],wp.int32),rigid_pin_references=upload(empty3,wp.vec3),rigid_pin_targets=upload(empty3,wp.vec3),
        collider_records=[],collider_data=wp.array(np.empty((0,32),np.float32),dtype=wp.float32,device=model.device),
        collider_host=np.empty((0,32),np.float32))
    model.ipc=ipc
    after=model_array_snapshot(model)
    if before!=after:
        del model.ipc
        raise RuntimeError('Bridge modified existing stock model arrays')
    receipt={'existing_arrays_unchanged':True,'native_arrays':portable_array_hashes(before),
             'particles':n,'triangles':model.tri_count,'edges':model.edge_count,'bodies':model.body_count,'tets':model.tet_count,
             'inactive_hard_pins':int(np.count_nonzero((flags&1)==0)),'finite_pins':0,'coefficient_formula':'edge_rest_length²/(adjacent stock tri_areas sum)',
             'edge_coefficients_sha256':_sha(coeff),'coefficient_range':[float(coeff.min()) if len(coeff) else 0.,float(coeff.max()) if len(coeff) else 0.],
             'bending_mapping':bending_mapping,'mapped_kB_formula':'stock edge_ke * rest_length / (2*c_e)' if bending_mapping=='matched-effective' else 'same numeric stock edge_ke (diagnostic)',
             'stock_edge_ke_sha256':_sha(edge_materials[:,0]),'stock_edge_kd_sha256':_sha(edge_materials[:,1]),
             'mapped_own_kB_sha256':_sha(mapped_bending),'stock_effective_bending_scale_sha256':_sha(stock_scale),
             'own_effective_bending_scale_sha256':_sha(own_scale),
             'effective_bending_scale_max_abs_error':float(np.abs(stock_scale.astype(np.float64)-own_scale.astype(np.float64)).max(initial=0)),
             'tri_mapping':'Stock MODEL slots are Lamé mu/lambda; native helper internally converts lambda_NH=lambda_slot+mu_slot. Copy both slots directly, with no subtraction/thickness scaling; large-strain energy changes intentionally.',
             'raw_stock_tri_materials_sha256':_sha(tri_materials),'own_triangle_materials_sha256':_sha(np.ascontiguousarray(tri_materials[:,:2],dtype=np.float32)),
             'small_strain_uniaxial_modulus_stock_and_own':(2.*tri_materials[:,0]+tri_materials[:,1]).astype(np.float64).tolist(),
             'params':p,'energy_note':'Own corotated/hinge and IPC/friction are deliberate energy substitutions; stock physical arrays and hard constraints are preserved.'}
    ipc.stock_bridge_receipt=receipt
    return receipt


class _DisabledCollisionPipeline:
    """Keep stock-scene metadata available without executing its pipeline."""
    def __init__(self, original):
        self.original = original

    def collide(self, *args, **kwargs):
        return None

    def contacts(self):
        return None

    def __getattr__(self, name):
        return getattr(self.original, name)


def make_example_type(original_example, angular_velocity=ANGULAR_VELOCITY, *, enable_collision_pipeline=True):
    if not math.isfinite(angular_velocity) or angular_velocity <= 0:
        raise ValueError("Twist angular velocity must be finite and positive")
    if not isinstance(enable_collision_pipeline, bool):
        raise TypeError("enable_collision_pipeline must be a bool")

    class DatClothTwist(original_example):
        def capture(self):
            actual = (self.model.particle_count, self.model.tri_count,
                      self.fps, self.sim_substeps, self.iterations)
            expected = (VERTICES, TRIANGLES, FPS, SUBSTEPS, ITERATIONS)
            if actual != expected:
                raise RuntimeError(f"DAT twist scene changed: {actual}, expected {expected}")
            # The original scene assigns opposite rotation axes to its edges.
            # Set their common speed before simulate() captures kernel inputs.
            self.rot_angular_velocity = float(angular_velocity)
            if not enable_collision_pipeline:
                if self.model.shape_count != 0:
                    raise RuntimeError("The shared cloth benchmark must have no collision shapes")
                if not isinstance(self.collision_pipeline, _DisabledCollisionPipeline):
                    self.collision_pipeline = _DisabledCollisionPipeline(self.collision_pipeline)
                self.contacts = None
            return super().capture()

    return DatClothTwist


def run(arguments, *, enable_collision_pipeline=True):
    import newton.examples
    from newton.examples.cloth.example_cloth_twist import Example

    original_arguments = sys.argv
    try:
        sys.argv = ["dat-cloth-twist", *arguments]
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=DEFAULT_FRAMES)
        viewer, args = newton.examples.init(parser)
        example = make_example_type(Example, enable_collision_pipeline=enable_collision_pipeline)(viewer, args)
        newton.examples.run(example, args)
        return 0
    except SystemExit as result:
        return int(result.code or 0)
    finally:
        sys.argv = original_arguments


def run_own(arguments, *, enable_planar_dat=True, enable_broad_phase_boxes=False,
            enable_barrier_coloring=False, enable_color_ccd=False,
            enable_color_ccd_contact_forces=False,color_ccd_method="direct_rounding",
            color_ccd_accept_uncertified=False,
            vertex_contact_capacity=None, edge_contact_capacity=None,
            level_based_ccd_guess=False,colored_ccd_guess_iters=1,colored_ccd_guess_fraction=1.,ccd_method=None):
    from .solver_dat import SolverDAT
    import newton.solvers

    original_solver = newton.solvers.SolverVBD
    solvers = []

    def energy_solver(model, **stock_options):
        if level_based_ccd_guess:
            attach_stock_ipc(model,gs_predictor="level",colored_ccd_guess_iters=colored_ccd_guess_iters,
                             colored_ccd_guess_fraction=colored_ccd_guess_fraction)
        else:
            attach_stock_ipc(model, gs_predictor="none")
        if ccd_method is not None:
            model.ipc.params["ccd_method"] = ccd_method
        if vertex_contact_capacity is not None:
            stock_options["particle_vertex_contact_buffer_size"] = vertex_contact_capacity
        if edge_contact_capacity is not None:
            stock_options["particle_edge_contact_buffer_size"] = edge_contact_capacity
        solver = SolverDAT(model, enable_planar_dat=enable_planar_dat,
                           enable_broad_phase_boxes=enable_broad_phase_boxes,
                           enable_barrier_coloring=enable_barrier_coloring,
                           enable_color_ccd=enable_color_ccd,
                           **({"color_ccd_method": color_ccd_method} if color_ccd_method != "direct_rounding" else {}),
                           **({"color_ccd_accept_uncertified": True} if color_ccd_accept_uncertified else {}),
                           **({"enable_color_ccd_contact_forces": True} if enable_color_ccd_contact_forces else {}),
                           **stock_options)
        solvers.append(solver)
        return solver

    try:
        newton.solvers.SolverVBD = energy_solver
        result = run(arguments, enable_collision_pipeline=enable_planar_dat)
        if result:
            return result
        # Check our latched numerical errors outside the official timer.
        for solver in solvers:
            solver.check_status()
            coloring = getattr(solver, "barrier_coloring", None) if enable_barrier_coloring else None
            if coloring is not None:
                print(json.dumps({"barrier_coloring_statistics": coloring.statistics()}), flush=True)
        return 0
    finally:
        newton.solvers.SolverVBD = original_solver


def run_gs(arguments, *, enable_collisions=True, ccd_method=None, accept_uncertified=False,
           guess_accept_uncertified=False):
    """Run the unchanged stock scene with our GS update and energy kernels."""
    from .solver import SolverIPC
    import newton.solvers

    original_solver = newton.solvers.SolverVBD
    solvers = []

    def gs_solver(model, **stock_options):
        attach_stock_ipc(model, gs_predictor="level" if guess_accept_uncertified else "colored")
        if ccd_method is not None:
            model.ipc.params["ccd_method"] = ccd_method
        if accept_uncertified:
            model.ipc.params["gs_ccd_accept_uncertified"] = True
        if guess_accept_uncertified:
            model.ipc.params["gs_ccd_guess_accept_uncertified"] = True
        solver = SolverIPC(model, iterations=stock_options.get("iterations", ITERATIONS),
            enable_collisions=enable_collisions,
            vertex_contact_capacity=stock_options.get("particle_vertex_contact_buffer_size", 512),
            edge_contact_capacity=stock_options.get("particle_edge_contact_buffer_size", 1024))
        solvers.append(solver)
        return solver

    try:
        newton.solvers.SolverVBD = gs_solver
        result = run(arguments, enable_collision_pipeline=enable_collisions)
        if result:
            return result
        for solver in solvers:
            solver.check_status()
        return 0
    finally:
        newton.solvers.SolverVBD = original_solver
