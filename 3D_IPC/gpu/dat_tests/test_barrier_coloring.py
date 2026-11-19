"""BVH candidate completeness, combined graph coloring and native VBD parity."""
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverVBD

from gpu.barrier_coloring import BarrierColoring
from gpu.broad_phase_boxes import BroadPhaseBoxes
from gpu.solver_dat import SolverDAT
from gpu.dat_tests.test_broad_phase_boxes import box_model, material_model, with_box_parameters
from gpu.dat_tests.test_planar_toggle import attach, solver, fixture, ForbiddenCollisionObject


def graph_model(device, points, triangles=(), edges=(), tets=()):
    m=box_model(device,points,triangles,edges)
    m.tet_indices=wp.array(np.asarray(tets,np.int32).reshape(-1,4),int,device=device)
    m.tet_count=len(tets)
    m.particle_flags=wp.ones(m.particle_count,int,device=device)
    m.particle_mass=wp.ones(m.particle_count,float,device=device)
    return m


def make_graph(model,vertex_capacity=32,edge_capacity=32):
    error=wp.zeros(1,int,device=model.device)
    boxes=BroadPhaseBoxes(model,error)
    graph=BarrierColoring(model,boxes,error,vertex_capacity=vertex_capacity,edge_capacity=edge_capacity)
    return boxes,graph,error


def solid_material_model(device, *, include_triangles):
    """A deformed solid with active vertex zero and one vertex per color."""
    builder=newton.ModelBuilder(gravity=(0.,0.,0.))
    points=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    builder.add_particles(points,[[0.,0.,0.]]*4,[1.]*4,radius=[0.]*4,flags=[1]*4)
    builder.add_tetrahedron(0,1,2,3,k_mu=9.,k_lambda=13.,k_damp=.02)
    # An inactive skin keeps the standard attach() fixture's triangle arrays
    # populated while still selecting the native tet-only specialization.
    builder.add_triangles([0,0],[1,3],[2,1],
                          tri_ke=[7. if include_triangles else 0.]*2,
                          tri_ka=[11. if include_triangles else 0.]*2,
                          tri_kd=[.01 if include_triangles else 0.]*2)
    builder.color()
    model=with_box_parameters(attach(builder.finalize(device=device),pin=8.))
    q=model.particle_q.numpy().copy()
    q[0]=[.13,-.07,.09]
    model.particle_q.assign(q)
    return model


def overlaps(lo,hi,lo2,hi2):
    return np.all(lo<=hi2) and np.all(lo2<=hi)


def actual_pairs(g):
    nt=g.node_triangle_pairs.numpy(); nc=g.node_triangle_counts.numpy()
    ee=g.edge_edge_pairs.numpy(); ec=g.edge_edge_counts.numpy()
    return ({(i,int(nt[i,j])) for i in range(len(nc)) for j in range(int(nc[i]))},
            {(i,int(ee[i,j])) for i in range(len(ec)) for j in range(int(ec[i]))})


def expected_graph(m,nt,ee):
    triangles=m.tri_indices.numpy(); edges=m.edge_indices.numpy(); tets=m.tet_indices.numpy()
    groups=list(triangles)+[e for e in edges if e[0]>=0 and e[1]>=0]+list(tets)
    groups += [[v,*triangles[t]] for v,t in nt]
    groups += [[*edges[a,2:],*edges[b,2:]] for a,b in ee]
    lower=[set() for _ in range(m.particle_count)]
    for group in groups:
        for a in group:
            for b in group:
                if a>b: lower[int(a)].add(int(b))
    colors=[]
    for neighbors in lower:
        forbidden={colors[j] for j in neighbors}
        c=0
        while c in forbidden:c+=1
        colors.append(c)
    return lower,np.asarray(colors,np.int32)


class BarrierColoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init();cls.devices=[wp.get_device('cpu'),*wp.get_cuda_devices()]

    def test_candidates_and_colors_match_brute_force_after_motion(self):
        points=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[.2,.2,.001],
                           [.8,.2,.001],[.2,.8,.001],[2.,0.,0.],[2.,1.,0.]],np.float32)
        triangles=[[0,1,2],[3,4,5]]
        edges=[[-1,-1,0,1],[-1,-1,1,2],[-1,-1,3,5],[-1,-1,6,7]]
        for device in self.devices:
            m=graph_model(device,points,triangles,edges)
            boxes,g,error=make_graph(m)
            for moved in (points,points+np.asarray([[0,0,0]]*3+[[0,0,3]]*3+[[0,0,0]]*2,np.float32)):
                m.particle_q.assign(moved);boxes.update(m.particle_q);g.update();g.validate_independence()
                self.assertEqual(int(error.numpy()[0]),0)
                lo,hi=boxes.node_lower.numpy(),boxes.node_upper.numpy()
                tl,th=boxes.triangle_lower.numpy(),boxes.triangle_upper.numpy()
                el,eh=boxes.edge_lower.numpy(),boxes.edge_upper.numpy()
                ql,qh=boxes.edge_query_lower.numpy(),boxes.edge_query_upper.numpy()
                nt={(i,t) for i in range(len(points)) for t,tri in enumerate(triangles)
                    if i not in tri and overlaps(lo[i],hi[i],tl[t],th[t])}
                ee={(a,b) for a in range(len(edges)) for b in range(a+1,len(edges))
                    if not set(edges[a][2:])&set(edges[b][2:]) and overlaps(ql[a],qh[a],el[b],eh[b])}
                self.assertEqual(actual_pairs(g),(nt,ee))
                lower,colors=expected_graph(m,nt,ee)
                np.testing.assert_array_equal(g.colors.numpy(),colors)
                rows=g.rows.numpy();adj=g.adjacency.numpy()
                for i in range(len(points)):
                    self.assertEqual(set(adj[int(rows[i]):int(rows[i+1])]),lower[i])
                self.assertEqual(int(g.candidate_count.numpy()[0]),len(nt)+len(ee))
                self.assertEqual(sorted(g.order.numpy().tolist()),list(range(len(points))))

    def test_hinge_and_solid_connectivity_even_without_candidates(self):
        for device in self.devices:
            m=graph_model(device,[[i*10.,0.,0.] for i in range(8)],
                          [[0,1,2],[1,0,3]],[[2,3,0,1]],[[4,5,6,7]])
            boxes,g,error=make_graph(m);g.update();g.validate_independence()
            self.assertEqual(int(error.numpy()[0]),0)
            nt,ee=actual_pairs(g);_,colors=expected_graph(m,nt,ee)
            np.testing.assert_array_equal(g.colors.numpy(),colors)
            self.assertEqual(len(set(colors[:4])),4)
            self.assertEqual(len(set(colors[4:])),4)

    def test_candidate_overflow_is_reported(self):
        for device in self.devices:
            m=graph_model(device,np.zeros((7,3)),[[1,2,3],[4,5,6]])
            boxes,g,error=make_graph(m,vertex_capacity=1)
            g.update()
            self.assertNotEqual(int(error.numpy()[0]),0)

    def test_fixed_surface_and_world_filters(self):
        from types import SimpleNamespace
        for device in self.devices:
            m=graph_model(device,np.zeros((8,3)),[[1,2,3],[5,6,7]],
                          [[-1,-1,1,2],[-1,-1,5,6]])
            m.ipc=SimpleNamespace(surface=wp.ones(8,int,device=device))
            m.particle_world.assign([0,0,0,0,1,1,1,1])
            m.particle_flags.zero_()
            boxes,g,error=make_graph(m)
            g.update();self.assertEqual(actual_pairs(g),(set(),set()))
            m.particle_flags.assign([1,0,0,0,0,0,0,0])
            g.update();self.assertEqual(actual_pairs(g),({(0,0)},set()))
            m.ipc.surface.assign([0,1,1,1,1,1,1,1])
            g.update();self.assertEqual(actual_pairs(g),(set(),set()))
            m.ipc.surface.fill_(1)
            m.particle_world.assign([-1,0,0,0,1,1,1,1])
            g.update();self.assertEqual(actual_pairs(g),({(0,0),(0,1)},set()))
            m.particle_flags.assign([0,1,0,0,0,0,0,0])
            g.update();self.assertFalse(actual_pairs(g)[1])
            m.particle_world.assign([0,0,0,0,-1,-1,-1,-1])
            g.update();self.assertEqual(actual_pairs(g)[1],{(0,1)})
            g.validate_independence();self.assertEqual(int(error.numpy()[0]),0)

    def test_combined_coloring_requires_boxes_and_collision_response_off(self):
        m=fixture('cpu')
        for opts in ({'enable_planar_dat':False},{'enable_planar_dat':True,'enable_broad_phase_boxes':True}):
            with self.assertRaises(ValueError):
                SolverDAT(m,allow_cpu=True,enable_barrier_coloring=True,**opts)
        for invalid in (0,1,'true',None):
            with self.assertRaises(TypeError):
                SolverDAT(m,allow_cpu=True,enable_planar_dat=False,enable_broad_phase_boxes=True,
                          enable_barrier_coloring=invalid)

    def test_dynamic_material_kernel_matches_native_kernel_with_same_colors(self):
        for device in self.devices:
            for tiled in ((False,True) if device.is_cuda else (False,)):
                with self.subTest(device=str(device),tiled=tiled):
                    models=[material_model(device),material_model(device)]
                    for m in models:
                        m.gravity.zero_();m.particle_qd.zero_()
                        q=m.particle_q.numpy().copy();q[3,2]=.2;m.particle_q.assign(q)
                    dynamic=solver(models[0],enabled=False,enable_broad_phase_boxes=True,
                                   enable_barrier_coloring=True,iterations=2,particle_enable_tile_solve=tiled)
                    dynamic.broad_phase_boxes.update(models[0].particle_q)
                    dynamic.barrier_coloring.update()
                    colors=dynamic.barrier_coloring.colors.numpy().copy()
                    models[1].particle_colors=wp.array(colors,int,device=device)
                    models[1].particle_color_groups=[wp.array(np.flatnonzero(colors==c).astype(np.int32),int,device=device)
                                                       for c in range(int(colors.max())+1)]
                    native=solver(models[1],enabled=False,iterations=2,particle_enable_tile_solve=tiled)
                    values=[]
                    for own,m in zip((dynamic,native),models):
                        a,b=m.state(),m.state()
                        own.step(a,b,m.control(),ForbiddenCollisionObject(),.01);own.check_status()
                        values.append((b.particle_q.numpy().copy(),b.particle_qd.numpy().copy()))
                    for a,b in zip(*values):
                        self.assertEqual(a.tobytes(),b.tobytes())
                    self.assertIsNone(dynamic._energy_contacts)
                    self.assertIsNone(dynamic.sdf_backend)
                    self.assertFalse(dynamic.particle_enable_self_contact)

    def test_capture_replay_matches_eager_and_records_every_epoch(self):
        for device in wp.get_cuda_devices():
            outcomes=[]
            for captured in (False,True):
                m=material_model(device)
                own=solver(m,enabled=False,enable_broad_phase_boxes=True,enable_barrier_coloring=True,iterations=3)
                a,b=m.state(),m.state();control=m.control()
                def two_steps():
                    own.step(a,b,control,None,.01);own.step(b,a,control,None,.01)
                if captured:
                    with wp.ScopedCapture(device=device) as capture:two_steps()
                    own.barrier_coloring.reset_statistics()
                    for _ in range(2):wp.capture_launch(capture.graph)
                else:
                    own.barrier_coloring.reset_statistics()
                    for _ in range(2):two_steps()
                own.check_status();own.barrier_coloring.validate_independence();own.check_status()
                outcomes.append((a.particle_q.numpy().copy(),a.particle_qd.numpy().copy()))
                self.assertTrue(own.barrier_coloring.statistics())
            for a,b in zip(*outcomes):self.assertEqual(a.tobytes(),b.tobytes())

    def test_cached_dynamic_kernel_restores_each_solver_compiler_policy(self):
        for device in self.devices:
            for tiled in ((False,True) if device.is_cuda else (False,)):
                with self.subTest(device=str(device),tiled=tiled):
                    instances=[]
                    for fast_math in (False,True):
                        own=SolverDAT(material_model(device),allow_cpu=True,iterations=0,
                                      enable_planar_dat=False,enable_broad_phase_boxes=True,
                                      enable_barrier_coloring=True,particle_enable_tile_solve=tiled,
                                      enable_fast_math=fast_math)
                        from gpu import dat_energies
                        source=own._tiled_elasticity_kernel if tiled else dat_energies.solve_elasticity
                        source_options=wp.get_module_options(source.module)
                        expected={key:source_options[key] for key in ('fast_math','fuse_fp')
                                  if key in source_options}
                        if tiled:
                            self.assertEqual(expected['fast_math'],fast_math)
                        actual=wp.get_module_options(own._dynamic_elasticity_kernel.module)
                        self.assertEqual({key:actual[key] for key in expected},expected)
                        instances.append((own,expected))
                    self.assertIs(instances[0][0]._dynamic_elasticity_kernel,
                                  instances[1][0]._dynamic_elasticity_kernel)
                    # Native step() restores per-instance module policies. No
                    # iteration is needed to test this constructor/cache hazard.
                    for own,expected in (instances[0],instances[1],instances[0]):
                        model=own.model
                        own.step(model.state(),model.state(),model.control(),None,.01)
                        own.check_status()
                        actual=wp.get_module_options(own._dynamic_elasticity_kernel.module)
                        self.assertEqual({key:actual[key] for key in expected},expected)

    def test_solid_material_schedule_preserves_native_solve_and_vertex_zero(self):
        for device in self.devices:
            for include_triangles in (False,True):
                for tiled in ((False,True) if device.is_cuda else (False,)):
                    with self.subTest(device=str(device),mixed=include_triangles,tiled=tiled):
                        models=[solid_material_model(device,include_triangles=include_triangles)
                                for _ in range(2)]
                        initial=models[0].particle_q.numpy().copy()
                        dynamic=solver(models[0],enabled=False,enable_broad_phase_boxes=True,
                                       enable_barrier_coloring=True,iterations=2,
                                       particle_enable_tile_solve=tiled)
                        dynamic.broad_phase_boxes.update(models[0].particle_q)
                        dynamic.barrier_coloring.update()
                        colors=dynamic.barrier_coloring.colors.numpy().copy()
                        np.testing.assert_array_equal(colors,np.arange(4,dtype=np.int32))
                        models[1].particle_colors=wp.array(colors,int,device=device)
                        models[1].particle_color_groups=[wp.array([i],int,device=device) for i in range(4)]
                        native=solver(models[1],enabled=False,iterations=2,particle_enable_tile_solve=tiled)
                        if tiled:
                            self.assertEqual(dynamic._dynamic_elasticity_block_dim,16)
                            self.assertEqual(dynamic._dynamic_elasticity_dim,64)
                        values=[]
                        for own,m in zip((dynamic,native),models):
                            a,b=m.state(),m.state()
                            own.step(a,b,m.control(),ForbiddenCollisionObject(),.01)
                            own.check_status()
                            values.append((b.particle_q.numpy().copy(),b.particle_qd.numpy().copy(),
                                           own.particle_displacements.numpy().copy()))
                        # Vertex zero solves first. Later colors launch three invalid
                        # 16-lane blocks, which must not clear its prior correction.
                        self.assertGreater(float(np.linalg.norm(values[0][0][0]-initial[0])),1.e-7)
                        self.assertGreater(float(np.linalg.norm(values[0][2][0])),1.e-7)
                        for a,b in zip(*values):self.assertEqual(a.tobytes(),b.tobytes())


if __name__=='__main__':unittest.main()
