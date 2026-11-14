"""Integer scheduling oracles; physical state and CCD are outside this module."""
from types import SimpleNamespace
import unittest
import numpy as np
import warp as wp

from gpu.parallel_levels import ParallelLevels
from gpu.policies import _levels_graph


def make_graph(device, n, edges, moving, movement, seed=0):
    neighbors=[set() for _ in range(n)]
    for left,right in edges:
        if left!=right:
            neighbors[left].add(right);neighbors[right].add(left)
    rows=[0];adjacency=[];colors=[]
    for vertex in range(n):
        adjacency.extend(sorted(neighbors[vertex]));rows.append(len(adjacency))
        forbidden={colors[other] for other in neighbors[vertex] if other<vertex}
        color=0
        while color in forbidden:color+=1
        colors.append(color)
    # Include non-node-ID tie order inside each existing color.
    trial=sorted(np.random.default_rng(seed).permutation(n).tolist(),key=lambda vertex:colors[vertex])
    array=lambda data,dtype:wp.array(data,dtype=dtype,device=device)
    return SimpleNamespace(n=n,device=wp.get_device(device),rows=array(rows,wp.uint64),
        adjacency=array(adjacency,int),colors=array(colors,int),trial_order=array(trial,int),
        moving=array(moving,wp.uint8),movement=array(movement,wp.float64),
        seen=wp.empty(n,int,device=device),component=wp.empty(n,int,device=device),
        queue=wp.empty(n,int,device=device),levels=wp.empty(n,int,device=device),
        order=wp.empty(n,int,device=device),offsets=wp.empty(n+1,int,device=device),
        level_offsets=wp.empty(n+1,int,device=device),metadata=array([max(colors)+1,0,0],int),
        keys=wp.empty(2*n,wp.int64,device=device),values=wp.empty(2*n,int,device=device))


def run_oracle(graph,error):
    g=graph
    wp.launch(_levels_graph,1,[g.rows,g.adjacency,g.colors,g.seen,g.levels,g.component,g.queue,
        g.trial_order,g.order,g.offsets,g.level_offsets,g.metadata,g.moving,g.movement,error],device=g.device)


class ParallelLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init();cls.devices=[wp.get_device('cpu'),*wp.get_cuda_devices()]

    def compare(self,device,n,edges,moving,movement,seed=0,captured=False):
        old=make_graph(device,n,edges,moving,movement,seed)
        new=make_graph(device,n,edges,moving,movement,seed)
        old_error=wp.zeros(1,int,device=device);new_error=wp.zeros(1,int,device=device)
        workspace=ParallelLevels(new);contacts=SimpleNamespace(error=new_error)
        run_oracle(old,old_error);workspace.apply(contacts)
        if captured:
            with wp.ScopedCapture(device=device) as capture:workspace.apply(contacts)
            wp.capture_launch(capture.graph)
        self.assertEqual(int(old_error.numpy()[0]),0);self.assertEqual(int(new_error.numpy()[0]),0)
        for field in ['metadata','levels','colors','trial_order','level_offsets']:
            np.testing.assert_array_equal(getattr(old,field).numpy(),getattr(new,field).numpy(),err_msg=field)
        active=int(old.metadata.numpy()[2]);groups=int(old.metadata.numpy()[1])
        np.testing.assert_array_equal(old.order.numpy()[:active],new.order.numpy()[:active])
        np.testing.assert_array_equal(old.offsets.numpy()[:groups+1],new.offsets.numpy()[:groups+1])
        return new

    def test_our_solver_installs_the_validated_execution_workspaces(self):
        from copy import deepcopy
        from gpu.scene import create_model
        from gpu.solver import SolverIPC
        from gpu.parallel_colors import ParallelColors
        from gpu.contact_original import ContactBackendIPC
        from gpu.ccd import resolve_cached
        from .test_solver_dat import point_over_triangle
        payload=point_over_triangle(height=.1,velocity=(.1,0,0),iterations=0)
        payload['params'].update(use_colored_ccd_guess_level_based=True,
            colored_ccd_guess_iters=1,colored_ccd_guess_fraction=.97)
        for device in self.devices:
            solver=SolverIPC(create_model(deepcopy(payload),device),allow_cpu=True)
            self.assertEqual(solver.ccd_backend.group_query_block_dim,64)
            self.assertIsInstance(solver.contact_backend,ContactBackendIPC)
            self.assertIs(solver.ccd_backend._resolve,resolve_cached)
            graph=solver.ccd_backend._initial_guess.graph
            self.assertIsInstance(graph._parallel_colors,ParallelColors)
            self.assertIsInstance(graph._parallel_levels,ParallelLevels)
            self.assertTrue(solver.contact_backend.track_query_epoch)

    def test_random_graphs_preserve_every_canonical_integer_output(self):
        rng=np.random.default_rng(7101)
        for device in self.devices:
            for n in [1,2,4,8,17,31]:
                for case in range(12):
                    edges=[(i,j) for i in range(n) for j in range(i) if rng.random()<.05+.07*case]
                    moving=(rng.random(n)<.6).astype(np.uint8)
                    movement=rng.integers(0,4,n).astype(np.float64)
                    with self.subTest(device=str(device),n=n,case=case):
                        self.compare(device,n,edges,moving,movement,case)

    def test_disconnected_components_all_maximum_ties_and_rigid_only_components(self):
        edges=[(0,1),(1,2),(2,3),(4,5),(5,6),(6,7)]
        moving=[1,1,0,1,0,0,0,0,1]
        movement=[4.,1.,100.,4.,0.,0.,0.,0.,-0.]
        for device in self.devices:
            g=self.compare(device,9,edges,moving,movement,8)
            np.testing.assert_array_equal(g.levels.numpy(),[0,1,1,0,0,0,0,0,0])

    def test_long_chain_multi_source_bfs_and_no_active_vertices(self):
        n=65;edges=[(i,i+1) for i in range(n-1)]
        for device in self.devices:
            movement=np.zeros(n);movement[[0,n-1]]=2
            graph=self.compare(device,n,edges,np.ones(n,np.uint8),movement)
            np.testing.assert_array_equal(graph.levels.numpy(),np.minimum(np.arange(n),np.arange(n)[::-1]))
            graph=self.compare(device,n,edges,np.zeros(n,np.uint8),np.ones(n))
            self.assertEqual(int(graph.metadata.numpy()[2]),0)
            np.testing.assert_array_equal(graph.levels.numpy(),np.zeros(n,int))

    def test_invalid_movement_fails_before_schedule_publication(self):
        for device in self.devices:
            for value in [float('nan'),float('inf'),-1.]:
                g=make_graph(device,2,[(0,1)],[1,1],[0.,value]);error=wp.zeros(1,int,device=device)
                ParallelLevels(g).apply(SimpleNamespace(error=error))
                self.assertEqual(int(error.numpy()[0]),373)

    def test_cuda_conditional_graph_replay_preserves_canonical_outputs(self):
        devices=wp.get_cuda_devices()
        if not devices:self.skipTest('CUDA device unavailable')
        n=65;edges=[(i,i+1) for i in range(n-1)];movement=np.zeros(n);movement[32]=1.
        for device in devices:self.compare(device,n,edges,np.ones(n,np.uint8),movement,3,captured=True)


if __name__=='__main__':unittest.main()
