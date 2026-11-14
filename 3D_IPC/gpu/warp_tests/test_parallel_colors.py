"""Canonical ordered-greedy colors, independent of dependency execution order."""
from types import SimpleNamespace
import unittest
import numpy as np
import warp as wp
from gpu.parallel_colors import ParallelColors
from gpu.policies import _color_graph
from .test_parallel_levels import make_graph


class ParallelColorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init();cls.devices=[wp.get_device('cpu'),*wp.get_cuda_devices()]

    def compare(self,device,n,edges,captured=False):
        a=make_graph(device,n,edges,np.ones(n),np.zeros(n))
        b=make_graph(device,n,edges,np.ones(n),np.zeros(n))
        old_error=wp.zeros(1,int,device=device);error=wp.zeros(1,int,device=device)
        wp.launch(_color_graph,1,[a.rows,a.adjacency,a.colors,a.seen,a.levels,a.component,a.queue,
            a.trial_order,a.order,a.offsets,a.level_offsets,a.metadata,old_error],device=device)
        workspace=ParallelColors(b);contacts=SimpleNamespace(error=error);workspace.apply(contacts)
        if captured:
            with wp.ScopedCapture(device=device) as capture:workspace.apply(contacts)
            wp.capture_launch(capture.graph)
        self.assertEqual(int(error.numpy()[0]),0);self.assertEqual(int(old_error.numpy()[0]),0)
        for field in ['colors','trial_order','metadata']:
            np.testing.assert_array_equal(getattr(a,field).numpy(),getattr(b,field).numpy(),err_msg=field)
        return b

    def test_random_graphs_match_serial_node_id_greedy_colors(self):
        rng=np.random.default_rng(10491)
        for device in self.devices:
            for n in [1,2,9,31,65]:
                for case in range(10):
                    edges=[(i,j) for i in range(n) for j in range(i) if rng.random()<case/9]
                    with self.subTest(device=str(device),n=n,case=case):self.compare(device,n,edges)

    def test_color_count_has_no_128_color_cap(self):
        n=129;edges=[(i,j) for i in range(n) for j in range(i)]
        for device in self.devices:
            graph=self.compare(device,n,edges)
            np.testing.assert_array_equal(graph.colors.numpy(),np.arange(n))
            self.assertEqual(int(graph.metadata.numpy()[0]),n)

    def test_disconnected_components_and_long_lower_id_dependency_chain(self):
        n=65;edges=[(i,i+1) for i in range(32)]+[(i,i+1) for i in range(34,64)]
        for device in self.devices:self.compare(device,n,edges)

    def test_invalid_csr_fails_closed(self):
        for device in self.devices:
            for adjacency in ([0],[2],[-1]):
                g=make_graph(device,2,[],np.ones(2),np.zeros(2))
                g.rows=wp.array([0,1,1],wp.uint64,device=device)
                g.adjacency=wp.array(adjacency,int,device=device)
                error=wp.zeros(1,int,device=device)
                ParallelColors(g).apply(SimpleNamespace(error=error))
                self.assertEqual(int(error.numpy()[0]),371)

    def test_cuda_partial_warps_and_conditional_capture_replay(self):
        devices=wp.get_cuda_devices()
        if not devices:self.skipTest('CUDA device unavailable')
        for device in devices:
            for n in [1,2,31,65,129]:
                edges=[(i,i+1) for i in range(n-1)]
                self.compare(device,n,edges,captured=True)


if __name__=='__main__':unittest.main()
