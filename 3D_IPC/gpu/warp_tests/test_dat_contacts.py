"""Strict stock-row execution with only our contact energy substituted."""
from pathlib import Path
import sys
import unittest
import numpy as np
import warp as wp

import gpu
_isolated_gpu = Path(__file__).resolve().parents[1]
if str(_isolated_gpu) not in gpu.__path__:
    gpu.__path__.append(str(_isolated_gpu))
from gpu import contact, dat_contacts as energy
from gpu.contact_exact import evaluate_self_exact
from newton._src.geometry.tri_mesh_collision import build_tri_mesh_collision_info
from newton._src.solvers.vbd import particle_vbd_kernels as stock


@wp.kernel
def packets(q:wp.array[wp.vec3], previous:wp.array[wp.vec3], nodes:wp.vec4i, kind:int,
            ctx:energy.EnergyContext, output:wp.array[contact.Result]):
    output[0]=contact.evaluate_self(q,previous,nodes,kind,ctx.parameters)


@wp.kernel
def exact_packet(q:wp.array[wp.vec3], previous:wp.array[wp.vec3], nodes:wp.vec4i, kind:int,
                 ctx:energy.EnergyContext, output:wp.array[contact.Result]):
    output[0]=evaluate_self_exact(q,previous,nodes,kind,ctx.parameters,ctx.words,ctx.slots,0)


contact.configure_module(__name__)


class DatContactEnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init();cls.devices=[wp.get_device('cpu'),*wp.get_cuda_devices()]

    def fixture(self,device,kind,points=None,*,friction=.2,exact=False,colors=None,k=100):
        if points is None:
            points=([[.2,.2,.001],[0,0,0],[1,0,0],[0,1,0]] if kind==0 else
                    [[0,0,0],[1,0,0],[.5,-.5,.001],[.5,.5,.001]])
        q=wp.array(points,wp.vec3,device=device)
        old=np.asarray(points,np.float32).copy();old[0,0]-=.001
        previous=wp.array(old,wp.vec3,device=device)
        triangles=wp.array([[1,2,3]],int,device=device) if kind==0 else wp.empty((0,3),int,device=device)
        edges=wp.empty((0,4),int,device=device) if kind==0 else wp.array([[-1,-1,0,1],[-1,-1,2,3]],int,device=device)
        info=build_tri_mesh_collision_info(4,1 if kind==0 else 0,0 if kind==0 else 2,
            vertex_collision_buffer_pre_alloc=1,edge_collision_buffer_pre_alloc=1,device=device)
        info.vertex_colliding_triangles.fill_(-1)
        info.vertex_colliding_triangles_count.zero_()
        info.edge_colliding_edges.fill_(-1)
        info.edge_colliding_edges_count.zero_()
        if kind==0:
            values=np.full(info.vertex_colliding_triangles.shape[0],-1,np.int32);values[:2]=[0,0]
            info.vertex_colliding_triangles.assign(values);info.vertex_colliding_triangles_count.assign([1,0,0,0])
        else:
            info.edge_colliding_edges.assign([0,1,1,0]);info.edge_colliding_edges_count.assign([1,1])
        groups=wp.array([0,0,0,0] if colors is None else colors,int,device=device)
        workspace=energy.EnergyWorkspace(info,device=device,k_barrier=k,exact_geometry=exact)
        workspace.begin(.01,.002,friction,.01)
        return q,previous,triangles,edges,info,groups,workspace

    def reference(self,q,previous,kind,workspace,*,exact=False):
        output=wp.empty(1,contact.Result,device=q.device)
        wp.launch(exact_packet if exact else packets,1,
            [q,previous,wp.vec4i(0,1,2,3),kind,workspace.context,output],device=q.device,block_dim=1 if exact else 256)
        return output.numpy()[0]

    def run_color(self,data,color=0,*,ke=1000,kd=.1,resolve=True):
        q,previous,triangles,edges,info,groups,workspace=data
        f=wp.zeros(4,wp.vec3,device=q.device);h=wp.zeros(4,wp.mat33,device=q.device)
        p=workspace.context.parameters
        dim=max(4*info.vertex_colliding_triangles_count.shape[0],4*info.edge_colliding_edges_count.shape[0])
        wp.launch(workspace.kernel,dim,
            [float(p.dt),color,previous,q,groups,triangles,edges,info,float(p.d_hat),ke,kd,
             float(p.friction),float(p.eps_v),True,1e-5,workspace.context,f,h],device=q.device,
             max_blocks=q.device.sm_count)
        if resolve:workspace.resolve(q,previous,groups,color,f,h)
        return f.numpy(),h.numpy()

    def assert_scatter(self,data,kind,actual,packet,color=0):
        self.assertEqual(int(packet[48]),0)
        # Stock atomic_add starts from +0; +0 + -0 is +0. Compare the
        # accumulated result, including that signed-zero operation.
        expected_f=np.zeros((4,3),np.float32)+packet[:48].reshape(4,12)[:,:3]
        expected_h=np.zeros((4,3,3),np.float32)+packet[:48].reshape(4,12)[:,3:].reshape(4,3,3)
        colors=data[5].numpy();expected_f[colors!=color]=0;expected_h[colors!=color]=0
        for x,y in zip(actual,(expected_f,expected_h)):
            self.assertEqual(x.tobytes(),y.tobytes())

    def test_source_copy_preserves_every_loop_atomic_and_reference_kernel(self):
        original=stock.accumulate_self_contact_force_and_hessian
        original_vt=stock.evaluate_vertex_triangle_collision_force_hessian_4_vertices
        original_ee=stock.evaluate_edge_edge_contact_2_vertices
        source,proof=energy.cloned_source()
        self.assertTrue(proof['restored_byte_identical'])
        self.assertEqual(proof['stock_source_sha256'],energy.STOCK_SOURCE_SHA256)
        kernel=energy.make_accumulate_self_contact_force_and_hessian()
        self.assertIs(stock.accumulate_self_contact_force_and_hessian,original)
        self.assertIs(stock.evaluate_vertex_triangle_collision_force_hessian_4_vertices,original_vt)
        self.assertIs(stock.evaluate_edge_edge_contact_2_vertices,original_ee)
        self.assertIsNot(kernel.module,original.module)

    def test_vt_role_order_friction_and_color_scatter_match_all_four_own_roles(self):
        for device in self.devices:
            for friction in (0.,.2):
                for colors in ([0,0,0,0],[0,1,0,1],[3,2,1,0]):
                    data=self.fixture(device,0,friction=friction,colors=colors)
                    packet=self.reference(data[0],data[1],0,data[-1]);actual=self.run_color(data)
                    data[-1].check();self.assert_scatter(data,0,actual,packet)
                    if friction:self.assertLess(packet[0],0)

    def test_directed_ee_rows_canonicalize_and_add_each_side_once_including_ties(self):
        fixtures=[None,[[0,0,0],[1,0,0],[.1,.001,0],[.9,.001,0]],
                  [[0,0,0],[0,0,0],[0,.001,0],[1,.001,0]]]
        for device in self.devices:
            for points in fixtures:
                for colors in ([0,0,0,0],[0,1,0,1],[1,1,0,0]):
                    data=self.fixture(device,1,points,colors=colors)
                    packet=self.reference(data[0],data[1],1,data[-1]);actual=self.run_color(data)
                    data[-1].check();self.assert_scatter(data,1,actual,packet)

    def test_kbarrier_is_explicit_and_stock_normal_damping_is_not_in_own_energy(self):
        for device in self.devices:
            data=self.fixture(device,0)
            first=self.run_color(data,ke=1,kd=0)
            data[-1].begin(.01,.002,.2,.01)
            second=self.run_color(data,ke=12345,kd=100)
            for a,b in zip(first,second):self.assertEqual(a.tobytes(),b.tobytes())
            stronger=self.fixture(device,0,k=200)
            for a,b in zip(first,self.run_color(stronger)):np.testing.assert_array_equal(2*a,b)

    def test_invalid_history_and_context_mismatch_latch_without_force_commit(self):
        for device in self.devices:
            data=self.fixture(device,0,points=[[.2,.2,1],[0,0,0],[1,0,0],[0,1,0]])
            old=data[1].numpy();old[0,0]=np.nan;data[1].assign(old)
            f,h=self.run_color(data)
            self.assertEqual(int(data[-1].context.error.numpy()[0]),101)
            self.assertFalse(np.any(f));self.assertFalse(np.any(h))
            data=self.fixture(device,0);data[-1].context.parameters.dt=.0123
            f,h=self.run_color(data)
            self.assertEqual(int(data[-1].context.error.numpy()[0]),101)
            self.assertFalse(np.any(f));self.assertFalse(np.any(h))

    def test_overflow_on_unselected_roles_is_still_an_error(self):
        for device in self.devices:
            # B and C have zero contact weight. Only they are selected, but
            # the all-role energy packet must reject A/V's overflowing F32 H.
            data=self.fixture(device,0,points=[[0,0,1e-25],[0,0,0],[1,0,0],[0,1,0]],
                              colors=[1,1,0,0],exact=False)
            packet=self.reference(data[0],data[1],0,data[-1])
            self.assertEqual(int(packet[48]),4)
            f,h=self.run_color(data)
            self.assertEqual(int(data[-1].context.error.numpy()[0]),104)
            self.assertFalse(np.any(f));self.assertFalse(np.any(h))

    def test_robust_ambiguous_geometry_replays_exact_with_stock_role_color_ownership(self):
        fixtures=[(0,[[.2,.0002,.001],[-1,0,0],[0,0,0],[1,0,0]]),
                  (1,[[-1,0,0],[1,0,0],[-.5,.001,.0002],[.5,.001,.0002]])]
        for device in self.devices:
            for kind,points in fixtures:
                for colors in ([0,0,0,0],[1,1,0,0],[0,1,1,0]):
                    data=self.fixture(device,kind,points,exact=True,colors=colors)
                    fast=self.reference(data[0],data[1],kind,data[-1]);self.assertEqual(int(fast[48]),2)
                    expected=self.reference(data[0],data[1],kind,data[-1],exact=True)
                    actual=self.run_color(data);data[-1].check()
                    expected_requests=(1 if kind==0 else int(0 in colors[:2])+int(0 in colors[2:]))
                    self.assertEqual(int(data[-1].context.pending_count.numpy()[0]),expected_requests)
                    self.assert_scatter(data,kind,actual,expected)

    def test_queue_reset_overflow_and_original_ambiguity_fail_closed(self):
        points=[[.2,.0002,.001],[-1,0,0],[0,0,0],[1,0,0]]
        for device in self.devices:
            data=self.fixture(device,0,points,exact=True)
            self.run_color(data);self.assertEqual(int(data[-1].context.pending_count.numpy()[0]),1)
            data[-1].begin(.01,.002,.2,.01)
            self.assertEqual(int(data[-1].context.pending_count.numpy()[0]),0)
            data[-1].context.pending_nodes=wp.empty(0,wp.vec4i,device=device)
            f,h=self.run_color(data)
            self.assertEqual(int(data[-1].context.error.numpy()[0]),12)
            self.assertFalse(np.any(f));self.assertFalse(np.any(h))
            # Original mode retains its own numerical status; no robust switch.
            data=self.fixture(device,0,points,exact=False)
            packet=self.reference(data[0],data[1],0,data[-1]);self.run_color(data)
            status=int(packet[48])
            self.assertEqual(int(data[-1].context.error.numpy()[0]),0 if status==0 else 100+status)
            self.assertEqual(int(data[-1].context.pending_count.numpy()[0]),0)


if __name__=='__main__':unittest.main()
