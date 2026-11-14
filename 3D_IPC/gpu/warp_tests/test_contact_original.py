"""All-field/status and one-sided proof checks for GS-owned contact assembly."""
import unittest
import numpy as np
import warp as wp
from gpu import contact
from gpu.contact_original import (ContactBackendIPC,HEADERS,evaluate_original,
    needs_original_evaluation,configure_module)


@wp.kernel
def _packets(q:wp.array[wp.vec3],previous:wp.array[wp.vec3],nodes:wp.vec4i,kind:int,p:contact.Parameters,
             baseline:wp.array[contact.Result],compact:wp.array[contact.Result],needs:wp.array[int]):
    baseline[0]=contact.evaluate_self(q,previous,nodes,kind,p)
    compact[0]=evaluate_original(q,previous,nodes,kind,p)
    needs[0]=needs_original_evaluation(q,previous,nodes,kind,p)


contact.configure_module(__name__)
configure_module(__name__,HEADERS)


def fixtures():
    yield 0,[[.2,.2,.002],[0,0,0],[1,0,0],[0,1,0]]
    yield 0,[[1.2,.2,.002],[0,0,0],[1,0,0],[0,1,0]]
    yield 0,[[.5,0,.002],[0,0,0],[1,0,0],[0,1,0]]
    yield 0,[[.5,.002,0],[0,0,0],[1,0,0],[2,0,0]]
    yield 0,[[.5,1e-11,.001],[0,0,0],[1,0,0],[1,1e-10,0]]
    yield 0,[[0,0,0],[0,0,0],[1,0,0],[0,1,0]]
    yield 0,[[0,0,1e-25],[0,0,0],[1,0,0],[0,1,0]]
    yield 1,[[0,0,0],[1,0,0],[.5,-.5,.002],[.5,.5,.002]]
    yield 1,[[0,0,0],[1,0,0],[.1,.002,0],[.9,.002,0]]
    yield 1,[[0,0,0],[0,0,0],[0,.002,0],[1,.002,0]]
    yield 1,[[0,0,0],[1,0,0],[1.002,0,0],[2,0,0]]
    yield 1,[[0,0,0],[1,0,0],[.2,.001,0],[.8,.001000000001,1e-7]]
    yield 0,[[.2,.2,1.],[0,0,0],[1,0,0],[0,1,0]]
    yield 1,[[0,0,0],[1,0,0],[.5,-.5,1.],[.5,.5,1.]]


class OriginalContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init();cls.devices=[wp.get_device('cpu'),*wp.get_cuda_devices()]

    def evaluate(self,device,q,kind,p,previous=None,nodes=None):
        q=np.asarray(q,np.float32);previous=q.copy() if previous is None else np.asarray(previous,np.float32)
        base=wp.empty(1,contact.Result,device=device);new=wp.empty_like(base);needs=wp.empty(1,int,device=device)
        wp.launch(_packets,1,[wp.array(q,wp.vec3,device=device),wp.array(previous,wp.vec3,device=device),
            wp.vec4i(0,1,2,3) if nodes is None else nodes,kind,p,base,new,needs],device=device)
        base,new,needs=base.numpy(),new.numpy(),int(needs.numpy()[0])
        self.assertEqual(base.tobytes(),new.tobytes(),'All48 force/H and status/active bits must match')
        if not needs:self.assertEqual(base.tobytes(),np.zeros((1,50),np.float32).tobytes(),'Only a truly inactive valid packet may be filtered')
        return base[0],needs

    def test_all_four_diagonals_regular_boundary_and_overflow_packet_bits(self):
        p=contact.parameters(d_hat=.005,k_barrier=100,dt=.01,exact_geometry=False)
        for device in self.devices:
            for kind,q in fixtures():self.evaluate(device,q,kind,p)

    def test_random_far_and_active_packets_and_certificates(self):
        rng=np.random.default_rng(18342);p=contact.parameters(d_hat=.005,k_barrier=100,dt=.01,exact_geometry=False)
        for device in self.devices:
            for _ in range(100):
                x,y=rng.uniform(-.2,1.2,2);z=10**rng.uniform(-5,1)
                self.evaluate(device,[[x,y,z],[0,0,0],[1,0,0],[0,1,0]],0,p)
                self.evaluate(device,[[0,0,0],[1,0,0],[x,-.5,z],[x,.5,z]],1,p)

    def test_invalid_history_scalars_positions_and_zero_stiffness_contract(self):
        points=np.asarray([[.2,.2,1.],[0,0,0],[1,0,0],[0,1,0]],np.float32)
        for device in self.devices:
            for field,value in (('dt',0.),('dt',float('nan')),('dt',1e-200),('d_hat',float('nan')),('d_hat',float('inf')),
                ('k_barrier',float('inf')),('k_barrier',0.),('eps_v',0.),('eps_v',float('nan')),('eps',float('nan')),('friction',-0.0)):
                p=contact.parameters(d_hat=.005,k_barrier=100,dt=.01,exact_geometry=False);setattr(p,field,value)
                self.evaluate(device,points,0,p)
                active=points.copy();active[0,2]=.002;self.evaluate(device,active,0,p)
            for far in (False,True):
                q=points.copy();q[0,2]=1. if far else .002
                for value in (np.nan,np.inf,-np.inf):
                    previous=q.copy();previous[0,0]=value
                    p=contact.parameters(d_hat=.005,k_barrier=100,dt=.01,exact_geometry=False)
                    result,needs=self.evaluate(device,q,0,p,previous)
                    self.assertEqual(needs,1);self.assertEqual(int(result[48]),1)
                    invalid=q.copy();invalid[1,0]=value;self.evaluate(device,invalid,0,p)
            p=contact.parameters(d_hat=.005,k_barrier=0,dt=.01,exact_geometry=False)
            for nodes in (wp.vec4i(-1,1,2,3),wp.vec4i(4,1,2,3)):
                result,needs=self.evaluate(device,points,0,p,nodes=nodes);self.assertEqual(needs,1);self.assertEqual(int(result[48]),1)

    def fixture(self,device,*,friction=0.,exact=False,z=.002):
        from .test_collision import CollisionFixture
        q,model,info,base=CollisionFixture().fixture(device,z=z,friction=friction)
        base.options['exact_geometry']=exact
        backend=ContactBackendIPC(model,info,d_hat=.005,k_barrier=100,friction=friction,exact_geometry=exact)
        backend.refresh()
        self.assertEqual(backend.fast_workers,base.fast_workers)
        if backend._original_zero_friction:
            self.assertEqual(backend._original_block_dim,64)
            self.assertEqual(backend._original_workers,max(128,backend.device.sm_count*256) if backend.device.is_cuda else backend.fast_workers)
        return q,model,base,backend

    def test_fused_scatter_and_rebuilt_queue_one_and_multiple_selected_roles(self):
        for device in self.devices:
            for color_values in ([0,1,2,3],[0,0,1,1],[0,0,0,0]):
                q,model,base,backend=self.fixture(device);model.particle_colors.assign(color_values)
                result=[]
                for worker in (base,backend):
                    f=wp.zeros(4,wp.vec3,device=device);h=wp.zeros(4,wp.mat33,device=device)
                    worker.accumulate(q,q,0,f,h,.01);worker.check();result.append((f.numpy(),h.numpy()))
                for a,b in zip(*result):self.assertEqual(a.tobytes(),b.tobytes())
                self.assertEqual(int(backend.pending_count.numpy()[0]),1)
                # Every call rebuilds the queue at current q. No stale contact
                # packet/certificate is reused when the pair becomes inactive.
                values=q.numpy();values[0,2]=1.;q.assign(values)
                f=wp.zeros(4,wp.vec3,device=device);h=wp.zeros(4,wp.mat33,device=device)
                backend.accumulate(q,q,0,f,h,.01);backend.check()
                self.assertEqual(int(backend.pending_count.numpy()[0]),0)
                self.assertFalse(np.any(f.numpy()));self.assertFalse(np.any(h.numpy()))

    def test_far_geometry_invalid_history_is_not_dropped_and_overflow_fails_closed(self):
        for device in self.devices:
            q,model,base,backend=self.fixture(device,z=1.)
            old=q.numpy();old[0,0]=np.nan;previous=wp.array(old,wp.vec3,device=device)
            f=wp.zeros(4,wp.vec3,device=device);h=wp.zeros(4,wp.mat33,device=device)
            backend.accumulate(q,previous,0,f,h,.01)
            with self.assertRaisesRegex(RuntimeError,'invalid contact input'):backend.check()
            self.assertEqual(int(backend.pending_count.numpy()[0]),1)
            q,model,base,backend=self.fixture(device)
            backend.pending=wp.empty(0,int,device=device)
            backend.accumulate(q,q,0,f,h,.01)
            with self.assertRaisesRegex(RuntimeError,'queue capacity'):backend.check()

    def test_options_changed_before_recapture_fall_back_to_current_policy(self):
        for device in self.devices:
            for exact,friction in ((True,0.),(False,.2),(True,.2)):
                q,model,base,backend=self.fixture(device)
                self.assertTrue(backend._original_zero_friction)
                base.options.update(exact_geometry=exact,friction=friction)
                backend.options.update(exact_geometry=exact,friction=friction)
                outputs=[]
                for worker in (base,backend):
                    f=wp.zeros(4,wp.vec3,device=device);h=wp.zeros(4,wp.mat33,device=device)
                    worker.accumulate(q,q,0,f,h,.01);worker.check();outputs.append((f.numpy(),h.numpy()))
                for a,b in zip(*outputs):self.assertEqual(a.tobytes(),b.tobytes())

    def test_robust_and_friction_inherit_shared_backend_and_dat_is_untouched(self):
        from .test_solver import point_payload
        from gpu.scene import create_model
        from gpu.reference.legacy_dat_adapter import SolverDAT
        original=contact.ContactBackend.accumulate
        for device in self.devices:
            for exact,friction in ((True,0.),(False,.2),(True,.2)):
                q,model,base,backend=self.fixture(device,exact=exact,friction=friction)
                self.assertFalse(backend._original_zero_friction)
                outputs=[]
                for worker in (base,backend):
                    f=wp.zeros(4,wp.vec3,device=device);h=wp.zeros(4,wp.mat33,device=device)
                    worker.accumulate(q,q,0,f,h,.01);worker.check();outputs.append((f.numpy(),h.numpy()))
                for a,b in zip(*outputs):self.assertEqual(a.tobytes(),b.tobytes())
            dat=SolverDAT(create_model(point_payload(),device),allow_cpu=True)
            self.assertIs(type(dat.contact_backend),contact.ContactBackend)
            self.assertIs(dat.contact_backend.accumulate.__func__,original)
        self.assertIs(contact.ContactBackend.accumulate,original)


if __name__=='__main__':unittest.main()
