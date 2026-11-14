"""CPU packet/interval proofs and CUDA parity through the actual cold kernels."""
import unittest
from types import SimpleNamespace
import shutil
from pathlib import Path
import subprocess
import sys
import textwrap
import tempfile
import hashlib
from unittest.mock import patch
import numpy as np
import warp as wp

from gpu.ccd import pair_clip as original_clip
from gpu.ccd import pair_clip_cached as cached_clip, configure_module
from .ccd_cache_cases import synthetic_cases
from .ccd_cache_fixture import SingleQueryResolver
from .run_ccd_cache_host import run_checks


class CcdCompilationIsolationTests(unittest.TestCase):
    def test_virtual_header_sections_keep_validation_and_portable_content_hashing(self):
        from gpu import warp_build
        with tempfile.TemporaryDirectory(prefix='ccd-virtual-header-') as temporary:
            root = Path(temporary)
            (root / 'core.h').write_text('#pragma once\n#include "wrapper.h"\n')
            (root / 'wrapper.h').write_text('#pragma once\n#include "core.h"\n')
            with patch.object(warp_build, 'GPU_ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Recursive project include'):
                    warp_build.flatten_headers(('wrapper.h',))
                first, dependencies = warp_build.flatten_headers(('wrapper.h',),
                    source_overrides={'core.h': '#pragma once\nconstexpr int selected_core=1;\n'})
                second, _ = warp_build.flatten_headers(('wrapper.h',),
                    source_overrides={'core.h': '#pragma once\nconstexpr int selected_core=2;\n'})
                self.assertIn('constexpr int selected_core=1;', first)
                self.assertNotIn(str(root), first)
                self.assertEqual(set(dependencies), {root/'core.h', root/'wrapper.h'})
                self.assertNotEqual(hashlib.sha256(first.encode()).digest(),
                                    hashlib.sha256(second.encode()).digest())
                with self.assertRaisesRegex(ValueError, 'Unsupported system header'):
                    warp_build.flatten_headers(('wrapper.h',),
                        source_overrides={'core.h': '#include <filesystem>\n'})
                with self.assertRaisesRegex(ValueError, 'Not a project device arithmetic header'):
                    warp_build.flatten_headers(('wrapper.h',),
                        source_overrides={root.parent/'outside.h': 'int outside;\n'})

    def test_cold_factory_construction_preserves_fast_hashes_and_preambles(self):
        # A fresh interpreter makes this check independent of test discovery
        # order. Hashing resolves dependencies without compiling CUDA kernels.
        script = textwrap.dedent('''
            import warp as wp
            from gpu import ccd
            wp.init()
            fast = [ccd._prepare, ccd._begin, ccd._begin_round,
                    ccd._certify, ccd._shorten, ccd._next, ccd._commit,
                    ccd.get_independent_certifier(), ccd.get_independent_shortener()]
            def snapshot(kernel):
                options = wp.get_module_options(kernel.module)
                extra = options['extra_build_options']
                source = extra.extra_cuda_preamble
                assert 'struct DevicePool' not in source
                assert 'resolve_original_exact' not in source
                assert 'namespace ipc_gpu::gs_ccd_cache' not in source
                assert options['fast_math'] is False
                assert options['fuse_fp'] is False
                return (kernel.module.get_module_hash(block_dim=64),
                        source, extra.extra_cpu_preamble)
            before = [snapshot(kernel) for kernel in fast]
            cold = [ccd.get_resolver(mode) for mode in ('original', 'cached', 'bound')]
            cold += [ccd.get_independent_resolver(bound) for bound in (False, True)]
            assert len({id(kernel.module) for kernel in cold + fast}) == len(cold + fast)
            assert [snapshot(kernel) for kernel in fast] == before
            for mode, kernel in zip(('original', 'cached', 'bound'), cold):
                assert ccd.get_resolver(mode) is kernel
                source = wp.get_module_options(kernel.module)['extra_build_options'].extra_cuda_preamble
                assert 'struct DevicePool' in source
            assert ccd.resolve_original is ccd.get_resolver('original')
            assert ccd.resolve_cached is ccd.get_resolver('cached')
            assert ccd.resolve_bound is ccd.get_resolver('bound')
            print('CCD factory isolation passed')
        ''')
        result = subprocess.run([sys.executable, '-c', script],
            cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('CCD factory isolation passed', result.stdout)


@wp.kernel
def _direct_packets(q:wp.array[wp.vec3],end:wp.array[wp.vec3],kinds:wp.array[int],
                    exact:wp.array[int],gap:wp.array[int],words:wp.array[wp.uint32],slots:wp.array[int],
                    original_packet:wp.array[wp.vec3d],cached_packet:wp.array[wp.vec3d]):
    i=wp.tid()
    nodes=wp.vec4i(4*i,4*i+1,4*i+2,4*i+3)
    original_packet[i]=original_clip(q,end,nodes,kinds[i],exact[i],gap[i],words,slots,i)
    cached_packet[i]=cached_clip(q,end,nodes,kinds[i],exact[i],gap[i],words,slots,i)


class CachedClipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init();cls.device=wp.get_device('cpu');cls.cases=synthetic_cases()
        q=wp.array(np.concatenate([x['start'] for x in cls.cases]),wp.vec3,device=cls.device)
        end=wp.array(np.concatenate([x['end'] for x in cls.cases]),wp.vec3,device=cls.device)
        kinds=wp.array([x['kind'] for x in cls.cases],int,device=cls.device)
        exact=wp.array([x['exact_geometry'] for x in cls.cases],int,device=cls.device)
        gap=wp.array([x['preserve_gap'] for x in cls.cases],int,device=cls.device)
        old=wp.empty(len(cls.cases),wp.vec3d,device=cls.device);new=wp.empty_like(old)
        wp.launch(_direct_packets,len(cls.cases),[q,end,kinds,exact,gap,
            wp.empty(0,wp.uint32,device=cls.device),wp.empty(0,int,device=cls.device),old,new],device=cls.device)
        cls.old=old.numpy();cls.new=new.numpy()

    def test_all_direct_fraction_status_full_packet_bits_match(self):
        np.testing.assert_array_equal(self.old.view(np.uint64),self.new.view(np.uint64))
        self.assertEqual(set(self.old[:,1].astype(int)),{0,1,2})

    def test_single_query_resolver_exactly_encodes_original_direct_packet(self):
        resolver=SingleQueryResolver(self.device)
        for case,packet in zip(self.cases,self.old):
            with self.subTest(case=case['name']):
                result=resolver.evaluate(case,cached=False)
                np.testing.assert_array_equal(packet.view(np.uint64),result.view(np.uint64))

    def test_single_query_resolver_exactly_encodes_cached_direct_packet(self):
        resolver=SingleQueryResolver(self.device)
        for case,packet in zip(self.cases,self.new):
            with self.subTest(case=case['name']):
                result=resolver.evaluate(case,cached=True)
                np.testing.assert_array_equal(packet.view(np.uint64),result.view(np.uint64))

    def test_opt_in_applies_only_to_our_original_solver_instance(self):
        from gpu.ccd import install_our_instance, resolve_cached as resolve
        from gpu.ccd import resolve_original as original_resolve
        from gpu.solver import SolverIPC
        from gpu.reference.legacy_dat_adapter import SolverDAT
        ours=object.__new__(SolverIPC)
        ours.ccd_backend=SimpleNamespace(exact_step_bound=False,_resolve=original_resolve,device=self.device)
        self.assertIs(install_our_instance(ours),original_resolve)
        self.assertIs(ours.ccd_backend._resolve,resolve)
        with self.assertRaises(ValueError):install_our_instance(ours)
        dat=object.__new__(SolverDAT)
        dat.ccd_backend=SimpleNamespace(exact_step_bound=False,_resolve=original_resolve,device=self.device)
        with self.assertRaises(TypeError):install_our_instance(dat)
        self.assertIs(dat.ccd_backend._resolve,original_resolve)
        with self.assertRaises(TypeError):install_our_instance(SimpleNamespace(ccd_backend=ours.ccd_backend))


class CachedInvariantHostTests(unittest.TestCase):
    def test_original_interval_values_and_all_statuses_match(self):
        if not (shutil.which('clang++') or shutil.which('g++')):
            self.skipTest('C++17 compiler unavailable')
        result=run_checks()
        self.assertTrue(result['all_passed']);self.assertTrue(result['source_unchanged'])
        self.assertEqual(result['cases'],505)
        self.assertEqual(result['interval_checks'],505*256)
        self.assertGreater(result['bitwise_invariant_checks'],1_000_000)
        self.assertEqual(result['represented_status_checks'],54)
        self.assertTrue(all(result['status_counts_safe_uncertain_initial_invalid_capacity']))


class CachedClipCudaTests(unittest.TestCase):
    def test_actual_resolvers_preserve_packets_and_captured_replay(self):
        wp.init();devices=wp.get_cuda_devices()
        if not devices:self.skipTest('CUDA device unavailable')
        cases=synthetic_cases()
        for device in devices:
            resolver=SingleQueryResolver(device);reference={}
            # This launches only the already-real resolve kernels. The giant
            # dual-native _direct_packets kernel above remains CPU-only.
            for case in cases:
                with self.subTest(device=str(device),case=case['name']):
                    expected=resolver.evaluate(case,cached=False)
                    actual=resolver.evaluate(case,cached=True)
                    np.testing.assert_array_equal(expected.view(np.uint64),actual.view(np.uint64))
                    reference[case['name']]=expected
            self.assertEqual({int(packet[1]) for packet in reference.values()},{0,1,2})
            self.assertTrue(any(packet[1]==0 and 0<packet[0]<1 for packet in reference.values()))
            selected=[case for case in cases if case['exact_geometry']==0 and case['preserve_gap']==0
                and any(name in case['name'] for name in ['common_translation','multiple_crossing','initial_contact','nan_start'])]
            for case in selected:
                resolver.upload(case)
                graphs=[]
                for cached in [False,True]:
                    resolver.launch(cached=cached)
                    with wp.ScopedCapture(device=device) as cap:resolver.launch(cached=cached)
                    graphs.append(cap.graph)
                for _ in range(3):
                    for graph in graphs:
                        wp.capture_launch(graph)
                        np.testing.assert_array_equal(reference[case['name']].view(np.uint64),resolver.packet().view(np.uint64))


configure_module(__name__, mode="cached")

if __name__=='__main__':unittest.main()
