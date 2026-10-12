#!/usr/bin/env python3
"""Build/test an isolated snapshot, then screen opt-in GPU optimizations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from profile_contact_round2_full import write_json
from record_contact_validation import run_test, seal, verify_source
from wait_example1_comparison import activity, sample


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parent.parent;root=args.root.resolve(strict=True)
    suffix=str(args.attempt);manifest=root/('source_manifest_'+suffix+'.json')
    status=dict(pid=os.getpid(),status='preparing',attempt=args.attempt)
    def save(phase,**extra):
        status.update(status=phase,updated_unix=time.time(),**extra)
        write_json(root/('finish_status_'+suffix+'.json'),status)
    def idle():
        quiet=0
        while quiet<2:
            a=sample();time.sleep(3);b=sample()
            reasons,load=activity(a,b,[],max_background_cores=2)
            reasons=[r for r in reasons if r!='I/O wait']
            quiet=0 if reasons else quiet+1
            save('waiting_for_server',reasons=reasons,external_cpu_cores=load)
    def command(name,argv):
        save(name)
        with (root/(name+'_'+suffix+'.log')).open('x') as output:
            subprocess.run(argv,cwd=repo,stdout=output,stderr=subprocess.STDOUT,check=True)
    try:
        verify_source(manifest);idle()
        command('configure',['cmake','-S',str(root/'source/gpu'),'-B',str(root/'build'),
            '-DIPC_CPU_SOURCE='+str(root/'source'),'-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc',
            '-DCMAKE_CXX_COMPILER=/usr/bin/g++','-DCMAKE_CUDA_ARCHITECTURES=89','-DCMAKE_BUILD_TYPE=Release',
            '-DFETCHCONTENT_SOURCE_DIR_EIGEN3='+str(repo/'gpu/deps/eigen3-src'),
            '-DFETCHCONTENT_SOURCE_DIR_TIGHT_INCLUSION='+str(repo/'gpu/deps/tight_inclusion-src')])
        command('build',['cmake','--build',str(root/'build'),'--target','gpu_resident_validation','-j','8'])
        verify_source(manifest);candidate=root/'build/3D_sim_gpu_resident'
        for key in list(os.environ):
            if key.startswith('IPC_GPU_'):del os.environ[key]
        os.environ.update(IPC_GPU_COMPACT_DEFORMABLE_CONTACTS='1',IPC_GPU_BOX_CONTACT_PRUNING='1',
            IPC_GPU_ELASTIC_COLOR_WORKLISTS='1',IPC_GPU_COLOR_PAIR_QUEUES='1',
            IPC_GPU_WARP_CONTACT_REDUCE='0',IPC_GPU_COOPERATIVE_EXACT_CONTACT='1',
            IPC_GPU_CONTACT_SELECTED_NODE_REDUCE='1',IPC_GPU_COMPACT_WARP_REDUCE='1',
            IPC_GPU_COOPERATIVE_GEOMETRY_CONVERSIONS='1',IPC_GPU_GLOBAL_GUESS_CANDIDATES_ONLY='1',
            IPC_GPU_GLOBAL_GUESS_WORKERS='2048',IPC_GPU_GLOBAL_GUESS_THREADS='8')
        tests=[
            ('ctest',['ctest','--test-dir',str(root/'build'),'-R','^gpu_resident_',
                '-E','^gpu_resident_global_guess$','--output-on-failure','--timeout','900','-j','1']),
            ('global_guess_quick',[str(root/'build/gpu_resident_global_guess_test'),'--quick']),
            ('contact_memcheck',['/usr/local/cuda-12.6/bin/compute-sanitizer','--tool','memcheck',
                '--error-exitcode','99',str(root/'build/gpu_resident_contact_worklist_test')]),
            ('cooperative_memcheck',['/usr/local/cuda-12.6/bin/compute-sanitizer','--tool','memcheck',
                '--error-exitcode','99',str(root/'build/gpu_resident_contact_cooperative_test')]),
            ('global_guess_initcheck',['/usr/local/cuda-12.6/bin/compute-sanitizer','--tool','initcheck',
                '--error-exitcode','99',str(root/'build/gpu_resident_global_guess_test'),'--quick']),
        ]
        receipts=[]
        for name,argv in tests:
            idle();save('validating',test=name)
            receipt=root/(name+'_'+suffix+'.receipt.json')
            code=run_test(candidate,manifest,root/(name+'_'+suffix+'.log'),receipt,argv,cwd=repo,timeout=7200)
            if code:raise RuntimeError(name+' failed with exit '+str(code))
            receipts.append(receipt)
        marker=root/('validation_exit_'+suffix+'.json');seal(candidate,manifest,receipts,marker)
        save('screening')
        command('screen',['python3',str(repo/'gpu/experiment_contact_round4.py'),'--mode','screen',
            '--output',str(root/('screen_'+suffix)),
            '--baseline-directory',str(repo/'gpu/contact_round3_20261011_VS0RpV/benchmarks/full_validation'),
            '--baseline',str(repo/'gpu/contact_round3_20261011_VS0RpV/build/3D_sim_gpu_resident'),
            '--candidate',str(candidate),'--validation-marker',str(marker),'--configs',str(root/'configurations.json')])
        save('complete')
    except BaseException as error:
        save('stopped',error=type(error).__name__+': '+str(error));raise


if __name__=='__main__':main()
