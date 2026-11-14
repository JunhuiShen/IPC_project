"""Standalone host proof of unchanged CCD interval operations and statuses.

Run: python -m gpu.warp_tests.run_ccd_cache_host
Only a C++17 compiler is needed. The original projection is extracted directly
from ccd.cuh and receives only trace stores. Generated source and binary live
in a temporary directory; no CUDA SDK, solver state or persistent build needed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def run_checks(*, source=None, compiler=None):
    test_root=Path(__file__).resolve().parent
    source=Path(source).resolve() if source is not None else test_root.parents[1]
    compiler=compiler or shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        raise FileNotFoundError('The CCD invariant proof needs a C++17 compiler')
    headers=[source/'gpu'/name for name in ['ccd.cuh','exact.cuh','original_ccd.cuh','original_rigid.cuh','ccd_cache.cuh']]
    def hashes():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in headers}
    before=hashes()
    text=(source/'gpu/ccd.cuh').read_text()
    begin=text.index('IPC_GPU_CCD_HD inline bool projection(')
    end=text.index('\nIPC_GPU_CCD_HD inline bool separated(',begin)
    body=text[begin:end].strip()
    body=body.replace('IPC_GPU_CCD_HD inline bool projection(', 'inline bool reference_projection(',1)
    needle='double time_lo, double time_hi, double required_gap) {'
    if body.count(needle)!=1:raise RuntimeError('Original projection signature changed; review the trace proof')
    body=body.replace(needle,'double time_lo, double time_hi, double required_gap, gs_ccd_cache::Axis* invariant, gs_ccd_cache::ProjectionTrace* trace) {',1)
    needle='    if (!finite(threshold)) return false;'
    if body.count(needle)!=1:raise RuntimeError('Original projection preamble changed; review the trace proof')
    body=body.replace(needle,needle+'\n    if(invariant) { invariant->direction=axis;invariant->norm=norm;invariant->threshold=threshold;invariant->usable=true; }')
    needle='    return maximum(ab, ba) > threshold;'
    if body.count(needle)!=1:raise RuntimeError('Original projection return changed; review the trace proof')
    body=body.replace(needle,'    if(trace) { trace->first_lo=first_lo;trace->first_hi=first_hi;trace->second_lo=second_lo;trace->second_hi=second_hi;trace->ab=ab;trace->ba=ba; }\n'+needle)
    with tempfile.TemporaryDirectory(prefix='ipc-ccd-cache-host-') as temporary:
        build=Path(temporary)
        (build/'reference_projection.cuh').write_text('// Only name/signature and trace stores differ from the original.\nnamespace ipc_gpu::ccd::detail {\n'+body+'\n}\n')
        command=[str(compiler),'-std=c++17','-O2','-fno-fast-math','-ffp-contract=off',
            '-I'+str(source/'gpu'),'-I'+str(build),str(test_root/'ccd_cache_invariants.cpp'),'-o',str(build/'check')]
        start=time.perf_counter();subprocess.run(command,check=True,capture_output=True,text=True)
        compile_seconds=time.perf_counter()-start
        result=json.loads(subprocess.check_output([str(build/'check')],text=True))
    if hashes()!=before:raise RuntimeError('CCD source changed during the host proof')
    result.update({'scope':'Host differential proof; not a CUDA or performance test.',
        'compiler':str(compiler),'compile_seconds':compile_seconds,'header_sha256':before,'source_unchanged':True})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--compiler');parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=run_checks(source=args.source,compiler=args.compiler)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
