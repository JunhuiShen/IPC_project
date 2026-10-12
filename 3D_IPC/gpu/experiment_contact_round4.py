#!/usr/bin/env python3
"""Isolated round-4 profiles and controlled execution-only GPU experiments.

Never edits solver sources or production binaries. Profiles are not clean
timings. Cold restarts compare against cold restarts, then a selected winner
must reproduce all continuous reference GEO/BIN files before claiming speedup.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
from statistics import median
import subprocess
import time

from analyze_example1_full_profile import read_phases
from example1_comparison import parse_trial, verify_readme_command
from profile_contact_round2_full import FEATURES, SCHEDULE, write_json
from profile_example1_full import checkpoint_manifest, sha256
from tune_example1_contact import SCENARIOS, parse_segment, segment_command
from validate_contact_round3 import CANDIDATE_FLAGS, validate_marker
from validate_optimized_example1_full import (
    build_report, checked_cpu_baseline, clean_environment, load_clean_comparison, read_reference,
    safe_completed_frame, verify_ready_frames, write_report,
)
from wait_example1_comparison import HZ, activity, physical_cpus, sample, stop_our_process

BASE_SHA = '48c0bc16ce1446b77ec548ba7e9a28501d7da7e0dc748984dfa81cb1f4aff402'
CPU_SHA = '25a3a8a6b06312bb141f7bc940a9981d7f3243837f6dff8c798db14c89335a0e'
POLICY = ('Routine system I/O wait is recorded but does not block, per user authorization. '
          'Competing simulations, GPU jobs, compilers, explicit transfers, swapping and '
          'aggregate external CPU work above 2 cores still block. This is not proof of zero I/O interference.')
OPTIONS = {
    'IPC_GPU_CONTACT_SELECTED_NODE_REDUCE': {'0','1'},
    'IPC_GPU_COMPACT_WARP_REDUCE': {'0','1'},
    'IPC_GPU_COOPERATIVE_GEOMETRY_CONVERSIONS': {'0','1'},
    'IPC_GPU_GLOBAL_GUESS_CANDIDATES_ONLY': {'0','1'},
    'IPC_GPU_CONTACT_SELECTED_NODE_THREADS': {'1','2','4','8','16','32','64','128'},
    'IPC_GPU_GLOBAL_GUESS_THREADS': {'1','2','4','8','16','32'},
}


def validate_flags(flags):
    if not isinstance(flags,dict):raise ValueError('Flags must be a dictionary')
    for key,value in flags.items():
        if not isinstance(value,str):raise ValueError('Flag values must be strings')
        if key in OPTIONS:
            if value not in OPTIONS[key]:raise ValueError('Invalid execution option '+key)
        elif key in ('IPC_GPU_COMPACT_WARP_REDUCE_MIN_DEGREE','IPC_GPU_GLOBAL_GUESS_WORKERS'):
            maximum=2147483647 if key.endswith('MIN_DEGREE') else 8192
            if not value.isascii() or not value.isdecimal() or not 1<=int(value)<=maximum:
                raise ValueError('Invalid integer execution option '+key)
        else:raise ValueError('Unreviewed/diagnostic option forbidden in clean experiments: '+key)


def environment(flags=None):
    result = clean_environment(SCHEDULE, FEATURES)
    result.update(CANDIDATE_FLAGS)
    result.update(flags or {})
    return result


def rank_variants(runs, configs):
    """Screening score is equal-weight geometric mean of paired stage ratios."""
    references = {(r['stage'], r['repeat']): r for r in runs
                  if r['variant'] == 'baseline' and r.get('status') == 'accepted'}
    rows = []
    for config in configs:
        selected = [r for r in runs if r['variant'] == config['name'] and r.get('status') == 'accepted']
        if len(selected) != 3 or {r['stage'] for r in selected} != set(SCENARIOS):
            continue
        ratios = {r['stage']: references[r['stage'], r['repeat']]['total_solver_s'] / r['total_solver_s']
                  for r in selected}
        rows.append(dict(**config, stage_speedups=ratios,
                         score=math.exp(sum(math.log(x) for x in ratios.values())/3)))
    return sorted(rows, key=lambda x: -x['score'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('profile', 'screen'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-directory', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--validation-marker', type=Path)
    parser.add_argument('--configs', type=Path)
    parser.add_argument('--stage', choices=tuple(SCENARIOS), action='append', help='Profile only; omit for all three stages')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    root = args.output.resolve();root.mkdir(exist_ok=True)
    if any(root.iterdir()):raise ValueError('Require a fresh output directory')
    verify_readme_command(repo)
    baseline = args.baseline.resolve(strict=True)
    clean_dir = args.baseline_directory.resolve(strict=True)
    clean_path = clean_dir/'clean_report.json'
    clean = json.loads(clean_path.read_text())
    if (sha256(baseline) != BASE_SHA or clean['command_metadata']['executable_sha256'] != BASE_SHA
            or clean['instrumented'] is not False or clean['byte_identical_to_reference'] is not True
            or abs(clean['metrics']['total_solver_s']-903.817696)>1e-6):
        raise ValueError('Wrong completed round-3 baseline')
    reference_path=clean_dir/'checkpoint_manifest.json'
    reference=read_reference(reference_path,240)
    clean=load_clean_comparison(clean_path,240,sha256(repo/'README.md'),sha256(Path(clean['reference_manifest'])))
    cpu=checked_cpu_baseline(clean_dir/'cpu_baseline.json')
    pinned={baseline:BASE_SHA, clean_path:sha256(clean_path), reference_path:sha256(reference_path),
            repo/'build/3D_sim':CPU_SHA, repo/'README.md':sha256(repo/'README.md'),
            clean_dir/'cpu_baseline.json':sha256(clean_dir/'cpu_baseline.json'),
            Path(__file__).resolve():sha256(Path(__file__).resolve())}
    candidate=None;configs=[];marker=None
    if args.mode=='screen':
        if args.stage:raise ValueError('Screen always requires all three stages')
        if not all((args.candidate,args.validation_marker,args.configs)):
            raise ValueError('Screen requires candidate, successful validation marker and configurations')
        candidate=args.candidate.resolve(strict=True)
        pinned[candidate]=sha256(candidate)
        marker,evidence=validate_marker(args.validation_marker.resolve(strict=True),pinned[candidate])
        pinned.update(evidence);pinned[args.configs.resolve()]=sha256(args.configs)
        configs=json.loads(args.configs.read_text())
        if (not configs or len({x['name'] for x in configs})!=len(configs)
                or any(x['name'] in ('baseline','control','combined') or not x['name'].replace('_','').isalnum()
                       or not isinstance(x['flags'],dict) or not x['group']
                       or any(not k.startswith('IPC_GPU_') or not isinstance(v,str) for k,v in x['flags'].items())
                       for x in configs)):
            raise ValueError('Invalid or duplicate trial configuration')
        for config in configs:validate_flags(config['flags'])
    cpus=physical_cpus()
    if cpus!=list(range(64)):raise ValueError('Physical CPU affinity changed')
    state=dict(status='preparing',pid=os.getpid(),mode=args.mode,updated_unix=time.time(),runs=[],
               baseline_sha256=BASE_SHA,candidate_sha256=pinned.get(candidate),monitoring_policy=POLICY,
               configurations=configs,validation_marker=marker)
    process=None;current=None;references={};first_hashes={};paired_times={}
    def save(status,**extra):
        state.update(status=status,updated_unix=time.time(),**extra)
        write_json(root/'status.json',state)
        lines=['Round 4 GPU experiment: '+status, POLICY,
               'variant | stage | solver seconds | baseline speedup | bytes']
        for r in state['runs']:
            if r.get('status')=='accepted':
                ratio='diagnostic' if r['instrumented'] else f"{r['speedup']:.4f}x" if 'speedup' in r else 'pending paired baseline'
                lines.append(f"{r['variant']} | {r['stage']} | {r['total_solver_s']:.6f} | {ratio} | identical")
        if state.get('error'):lines.append(state['error'])
        if state.get('full_summary'):lines.append(json.dumps(state['full_summary']))
        (root/'summary.txt').write_text('\n'.join(lines)+'\n')
    def verify():
        for path,digest in pinned.items():
            if sha256(path)!=digest:raise RuntimeError('Pinned input changed: '+str(path))
    def check(a,b,group=None):
        reasons,load=activity(a,b,[],group,max_background_cores=2)
        for p in b['processes'].values():
            if p['state'] not in 'TtZ' and (not group or p['group']!=group) and p['name'] in (
                    'gcc','g++','cc','c++','cc1','cc1plus','clang','clang++','clang-18','clang-19','ld','ld.lld'):
                reasons.append(f"active compiler {p['pid']} ({p['name']})")
        # Explicit, recorded exception for ordinary editor/network-filesystem I/O.
        return [r for r in reasons if r!='I/O wait'],load
    def wait_idle(label):
        quiet=0;start=time.monotonic()
        while quiet<2:
            a=sample();time.sleep(3);b=sample();reasons,load=check(a,b)
            quiet=0 if reasons else quiet+1
            save('waiting_for_available_server',current=label,reasons=reasons,external_cpu_cores=load,
                 observed_idle_iowait_cores=max(0.,(b['iowait']-a['iowait'])/HZ/(b['time']-a['time'])))
            if time.monotonic()-start>172800:raise TimeoutError('Server remained busy for two days')
    def execute(variant,flags,stage,repeat=1,profile=False,full=False):
        nonlocal process,current
        spec=dict(restart_frame=None,first_frame=1,last_frame=240) if full else SCENARIOS[stage]
        name=f"{len(state['runs']):03d}_{stage}_{variant}_r{repeat}"+('_profile' if profile else '')
        current=dict(name=name,variant=variant,flags=flags,stage=stage,repeat=repeat,
                     status='preparing',instrumented=profile,background_observations=[])
        state['runs'].append(current);wait_idle(name);verify()
        directory=root/name;directory.mkdir();out=directory/'frames'
        restart=spec['restart_frame']
        if restart is not None:
            out.mkdir()
            source=clean_dir/'frames'/f'state_{restart:04d}.bin'
            if sha256(source)!=reference['files'][source.name]['sha256']:raise ValueError('Checkpoint changed')
            shutil.copy2(source,out/source.name)
            if sha256(out/source.name)!=reference['files'][source.name]['sha256']:raise ValueError('Checkpoint copy differs')
        binary=baseline if variant=='baseline' or args.mode=='profile' else candidate
        argv=segment_command(repo,out,binary,spec,cpus);env=environment(flags)
        if profile:env['IPC_GPU_PROFILE_CSV']=str(directory/'phases.csv')
        log=directory/'run.log'
        current.update(command=argv,environment={k:v for k,v in env.items() if k.startswith(('OMP_','IPC_GPU_'))},
                       log=str(log),outdir=str(out),executable_sha256=pinned[binary],status='running')
        write_json(directory/'command.json',current)
        checked={};before=sample();start=time.monotonic()
        print('Starting',name,flush=True)
        with log.open('x') as stream:
            process=subprocess.Popen(argv,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            save('running',current=name,child_pid=process.pid)
            while True:
                try:process.wait(timeout=5);done=True
                except subprocess.TimeoutExpired:done=False
                now=sample(process.pid);reasons,load=check(before,now,process.pid)
                current['background_observations'].append(dict(external_cpu_cores=load,reasons=reasons,
                    system_iowait_cpu_s=max(0.,(now['iowait']-before['iowait'])/HZ)))
                before=now
                if reasons:raise RuntimeError('Competing workload: '+'; '.join(reasons))
                if done and process.returncode:raise RuntimeError('GPU exit '+str(process.returncode))
                if full:
                    through,count=safe_completed_frame(log,240,process_complete=done)
                    verify_ready_frames(out,reference,checked,through)
                    current.update(completed_frames=count,verified_files=len(checked))
                save('running',elapsed_s=time.monotonic()-start)
                if time.monotonic()-start>(43200 if full else 1800):raise TimeoutError('Trial timeout')
                if done:break
        current['observed_process_wall_s']=time.monotonic()-start;process=None;verify()
        if full:
            parsed=parse_trial(log,out,240,'gpu',readme_command=True)
            parsed.update(instrumented=False,process_elapsed_s=current['observed_process_wall_s'])
            actual=checkpoint_manifest(out,240)
            if actual['files']!=reference['files']:raise ValueError('Full 482-file output mismatch')
            write_json(directory/'checkpoint_manifest.json',actual)
            meta=dict(command=argv,solver_command=argv[6:],environment=current['environment'],
                      executable_sha256=pinned[binary],cpu_affinity=cpus,readme_sha256=pinned[repo/'README.md'],
                      experiment_flags=flags,monitoring_policy=POLICY)
            report=build_report(directory,parsed,cpu,meta,reference_path,pinned[reference_path],clean)
            report['notes']=[n for n in report['notes'] if 'Idle checks reject system I/O wait' not in n]+[POLICY]
            write_report(directory,report)
            current.update(status='accepted',total_solver_s=parsed['total_solver_s'],byte_identical=True)
            state['full_summary']=dict(total_solver_s=parsed['total_solver_s'],
                average_solver_frame_s=parsed['average_frame_s'],speedup=903.817696/parsed['total_solver_s'],
                gpu_over_cpu=report['gpu_over_cpu_solver_ratio'],verified_files=len(actual['files']),
                clean_report=str(directory/'clean_report.json'),flags=flags)
        else:
            parsed=parse_segment(log,out,spec)
            if variant=='baseline' and not profile:
                first_hashes.setdefault(stage,parsed['output_hashes'])
                references[stage]=parsed
                paired_times[stage,repeat]=parsed['total_solver_s']
            if parsed['output_hashes']!=first_hashes[stage]:raise ValueError('Byte mismatch: '+name)
            current.update(parsed,byte_identical=True)
            if (stage,repeat) in paired_times and not profile:
                current['speedup']=paired_times[stage,repeat]/parsed['total_solver_s']
            if profile:
                phases=read_phases(directory/'phases.csv');totals={}
                if set(phases)!=set(range(spec['first_frame'],spec['last_frame']+1)):
                    raise ValueError('Profile is missing requested frames')
                for entry in phases.values():
                    for label,values in entry['stages'].items():
                        totals[label]=totals.get(label,0)+values['cuda_exclusive_ms']/1000
                current['top_stream_intervals_s']=sorted(totals.items(),key=lambda x:-x[1])[:12]
                current['profile_note']='CUDA stream intervals include idle gaps, not kernel-active time; not clean speedup.'
                current['profile_over_clean_time']=parsed['total_solver_s']/references[stage]['total_solver_s']
                current.pop('speedup',None)
            current['status']='accepted'
            if variant=='baseline' and not profile:
                for earlier in state['runs']:
                    if earlier['stage']==stage and earlier['repeat']==repeat and earlier.get('status')=='accepted' and not earlier['instrumented']:
                        earlier['speedup']=paired_times[stage,repeat]/earlier['total_solver_s']
                        if earlier is not current:write_json(root/earlier['name']/'result.json',earlier)
        write_json(directory/'result.json',current);save('between_trials',child_pid=None)
        print('Accepted',name,f"solver={current['total_solver_s']:.6f}s",flush=True)
        return current
    def interrupt(signum,frame):raise KeyboardInterrupt('Termination '+str(signum))
    signal.signal(signal.SIGTERM,interrupt)
    lock=(repo/'gpu/.example1_comparison.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        if args.mode=='profile':
            for stage in args.stage or SCENARIOS:
                execute('baseline',{},stage)
                execute('profile',{},stage,profile=True)
        else:
            # First verify identical outputs with every new optimization OFF.
            choices=[dict(name='control',flags={},group='control'),*configs]
            for number,stage in enumerate(SCENARIOS):
                execute('baseline',{},stage)
                for config in (choices if number%2==0 else list(reversed(choices))):
                    execute(config['name'],config['flags'],stage)
            ranking=rank_variants(state['runs'],configs);state['screening']=ranking
            winners={}
            for item in ranking:
                if item['group'] not in winners and item['score']>1.02:winners[item['group']]=item
            combined={}
            for item in winners.values():
                for k,v in item['flags'].items():
                    if k in combined and combined[k]!=v:raise ValueError('Conflicting combination flags')
                    combined[k]=v
            state['selected_group_winners']=winners;state['combined_flags']=combined
            # Confirmation uses new baselines and alternates order across repeats.
            # The first screening baseline hashes remain valid for the same checkpoint.
            for repeat in (2,3):
                for stage in SCENARIOS:
                    order=('baseline','combined') if repeat==2 else ('combined','baseline')
                    for variant in order:execute(variant,combined if variant=='combined' else {},stage,repeat)
            confirmation=[]
            for stage in SCENARIOS:
                selected=[r for r in state['runs'] if r['stage']==stage and r['repeat'] in (2,3)]
                old=median(r['total_solver_s'] for r in selected if r['variant']=='baseline')
                new=median(r['total_solver_s'] for r in selected if r['variant']=='combined')
                confirmation.append(dict(stage=stage,baseline_median_s=old,candidate_median_s=new,speedup=old/new))
            state['confirmation']=confirmation
            score=math.exp(sum(math.log(x['speedup']) for x in confirmation)/3)
            if combined and score>1.02 and min(x['speedup'] for x in confirmation)>.95:
                execute('combined',combined,'full',full=True)
            else:
                state['full_skipped_reason']='No repeat-confirmed combined improvement above 2% without a >5% stage regression.'
        verify();save('complete',current=None,child_pid=None)
    except BaseException as error:
        if process is not None:stop_our_process(process)
        if current and current.get('status')!='accepted':current.update(status='rejected',error=str(error))
        save('stopped',error=type(error).__name__+': '+str(error),child_pid=None);raise
    finally:
        if process is not None:stop_our_process(process)
        lock.close()


if __name__=='__main__':main()
