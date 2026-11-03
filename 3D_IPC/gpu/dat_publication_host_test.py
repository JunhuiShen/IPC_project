#!/usr/bin/env python3
"""Standalone host DAT publication regression; no CUDA SDK/device required.

Compiles actual DeviceArray/publication/view/status/snapshot helper code with
fake device storage under ASan/UBSan. This does not qualify GPU arithmetic,
CUDA stream ordering, collision safety or CPU/GPU trajectory parity.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,tempfile,sys,shutil

FAKE_RUNTIME = r"""#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
using cudaStream_t=void*;
using cudaError_t=int;
inline constexpr cudaError_t cudaSuccess=0;
enum cudaMemcpyKind {cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice};
namespace fake_cuda {
struct Counters {std::size_t mallocs=0,frees=0,copies=0,sets=0;};
inline Counters calls;
inline std::map<void*,std::size_t> live;
inline bool equals(Counters a,Counters b) {return a.mallocs==b.mallocs&&a.frees==b.frees&&a.copies==b.copies&&a.sets==b.sets;}
inline void range(const void* p,std::size_t n) {
 if(!n)return;
 const auto a=reinterpret_cast<std::size_t>(p);
 for(const auto& item:live) {
  const auto b=reinterpret_cast<std::size_t>(item.first);
  if(a>=b&&a-b<=item.second&&n<=item.second-(a-b))return;
 }
 throw std::runtime_error("fake CUDA invalid device range");
}
}
inline cudaError_t cudaMalloc(void** p,std::size_t n) {
 ++fake_cuda::calls.mallocs;*p=std::malloc(n);if(!*p&&n)return 1;
 if(*p)fake_cuda::live.emplace(*p,n);return 0;
}
inline cudaError_t cudaFree(void* p) {
 ++fake_cuda::calls.frees;
 if(p) {auto i=fake_cuda::live.find(p);if(i==fake_cuda::live.end())throw std::runtime_error("double/foreign cudaFree");fake_cuda::live.erase(i);std::free(p);}
 return 0;
}
inline cudaError_t cudaMemcpy(void* to,const void* from,std::size_t n,cudaMemcpyKind kind) {
 ++fake_cuda::calls.copies;
 if(kind!=cudaMemcpyHostToDevice)fake_cuda::range(from,n);
 if(kind!=cudaMemcpyDeviceToHost)fake_cuda::range(to,n);
 if(n)std::memmove(to,from,n);return 0;
}
inline cudaError_t cudaMemset(void* p,int value,std::size_t n) {
 ++fake_cuda::calls.sets;fake_cuda::range(p,n);if(n)std::memset(p,value,n);return 0;
}
inline cudaError_t cudaGetLastError() {return 0;}
inline const char* cudaGetErrorString(cudaError_t) {return "fake CUDA error";}
"""

HARNESS = r"""#include "device_memory.h"
#include "solver.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

using namespace ipc_gpu;
using namespace ipc_gpu::solver;
namespace {
std::size_t checks=0;
void require(bool ok,const char* message) {++checks;if(!ok)throw std::runtime_error(message);}
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b) {
 return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
bool same(const StateSnapshot& a,const StateSnapshot& b) {
 return same(a.positions,b.positions)&&same(a.velocities,b.velocities)&&same(a.com,b.com)&&same(a.com_velocity,b.com_velocity)&&same(a.omega,b.omega)&&same(a.orientation,b.orientation);
}
std::vector<Vector3> values(std::size_t n,double seed) {
 std::vector<Vector3> a(n);for(std::size_t i=0;i<n;++i)a[i]={seed+double(i)/8,-0.0,std::numeric_limits<double>::denorm_min()};return a;
}
std::vector<Quaternion> quats(std::size_t n,double seed) {
 std::vector<Quaternion>a(n);for(std::size_t i=0;i<n;++i)a[i]={seed+double(i)/8,-0.0,.25,std::numeric_limits<double>::denorm_min()};return a;
}
struct GpuProfileScope {explicit GpuProfileScope(const char*) {}};
std::mutex gpu_mutex;
double (*v3(DeviceArray<Vector3>& a))[3] {return reinterpret_cast<double(*)[3]>(a.data());}
double (*q4(DeviceArray<Quaternion>& a))[4] {return reinterpret_cast<double(*)[4]>(a.data());}

struct Impl {
 DeviceArray<Vector3> positions,velocities,previous,predicted,dat_next;
 DeviceArray<Vector3> com,com_velocity,omega,previous_com,previous_com_velocity,previous_omega;
 DeviceArray<Vector3> proposed_com,proposed_omega,accepted_com,accepted_omega;
 DeviceArray<Quaternion> orientation,previous_orientation,proposed_orientation,accepted_orientation;
 DeviceArray<Derivatives> body_systems;
 DeviceArray<int> error;
 DeformableScene scene{};RigidScene rigid{};DeformableState state{};RigidState rigid_state{};RigidWorkspace rigid_work{};
 bool invalid=false;
 Impl(std::size_t n,std::size_t b) {
  scene.nodes=n;rigid.bodies=b;
  positions.upload(values(n,1));velocities.upload(values(n,2));previous.upload(values(n,3));predicted.upload(values(n,4));dat_next.upload(values(n,101));
  com.upload(values(b,5));com_velocity.upload(values(b,6));omega.upload(values(b,7));previous_com.upload(values(b,8));previous_com_velocity.upload(values(b,9));previous_omega.upload(values(b,10));
  proposed_com.upload(values(b,11));proposed_omega.upload(values(b,12));accepted_com.upload(values(b,102));accepted_omega.upload(values(b,103));
  orientation.upload(quats(b,13));previous_orientation.upload(quats(b,14));proposed_orientation.upload(quats(b,15));accepted_orientation.upload(quats(b,104));body_systems.resize(b);error.upload({0});
  refresh_state_views();
 }
 __REFRESH_METHOD__
 __PUBLISH_METHOD__
 __CHECK_STATUS_METHOD__
 void baseline_publish() { __BASELINE_PUBLICATION__ }
 void candidate_checked_publish() { __CANDIDATE_GUARDED_PUBLICATION__ }
 void baseline_checked_publish() {cuda_checked(cudaGetLastError());check_status();baseline_publish();}
 void assert_views() {
  require(state.positions==v3(positions),"stale particle positions view");require(state.velocities==v3(velocities),"stale velocity view");require(state.previous==v3(previous),"stale previous view");require(state.predicted==v3(predicted),"stale predicted view");
  require(rigid_state.com==v3(com),"stale COM view");require(rigid_state.velocity==v3(com_velocity),"stale COM velocity view");require(rigid_state.omega==v3(omega),"stale angular view");require(rigid_state.orientation==q4(orientation),"stale orientation view");
  require(rigid_state.previous_com==v3(previous_com),"stale previous COM view");require(rigid_state.previous_velocity==v3(previous_com_velocity),"stale previous COM velocity view");require(rigid_state.previous_orientation==q4(previous_orientation),"stale q_n view");require(rigid_state.previous_omega==v3(previous_omega),"stale previous omega view");
  require(rigid_work.systems==body_systems.data(),"stale systems view");require(rigid_work.proposed_com==v3(proposed_com),"stale proposed COM view");require(rigid_work.proposed_omega==v3(proposed_omega),"stale proposed omega view");require(rigid_work.proposed_orientation==q4(proposed_orientation),"stale proposed orientation view");
  require(rigid_work.accepted_com==v3(accepted_com),"stale accepted COM scratch view");require(rigid_work.accepted_omega==v3(accepted_omega),"stale accepted omega scratch view");require(rigid_work.accepted_orientation==q4(accepted_orientation),"stale accepted orientation scratch view");
  if(scene.nodes)require(state.positions!=v3(dat_next),"particle input/output alias after swap");
  if(rigid.bodies)require(rigid_state.com!=rigid_work.accepted_com&&rigid_state.omega!=rigid_work.accepted_omega&&rigid_state.orientation!=rigid_work.accepted_orientation,"pose input/output alias after swap");
 }
 void prepare(double step) {
  for(std::size_t i=0;i<scene.nodes;++i)for(int k=0;k<3;++k)v3(dat_next)[i][k]=state.positions[i][k]+step*(k+1);
  for(std::size_t i=0;i<rigid.bodies;++i) {
   for(int k=0;k<3;++k) {rigid_work.accepted_com[i][k]=rigid_state.com[i][k]+step*(k+1);rigid_work.accepted_omega[i][k]=rigid_state.omega[i][k]-step*(k+1);}
   for(int k=0;k<4;++k)rigid_work.accepted_orientation[i][k]=rigid_state.orientation[i][k]+step*(k+1);
  }
 }
 void ordinary_consumer(double step) {
  // Deliberately consume the actual published views, as later kernels do.
  for(std::size_t i=0;i<scene.nodes;++i)state.positions[i][1]-=step;
  for(std::size_t i=0;i<rigid.bodies;++i) {rigid_state.com[i][2]+=step;rigid_state.omega[i][0]-=step;rigid_state.orientation[i][2]+=step;}
 }
};
struct Solver {
 std::unique_ptr<Impl> impl_;
 Solver(std::size_t n,std::size_t b):impl_(std::make_unique<Impl>(n,b)){}
 StateSnapshot snapshot()const { __SNAPSHOT_BODY__ }
};

void device_array_ownership() {
 const auto initial=fake_cuda::live.size();
 {
  DeviceArray<int>a(std::vector<int>{11,22,33}),b(std::vector<int>{99}),empty;
  auto* ap=a.data();auto* bp=b.data();auto calls=fake_cuda::calls;a.swap(b);
  require(fake_cuda::equals(calls,fake_cuda::calls),"swap performed CUDA operation");require(a.data()==bp&&a.size()==1&&b.data()==ap&&b.size()==3,"swap did not exchange pointer AND size");require(a.download()==std::vector<int>{99}&&b.download()==std::vector<int>({11,22,33}),"swap changed contents");
  calls=fake_cuda::calls;a.swap(a);require(fake_cuda::equals(calls,fake_cuda::calls)&&a.data()==bp&&a.size()==1,"self swap changed storage");
  calls=fake_cuda::calls;empty.swap(b);require(fake_cuda::equals(calls,fake_cuda::calls)&&empty.data()==ap&&empty.size()==3&&b.data()==nullptr&&b.size()==0,"empty/full swap invalid");
  for(int i=0;i<101;++i) {calls=fake_cuda::calls;empty.swap(b);require(fake_cuda::equals(calls,fake_cuda::calls),"repeated swap performed CUDA operation");}
  require(b.data()==ap&&b.download()==std::vector<int>({11,22,33}),"odd repeated swaps lost ownership");
  DeviceArray<int>e1,e2;calls=fake_cuda::calls;e1.swap(e2);require(fake_cuda::equals(calls,fake_cuda::calls)&&!e1.data()&&!e2.data(),"empty swap invalid");
 }
 require(fake_cuda::live.size()==initial,"device-array lifecycle leaked storage");
}
void publication_case(std::size_t n,std::size_t b,int sweeps) {
 Solver baseline(n,b),candidate(n,b);
 require(same(baseline.snapshot(),candidate.snapshot()),"different initial state");
 for(int iteration=0;iteration<sweeps;++iteration) {
  auto before=candidate.snapshot();baseline.impl_->prepare(.125+iteration/16.);candidate.impl_->prepare(.125+iteration/16.);
  require(same(before,candidate.snapshot()),"writing accepted scratch mutated frozen inputs");
  auto calls=fake_cuda::calls;baseline.impl_->baseline_publish();const auto copies=fake_cuda::calls.copies-calls.copies;
  require(copies==(n?1u:0u)+(b?3u:0u),"baseline copy count changed");
  calls=fake_cuda::calls;candidate.impl_->publish_dat_state();require(fake_cuda::equals(calls,fake_cuda::calls),"candidate publication performed CUDA operation");
  baseline.impl_->assert_views();candidate.impl_->assert_views();require(same(baseline.snapshot(),candidate.snapshot()),"publication changed exact accepted values");
  require(same(candidate.snapshot(),candidate.snapshot()),"snapshot changed state");
  // Stress DAT→ordinary consumer→DAT across odd/even owner identities.
  baseline.impl_->ordinary_consumer(.0625);candidate.impl_->ordinary_consumer(.0625);require(same(baseline.snapshot(),candidate.snapshot()),"ordinary consumer used stale state view");
 }
 auto before=candidate.snapshot();candidate.impl_->prepare(.75);candidate.impl_->error.upload({21});auto* positions=candidate.impl_->positions.data();auto* scratch=candidate.impl_->dat_next.data();
 bool rejected=false;try {candidate.impl_->candidate_checked_publish();}catch(const std::runtime_error&){rejected=true;}
 require(rejected&&candidate.impl_->invalid,"production check_status did not reject failed publication");require(candidate.impl_->positions.data()==positions&&candidate.impl_->dat_next.data()==scratch,"failed status exchanged owners");
 require(same(before,baseline.snapshot()),"baseline ordinary state unexpectedly changed");
 bool snapshot_rejected=false;try{(void)candidate.snapshot();}catch(const std::runtime_error&){snapshot_rejected=true;}require(snapshot_rejected,"production snapshot exposed invalid state");
 candidate.impl_->assert_views();
}
}
int main() {
 try {
  device_array_ownership();
  for(const auto sizes:std::vector<std::array<std::size_t,2>>{{0,0},{1,0},{9,0},{6,2},{9,2}})
   for(int sweeps:{0,1,2,3,4,7,16,33}) {
    const auto live=fake_cuda::live.size();publication_case(sizes[0],sizes[1],sweeps);require(fake_cuda::live.size()==live,"published/destructed solver leaked storage");
   }
  require(fake_cuda::live.empty(),"live device storage after all tests");
  std::cout<<"HOST_PUBLICATION_ACCEPTANCE_PASS checks="<<checks<<" allocations="<<fake_cuda::calls.mallocs<<" frees="<<fake_cuda::calls.frees<<"\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<"HOST_PUBLICATION_ACCEPTANCE_FAIL: "<<e.what()<<'\n';return 1;}
}
"""

COPY_PUBLICATION_REFERENCE = r"""
            // Publish only the already-certified represented arrays, after all
            // particle and rigid checks join. No endpoint arithmetic is repeated.
            if(scene.nodes)cuda_checked(cudaMemcpy(positions.data(),dat_next.data(),scene.nodes*sizeof(Vector3),cudaMemcpyDeviceToDevice));
            if(rigid.bodies) {
                cuda_checked(cudaMemcpy(com.data(),accepted_com.data(),rigid.bodies*sizeof(Vector3),cudaMemcpyDeviceToDevice));
                cuda_checked(cudaMemcpy(omega.data(),accepted_omega.data(),rigid.bodies*sizeof(Vector3),cudaMemcpyDeviceToDevice));
                cuda_checked(cudaMemcpy(orientation.data(),accepted_orientation.data(),rigid.bodies*sizeof(Quaternion),cudaMemcpyDeviceToDevice));
            }
        """

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--cxx')
    p.add_argument('--report',type=Path)
    a=p.parse_args()
    source=a.source.resolve()
    managed=tempfile.TemporaryDirectory(prefix='gpu_dat_publish_host_')
    output=Path(managed.name);(output/'fake_cuda').mkdir()
    (output/'fake_cuda/cuda_runtime_api.h').write_text(FAKE_RUNTIME)

    def balanced(s,opening):
        depth=0
        for i in range(opening,len(s)):
            if s[i]=='{':depth+=1
            elif s[i]=='}':
                depth-=1
                if depth==0:return i
        raise ValueError('unbalanced C++ scope')
    def method(s,signature):
        start=s.index(signature);opening=s.index('{',start);end=balanced(s,opening)
        return s[start:end+1],s[opening+1:end]
    def loop_and_guard(s):
        all_method,body=method(s,'void run_dat(')
        loop=body.index('for(int iteration=');opening=body.index('{',loop);end=balanced(body,opening)
        inside=body[opening+1:end];guard='cuda_checked(cudaGetLastError());check_status();';at=inside.rindex(guard)
        return inside[:at],inside[at:],inside[at+len(guard):]

    new=(source/'gpu/solver.cu').read_text();_,new_snapshot=method(new,'StateSnapshot GpuSolver::snapshot()const')
    _,guarded,_=loop_and_guard(new);publication=COPY_PUBLICATION_REFERENCE
    refresh,_=method(new,'void refresh_state_views()');publish,_=method(new,'void publish_dat_state()');check,_=method(new,'void check_status()')
    template=HARNESS
    for key,value in {'__REFRESH_METHOD__':refresh,'__PUBLISH_METHOD__':publish,'__CHECK_STATUS_METHOD__':check,'__BASELINE_PUBLICATION__':publication,'__CANDIDATE_GUARDED_PUBLICATION__':guarded,'__SNAPSHOT_BODY__':new_snapshot}.items():
        assert key in template;template=template.replace(key,value)
    cpp=output/'publication_host_acceptance.cpp';cpp.write_text(template)
    compiler=a.cxx or os.environ.get('CXX') or shutil.which('clang++') or 'c++'
    cmd=[compiler,'-std=c++17','-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I'+str(output/'fake_cuda'),'-I'+str(source/'gpu'),str(cpp),'-o',str(output/'publication_host_acceptance')]
    build=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(output/'build.log').write_text(build.stdout)
    if build.returncode:raise SystemExit(build.stdout)
    env=os.environ.copy();env['ASAN_OPTIONS']='detect_leaks=0'if sys.platform=='darwin'else'detect_leaks=1';run=subprocess.run([str(output/'publication_host_acceptance')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env);(output/'run.log').write_text(run.stdout)
    report={'status':'PASS'if run.returncode==0 else'FAIL','validation':'Compiled host ownership/publication harness with fake CUDA storage, real candidate DeviceArray, actual extracted publication/view/status/snapshot helpers and exact baseline publication block.','sanitizers':['AddressSanitizer','UndefinedBehaviorSanitizer'],'compile_command':cmd,'source_sha256':{str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest()for path in [source/'gpu/device_memory.h',source/'gpu/solver.cu']},'copy_reference_sha256':hashlib.sha256(COPY_PUBLICATION_REFERENCE.encode()).hexdigest(),'run_output':run.stdout,'native_cuda_execution':False,'limitations':['Does not compile or run device kernels','Does not establish CPU/GPU numerical parity or collision safety','Does not validate asynchronous stream ordering or actual GpuSolver mode transitions; host consumers exercise the published production views.']}
    report['leak_check']='Every fake-CUDA allocation is tracked; all pointer owners must be released exactly once. macOS LeakSanitizer is unavailable.'if sys.platform=='darwin'else'LeakSanitizer and fake-CUDA ownership accounting.'
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(run.stdout.strip())
    print('Host ownership/publication checks only; native NVIDIA validation remains required.')
    managed.cleanup()
    return run.returncode


if __name__=="__main__":
    raise SystemExit(main())
